#!/usr/bin/env python3
"""A variable reassigned from its own value, when the old value is a constant
and the new right-hand side is not: ``x = a; x = x + tc * ($temperature -
300.15);``.

THE BUG (2026-10-25, PYMS1-INT): vae/ginac_emitter.py ``_walk_statement``
(the ASSIGN case) dropped the left-hand side's known value (``var_values``)
BEFORE it translated the right-hand side.  ``$temperature`` and any voltage
make an expression non-constant, so the second assignment read ``x`` as the
GiNaC symbol of itself: the self-reference ``x = x + ...``, emitted as
``double x = 0.0;`` (a module-level variable) or ``double y = y;`` (a
block-local one, an uninitialized read).  BSIM-SOI 4.5/4.6.1/4.7 computes its
gate-sidewall junction potential this way (``PhiBSWG = ...Potential;
PhiBSWG = PhiBSWG + dPhiBSWG_dT * (DevTemp - B4SOItnom);``): PhiBSWG was 0,
the junction charge NaN, and the DC operating point never converged.  With
the fix BSIM-SOI 4.5 matches ngspice's B4SOI (level 10) to the gmin current.

Each module below sits in a 1 V / 1k divider (out = 1k / (r_eff + 1k)).
Skips without a Xyce binary.

Run:  python3 vae/tests/test_self_reference_assign.py
"""
import os
import shutil
import subprocess
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
RTOL = 1e-6

HEAD = """`include "disciplines.vams"
module sr(p, n);
  inout p, n;
  electrical p, n;
  parameter real a = 1k;
  parameter real tc = 0;
  parameter integer mode = 0;
  real x;
  analog begin
"""
TAIL = """    I(p, n) <+ V(p, n) / x;
  end
endmodule
"""

# name -> (analog statements, expected v(out))
CASES = {
    # the BSIM-SOI form: the constant old value, a $temperature right-hand side
    'temperature': ("""    x = a;
    x = x + tc * ($temperature - 300.15);
""", 0.5),
    # the same under a parameter-only (resolved) condition
    'param_branch': ("""    if (mode == 2) begin
      x = 0.0;
    end else begin
      x = a;
      x = x + tc * ($temperature - 300.15);
    end
""", 0.5),
    # a block-local variable (was `double y = y;`)
    'block_local': ("""    begin : blk
      real y;
      y = a;
      y = y + tc * ($temperature - 300.15);
      x = y;
    end
""", 0.5),
    # a voltage-dependent right-hand side that changes the value: x = 2k
    'voltage_rhs': ("""    x = a;
    x = x + 1k * (1.0 + 0.0 * V(p, n));
""", 1.0 / 3.0),
    # control: a constant reassignment (always right)
    'constant_rhs': ("""    x = a;
    x = x + 1k;
""", 1.0 / 3.0),
}

DECK = """* sr in a 1 V / 1k divider
.hdl "sr.va"
V1 in 0 1
.model msr sr
YSR X1 in out msr
R2 out 0 1k
.op
.print dc v(out)
.end
"""


def _xyce_env(work):
    env = dict(os.environ)
    env["PYMS_DIR"] = _PYMS_DIR
    env["PYMS_CACHE"] = os.path.join(work, "hdl_cache")
    env["PYMS_VAE_CACHE"] = os.path.join(work, "vae_cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    for k in ("VAE_SO_PATH", "VAE_SO_DIR", "PYMS_DEFINES"):
        env.pop(k, None)
    return env


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce at %s" % XYCE)
class SelfReferenceTests(unittest.TestCase):

    def _run(self, name):
        body, want = CASES[name]
        work = tempfile.mkdtemp(prefix="pyms_selfref_%s_" % name)
        try:
            with open(os.path.join(work, "sr.va"), "w") as f:
                f.write(HEAD + body + TAIL)
            with open(os.path.join(work, "d.cir"), "w") as f:
                f.write(DECK)
            r = subprocess.run([XYCE, "d.cir"], capture_output=True, text=True,
                               timeout=600, cwd=work, env=_xyce_env(work))
            prn = os.path.join(work, "d.cir.prn")
            self.assertTrue(r.returncode == 0 and os.path.exists(prn),
                            "%s: Xyce rc %d\n%s" % (name, r.returncode,
                                                    (r.stdout + r.stderr)[-3000:]))
            lines = [ln for ln in open(prn).read().splitlines() if ln.strip()]
            hdr = [h.upper() for h in lines[0].split()]
            out = float(lines[1].split()[hdr.index('V(OUT)')])
            self.assertTrue(abs(out - want) <= RTOL * abs(want),
                            "%s: v(out) = %.12g, expected %.12g" % (name, out, want))
        finally:
            if not os.environ.get("HDLTEST_KEEP"):
                shutil.rmtree(work, ignore_errors=True)

    def test_temperature_rhs_reads_the_old_value(self):
        self._run('temperature')

    def test_under_a_parameter_condition(self):
        self._run('param_branch')

    def test_block_local_variable(self):
        self._run('block_local')

    def test_voltage_rhs_reads_the_old_value(self):
        self._run('voltage_rhs')

    def test_constant_reassignment_control(self):
        self._run('constant_rhs')


if __name__ == "__main__":
    unittest.main()
