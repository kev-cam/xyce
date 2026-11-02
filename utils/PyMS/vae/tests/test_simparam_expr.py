"""``$simparam`` in analog code.

THE DEFECT (the old BUGS.md B10, corrected by the PYMS1 skeptic, 2026-10-25): in an
analog expression vae/ginac_emitter.py rewrote ``$simparam("name",
default)`` to the default whatever the simulator's value, and
``$simparam("gmin")`` to 0, so ``I(p, n) <+ V(p, n) * $simparam("gmin");``
was a 0 S device: V(out) = 0 on the 1 V / 1k divider, rc 0 -- a silent 0,
not the non-convergence the 1.1.0 BUGS.md said.

THE FIX: GiNaCEmitter._sub_simparam substitutes the value the device shell
passes (gmin and tnom, the __SIMPARAM__ lines of the params file) when it
has one, the default otherwise, and refuses a name with neither.  1.1.2: the
same in a condition (GiNaCEmitter._preprocess_sys_funcs, the one entry of
every condition lowering).

Run:  cd utils/PyMS && python3 vae/tests/test_simparam_expr.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a  # noqa: E402
from vae.ginac_emitter import emit_ginac_program, EmitError  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'


def mod(expr):
    return (HDR + "module a1(p, n);\n  inout p, n;\n  electrical p, n;\n"
            "  analog I(p, n) <+ V(p, n) * %s;\nendmodule\n" % expr)


class TestEmitter(unittest.TestCase):

    def _f(self, expr, simparams):
        src = emit_ginac_program(parse_verilog_a(mod(expr), filename="a1.va"), simparams=simparams)
        return [l.strip() for l in src.split('\n') if 'F_contribs[br_idx] +=' in l][0]

    def test_known_value_wins(self):
        self.assertIn('0.001', self._f('$simparam("gmin")', {'gmin': 1e-3}))
        self.assertIn('0.001', self._f('$simparam("gmin", 5.0)', {'gmin': 1e-3}))

    def test_default_when_unknown(self):
        self.assertIn('5.0', self._f('$simparam("foo", 5.0)', {'gmin': 1e-3}))

    def test_unknown_without_default_is_an_error(self):
        with self.assertRaisesRegex(EmitError, r'\$simparam\("foo"\)'):
            self._f('$simparam("foo")', {'gmin': 1e-3})

    # 1.1.2 (PYMS2 skeptic, 2026-10-26): $simparam in a CONDITION reached g++
    # verbatim ("'$simparam' was not declared in this scope").
    def _fc(self, cond, simparams):
        va = (HDR + "module a1(p, n);\n  inout p, n;\n  electrical p, n;\n"
              "  analog begin\n    if (%s) I(p, n) <+ V(p, n) / 1000;\n"
              "    else I(p, n) <+ V(p, n) / 3000;\n  end\nendmodule\n" % cond)
        src = emit_ginac_program(parse_verilog_a(va, filename="a1.va"), simparams=simparams)
        return [l.strip() for l in src.split('\n') if 'F_contribs[br_idx] +=' in l]

    def test_in_a_condition(self):
        # one arm only, resolved at elaboration
        self.assertEqual(self._fc('$simparam("gmin", 1.0) < 1e-6', {'gmin': 1e-12}),
                         ['F_contribs[br_idx] += (V_p - V_n) / 1000;'])
        self.assertEqual(self._fc('$simparam("gmin", 1.0) < 1e-6', {'gmin': 1e-3}),
                         ['F_contribs[br_idx] += (V_p - V_n) / 3000;'])
        self.assertEqual(self._fc('$simparam("foo", 3.0) > 2', {'gmin': 1e-12}),
                         ['F_contribs[br_idx] += (V_p - V_n) / 1000;'])
        with self.assertRaisesRegex(EmitError, r'\$simparam\("foo"\)'):
            self._fc('$simparam("foo") > 2', {'gmin': 1e-12})


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_gmin_conductance(self):
        work = tempfile.mkdtemp(prefix="pyms_sp_")
        try:
            env = dict(os.environ)
            env.update(PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(work, "hc"),
                       PYMS_VAE_CACHE=os.path.join(work, "vc"), PYTHONDONTWRITEBYTECODE="1")
            env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            with open(os.path.join(work, "a1.va"), "w") as f:
                f.write(mod('$simparam("gmin")'))
            with open(os.path.join(work, "d.cir"), "w") as f:
                f.write('* g\n.hdl "a1.va"\n.options device gmin=1e-3\nV1 in 0 DC 1\n.model m1 a1\n'
                        'Ya1 X1 in out m1\nR2 out 0 1k\n.op\n.print dc v(out)\n.end\n')
            r = subprocess.run([XYCE, "d.cir"], cwd=work, env=env, capture_output=True, text=True, timeout=900)
            with open(os.path.join(work, "d.cir.prn")) as f:
                got = float(f.read().split("\n")[1].split()[1])
            self.assertEqual(r.returncode, 0)
            # the device: gmin = 1 mS from in to out; R2 1 mS to ground; and the
            # shell's own diagonal gmin (the same option) from out to ground:
            # out = 1m / (1m + 1m + 1m)
            self.assertAlmostEqual(got, 1.0 / 3.0, delta=1e-6)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_condition(self):
        work = tempfile.mkdtemp(prefix="pyms_spc_")
        try:
            env = dict(os.environ)
            env.update(PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(work, "hc"),
                       PYMS_VAE_CACHE=os.path.join(work, "vc"), PYTHONDONTWRITEBYTECODE="1")
            env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            with open(os.path.join(work, "a1.va"), "w") as f:
                f.write(HDR + "module a1(p, n);\n  inout p, n;\n  electrical p, n;\n"
                        "  analog begin\n    if ($simparam(\"gmin\", 1.0) < 1e-6) I(p, n) <+ V(p, n) / 1000;\n"
                        "    else I(p, n) <+ V(p, n) / 3000;\n  end\nendmodule\n")
            with open(os.path.join(work, "d.cir"), "w") as f:
                f.write('* c\n.hdl "a1.va"\nV1 in 0 DC 1\n.model m1 a1\n'
                        'Ya1 X1 in out m1\nR2 out 0 1k\n.op\n.print dc v(out)\n.end\n')
            r = subprocess.run([XYCE, "d.cir"], cwd=work, env=env, capture_output=True, text=True, timeout=900)
            got = None
            try:
                with open(os.path.join(work, "d.cir.prn")) as f:
                    got = float(f.read().split("\n")[1].split()[1])
            except (OSError, IndexError, ValueError):
                pass
            self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-1500:])
            self.assertAlmostEqual(got, 0.5, delta=1e-6)   # gmin 1e-12 < 1e-6: the 1k arm
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
