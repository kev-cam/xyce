"""A .step of a baked parameter; a module's attributes; one module name in
two .hdl files; the regime of a run-time loop in ac (1.1.5).

THE DEFECTS (PYMS5 skeptic, 2026-11-01, on 1.1.4):
  * a .step of a parameter the math library bakes changed nothing, with no
    message (all versions): ``.step m1:rp list 1000 4000`` with
    ``I(p, n) <+ V(p, n) / rp`` gave 0.5, 0.5 for 0.5, 0.2.  The device shell
    loaded the library once; processParams now computes the key of the
    current values again and loads (builds once, caches) the library of
    those values.
  * the xyceModelGroup / xyceLevelNumber attributes of a later module of a
    file landed on the first (m5_second_has_group_attr: a1 registered as a
    Diode, the netlist's ``.model m1 a1`` an error): the attribute scan now
    reads the declaration of the module the shell is for.
  * two .hdl files that both define module helper: the later one was
    registered over the first, silently.  The first stays and the run warns
    (N_DEV_PyMS.C).
  * a voltage condition inside a run-time loop at exactly 0 V in ac
    (s5_ac_regime_in_rtloop 0.8 for 0.2): the difference quotient stepped
    across the condition; the exact Jacobian takes the slope of the branch
    the operating point is in.

Run:  cd utils/PyMS && python3 vae/tests/test_step_params_and_modules.py
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.tests._decks import HDR, mod, XYCE, XYCE_LIBDIR  # noqa: E402

W = 2 * math.pi * 1e3
TIMES = [0.1e-3, 0.25e-3, 0.6e-3, 1.25e-3, 1.9e-3]
RP = mod("a1", "  parameter real rp = 1000;\n  analog I(p, n) <+ V(p, n) / rp;")
RP_IS = mod("a1", "  parameter real rp = 1000;\n  real r;\n  analog begin\n    @(initial_step) r = rp;\n    I(p, n) <+ V(p, n) / r;\n  end")
A1R = "module a1(p, n);\n  inout p, n;\n  electrical p, n;\n  parameter real r = 1000;\n  analog I(p, n) <+ V(p, n) / r;\nendmodule\n"
DD = ('(* xyceModelGroup="Diode" xyceLevelNumber="2001" *)\nmodule dd(p, n);\n  inout p, n;\n  electrical p, n;\n'
      '  analog I(p, n) <+ V(p, n) / 4000;\nendmodule\n')
HELP1 = "module helper(a);\n  inout a;\n  electrical a;\n  analog I(a) <+ V(a) / 7000;\nendmodule\n"
HELP2 = "module helper(a);\n  inout a;\n  electrical a;\n  analog I(a) <+ V(a) / 9000;\nendmodule\n"
BB = "module bb(p, n);\n  inout p, n;\n  electrical p, n;\n  analog I(p, n) <+ V(p, n) / 1000;\nendmodule\n"
RTLOOP = mod("a1", "  integer k;\n  real g;\n  analog begin\n    k = 0; g = 0;\n    while (k < 1 && V(p, n) > -1) begin\n"
                   "      if (V(p, n) > 0) g = 1m; else g = 0.25m;\n      k = k + 1;\n    end\n    I(p, n) <+ V(p, n) * g;\n  end")
TRAN = ".options timeint reltol=1e-6 abstol=1e-12\n.tran 1u 2m\n.print tran v(out)"


def deck(body, card="", src="V1 in 0 DC 1", dev="Ya1 X1 in out m1\nR2 out 0 1k", hdl='.hdl "a1.va"'):
    return "* t\n%s\n%s\n.model m1 a1 %s\n%s\n%s\n.end\n" % (hdl, src, card, dev, body)


def tr(g):
    return [g * math.sin(W * t) for t in TIMES]


# name -> (files, deck, kind, want, log text that must appear)
CASES = {
    "mstep_op": ({"a1.va": RP}, deck(".step m1:rp list 1000 4000\n.op\n.print dc v(out)", "rp=1000"), "rows", [0.5, 0.2], None),
    "gstep_op": ({"a1.va": RP}, deck(".global_param rpv=1000\n.step rpv list 1000 4000\n.op\n.print dc v(out)", "rp={rpv}"),
                 "rows", [0.5, 0.2], None),
    "mstep_tran_is": ({"a1.va": RP_IS}, deck(".step m1:rp list 1000 4000\n" + TRAN, "rp=1000", src="V1 in 0 SIN(0 1 1k)"),
                      "tran", tr(0.5) + tr(0.2), None),
    "second_has_group_attr": ({"a1.va": HDR + A1R + DD}, deck(".op\n.print dc v(out)"), "rows", [0.5], None),
    "same_module_two_files": ({"a1.va": HDR + A1R + HELP1, "b.va": HDR + BB + HELP2},
                              "* t\n.hdl \"a1.va\"\n.hdl \"b.va\"\nV1 in 0 DC 1\n.model m1 a1\n.model mb bb\n.model mh helper\n"
                              "Ya1 X1 in out m1\nYbb X2 out 0 mb\nYhelper X3 out mh\n.op\n.print dc v(out)\n.end\n",
                              "rows", [875.0 / 1875.0], "has the name of the module helper"),
    "ac_regime_in_rtloop": ({"a1.va": RTLOOP}, deck(".ac dec 1 1k 1k\n.print ac vm(out)", src="V1 in 0 DC 0 AC 1"),
                            "ac", [0.2], None),
}


def table(path, sig):
    with open(path) as fh:
        rows = [l.split() for l in fh.read().splitlines() if l.strip()]
    hdr = [c.upper() for c in rows[0]]
    k = hdr.index(sig)
    tk = hdr.index("TIME") if "TIME" in hdr else None
    out = []
    for c in rows[1:]:
        try:
            out.append((float(c[tk]) if tk is not None else 0.0, float(c[k])))
        except (ValueError, IndexError):
            continue
    return out


def segments(rows):
    segs, last = [], None
    for t, v in rows:
        if last is None or t < last:
            segs.append([])
        segs[-1].append((t, v))
        last = t
    return segs


def interp(rows, t):
    for (t0, v0), (t1, v1) in zip(rows, rows[1:]):
        if t0 <= t <= t1:
            return v0 if t1 == t0 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return None


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        work = tempfile.mkdtemp(prefix='pyms_stepmod_')
        env = dict(os.environ, PYMS_DIR=_PYMS_DIR, PYTHONDONTWRITEBYTECODE='1')
        env['LD_LIBRARY_PATH'] = XYCE_LIBDIR + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        bad = []
        try:
            for name, (files, text, kind, want, logwant) in CASES.items():
                d = os.path.join(work, name)
                os.makedirs(d)
                for fn, t in files.items():
                    with open(os.path.join(d, fn), 'w') as fh:
                        fh.write(t)
                with open(os.path.join(d, 'c.cir'), 'w') as fh:
                    fh.write(text)
                env.update(PYMS_CACHE=os.path.join(d, 'hc'), PYMS_VAE_CACHE=os.path.join(d, 'vc'))
                r = subprocess.run([XYCE, 'c.cir'], cwd=d, env=env, capture_output=True, text=True, timeout=1800)
                log = r.stdout + r.stderr
                got = None
                try:
                    if kind == "ac":
                        got = [v for _t, v in table(os.path.join(d, 'c.cir.FD.prn'), 'VM(OUT)')]
                    elif kind == "tran":
                        got = [interp(s, t) for s in segments(table(os.path.join(d, 'c.cir.prn'), 'V(OUT)')) for t in TIMES]
                    else:
                        got = [v for _t, v in table(os.path.join(d, 'c.cir.prn'), 'V(OUT)')]
                except (OSError, ValueError, IndexError):
                    pass
                tol = 1e-3 if kind == "tran" else 1e-6
                ok = (r.returncode == 0 and got is not None and len(got) == len(want) and None not in got
                      and all(abs(g - w) <= tol for g, w in zip(got, want))
                      and (logwant is None or logwant in log))
                print('%-24s want %s got %s rc %d%s' % (name, ['%.6g' % w for w in want],
                      None if got is None else ['-' if g is None else '%.6g' % g for g in got], r.returncode,
                      '' if logwant is None else ' warned=%s' % (logwant in log)), flush=True)
                if not ok:
                    bad.append((name, got, log[-600:]))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
