"""A device parameter given as an expression of a .global_param, or of a .param
that a .DC / .STEP sweeps (1.1.8, BUGS.md A23, A24).

THE DEFECTS (vamos round 8, sv2ghdl FIX2E8 raw/probe.py r21-r24, q1-q12, on 1.1.7):
  * A23: ``.global_param rr=2k`` with ``.model c1 a2 r={rr}`` gave 0.5 for 1/3
    (the declaration default), exit 0; the same on the Y line and for an
    integer parameter.  A built-in R read it right.
  * A24: ``.param rr=2k``, card ``r={rr}``, ``.DC rr LIST 2k 4k`` gave 0.5 0.2
    for 1/3 0.2: the first point equal to the nominal ran with the default
    (also .STEP, .STEP + .TRAN, an expression ``{2*rr}``, the Y line).
  One cause: such a value is a Xyce "dependent parameter".  setParams leaves
  the member at its default and only DeviceEntity::updateDependentParameters()
  writes the expression's value; every built-in device calls it in its
  constructor, the generated PyMS shell did not.  Xyce reprocesses an instance
  only when the global's value changes, so a .global_param that never changes,
  and a first sweep point equal to the nominal, kept the default.

Every deck: V1 = 1 V, the module between in and out, R2 = 1k to ground:
V(out) = 1k / (1k + Req), Req = 1 / (k/r + g).

Run:  cd utils/PyMS && python3 vae/tests/test_global_dependent_params.py
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

from vae.tests._decks import HDR, XYCE, XYCE_LIBDIR  # noqa: E402

A2 = HDR + """module a2(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  parameter real g = 0 (* type="instance" *);
  parameter integer k = 1;
  analog I(p, n) <+ V(p, n) * (k / r + g);
endmodule
"""
Y = ".model c1 a2 %s\nya2 y1 in out c1 %s\n"


def deck(card, line, extra, analysis=".dc V1 1 1 1"):
    return ("* t\n.hdl \"a2.va\"\nV1 in 0 1\n" + Y % (card, line) + "R2 out 0 1k\n" + extra
            + analysis + "\n.print dc v(out)\n.end\n")


T3, T5, T6, T2, T8 = 1 / 3, 0.2, 2 / 3, 0.5, 0.8
# name -> (deck, V(out) of every printed row)
CASES = {
    "a23_global_card": (deck("r={rr}", "", ".global_param rr=2k\n"), [T3]),
    "a23_global_line": (deck("", "g={gg}", ".global_param gg=1m\n"), [T6]),
    "a23_global_card_int": (deck("k={kk}", "", ".global_param kk=2\n"), [T6]),
    "a24_dc_card_nominal_first": (deck("r={rr}", "", ".param rr=2k\n", ".dc rr list 2k 4k"), [T3, T5]),
    "a24_dc_line_nominal_first": (deck("", "g={gg}", ".param gg=1m\n", ".dc gg list 1m 3m"), [T6, T8]),
    "a24_dc_card_expr_of_swept": (deck("r={2*rr}", "", ".param rr=1k\n", ".dc rr list 1k 2k"), [T3, T5]),
    "a24_step_card_nominal_first": (deck("r={rr}", "", ".param rr=2k\n.step rr list 2k 4k\n"), [T3, T5]),
    # controls that were right already
    "ctl_dc_nominal_second": (deck("r={rr}", "", ".param rr=2k\n", ".dc rr list 4k 2k"), [T5, T3]),
    "ctl_param_no_sweep": (deck("r={rr}", "", ".param rr=2k\n"), [T3]),
    "ctl_default": (deck("", "", ""), [T2]),
}


def rows(path):
    out = []
    with open(path) as fh:
        for ln in fh:
            c = ln.split()
            if not c or c[0] == "Index" or c[0].lower().startswith("end"):
                continue
            out.append(float(c[-1]))
    return out


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestGlobalDependentParams(unittest.TestCase):

    def test_decks(self):
        work = tempfile.mkdtemp(prefix='pyms_globdep_')
        env = dict(os.environ, PYMS_DIR=_PYMS_DIR, PYTHONDONTWRITEBYTECODE='1')
        env['LD_LIBRARY_PATH'] = XYCE_LIBDIR + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        env.pop('PYMS_DEFINES', None)
        bad = []
        try:
            for name, (text, want) in CASES.items():
                d = os.path.join(work, name)
                os.makedirs(d)
                with open(os.path.join(d, 'a2.va'), 'w') as fh:
                    fh.write(A2)
                with open(os.path.join(d, 'c.cir'), 'w') as fh:
                    fh.write(text)
                env.update(PYMS_CACHE=os.path.join(d, 'hc'), PYMS_VAE_CACHE=os.path.join(d, 'vc'))
                r = subprocess.run([XYCE, 'c.cir'], cwd=d, env=env, capture_output=True, text=True, timeout=1800)
                try:
                    got = rows(os.path.join(d, 'c.cir.prn'))
                except OSError:
                    got = None
                ok = (r.returncode == 0 and got is not None and len(got) == len(want)
                      and all(abs(a - b) <= 1e-6 for a, b in zip(got, want)))
                print('%-28s want %s got %s rc %d' % (name, ['%.6g' % w for w in want],
                      got and ['%.6g' % g for g in got], r.returncode), flush=True)
                if not ok:
                    bad.append((name, got, (r.stdout + r.stderr)[-500:]))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
