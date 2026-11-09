"""A Newton iterate on the other branch of a held update is not converged
(1.1.8, BUGS.md B25).

THE DEFECT (found by the 1.1.7 round, FIX8/decks/h1.cir; 1.1.6 the same): a held
peak (`@(initial_step) pk = 0; if (V(p,n) > pk) pk = V(p,n); I(p,n) <+ V(p,n)/r
+ 1e-4*pk;`) straight on `SIN(0 1 1k)`, R2 = 1k to ground, with `.options timeint
reltol=1e-6 abstol=1e-12`: the time stopped at 1.4877 ms and Xyce printed millions
of rows there, V(out) alternating by 1.6e-7.  At the second positive peak V(p,n)
grazes its own held value: the predictor is above pk (the update is written), the
next iterate below (it is not), and the solver accepted that iterate -- a solution
of neither branch.  Xyce's default trapezoid rule carries the residual error of an
algebraic row on, alternating in sign, without damping it; near the zero crossing
the error test crushed the time step to 1e-17 s.  At the default tolerances the
same error was 2e-5 V at the end of the run.

Now the device shell reports such a Newton evaluation as not converged
(Instance::isConverged, Xyce's device convergence test): the next iteration solves
the branch the iterate is in.

After the first peak (t > 0.25 ms) pk = 1/2.1 and V(out) = (V(in) + 0.1 pk) / 2;
before it pk follows V(p,n) and V(out) = 1.1 V(in) / 2.1.

Run:  cd utils/PyMS && python3 vae/tests/test_held_branch_newton.py
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

from vae.tests._decks import XYCE, XYCE_LIBDIR  # noqa: E402

VA = """`include "disciplines.vams"
module stp2(p, n);
inout p, n; electrical p, n;
parameter real r = 1k;
real pk;
analog begin
  @(initial_step) pk = 0;
  if (V(p,n) > pk) pk = V(p,n);
  I(p,n) <+ V(p,n)/r + 1e-4*pk;
end
endmodule
"""
DECK = """* held peak on a sine
V1 in 0 SIN(0 1 1k)
YSTP2 X1 in out m1
R2 out 0 1k
.model m1 stp2
.hdl "stp2.va"
%s
.tran 10u 2m
.print tran precision=15 v(in) v(out)
.end
"""
TIGHT = ".options timeint reltol=1e-6 abstol=1e-12"


def table(path):
    rows = []
    with open(path) as fh:
        for ln in fh:
            c = ln.split()
            if len(c) < 4 or c[0] == "Index":
                continue
            try:
                rows.append((float(c[1]), float(c[2]), float(c[3])))
            except ValueError:
                pass
    return rows


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestHeldBranchNewton(unittest.TestCase):

    def run_deck(self, options, timeout):
        d = tempfile.mkdtemp(prefix='pyms_heldnewton_')
        try:
            with open(os.path.join(d, 'stp2.va'), 'w') as fh:
                fh.write(VA)
            with open(os.path.join(d, 'c.cir'), 'w') as fh:
                fh.write(DECK % options)
            env = dict(os.environ, PYMS_DIR=_PYMS_DIR, PYTHONDONTWRITEBYTECODE='1',
                       PYMS_CACHE=os.path.join(d, 'hc'), PYMS_VAE_CACHE=os.path.join(d, 'vc'))
            env['LD_LIBRARY_PATH'] = XYCE_LIBDIR + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
            env.pop('PYMS_DEFINES', None)
            try:
                r = subprocess.run([XYCE, 'c.cir'], cwd=d, env=env, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                self.fail("Xyce did not finish in %d s (the time step stalls)" % timeout)
            self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-800:])
            return table(os.path.join(d, 'c.cir.prn'))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def check(self, rows, tol):
        # the algebraic equation of every printed point: before the peak
        # V(out) = 1.1 V(in) / 2.1; after it V(out) = (V(in) + 0.1 pk) / 2 with
        # the held pk (the largest V(p,n) the run accepted: the median of the
        # rows' estimates; it is 1/2.1 to the time step's resolution)
        pre = [abs(vo - 1.1 * vi / 2.1) for t, vi, vo in rows if t < 0.2e-3]
        ests = sorted(10 * (2 * vo - vi) for t, vi, vo in rows if t > 0.3e-3)
        pk = ests[len(ests) // 2]
        post = [abs(vo - (vi + 0.1 * pk) / 2) for t, vi, vo in rows if t > 0.3e-3]
        print('  rows %d, pk %.12f (1/2.1 = %.12f), max error before the peak %.3g, after %.3g, end at %.9g'
              % (len(rows), pk, 1 / 2.1, max(pre), max(post), rows[-1][0]), flush=True)
        self.assertAlmostEqual(rows[-1][0], 2e-3, delta=1e-12)
        self.assertLess(abs(pk - 1 / 2.1), 1e-3)
        self.assertLess(max(pre), tol)
        self.assertLess(max(post), tol)

    def test_tight_tolerances_finish(self):
        rows = self.run_deck(TIGHT, 300)
        print('tight: %d rows' % len(rows))
        self.assertLess(len(rows), 50000)
        self.check(rows, 1e-8)

    def test_default_tolerances_no_ringing(self):
        rows = self.run_deck("", 300)
        print('default: %d rows' % len(rows))
        self.check(rows, 1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
