"""pow with a zero base and a bias-dependent exponent has slope 0, not NaN (1.1.7).

THE DEFECT (PYMS7 code review F2, 2026-11-07, on 1.1.6): the dual-number
pow (vae/dual_ad.py) took the slope along the exponent as r.v * log(a.v);
at a zero base that is 0 * (-inf) = NaN, and every Jacobian column the
exponent depends on became NaN: `I(p,n) <+ ... + 1e-6*pow(max(V(p,n),0.0),
1.0+V(p,n))` failed every DC point (Failed Linear Solves), while a constant
exponent (pow(x, 1.5)) converged.

THE FIX: dual_ad.py -- pow(double, dual) and pow(dual, dual) give the
exponent slope 0 where the value is 0 (pow(0, y) is 0 for every y > 0).

Run:  cd utils/PyMS && python3 vae/tests/test_pow_zero_base.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.tests._decks import mod, run_decks, rv, XYCE  # noqa: E402
from vae.tests.test_exact_jacobian import Lib, close  # noqa: E402

# the base is 0 below 2 V, the exponent depends on the bias
ZB = mod("zb", "  analog I(p, n) <+ V(p, n)/1k + 1e-6*pow(max(V(p, n) - 2.0, 0.0), 1.0 + V(p, n));")


class TestLibrary(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix='pyms_pow0_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_zero_base_slope(self):
        L = Lib(ZB, self.work)
        p, n = L.nodes.index('p'), L.nodes.index('n')
        for v in (0.0, 0.5, 2.0):
            dF, _ = L.jac(L.state([v, 0.0]))
            self.assertTrue(all(x == x for x in dF[0]), (v, dF))
            self.assertTrue(close(dF[0][p], 1e-3), (v, dF))
            self.assertTrue(close(dF[0][n], -1e-3), (v, dF))
        # above 2 V the slope is the full one: d/dV (V-2)^(1+V)
        import math
        v = 3.0
        dF, _ = L.jac(L.state([v, 0.0]))
        b, e = v - 2.0, 1.0 + v
        want = 1e-3 + 1e-6 * (b ** e) * (e / b + math.log(b))
        self.assertTrue(close(dF[0][p], want), (dF[0][p], want))


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_op(self):
        bad = run_decks({"zb": (ZB, "", rv(1000.0))}, prefix="pyms_pow0x_")
        self.assertEqual(bad, [])


if __name__ == '__main__':
    unittest.main()
