"""The Jacobian of a math library is exact (1.1.5).

THE DEFECT (PYMS5 skeptic, 2026-11-01, on 1.1.4; since 739eb5ca, before
1.1.0): build_vae_so.py's vae_jacobian was a forward difference quotient of
the eval, with a step of 1e-4*(|V|+1) on the ABSOLUTE node potential, and
the GiNaC analytic Jacobian was never called.  The truncation error grew
with the bias: the ac gain of a diode on 1 kOhm -0.29 %, of a diode whose
cathode sits at 10 V -2.1 %, at 50 V -9.2 % (a passive diode + R biased from
50 V had a gain of 1.216 instead of 0.99948), Mextram bjtd504 with its
emitter at 20 V +7.5 %; a 45 V rectifier's transient stalled in 1.8e7 steps
of 3e-17 s.

THE FIX: vae/dual_ad.py -- the eval code compiled a second time on dual
numbers (forward-mode automatic differentiation): one evaluation gives every
column, exact to rounding, and it differentiates the very value path of the
eval (ternaries, run-time loops, held values, regime latches).

The library tests build a math library with build_vae_so.py, load it with
ctypes and compare vae_jacobian with the analytic derivatives (relative
1e-9; the difference quotient of 1.1.4 misses them by 1e-5 to 1e-1); the
Xyce tests compare ac gains with the analytic small-signal gain of the
operating point Xyce found (relative 1e-6).

Run:  cd utils/PyMS && python3 vae/tests/test_exact_jacobian.py
"""
from __future__ import annotations

import ctypes
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

from vae.ginac_emitter import VAE_STATE_CAP  # noqa: E402
from vae.tests._decks import HDR, mod, XYCE, XYCE_LIBDIR  # noqa: E402

BUILD = os.path.join(_PYMS_DIR, 'vae', 'build_vae_so.py')
T0 = 300.15                         # the device temperature of these tests
VT = 8.617086918058125e-05 * T0      # the eval code's $vt there (Xyce's k/q, 1.1.6)


class VaeState(ctypes.Structure):
    _fields_ = [('V', ctypes.c_double * VAE_STATE_CAP), ('Vt', ctypes.c_double),
                ('I', ctypes.c_double * VAE_STATE_CAP), ('t', ctypes.c_double),
                ('an', ctypes.c_int), ('phase', ctypes.c_int),
                ('Hr', ctypes.POINTER(ctypes.c_double)), ('Hw', ctypes.POINTER(ctypes.c_double)),
                ('Hl', ctypes.POINTER(ctypes.c_double)), ('T', ctypes.c_double)]


class Lib:
    """A math library of one module, built into a private directory."""

    def __init__(self, va_text, work, full=False):
        self.dir = tempfile.mkdtemp(dir=work)
        va = os.path.join(self.dir, 'c.va')
        with open(va, 'w') as fh:
            fh.write(va_text)
        pf = os.path.join(self.dir, 'params.txt')
        with open(pf, 'w') as fh:
            fh.write('__GIVEN__=\n')
        so = os.path.join(self.dir, 'c.so')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYMS_AD_FULL', None)
        if full:
            env['PYMS_AD_FULL'] = '1'
        r = subprocess.run([sys.executable, BUILD, va, so, pf], capture_output=True, text=True,
                           env=env, timeout=900)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-2000:])
        self.lib = ctypes.CDLL(so)
        self.n = self.lib.vae_n_nodes()
        self.nc = self.lib.vae_n_currents()
        self.nb = self.lib.vae_n_branches()
        self.nh = self.lib.vae_n_held()
        self.nt = self.n + self.nc
        self.lib.vae_branch_label.restype = ctypes.c_char_p
        self.lib.vae_node_label.restype = ctypes.c_char_p
        self.labels = [self.lib.vae_branch_label(i).decode() for i in range(self.nb)]
        self.nodes = [self.lib.vae_node_label(i).decode() for i in range(self.n)]
        with open(os.path.join(self.dir, 'c.so.build', 'wrap.cpp')) as fh:
            self.plan_line = next((l for l in fh if l.startswith('// The Jacobian (')), '')
        self.Hr = (ctypes.c_double * max(self.nh, 1))()
        self.Hw = (ctypes.c_double * max(self.nh, 1))()
        if self.nh:
            self.lib.vae_held_init(self.Hr)

    def state(self, v, i=(), an=0, phase=0, held=False):
        s = VaeState()
        for k, x in enumerate(v):
            s.V[k] = x
        for k, x in enumerate(i):
            s.I[k] = x
        s.Vt, s.T, s.an, s.phase = VT, T0, an, phase
        if held and self.nh:
            s.Hr = ctypes.cast(self.Hr, ctypes.POINTER(ctypes.c_double))
            s.Hw = ctypes.cast(self.Hw, ctypes.POINTER(ctypes.c_double))
        return s

    def eval(self, s):
        F = (ctypes.c_double * max(self.nb, 1))()
        Q = (ctypes.c_double * max(self.nb, 1))()
        self.lib.vae_eval(ctypes.byref(s), F, Q)
        return list(F)[:self.nb], list(Q)[:self.nb]

    def jac(self, s):
        dF = (ctypes.c_double * max(self.nb * self.nt, 1))()
        dQ = (ctypes.c_double * max(self.nb * self.nt, 1))()
        self.lib.vae_jacobian(ctypes.byref(s), dF, dQ)
        return ([[dF[r * self.nt + c] for c in range(self.nt)] for r in range(self.nb)],
                [[dQ[r * self.nt + c] for c in range(self.nt)] for r in range(self.nb)])


def close(a, b, rel=1e-9, ab=1e-30):
    return abs(a - b) <= rel * max(abs(a), abs(b)) + ab


DIODE = mod("d1", "  parameter real is = 1e-14;\n  analog I(p, n) <+ is * (limexp(V(p, n) / $vt) - 1);")


class TestLibrary(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix='pyms_jac_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_diode_at_50v(self):
        # anode 50.6 V, cathode 50 V: dI/dV(p) = is/vt * exp(0.6/vt) exactly
        L = Lib(DIODE, self.work)
        dF, dQ = L.jac(L.state([50.6, 50.0]))
        g = 1e-14 / VT * math.exp(0.6 / VT)
        p, n = L.nodes.index('p'), L.nodes.index('n')
        self.assertTrue(close(dF[0][p], g), (dF[0][p], g))
        self.assertTrue(close(dF[0][n], -g), (dF[0][n], -g))

    def test_charge(self):
        # Q = c0 * V^2 / 2 + c1 * V^3: dQ/dV = c0 V + 3 c1 V^2
        L = Lib(mod("c1", "  analog I(p, n) <+ ddt(1e-12 * V(p, n) * V(p, n) / 2 + 1e-13 * pow(V(p, n), 3));"), self.work)
        v = 30.7
        dF, dQ = L.jac(L.state([v + 12.0, 12.0]))
        want = 1e-12 * v + 3e-13 * v * v
        self.assertTrue(close(dQ[0][L.nodes.index('p')], want), (dQ[0], want))

    def test_select_untaken_branch(self):
        # the untaken branch divides by zero: no NaN in the slope of the taken one
        va = mod("s1", "  real g;\n  analog begin\n    if (V(p, n) > 0) g = V(p, n) * V(p, n);\n"
                       "    else g = 1 / (V(p, n) * 0);\n    I(p, n) <+ 1e-3 * g;\n  end")
        L = Lib(va, self.work)
        dF, _ = L.jac(L.state([3.0, 1.0]))
        self.assertTrue(close(dF[0][L.nodes.index('p')], 1e-3 * 2 * 2.0), dF)

    def test_runtime_loop(self):
        # a loop the run time ends: x = V^k with k from a voltage condition
        va = mod("l1", "  integer k;\n  real x;\n  analog begin\n    x = 1; k = 0;\n"
                       "    while (k < 3 && V(p, n) > -100) begin\n      x = x * V(p, n);\n      k = k + 1;\n    end\n"
                       "    I(p, n) <+ 1e-3 * x;\n  end")
        L = Lib(va, self.work)
        v = 1.7
        dF, _ = L.jac(L.state([v + 20.0, 20.0]))
        self.assertTrue(close(dF[0][L.nodes.index('p')], 3e-3 * v * v), dF)

    def test_sqrt_at_zero(self):
        # sqrt(x*x + 0) at x = 0: a slope 0/0 -> 0, never NaN
        va = mod("q1", "  analog I(p, n) <+ 1e-3 * V(p, n) + 1e-3 * sqrt(V(p, n) * V(p, n) * 0 + (V(p) - V(p)));")
        L = Lib(va, self.work)
        dF, _ = L.jac(L.state([0.0, 0.0]))
        self.assertTrue(all(x == x for x in dF[0]), dF)
        self.assertTrue(close(dF[0][L.nodes.index('p')], 1e-3), dF)

    def test_branch_current_column(self):
        # V(p, n) <+ 2k * I(p, n) + 1e-6 * I(p, n)^2: the current column exact
        va = mod("v1", "  analog V(p, n) <+ 2000 * I(p, n) + 1e6 * I(p, n) * I(p, n);")
        L = Lib(va, self.work)
        self.assertEqual(L.nc, 1)
        i = 3.5e-3
        dF, _ = L.jac(L.state([50.0, 40.0], [i]))
        row = [r for r in range(L.nb) if dF[r][L.n] != 0.0]
        self.assertTrue(row, dF)
        got = abs(dF[row[0]][L.n])
        self.assertTrue(close(got, 2000 + 2e6 * i), (dF, 2000 + 2e6 * i))

    def test_initial_step_value_held(self):
        # @(initial_step) v0 = V(p, n): at the first point (phase 0) v0 moves
        # with V (slope 1/2000 + 1/2000); at a later point (phase 1) it is the
        # held value (slope 1/2000)
        va = mod("h1", "  real v0;\n  analog begin\n    @(initial_step) v0 = V(p, n);\n"
                       "    I(p, n) <+ V(p, n) / 2000 + v0 / 2000;\n  end")
        L = Lib(va, self.work)
        p = L.nodes.index('p')
        s0 = L.state([0.7, 0.0], an=1, phase=0, held=True)
        dF0, _ = L.jac(s0)
        self.assertTrue(close(dF0[0][p], 1e-3), dF0)
        L.eval(s0)                          # writes the held value
        for k in range(L.nh):
            L.Hr[k] = L.Hw[k]
        dF1, _ = L.jac(L.state([0.9, 0.0], an=1, phase=1, held=True))
        self.assertTrue(close(dF1[0][p], 0.5e-3), dF1)


MOS = mod("m1", "  inout d, g, s, b;\n  electrical d, g, s, b, di;\n  real vgs, vds, ids, k;\n"
               "  analog begin\n    vgs = V(g, s); vds = V(di, s); k = 0; ids = 0;\n"
               "    while (k < 2 && vds > -10) begin ids = ids + 1e-4 * pow(vgs - 0.4 + 0.1 * k, 2) * tanh(3 * vds); k = k + 1; end\n"
               "    if (vgs < 0.4) ids = 1e-12 * vds;\n"
               "    I(di, s) <+ ids;\n    I(d, di) <+ V(d, di) / 50;\n    I(g, s) <+ ddt(1e-15 * vgs * vgs + 2e-15 * V(g, b));\n"
               "    I(b, s) <+ 1e-14 * (exp(V(b, s) / $vt) - 1);\n  end",
          ports="d, g, s, b", decl="")


class TestSparsePlan(unittest.TestCase):
    """The sparse plan (dependency analysis, colouring, variables kept
    double) gives the full plan's Jacobian."""

    def test_sparse_equals_full(self):
        work = tempfile.mkdtemp(prefix='pyms_jacs_')
        try:
            S, Fu = Lib(MOS, work), Lib(MOS, work, full=True)
            self.assertIn('(sparse:', S.plan_line)
            self.assertIn('(full: requested', Fu.plan_line)
            import random
            rnd = random.Random(3)
            for _ in range(40):
                v = [rnd.uniform(-1, 2) for _ in range(S.n)]
                for ph in (0, 1):
                    a, b = S.jac(S.state(v, phase=ph)), Fu.jac(Fu.state(v, phase=ph))
                    for A, B in zip(a, b):
                        for ra, rb in zip(A, B):
                            for x, y in zip(ra, rb):
                                self.assertTrue(close(x, y, 1e-13), (x, y))
        finally:
            shutil.rmtree(work, ignore_errors=True)


def run_xyce(deck, work, name):
    d = os.path.join(work, name)
    os.makedirs(d)
    with open(os.path.join(d, 'c.va'), 'w') as fh:
        fh.write(DIODE)
    with open(os.path.join(d, 'c.cir'), 'w') as fh:
        fh.write(deck)
    env = dict(os.environ, PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(d, 'hc'),
               PYMS_VAE_CACHE=os.path.join(d, 'vc'), PYTHONDONTWRITEBYTECODE='1')
    env['LD_LIBRARY_PATH'] = XYCE_LIBDIR + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    r = subprocess.run([XYCE, 'c.cir'], cwd=d, env=env, capture_output=True, text=True, timeout=1800)
    return d, r


def cols(path):
    with open(path) as fh:
        rows = [l.split() for l in fh.read().splitlines() if l.strip()]
    hdr = [c.upper() for c in rows[0]]
    return {h: float(rows[1][k]) for k, h in enumerate(hdr)}


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):
    """ac gain of a diode d1 from a source at E0 (anode) to the node d, R to
    ground: |v(d)| = gd / (gd + 1/R) with gd = is/vt exp((V(in)-V(d))/vt) of
    the operating point Xyce printed."""

    def test_ac_gain_on_bias(self):
        work = tempfile.mkdtemp(prefix='pyms_jacx_')
        bad = []
        try:
            for e0, r in ((10, 9.3e3), (50, 49.3e3), (50, 4.93e3)):
                name = 'ac_%d_%d' % (e0, r)
                ckt = ('* %s\n.hdl "c.va"\nV1 in 0 DC %g AC 1\n.model m1 d1\nYd1 X1 in d m1\nR1 d 0 %g\n'
                       % (name, e0, r))
                d, res = run_xyce(ckt + '.op\n.print dc v(in) v(d)\n.end\n', work, name + '_op')
                op = cols(os.path.join(d, 'c.cir.prn'))
                vd = op['V(D)']
                gd = 1e-14 / VT * math.exp((op['V(IN)'] - vd) / VT)
                want = gd / (gd + 1.0 / r)
                d, res = run_xyce(ckt + '.ac dec 1 1k 1k\n.print ac vm(d)\n.end\n', work, name)
                got = cols(os.path.join(d, 'c.cir.FD.prn'))['VM(D)']
                print('%-14s V(d) %.9f  ac %.9f  analytic %.9f' % (name, vd, got, want), flush=True)
                if res.returncode != 0 or abs(got - want) > 1e-6 * want:
                    bad.append((name, got, want))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(bad, [])

    def test_flat_branch_equation(self):
        # V(p, n) <+ tanh(V(p, n)): the solution V(p, n) = 0 has an exactly
        # zero slope of its branch equation (1 - tanh'(0)); the exact
        # Jacobian alone is singular there (VACASK fails), the central
        # difference of that row lets Newton through: V(out) = 1
        from vae.tests._decks import run_decks
        bad = run_decks({"v_tanh": (mod("v_tanh", "  analog V(p, n) <+ tanh(V(p, n));"), "", 1.0)},
                        "pyms_jacflat_")
        self.assertEqual(bad, [])

    def test_nan_value_fails_loudly(self):
        # a contribution whose value is not a number: the device shell sets
        # the value to 0, so its Jacobian row must stay NaN (as the
        # difference quotient's was) and the operating point fail -- a
        # finite slope there let the run end rc 0 with V(out) = 1, silently
        work = tempfile.mkdtemp(prefix='pyms_jacnan_')
        try:
            d = os.path.join(work, 'nan')
            os.makedirs(d)
            with open(os.path.join(d, 'c.va'), 'w') as fh:
                fh.write(mod("vnan", "  analog V(p, n) <+ sqrt(-1.0 - V(p, n) * 0);"))
            with open(os.path.join(d, 'c.cir'), 'w') as fh:
                fh.write('* nan\n.hdl "c.va"\nV1 in 0 DC 1\n.model m1 vnan\nYvnan X1 in out m1\nR2 out 0 1k\n'
                         '.op\n.print dc v(out)\n.end\n')
            env = dict(os.environ, PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(d, 'hc'),
                       PYMS_VAE_CACHE=os.path.join(d, 'vc'), PYTHONDONTWRITEBYTECODE='1')
            env['LD_LIBRARY_PATH'] = XYCE_LIBDIR + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
            r = subprocess.run([XYCE, 'c.cir'], cwd=d, env=env, capture_output=True, text=True, timeout=1800)
            self.assertNotEqual(r.returncode, 0, (r.stdout + r.stderr)[-800:])
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
