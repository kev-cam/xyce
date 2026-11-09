"""A gated contribution whose untaken arm is not finite (1.1.9, BUGS.md B24).

THE DEFECT (the 1.1.6 code review F3; REV7/t3): a branch with a current
contribution in one arm and a voltage contribution in the other of a run-time
condition is one blended row, and a gated contribution was the product
e * S of its value and the 0/1 indicator:

    parameter real rs = rs0*($temperature/300.15);
    if (rs > 0) I(p,m) <+ V(p,m)/rs; else V(p,m) <+ 0;

with rs0 = 0 gave 1/rs * V * 0 = NaN in the value and the Jacobian: every
point failed (tc0, tc2; tv0 with a variable computed from $temperature).

THE FIX (ginac_emitter.py, the contribution walk): a gated contribution is
an intermediate whose value is the C++ ternary (S != 0 ? e : 0); the
dual-number Jacobian follows the taken arm only.

Expected V(2) = (1k + rs) / (2k + rs), rs = rs0*T/300.15 (V1 = 1 V, R1 = 1k,
the module from 2 to ground with its 1k).  Private caches, PYMS_DIR = the
PyMS copy this file sits in.

Run:
    cd utils/PyMS && python3 vae/tests/test_switch_row_select.py
"""
import os
import shutil
import subprocess
import tempfile
import unittest

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TC = '''`include "disciplines.vams"
module tc(p, n);
inout p, n; electrical p, n, m;
parameter real rs0 = 0;
parameter real rs = rs0*($temperature/300.15);
analog begin
  if (rs > 0) I(p,m) <+ V(p,m)/rs; else V(p,m) <+ 0;
  I(m,n) <+ V(m,n)/1k;
end
endmodule
'''
TV = '''`include "disciplines.vams"
module tv(p, n);
inout p, n; electrical p, n, m;
parameter real rs0 = 0;
real rt;
analog begin
  rt = rs0*($temperature/300.15);
  if (rt > 0) I(p,m) <+ V(p,m)/rt; else V(p,m) <+ 0;
  I(m,n) <+ V(m,n)/1k;
end
endmodule
'''
# a current contribution gated alone (no switch row): g/abs(g) = 0/0 untaken at g = 0
TG = '''`include "disciplines.vams"
module tg(p, n);
inout p, n; electrical p, n;
parameter real g0 = 0;
real g;
analog begin
  g = g0*($temperature/300.15);
  I(p,n) <+ V(p,n)/1k;
  if (g > 0) I(p,n) <+ V(p,n)*1m*g/abs(g);
end
endmodule
'''

DECK = '''* switch row
V1 1 0 DC 1
R1 1 2 1k
Y{M} x1 2 0 m1
.model m1 {m} {card}
.hdl "a.va"
.step temp list 27 327.15
.op
.print dc format=noindex v(2)
.end
'''


def run_xyce(va, m, card):
    work = tempfile.mkdtemp(prefix='pyms_swsel_')
    try:
        with open(os.path.join(work, 'a.va'), 'w') as f:
            f.write(va)
        with open(os.path.join(work, 'd.cir'), 'w') as f:
            f.write(DECK.format(M=m.upper(), m=m, card=card))
        env = dict(os.environ, PYMS_DIR=PYMS_DIR, PYMS_CACHE=os.path.join(work, 'ch'),
                   PYMS_VAE_CACHE=os.path.join(work, 'cv'), PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run([XYCE, 'd.cir'], cwd=work, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=600)
        vals = []
        prn = os.path.join(work, 'd.cir.prn')
        if os.path.exists(prn):
            with open(prn) as f:
                for line in f.read().splitlines():
                    s = line.split()
                    if len(s) == 1:
                        try:
                            vals.append(float(s[0]))
                        except ValueError:
                            pass
        return r.returncode, r.stdout, vals
    finally:
        shutil.rmtree(work, ignore_errors=True)


def expect(rs0):
    out = []
    for T in (300.15, 600.3):
        rs = rs0 * T / 300.15
        out.append((1000.0 + rs) / (2000.0 + rs))
    return out


@unittest.skipUnless(os.path.exists(XYCE), 'no Xyce binary')
class TestSwitchRowSelect(unittest.TestCase):
    def check(self, va, m, card, want):
        rc, log, vals = run_xyce(va, m, card)
        self.assertEqual(rc, 0, log[-2500:])
        self.assertEqual(len(vals), len(want), log[-2500:])
        for got, w in zip(vals, want):
            self.assertAlmostEqual(got, w, delta=1e-8, msg=vals)

    def test_thermal_parameter_rs0_zero(self):
        self.check(TC, 'tc', 'rs0=0', expect(0.0))

    def test_thermal_parameter_default(self):
        self.check(TC, 'tc', '', expect(0.0))

    def test_thermal_parameter_rs0_1k(self):
        self.check(TC, 'tc', 'rs0=1k', expect(1000.0))

    def test_thermal_variable_rs0_zero(self):
        self.check(TV, 'tv', 'rs0=0', expect(0.0))

    def test_thermal_variable_rs0_1k(self):
        self.check(TV, 'tv', 'rs0=1k', expect(1000.0))

    def test_gated_current_alone(self):
        # g0 = 0: the module is its 1k (V(2) = 0.5); g0 = 1m: 1k || 1k (V(2) = 1/3)
        self.check(TG, 'tg', 'g0=0', [0.5, 0.5])
        want = [1.0 / 3.0, 1.0 / 3.0]
        self.check(TG, 'tg', 'g0=1m', want)


if __name__ == '__main__':
    unittest.main(verbosity=2)
