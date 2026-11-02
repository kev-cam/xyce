"""The device shell's node regularization is the device gmin of the options
(2026-10-25, PYMS round 1 E-4).

THE BUG: xyce_device_gen.py stamped a conductance to ground on every node of
a PyMS device, 1e-12 S on the ports and a fixed 1e-9 S on the internal nodes,
whatever the netlist's ``.options device gmin``. A series divider written as
a voltage contribution into an internal node and a current contribution out
of it (``V(p, m) <+ r * I(p, m); I(m, n) <+ V(m, n) / r;``, R2 = 1k to
ground, analytic out = 1/3) gave 0.333333111: a relative error of 6.7e-7,
the same with gmin=1e-12 or 1e-9 (VACASK 0.333333333).

THE FIX: the shell reads getDeviceOptions().gmin (default 1e-12), as Xyce's
own devices read it, for every node.

Through Xyce (private caches, PYMS_DIR = the PyMS copy this file sits in;
skipped without a Xyce binary).

Run:
    cd utils/PyMS && python3 vae/tests/test_device_gmin.py
"""
import os
import shutil
import subprocess
import tempfile
import unittest

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

SER = ('`include "disciplines.vams"\n`include "constants.vams"\n'
       'module a1(p, n);\n  inout p, n;\n  electrical p, n;\n  electrical m;\n'
       '  parameter real r = 1k;\n'
       '  analog begin\n    V(p, m) <+ r * I(p, m);\n    I(m, n) <+ V(m, n) / r;\n  end\nendmodule\n')
PLAIN = ('`include "disciplines.vams"\n`include "constants.vams"\n'
         'module a1(p, n);\n  inout p, n;\n  electrical p, n;\n  parameter real r = 1k;\n'
         '  analog I(p, n) <+ V(p, n) / r;\nendmodule\n')

DECK = '''* gmin: the reference divider
.hdl "a1.va"
V1 in 0 DC 1.0
.model a1__t a1
ya1 X1 in out a1__t
R2 out 0 1000.0
.options device temp=27.0 tnom=27.0 gmin={gmin}
.OP
.print dc format=noindex v(out)
.end
'''


def run_xyce(va, gmin):
    work = tempfile.mkdtemp(prefix='pyms_gmin_')
    try:
        with open(os.path.join(work, 'a1.va'), 'w') as f:
            f.write(va)
        with open(os.path.join(work, 'deck.cir'), 'w') as f:
            f.write(DECK.format(gmin=gmin))
        env = dict(os.environ, PYMS_DIR=PYMS_DIR, PYMS_CACHE=os.path.join(work, 'ch'),
                   PYMS_VAE_CACHE=os.path.join(work, 'cv'), PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run([XYCE, 'deck.cir'], cwd=work, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=600)
        prn = os.path.join(work, 'deck.cir.prn')
        if os.path.exists(prn):
            with open(prn) as f:
                prn_lines = f.read().splitlines()
            for line in prn_lines:
                parts = line.split()
                if len(parts) == 1:
                    try:
                        return r.returncode, r.stdout, float(parts[0])
                    except ValueError:
                        pass
        return r.returncode, r.stdout, None
    finally:
        shutil.rmtree(work, ignore_errors=True)


@unittest.skipUnless(os.path.exists(XYCE), 'no Xyce binary')
class TestDeviceGmin(unittest.TestCase):
    def out(self, va, gmin):
        rc, log, val = run_xyce(va, gmin)
        self.assertEqual(rc, 0, log[-2500:])
        self.assertIsNotNone(val, log[-2500:])
        return val

    def test_series_divider_with_the_default_gmin(self):
        # 1e-12 S on the internal node m: out within 1e-8 of 1/3 (it was 6.7e-7 off)
        val = self.out(SER, '1e-12')
        self.assertLess(abs(val - 1.0 / 3.0) * 3.0, 1e-8, repr(val))

    def test_the_options_gmin_reaches_the_internal_node(self):
        # with gmin = 1e-9 S on m (and on the ports) the leak is the analytic one:
        # m sees r up to p, r and R2 down to ground, and 1e-9 S to ground
        lo = self.out(SER, '1e-12')
        hi = self.out(SER, '1e-9')
        self.assertGreater(abs(hi - lo) / lo, 1e-7, f'gmin=1e-9 {hi!r} vs gmin=1e-12 {lo!r}')
        g = 1e-9
        r, r2 = 1000.0, 1000.0
        # nodal analysis: node m (conductances 1/r to in=1 V, 1/r to out, g to ground),
        # node out (1/r to m, 1/r2 and g to ground)
        a11, a12, b1 = 2 / r + g, -1 / r, 1 / r
        a21, a22 = -1 / r, 1 / r + 1 / r2 + g
        out = (a11 * 0 - a21 * b1) / (a11 * a22 - a12 * a21)
        self.assertLess(abs(hi - out) / out, 1e-9, f'{hi!r} vs nodal {out!r}')

    def test_a_plain_divider_is_unchanged(self):
        val = self.out(PLAIN, '1e-12')
        self.assertLess(abs(val - 0.5) / 0.5, 1e-8, repr(val))


if __name__ == '__main__':
    unittest.main(verbosity=2)
