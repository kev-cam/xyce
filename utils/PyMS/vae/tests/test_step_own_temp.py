"""A .step of a module's own TEMP parameter to the circuit temperature (1.1.9, BUGS.md A20).

THE DEFECT: ``.step m1:TEMP list 50 27 80`` with the circuit at 27 C gave
323.15 / 323.15 / 353.15 K: the second point kept 50.  Xyce's device manager
writes the circuit temperature into the TEMP of every device at a
temperature change (DeviceMgr::updateTemperature); the shell undoes that
write (1.1.6), and told it by the value alone: a TEMP equal to the circuit
temperature was taken for the manager's write and went back to the previous
value.

THE FIX (xyce_device_gen.py): the manager writes XYCEADMSINSTTEMP /
XYCEADMSMODTEMP (K) right after TEMP; processParams reads that parameter's
given flag (and clears it), so only that write is undone.  A TEMP a .step
writes is the netlist's and given, whatever its value.  The same for the
shell's instance TEMP (the device temperature of a module without a TEMP
parameter) and for an instance-parameter TEMP.

Module: V(o) = TEMP + 273.15 when TEMP > -500 (given), else $temperature;
V(q) = $temperature.  Private caches, PYMS_DIR = the PyMS copy this file
sits in.

Run:
    cd utils/PyMS && python3 vae/tests/test_step_own_temp.py
"""
import os
import shutil
import subprocess
import tempfile
import unittest

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'


def et(kind):
    attr = ' (*type="instance"*)' if kind == 'inst' else ''
    return HDR + '''module et(o, q);
  inout o, q;
  electrical o, q;
  parameter real TEMP = -1e3%s;
  analog begin
    V(o) <+ (TEMP > -500) ? TEMP + 273.15 : $temperature;
    V(q) <+ $temperature;
  end
endmodule
''' % attr


# a module without TEMP: the shell's instance TEMP is the device temperature
SHELL = HDR + '''module et(o, q);
  inout o, q;
  electrical o, q;
  analog begin
    V(o) <+ $temperature;
    V(q) <+ $temperature;
  end
endmodule
'''

DECK = '''* own TEMP stepped
.hdl "a.va"
R1 o 0 1k
R2 q 0 1k
.model m1 et {card}
Yet X1 o q m1 {inst}
{extra}
.op
.print dc format=noindex v(o) v(q)
.end
'''


def run_xyce(va, extra, card='', inst=''):
    work = tempfile.mkdtemp(prefix='pyms_owntemp_')
    try:
        with open(os.path.join(work, 'a.va'), 'w') as f:
            f.write(va)
        with open(os.path.join(work, 'x.cir'), 'w') as f:
            f.write(DECK.format(card=card, inst=inst, extra=extra))
        env = dict(os.environ, PYMS_DIR=PYMS_DIR, PYMS_CACHE=os.path.join(work, 'ch'),
                   PYMS_VAE_CACHE=os.path.join(work, 'cv'), PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run([XYCE, 'x.cir'], cwd=work, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=600)
        rows = []
        prn = os.path.join(work, 'x.cir.prn')
        if os.path.exists(prn):
            with open(prn) as f:
                for line in f.read().splitlines():
                    try:
                        v = [float(x) for x in line.split()]
                    except ValueError:
                        continue
                    if len(v) == 2:
                        rows.append(v)
        return r.returncode, r.stdout, rows
    finally:
        shutil.rmtree(work, ignore_errors=True)


@unittest.skipUnless(os.path.exists(XYCE), 'no Xyce binary')
class TestStepOwnTemp(unittest.TestCase):
    def check(self, va, extra, want, **k):
        rc, log, rows = run_xyce(va, extra, **k)
        self.assertEqual(rc, 0, log[-2500:])
        self.assertEqual(len(rows), len(want), (rows, log[-1500:]))
        for got, w in zip(rows, want):
            self.assertAlmostEqual(got[0], w[0], delta=1e-6, msg=rows)
            self.assertAlmostEqual(got[1], w[1], delta=1e-6, msg=rows)

    def test_model_temp_stepped_to_the_circuit_temperature(self):
        self.check(et('model'), '.step m1:TEMP list 50 27 80',
                   [(323.15, 300.15), (300.15, 300.15), (353.15, 300.15)])

    def test_model_temp_stepped_to_the_circuit_temperature_127(self):
        self.check(et('model'), '.options device temp=127\n.step m1:TEMP list 50 127 80',
                   [(323.15, 400.15), (400.15, 400.15), (353.15, 400.15)])

    def test_instance_temp_on_the_card_stepped(self):
        # an instance-parameter TEMP (on the card since 1.1.9, A25)
        self.check(et('inst'), '.step m1:TEMP list 50 27 80',
                   [(323.15, 300.15), (300.15, 300.15), (353.15, 300.15)])

    def test_step_temp_still_undoes_the_managers_write(self):
        # controls: the device manager's write at .step temp is still undone
        self.check(et('model'), '.step temp list 27 127 27',
                   [(300.15, 300.15), (400.15, 400.15), (300.15, 300.15)])
        self.check(et('model'), '.step temp list 27 127 50',
                   [(323.15, 300.15), (323.15, 400.15), (323.15, 323.15)], card='TEMP=50')
        self.check(SHELL, '.step temp list 27 127', [(400.15, 400.15), (400.15, 400.15)], inst='TEMP=127')


if __name__ == '__main__':
    unittest.main(verbosity=2)
