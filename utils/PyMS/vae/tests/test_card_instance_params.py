"""An instance parameter on a .model card (1.1.9, BUGS.md A25).

THE DEFECT: ``(* type="instance" *) parameter real g`` given on the .model
card was ignored with only Xyce's "No model parameter G found for model C1
of type A2, parameter ignored" and exit 0; the declaration default was used.
openvaf-r/VACASK take the card value as the default of every instance of
that model, and a value on the instance line wins (FIX10/vac/a25: card r=2k
-> 1/3, card 2k + instance 500 -> 2/3, neither -> 1/2).

THE FIX (xyce_device_gen.py): every instance parameter is registered on the
model too (the xyceAlsoModel machinery of 1.1.1, now for all of them); an
instance that does not give it takes the card's value in processParams,
counts it as given for $param_given, and a parameter whose default reads
another parameter is no longer re-evaluated over a card value. A .step of
the card value, or a card value that is an expression of a swept .param,
reaches every instance that does not give its own.

Through Xyce on the reference divider (V1 = 1 V into the module, R2 = 1k;
private caches, PYMS_DIR = the PyMS copy this file sits in).

Run:
    cd utils/PyMS && python3 vae/tests/test_card_instance_params.py
"""
import os
import shutil
import subprocess
import tempfile
import unittest

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

HEAD = ('`include "disciplines.vams"\n`include "constants.vams"\n'
        'module a1(p, n);\n  inout p, n;\n  electrical p, n;\n')
VA = HEAD + ('  (*type="instance"*) parameter real r = 1k;\n'
             '  analog I(p, n) <+ V(p, n) / r;\nendmodule\n')
# $param_given sees a card value
VA_GIVEN = HEAD + ('  (*type="instance"*) parameter real r = 1k;\n'
                   '  analog I(p, n) <+ V(p, n) / ($param_given(r) ? r : 4000.0);\nendmodule\n')
# an instance parameter whose default reads another one: the card value is not re-evaluated away
VA_DEP = HEAD + ('  parameter real k = 1.0;\n'
                 '  (*type="instance"*) parameter real r = 1k * k;\n'
                 '  analog I(p, n) <+ V(p, n) / r;\nendmodule\n')

DECK = '''* model-card instance parameters
.hdl "a1.va"
{pre}
V1 in 0 DC 1.0
.model mm a1 {card}
ya1 X1 in out mm {inst}
R2 out 0 1000.0
{extra}
.options device temp=27.0 tnom=27.0 gmin=1e-12
.OP
.print dc format=noindex v(out)
.end
'''


def run_xyce(va, card, inst='', pre='', extra=''):
    work = tempfile.mkdtemp(prefix='pyms_cardinst_')
    try:
        with open(os.path.join(work, 'a1.va'), 'w') as f:
            f.write(va)
        with open(os.path.join(work, 'deck.cir'), 'w') as f:
            f.write(DECK.format(card=card, inst=inst, pre=pre, extra=extra))
        env = dict(os.environ, PYMS_DIR=PYMS_DIR, PYMS_CACHE=os.path.join(work, 'ch'),
                   PYMS_VAE_CACHE=os.path.join(work, 'cv'), PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run([XYCE, 'deck.cir'], cwd=work, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=600)
        vals = []
        prn = os.path.join(work, 'deck.cir.prn')
        if os.path.exists(prn):
            with open(prn) as f:
                for line in f.read().splitlines():
                    parts = line.split()
                    if len(parts) == 1:
                        try:
                            vals.append(float(parts[0]))
                        except ValueError:
                            pass
        return r.returncode, r.stdout, vals
    finally:
        shutil.rmtree(work, ignore_errors=True)


@unittest.skipUnless(os.path.exists(XYCE), 'no Xyce binary')
class TestCardInstanceParams(unittest.TestCase):
    def vals(self, *a, **k):
        rc, log, vals = run_xyce(*a, **k)
        self.assertEqual(rc, 0, log[-2500:])
        self.assertTrue(vals, log[-2500:])
        self.assertNotIn('No model parameter', ' '.join(log.split()))
        return vals

    def test_the_card_value_is_the_default(self):
        self.assertAlmostEqual(self.vals(VA, 'r=2k')[0], 1.0 / 3.0, delta=1e-8)

    def test_the_instance_line_wins(self):
        self.assertAlmostEqual(self.vals(VA, 'r=2k', 'r=500')[0], 2.0 / 3.0, delta=1e-8)

    def test_neither_gives_it(self):
        self.assertAlmostEqual(self.vals(VA, '')[0], 0.5, delta=1e-8)

    def test_the_card_value_counts_as_given(self):
        self.assertAlmostEqual(self.vals(VA_GIVEN, 'r=2k')[0], 1.0 / 3.0, delta=1e-8)
        self.assertAlmostEqual(self.vals(VA_GIVEN, '')[0], 0.2, delta=1e-8)

    def test_a_dependent_default_keeps_the_card_value(self):
        self.assertAlmostEqual(self.vals(VA_DEP, 'r=2k')[0], 1.0 / 3.0, delta=1e-8)
        self.assertAlmostEqual(self.vals(VA_DEP, 'k=2')[0], 1.0 / 3.0, delta=1e-8)

    def test_a_step_of_the_card_value_reaches_the_instance(self):
        v = self.vals(VA, 'r=2k', '', extra='.step mm:r list 2k 500 1k')
        self.assertEqual(len(v), 3, v)
        for got, want in zip(v, (1.0 / 3.0, 2.0 / 3.0, 0.5)):
            self.assertAlmostEqual(got, want, delta=1e-8)

    def test_a_step_of_the_card_value_leaves_an_instance_value(self):
        v = self.vals(VA, 'r=2k', 'r=500', extra='.step mm:r list 2k 1k')
        self.assertEqual(len(v), 2, v)
        for got in v:
            self.assertAlmostEqual(got, 2.0 / 3.0, delta=1e-8)

    def test_a_card_expression_of_a_swept_param(self):
        v = self.vals(VA, 'r={rr}', '', pre='.param rr=1k', extra='.step rr list 2k 500')
        self.assertEqual(len(v), 2, v)
        for got, want in zip(v, (1.0 / 3.0, 2.0 / 3.0)):
            self.assertAlmostEqual(got, want, delta=1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
