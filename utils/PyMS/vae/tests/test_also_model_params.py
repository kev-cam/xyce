"""Instance parameters on a .model card: the Xyce/ADMS rule (2026-10-25,
PYMS round 1 E-6).

THE QUESTION: ``.model mm vsrc Voltage=0.3`` with utils/ADMS/examples/toys
vsrc.va stops Xyce with "No model parameter VOLTAGE found". vsrc declares
``(*type="instance"*) parameter real Voltage``: an instance parameter, which
under the Xyce/ADMS convention (utils/ADMS/xyceImplementationFile_nosac.xml)
a model card can give only when the declaration also carries
``xyceAlsoModel="yes"``; Xyce's ADMS-built devices answer the same deck with
the same message. So that deck is in the wrong form (the value belongs on
the instance line) and the message is right.

THE DEFECT behind it: PyMS ignored xyceAlsoModel, so a card could never give
such a parameter, even where the module allows it (PSP103, BSIM-CMG,
BSIM-SOI, BSIM6, Mextram, VBIC, EKV declare 270 of them, e.g. W, L, NF).
THE FIX: the parser keeps the attribute (Param.also_model); the shell
registers the parameter on the model too, and an instance that does not give
it takes the card's value (as ADMS's Instance::processParams does); it
counts as given for $param_given.

Through Xyce on the reference divider (V1 = 1 V into the module, R2 = 1k;
private caches, PYMS_DIR = the PyMS copy this file sits in).

Run:
    cd utils/PyMS && python3 vae/tests/test_also_model_params.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PYMS_DIR not in sys.path:
    sys.path.insert(0, PYMS_DIR)

from vae.parser import parse_verilog_a  # noqa: E402
from vae.preprocess import preprocess_source  # noqa: E402

HEAD = ('`include "disciplines.vams"\n`include "constants.vams"\n'
        'module a1(p, n);\n  inout p, n;\n  electrical p, n;\n')
ALSO = HEAD + ('  (*type="instance", xyceAlsoModel="yes"*) parameter real r = 1k;\n'
               '  analog I(p, n) <+ V(p, n) / r;\nendmodule\n')
# the CMC macro form, after the declaration
ALSO_MACRO = ('`include "disciplines.vams"\n`include "constants.vams"\n'
              '`define attr(txt) (*txt*)\n'
              'module a1(p, n);\n  inout p, n;\n  electrical p, n;\n'
              '  parameter real r = 1k from (0:inf) `attr(type="instance" xyceAlsoModel="YES" info="r");\n'
              '  analog I(p, n) <+ V(p, n) / ($param_given(r) ? r : 4000.0);\nendmodule\n')
INST_ONLY = HEAD + ('  (*type="instance"*) parameter real r = 1k;\n'
                    '  analog I(p, n) <+ V(p, n) / r;\nendmodule\n')

DECK = '''* model-card instance parameters
.hdl "a1.va"
V1 in 0 DC 1.0
.model mm a1 {card}
ya1 X1 in out mm {inst}
R2 out 0 1000.0
.options device temp=27.0 tnom=27.0 gmin=1e-12
.OP
.print dc format=noindex v(out)
.end
'''


def run_xyce(va, card, inst):
    work = tempfile.mkdtemp(prefix='pyms_also_')
    try:
        with open(os.path.join(work, 'a1.va'), 'w') as f:
            f.write(va)
        with open(os.path.join(work, 'deck.cir'), 'w') as f:
            f.write(DECK.format(card=card, inst=inst))
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


class TestParser(unittest.TestCase):
    def test_the_attribute_is_kept(self):
        for va in (ALSO, ALSO_MACRO):
            mod = parse_verilog_a(preprocess_source(va, filename='a1.va'), filename='a1.va')
            (r,) = mod.params
            self.assertTrue(r.is_instance)
            self.assertTrue(r.also_model)
        mod = parse_verilog_a(preprocess_source(INST_ONLY, filename='a1.va'), filename='a1.va')
        self.assertFalse(mod.params[0].also_model)


@unittest.skipUnless(os.path.exists(XYCE), 'no Xyce binary')
class TestXyce(unittest.TestCase):
    def out(self, va, card, inst=''):
        rc, log, val = run_xyce(va, card, inst)
        self.assertEqual(rc, 0, log[-2500:])
        self.assertIsNotNone(val, log[-2500:])
        return val

    def test_the_card_gives_an_also_model_parameter(self):
        self.assertAlmostEqual(self.out(ALSO, 'r=2k'), 1.0 / 3.0, delta=1e-8)

    def test_the_instance_overrides_the_card(self):
        self.assertAlmostEqual(self.out(ALSO, 'r=2k', 'r=500'), 2.0 / 3.0, delta=1e-8)

    def test_neither_gives_it(self):
        self.assertAlmostEqual(self.out(ALSO, ''), 0.5, delta=1e-8)

    def test_the_card_value_counts_as_given(self):
        # $param_given(r) ? r : 4000.0  -- r = 2k from the card
        self.assertAlmostEqual(self.out(ALSO_MACRO, 'r=2k'), 1.0 / 3.0, delta=1e-8)
        self.assertAlmostEqual(self.out(ALSO_MACRO, ''), 0.2, delta=1e-8)

    def test_an_instance_only_parameter_on_a_card_is_the_default(self):
        # 1.1.9 (BUGS.md A25): the openvaf-r/VACASK rule replaces the
        # Xyce/ADMS one -- every instance parameter is accepted on the card
        # as the instance default (until 1.1.8: "No model parameter R found
        # ..., parameter ignored", the default r = 1k used).  The full set is
        # in test_card_instance_params.
        rc, log, val = run_xyce(INST_ONLY, 'r=2k', '')
        self.assertEqual(rc, 0, log[-2000:])
        self.assertNotIn('No model parameter', ' '.join(log.split()))
        self.assertAlmostEqual(val, 1.0 / 3.0, delta=1e-8)
        self.assertAlmostEqual(self.out(INST_ONLY, '', 'r=2k'), 1.0 / 3.0, delta=1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
