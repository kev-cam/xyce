"""R/C/L model groups: the simple stamp only for the canonical forms (1.1.9, BUGS.md B27).

THE DEFECT: a module with xyceModelGroup Resistor / Capacitor / Inductor ran
PyMS's simple linear stamp whenever it had a value parameter (R, C, L; rsh,
l, w; rser/rpar of a lossy C/L), whatever its analog code computed: the
stamp reads only those parameters, so ``I(p,n) <+ V(p,n)/(R*rm)`` ignored rm
on the card (FIX2E8 raw l02/l06: 0.5 for 1/3), ``V(p,n)/(R*$temperature/
300.15)`` at 77 C ignored the temperature (0.5 for 0.4616), a constant
factor or an intermediate variable was dropped -- silently.

THE FIX (xyce_device_gen.py, simple_rlc_form / rlc_expected_form): the
analog block is reduced to one linear form per branch (straight-line
variables inlined, noise contributions dropped) and the simple stamp runs
only when that form is exactly the one it implements (I(p,n) <+ V(p,n)/R,
rsh*l/w, I <+ ddt(C*V) with the QSPICE rser/rpar network); anything else runs
the full evaluation.  The Inductor stamp (a 1e3 S operating-point
placeholder) is never used.

Reference divider: V1 = 1 V, the device from in to out, R2 = 1k to ground,
.op (private caches, PYMS_DIR = the PyMS copy this file sits in).  The
controls also check which route the device shell took (the simple stamp's
marker in the generated shell source).

Run:
    cd utils/PyMS && python3 vae/tests/test_rlc_group_other_params.py
"""
import glob
import os
import shutil
import subprocess
import tempfile
import unittest

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'
SIMPLE_MARK = 'stamp linear contribution'


def mod(group, letter, body, params):
    return HDR + '''(*xyceModelGroup="%s", xyceSpiceDeviceName="%s", xyceLevelNumber="6"*)
module dev (p,n);
  electrical p,n;
  inout p,n;
%s
  analog begin
%s
  end
endmodule
''' % (group, letter, params, body)


RINST = '  (*type="instance"*) parameter real R=1k from (0:inf);'
R_RM = mod('Resistor', 'R', '    I(p,n) <+ V(p,n)/(R*rm);', RINST + '\n  parameter real rm=1;')
R_PLAIN = mod('Resistor', 'R', '    I(p,n) <+ V(p,n)/R;', RINST)
R_TEMP = mod('Resistor', 'R', '    I(p,n) <+ V(p,n)/(R*$temperature/300.15);', RINST)
R_FACTOR = mod('Resistor', 'R', '    I(p,n) <+ V(p,n)/R*2;', RINST)
R_VAR = mod('Resistor', 'R', '    rr = 2*R;\n    I(p,n) <+ V(p,n)/rr;', RINST + '\n  real rr;')
# the toys/resistor.va shape: intermediates and a noise contribution, still canonical
R_TOY = mod('Resistor', 'R', '    g = 1.0/R;\n    ir = V(p,n)*g;\n    I(p,n) <+ ir;\n'
            '    I(p,n) <+ white_noise(4*1.38e-23*$temperature*g, "thermal");', RINST + '\n  real g, ir;')
C_G = mod('Capacitor', 'C', '    I(p,n) <+ ddt(C*V(p,n)) + g*V(p,n);',
          '  (*type="instance"*) parameter real C=1p from (0:inf);\n  parameter real g=0;')
C_PLAIN = mod('Capacitor', 'C', '    I(p,n) <+ ddt(C*V(p,n));', '  (*type="instance"*) parameter real C=1p from (0:inf);')
L_RS = mod('Inductor', 'L', '    V(p,n) <+ L*ddt(I(p,n)) + rs*I(p,n);',
           '  (*type="instance"*) parameter real L=1u from (0:inf);\n  parameter real rs=0;')
R_SHEET_K = mod('Resistor', 'R', '    I(p,n) <+ V(p,n)/(rsh*l/w*k);',
                '  parameter real rsh=100;\n  parameter real k=1;\n'
                '  (*type="instance"*) parameter real l=10u;\n  (*type="instance"*) parameter real w=1u;')

DECK = '''* rlc group
.hdl "dev.va"
V1 in 0 DC 1.0
.model m1 {letter} level=6 {card}
{letter}1 in out m1 {inst}
R2 out 0 1000.0
{extra}
.op
.print dc format=noindex v(out)
.end
'''


def run_xyce(va, letter, card='', inst='', extra=''):
    work = tempfile.mkdtemp(prefix='pyms_rlcgrp_')
    try:
        with open(os.path.join(work, 'dev.va'), 'w') as f:
            f.write(va)
        with open(os.path.join(work, 'd.cir'), 'w') as f:
            f.write(DECK.format(letter=letter, card=card, inst=inst, extra=extra))
        env = dict(os.environ, PYMS_DIR=PYMS_DIR, PYMS_CACHE=os.path.join(work, 'ch'),
                   PYMS_VAE_CACHE=os.path.join(work, 'cv'), PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run([XYCE, 'd.cir'], cwd=work, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=900)
        val = None
        prn = os.path.join(work, 'd.cir.prn')
        if os.path.exists(prn):
            with open(prn) as f:
                lines = f.read().splitlines()
            for line in lines:
                s = line.split()
                if len(s) == 1:
                    try:
                        val = float(s[0])
                    except ValueError:
                        pass
        simple = False
        for fn in glob.glob(os.path.join(work, 'ch', '**', '*'), recursive=True):
            if fn.endswith(('.C', '.cpp', '.h')) and os.path.isfile(fn):
                with open(fn, errors='replace') as f:
                    if SIMPLE_MARK in f.read():
                        simple = True
        return r.returncode, r.stdout, val, simple
    finally:
        shutil.rmtree(work, ignore_errors=True)


@unittest.skipUnless(os.path.exists(XYCE), 'no Xyce binary')
class TestRlcGroupOtherParams(unittest.TestCase):
    def out(self, *a, **k):
        rc, log, val, _simple = run_xyce(*a, **k)
        self.assertEqual(rc, 0, log[-2500:])
        self.assertIsNotNone(val, log[-2500:])
        return val

    def test_resistor_reads_a_card_multiplier(self):
        self.assertAlmostEqual(self.out(R_RM, 'R', 'rm=2', 'R=1k'), 1.0 / 3.0, delta=1e-8)
        self.assertAlmostEqual(self.out(R_RM, 'R', '', 'R=2k'), 1.0 / 3.0, delta=1e-8)

    def test_resistor_reads_the_temperature(self):
        # 77 C: R*350.15/300.15 in series with 1k
        r = 1000.0 * 350.15 / 300.15
        self.assertAlmostEqual(self.out(R_TEMP, 'R', '', 'R=1k', '.options device temp=77'),
                               1000.0 / (1000.0 + r), delta=1e-8)

    def test_resistor_with_a_constant_factor(self):
        # V/R*2: 500 ohm
        self.assertAlmostEqual(self.out(R_FACTOR, 'R', '', 'R=1k'), 2.0 / 3.0, delta=1e-8)

    def test_resistor_through_an_intermediate_variable(self):
        # rr = 2R: 2k
        self.assertAlmostEqual(self.out(R_VAR, 'R', '', 'R=1k'), 1.0 / 3.0, delta=1e-8)

    def test_canonical_forms_keep_the_simple_stamp(self):
        for va, letter, inst, want in ((R_PLAIN, 'R', 'R=2k', 1.0 / 3.0), (R_TOY, 'R', 'R=2k', 1.0 / 3.0),
                                       (C_PLAIN, 'C', 'C=1n', 0.0)):
            rc, log, val, simple = run_xyce(va, letter, '', inst)
            self.assertEqual(rc, 0, log[-2500:])
            self.assertAlmostEqual(val, want, delta=1e-8)
            self.assertTrue(simple, 'the simple stamp was not used: ' + va)

    def test_non_canonical_forms_take_the_full_evaluation(self):
        for va, letter in ((R_TEMP, 'R'), (R_FACTOR, 'R'), (R_VAR, 'R'), (L_RS, 'L')):
            rc, log, _val, simple = run_xyce(va, letter, '', '')
            self.assertEqual(rc, 0, log[-2500:])
            self.assertFalse(simple, 'the simple stamp was used: ' + va)

    def test_capacitor_with_a_conductance(self):
        # DC: the 1 mS conductance across the capacitor, 1k to ground
        self.assertAlmostEqual(self.out(C_G, 'C', 'g=1m', 'C=1n'), 0.5, delta=1e-8)

    def test_inductor_with_a_series_resistance(self):
        # DC: the inductor is a short; rs = 1k in series, 1k to ground
        self.assertAlmostEqual(self.out(L_RS, 'L', 'rs=1k', 'L=1u'), 0.5, delta=1e-8)

    def test_sheet_resistor_reads_another_parameter(self):
        # rsh*l/w = 1k, k = 2 -> 2k
        self.assertAlmostEqual(self.out(R_SHEET_K, 'R', 'k=2'), 1.0 / 3.0, delta=1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
