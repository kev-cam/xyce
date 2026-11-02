"""An SI scale factor or a digit separator inside an analog-block expression
reaches the math library as a number (2026-10-25, PYMS round 1 D-5).

THE BUG: the parser kept a NUMBER token as written, so ``r / 1k`` in a
contribution reached the emitted GiNaC/C++ program as ``1k`` (g++: "unable to
find numeric literal operator"; the math library did not build) and
``1_000`` as ``1_000``. Only a parameter default was converted (va_const,
parser.parse_number). VACASK reads both (LRM 2.6.2: T G M K k m u n p f a).

THE FIX: parser.tokenize turns every NUMBER token into the number C++ and
Python read (parser.plain_number): a scale factor into the real it stands
for (``4.7p`` the double nearest 4.7e-12), the separators dropped (an
integer stays an integer). Every translator of the emitter reads the tokens.

Two layers: the tokenizer alone, and the forms on the reference divider
through Xyce (private PYMS_CACHE / PYMS_VAE_CACHE, PYMS_DIR = the PyMS copy
this file sits in; skipped without a Xyce binary).

Run:
    cd utils/PyMS && python3 vae/tests/test_expr_scale_factors.py
$XYCE names the binary (default /usr/local/src/xyce-build/src/Xyce).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_PYMS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS not in sys.path:
    sys.path.insert(0, _PYMS)

from vae.parser import parse_verilog_a, tokenize, NodeKind  # noqa: E402
from vae.preprocess import preprocess_source  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
RTOL = 1e-6
HEAD = ('`include "disciplines.vams"\n`include "constants.vams"\n'
        'module a1(p, n);\n  inout p, n;\n  electrical p, n;\n  parameter real r = 1k;\n')

DECK = '''* scale factors in an expression: the reference divider
.hdl "a1.va"
V1 in 0 DC 1.0
.model a1__t a1
ya1 X1 in out a1__t
R2 out 0 1000.0
.options device temp=27.0 tnom=27.0 gmin=1e-12
.OP
.print dc format=noindex v(out)
.end
'''

# body -> analytic out (V1 = 1 V into the module, R2 = 1k to ground)
FORMS = {
    'divide_by_1k':        ('analog I(p, n) <+ V(p, n) / 1k;', 0.5),
    'times_1m':            ('analog I(p, n) <+ V(p, n) * 1m;', 0.5),
    'parameter_over_1k':   ('analog I(p, n) <+ (r / 1k) * V(p, n) / 2k;', 1.0 / 3.0),
    'in_a_condition':      ('analog begin\n    if (V(p, n) > 1m) I(p, n) <+ V(p, n) / 1k;\n'
                            '    else I(p, n) <+ V(p, n) / 2k;\n  end', 0.5),
    'in_a_variable':       ('real g;\n  analog begin\n    g = 1.0 / 4k;\n    I(p, n) <+ V(p, n) * g;\n  end',
                            0.2),
    'digit_separator':     ('analog I(p, n) <+ V(p, n) / 1_000.0;', 0.5),
    'small_factors':       ('analog I(p, n) <+ V(p, n) * (1000p * 1e6 + 1u * 0.0);', 0.5),
}


def module(body):
    return HEAD + '  ' + body + '\nendmodule\n'


def run_xyce(va):
    work = tempfile.mkdtemp(prefix='pyms_scale_')
    try:
        with open(os.path.join(work, 'a1.va'), 'w') as f:
            f.write(va)
        with open(os.path.join(work, 'deck.cir'), 'w') as f:
            f.write(DECK)
        env = dict(os.environ, PYMS_DIR=_PYMS, PYMS_CACHE=os.path.join(work, 'ch'),
                   PYMS_VAE_CACHE=os.path.join(work, 'cv'), PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run([XYCE, 'deck.cir'], cwd=work, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=600)
        val = None
        prn = os.path.join(work, 'deck.cir.prn')
        if os.path.exists(prn):
            with open(prn) as f:
                prn_lines = f.read().splitlines()
            for line in prn_lines:
                parts = line.split()
                if len(parts) == 1:
                    try:
                        val = float(parts[0])
                        break
                    except ValueError:
                        pass
        return r.returncode, r.stdout, val
    finally:
        shutil.rmtree(work, ignore_errors=True)


class TestTokens(unittest.TestCase):
    def values(self, text):
        return [t.value for t in tokenize(text) if t.type == 'NUMBER']

    def test_scale_factors_become_reals(self):
        self.assertEqual(self.values('1k 2.5m 1K 3M'), ['1000.0', '0.0025', '1000.0', '3000000.0'])
        self.assertEqual(float(self.values('4.7p')[0]), 4.7e-12)       # the nearest double, not 4.7 * 1e-12
        self.assertEqual(float(self.values('1a')[0]), 1e-18)
        self.assertEqual(float(self.values('2T')[0]), 2e12)

    def test_digit_separators_go_and_integers_stay_integers(self):
        self.assertEqual(self.values('1_000 1_000.5 1_0k'), ['1000', '1000.5', '10000.0'])

    def test_plain_numbers_are_unchanged(self):
        self.assertEqual(self.values('1000 1.5e-3 .5 7'), ['1000', '1.5e-3', '.5', '7'])

    def test_no_scale_factor_reaches_a_contribution(self):
        for name, (body, _out) in FORMS.items():
            with self.subTest(name):
                mod = parse_verilog_a(preprocess_source(module(body), filename='a1.va'), filename='a1.va')
                texts = []

                def walk(n):
                    if n is None:
                        return
                    texts.extend(x for x in (n.expr, n.condition) if x)
                    for c in list(n.children or []) + ([n.else_body] if n.else_body else []):
                        walk(c)
                walk(mod.analog_block)
                joined = ' '.join(texts)
                self.assertNotRegex(joined, r'\d[TGMKkmunpfa]\b|\d_\d', joined)


@unittest.skipUnless(os.path.exists(XYCE), 'no Xyce binary')
class TestXyce(unittest.TestCase):
    def test_forms_run_right(self):
        for name, (body, want) in FORMS.items():
            with self.subTest(name):
                rc, log, val = run_xyce(module(body))
                self.assertEqual(rc, 0, f'{name}: Xyce exit {rc}:\n{log[-2500:]}')
                self.assertIsNotNone(val, f'{name}: no value:\n{log[-2500:]}')
                self.assertLess(abs(val - want) / want, RTOL, f'{name}: out {val!r}, want {want!r}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
