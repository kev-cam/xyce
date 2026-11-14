"""Integer bitwise and shift operators in the analog block (1.1.10, BUGS.md B28).

THE DEFECT: an integer variable assigned from `a >> 1`, `a >>> 1`, `~a`,
`a << 2`, `a & b`, `a | b`, `a ^ b` or `a ^~ b` in the analog block made the
generated ginac.cpp fail to compile (the integer is carried as a double
there: "invalid operands of types 'double' and 'int' to binary
'operator>>'", "wrong type argument to bit-complement"), and the main
tokenizer split `>>>` into `>>` `>`: the device was refused.

THE FIX: int_ops rewrites each integer bitwise or shift operation to
_vae_ibit(a, b, code) with 32-bit semantics (>> logical, >>> arithmetic,
a shift by 32 or more gives 0, or -1 for >>> of a negative), folded in
Python at elaboration (_py_ibit), held as the GiNaC function vae_ibit
(slope 0) and printed as the C++ helper in the eval code and its dual copy;
parser.py tokenizes >>> <<< ^~ ~^ as single operators.

Each output of a module is one expression, built to be 1000 V at the
operating point; KF folds everything at elaboration (model parameters),
KR reads instance parameters and a shift count that depends on a voltage
(the run-time path).  Private caches, PYMS_DIR = the PyMS copy this file
sits in.

Run:
    cd utils/PyMS && python3 vae/tests/test_int_bitwise.py
"""
import os
import shutil
import subprocess
import tempfile
import unittest

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# (expression, a): every expression is 1000 for its a
EXPRS = [
    ('(a >> 28) + 985', -8),           # logical: 0xFFFFFFF8 >> 28 = 15
    ('(a >> 1) - 2147482644', -8),     # logical: 2147483644
    ('(a >>> 1) + 1004', -8),          # arithmetic: -4
    ('~a + 2001', 1000),               # ~1000 = -1001
    ('(~a & 255) + 977', 1000),        # 0xFFFFFC17 & 0xFF = 23
    ('(a << 2) + 1032', -8),
    ('((a + 9) << 31) + 2147484648', -8),   # 1 << 31 = -2147483648
    ('(a & 12) + 992', -8),
    ('(a | 3) + 1005', -8),
    ('(a ^ 5) + 1003', -8),
    ('(a ^~ 5) + 998', -8),
    ('(a >> 40) + 1000', -8),          # a shift by 32 or more: 0
    ('(a >>> 40) + 1001', -8),         # ... -1 for >>> of a negative
]
N = len(EXPRS)


def module(name, runtime):
    ports = ', '.join(f'o{i}' for i in range(N))
    lines = ['`include "disciplines.vams"',
             f'module {name}({ports}, n);',
             f'inout {ports}, n; electrical {ports}, n;',
             ]
    if runtime:
        lines.append('(*type="instance"*) parameter integer am = -8;')
        lines.append('(*type="instance"*) parameter integer ap = 1000;')
    else:
        lines.append('parameter integer am = -8;')
        lines.append('parameter integer ap = 1000;')
    lines.append('integer k, r;')
    lines.append('analog begin')
    # k = 1 at the operating point; a voltage decides it, so it is a run-time value
    lines.append('  k = (V(o0, n) > 1e9) ? 3 : 1;' if runtime else '  k = 1;')
    for i, (e, a) in enumerate(EXPRS):
        e = e.replace('a', 'am' if a == -8 else 'ap')
        if runtime:
            # the shift counts and operands read k (k == 1 here)
            e = e.replace('>> 1)', '>> k)').replace('<< 2)', '<< (k + 1))')
        lines.append(f'  r = {e}; V(o{i}, n) <+ r;')
    lines.append('end')
    lines.append('endmodule')
    return '\n'.join(lines) + '\n'


def deck(name):
    nodes = ' '.join(f'o{i}' for i in range(N))
    out = ['* integer bitwise', f'.hdl "{name}.va"', f'.model m1 {name}',
           f'Y{name.upper()} x1 {nodes} 0 m1']
    out += [f'R{i} o{i} 0 1k' for i in range(N)]
    out += ['.op', '.print dc format=noindex ' + ' '.join(f'v(o{i})' for i in range(N)), '.end']
    return '\n'.join(out) + '\n'


def run_xyce(name, runtime):
    work = tempfile.mkdtemp(prefix='pyms_ibit_')
    try:
        with open(os.path.join(work, f'{name}.va'), 'w') as f:
            f.write(module(name, runtime))
        with open(os.path.join(work, 'd.cir'), 'w') as f:
            f.write(deck(name))
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
                    if len(s) == N:
                        try:
                            vals = [float(x) for x in s]
                        except ValueError:
                            pass
        return r.returncode, r.stdout, vals
    finally:
        shutil.rmtree(work, ignore_errors=True)


class TestPyHelper(unittest.TestCase):
    """_py_ibit, the elaboration-time fold, against the table."""
    def test_table(self):
        from vae.ginac_emitter import _py_ibit
        self.assertEqual(_py_ibit(-8, 1, 5), 2147483644.0)
        self.assertEqual(_py_ibit(-8, 1, 6), -4.0)
        self.assertEqual(_py_ibit(1000, 0, 7), -1001.0)
        self.assertEqual(_py_ibit(1, 31, 4), -2147483648.0)
        self.assertEqual(_py_ibit(-8, 40, 5), 0.0)
        self.assertEqual(_py_ibit(-8, 40, 6), -1.0)
        self.assertEqual(_py_ibit(-8, -1, 5), 0.0)
        self.assertEqual(_py_ibit(-8, 5, 3), 2.0)


@unittest.skipUnless(os.path.exists(XYCE), 'no Xyce binary')
class TestIntBitwise(unittest.TestCase):
    def check(self, name, runtime):
        rc, log, vals = run_xyce(name, runtime)
        self.assertEqual(rc, 0, log[-2500:])
        self.assertEqual(len(vals), N, log[-2500:])
        for (e, a), got in zip(EXPRS, vals):
            self.assertAlmostEqual(got, 1000.0, delta=1e-9, msg=f'{e} (a = {a}): {got}')

    def test_folded_at_elaboration(self):
        self.check('kf', False)

    def test_run_time(self):
        self.check('kr', True)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, PYMS_DIR)
    unittest.main(verbosity=2)
