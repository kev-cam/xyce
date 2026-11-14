"""Integer + - * ** and unary - wrap at 32 bits in the analog block (1.1.11, BUGS.md B29).

THE DEFECT: the analog block carries an integer as a double, so an integer
sum, difference, product or power past 32 bits kept growing where the LRM
(and VACASK, and PyMS's own parameter defaults, va_const) wrap:
`(a >> 1) + 1004` with a = -8 was 2147484648, not -2147482648, and a
product past 2**53 was not even exact.

THE FIX: int_ops rewrites each integer + - * ** and unary - to
_vae_ibit(a, b, code) (codes 8-12; the B28 helper), wrapped to 32 bits;
** follows LRM table 4-6 (a negative exponent gives 0 unless the base is
1 or -1).

A wrap shows only where a later operation is not a ring operation (a
division, a modulus, a comparison, a real variable) -- a chain of + - *
agrees modulo 2**32 either way -- so each expression ends in one.  Each
output is 1000 V at the operating point; KF folds at elaboration, KR reads
instance parameters and a voltage-dependent k (the run-time path).

The last two outputs are real powers (B30, same release): a real `a ** b`
in the analog block reached the C++ as `**` and did not build; int_ops
now writes it as pow(a, b).

Run:
    cd utils/PyMS && python3 vae/tests/test_int_wrap.py
"""
import os
import shutil
import subprocess
import tempfile
import unittest

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# every expression is 1000; am = -8, ap = 1000, ab = 123456789,
# an = -2147483648, k = 1, n = 4
EXPRS = [
    '((am >> 1) + 1004) / 2 + 1073742324',     # -2147482648 / 2
    '(ap * ap * ap * ap < 0) ? 1000 : 0',      # 10**12 wraps to -727379968
    '(ab * ab) % 1000 + 1751',                 # -1757895751 (exact)
    '(-an < 0) ? 1000 : 0',                    # -(-2**31) is -2**31
    '((an - k) > 0) ? 1000 : 0',               # -2**31 - 1 is 2**31 - 1
    '((2 ** (k + 30)) < 0) ? 1000 : 0',        # 2**31 wraps
    '(2 ** (k - 2)) + 1000',                   # 2 ** -1 is 0 (LRM table 4-6)
    '((-1) ** (k - 4)) + 1001',                # (-1) ** -3 is -1
    '(3 ** (k + 39)) - 689955897',             # 3**40 mod 2**32
    'x + 1294968296',                          # x = ap * 3000000: -1294967296
    's',                                       # s = 250 n times in a loop
    'xr ** 3.0',                               # a real power (B30): xr = 10
    'xr ** (k + 2)',                           # ... with an integer exponent
]
N = len(EXPRS)


def module(name, runtime):
    ports = ', '.join(f'o{i}' for i in range(N))
    inst = '(*type="instance"*) ' if runtime else ''
    lines = ['`include "disciplines.vams"',
             f'module {name}({ports}, n);',
             f'inout {ports}, n; electrical {ports}, n;']
    for p, v in (('am', '-8'), ('ap', '1000'), ('ab', '123456789'),
                 ('an', '-2147483647 - 1'), ('nn', '4')):
        lines.append(f'{inst}parameter integer {p} = {v};')
    lines += ['integer k, r, s, i;', 'real x, xr;', 'analog begin',
              '  k = (V(o0, n) > 1e9) ? 3 : 1;' if runtime else '  k = 1;',
              '  x = ap * 3000000;',
              '  xr = 10.0;',
              '  s = 0;',
              '  for (i = 0; i < nn; i = i + 1) s = s + 250;']
    for i, e in enumerate(EXPRS):
        lhs = 'x' if e.startswith('x') else 'r'
        if lhs == 'x':
            lines.append(f'  V(o{i}, n) <+ {e};')
        else:
            lines.append(f'  r = {e}; V(o{i}, n) <+ r;')
    lines += ['end', 'endmodule']
    return '\n'.join(lines) + '\n'


def deck(name):
    nodes = ' '.join(f'o{i}' for i in range(N))
    out = ['* integer wrap', f'.hdl "{name}.va"', f'.model m1 {name}',
           f'Y{name.upper()} x1 {nodes} 0 m1']
    out += [f'R{i} o{i} 0 1k' for i in range(N)]
    out += ['.op', '.print dc format=noindex ' + ' '.join(f'v(o{i})' for i in range(N)), '.end']
    return '\n'.join(out) + '\n'


def run_xyce(name, runtime):
    work = tempfile.mkdtemp(prefix='pyms_iwrap_')
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
    """_py_ibit codes 8-12, the elaboration-time fold."""
    def test_table(self):
        from vae.ginac_emitter import _py_ibit
        self.assertEqual(_py_ibit(2147483644, 1004, 8), -2147482648.0)
        self.assertEqual(_py_ibit(-2147483648, 1, 9), 2147483647.0)
        self.assertEqual(_py_ibit(123456789, 123456789, 10), -1757895751.0)
        self.assertEqual(_py_ibit(-2147483648, 0, 11), -2147483648.0)
        self.assertEqual(_py_ibit(2, 31, 12), -2147483648.0)
        self.assertEqual(_py_ibit(2, -1, 12), 0.0)
        self.assertEqual(_py_ibit(-1, -3, 12), -1.0)
        self.assertEqual(_py_ibit(1, -5, 12), 1.0)
        self.assertEqual(_py_ibit(0, -1, 12), 0.0)


@unittest.skipUnless(os.path.exists(XYCE), 'no Xyce binary')
class TestIntWrap(unittest.TestCase):
    def check(self, name, runtime):
        rc, log, vals = run_xyce(name, runtime)
        self.assertEqual(rc, 0, log[-2500:])
        self.assertEqual(len(vals), N, log[-2500:])
        for e, got in zip(EXPRS, vals):
            self.assertAlmostEqual(got, 1000.0, delta=1e-9, msg=f'{e}: {got}')

    def test_folded_at_elaboration(self):
        self.check('kf', False)

    def test_run_time(self):
        self.check('kr', True)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, PYMS_DIR)
    unittest.main(verbosity=2)
