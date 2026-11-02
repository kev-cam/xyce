"""Two emitter defects found by the 2026-10-24 corpus scans (PYMS_BUGS.md,
cluster C's report items 1 and 4).

1. A math call with all-numeric arguments and an integer literal among them
   did not compile: DIODE_CMC 2.0 writes ``pow(2, -1.0 / MJ)``; once MJ is
   substituted the GiNaC program says ``pow ( 2 , ( -1.0 / ( 0.5 ) ) )`` and
   g++ stops with "call of overloaded 'pow(int, double)' is ambiguous"
   (<cmath>'s template and GiNaC's both match). The emitter now casts the
   arguments of such a call to double (GiNaCEmitter._real_numeric_call_args).

2. emit_ginac_program was not deterministic within one process: the ternary
   temporaries _vae_t<N> were numbered by a counter in parser.py that no parse
   reset, so emitting the same module twice gave two programs (PSP102/103,
   BSIM6 in a corpus scan). The counter is now reset per parse.

Run:
    cd utils/PyMS && python3 vae/tests/test_emit_numeric_and_determinism.py
The Xyce test skips without a Xyce binary ($XYCE, default
/usr/local/src/xyce-build/src/Xyce); HDLTEST_KEEP=1 keeps its directory.
"""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PYMS_DIR)

from vae.parser import parse_verilog_a              # noqa: E402
from vae.ginac_emitter import emit_ginac_program    # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
TIMEOUT = 600

TERNARY_MODULE = '''`include "disciplines.vams"
module t1(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  real g, h;
  analog begin
    g = V(p, n) > 0 ? 1.0 / r : 2.0 / r;
    h = V(p, n) > 1 ? g : (V(p, n) < -1 ? 2 * g : g);
    I(p, n) <+ h * V(p, n);
  end
endmodule
'''

POW_MODULE = '''`include "disciplines.vams"
`include "constants.vams"
module a1(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  parameter real m = 0.5;
  analog I(p, n) <+ V(p, n) / r * 4 * pow(2, -1.0 / m);
endmodule
'''

DC_DECK = '''* pow(2, -1.0 / m) = 0.25: the module is r, out = 0.5
.hdl "a1.va"
V1 in 0 DC 1.0
.model a1__t a1
ya1 X1 in out a1__t
R2 out 0 1000.0
.OP
.print dc format=noindex v(out)
.end
'''


def emit(source):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return emit_ginac_program(parse_verilog_a(source, filename='t1.va'))


class TestEmitDeterminism(unittest.TestCase):

    def test_same_module_twice_in_one_process(self):
        """Emitting one module twice gives one program."""
        first = emit(TERNARY_MODULE)
        self.assertIn('_vae_t1', first)
        for _ in range(2):
            self.assertEqual(emit(TERNARY_MODULE), first,
                             'the second emit renamed the ternary temporaries')

    def test_numeric_pow_call_is_real(self):
        """pow(2, -1.0 / m) with m substituted calls the real pow."""
        src = emit(POW_MODULE.replace('module a1', 'module t1'))
        self.assertNotRegex(src, r'pow \( 2 ,', 'an integer literal is left in a numeric pow call')
        self.assertIn('pow ( double(2) ,', src)


@unittest.skipUnless(os.path.exists(XYCE), f'no Xyce binary at {XYCE} (set $XYCE)')
class TestNumericPowOnXyce(unittest.TestCase):

    def test_numeric_pow_builds_and_runs(self):
        work = tempfile.mkdtemp(prefix='pyms_numpow_')
        try:
            with open(os.path.join(work, 'a1.va'), 'w') as f:
                f.write(POW_MODULE)
            with open(os.path.join(work, 'deck.cir'), 'w') as f:
                f.write(DC_DECK)
            env = dict(os.environ, PYMS_DIR=PYMS_DIR, PYTHONDONTWRITEBYTECODE='1',
                       PYMS_CACHE=os.path.join(work, 'cache_hdl'),
                       PYMS_VAE_CACHE=os.path.join(work, 'cache_vae'))
            r = subprocess.run([XYCE, 'deck.cir'], cwd=work, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, timeout=TIMEOUT)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:])
            vals = []
            with open(os.path.join(work, 'deck.cir.prn')) as f:
                for line in f:
                    try:
                        vals.append(float(line.split()[0]))
                    except (ValueError, IndexError):
                        pass
            self.assertTrue(vals, 'no dc value')
            self.assertAlmostEqual(vals[0], 0.5, delta=1e-6)
        finally:
            if not os.environ.get('HDLTEST_KEEP'):
                shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
