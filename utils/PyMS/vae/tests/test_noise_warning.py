"""The noise sources PyMS drops are named in a warning (1.1.8, BUGS.md A19).

THE DEFECT (the 1.1.6 skeptic): ginac_emitter replaces white_noise /
flicker_noise with 0 without a word: a module with ``I(p, n) <+
white_noise(...)`` gave the .noise onoise of the rest of the circuit only,
silently.  PyMS still has no noise; the generator now prints a note naming
each noise source and its line, which Xyce shows as a ".HDL:" warning on
every run.  A module without noise sources prints none.

Run:  cd utils/PyMS && python3 vae/tests/test_noise_warning.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.tests._decks import HDR, XYCE, XYCE_LIBDIR  # noqa: E402
from vae.parser import parse_verilog_a  # noqa: E402
from vae.xyce_device_gen import noise_sources  # noqa: E402

NOISY = """module a1(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  analog begin
    I(p, n) <+ V(p, n) / r;
    I(p, n) <+ white_noise(4 * 1.380649e-23 * 300.15 / r, "thermal");
    if (V(p, n) > 0)
      I(p, n) <+ flicker_noise(1e-20 * abs(V(p, n) / r), 1.0, "flicker");
  end
endmodule
"""
QUIET = """module a1(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  analog I(p, n) <+ V(p, n) / r;
endmodule
"""
FN = """module a1(p, n);
  inout p, n;
  electrical p, n;
  analog function real my_white_noise;
    input x;
    real x;
    my_white_noise = 0 * x;
  endfunction
  analog I(p, n) <+ V(p, n) / 1k + my_white_noise(V(p, n));
endmodule
"""
DECK = ("* t\n.hdl \"a1.va\"\nV1 in 0 DC 1 AC 1\n.model m1 a1\nYa1 X1 in out m1\nR2 out 0 1k\n"
        ".op\n.print dc v(out)\n.end\n")


class TestNoiseSources(unittest.TestCase):

    def test_found_with_lines(self):
        self.assertEqual(noise_sources(parse_verilog_a(NOISY)), [('white_noise', 7), ('flicker_noise', 9)])

    def test_none(self):
        self.assertEqual(noise_sources(parse_verilog_a(QUIET)), [])
        self.assertEqual(noise_sources(parse_verilog_a(FN)), [])


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyceWarning(unittest.TestCase):

    def run_deck(self, va):
        d = tempfile.mkdtemp(prefix='pyms_nz_')
        try:
            with open(os.path.join(d, 'a1.va'), 'w') as fh:
                fh.write(HDR + va)
            with open(os.path.join(d, 'c.cir'), 'w') as fh:
                fh.write(DECK)
            env = dict(os.environ, PYMS_DIR=_PYMS_DIR, PYTHONDONTWRITEBYTECODE='1',
                       PYMS_CACHE=os.path.join(d, 'hc'), PYMS_VAE_CACHE=os.path.join(d, 'vc'))
            env['LD_LIBRARY_PATH'] = XYCE_LIBDIR + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
            env.pop('PYMS_DEFINES', None)
            r = subprocess.run([XYCE, 'c.cir'], cwd=d, env=env, capture_output=True, text=True, timeout=1800)
            return r.returncode, r.stdout + r.stderr
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_noisy_module_warns(self):
        rc, log = self.run_deck(NOISY)
        self.assertEqual(rc, 0, log[-800:])
        flat = ' '.join(log.split())   # Xyce wraps long messages
        self.assertEqual(flat.count('does not implement noise'), 1, log[-800:])
        msg = flat[flat.rfind('warning', 0, flat.index('does not implement noise')):]
        self.assertIn('white_noise (line 9)', msg)   # +2 lines: the HDR includes
        self.assertIn('flicker_noise (line 11)', msg)

    def test_quiet_module_does_not(self):
        rc, log = self.run_deck(QUIET)
        self.assertEqual(rc, 0, log[-800:])
        self.assertNotIn('does not implement noise', ' '.join(log.split()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
