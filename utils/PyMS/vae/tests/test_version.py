#!/usr/bin/env python3
"""PyMS's version marker: ``vae.__version__``, the ``VERSION`` file beside
``vae/`` and ``xyce_device_gen.py --version`` name one and the same version.

WHY (2026-10-24, PYMS1-I; PYMS_BUGS.md P11): vamos's spectre personality
guards every Verilog-A form PyMS once got wrong; with a version marker it can
require the PyMS that has the fixes and refuse Verilog-A on Xyce with an
older one.  The three spellings must agree, or a layer reading one of them
is misled.  No engine needed.

Run:  python3 vae/tests/test_version.py
"""
import os
import re
import subprocess
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

import vae  # noqa: E402

GEN = os.path.join(_PYMS_DIR, 'vae', 'xyce_device_gen.py')
VERSION_FILE = os.path.join(_PYMS_DIR, 'VERSION')


class VersionTests(unittest.TestCase):

    def test_version_is_dotted_numbers(self):
        self.assertTrue(re.fullmatch(r'\d+\.\d+\.\d+', vae.__version__), vae.__version__)

    def test_version_carries_the_round_1_fixes(self):
        # 1.1.0: the PYMS1 round (2026-10-24/25: parameter defaults,
        # attributes and undefined macros, defines and headers, ddt, current
        # probes, line map, cache race, model cards, disciplines, --define,
        # declaration syntax).  No marker at all before it.
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 0))

    def test_version_carries_the_round_2_fixes(self):
        # 1.1.1: the PYMS2 round (2026-10-25: node collapse, sibling model
        # groups, loud math-library failures, value ranges, string
        # parameters, mid-line conditionals, one-line errors; the version is
        # part of the math-library cache key).
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 1))

    def test_version_carries_the_round_3_fixes(self):
        # 1.1.2: the PYMS3 round (2026-10-26: a probed short never
        # collapses, case statements as the LRM defines them, undeclared
        # names refused, $simparam in conditions, string compares in
        # parameter defaults; the device-shell cache keys on the shell source).
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 2))

    def test_version_carries_the_round_4_fixes(self):
        # 1.1.3: the PYMS4 round (2026-10-26: ground declarations ground
        # their net and a net named gnd is a node, casex/casez, block-local
        # scoping, analog functions parsed / checked / inlined, $simparam
        # defaults at any depth, integer / % and rounding, loops that
        # iterate, ANSI and discipline port declarations, @(initial_step)
        # run, deterministic eval code).
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 3))

    def test_version_carries_the_round_5_fixes(self):
        # 1.1.4: the PYMS5 round (2026-10-30: analysis shadows -- an
        # @(initial_step(list)) runs only where the list says, held from
        # point to point, analysis(), $abstime, final_step, cross / above
        # events; the regime of the operating point held for ac; the last
        # reaching assignment in conditions; run-time loops of the eval code;
        # static block-local declaration values; run-time division-by-zero
        # warnings; function scoping and typing; recursion, several modules
        # in a file, parameter lists; the determinism guard warns).
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 4))

    def test_version_carries_the_round_6_fixes(self):
        # 1.1.5: the PYMS6 round (2026-11-01: the exact Jacobian -- forward-
        # mode automatic differentiation of the eval code instead of a
        # difference quotient; the held state starts from its declaration
        # values at every analysis and .step point; a .step of a baked
        # parameter rebuilds the math library; function locals that shadow
        # a parameter; untyped parameters typed by their default expression;
        # a module's attributes are its own; a module name of two .hdl files
        # warns, the first stays).
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 5))

    def test_version_carries_the_round_7_fixes(self):
        # 1.1.6: the PYMS7 round (2026-11-01: the device temperature reaches
        # a module -- $temperature / $vt at the instance's temperature, read
        # at run time through the instance's temperature source; parameter
        # defaults that read them; Xyce's k/q; a module's own TEMP / TNOM).
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 6))

    def test_version_carries_the_round_8_fixes(self):
        # 1.1.7: the PYMS8 round (2026-11-07: the code review's findings --
        # acceptStep's vae_needs_accept pointer dropped with the math library
        # it came from (a .step of a baked parameter crashed); pow with a zero
        # base and a bias-dependent exponent has slope 0; macro arguments
        # substituted in one pass; an @(initial_step) capture of the
        # temperature is held under .dc temp).
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 7))

    def test_version_carries_the_round_9_fixes(self):
        # 1.1.8: the PYMS9 round (2026-11-08: a device parameter given as an
        # expression of a .global_param, or of a .param a .DC/.STEP sweeps,
        # reaches the math library (updateDependentParameters in the shell's
        # constructors); aliasparam names are parameters of the card and the
        # instance line; a Newton iterate on the other branch of a held update
        # is not converged (the trapezoid ringing that stalled a held peak);
        # dropped noise sources are named in a warning).
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 8))

    def test_version_carries_the_round_10_fixes(self):
        # 1.1.9: the PYMS10 round (2026-11-09: an instance parameter on a
        # .model card is the default of the model's instances, as openvaf-r
        # has it; an R/C/L-group module that reads other parameters runs the
        # full evaluation; a gated contribution is a select (a V/I switch row
        # with a non-finite untaken arm is no longer NaN); a .step of a
        # module's own TEMP to the circuit temperature is taken).
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 9))

    def test_version_carries_the_b28_fix(self):
        # 1.1.10 (2026-11-13): the integer bitwise and shift operators of the
        # analog block build (>> logical, >>> arithmetic, ~, <<, &, |, ^, ^~),
        # and >>> <<< ^~ ~^ are single tokens (BUGS.md B28).
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 10))

    def test_version_carries_the_b29_fix(self):
        # 1.1.11 (2026-11-14): integer + - * ** and unary - of the analog
        # block wrap at 32 bits (BUGS.md B29), and a real ** builds (B30).
        self.assertGreaterEqual(tuple(int(x) for x in vae.__version__.split('.')), (1, 1, 11))

    def test_version_file_agrees(self):
        with open(VERSION_FILE) as f:
            text = f.read()
        self.assertEqual(text, vae.__version__ + '\n')

    def test_cli_reports_it(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run([sys.executable, GEN, '--version'], capture_output=True,
                           text=True, timeout=60, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), 'PyMS ' + vae.__version__)


if __name__ == "__main__":
    unittest.main()
