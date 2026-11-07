"""Macro arguments are substituted in one pass and inserted literally (1.1.7).

THE DEFECT (PYMS7 code review F4, 2026-11-07; since before 1.1.0):
preprocess._Preprocessor._substitute replaced the macro parameters one after
another in the body, so an argument containing the name of a LATER parameter
was substituted again: with `define ADD2(a, b) ((a) + 2*(b)) the use
`ADD2(b, 250) became ((250) + 2*(250)) instead of ((b) + 2*(250)), a divider
gave V(2) = 0.428571428 for 0.6, silently.  And re.sub took each argument as
a replacement template, so a backslash in an argument (an escaped identifier)
was read as an escape.

THE FIX: preprocess.py -- one combined pattern of every parameter, one pass,
a function replacement (the argument text is inserted as it is).

Run:  cd utils/PyMS && python3 vae/tests/test_macro_args_one_pass.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.preprocess import preprocess_source  # noqa: E402
from vae.tests._decks import mod, run_decks, XYCE  # noqa: E402


def _pp(text):
    return str(preprocess_source(text, "t.va"))


class TestPreprocess(unittest.TestCase):

    def test_argument_names_a_later_parameter(self):
        out = _pp("`define ADD2(a, b) ((a) + 2*(b))\nx = `ADD2(b, 250);\n")
        self.assertIn("x = ((b) + 2*(250));", out)

    def test_swapped_arguments(self):
        out = _pp("`define SUB(a, b) ((a) - (b))\nx = `SUB(b, a);\n")
        self.assertIn("x = ((b) - (a));", out)

    def test_backslash_in_argument(self):
        out = _pp("`define ID(a) (a)\nx = `ID(\\n1 );\n")
        # the backslash and the n stay two characters (re.sub made a newline)
        self.assertIn("x = (\\n1", out)
        self.assertNotIn("\n1", out.replace("\\n1", ""))

    def test_parameter_prefix_of_another(self):
        out = _pp("`define P2(v, vv) (v*vv)\nx = `P2(vv, v);\n")
        self.assertIn("x = (vv*v);", out)


MS = mod("ms", "  parameter real b = 1k;\n  analog I(p, n) <+ V(p, n) / `ADD2(b, 250);").replace(
    "module ms", "`define ADD2(a, b) ((a) + 2*(b))\nmodule ms")


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_divider(self):
        # R = b + 2*250 = 1.5k on the 1k load: V(out) = 1k / 2.5k = 0.4
        bad = run_decks({"ms": (MS, "", 0.4)}, prefix="pyms_macro1_")
        self.assertEqual(bad, [])


if __name__ == '__main__':
    unittest.main()
