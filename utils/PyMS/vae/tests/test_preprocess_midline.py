"""Conditional directives in the middle of a line.

THE DEFECT (PYMS1 skeptic, 2026-10-25; BUGS.md CHANGES 1.1.1): vae/preprocess.py
recognised `ifdef / `ifndef / `elsif / `else / `endif only at the start of
a line (_IFDEF_RE and friends) and ignored everything after one, so
``parameter real r = `ifdef BIG 2k `else 1k `endif;`` reached the parser
with the directives in it (8 decks of the skeptic's set refused), and an
undefined macro in the inactive arm of such a line was reported as an error.
The LRM puts compiler directives anywhere in the source text.

THE FIX: a line holding a conditional directive with other text around it
is split at the directives; the arms are kept or dropped by the conditional
stack exactly like whole-line directives, the kept text stays ONE output
line (the line map is unchanged) and only kept text is checked for
undefined macros.

Run:  cd utils/PyMS && python3 vae/tests/test_preprocess_midline.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.preprocess import preprocess_source, PreprocessError  # noqa: E402


def pp(text):
    return preprocess_source(text, filename="m.va")


def body(out):
    return [l.strip() for l in str(out).split('\n')]


class TestMidline(unittest.TestCase):

    def test_ifdef_else_in_a_default(self):
        out = pp("`define FOO\nparameter real r = `ifdef FOO 1k `else 2k `endif;\nx;\n")
        self.assertEqual(body(out)[:2], ["parameter real r = 1k ;", "x;"])
        out = pp("parameter real r = `ifdef FOO 1k `else 2k `endif;\nx;\n")
        self.assertEqual(body(out)[:2], ["parameter real r = 2k ;", "x;"])

    def test_line_map_is_kept(self):
        out = pp("a;\nparameter real r = `ifdef FOO 1k `else 2k `endif;\nb;\n")
        lines = body(out)
        self.assertEqual(lines.index("b;") + 1, 3)
        self.assertEqual(out.line_map[lines.index("b;")], ("m.va", 3))
        self.assertEqual(out.line_map[1], ("m.va", 2))

    def test_nested(self):
        src = "`define X\nr = `ifdef X `ifdef Y 3k `else 1k `endif `else 2k `endif;\n"
        self.assertEqual(body(pp(src))[0], "r = 1k ;")
        src = "`define X\n`define Y\nr = `ifdef X `ifdef Y 3k `else 1k `endif `else 2k `endif;\n"
        self.assertEqual(body(pp(src))[0], "r = 3k ;")

    def test_ifndef_and_elsif(self):
        self.assertEqual(body(pp("r = `ifndef A 1k `else 2k `endif;\n"))[0], "r = 1k ;")
        self.assertEqual(body(pp("`define B\nr = `ifdef A 3k `elsif B 1k `else 2k `endif;\n"))[0], "r = 1k ;")

    def test_macro_in_the_kept_arm_is_expanded(self):
        src = "`define RV 1000.0\nr = `ifdef BIG 2000.0 `else `RV `endif;\n"
        self.assertEqual(body(pp(src))[0], "r = 1000.0 ;")

    def test_undefined_macro_in_the_inactive_arm_is_not_an_error(self):
        out = pp("r = `ifdef NOTDEF `NOPE `else 1k `endif;\n")
        self.assertEqual(body(out)[0], "r = 1k ;")

    def test_undefined_macro_in_the_active_arm_is_an_error(self):
        with self.assertRaisesRegex(PreprocessError, "m.va:1"):
            pp("r = `ifdef NOTDEF 1k `else `NOPE `endif;\n")

    def test_spans_lines(self):
        src = "r = 1 `ifdef HALF * 0.5\n + 2 `else + 3 `endif;\n"
        self.assertEqual(body(pp(src))[:2], ["r = 1", "+ 3 ;"])

    def test_statement_with_directive_and_trailing_text(self):
        src = "`ifdef A x = 1; `endif y = 2;\n"
        self.assertEqual(body(pp(src))[0], "y = 2;")
        src = "`define A\n`ifdef A x = 1; `endif y = 2;\n"
        self.assertEqual(body(pp(src))[0], "x = 1; y = 2;")

    def test_directive_word_in_a_string_is_text(self):
        src = 'a = "`ifdef X";\n'
        self.assertEqual(body(pp(src))[0], 'a = "`ifdef X";')

    def test_whole_line_forms_unchanged(self):
        src = "`ifdef X\nr = 3k;\n`else\nr = 1k;\n`endif // X\n"
        self.assertEqual([l for l in body(pp(src)) if l], ["r = 1k;"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
