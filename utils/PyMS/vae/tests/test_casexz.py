"""casex / casez.

THE DEFECT (PYMS3 skeptic, 2026-10-26; BUGS.md CHANGES 1.1.3): the parser
knew only ``case``; ``casex (mode) 1: r = 1000; default: r = 4000;
endcase`` fell through to the expression-statement fallback, its arms were
dropped and r read 0 (V(out) = 0 with exit status 0; openvaf refuses it).

THE FIX: casex and casez parse as case.  In them the x / z (casez: z and ?)
digits of a based literal are don't-care bits; an analog value has no x or z
bits, so an item without such digits compares exactly as in case -- which is
what is built.  An item WITH x / z / ? digits needs a bit-mask compare PyMS
does not build: a file:line ParseError, never a silent 0.

Run:  cd utils/PyMS && python3 vae/tests/test_casexz.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a, NodeKind, ParseError  # noqa: E402
from vae.tests._decks import mod, run_decks, XYCE  # noqa: E402

BODY = ("  parameter integer mode = 1;\n  real r;\n  analog begin\n    %s (mode)\n"
        "      %s: r = 1000;\n      2, 3: r = 3000;\n      default: r = 4000;\n    endcase\n"
        "    I(p, n) <+ V(p, n) / r;\n  end")


def m(name, kw, item="1"):
    return mod(name, BODY % (kw, item))


CASES = {
    "casex_1": (m("casex_1", "casex"), "", 0.5),
    "casex_3": (m("casex_3", "casex"), "mode=3", 0.25),
    "casex_9": (m("casex_9", "casex"), "mode=9", 0.2),
    "casez_1": (m("casez_1", "casez"), "", 0.5),
    "casez_2": (m("casez_2", "casez"), "mode=2", 0.25),
    "casex_based": (m("casex_based", "casex", "4'b0001"), "", 0.5),
    "casex_xdigit": (m("casex_xdigit", "casex", "4'b000x"), "", "ERR"),
}


class TestParse(unittest.TestCase):

    def test_casex_is_an_if_chain(self):
        top = parse_verilog_a(m("t", "casex"), filename="c.va").analog_block.children[0]
        self.assertEqual(top.kind, NodeKind.IF)
        self.assertEqual(top.condition, "mode == 1")
        self.assertEqual(top.else_body.condition, "mode == 2 || mode == 3")

    def test_dont_care_digits_are_an_error(self):
        for kw, item in (("casex", "4'b000x"), ("casez", "4'b000?"), ("casez", "'b0z"), ("casex", "'hx")):
            with self.assertRaisesRegex(ParseError, r"c\.va:\d+: %s: .*don't-care" % kw):
                parse_verilog_a(m("e", kw, item), filename="c.va")

    def test_casez_item_without_dont_care(self):
        top = parse_verilog_a(m("t", "casez", "'h1"), filename="c.va").analog_block.children[0]
        self.assertEqual(top.condition, "mode == ( 1 )")      # the based literal as its value


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_casexz_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
