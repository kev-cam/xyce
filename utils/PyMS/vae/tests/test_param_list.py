"""A parameter declaration declares a list of parameters (LRM 3.4; 1.1.4).

THE DEFECT (BUGS.md B20, 2026-10-26): ``parameter real is = 1e-14, nf =
1.5;`` -- _parse_param read one name and took the rest of the statement
(``1e-14 , nf = 1.5``) as its default, which va_const refused ("the
character '=' is not part of a constant expression").

THE FIX: parser.py _parse_param returns every ``name = default {range}``
item of the declaration, each with its own ranges; the attributes of the
declaration belong to all.  An untyped parameter takes its default's type.

Run:  cd utils/PyMS && python3 vae/tests/test_param_list.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a  # noqa: E402
from vae.tests._decks import mod, run_decks, XYCE  # noqa: E402

CASES = {
    "plist": (mod("plist", "  parameter real ra = 500, rb = 500;\n  analog I(p, n) <+ V(p, n) / (ra + rb);"), "", 0.5),
    "plist_ranges": (mod("plist_ranges", "  parameter real ra = 500 from (0:inf), rb = 250 from [0:inf);\n  analog I(p, n) <+ V(p, n) / (ra + 2 * rb);"), "", 0.5),
    "plist_int": (mod("plist_int", "  parameter integer n1 = 2, n2 = 2;\n  analog I(p, n) <+ V(p, n) * (n1 + n2) / 4000;"), "", 0.5),
    "plist_card": (mod("plist_card", "  parameter real ra = 100, rb = 500;\n  analog I(p, n) <+ V(p, n) / (ra + rb);"), "rb=900", 0.5),
    # a value set holds commas of its own
    "plist_valueset": (mod("plist_valueset", "  parameter string mode = \"fast\" from '{\"fast\", \"slow\"}, other = \"x\";\n  analog I(p, n) <+ V(p, n) / ((mode == \"fast\") ? 1000 : 2000);"), "", 0.5),
    "plist_range_bad": (mod("plist_range_bad", "  parameter real ra = 500 from (0:inf), rb = -5 from [0:inf);\n  analog I(p, n) <+ V(p, n) / (ra + rb);"), "", "ERR"),
}


class TestParse(unittest.TestCase):

    def test_items(self):
        m = parse_verilog_a(mod("x", "  (* desc=\"d\" *) parameter real is = 1e-14, nf = 1.5 from (0:inf);\n"
                                     "  parameter rr = 7;\n  analog I(p, n) <+ V(p, n) / rr;"), filename="c.va")
        ps = {p.name: p for p in m.params}
        self.assertEqual([p.name for p in m.params], ['is', 'nf', 'rr'])
        self.assertEqual(ps['is'].default.split(), ['1e-14'])
        self.assertEqual(ps['nf'].default.split(), ['1.5'])
        self.assertEqual(ps['nf'].value_ranges, [('from', '( 0 : inf )')])
        self.assertEqual(ps['is'].value_ranges, [])
        self.assertEqual(ps['is'].desc, 'd')
        self.assertEqual(ps['nf'].desc, 'd')
        self.assertEqual(ps['rr'].type, 'integer')


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_plist_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
