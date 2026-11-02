"""Analog functions keep the scope and the type of what they declare (1.1.4).

THE DEFECTS (PYMS4 skeptic, 2026-10-27, on 1.1.3; the 1 V / 1k divider):
  * two named blocks of a function body declaring ``real t`` and ``integer
    t`` became one real variable (parser.py _parse_block_variables: the
    function path never renamed): s4_func_two_blocks_types 0.518867924 for
    0.5;
  * a function body's ternaries were lowered to temporaries BEFORE
    int_ops typed the body, so ``((a > 0) ? 7 : 9) / 2`` read an untyped
    temporary and was a real division: x4_func_int_tern_div 0.428632662,
    x4_func_real_tern_div 0.461538461 for 0.5;
  * a recursive function reached g++ as an undeclared function (the
    inliner renamed the recursive call with the copy's locals).

THE FIX: the function path scopes block-locals like the module path (a
taken name gets a variable of its own); function bodies are lowered after
lower_integer_ops, as the analog block is; a recursion (direct or through
another function) is a ParseError naming the function's file:line.

Run:  cd utils/PyMS && python3 vae/tests/test_function_scope_type.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a, ParseError  # noqa: E402
from vae.tests._decks import mod, run_decks, XYCE  # noqa: E402

CASES = {
    "two_blocks_types": (mod("two_blocks_types", "  analog function real f;\n    input a;\n    real a;\n    real s;\n    begin\n      begin : b1\n        real t;\n        t = 2.5;\n        s = t;\n      end\n      begin : b2\n        integer t;\n        t = 2.6;\n        f = a * (s + t) / 5.5;\n      end\n    end\n  endfunction\n  analog I(p, n) <+ V(p, n) / f(1000);"), "", 0.5),
    "int_tern_div": (mod("int_tern_div", "  analog function integer h;\n    input a;\n    integer a;\n    h = ((a > 0) ? 7 : 9) / 2;\n  endfunction\n  analog I(p, n) <+ V(p, n) / (1000 * h(1) / 3);"), "", 0.5),
    "real_tern_div": (mod("real_tern_div", "  analog function real f;\n    input a;\n    integer a;\n    f = (((a > 0) ? 7 : 9) / 2) * 1000 / 3;\n  endfunction\n  analog I(p, n) <+ V(p, n) / f(1);"), "", 0.5),
    "block_shadows_arg": (mod("block_shadows_arg", "  analog function real f;\n    input a;\n    real a;\n    begin\n      begin : b\n        integer a;\n        a = 2.6;\n        f = a * 1000 / 3;\n      end\n    end\n  endfunction\n  analog I(p, n) <+ V(p, n) / f(5);"), "", 0.5),
    "tern_nested_int": (mod("tern_nested_int", "  analog function integer h;\n    input a;\n    integer a;\n    h = (a > 0) ? ((a > 5) ? 9 : 7) / 2 : 1;\n  endfunction\n  analog I(p, n) <+ V(p, n) / (1000 * h(1) / 3);"), "", 0.5),
}

RECURSIVE = mod("rec", "  analog function real fact;\n    input m;\n    real m;\n    fact = (m <= 1) ? 1 : m * fact(m - 1);\n  endfunction\n  analog I(p, n) <+ V(p, n) / (1000 * fact(3) / 6);")
MUTUAL = mod("mut", "  analog function real g;\n    input m;\n    real m;\n    g = h(m);\n  endfunction\n  analog function real h;\n    input m;\n    real m;\n    h = (m < 1) ? 1 : g(m - 1);\n  endfunction\n  analog I(p, n) <+ V(p, n) / (1000 * g(2));")


class TestParse(unittest.TestCase):

    def test_two_blocks_two_variables(self):
        m = parse_verilog_a(CASES["two_blocks_types"][0], filename="c.va")
        types = {v.name: v.type for v in m.variables if v.name.startswith('_vae_f1_t')}
        self.assertEqual(sorted(types.values()), ['integer', 'real'], types)

    def test_recursion_is_a_clear_error(self):
        for va in (RECURSIVE, MUTUAL):
            with self.assertRaisesRegex(ParseError, r"c\.va:\d+: analog function \w+ is recursive"):
                parse_verilog_a(va, filename="c.va")


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_fscope_"), [])

    def test_recursion_deck(self):
        self.assertEqual(run_decks({"rec": (RECURSIVE, "", "ERR")}, "pyms_frec_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
