"""Analog functions (LRM 4.7): bodies checked, calls inlined, case expressions once.

THE DEFECTS (PYMS3 skeptic, 2026-10-26; BUGS.md CHANGES 1.1.3):
  * the parser skipped ``analog function ... endfunction``: a call reached the
    generated C++ as an undeclared function (no math library: a netlist error
    for an analog function with a local, an output argument, a case body);
  * so check_declarations never saw a function body: ``zz = a;`` (zz
    undeclared) inside one passed generation (exit 0);
  * a case expression was copied into each item compare, so a call with a
    side effect (an inout argument) ran once per item.

THE FIX: Parser._parse_analog_function parses the body (arguments with their
directions, locals, the statement); check_declarations walks it (an
assignment target must be an output / inout argument, a local or the
function name; a node access is refused); parser._FunctionInliner turns
every call into, before its statement, the inputs assigned to fresh
variables, a renamed copy of the body and the outputs assigned back; a case
expression with such a call is stored in a temporary first, so it is
evaluated once.  A function named like a PyMS builtin (lexp, lln,
hypsmooth, ...) keeps the builtin, as before.

Run:  cd utils/PyMS && python3 vae/tests/test_analog_functions.py
"""
from __future__ import annotations

import os
import re
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a, NodeKind  # noqa: E402
from vae.ginac_emitter import check_declarations, EmitError, _ast_walk  # noqa: E402
from vae.tests._decks import mod, run_decks, XYCE  # noqa: E402

BUMP = ("  analog function integer bump;\n    inout c;\n    integer c;\n    begin\n      c = c + 1;\n"
        "      bump = c;\n    end\n  endfunction\n")
CASES = {
    "local": (mod("local", "  analog function real f;\n    input a;\n    real a;\n    real t;\n    begin\n      t = 2 * a;\n      f = t;\n    end\n  endfunction\n  analog I(p, n) <+ V(p, n) / f(500);"), "", 0.5),
    "output_arg": (mod("output_arg", "  analog function real f;\n    input a;\n    output b;\n    real a, b;\n    begin\n      b = 2 * a;\n      f = a;\n    end\n  endfunction\n  real q, z;\n  analog begin z = f(500, q); I(p, n) <+ V(p, n) / q; end"), "", 0.5),
    "case_body": (mod("case_body", "  analog function real f;\n    input m;\n    integer m;\n    begin\n      case (m)\n        1: f = 1000;\n        default: f = 4000;\n      endcase\n    end\n  endfunction\n  parameter integer mode = 1;\n  analog I(p, n) <+ V(p, n) / f(mode);"), "", 0.5),
    "vdep_if": (mod("vdep_if", "  analog function real lim;\n    input v, hi;\n    real v, hi;\n    begin\n      if (v > hi) lim = hi; else lim = v;\n    end\n  endfunction\n  analog I(p, n) <+ lim(V(p, n), 10) / 1000;"), "", 0.5),
    "nested": (mod("nested", "  analog function real dbl;\n    input a;\n    real a;\n    dbl = 2 * a;\n  endfunction\n  analog function real quad;\n    input a;\n    real a;\n    quad = dbl(dbl(a));\n  endfunction\n  analog I(p, n) <+ V(p, n) / quad(250);"), "", 0.5),
    "int_func": (mod("int_func", "  analog function integer half;\n    input a;\n    integer a;\n    half = a / 2;\n  endfunction\n  analog I(p, n) <+ V(p, n) / (500 * half(5));"), "", 0.5),
    "case_once": (mod("case_once", BUMP + "  integer k, z;\n  analog begin\n    k = 0;\n    case (bump(k))\n      5: z = 1;\n      6: z = 2;\n      7: z = 3;\n      default: z = 0;\n    endcase\n    I(p, n) <+ V(p, n) / (1000 * k);\n  end"), "", 0.5),
    "case_once_match": (mod("case_once_match", BUMP + "  integer k;\n  real r;\n  analog begin\n    k = 0;\n    case (bump(k))\n      2: r = 4000;\n      1: r = 1000;\n      default: r = 2000;\n    endcase\n    I(p, n) <+ V(p, n) / r;\n  end"), "", 0.5),
    "bad_assign": (mod("bad_assign", "  analog function real f;\n    input a;\n    real a;\n    begin\n      zz = a;\n      f = a;\n    end\n  endfunction\n  analog I(p, n) <+ V(p, n) / f(1000);"), "", "ERR"),
}


class TestDeclarations(unittest.TestCase):

    def _check(self, name):
        check_declarations(parse_verilog_a(CASES[name][0], filename="c.va"))

    def test_body_assignment_target_checked(self):
        with self.assertRaisesRegex(EmitError, r"c\.va:\d+: assignment to 'zz' in analog function f"):
            self._check("bad_assign")

    def test_input_is_read_only_and_no_node_access(self):
        va = mod("e", "  analog function real f;\n    input a;\n    real a;\n    begin\n      a = 2 * a;\n      f = a + V(p);\n    end\n  endfunction\n  analog I(p, n) <+ V(p, n) / f(500);")
        with self.assertRaisesRegex(EmitError, r"input argument 'a'"):
            check_declarations(parse_verilog_a(va, filename="c.va"))

    def test_legal_bodies_pass(self):
        for name in ("local", "output_arg", "case_body", "nested", "case_once"):
            self._check(name)


class TestInlining(unittest.TestCase):

    def test_case_expression_evaluated_once(self):
        m = parse_verilog_a(CASES["case_once"][0], filename="c.va")
        incs = [n for n in _ast_walk(m.analog_block) if n.kind == NodeKind.ASSIGN
                and re.fullmatch(r'\S+ \+ 1', ' '.join((n.expr or '').split()))]
        self.assertEqual(len(incs), 1)          # one copy of the body: c = c + 1

    def test_no_call_left(self):
        m = parse_verilog_a(CASES["nested"][0], filename="c.va")
        for n in _ast_walk(m.analog_block):
            self.assertNotRegex(n.expr or '', r'\b(dbl|quad)\s*\(')


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_afunc_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
