"""A function local that shadows a parameter; the type of an untyped
parameter (1.1.5).

THE DEFECTS (PYMS5 skeptic, 2026-11-01, on 1.1.4):
  * parser._is_constant_init took every module parameter name for a
    constant, also inside an analog function whose argument or local of the
    same name shadows it: the block-local ``real t = r + 1000;`` (r a local
    of the function, assigned 3 * x above) was hoisted to the head of the
    function body and read r = 0 (b5b_func_local_shadow_silent 0.5 for 0.2;
    b5_hoist_func_local_param a singular operating point).  A regression
    against 1.1.3.  PyMS keeps a non-constant declaration value where it is
    written (BUGS.md C11), and a name of the function's own scope is not a
    constant: the declaration now stays in place and reads the local.
  * an untyped parameter was an integer only when its default was a bare
    integer literal: ``parameter m = 2 * 2`` and ``parameter b = a`` (a an
    integer parameter) were real, so m / 3 was 1.33 (LRM 3.4.1: the type of
    the default expression; 0.571428571 for 0.5).  The default is now typed
    with the expression typer of the integer arithmetic (int_ops).

Run:  cd utils/PyMS && python3 vae/tests/test_scope_and_param_type.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a  # noqa: E402
from vae.tests._decks import mod, rv, run_decks, XYCE  # noqa: E402

FUNC_SHADOW = ("  parameter real r = 1000;\n  analog function real f;\n    input x;\n    real x;\n    real r;\n"
               "    begin\n      r = 3 * x;\n      begin : b\n        real t = r + 1000;\n        f = t;\n      end\n"
               "    end\n  endfunction\n  analog I(p, n) <+ V(p, n) / f(1000);")
FUNC_SHADOW2 = ("  parameter real r = 1000;\n  analog function real f;\n    input x;\n    real x;\n    real r;\n"
                "    begin\n      r = 2 * x;\n      begin : b\n        real t = r;\n        f = t;\n      end\n    end\n"
                "  endfunction\n  analog I(p, n) <+ V(p, n) / f(2000);")
FUNC_ARG = ("  parameter real r = 1000;\n  analog function real f;\n    input r;\n    real r;\n    begin : b\n"
            "      real t = r;\n      f = t;\n    end\n  endfunction\n  analog I(p, n) <+ V(p, n) / f(4000);")
FUNC_NOSHADOW = ("  parameter real q = 1000;\n  analog function real f;\n    input x;\n    real x;\n    real r;\n"
                 "    begin\n      r = 3 * x;\n      begin : b\n        real t = r + 1000;\n        f = t;\n      end\n"
                 "    end\n  endfunction\n  analog I(p, n) <+ V(p, n) / f(1000);")
FUNC_CONST = ("  parameter real r = 1000;\n  analog function real f;\n    input x;\n    real x;\n    begin : b\n"
              "      real r = 4000;\n      f = r * x;\n    end\n  endfunction\n  analog I(p, n) <+ V(p, n) / f(1);")
FUNC_PARAM_CONST = ("  parameter real r = 4000;\n  analog function real f;\n    input x;\n    real x;\n    begin : b\n"
                    "      real t = r;\n      f = t * x;\n    end\n  endfunction\n  analog I(p, n) <+ V(p, n) / f(1);")

CASES = {
    "func_local_shadow": (mod("func_local_shadow", FUNC_SHADOW), "", 0.2),
    "func_local_shadow2": (mod("func_local_shadow2", FUNC_SHADOW2), "", 0.2),
    "func_arg_shadow": (mod("func_arg_shadow", FUNC_ARG), "", 0.2),
    "func_noshadow": (mod("func_noshadow", FUNC_NOSHADOW), "", 0.2),
    "func_local_const": (mod("func_local_const", FUNC_CONST), "", 0.2),
    # a parameter read in a function (not shadowed): a constant, hoisted
    "func_param_const": (mod("func_param_const", FUNC_PARAM_CONST), "", 0.2),
    "untyped_int_expr": (mod("untyped_int_expr", "  parameter m = 2 * 2;\n  analog I(p, n) <+ V(p, n) * (m / 3) * 1m;"), "", 0.5),
    "untyped_ref_int": (mod("untyped_ref_int", "  parameter integer a = 4;\n  parameter b = a;\n  analog I(p, n) <+ V(p, n) * (b / 3) * 1m;"), "", 0.5),
    "untyped_list_ref": (mod("untyped_list_ref", "  parameter a = 4, b = a + 0;\n  analog I(p, n) <+ V(p, n) * (b / 3) * 1m;"), "", 0.5),
    "untyped_int_lit": (mod("untyped_int_lit", "  parameter m = 4;\n  analog I(p, n) <+ V(p, n) * (m / 3) * 1m;"), "", 0.5),
    "untyped_real_expr": (mod("untyped_real_expr", "  parameter m = 2 * 2.0;\n  analog I(p, n) <+ V(p, n) * (m / 3) * 1m;"), "", rv(750.0)),
    "untyped_real_ref": (mod("untyped_real_ref", "  parameter real a = 4;\n  parameter b = a;\n  analog I(p, n) <+ V(p, n) * (b / 3) * 1m;"), "", rv(750.0)),
    "untyped_scale": (mod("untyped_scale", "  parameter m = 4k;\n  analog I(p, n) <+ V(p, n) * (m / 3000) * 1m;"), "", rv(750.0)),
    "untyped_card": (mod("untyped_card", "  parameter m = 2 * 2;\n  analog I(p, n) <+ V(p, n) * (m / 3) * 1m;"), "m=5", 0.5),
}


class TestParse(unittest.TestCase):

    def types(self, body):
        m = parse_verilog_a(mod("m", body), filename="c.va")
        return {p.name: p.type for p in m.params}

    def test_untyped(self):
        t = self.types("  parameter integer a = 4;\n  parameter real ra = 1;\n  parameter b = a, c = 2 * 2, d = 1.5 * a, "
                       "e = 4, f = 1m, g = ra, h = (a > 1) ? 2 : 3, i = \"s\", j = -3, k = 1e3;\n"
                       "  analog I(p, n) <+ V(p, n) / 1000;")
        self.assertEqual(t, dict(a='integer', ra='real', b='integer', c='integer', d='real', e='integer',
                                 f='real', g='real', h='integer', i='string', j='integer', k='real'))

    def test_function_decl_stays(self):
        m = parse_verilog_a(mod("m", FUNC_SHADOW), filename="c.va")
        body = m.functions[0].body
        # the declaration assignment t = r + 1000 is not the first statement
        first = body.children[0] if body.children else body
        self.assertNotEqual(getattr(first, 'lhs', None), 't')


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_scope_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
