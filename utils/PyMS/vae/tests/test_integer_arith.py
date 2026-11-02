"""Integer arithmetic in the analog block (LRM 4.2: Verilog-A typing).

THE DEFECT (BUGS.md A8, PYMS3 fixer / skeptic, 2026-10-26): the statements of
the analog block were translated with real arithmetic: ``integer k = 3; t =
k / 2;`` gave t = 1.5 (Verilog-A: 1, the quotient of two integers truncates),
``7 % 3`` reached the C++ on doubles, and ``k = 2.5`` kept 2.5 in an integer
variable (Verilog-A: 3, a real assigned to an integer is rounded, halves away
from zero).  The parameter defaults had these semantics since 1.1.0.

THE FIX: int_ops.lower_integer_ops types every analog-block expression with
the declared types and rewrites an integer ``a / b`` to ``_vae_idiv(a, b)``
and ``a % b`` to ``_vae_imod(a, b)``; an assignment of a real to an integer
variable is rounded (``_vae_rint``).  The emitter evaluates them in Python at
elaboration, as GiNaC functions in the metaprogram (piecewise constant,
derivative 0) and as C++ helpers in the eval code.

Run:  cd utils/PyMS && python3 vae/tests/test_integer_arith.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a, NodeKind  # noqa: E402
from vae.tests._decks import mod, rv, run_decks, XYCE  # noqa: E402

CASES = {
    "var_div": (mod("var_div", "  integer k;\n  real t;\n  analog begin\n    k = 3; t = k / 2;\n    I(p, n) <+ V(p, n) / (1000 * t);\n  end"), "", 0.5),
    "lit_div": (mod("lit_div", "  real t;\n  analog begin\n    t = 7 / 2;\n    I(p, n) <+ V(p, n) / (1000 * t / 3);\n  end"), "", 0.5),
    "real_div": (mod("real_div", "  real t;\n  analog begin\n    t = 7.0 / 2;\n    I(p, n) <+ V(p, n) / (1000 * t / 3.5);\n  end"), "", 0.5),
    "neg_div": (mod("neg_div", "  integer k;\n  analog begin\n    k = -7 / 2;\n    I(p, n) <+ V(p, n) / (-1000 * k / 3);\n  end"), "", 0.5),
    "neg_mod": (mod("neg_mod", "  integer k;\n  analog begin\n    k = -7 % 3;\n    I(p, n) <+ V(p, n) / (-1000 * k);\n  end"), "", 0.5),
    "param_div_card": (mod("param_div_card", "  parameter integer m = 5;\n  analog I(p, n) <+ V(p, n) / (500 * (m / 2));"), "m=7", rv(1500)),
    "in_contrib": (mod("in_contrib", "  parameter integer m = 3;\n  analog I(p, n) <+ V(p, n) * (m / 2) / 1000;"), "", 0.5),
    "round_assign": (mod("round_assign", "  integer k;\n  analog begin\n    k = 2.5;\n    I(p, n) <+ V(p, n) / (1000 * k / 3);\n  end"), "", 0.5),
    "round_assign_neg": (mod("round_assign_neg", "  integer k;\n  analog begin\n    k = -2.5;\n    I(p, n) <+ V(p, n) / (-1000 * k / 3);\n  end"), "", 0.5),
    "round_vdep": (mod("round_vdep", "  integer k;\n  analog begin\n    k = 1.4 + V(p, n) * 0.01;\n    I(p, n) <+ V(p, n) / (1000 * k);\n  end"), "", 0.5),
    "in_cond": (mod("in_cond", "  parameter integer m = 3;\n  analog begin\n    if (m / 2 == 1) I(p, n) <+ V(p, n) / 1000;\n    else I(p, n) <+ V(p, n) / 2000;\n  end"), "", 0.5),
    "vdep_intvar_div": (mod("vdep_intvar_div", "  integer k;\n  real r;\n  analog begin\n    k = (V(p, n) > 0.1) ? 5 : 9;\n    r = 500 * (k / 2);\n    I(p, n) <+ V(p, n) / r;\n  end"), "", 0.5),
    "mixed_real": (mod("mixed_real", "  integer k;\n  real t;\n  analog begin\n    k = 3; t = k / 2.0;\n    I(p, n) <+ V(p, n) / (1000 * t / 1.5);\n  end"), "", 0.5),
}


class TestRewrite(unittest.TestCase):

    def _assigns(self, name):
        m = parse_verilog_a(CASES[name][0], filename="c.va")
        return [s for s in m.analog_block.children if s.kind == NodeKind.ASSIGN]

    def test_integer_division_is_rewritten(self):
        a = self._assigns("var_div")[1]
        self.assertEqual(a.expr.split(), "_vae_idiv ( k , 2 )".split())
        self.assertEqual(self._assigns("neg_mod")[0].expr.split(), "_vae_imod ( - 7 , 3 )".split())

    def test_real_division_is_not(self):
        self.assertEqual(self._assigns("real_div")[0].expr.split(), "7.0 / 2".split())
        self.assertEqual(self._assigns("mixed_real")[1].expr.split(), "k / 2.0".split())

    def test_integer_assignment_is_marked_for_rounding(self):
        self.assertTrue(self._assigns("round_assign")[0].int_lhs)
        self.assertFalse(self._assigns("var_div")[0].int_lhs)      # k = 3: already an integer


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_iarith_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
