"""A block-local declaration assignment gives the variable its value once
(Verilog-A variables are static), not every time the block is entered (1.1.4).

THE DEFECT (PYMS4 skeptic, s4_loop_body_local_init, 2026-10-27): ``for (i = 0;
i < 3; i = i + 1) begin : lb real t = 0; t = t + 1; g = g + t * (1m / 3);
end`` re-ran ``t = 0`` on every pass (t = 1, 1, 1: g = 1m, V(out) 0.5); VACASK
keeps t (1, 2, 3: g = 2m, V(out) 2/3).

THE READING (LRM 2.4 Annex A, real_declaration: real_type ::=
real_identifier [ = constant_expression ]): the declaration assignment is a
CONSTANT expression and an analog variable is static, so it is the
variable's initial value, given once -- for PyMS, whose variables start
every evaluation afresh, once per evaluation -- and a block entered again
does not reset it.  parser.py hoists a constant block-local declaration
assignment to the head of the analog block (of the function body inside an
analog function); a non-constant one (not LRM Verilog-A) stays where it is
written.

Run:  cd utils/PyMS && python3 vae/tests/test_decl_init_static.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a, NodeKind  # noqa: E402
from vae.tests._decks import mod, run_decks, XYCE  # noqa: E402

CASES = {
    "loop_body_local_init": (mod("loop_body_local_init", "  integer i;\n  real g;\n  analog begin\n    g = 0;\n    for (i = 0; i < 3; i = i + 1) begin : lb\n      real t = 0;\n      t = t + 1;\n      g = g + t * (1m / 3);\n    end\n    I(p, n) <+ V(p, n) * g;\n  end"), "", 2.0 / 3.0),
    "init_param": (mod("init_param", "  parameter real g0 = 0.5m;\n  integer i;\n  real g;\n  analog begin\n    g = 0;\n    for (i = 0; i < 2; i = i + 1) begin : lb\n      real t = g0;\n      g = g + t;\n      t = 0;\n    end\n    I(p, n) <+ V(p, n) * g * 2;\n  end"), "", 0.5),
    "init_nonconst_stays": (mod("init_nonconst_stays", "  integer i;\n  real g;\n  analog begin\n    g = 0;\n    for (i = 0; i < 2; i = i + 1) begin : lb\n      real t = g + 0.25m;\n      g = g + t;\n    end\n    I(p, n) <+ V(p, n) * g * 4 / 3;\n  end"), "", 0.5),
    "func_loop_init": (mod("func_loop_init", "  analog function real f;\n    input m;\n    integer m;\n    integer i;\n    real g;\n    begin\n      g = 0;\n      for (i = 0; i < m; i = i + 1) begin : lb\n        real t = 0;\n        t = t + 1;\n        g = g + t;\n      end\n      f = g;\n    end\n  endfunction\n  analog I(p, n) <+ V(p, n) / (f(3) * 1000 / 6);"), "", 0.5),
}


class TestParse(unittest.TestCase):

    def test_hoisted_to_the_head(self):
        blk = parse_verilog_a(CASES["loop_body_local_init"][0], filename="c.va").analog_block
        first = blk.children[0]
        self.assertEqual(first.kind, NodeKind.ASSIGN)
        self.assertEqual(first.decl_init, 'block')


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_dinit_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
