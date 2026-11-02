"""for / while / repeat loops iterate (LRM 5.10).

THE DEFECT (BUGS.md A9, PYMS3 fixer, 2026-10-26): the parser dropped a loop's
header and kept its body as one pass: ``for (i = 0; i < 3; i = i + 1) I(p, n)
<+ V(p, n) / 3000;`` contributed 1/3000 S once (0.25 on the 1 V / 1k divider,
0.5 is right); a zero-trip loop ran once (0.6667 for 0.5); ``repeat`` was not
a statement.

THE FIX: a loop is a BLOCK [init, body, step] with node.loop set;
GiNaCEmitter._walk_loop unrolls it: exactly, when the condition is known at
elaboration (parameters, folded variables; a zero-trip loop runs not at all);
under a chain of runtime indicators when the condition reads the node
voltages, up to the bound of a counter conjunct (``k < N`` with k stepped once
per iteration: HiCUM L2's ``l_it <= l_itmax``); with no such bound, and for a
repeat count unknown at elaboration, a file:line error.

1.1.4 (PYMS4 skeptic C14: past 16 run-time passes 1.1.3 warned once and went
on with the values after 16, l4_vdep_counter_20 0.444 / l4_vdep_unbounded_17
0.667 for 0.5): a loop whose condition only the run time knows is a C++ loop
of the eval code (GiNaCEmitter._walk_runtime_loop), its body walked once and
run as often as the condition says, exactly; a current contribution in it
adds to an accumulator.  Only a body with ddt(), a voltage contribution or an
event statement is still unrolled (16 passes), and an evaluation that would
need more is a hard error of the simulation naming the loop; so is a
run-time loop past 100000 passes.

Run:  cd utils/PyMS && python3 vae/tests/test_loops.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a, NodeKind  # noqa: E402
from vae.ginac_emitter import GiNaCEmitter, EmitError  # noqa: E402
from vae.tests._decks import mod, rv, tran, run_decks, XYCE  # noqa: E402

CASES = {
    "for_const": (mod("for_const", "  genvar i;\n  analog begin\n    for (i = 0; i < 3; i = i + 1) I(p, n) <+ V(p, n) / 3000;\n  end"), "", 0.5),
    "for_zero": (mod("for_zero", "  genvar i;\n  analog begin\n    I(p, n) <+ V(p, n) / 1000;\n    for (i = 0; i < 0; i = i + 1) I(p, n) <+ V(p, n) / 1000;\n  end"), "", 0.5),
    "for_param_card": (mod("for_param_card", "  parameter integer N = 4;\n  integer i;\n  analog begin\n    for (i = 0; i < N; i = i + 1) I(p, n) <+ V(p, n) / 4000;\n  end"), "N=2", rv(2000)),
    "while_sum": (mod("while_sum", "  integer k;\n  real g;\n  analog begin\n    k = 0; g = 0;\n    while (k < 2) begin g = g + 0.5m; k = k + 1; end\n    I(p, n) <+ V(p, n) * g;\n  end"), "", 0.5),
    "while_zero": (mod("while_zero", "  integer k;\n  real g;\n  analog begin\n    k = 5; g = 1m;\n    while (k < 2) begin g = g + 1m; k = k + 1; end\n    I(p, n) <+ V(p, n) * g;\n  end"), "", 0.5),
    "repeat": (mod("repeat", "  real g;\n  analog begin\n    g = 0;\n    repeat (4) g = g + 0.25m;\n    I(p, n) <+ V(p, n) * g;\n  end"), "", 0.5),
    "nested": (mod("nested", "  integer i, j;\n  real g;\n  analog begin\n    g = 0;\n    for (i = 0; i < 2; i = i + 1)\n      for (j = 0; j < 2; j = j + 1) g = g + 0.25m;\n    I(p, n) <+ V(p, n) * g;\n  end"), "", 0.5),
    "vdep_counter": (mod("vdep_counter", "  integer k;\n  real r;\n  analog begin\n    k = 0; r = 8000;\n    while (r > 1000 * (1 + V(p, n)) && k < 10) begin r = r / 2; k = k + 1; end\n    I(p, n) <+ V(p, n) / r;\n  end"), "", 0.5),
    "vdep_newton": (mod("vdep_newton", "  integer it;\n  real x, d;\n  analog begin\n    x = 1 + 0.001 * V(p, n); d = 1; it = 0;\n    while (abs(d) > 1e-14 && it < 60) begin d = (x * x - 2) / (2 * x); x = x - d; it = it + 1; end\n    I(p, n) <+ V(p, n) / (1000 * x / sqrt(2));\n  end"), "", 0.5),
    # past 16 passes: exact (the skeptic's l4_vdep_counter_20, l4_vdep_unbounded_17)
    "vdep_counter_20": (mod("vdep_counter_20", "  integer k;\n  real g;\n  analog begin\n    k = 0; g = 0;\n    while (k < 20 && V(p, n) > -1) begin g = g + 0.05m; k = k + 1; end\n    I(p, n) <+ V(p, n) * g;\n  end"), "", 0.5),
    "vdep_unbounded_17": (mod("vdep_unbounded_17", "  real x;\n  analog begin\n    x = 1;\n    while (x < 1e5 + V(p, n)) x = x * 2;\n    I(p, n) <+ V(p, n) / (1000 * x / 131072);\n  end"), "", 0.5),
    "vdep_contrib_20": (mod("vdep_contrib_20", "  integer k;\n  analog begin\n    k = 0;\n    while (k < 20 && V(p, n) > -1) begin I(p, n) <+ V(p, n) / 20000; k = k + 1; end\n  end"), "", 0.5),
    "vdep_endless": (mod("vdep_endless", "  real x;\n  analog begin\n    x = 1;\n    while (x > 0 * V(p, n)) x = x + 1;\n    I(p, n) <+ V(p, n) / 1000;\n  end"), "", "ERR"),
    # no counter: a run-time loop; 3 halvings are needed
    "vdep_unbounded": (mod("vdep_unbounded", "  real r;\n  analog begin\n    r = 8000;\n    while (r > 1000 * (1 + V(p, n))) r = r / 2;\n    I(p, n) <+ V(p, n) / r;\n  end"), "", 0.5),
    "tran": (mod("tran", "  integer i;\n  analog begin\n    for (i = 0; i < 2; i = i + 1) I(p, n) <+ V(p, n) / 2000;\n  end"), "", tran(0.5)),
}


class TestParse(unittest.TestCase):

    def test_loop_node(self):
        top = parse_verilog_a(CASES["for_const"][0], filename="c.va").analog_block.children[0]
        self.assertEqual(top.kind, NodeKind.BLOCK)
        self.assertEqual(top.loop["kind"], "for")
        self.assertEqual(top.loop["cond"].split(), "i < 3".split())
        self.assertEqual([c.kind for c in top.children], [NodeKind.ASSIGN, NodeKind.CONTRIB, NodeKind.ASSIGN])

    def _prog(self, name):
        return GiNaCEmitter(parse_verilog_a(CASES[name][0], filename="c.va")).emit()

    def test_unrolled_count(self):
        for name, n in (("for_const", 3), ("for_zero", 1)):
            self.assertEqual(self._prog(name).count("F_contribs[br_idx] +="), n, name)

    def test_folded_sums(self):
        # g = 4 x 0.25m (repeat), 2 x 2 x 0.25m (nested), 2 x 0.5m (while): 1m
        for name in ("repeat", "nested", "while_sum", "while_zero"):
            line = [l for l in self._prog(name).splitlines() if "F_contribs[br_idx] +=" in l][0]
            self.assertIn("0.001", line, name)

    def test_runtime_loop(self):
        # a loop of the eval code, its body once, no guard
        for name in ("vdep_unbounded", "vdep_counter", "vdep_counter_20"):
            prog = GiNaCEmitter(parse_verilog_a(CASES[name][0], filename="c.va")).emit()
            self.assertEqual(prog.count("for (int _it"), 1, name)
            self.assertEqual(prog.count("_vae_loop_over(("), 0, name)

    def test_unrolled_body_with_ddt_guarded(self):
        va = mod("ddtloop", "  integer k;\n  analog begin\n    I(p, n) <+ V(p, n) / 1000;\n    k = 0;\n    while (k < 20 && V(p, n) > -10) begin I(n) <+ ddt(0.05u * V(n)); k = k + 1; end\n  end")
        prog = GiNaCEmitter(parse_verilog_a(va, filename="c.va")).emit()
        self.assertEqual(prog.count("for (int _it"), 0)
        self.assertEqual(prog.count("_vae_loop_over(("), 1)

    def test_errors(self):
        va = mod("e", "  real g;\n  analog begin\n    g = 0;\n    repeat (1 + V(p, n)) g = g + 0.25m;\n    I(p, n) <+ V(p, n) * g;\n  end")
        with self.assertRaisesRegex(EmitError, r"c\.va:\d+: repeat"):
            GiNaCEmitter(parse_verilog_a(va, filename="c.va")).emit()


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_loops_"), [])

    def test_twenty_halvings_exact(self):
        # 20 halvings are needed (1.1.3 stopped after 16: r = 1000 * 2^4, and a warning)
        va = mod("halvings", "  real r;\n  analog begin\n    r = 1000 * 1048576;\n    while (r > 1000 * (1 + V(p, n))) r = r / 2;\n    I(p, n) <+ V(p, n) / r;\n  end")
        self.assertEqual(run_decks({"halvings": (va, "", 0.5)}, "pyms_loopg_"), [])

    def test_unrolled_past_bound_is_an_error(self):
        va = mod("ddtover", "  integer k;\n  analog begin\n    I(p, n) <+ V(p, n) / 1000;\n    k = 0;\n    while (k < 20 && V(p, n) > -10) begin I(n) <+ ddt(0.05u * V(n)); k = k + 1; end\n  end")
        logs = {}
        self.assertEqual(run_decks({"ddtover": (va, "", "ERR", "tran")}, "pyms_loope_", logs=logs), [])
        self.assertRegex(" ".join(logs["ddtover"].split()), r"c\.va:\d+: this while loop needs more than the 16 iterations PyMS unrolls")


if __name__ == "__main__":
    unittest.main(verbosity=2)
