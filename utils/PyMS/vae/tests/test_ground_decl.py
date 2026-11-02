"""``ground`` declarations ground their net; a net merely NAMED gnd is a node.

THE DEFECTS (PYMS3 skeptic, 2026-10-26; BUGS.md CHANGES 1.1.3):
  * ``ground g;`` was recorded and never applied: ``electrical g; ground g;``
    left g an ordinary internal node, so I(n, g) <+ V(n, g)/1k floated (0.5
    on the 1 V / 1k divider where 1/3 is right), V(n, g) <+ 0.25 gave 0.5
    (0.25), a probe I(a, g) read 5e-10 (0.5), and a short V(a, g) <+ 0
    merged a and g into one floating node (0.5 for 1/3; 1.1.0 had 1/3: a
    regression of 1.1.1).  ``ground g;`` alone reached the C++ as an
    undeclared symbol (no math library).
  * the emitter's ground was the string 'gnd': an internal node or a PORT
    merely named gnd was taken for ground (V(x, gnd) <+ 0 collapsed x into
    the reference node: 0 for 1/3; I(p, gnd) on a port gnd went to ground).

THE FIX: parser.Parser._apply_grounds rewrites every reference to a declared
ground to the implicit-ground form (V(a, g) -> V(a), V(g, a) -> (-V(a)),
I(g, a) <+ e -> I(a) <+ -(e), a branch (g, a) b -> (a) b negated, ...) and
drops the net from the internal nodes; the emitter's ground is the
non-identifier GND = '<ground>'.

Run:  cd utils/PyMS && python3 vae/tests/test_ground_decl.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a, NodeKind  # noqa: E402
from vae.tests._decks import mod, rv, tran, run_decks, XYCE  # noqa: E402

GE = "  electrical a, g;\n  ground g;\n"
G1 = "  electrical g;\n  ground g;\n"
PG = dict(ports="p, gnd", decl="  inout p, gnd;\n  electrical p, gnd;\n")

CASES = {
    "gd_contrib": (mod("gd_contrib", G1 + "  analog begin I(p, n) <+ V(p, n) / 1000; I(n, g) <+ V(n, g) / 1000; end"), "", 1 / 3),
    "gd_contrib_rev": (mod("gd_contrib_rev", "  ground g;\n  electrical g;\n  analog begin I(p, n) <+ V(p, n) / 1000; I(g, n) <+ V(g, n) / 1000; end"), "", 1 / 3),
    "gd_ground_only": (mod("gd_ground_only", "  ground g;\n  analog begin I(p, n) <+ V(p, n) / 1000; I(n, g) <+ V(n, g) / 1000; end"), "", 1 / 3),
    "gd_vsrc": (mod("gd_vsrc", G1 + "  analog begin I(p, n) <+ V(p, n) / 1000; V(n, g) <+ 0.25; end"), "", 0.25),
    "gd_vsrc_rev": (mod("gd_vsrc_rev", G1 + "  analog begin I(p, n) <+ V(p, n) / 1000; V(g, n) <+ -0.25; end"), "", 0.25),
    "gd_short": (mod("gd_short", GE + "  analog begin I(p, n) <+ V(p, n) / 1000; I(n, a) <+ V(n, a) / 1000; V(a, g) <+ 0; end"), "", 1 / 3),
    "gd_short_rev": (mod("gd_short_rev", GE + "  analog begin I(p, n) <+ V(p, n) / 1000; I(n, a) <+ V(n, a) / 1000; V(g, a) <+ 0; end"), "", 1 / 3),
    "gd_probe_pair": (mod("gd_probe_pair", GE + "  analog begin I(p, a) <+ V(p, a) / 1000; V(a, g) <+ 0; I(n) <+ -0.5 * I(a, g); end"), "", 0.5),
    "gd_probe_rev": (mod("gd_probe_rev", GE + "  analog begin I(p, a) <+ V(p, a) / 1000; V(a, g) <+ 0; I(n) <+ 0.5 * I(g, a); end"), "", 0.5),
    "gd_branch_ga": (mod("gd_branch_ga", GE + "  branch (g, a) br;\n  analog begin I(p, a) <+ V(p, a) / 1000; V(br) <+ 0; I(n) <+ 0.5 * I(br); end"), "", 0.5),
    "gd_branch_contrib": (mod("gd_branch_contrib", G1 + "  branch (g, n) bg;\n  analog begin I(p, n) <+ V(p, n) / 1000; I(bg) <+ V(bg) / 1000; end"), "", 1 / 3),
    "gd_tran": (mod("gd_tran", G1 + "  analog begin I(p, n) <+ V(p, n) / 1000; I(n, g) <+ V(n, g) / 1000; end"), "", tran(1 / 3)),
    "gn_internal_short": (mod("gn_internal_short", "  electrical gnd, x;\n  analog begin I(p, x) <+ V(p, x) / 1000; V(x, gnd) <+ 0; I(gnd, n) <+ V(gnd, n) / 1000; end"), "", 1 / 3),
    "gn_port": (mod("gn_port", "  analog begin I(p, gnd) <+ V(p, gnd) / 1000; end", **PG), "", 0.5),
    "gn_port_and_ground": (mod("gn_port_and_ground", G1 + "  analog begin I(p, gnd) <+ V(p, gnd) / 1000; I(gnd, g) <+ V(gnd, g) / 1000; end", **PG), "", 1 / 3),
}


class TestParse(unittest.TestCase):

    def _mod(self, name):
        return parse_verilog_a(CASES[name][0], filename="c.va")

    def test_ground_is_no_internal_node(self):
        m = self._mod("gd_contrib")
        self.assertEqual(m.grounds, ["g"])
        self.assertEqual(m.internal_nodes, [])

    def test_references_take_the_implicit_ground_form(self):
        blk = self._mod("gd_contrib_rev").analog_block
        c = [s for s in blk.children if s.kind == NodeKind.CONTRIB][1]
        self.assertEqual(c.branch, ("n",))             # I(g, n) <+ e  ->  I(n) <+ -(e)
        self.assertEqual(c.expr.replace(" ", ""), "-((-V(n))/1000)")
        s = [x for x in self._mod("gd_short").analog_block.children if x.kind == NodeKind.CONTRIB][-1]
        self.assertEqual((s.branch, s.expr.strip()), (("a",), "0"))   # still a short, to ground

    def test_named_branch_to_ground(self):
        m = self._mod("gd_branch_ga")
        self.assertEqual(m.branch_map["br"], "a")
        self.assertNotIn("br", m.branch_neg_map)

    def test_a_node_named_gnd_stays_a_node(self):
        self.assertEqual(self._mod("gn_internal_short").internal_nodes, ["gnd", "x"])


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_gnd_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
