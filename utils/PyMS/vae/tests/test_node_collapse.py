"""Node collapse: ``V(x, y) <+ 0`` between internal nodes.

THE DEFECT (PYMS1 skeptic, 2026-10-25; BUGS.md CHANGES 1.1.1): a short ``V(x, y) <+ 0``
whose two ends are both internal nodes removed BOTH nodes
(ginac_emitter.GiNaCEmitter._find_shorted_nodes added every internal end),
and _resolve_shorted_node then found no surviving node and sent every
contribution at x and y to ground.  ``if (flag > 0) I(x, y) <+ V(x, y)/250;
else V(x, y) <+ 0;`` with 500 ohm and 250 ohm around it gave 0 on the 1 V /
1k divider (analytic 0.5714); a chain of shorts (the Mextram c3/c4/c1
topology) gave 0, silently, exit status 0.  The shell decided its node list
from the parameter defaults and the math library from the instance's
values, so a card that flipped such a condition also disagreed on the node
count (the old BUGS.md D4).

THE FIX: only a short that runs unconditionally collapses (both the shell
and the library see the same node list for every parameter set); the nodes
of one short class merge into ONE survivor (a port first, then ground, then
the first internal node declared), and a short inside a conditional is a
0 V source on the branch-current unknown of its branch.

THE AMMETER (PYMS2 skeptic, 2026-10-26; BUGS.md CHANGES 1.1.2): 1.1.1 collapsed
an unconditional short ``V(am) <+ 0`` even when its current was read
(``I(p, n) <+ I(am)``): the probe then sat on a degenerate branch and read 0
(the 1.1.0 value 0.6667 became 0.5 on c_ammeter_double).  A short whose
current is probed anywhere -- I(br), or I(a, b) on its node pair in either
order -- never collapses; it stays a 0 V source on its branch-current unknown.

Run:  cd utils/PyMS && python3 vae/tests/test_node_collapse.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a  # noqa: E402
from vae.ginac_emitter import GiNaCEmitter  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'


def mod(name, body, decl="  electrical x, y;\n"):
    return (HDR + "module %s(p, n);\n  inout p, n;\n  electrical p, n;\n%s%s\nendmodule\n"
            % (name, decl, body))


def rv(r):
    """V(out) of the 1 V / 1k divider with r ohm in the device."""
    return 1000.0 / (r + 1000.0)


COL = ("  parameter integer flag = 0;\n  analog begin\n    I(p, x) <+ V(p, x) / 500;\n"
       "    if (flag > 0) I(x, y) <+ V(x, y) / 250; else V(x, y) <+ 0;\n"
       "    I(y, n) <+ V(y, n) / 250;\n  end")
CHAIN = ("  analog begin\n    I(p, c3) <+ V(p, c3) / 12;\n    V(c3, c4) <+ 0;\n    V(c4, c1) <+ 0;\n"
         "    I(c3, c1) <+ V(c3, c1) / 7;\n    I(c1, n) <+ V(c1, n) / 988;\n  end")
MEXSHAPE = ("  parameter real rcbli = 0;\n  analog begin\n    I(p, c3) <+ V(p, c3) / 12;\n"
            "    if (rcbli > 0) begin I(c4, c1) <+ V(c4, c1) / rcbli; I(c3, c4) <+ V(c3, c4) / 5; end\n"
            "    else begin V(c4, c1) <+ 0.0; I(c3, c1) <+ V(c3, c1) / 5; end\n"
            "    I(c1, n) <+ V(c1, n) / 988;\n  end")
TOPORT = ("  analog begin\n    I(p, x) <+ V(p, x) / 1000;\n    V(x, n) <+ 0;\n  end")
TOGND = ("  analog begin\n    I(p, x) <+ V(p, x) / 1000;\n    V(x) <+ 0;\n"
         "    I(p, n) <+ V(p, n) / 3000;\n  end")

AM = "  electrical x;\n  branch (p, x) am;\n"
AMB = "  analog begin V(am) <+ 0; I(x, n) <+ V(x, n) / 1000; %s end"

# name -> (module, .model card, analytic V(out))
CASES = {
    "col_flag0": (mod("col_flag0", COL), "", rv(750)),
    "col_flag1": (mod("col_flag1", COL.replace("flag = 0", "flag = 1")), "", 0.5),
    # a card that flips the condition: the collapse follows the instance
    "col_flag0_card1": (mod("col_flag0_card1", COL), "flag=1", 0.5),
    "col_flag1_card0": (mod("col_flag1_card0", COL.replace("flag = 0", "flag = 1")), "flag=0", rv(750)),
    "col_chain": (mod("col_chain", CHAIN, "  electrical c1, c3, c4;\n"), "", rv(12 + 988)),
    "col_mextram_shape": (mod("col_mextram_shape", MEXSHAPE, "  electrical c1, c3, c4;\n"), "",
                          rv(12 + 5 + 988)),
    "col_mextram_shape_card": (mod("col_mextram_shape_card", MEXSHAPE, "  electrical c1, c3, c4;\n"),
                               "rcbli=10", rv(12 + 5 + 10 + 988)),
    "col_toport": (mod("col_toport", TOPORT, "  electrical x;\n"), "", 0.5),
    "col_tognd": (mod("col_tognd", TOGND, "  electrical x;\n"), "", 0.25),
    # ammeters: the probed short carries the current the probe reads
    "am_double": (mod("am_double", AMB % "I(p, n) <+ I(am);", AM), "", 2 / 3),
    "am_scaled": (mod("am_scaled", AMB % "I(p, n) <+ 0.5 * I(am);", AM), "", 0.6),
    "am_assign": (mod("am_assign", AMB % "i = I(am); I(p, n) <+ i;", AM + "  real i;\n"), "", 2 / 3),
    "am_unnamed": (mod("am_unnamed", "  analog begin V(p, x) <+ 0; I(x, n) <+ V(x, n) / 1000; "
                       "I(p, n) <+ I(p, x); end", "  electrical x;\n"), "", 2 / 3),
    "am_reversed": (mod("am_reversed", "  analog begin V(p, x) <+ 0; I(x, n) <+ V(x, n) / 1000; "
                        "I(n, p) <+ I(x, p); end", "  electrical x;\n"), "", 2 / 3),
    "am_gnd": (mod("am_gnd", "  analog begin V(am) <+ 0; I(x) <+ V(x) / 1000; I(p, n) <+ V(p, n) / 1000; "
                   "I(p, n) <+ I(am); end", "  electrical x;\n  branch (n, x) am;\n"), "", 0.5),
    "am_int": (mod("am_int", "  analog begin I(p, a) <+ V(p, a) / 500; V(am) <+ 0; I(b, n) <+ V(b, n) / 500; "
                   "I(p, n) <+ I(am); end", "  electrical a, b;\n  branch (a, b) am;\n"), "", 2 / 3),
}


class TestCollapseSets(unittest.TestCase):
    """The collapse set, without Xyce."""

    def _shorted(self, va, pv=None):
        m = parse_verilog_a(va, filename="t.va")
        return GiNaCEmitter(m, param_values=pv or {})._find_shorted_nodes(m.analog_block), m

    def test_conditional_short_does_not_collapse(self):
        for pv in ({}, {"flag": 0}, {"flag": 1}):
            s, _m = self._shorted(CASES["col_flag0"][0], pv)
            self.assertEqual(s, set(), pv)

    def test_chain_keeps_one_survivor(self):
        s, m = self._shorted(CASES["col_chain"][0])
        self.assertEqual(len(s), 2)
        surv = [n for n in m.internal_nodes if n not in s]
        self.assertEqual(len(surv), 1)
        em = GiNaCEmitter(m, param_values={})
        act = ["p", "n"] + surv
        for n in s:
            self.assertEqual(em._resolve_shorted_node(n, act), surv[0])

    def test_short_to_port_merges_into_the_port(self):
        s, m = self._shorted(CASES["col_toport"][0])
        self.assertEqual(s, {"x"})
        self.assertEqual(GiNaCEmitter(m, param_values={})._resolve_shorted_node("x", ["p", "n"]), "n")

    def test_short_to_ground(self):
        s, m = self._shorted(CASES["col_tognd"][0])
        self.assertEqual(s, {"x"})
        self.assertIsNone(GiNaCEmitter(m, param_values={})._resolve_shorted_node("x", ["p", "n"]))

    def test_probed_short_does_not_collapse(self):
        for name in ("am_double", "am_unnamed", "am_reversed", "am_gnd", "am_int"):
            s, _m = self._shorted(CASES[name][0])
            self.assertEqual(s, set(), name)

    def test_unprobed_short_still_collapses(self):
        s, _m = self._shorted(mod("am_plain", AMB % "", AM))
        self.assertEqual(s, {"x"})


    def test_probed_short_does_not_collapse(self):
        for name in ("am_double", "am_unnamed", "am_reversed", "am_gnd", "am_int"):
            s, _m = self._shorted(CASES[name][0])
            self.assertEqual(s, set(), name)

    def test_unprobed_short_still_collapses(self):
        s, _m = self._shorted(mod("am_plain", AMB % "", AM))
        self.assertEqual(s, {"x"})


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        work = tempfile.mkdtemp(prefix="pyms_collapse_")
        env = dict(os.environ)
        env.update(PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(work, "hc"),
                   PYMS_VAE_CACHE=os.path.join(work, "vc"), PYTHONDONTWRITEBYTECODE="1")
        env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        bad = []
        try:
            for name, (va, card, want) in CASES.items():
                d = os.path.join(work, name)
                os.makedirs(d)
                open(os.path.join(d, "a1.va"), "w").write(va)
                open(os.path.join(d, "a1.cir"), "w").write(
                    "* %s\n.hdl \"a1.va\"\nV1 in 0 DC 1\n.model m1 %s %s\nY%s X1 in out m1\n"
                    "R2 out 0 1k\n.op\n.print dc v(out)\n.end\n" % (name, name, card, name))
                r = subprocess.run([XYCE, "a1.cir"], cwd=d, env=env, capture_output=True,
                                   text=True, timeout=900)
                got = None
                try:
                    rows = [l.split() for l in open(os.path.join(d, "a1.cir.prn"))]
                    got = float(rows[1][1])
                except (OSError, IndexError, ValueError):
                    pass
                ok = r.returncode == 0 and got is not None and abs(got - want) <= 1e-6 * max(1.0, abs(want))
                print("%-24s want %.10g got %s rc %d" % (name, want, got, r.returncode), flush=True)
                if not ok:
                    bad.append("%s: want %.10g got %s (rc %d) %s" % (
                        name, want, got, r.returncode, (r.stdout + r.stderr)[-600:] if got is None else ""))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
