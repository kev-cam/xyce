"""case statements as the LRM defines them.

THE DEFECT (PYMS2 skeptic, 2026-10-26; BUGS.md CHANGES 1.1.2): Parser._parse_case
made every item an independent ``if`` and put the default arm's statements
in the enclosing block UNCONDITIONALLY, so the default ran beside the arm
that matched: ``case (mode) 0: I(p,n) <+ V(p,n)/1000; default: I(p,n) <+
V(p,n)/2000; endcase`` with mode 0 gave 0.6 on the 1 V / 1k divider (0.5 is
right); a V contribution in an arm met the default's I contribution on the
same branch and lost (0.333 for 0.5; a 0.2 V source gave 0.5 for 0.8); a
short in an arm gave 0.4 for 0.3333.  An item of more than one token
(``-1``, ``1 + 1``) became one item per token.

THE FIX: the case is the equivalent if / else-if / else chain: the items are
compared with the case expression in source order, the first arm with a
match runs and no other, the default (wherever written, its ``:``
optional) runs when none matches, an arm is one statement or the null
statement, an item is any expression.  Contributions, shorts and branch
kinds per arm behave as under if/else.

Run:  cd utils/PyMS && python3 vae/tests/test_case_statement.py
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

from vae.parser import parse_verilog_a, NodeKind, ParseError  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'


def mod(name, body, decl="  parameter integer mode = 0;\n"):
    return (HDR + "module %s(p, n);\n  inout p, n;\n  electrical p, n;\n%s"
            "  analog begin\n%s\n  end\nendmodule\n" % (name, decl, body))


def case(*arms, expr="mode"):
    return "    case (%s)\n%s\n    endcase" % (expr, "\n".join("      " + a for a in arms))


def g(r):
    return "I(p, n) <+ V(p, n) / %d;" % r


def rv(r):
    return 1000.0 / (r + 1000.0)


SER = ("  parameter integer mode = 0;\n  electrical x, y;\n", "    I(p, x) <+ V(p, x) / 500;\n%s\n"
       "    I(y, n) <+ V(y, n) / 500;")
GND = ("  parameter integer mode = 0;\n  electrical x;\n", "    I(p, n) <+ V(p, n) / 1000;\n"
       "    I(n, x) <+ V(n, x) / 1000;\n%s")
PLAIN = case("0: " + g(1000), "default: " + g(2000))
MULTI = case("0, 1: " + g(1000), "2: " + g(2000), "default: " + g(4000))
NEST = case("0: case (sub) 0: " + g(1000) + " default: " + g(2000) + " endcase", "default: " + g(4000))
SC = case("0: V(x, y) <+ 0;", "default: I(x, y) <+ V(x, y) / 1000;")

# name -> (module, .model card, analytic V(out))
CASES = {
    "plain0": (mod("plain0", PLAIN), "", 0.5),
    "plain1": (mod("plain1", PLAIN), "mode=1", rv(2000)),
    "multi1": (mod("multi1", MULTI), "mode=1", 0.5),
    "multi2": (mod("multi2", MULTI), "mode=2", rv(2000)),
    "multi5": (mod("multi5", MULTI), "mode=5", rv(4000)),
    "first_wins": (mod("first_wins", case("0: " + g(1000), "0: " + g(2000))), "", 0.5),
    "default_first": (mod("default_first", case("default: " + g(2000), "0: " + g(1000))), "", 0.5),
    "default_first3": (mod("default_first3", case("default: " + g(2000), "0: " + g(1000))), "mode=3", rv(2000)),
    "neg_item": (mod("neg_item", case("-1: " + g(1000), "default " + g(2000))), "mode=-1", 0.5),
    "expr_item": (mod("expr_item", case("1 + 1: " + g(1000), "default: " + g(2000))), "mode=2", 0.5),
    "null_arm": (mod("null_arm", "    " + g(4000) + "\n" + case("0: ;", "default: " + g(4000))), "", rv(4000)),
    "nested01": (mod("nested01", NEST, "  parameter integer mode = 0;\n  parameter integer sub = 1;\n"), "", rv(2000)),
    "nested1": (mod("nested1", NEST, "  parameter integer mode = 0;\n  parameter integer sub = 1;\n"), "mode=1",
                rv(4000)),
    "vsrc0": (mod("vsrc0", case("0: V(p, n) <+ 0.2;", "default: " + g(1000))), "", 0.8),
    "vsrc1": (mod("vsrc1", case("0: V(p, n) <+ 0.2;", "default: " + g(1000))), "mode=1", 0.5),
    "short0": (mod("short0", SER[1] % SC, SER[0]), "", 0.5),
    "short1": (mod("short1", SER[1] % SC, SER[0]), "mode=1", rv(2000)),
    "short_block0": (mod("short_block0", SER[1] % case("0: begin V(x, y) <+ 0; end",
                                                         "default: begin I(x, y) <+ V(x, y) / 1000; end"),
                         SER[0]), "", 0.5),
    "short_gnd0": (mod("short_gnd0", GND[1] % case("0: V(x) <+ 0;", "default: I(x) <+ V(x) / 1000;"),
                       GND[0]), "", 1 / 3),
    "short_gnd1": (mod("short_gnd1", GND[1] % case("0: V(x) <+ 0;", "default: I(x) <+ V(x) / 1000;"),
                       GND[0]), "mode=1", 0.4),
    "vdep": (mod("vdep", case("1: " + g(1000), "default: " + g(2000), expr="V(p, n) > 0.3")), "", 0.5),
    "assign": (mod("assign", case("0: r = 2000;", "1: r = 1000;", "default: r = 4000;")
                   + "\n    I(p, n) <+ V(p, n) / r;", "  parameter integer mode = 1;\n  real r;\n"), "", 0.5),
}


class TestParse(unittest.TestCase):

    def _top(self, va):
        return parse_verilog_a(va, filename="c.va").analog_block.children[0]

    def test_chain_shape(self):
        n = self._top(CASES["multi1"][0])
        self.assertEqual(n.kind, NodeKind.IF)
        self.assertEqual(n.condition, "mode == 0 || mode == 1")
        self.assertEqual(n.else_body.kind, NodeKind.IF)
        self.assertEqual(n.else_body.condition, "mode == 2")
        self.assertEqual(n.else_body.else_body.kind, NodeKind.CONTRIB)   # the default

    def test_default_written_first_is_still_the_last_else(self):
        n = self._top(CASES["default_first"][0])
        self.assertEqual(n.condition, "mode == 0")
        self.assertEqual(n.else_body.expr.split(), "V ( p , n ) / 2000".split())

    def test_items_are_expressions(self):
        self.assertEqual(self._top(CASES["neg_item"][0]).condition, "mode == (-1)")
        self.assertEqual(self._top(CASES["expr_item"][0]).condition, "mode == (1 + 1)")
        self.assertEqual(self._top(CASES["vdep"][0]).condition, "(V ( p , n ) > 0.3) == 1")

    def test_errors(self):
        with self.assertRaisesRegex(ParseError, r"c\.va:\d+: a case statement has more than one default"):
            parse_verilog_a(mod("e1", case("default: " + g(1000), "default: " + g(2000))), filename="c.va")
        with self.assertRaisesRegex(ParseError, r"c\.va:\d+: "):
            parse_verilog_a(mod("e2", "    case (mode)\n      0: " + g(1000)), filename="c.va")


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        work = tempfile.mkdtemp(prefix="pyms_case_")
        env = dict(os.environ)
        env.update(PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(work, "hc"),
                   PYMS_VAE_CACHE=os.path.join(work, "vc"), PYTHONDONTWRITEBYTECODE="1")
        env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        bad = []
        try:
            for name, (va, card, want) in CASES.items():
                d = os.path.join(work, name)
                os.makedirs(d)
                open(os.path.join(d, "c.va"), "w").write(va)
                open(os.path.join(d, "c.cir"), "w").write(
                    "* %s\n.hdl \"c.va\"\nV1 in 0 DC 1\n.model m1 %s %s\nY%s X1 in out m1\n"
                    "R2 out 0 1k\n.op\n.print dc v(out)\n.end\n" % (name, name, card, name))
                r = subprocess.run([XYCE, "c.cir"], cwd=d, env=env, capture_output=True,
                                   text=True, timeout=900)
                got = None
                try:
                    got = float([l.split() for l in open(os.path.join(d, "c.cir.prn"))][1][1])
                except (OSError, IndexError, ValueError):
                    pass
                ok = r.returncode == 0 and got is not None and abs(got - want) <= 1e-6 * max(1.0, abs(want))
                print("%-16s want %.10g got %s rc %d" % (name, want, got, r.returncode), flush=True)
                if not ok:
                    bad.append("%s: want %.10g got %s (rc %d) %s" % (
                        name, want, got, r.returncode, (r.stdout + r.stderr)[-600:] if got is None else ""))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
