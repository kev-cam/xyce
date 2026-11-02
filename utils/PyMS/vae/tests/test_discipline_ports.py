#!/usr/bin/env python3
"""A port (or node) of a non-electrical discipline is a node of MNA like any
other: ``thermal t;`` with ``Temp(t)`` reads and ``Pwr(t) <+`` contributions.

THE BUG (2026-10-24, PYMS1-I; PYMS_BUGS.md P11): the parser knew only the
``electrical`` declaration and the ``I``/``V`` access functions.  ``thermal t;``
fell through the unknown-token loop (the port stayed "electrical", which is
harmless to MNA), ``Temp(t)`` was a node voltage in the emitter (right), but a
``Pwr(t) <+ ...`` statement was not a contribution: _parse_simple_statement
collected it as a raw EXPR node that both emitters skip.  Measured on the
module below with the thermal port on a node of its own (Rth = 1k to ground):
Xyce out = 0.5, V(th) = 0, exit 0, no message; VACASK out = 0.546593,
th = 0.205566; analytic out = 1 - u, th = u^2, u the real root of
u^3 + 2u - 1 = 0 (out = 0.5466023485, th = 0.2055694304).  With the port
tied to ground or to a driven node both engines agree (0.5 and 2/3), which is
how it went unnoticed.

The same root cause made an INTERNAL thermal node no node at all: VBIC 1.3's
``thermal dt; branch (dt) b_rth; ... Temp(b_rth)`` and MVSG 1.1 reached the
math library's g++ as "'V_dt' was not declared in this scope" (the emitter
turned Temp(b_rth) into V_dt, but dt was never declared).

THE FIX (parser.py): LRM_DISCIPLINES names the continuous disciplines of
disciplines.vams with their access functions; a declaration with any of them
is a node declaration (Parser._parse_node_decl) that records
Port.discipline or makes an internal node; the access functions of the
disciplines a module declares nodes with (V and I always) are mapped to V
(potential) or I (flow) in contributions and expressions, so the rest of PyMS
sees electrical nodes only; natures and disciplines declared in the source
(and those of the shipped disciplines.vams) extend the table; a contribution
through a name that is no such access function is a ParseError naming it,
never a dropped statement.  An attribute instance after a node name, hicum
L0's ``electrical c `P(info="...");``, names no node.

The parse-level tests need no engine.  The Xyce class runs the three decks
(thermal port on its own node, tied to ground, tied to a driven node) through
THIS copy of PyMS (PYMS_DIR = the tree this file sits in, private caches) and
skips without a Xyce binary ($XYCE, default /usr/local/src/xyce-build/src/Xyce).

Run:  python3 vae/tests/test_discipline_ports.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# Make `vae.*` importable when run as a plain script, from THIS tree.
_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

import vae.parser as _parser  # noqa: E402
from vae.parser import parse_verilog_a, ContribKind, NodeKind, ParseError  # noqa: E402

# Absent before the fix: the tests that need it fail on it, cleanly.
LRM_DISCIPLINES = getattr(_parser, 'LRM_DISCIPLINES', None)

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)

# Analytic solution of the "own node" deck: V(p,n) = u with u^3 + 2u - 1 = 0
# (r = rth = 1k, alpha = 1): out = 1 - u, Temp(t) = u^2.
U = 0.45339765151640377
OUT_OWN = 1.0 - U            # 0.5466023484835962
TH_OWN = U * U               # 0.2055694304005903
OUT_GND = 0.5                # Temp(t) = 0: a plain 1k resistor in the divider
OUT_DRV = 2.0 / 3.0          # Temp(t) = V(in) = 1: conductance doubled
RTOL = 1e-6

VA_T1 = """`include "disciplines.vams"
`include "constants.vams"
module t1(p, n, t);
  inout p, n, t;
  electrical p, n;
  thermal t;
  parameter real r = 1k;
  parameter real alpha = 1.0;
  analog begin
    I(p, n) <+ V(p, n) / r * (1 + alpha * Temp(t));
    Pwr(t) <+ -V(p, n) * V(p, n) / r;
  end
endmodule
"""

DECKS = {
    "own": """* thermal port on a node of its own (Rth = 1k to ground)
.hdl "t1.va"
V1 in 0 1
.model m1 t1
YT1 X1 in out th m1
R2 out 0 1k
Rth th 0 1k
.op
.print dc v(out) v(th)
.end
""",
    "gnd": """* thermal port tied to ground
.hdl "t1.va"
V1 in 0 1
.model m1 t1
YT1 X1 in out 0 m1
R2 out 0 1k
.op
.print dc v(out)
.end
""",
    "drv": """* thermal port tied to the driven node
.hdl "t1.va"
V1 in 0 1
.model m1 t1
YT1 X1 in out in m1
R2 out 0 1k
.op
.print dc v(out)
.end
""",
}


def _contribs(node, out=None):
    """Every CONTRIB node under `node`, in source order."""
    if out is None:
        out = []
    if node is None:
        return out
    if node.kind == NodeKind.CONTRIB:
        out.append(node)
    for ch in (node.children or []):
        _contribs(ch, out)
    if getattr(node, 'else_body', None) is not None:
        _contribs(node.else_body, out)
    return out


class ParseTests(unittest.TestCase):

    def test_thermal_port_keeps_its_discipline(self):
        mod = parse_verilog_a(VA_T1, filename='t1.va')
        self.assertEqual([p.name for p in mod.ports], ['p', 'n', 't'])
        self.assertEqual([p.discipline for p in mod.ports],
                         ['electrical', 'electrical', 'thermal'])
        self.assertEqual(mod.internal_nodes, [])

    def test_pwr_contribution_is_a_flow_contribution(self):
        mod = parse_verilog_a(VA_T1, filename='t1.va')
        cs = _contribs(mod.analog_block)
        self.assertEqual(len(cs), 2, [c.expr for c in cs])
        self.assertEqual(cs[1].contrib_kind, ContribKind.I)
        self.assertEqual(tuple(cs[1].branch), ('t',))
        self.assertIn('V ( p , n )', cs[1].expr)

    def test_temp_read_is_a_potential_read(self):
        mod = parse_verilog_a(VA_T1, filename='t1.va')
        first = _contribs(mod.analog_block)[0]
        self.assertEqual(first.contrib_kind, ContribKind.I)
        self.assertNotIn('Temp', first.expr)
        self.assertIn('V ( t )', first.expr)

    def test_access_functions_in_conditions(self):
        src = VA_T1.replace("Pwr(t) <+ -V(p, n) * V(p, n) / r;",
                            "if (Temp(t) > 0.1) Pwr(t) <+ -1m; else Pwr(t) <+ 0;")
        mod = parse_verilog_a(src, filename='t1.va')
        ifs = [c for c in mod.analog_block.children if c.kind == NodeKind.IF]
        self.assertEqual(len(ifs), 1)
        self.assertIn('V ( t )', ifs[0].condition)
        self.assertNotIn('Temp', ifs[0].condition)
        self.assertEqual(len(_contribs(mod.analog_block)), 3)

    def test_internal_thermal_node(self):
        src = VA_T1.replace("module t1(p, n, t);", "module t1(p, n);") \
                   .replace("inout p, n, t;", "inout p, n;") \
                   .replace("thermal t;", "thermal t, dt;")
        mod = parse_verilog_a(src, filename='t1.va')
        self.assertEqual([p.name for p in mod.ports], ['p', 'n'])
        self.assertEqual(mod.internal_nodes, ['t', 'dt'])

    def test_internal_thermal_node_through_a_branch_is_declared(self):
        # VBIC 1.3's form: the GiNaC program must declare V_dt (it was a g++
        # error: 'V_dt' was not declared in this scope).
        from vae.ginac_emitter import emit_ginac_program
        src = VA_T1.replace("module t1(p, n, t);", "module t1(p, n);") \
                   .replace("inout p, n, t;", "inout p, n;") \
                   .replace("thermal t;", "thermal dt;\n  branch (dt) b_rth;") \
                   .replace("Temp(t)", "Temp(b_rth)") \
                   .replace("    Pwr(t) <+ -V(p, n) * V(p, n) / r;\n",
                            "    Pwr(b_rth) <+ Temp(b_rth) / 1k;\n"
                            "    Pwr(b_rth) <+ -V(p, n) * V(p, n) / r;\n")
        mod = parse_verilog_a(src, filename='t1.va')
        self.assertEqual(mod.internal_nodes, ['dt'])
        cs = _contribs(mod.analog_block)
        self.assertEqual([(c.contrib_kind, tuple(c.branch)) for c in cs[1:]],
                         [(ContribKind.I, ('b_rth',))] * 2)
        self.assertIn('V ( b_rth )', cs[0].expr)
        prog = emit_ginac_program(mod)
        self.assertIn('symbol V_dt("V_dt");', prog)

    def test_attribute_in_a_node_declaration_names_no_node(self):
        # hicum L0: ``electrical c `P(info="external collector node");``
        src = VA_T1.replace("electrical p, n;",
                            'electrical p (* info="plus" *);\n  electrical n (*info="minus"*);')
        mod = parse_verilog_a(src, filename='t1.va')
        self.assertEqual(mod.internal_nodes, [])
        self.assertEqual([p.discipline for p in mod.ports],
                         ['electrical', 'electrical', 'thermal'])

    def test_every_lrm_discipline_is_a_node_declaration(self):
        self.assertIsNotNone(LRM_DISCIPLINES, "parser.LRM_DISCIPLINES is missing")
        self.assertIn('thermal', LRM_DISCIPLINES)
        for disc, (pot, flow) in LRM_DISCIPLINES.items():
            src = VA_T1.replace("thermal t;", "%s t;" % disc)
            if pot:
                src = src.replace("Temp(t)", "%s(t)" % pot)
            else:
                src = src.replace("alpha * Temp(t)", "0")
            if flow:
                src = src.replace("Pwr(t)", "%s(t)" % flow)
            else:
                src = src.replace("    Pwr(t) <+ -V(p, n) * V(p, n) / r;\n", "")
            mod = parse_verilog_a(src, filename='t1.va')
            self.assertEqual(mod.ports[2].discipline, disc)
            cs = _contribs(mod.analog_block)
            if flow:
                self.assertEqual(cs[-1].contrib_kind, ContribKind.I, disc)
                self.assertEqual(tuple(cs[-1].branch), ('t',), disc)
            if pot:
                self.assertIn('V ( t )', cs[0].expr, disc)

    def test_source_declared_nature_and_discipline(self):
        src = """
nature Heat
  units = "W"; access = Hx; abstol = 1e-9;
endnature
nature Deg
  units = "K"; access = Dx; abstol = 1e-4;
endnature
discipline mytherm
  potential Deg;
  flow Heat;
enddiscipline
module t2(p, n, t);
  inout p, n, t;
  electrical p, n;
  mytherm t;
  parameter real r = 1k;
  real x;
  analog begin
    x = Dx(t);
    I(p, n) <+ V(p, n) / r * (1 + x);
    Hx(t) <+ -V(p, n) * V(p, n) / r;
  end
endmodule
"""
        mod = parse_verilog_a(src, filename='t2.va')
        self.assertEqual(mod.ports[2].discipline, 'mytherm')
        assigns = [c for c in mod.analog_block.children if c.kind == NodeKind.ASSIGN]
        self.assertEqual(assigns[0].expr.replace(' ', ''), 'V(t)')
        cs = _contribs(mod.analog_block)
        self.assertEqual(cs[-1].contrib_kind, ContribKind.I)
        self.assertEqual(tuple(cs[-1].branch), ('t',))

    def test_digital_discipline_is_not_a_node_declaration(self):
        src = """
discipline logic
  domain discrete;
enddiscipline
module t3(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  analog I(p, n) <+ V(p, n) / r;
endmodule
"""
        mod = parse_verilog_a(src, filename='t3.va')
        self.assertEqual(mod.internal_nodes, [])
        self.assertEqual(len(_contribs(mod.analog_block)), 1)

    def test_unknown_access_function_is_an_error(self):
        src = VA_T1.replace("Pwr(t) <+", "Heat(t) <+")
        with self.assertRaises(ParseError) as cm:
            parse_verilog_a(src, filename='t1.va')
        msg = str(cm.exception)
        self.assertIn("'Heat'", msg)
        self.assertIn('t1.va:11', msg)

    def test_access_function_of_an_undeclared_discipline_is_an_error(self):
        # Pwr applies to thermal nodes: with t electrical it is no access
        # function of this module ...
        src = VA_T1.replace("thermal t;", "electrical t;")
        with self.assertRaises(ParseError) as cm:
            parse_verilog_a(src, filename='t1.va')
        self.assertIn("'Pwr'", str(cm.exception))
        self.assertIn("declared in this module", str(cm.exception))
        # ... and a module-defined function sharing its name with some other
        # discipline's access (kinematic's F) is left alone
        src2 = VA_T1.replace("    Pwr(t) <+ -V(p, n) * V(p, n) / r;\n", "") \
                    .replace("thermal t;", "electrical t;") \
                    .replace("Temp(t)", "F(V(t))")
        mod = parse_verilog_a(src2, filename='t1.va')
        self.assertIn('F ( V ( t ) )', _contribs(mod.analog_block)[0].expr)


def _xyce_env(work):
    env = dict(os.environ)
    env["PYMS_DIR"] = _PYMS_DIR
    env["PYMS_CACHE"] = os.path.join(work, "hdl_cache")
    env["PYMS_VAE_CACHE"] = os.path.join(work, "vae_cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.pop("VAE_SO_PATH", None)
    env.pop("VAE_SO_DIR", None)
    env.pop("PYMS_DEFINES", None)
    return env


def _prn_values(prn_text):
    """{column: value} of the first data row of a Xyce .prn."""
    lines = [ln for ln in prn_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise AssertionError("no data row in .prn: %r" % prn_text[:200])
    header = lines[0].split()
    vals = lines[1].split()
    return {h.upper(): float(v) for h, v in zip(header, vals)}


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce at %s" % XYCE)
class XyceTests(unittest.TestCase):
    """The three decks end to end; one scratch dir, one cache pair."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="pyms_thermal_")
        with open(os.path.join(cls.work, "t1.va"), "w") as f:
            f.write(VA_T1)
        cls.results = {}
        for tag, deck in DECKS.items():
            name = "t1_%s.cir" % tag
            with open(os.path.join(cls.work, name), "w") as f:
                f.write(deck)
            r = subprocess.run([XYCE, name], capture_output=True, text=True,
                               timeout=600, cwd=cls.work, env=_xyce_env(cls.work))
            prn = os.path.join(cls.work, name + ".prn")
            log = r.stdout + r.stderr
            if r.returncode != 0 or not os.path.exists(prn):
                raise AssertionError("Xyce failed on %s (rc %d):\n%s" % (name, r.returncode, log[-3000:]))
            with open(prn) as f:
                cls.results[tag] = _prn_values(f.read())

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("HDLTEST_KEEP"):
            shutil.rmtree(cls.work, ignore_errors=True)

    def _close(self, got, want, what):
        self.assertTrue(abs(got - want) <= RTOL * abs(want),
                        "%s = %.12g, expected %.12g (rel %.3g)"
                        % (what, got, want, abs(got - want) / abs(want)))

    def test_own_node(self):
        # the Pwr(t) contribution heats the node: out and Temp(t) analytic
        self._close(self.results["own"]["V(OUT)"], OUT_OWN, "own: v(out)")
        self._close(self.results["own"]["V(TH)"], TH_OWN, "own: v(th)")

    def test_tied_to_ground(self):
        self._close(self.results["gnd"]["V(OUT)"], OUT_GND, "gnd: v(out)")

    def test_tied_to_driven_node(self):
        self._close(self.results["drv"]["V(OUT)"], OUT_DRV, "drv: v(out)")


# BSIM-CMG's self-heating switch (bsimcmg_body.include l.3005-3024): with
# self-heating off the thermal node is held by a potential contribution,
# ``Temp(t) <+ 0.0``.  With t grounded (or a port left open) that branch has
# both ends on one circuit node: its current unknown had no equation and no
# KCL entry and Xyce's matrix was singular ("Numerically singular matrix",
# no operating point: BSIM-CMG 111, BSIM-SOI, BSIM6 in the corpus deck).
VA_SH = """`include "disciplines.vams"
module t4(p, n, t);
  inout p, n, t;
  electrical p, n;
  thermal t;
  parameter real r = 1k;
  parameter real rth = 0;
  analog begin
    I(p, n) <+ V(p, n) / r * (1 + Temp(t));
    if (rth > 0) begin
      Pwr(t) <+ Temp(t) / rth;
      Pwr(t) <+ -V(p, n) * V(p, n) / r;
    end else
      Temp(t) <+ 0.0;
  end
endmodule
"""

DECKS_SH = {
    "gnd": ".hdl \"t4.va\"\nV1 in 0 1\n.model m1 t4\nYT4 X1 in out 0 m1\nR2 out 0 1k\n"
           ".op\n.print dc v(out)\n.end\n",
    "own": ".hdl \"t4.va\"\nV1 in 0 1\n.model m1 t4\nYT4 X1 in out th m1\nR2 out 0 1k\n"
           "Rth th 0 1k\n.op\n.print dc v(out) v(th)\n.end\n",
    "on":  ".hdl \"t4.va\"\nV1 in 0 1\n.model m1 t4 rth=1k\nYT4 X1 in out 0 m1\nR2 out 0 1k\n"
           ".op\n.print dc v(out)\n.end\n",
}


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce at %s" % XYCE)
class XyceShortedBranchTests(unittest.TestCase):
    """A potential contribution whose two ends are one circuit node."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="pyms_thermal_sh_")
        with open(os.path.join(cls.work, "t4.va"), "w") as f:
            f.write(VA_SH)
        cls.results, cls.logs = {}, {}
        for tag, deck in DECKS_SH.items():
            name = "t4_%s.cir" % tag
            with open(os.path.join(cls.work, name), "w") as f:
                f.write("* " + tag + "\n" + deck)
            r = subprocess.run([XYCE, name], capture_output=True, text=True,
                               timeout=600, cwd=cls.work, env=_xyce_env(cls.work))
            prn = os.path.join(cls.work, name + ".prn")
            cls.logs[tag] = (r.returncode, (r.stdout + r.stderr)[-3000:])
            if r.returncode == 0 and os.path.exists(prn):
                with open(prn) as f:
                    cls.results[tag] = _prn_values(f.read())

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("HDLTEST_KEEP"):
            shutil.rmtree(cls.work, ignore_errors=True)

    def _out(self, tag):
        self.assertIn(tag, self.results, "Xyce failed on t4_%s (rc %d):\n%s" % ((tag,) + self.logs[tag]))
        return self.results[tag]

    def test_held_node_tied_to_ground(self):
        out = self._out("gnd")["V(OUT)"]
        self.assertTrue(abs(out - 0.5) <= RTOL * 0.5, "v(out) = %.12g, expected 0.5" % out)

    def test_held_node_on_its_own_node(self):
        res = self._out("own")
        self.assertTrue(abs(res["V(OUT)"] - 0.5) <= RTOL * 0.5, res)
        self.assertTrue(abs(res["V(TH)"]) <= 1e-9, res)

    def test_heating_branch_with_the_port_grounded(self):
        # rth = 1k, t grounded: Temp(t) = 0, the Pwr branch is shorted by the netlist
        out = self._out("on")["V(OUT)"]
        self.assertTrue(abs(out - 0.5) <= RTOL * 0.5, "v(out) = %.12g, expected 0.5" % out)


if __name__ == "__main__":
    unittest.main()
