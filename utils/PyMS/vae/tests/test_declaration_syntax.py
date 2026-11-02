#!/usr/bin/env python3
"""The declaration statements of a module body are read by their grammar:
``real``/``integer`` (names, ``[lo:hi]`` dimensions, ``= expr``
initializers), the port directions, the discipline declarations
(``electrical``, ``thermal``, ...) and ``branch``; anything else is a
ParseError naming file and line.  A parameter may not share its name with a
port.

THE BUG (2026-10-25, found by PYMS1-F, fixed by PYMS1-I; PYMS_BUGS.md P11 and
the F/A hand-overs): the declaration loops took every identifier up to the next
';' as a name and skipped every other token.  A missing ';' after a variable
declaration,

    real v
    analog begin
      v = V(p, n);
      I(p, n) <+ v / r;
    end

made ``v analog begin v`` and the rest variable names: no analog block, a
device that conducts nothing, Xyce out = 0 with exit status 0 and no message
(VACASK: a syntax error).  The same loops dropped a variable's initializer
(``real g = 1m;`` read as g = 0, and its operands as more variables: IHP
sg13g2's resistor.va and capacitor.va keep constants this way,
``real CONSTvt0 = 1.38064852e-23*(27+273.15)/1.6021766208e-19;``), and a port
and a parameter of one name were accepted (openvaf-r: an error).

THE FIX (parser.py): Parser._parse_decl_names reads ``name {, name} ;`` with
the optional parts each declaration allows and raises ParseError (file:line,
what it expected, what it got, "a missing ';'?") otherwise; used by
_parse_variables, _parse_port_dir, _parse_node_decl and the branch
declaration.  Var.init keeps an initializer and
Parser._apply_variable_initializers makes it an assignment at the head of
the analog block (LRM: the variable declaration assignment).
Parser._check_port_param_names refuses a parameter named like a port.

The parse-level tests need no engine; the Xyce class runs the initializer
deck through THIS copy of PyMS (PYMS_DIR = the tree this file sits in,
private caches) and skips without a Xyce binary.

Run:  python3 vae/tests/test_declaration_syntax.py
"""
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

GEN = os.path.join(_PYMS_DIR, 'vae', 'xyce_device_gen.py')
XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
RTOL = 1e-6

HEAD = """`include "disciplines.vams"
module a1(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
"""

# Line 6 is the declaration under test in every one of these.
GOOD = HEAD + """  real v;
  analog begin
    v = V(p, n);
    I(p, n) <+ v / r;
  end
endmodule
"""

MISSING_SEMI = {
    # declaration (line 6)          -> the token PyMS must name
    "real v":                         ("real", "v", "analog"),
    "integer k":                      ("integer", "k", "analog"),
    "real v, w":                      ("real", "w", "analog"),
    "real v[0:1]":                    ("real", "v", "analog"),
    "real v = 1.0":                   ("real", "v", "analog"),
    "electrical m":                   ("electrical", "m", "analog"),
    "thermal t":                      ("thermal", "t", "analog"),
    "branch (p, n) b":                ("branch", "b", "analog"),
}


def _module(decl6, body="    v = V(p, n);\n    I(p, n) <+ v / r;\n"):
    return HEAD + "  %s\n  analog begin\n%s  end\nendmodule\n" % (decl6, body)


class MissingSemicolonTests(unittest.TestCase):

    def test_good_module_parses(self):
        mod = parse_verilog_a(GOOD, filename='a1.va')
        self.assertIsNotNone(mod.analog_block)
        self.assertEqual([v.name for v in mod.variables], ['v'])

    def test_missing_semicolon_is_an_error_at_its_line(self):
        for decl, (what, name, got) in MISSING_SEMI.items():
            with self.assertRaises(ParseError, msg=decl) as cm:
                parse_verilog_a(_module(decl), filename='a1.va')
            msg = str(cm.exception)
            self.assertTrue(msg.startswith("a1.va:6: "), (decl, msg))
            self.assertIn("'%s' declaration" % what, msg, decl)
            if name:
                self.assertIn("after '%s', got '%s'" % (name, got), msg, decl)
                self.assertIn("a missing ';'?", msg, decl)

    def test_a_keyword_is_no_name(self):
        with self.assertRaises(ParseError) as cm:
            parse_verilog_a(_module("real analog;"), filename='a1.va')
        self.assertIn("expected a name, got 'analog'", str(cm.exception))
        with self.assertRaises(ParseError) as cm:
            parse_verilog_a(_module("electrical [0:1] bus;"), filename='a1.va')
        self.assertIn("expected a name, got '['", str(cm.exception))

    def test_cli_reports_it_in_one_line(self):
        work = tempfile.mkdtemp(prefix="pyms_decl_")
        try:
            va = os.path.join(work, 'a1.va')
            with open(va, 'w') as f:
                f.write(_module("real v"))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            env.pop('PYMS_DEFINES', None)
            r = subprocess.run([sys.executable, GEN, va, '--output', work],
                               capture_output=True, text=True, timeout=120, env=env)
            out = r.stdout + r.stderr
            self.assertNotEqual(r.returncode, 0, out)
            self.assertIn("%s:6: in the 'real' declaration: expected ',' or ';' after 'v', "
                          "got 'analog'" % va, out)
            self.assertNotIn('Traceback', out)
            self.assertNotIn('Module: a1', out)
        finally:
            shutil.rmtree(work, ignore_errors=True)


class AcceptedFormsTests(unittest.TestCase):

    def test_lists_dimensions_attributes(self):
        src = _module('real v, w[0:3], x (* desc="x" *), (* desc="y" *) y;\n'
                      '  integer i, j[1:2];\n'
                      '  electrical m (* info="mid" *), q;\n'
                      '  branch (p, m) b1, b2;')
        mod = parse_verilog_a(src, filename='a1.va')
        self.assertEqual([(v.name, v.type) for v in mod.variables],
                         [('v', 'real'), ('w', 'real'), ('x', 'real'), ('y', 'real'),
                          ('i', 'integer'), ('j', 'integer')])
        self.assertEqual(mod.internal_nodes, ['m', 'q'])
        self.assertEqual(mod.branch_map, {'b1': 'p', 'b2': 'p'})
        self.assertIsNotNone(mod.analog_block)

    def test_initializer_is_an_assignment_at_the_head(self):
        src = _module("real v, g = 2 * (1m / 2), h;\n  integer k = 3;",
                      "    v = V(p, n);\n    I(p, n) <+ v * g * k / 3;\n")
        mod = parse_verilog_a(src, filename='a1.va')
        self.assertEqual([v.name for v in mod.variables], ['v', 'g', 'h', 'k'])
        head = mod.analog_block.children[:2]
        self.assertEqual([(c.kind, c.lhs) for c in head],
                         [(NodeKind.ASSIGN, 'g'), (NodeKind.ASSIGN, 'k')])
        self.assertEqual(head[0].expr.replace(' ', ''), '2*(0.001/2)')
        self.assertEqual(head[1].expr.strip(), '3')

    def test_initializer_with_a_single_statement_block(self):
        src = HEAD + "  real g = 1m;\n  analog I(p, n) <+ V(p, n) * g;\nendmodule\n"
        mod = parse_verilog_a(src, filename='a1.va')
        self.assertEqual(mod.analog_block.kind, NodeKind.BLOCK)
        self.assertEqual([c.kind for c in mod.analog_block.children],
                         [NodeKind.ASSIGN, NodeKind.CONTRIB])


class PortParameterNameTests(unittest.TestCase):

    def test_parameter_named_like_a_port_is_an_error(self):
        src = GOOD.replace("parameter real r = 1k;", "parameter real r = 1k;\n  parameter real n = 2;")
        with self.assertRaises(ParseError) as cm:
            parse_verilog_a(src, filename='a1.va')
        msg = str(cm.exception)
        self.assertIn("a1.va:6: parameter 'n' has the name of a port of module 'a1'", msg)

    def test_other_names_are_fine(self):
        src = GOOD.replace("parameter real r = 1k;", "parameter real r = 1k;\n  parameter real P = 2;")
        self.assertEqual([p.name for p in parse_verilog_a(src, filename='a1.va').params],
                         ['r', 'P'])


def _xyce_env(work):
    env = dict(os.environ)
    env["PYMS_DIR"] = _PYMS_DIR
    env["PYMS_CACHE"] = os.path.join(work, "hdl_cache")
    env["PYMS_VAE_CACHE"] = os.path.join(work, "vae_cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    for k in ("VAE_SO_PATH", "VAE_SO_DIR", "PYMS_DEFINES"):
        env.pop(k, None)
    return env


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce at %s" % XYCE)
class XyceInitializerTests(unittest.TestCase):
    """``real g = 1m;`` read by the device: a 1k conductance in the 1k
    divider, out = 0.5 (it was g = 0: out = 0, silent)."""

    def test_initializer_value_reaches_the_device(self):
        work = tempfile.mkdtemp(prefix="pyms_decl_xyce_")
        try:
            with open(os.path.join(work, "a1.va"), "w") as f:
                f.write(HEAD + "  real g = 1m;\n  analog I(p, n) <+ V(p, n) * g;\nendmodule\n")
            with open(os.path.join(work, "d.cir"), "w") as f:
                f.write("* initializer\n.hdl \"a1.va\"\nV1 in 0 1\n.model m1 a1\n"
                        "YA1 X1 in out m1\nR2 out 0 1k\n.op\n.print dc v(out)\n.end\n")
            r = subprocess.run([XYCE, "d.cir"], capture_output=True, text=True,
                               timeout=600, cwd=work, env=_xyce_env(work))
            prn = os.path.join(work, "d.cir.prn")
            self.assertTrue(r.returncode == 0 and os.path.exists(prn),
                            (r.stdout + r.stderr)[-3000:])
            lines = [ln for ln in open(prn).read().splitlines() if ln.strip()]
            hdr = [h.upper() for h in lines[0].split()]
            out = float(lines[1].split()[hdr.index('V(OUT)')])
            self.assertTrue(abs(out - 0.5) <= RTOL * 0.5, "v(out) = %.12g, expected 0.5" % out)
        finally:
            if not os.environ.get("HDLTEST_KEEP"):
                shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
