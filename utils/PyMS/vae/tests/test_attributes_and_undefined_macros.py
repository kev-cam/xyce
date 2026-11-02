"""Attribute instances before analog statements, and undefined macros
(PYMS_BUGS.md P2 and P3, 2026-10-24).

THE BUGS, on the reference divider (V1 = 1 V into the module over R2 = 1 k,
out = 0.5; VACASK 0.5):

  P2  analog begin (*info="x"*) I(p, n) <+ V(p, n) / r; end
      out = 0, exit 0, no message: Parser._parse_statement had no case for
      ``(*``; _parse_simple_statement's fallback read the attribute AND the
      statement after it, up to the ``;``, as one expression statement and
      the emitter dropped it.  The same with ``analog (*...*) I(...) <+ ...``,
      ``else (*...*) ...``, before an assignment, an ``if``, a nested block,
      and with an `attr macro that the source defines as (*txt*) (TRIAL2EJ-1).

  P3  an undefined macro was left in the text by preprocess._expand_once and
      the parser mis-read it by where it stood: dropped as an annotation
      (parameter end, module header, a line of its own, port lists), merged
      with the next statement and dropped with it (out = 0, exit 0: analog
      block before a statement, and through a define chain
      `define ANN `attr(info="x")), a g++ failure (after ``electrical``: a
      node named `attr), a Python traceback (before a parameter's name or its
      from range), a parameter default of 0 (parameter real r = `RVAL;).
      The LRM makes an undefined macro an error; openvaf-r refuses it
      ("macro '`attr' has not been declared").

THE FIX: preprocess._Preprocessor._check_macro_uses checks every `name of an
emitted source line (outside strings and comments, with the macros defined
at that point, and through the bodies of the macros the line uses); an
undefined one is a PreprocessError naming the file, the source line and the
macro, which xyce_device_gen.py prints as one line (no traceback).  There is
no annotation exception: the corpus survey of 2026-10-24 found no real-world
file that needs one (bsimcmg_main.va is rerouted to its wrapper bsimcmg.va,
which defines `attr; bsimcmg_nqsmod3.va of BSIM-CMG 108 built with every
type="instance" attribute lost, 0 instance parameters instead of 25).
HICUM L0 and L2 `include "constants.h", the ADMS name of the standard
header, which no tree on the box has: their `P_K, `P_Q, `M_PI were undefined
(left in the analog block, as an undefined macro in a contribution is a
g++ error of the math library), so a missing "constants.h" now
resolves to PyMS's constants.vams.  Parser._parse_statement consumes the attribute instances before a statement
and drops them; an attribute with no statement after it is a ParseError.
_collect_expr_until drops attribute instances at the end of a statement,
before its ``;`` (HICUM L0's ``I(br_sci) <+ ijsc (*spectre:gmin="add"*);``,
which used to become part of the contribution's text) and refuses one
anywhere else inside an expression.

Two layers of tests: the preprocessor and parser alone (fast, no Xyce), and
the forms end to end through Xyce (private PYMS_CACHE / PYMS_VAE_CACHE,
PYMS_DIR = the PyMS copy this file sits in; skipped without a Xyce binary).

Run:
    cd utils/PyMS && python3 vae/tests/test_attributes_and_undefined_macros.py
$XYCE names the binary (default /usr/local/src/xyce-build/src/Xyce);
ATTRTEST_KEEP=1 keeps the scratch directory.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

_VAE_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VAE_PKG_DIR not in sys.path:
    sys.path.insert(0, _VAE_PKG_DIR)

from vae.parser import parse_verilog_a, ParseError, NodeKind  # noqa: E402
from vae.preprocess import preprocess_source, preprocess_file, PreprocessError  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
RTOL = 1e-6

HEAD = '`include "disciplines.vams"\n`include "constants.vams"\n'


def module(decl="parameter real r = 1k;", header="module a1(p, n);", body=None, extra="", pre="",
           ports="  inout p, n;\n  electrical p, n;\n"):
    """The reference module a1 (r = 1k between p and n) with a piece changed."""
    body = body or "analog I(p, n) <+ V(p, n) / r;"
    return HEAD + pre + header + "\n" + ports + "  " + decl + "\n" + extra + "  " + body + "\nendmodule\n"


def parse(va):
    return parse_verilog_a(preprocess_source(va, filename="a1.va"), filename="a1.va")


def contributions(node):
    """The contribution statements of an analog block, in order."""
    if node is None:
        return []
    found = [node] if node.kind == NodeKind.CONTRIB else []
    for child in list(node.children or []) + ([node.else_body] if node.else_body else []):
        found += contributions(child)
    return found


def line_of(va, needle):
    """The 1-based line of the first occurrence of needle in va."""
    return va[:va.index(needle)].count("\n") + 1


class TestAttributeBeforeStatement(unittest.TestCase):
    """P2: an attribute instance before an analog statement belongs to it."""

    FORMS = {
        "before a contribution in a block":
            'analog begin\n    (*info="x"*) I(p, n) <+ V(p, n) / r;\n  end',
        "after the analog keyword, before begin":
            'analog (*info="x"*) begin\n    I(p, n) <+ V(p, n) / r;\n  end',
        "after the analog keyword, a single contribution":
            'analog (*info="x"*) I(p, n) <+ V(p, n) / r;',
        "two attribute instances":
            'analog begin\n    (*info="x"*) (*type="y"*) I(p, n) <+ V(p, n) / r;\n  end',
        "in a nested block":
            'analog begin\n    begin\n      (*info="x"*) I(p, n) <+ V(p, n) / r;\n    end\n  end',
        "before an if":
            'analog begin\n    (*info="x"*) if (r > 0) I(p, n) <+ V(p, n) / r;\n  end',
        "after else":
            'analog begin\n    if (r < 0) I(p, n) <+ 0.0; else (*info="x"*) I(p, n) <+ V(p, n) / r;\n  end',
        "on its own line before the statement":
            'analog begin\n    (*info="x"*)\n    I(p, n) <+ V(p, n) / r;\n  end',
    }

    def test_the_statement_after_the_attribute_is_kept(self):
        for what, body in self.FORMS.items():
            with self.subTest(what):
                mod = parse(module(body=body))
                exprs = [c.expr.replace(" ", "") for c in contributions(mod.analog_block)]
                self.assertIn("V(p,n)/r", exprs, "the contribution after the attribute was lost")

    def test_an_assignment_after_the_attribute_is_kept(self):
        mod = parse(module(extra="  real v;\n",
                           body='analog begin\n    (*info="x"*) v = V(p, n);\n    I(p, n) <+ v / r;\n  end'))
        kinds = [c.kind for c in mod.analog_block.children]
        self.assertEqual(kinds, [NodeKind.ASSIGN, NodeKind.CONTRIB])

    def test_a_defined_attribute_macro_before_a_statement(self):
        # TRIAL2EJ-1 vattr3: the macro of the CMC trees, expanding to (* ... *)
        mod = parse(module(pre="`define attr(txt) (*txt*)\n",
                           body='analog begin\n    `attr(info="before a statement")\n'
                                '    I(p, n) <+ V(p, n) / r;\n  end'))
        self.assertEqual(len(contributions(mod.analog_block)), 1)

    def test_an_attribute_on_a_null_statement(self):
        mod = parse(module(body='analog begin\n    (*info="x"*);\n    I(p, n) <+ V(p, n) / r;\n  end'))
        self.assertEqual(len(contributions(mod.analog_block)), 1)

    def test_an_attribute_at_the_end_of_a_contribution(self):
        # HICUM L0: I(br_sci) <+ ijsc (*spectre:gmin="add" ...*);  The attribute became part of
        # the contribution's text: out = 0 on pristine, a math-library g++ error on PyMS.base.
        mod = parse(module(body='analog I(p, n) <+ V(p, n) / r (*spectre:gmin="add" info="x"*);'))
        self.assertEqual([c.expr.replace(" ", "") for c in contributions(mod.analog_block)], ["V(p,n)/r"])

    def test_an_attribute_inside_an_expression_is_an_error(self):
        # the source line (the preprocessor's line map), not the line of the
        # preprocessed text with the standard headers inlined
        va = module(body='analog I(p, n) <+ V(p, n) / (*info="x"*) r;')
        with self.assertRaisesRegex(ParseError, r"^a1\.va:%d: an attribute instance \(\* \.\.\. \*\) inside "
                                                r"an expression is not supported" % line_of(va, "(*info")):
            parse(va)

    def test_an_attribute_with_no_statement_is_an_error(self):
        va = module(body='analog begin\n    I(p, n) <+ V(p, n) / r;\n    (*info="x"*)\n  end')
        with self.assertRaisesRegex(ParseError, r"^a1\.va:%d: an attribute instance \(\* \.\.\. \*\) must be "
                                                r"followed by a statement, found 'end'" % (line_of(va, "(*info") + 1)):
            parse(va)


class TestUndefinedMacro(unittest.TestCase):
    """P3: an undefined macro is an error naming file, line and macro, wherever it stands."""

    ANN = '`attr(info="x")'
    POSITIONS = {
        "at a parameter's end": dict(decl='parameter real r = 1k `attr(info="x");'),
        "after the module header": dict(header='module a1(p, n) `attr(info="x");'),
        "as a line of its own": dict(extra='  `attr(info="x")\n'),
        "as a line of its own with a semicolon": dict(extra='  `attr(info="x");\n'),
        "over two lines": dict(extra='  `attr(info="x",\n        type="instance")\n'),
        "in the analog block before a statement":
            dict(body='analog begin\n    `attr(info="x")\n    I(p, n) <+ V(p, n) / r;\n  end'),
        "in the analog block with a semicolon":
            dict(body='analog begin\n    `attr(info="x");\n    I(p, n) <+ V(p, n) / r;\n  end'),
        "in a contribution": dict(body='analog I(p, n) <+ V(p, n) / r `attr(info="x");'),
        "after a port direction": dict(ports='  inout p, n `attr(info="x");\n  electrical p, n;\n'),
        "after electrical": dict(ports='  inout p, n;\n  electrical p, n `attr(info="x");\n'),
        "after a from range": dict(decl='parameter real r = 1k from (0:inf) `attr(info="x");'),
        "before the parameter's name": dict(decl='parameter real `attr(info="x") r = 1k;'),
        "before a from range": dict(decl='parameter real r = 1k `attr(info="x") from (0:inf);'),
        "in a variable declaration": dict(extra='  real v `attr(info="x");\n'),
        "without a parenthesis": dict(decl='parameter real r = 1k `attr;'),
        "with a blank before the parenthesis": dict(decl='parameter real r = 1k `attr (info="x");'),
        "as a parameter default": dict(decl='parameter real r = `attr;'),
    }

    def check_error(self, va, macro="attr", needle=None, filename="a1.va"):
        line = line_of(va, needle or "`" + macro)
        with self.assertRaises(PreprocessError) as cm:
            preprocess_source(va, filename=filename)
        msg = str(cm.exception)
        self.assertRegex(msg, r"^%s:%d: `%s is not defined" % (re.escape(filename), line, macro))
        return msg

    def test_every_position_is_an_error(self):
        for what, kw in self.POSITIONS.items():
            with self.subTest(what):
                self.check_error(module(**kw))

    def test_through_a_define_chain(self):
        va = module(pre='`define ANN `attr(info="x")\n',
                    body='analog begin\n    `ANN\n    I(p, n) <+ V(p, n) / r;\n  end')
        msg = self.check_error(va, needle="    `ANN")
        self.assertIn("the body of `ANN (defined at a1.va:3)", msg)
        va = module(pre='`define ANN2(t) `attr(info=t)\n', decl='parameter real r = 1k `ANN2("resistance");')
        self.assertIn("the body of `ANN2", self.check_error(va, needle="1k `ANN2"))

    def test_undef_then_use_and_use_before_define(self):
        self.check_error(module(pre="`define attr(txt) (*txt*)\n`undef attr\n",
                                decl='parameter real r = 1k `attr(info="x");'), needle="1k `attr")
        self.check_error(module(decl='parameter real r = 1k `attr(info="x");',
                                extra="`define attr(txt) (*txt*)\n"))

    def test_the_line_survives_continuations(self):
        va = module(pre="`define TWO(a) \\\n   (2 * \\\n    a)\n", decl='parameter real r = `TWO(500.0) `attr(info="x");')
        self.check_error(va)

    def test_an_undefined_macro_in_an_included_file(self):
        work = tempfile.mkdtemp(prefix="attrtest_inc_")
        try:
            with open(os.path.join(work, "inc.vams"), "w") as f:
                f.write('// local include\n\nparameter real r = 1k `attr(info="x");\n')
            with open(os.path.join(work, "a1.va"), "w") as f:
                f.write(module(decl='`include "inc.vams"'))
            with self.assertRaisesRegex(PreprocessError, r"inc\.vams:3: `attr is not defined"):
                preprocess_file(os.path.join(work, "a1.va"))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_the_adms_name_of_the_constants_header(self):
        # HICUM L0/L2 `include "constants.h" (admsXml supplies it): without a
        # file of that name PyMS's constants.vams stands in, so `P_K is defined
        # (with the Verilog-AMS 2.4 Annex value, test_std_headers.py).
        work = tempfile.mkdtemp(prefix="attrtest_adms_")
        try:
            path = os.path.join(work, "a1.va")
            with open(path, "w") as f:
                f.write('`include "constants.h"\n' + module(decl="parameter real r = 1k * `P_K / `P_K;"))
            self.assertIn("1.3806503e-23", preprocess_file(path))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_what_is_not_a_use(self):
        controls = {
            "a line comment": dict(extra='  // `attr(info="x") in a comment\n'),
            "a block comment": dict(extra='  /* `attr(info="x") in a block\n     comment `RVAL */\n'),
            "a string": dict(body='analog begin\n    $strobe("`attr(info=x)");\n    I(p, n) <+ V(p, n) / r;\n  end'),
            "an ifdef that is not taken": dict(extra='`ifdef NEVER_DEFINED\n  `attr(info="x")\n`endif\n'),
            "a defined macro": dict(pre="`define attr(txt) (*txt*)\n", decl='parameter real r = 1k `attr(info="x");'),
            "a directive the preprocessor passes on": dict(pre="`timescale 1ns / 1ps\n`default_discipline electrical\n"),
            "a constants.vams macro": dict(decl="parameter real r = 1000.0 * `M_PI / `M_PI;"),
        }
        for what, kw in controls.items():
            with self.subTest(what):
                mod = parse(module(**kw))
                self.assertEqual(len(contributions(mod.analog_block)), 1)


# ---------------------------------------------------------------------------
# End to end through Xyce
# ---------------------------------------------------------------------------

DECK = """* attributes and undefined macros (PYMS_BUGS P2, P3): the reference divider
.hdl "a1.va"
V1 in 0 DC 1
.model m1 a1
Ya1 X1 in out m1
R2 out 0 1k
.op
.print dc v(out)
.end
"""


def _xyce_env(work):
    env = dict(os.environ)
    env["PYMS_DIR"] = _VAE_PKG_DIR                 # the PyMS copy this test sits in
    env["PYMS_CACHE"] = os.path.join(work, "hdl_cache")
    env["PYMS_VAE_CACHE"] = os.path.join(work, "vae_cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.pop("VAE_SO_PATH", None)
    env.pop("VAE_SO_DIR", None)
    return env


def _prn_vout(prn_text):
    hdr = None
    for ln in prn_text.splitlines():
        cols = ln.split()
        if not cols:
            continue
        if hdr is None:
            if cols[0].lower() == "index":
                hdr = [c.upper() for c in cols]
            continue
        if cols[0].lower().startswith("end"):
            break
        return float(cols[hdr.index("V(OUT)")])
    return None


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary at %s" % XYCE)
class TestXyce(unittest.TestCase):
    """The forms through Xyce: out = 0.5 on the reference divider, or an error."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="attrtest_")
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("ATTRTEST_KEEP"):
            print("kept", cls.work)
        else:
            shutil.rmtree(cls.work, ignore_errors=True)

    def run_deck(self, va):
        type(self).n += 1
        d = os.path.join(self.work, "c%02d" % self.n)
        os.makedirs(d)
        with open(os.path.join(d, "a1.va"), "w") as f:
            f.write(va)
        with open(os.path.join(d, "a1.cir"), "w") as f:
            f.write(DECK)
        r = subprocess.run([XYCE, "a1.cir"], capture_output=True, text=True,
                           timeout=600, cwd=d, env=_xyce_env(d))
        log = r.stdout + r.stderr
        prn = os.path.join(d, "a1.cir.prn")
        out = None
        if os.path.exists(prn):
            with open(prn) as f:
                out = _prn_vout(f.read())
        return r.returncode, out, log

    def check(self, va, expected):
        rc, out, log = self.run_deck(va)
        self.assertEqual(rc, 0, "Xyce exit %s:\n%s" % (rc, log[-3000:]))
        self.assertIsNotNone(out, "no V(OUT) in the .prn:\n%s" % log[-3000:])
        self.assertTrue(abs(out - expected) <= RTOL * max(abs(expected), 1e-3),
                        "out = %.12g, expected %.12g\n%s" % (out, expected, log[-3000:]))

    def test_attribute_before_a_statement(self):
        self.check(module(body='analog begin\n    (*info="x"*) I(p, n) <+ V(p, n) / r;\n  end'), 0.5)
        self.check(module(body='analog (*info="x"*) I(p, n) <+ V(p, n) / r;'), 0.5)
        self.check(module(body='analog I(p, n) <+ V(p, n) / r (*info="x"*);'), 0.5)

    def test_defined_attribute_macro_before_a_statement(self):
        self.check(module(pre="`define attr(txt) (*txt*)\n",
                          body='analog begin\n    `attr(info="before a statement")\n'
                               '    I(p, n) <+ V(p, n) / r;\n  end'), 0.5)

    def test_undefined_macro_is_a_compile_error(self):
        va = module(body='analog begin\n    `attr(info="x")\n    I(p, n) <+ V(p, n) / r;\n  end')
        rc, out, log = self.run_deck(va)
        self.assertNotEqual(rc, 0, "Xyce ran a module with an undefined macro:\n%s" % log[-2000:])
        flat = " ".join(log.split())                 # Xyce wraps its messages
        self.assertIn("PyMS compilation failed", flat)
        self.assertRegex(flat, r"a1\.va:%d: `attr is not defined" % line_of(va, "`attr"))
        self.assertNotIn("Traceback", flat)


if __name__ == "__main__":
    unittest.main(verbosity=2)
