"""`define NAME (body): a blank between the macro name and the parenthesis
makes the macro OBJECT-LIKE and the parenthesis opens its body.  IEEE
1364-2005 19.3.1 (the preprocessor Verilog-AMS 2.4 10.3 inherits): "the
left parenthesis shall follow the text macro name immediately, with no
space in between" -- only then is the parenthesis a formal-argument list.

THE BUG (PyMS defect P4, vamos spectre guide trials 2026-10-19..24):
preprocess._DEFINE_RE allowed blanks before the parenthesis, so
``\\`define R_VAL (1000.0)`` was read as a macro with the formal argument
"1000.0" and no body.  Its use ``\\`R_VAL`` (no argument list follows) was
left in the text, the emitter wrote ``R_VAL`` into the math library, g++
failed ("'R_VAL' was not declared in this scope") and Xyce printed
V(out) = 0 with exit status 0 -- no error anywhere (VACASK: 0.5).  Fifty
of the 111 files of the two Verilog-A corpora (/usr/local/share/xyce/verilog-a,
utils/ADMS/examples) write defines this way, in the file or its includes
(measured 2026-10-24): VBIC 1.3's and BSIM-SOI's ``\\`define IPM
(*units="" ...*) parameter real m=1.0 from(0.0:inf);`` (a declaration
line that silently vanished, the module lost its parameter m), fbh_hbt's
``\\`define TWO_Q (2 * 1.6021918e-19)``, BSIM6's and BSIM-SOI's ``\\`define
REFTEMP (300.15)``, DIODE_CMC's ``\\`define QN_INJ_SCALING (NQS)``.

What this test pins:
  * a blank before the parenthesis: object-like, body ``(...)``
  * no blank: function-like, as before (``\\`SCALE(2.0)`` substitutes)
  * the BSIM IPM declaration line reaches the parser as a parameter
  * a chain of two such macros
  * a formal argument that is not an identifier is a PreprocessError
    naming file, line and macro (not a silently mangled macro)
  * a macro with arguments used without an argument list is a
    PreprocessError naming the use and the definition (it was left in the
    text); arguments after a body that ends in such a macro still work
  * end to end through Xyce (skipped without the binary): the divider
    with ``\\`define R_VAL (1000.0)`` and the IPM module both give
    V(out) = 0.5 (VACASK 5.000000000000000e-01).

The Xyce runs use THIS copy of PyMS (PYMS_DIR = the tree this file sits
in) and private PYMS_CACHE / PYMS_VAE_CACHE under a scratch directory.

Run:
    python3 vae/tests/test_preprocess_define.py
$XYCE names the binary (default /usr/local/src/xyce-build/src/Xyce);
HDLTEST_KEEP=1 keeps the scratch directory.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# Make `vae.*` importable when run as a plain script, from THIS tree.
_VAE_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VAE_PKG_DIR not in sys.path:
    sys.path.insert(0, _VAE_PKG_DIR)

from vae.preprocess import preprocess_source, PreprocessError  # noqa: E402
from vae.parser import parse_verilog_a  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)

V_EXPECTED = 0.5                # 1 V into the 1k module over the 1k load
RTOL = 1e-6

# The reference divider with the macro in the analog expression.
VA_BLANK_DEFINE = """`include "disciplines.vams"
`include "constants.vams"
`define R_VAL (1000.0)
module a1(p, n);
  inout p, n;
  electrical p, n;
  analog I(p, n) <+ V(p, n) / `R_VAL;
endmodule
"""

# BSIM-CMG's multiplicity-factor declaration macro, verbatim form.
VA_IPM = """`include "disciplines.vams"
`include "constants.vams"
`define IPM (*units="" , type="instance",  desc="multiplicity factor"*) parameter real m=1.0 from(0.0:inf) ;
module a1(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  `IPM
  analog I(p, n) <+ m * V(p, n) / r;
endmodule
"""

DECK = """* P4: `define with a blank before the parenthesis
.hdl "a1.va"
v1 in 0 1
.model m1 a1
ya1 x1 in out m1
r2 out 0 1k
.op
.print dc format=noindex v(out)
.end
"""


class TestDefineBlankBeforeParen(unittest.TestCase):
    """The directive alone, through preprocess_source."""

    def test_blank_is_object_like(self):
        out = preprocess_source("`define R_VAL (1000.0)\nx = `R_VAL;\n", "t.va")
        self.assertIn("x = (1000.0);", out)
        self.assertNotIn("`", out)

    def test_adjacent_paren_is_function_like(self):
        out = preprocess_source("`define SCALE(x) ((x) * 1000.0)\ny = `SCALE(2.0);\n", "t.va")
        self.assertIn("y = ((2.0) * 1000.0);", out)

    def test_no_arguments_formal_list(self):
        out = preprocess_source("`define ONE() 1.0\nz = `ONE();\n", "t.va")
        self.assertIn("z = 1.0;", out)

    def test_chain_of_blank_defines(self):
        out = preprocess_source(
            "`define HALF_R (500.0)\n`define FULL_R (2 * `HALF_R)\nx = `FULL_R;\n", "t.va")
        self.assertIn("x = (2 * (500.0));", out)

    def test_ipm_declaration_line_reaches_the_parser(self):
        out = preprocess_source(VA_IPM, "a1.va")
        self.assertIn("parameter real m=1.0 from(0.0:inf)", out)
        mod = parse_verilog_a(out, filename="a1.va")
        self.assertIn("m", [p.name for p in mod.params],
                      "the IPM macro's parameter m is missing from the module")

    def test_function_like_without_arguments_is_an_error(self):
        # Left in the text it became an unknown name in the generated C++
        # (or vanished); B's round-1 check exempted it for this fix.
        with self.assertRaises(PreprocessError) as cm:
            preprocess_source("`define SCALE(x) ((x) * 1000.0)\ny = `SCALE;\n", "t.va")
        msg = str(cm.exception)
        self.assertIn("t.va:2", msg)
        self.assertIn("`SCALE", msg)
        self.assertIn("t.va:1", msg)          # where it was defined

    def test_arguments_after_a_body_that_ends_in_a_macro(self):
        out = preprocess_source("`define G(a) a*2\n`define F `G\nx = `F(3);\n", "t.va")
        self.assertIn("x = 3*2;", out)

    def test_formal_argument_must_be_an_identifier(self):
        with self.assertRaises(PreprocessError) as cm:
            preprocess_source("`define X(2) 2\n", "t.va")
        msg = str(cm.exception)
        self.assertIn("t.va:1", msg)
        self.assertIn("X", msg)
        self.assertIn("'2'", msg)


def _prn_value(prn_text, col, row=0):
    lines = [ln for ln in prn_text.splitlines() if ln.strip()]
    if not lines:
        raise AssertionError("empty .prn")
    header = lines[0].split()
    if col not in header:
        raise AssertionError("column %s not in .prn header %r" % (col, header))
    return float(lines[1 + row].split()[header.index(col)])


def _run_deck(work, va_text):
    """Run DECK with the module text in `work`; (log, V(out) or None)."""
    with open(os.path.join(work, "a1.va"), "w") as f:
        f.write(va_text)
    with open(os.path.join(work, "a1.cir"), "w") as f:
        f.write(DECK)
    env = dict(os.environ)
    env["PYMS_DIR"] = _VAE_PKG_DIR
    env["PYMS_CACHE"] = os.path.join(work, "hdl_cache")
    env["PYMS_VAE_CACHE"] = os.path.join(work, "vae_cache")
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.pop("VAE_SO_PATH", None)
    env.pop("VAE_SO_DIR", None)
    r = subprocess.run([XYCE, "a1.cir"], capture_output=True, text=True,
                       timeout=600, cwd=work, env=env)
    log = r.stdout + r.stderr
    prn = os.path.join(work, "a1.cir.prn")
    if not os.path.exists(prn):
        return log, None
    with open(prn) as f:
        return log, _prn_value(f.read(), "V(OUT)")


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce at %s" % XYCE)
class TestDefineBlankThroughXyce(unittest.TestCase):
    """End to end: Xyce .hdl -> PyMS of this tree -> device -> V(out)."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="pymsdefine_")

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("HDLTEST_KEEP"):
            sys.stderr.write("kept %s\n" % cls.work)
        else:
            shutil.rmtree(cls.work, ignore_errors=True)

    def _check(self, tag, va_text):
        work = os.path.join(self.work, tag)
        os.makedirs(work)
        log, v = _run_deck(work, va_text)
        self.assertIsNotNone(v, "Xyce produced no .prn; log:\n" + log[-3000:])
        self.assertNotIn("was not declared in this scope", log,
                         "the macro reached g++ unexpanded")
        self.assertTrue(abs(v - V_EXPECTED) <= RTOL * V_EXPECTED,
                        "V(out) = %.12g, expected %.12g" % (v, V_EXPECTED))

    def test_blank_define_in_expression(self):
        self._check("expr", VA_BLANK_DEFINE)

    def test_ipm_declaration_macro(self):
        self._check("ipm", VA_IPM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
