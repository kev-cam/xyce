"""Macros expand where they are used, and directives in /* */ comments are
comment text (IEEE 1364-2005 19.3.1 and 19, which Verilog-AMS 2.4 10.3
inherits: a text macro is substituted at its use with the definition in
force there; a redefinition applies from that point on; compiler directives
are not recognised inside comments).

THE BUGS (left open by the PyMS round-1 fix of P2/P3, 2026-10-24):
  1. preprocess._Preprocessor._process collected every ordinary line and
     expanded the whole file once at its end, with the macro table as it
     stood after the LAST directive, so
         `define RV 1000.0  ...  parameter real r = `RV;  ...  `define RV 2000.0
     built r = 2000.0 (the divider gave V(out) = 1/3 for the LRM's 0.5), and
     an `undef after a use expanded the use with whatever was defined later.
  2. _process matched directives on each line after removing only its
     ``//`` comment, so a ``\\`define``, ``\\`ifdef`` or ``\\`endif`` on a line
     inside a /* ... */ comment was obeyed: a commented-out redefinition
     took effect, a commented-out `ifdef swallowed the rest of the file, a
     commented-out `endif was "`endif without `ifdef".  A ``//`` inside a
     block comment (a URL, or ``*//*`` between two adjacent comments) cut
     the line and left the comment open, so the code after it vanished up
     to the next ``*/``: BSIM-SOI 4.5, 4.6.1 and 4.7 lost 15 statements of
     their charge model (Tcen, Coxeff, CoxWLcen ...) this way.

What this test pins, through preprocess_source and end to end through Xyce
(skipped without the binary): the uses before a redefinition or an `undef
keep the earlier body; a macro whose body names another macro sees the
other's definition at the use; a function-like macro invocation spanning
lines still expands; directives inside block comments (one line and
several lines) are ignored; strings with "/*" are not comments.

The Xyce runs use THIS copy of PyMS (PYMS_DIR = the tree this file sits
in) and private PYMS_CACHE / PYMS_VAE_CACHE under a scratch directory.

Run:
    python3 vae/tests/test_preprocess_point_of_use.py
$XYCE names the binary (default /usr/local/src/xyce-build/src/Xyce);
HDLTEST_KEEP=1 keeps the scratch directory.
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

from vae.preprocess import preprocess_source  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
RTOL = 1e-6


def _pp(text):
    return preprocess_source(text, "t.va")


class TestPointOfUse(unittest.TestCase):

    def test_redefinition_does_not_reach_back(self):
        out = _pp("`define RV 1000.0\nr = `RV;\n`define RV 2000.0\ns = `RV;\n")
        self.assertIn("r = 1000.0;", out)
        self.assertIn("s = 2000.0;", out)

    def test_undef_then_define(self):
        out = _pp("`define RV 1000.0\nr = `RV;\n`undef RV\n`define RV 3.0\ns = `RV;\n")
        self.assertIn("r = 1000.0;", out)
        self.assertIn("s = 3.0;", out)

    def test_body_sees_the_definition_at_the_use(self):
        out = _pp("`define A 1\n`define B (`A+1)\nx = `B;\n`define A 5\ny = `B;\n")
        self.assertIn("x = (1+1);", out)
        self.assertIn("y = (5+1);", out)

    def test_function_like_redefinition(self):
        out = _pp("`define F(a) (a*2)\nx = `F(1);\n`define F(a) (a*3)\ny = `F(1);\n")
        self.assertIn("x = (1*2);", out)
        self.assertIn("y = (1*3);", out)

    def test_multiline_invocation_still_expands(self):
        out = _pp("`define P(a,b) a+b\nx = `P(1,\n 2);\n")
        self.assertIn("x = 1+2 ;", " ".join(out.split()))
        # the expansion stays on the invocation's first line and the text after
        # it on its own source line (preprocess.PreprocessedText.line_map)
        self.assertEqual(out.split("\n")[:2], ["x = 1+2", ";"])
        self.assertEqual(out.line_map[:2], [("t.va", 2), ("t.va", 3)])


class TestDirectivesInComments(unittest.TestCase):

    def test_define_in_one_line_comment(self):
        out = _pp("`define RV 1000.0\n/* `define RV 2000.0 */\nr = `RV;\n")
        self.assertIn("r = 1000.0;", out)

    def test_define_in_multiline_comment(self):
        out = _pp("`define RV 1000.0\n/*\n`define RV 2000.0\n*/\nr = `RV;\n")
        self.assertIn("r = 1000.0;", out)

    def test_ifdef_in_comment_swallows_nothing(self):
        out = _pp("/*\n`ifdef NOT_DEFINED\n*/\nr = 1;\n")
        self.assertIn("r = 1;", out)

    def test_endif_in_comment_is_not_an_error(self):
        out = _pp("`ifdef insideADMS\nr = 1;\n/*\n`endif\n*/\n`endif\ns = 2;\n")
        self.assertIn("r = 1;", out)
        self.assertIn("s = 2;", out)

    def test_line_comment_marker_inside_block_comment(self):
        out = _pp("/* see http://example.org */ `define Q 7\nq = `Q;\n")
        self.assertIn("q = 7;", out)

    def test_adjacent_block_comments_keep_the_next_lines(self):
        # BSIM-SOI 4.5-4.7: ``/*Tcen = 1.9e-9 / T1;*//*ADOS added ...*/``.
        # The ``//`` inside ``*//*`` was taken for a line comment, the line
        # was cut to ``/*Tcen = 1.9e-9 / T1;*`` and the comment it left open
        # swallowed the model's code up to the next ``*/`` (15 statements).
        out = _pp("/*x = 1;*//*old*/\nr = 1;\n/* end */\n")
        code = re.sub(r'/\*.*?\*/', '', out, flags=re.S)
        self.assertIn("r = 1;", code)

    def test_comment_opener_in_a_string(self):
        out = _pp('`define Z 3\ns = "/* no comment"; z = `Z;\n')
        self.assertIn('s = "/* no comment"; z = 3;', out)


# The divider: V(out) = 0.5 with r = 1000, 1/3 with r = 2000.
VA_REDEFINE = """`include "disciplines.vams"
`include "constants.vams"
`define RV 1000.0
module a1(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = `RV;
  analog I(p, n) <+ V(p, n) / r;
endmodule
`define RV 2000.0
"""

VA_COMMENTED_DEFINE = """`include "disciplines.vams"
`include "constants.vams"
`define RV 1000.0
/* the old value:
`define RV 2000.0
*/
module a1(p, n);
  inout p, n;
  electrical p, n;
  analog I(p, n) <+ V(p, n) / `RV;
endmodule
"""

VA_ADJACENT_COMMENTS = """`include "disciplines.vams"
`include "constants.vams"
module a1(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  analog begin
    /*I(p, n) <+ 0.0;*//*the old contribution*/
    I(p, n) <+ V(p, n) / r;
    /* end of the contributions */
  end
endmodule
"""

DECK = """* macros expand at the point of use
.hdl "a1.va"
v1 in 0 1
.model m1 a1
ya1 x1 in out m1
r2 out 0 1k
.op
.print dc format=noindex v(out)
.end
"""


def _run_deck(work, va_text):
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
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    return log, float(lines[1].split()[lines[0].split().index("V(OUT)")])


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce at %s" % XYCE)
class TestPointOfUseThroughXyce(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="pymspou_")

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
        self.assertTrue(abs(v - 0.5) <= RTOL * 0.5, "V(out) = %.12g, expected 0.5" % v)

    def test_redefinition_after_the_module(self):
        self._check("redef", VA_REDEFINE)

    def test_define_in_a_block_comment(self):
        self._check("comment", VA_COMMENTED_DEFINE)

    def test_adjacent_block_comments(self):
        self._check("adjacent", VA_ADJACENT_COMMENTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
