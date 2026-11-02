"""The Verilog-AMS standard headers: PyMS ships disciplines.vams and
constants.vams (Accellera Verilog-AMS 2.4.0, Annex D and E, verbatim: the
notice permits verbatim copies; the same text openvaf-r carries built in)
beside its preprocessor and resolves an `include of them -- and of the
legacy names openvaf-r also knows, discipline.h, disciplines.h,
disciplines.va, constants.h, constants.va -- silently.  An include that is
found nowhere keeps its warning ``preprocess: warning: 'x' not found
(skipped)`` (vamos reads that form) and the module is built without it.

THE BUG (PyMS defect P9, vamos spectre guide trials 2026-10-19..24):
every build printed ``preprocess: warning: 'disciplines.vams' not found
(skipped)`` (PyMS shipped no disciplines.vams), and its constants.vams
carried physical constants of several CODATA vintages instead of the
Annex's: P_K 1.3806488e-23 where the LRM has 1.3806503e-23 (1.1e-6
relative, over the 1e-6 acceptance), P_Q 1.6021766208e-19 (LRM
1.602176462e-19), P_H 6.62607015e-34 (LRM 6.62606876e-34), P_U0 as a
number (LRM (4.0e-7 * `M_PI)); and of the legacy names only constants.h
(HICUM) resolved.  Measured 2026-10-24, the
constant times a scale as a current into 1k: Xyce V(k) = 1.3806488 vs
VACASK 1.3806503 (openvaf-r's built-in constants.vams is the Annex).

What this test pins:
  * the two headers sit beside vae/preprocess.py and carry the Accellera
    2.4.0 notice
  * an `include of either, and of the five legacy names, resolves with
    nothing on stderr and no `include left in the output
  * every macro of constants.vams expands to the Annex's text, verbatim
    (M_PI 3.14159265358979323846, P_K 1.3806503e-23, ...), and the
    PHYSICAL_CONSTANTS_NIST2010 selector works as in the Annex
  * a header beside the .va wins over the shipped one (search order)
  * a missing include still warns, once per name, in the form vamos reads,
    and the module is still built
  * end to end through Xyce (skipped without the binary): eight constants
    as node voltages equal the Annex values to 1e-6 (VACASK: k
    1.380650300000000e+00, q 1.602176462000000e+00, h 6.626068759999999e+00,
    e 8.854187817000000e+00, u 1.256637061435917e+00, pi
    3.141592653589793e+00, c0 2.731500000000000e+00, cc
    2.997924580000000e+00), and the divider's Xyce log carries no
    "preprocess: warning".

The Xyce runs use THIS copy of PyMS (PYMS_DIR = the tree this file sits
in) and private PYMS_CACHE / PYMS_VAE_CACHE under a scratch directory.

Run:
    python3 vae/tests/test_std_headers.py
$XYCE names the binary (default /usr/local/src/xyce-build/src/Xyce);
HDLTEST_KEEP=1 keeps the scratch directory.
"""
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_VAE_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VAE_PKG_DIR not in sys.path:
    sys.path.insert(0, _VAE_PKG_DIR)

from vae import preprocess as pp  # noqa: E402
from vae.preprocess import preprocess_source, PreprocessError  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
RTOL = 1e-6

VAE_DIR = os.path.dirname(os.path.abspath(pp.__file__))
STD_HEADERS = ("disciplines.vams", "constants.vams")
LEGACY_NAMES = {"discipline.h": "disciplines.vams", "disciplines.h": "disciplines.vams",
                "disciplines.va": "disciplines.vams", "constants.h": "constants.vams",
                "constants.va": "constants.vams"}

# Verilog-AMS 2.4.0 Annex D (constants.vams), verbatim values.
CONSTANTS = {
    "M_E": "2.7182818284590452354", "M_LOG2E": "1.4426950408889634074",
    "M_LOG10E": "0.43429448190325182765", "M_LN2": "0.69314718055994530942",
    "M_LN10": "2.30258509299404568402", "M_PI": "3.14159265358979323846",
    "M_TWO_PI": "6.28318530717958647693", "M_PI_2": "1.57079632679489661923",
    "M_PI_4": "0.78539816339744830962", "M_1_PI": "0.31830988618379067154",
    "M_2_PI": "0.63661977236758134308", "M_2_SQRTPI": "1.12837916709551257390",
    "M_SQRT2": "1.41421356237309504880", "M_SQRT1_2": "0.70710678118654752440",
    "P_Q": "1.602176462e-19", "P_C": "2.99792458e8", "P_K": "1.3806503e-23",
    "P_H": "6.62606876e-34", "P_EPS0": "8.854187817e-12",
    "P_U0": "(4.0e-7 * 3.14159265358979323846)", "P_CELSIUS0": "273.15",
}
NIST2010 = {"P_Q": "1.602176565e-19", "P_K": "1.3806488e-23",
            "P_H": "6.62606957e-34", "P_EPS0": "8.854187817e-12"}

VA_DIVIDER = """`include "disciplines.vams"
`include "constants.vams"
module a1(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  analog I(p, n) <+ V(p, n) / r;
endmodule
"""

DECK_DIVIDER = """* P9: the standard headers resolve silently
.hdl "a1.va"
v1 in 0 1
.model m1 a1
ya1 x1 in out m1
r2 out 0 1k
.op
.print dc format=noindex v(out)
.end
"""

# Each constant times a scale, as a current into its own 1k: V(node) = value * scale * 1k.
VA_CONSTANTS = """`include "disciplines.vams"
`include "constants.vams"
module c1(k, q, h, e, u, pi, c0, cc, n);
  inout k, q, h, e, u, pi, c0, cc, n;
  electrical k, q, h, e, u, pi, c0, cc, n;
  analog begin
    I(n, k)  <+ `P_K * 1e20;
    I(n, q)  <+ `P_Q * 1e16;
    I(n, h)  <+ `P_H * 1e31;
    I(n, e)  <+ `P_EPS0 * 1e9;
    I(n, u)  <+ `P_U0 * 1e3;
    I(n, pi) <+ `M_PI * 1e-3;
    I(n, c0) <+ `P_CELSIUS0 * 1e-5;
    I(n, cc) <+ `P_C * 1e-11;
  end
endmodule
"""

DECK_CONSTANTS = """* P9: constants.vams values as node voltages
.hdl "c1.va"
.model m1 c1
yc1 x1 k q h e u pi c0 cc 0 m1
rk k 0 1k
rq q 0 1k
rh h 0 1k
re e 0 1k
ru u 0 1k
rpi pi 0 1k
rc0 c0 0 1k
rcc cc 0 1k
.op
.print dc format=noindex precision=12 width=22 v(k) v(q) v(h) v(e) v(u) v(pi) v(c0) v(cc)
.end
"""
# node: (macro, scale * 1k)
EXPECTED_V = {"V(K)": ("P_K", 1e23), "V(Q)": ("P_Q", 1e19), "V(H)": ("P_H", 1e34),
              "V(E)": ("P_EPS0", 1e12), "V(U)": ("P_U0", 1e6), "V(PI)": ("M_PI", 1.0),
              "V(C0)": ("P_CELSIUS0", 1e-2), "V(CC)": ("P_C", 1e-8)}


def _expand(text, filename):
    """preprocess_source with stderr captured; (output, stderr)."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        out = preprocess_source(text, filename)
    return out, err.getvalue()


class TestShippedHeaders(unittest.TestCase):

    def test_headers_sit_beside_the_preprocessor(self):
        for name in STD_HEADERS:
            path = os.path.join(VAE_DIR, name)
            self.assertTrue(os.path.exists(path), "%s is not shipped" % path)
            with open(path) as f:
                text = f.read()
            self.assertIn("Accellera", text)
            self.assertIn("Version 2.4.0", text)

    def test_includes_resolve_silently(self):
        work = tempfile.mkdtemp(prefix="pymshdr_")
        try:
            out, err = _expand(VA_DIVIDER, os.path.join(work, "a1.va"))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(err, "", "stderr not empty: %r" % err)
        self.assertNotIn("`include", out)
        self.assertIn("discipline electrical", out)
        self.assertIn("nature Current", out)
        self.assertIn("module a1(p, n)", out)

    def test_legacy_names_resolve(self):
        # constants.vams is directives only (its expansion is blank lines), so
        # a macro's value shows it was read; disciplines.vams emits its text.
        for legacy, std in LEGACY_NAMES.items():
            disc = std.startswith("disc")
            out, err = _expand('`include "%s"\n%s' % (legacy, "" if disc else "X=`P_CELSIUS0;\n"),
                               "t.va")
            self.assertEqual(err, "", "%s: stderr %r" % (legacy, err))
            self.assertNotIn("`include", out, legacy)
            marker = "discipline electrical" if disc else "X=273.15;"
            self.assertIn(marker, out, "%s did not resolve to %s" % (legacy, std))

    def test_constants_are_the_annex_values(self):
        for name, value in sorted(CONSTANTS.items()):
            out, err = _expand('`include "constants.vams"\nX=`%s;\n' % name, "t.va")
            self.assertEqual(err, "", name)
            self.assertIn("X=%s;" % value, out,
                          "%s expands to %r, not %s" % (
                              name, [ln for ln in out.splitlines() if ln.startswith("X=")], value))

    def test_nist2010_selector(self):
        for name, value in sorted(NIST2010.items()):
            out, _ = _expand('`define PHYSICAL_CONSTANTS_NIST2010\n`include "constants.vams"\n'
                             'X=`%s;\n' % name, "t.va")
            self.assertIn("X=%s;" % value, out, name)

    def test_header_beside_the_source_wins(self):
        work = tempfile.mkdtemp(prefix="pymshdr_")
        try:
            with open(os.path.join(work, "constants.vams"), "w") as f:
                f.write("`define P_K 1\n")
            out, err = _expand('`include "constants.vams"\nX=`P_K;\n',
                               os.path.join(work, "t.va"))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(err, "")
        self.assertIn("X=1;", out)

    def test_missing_include_still_warns(self):
        # A header that is found nowhere keeps the warning, in the form vamos
        # reads (run_xyce.py _PYMS_SKIPPED), once per name; the module is
        # still built (real-world files build without a Cadence include they
        # name: EKV 2.6 SDext, compact.vams, test_va_terminal_split).
        out, err = _expand('`include "disciplines.vams"\n`include "nosuch_macros.include"\n'
                           '`include "nosuch_macros.include"\n'
                           'module a1(p, n); endmodule\n', "/nowhere/t.va")
        self.assertEqual(err, "preprocess: warning: 'nosuch_macros.include' not found (skipped)\n")
        self.assertIn("module a1(p, n); endmodule", out)


def _xyce_env(work):
    env = dict(os.environ)
    env["PYMS_DIR"] = _VAE_PKG_DIR
    env["PYMS_CACHE"] = os.path.join(work, "hdl_cache")
    env["PYMS_VAE_CACHE"] = os.path.join(work, "vae_cache")
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.pop("VAE_SO_PATH", None)
    env.pop("VAE_SO_DIR", None)
    return env


def _run_deck(work, deck_name, deck_text, va_name, va_text):
    with open(os.path.join(work, va_name), "w") as f:
        f.write(va_text)
    with open(os.path.join(work, deck_name), "w") as f:
        f.write(deck_text)
    r = subprocess.run([XYCE, deck_name], capture_output=True, text=True,
                       timeout=600, cwd=work, env=_xyce_env(work))
    log = r.stdout + r.stderr
    prn = os.path.join(work, deck_name + ".prn")
    prn_text = ""
    if os.path.exists(prn):
        with open(prn) as f:
            prn_text = f.read()
    return log, prn_text


def _prn_row(prn_text):
    lines = [ln for ln in prn_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise AssertionError("no data row in .prn")
    return dict(zip(lines[0].split(), (float(x) for x in lines[1].split())))


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce at %s" % XYCE)
class TestStdHeadersThroughXyce(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="pymshdr_xyce_")

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("HDLTEST_KEEP"):
            sys.stderr.write("kept %s\n" % cls.work)
        else:
            shutil.rmtree(cls.work, ignore_errors=True)

    def test_divider_builds_without_a_warning(self):
        work = os.path.join(self.work, "div")
        os.makedirs(work)
        log, prn = _run_deck(work, "a1.cir", DECK_DIVIDER, "a1.va", VA_DIVIDER)
        self.assertNotIn("preprocess: warning", log)
        self.assertNotIn("not found", log)
        self.assertTrue(prn, "no .prn; log:\n" + log[-3000:])
        v = _prn_row(prn)["V(OUT)"]
        self.assertTrue(abs(v - 0.5) <= RTOL * 0.5, "V(out) = %.12g" % v)

    def test_constants_reach_the_device(self):
        work = os.path.join(self.work, "const")
        os.makedirs(work)
        log, prn = _run_deck(work, "c1.cir", DECK_CONSTANTS, "c1.va", VA_CONSTANTS)
        self.assertTrue(prn, "no .prn; log:\n" + log[-3000:])
        row = _prn_row(prn)
        for col, (macro, scale) in sorted(EXPECTED_V.items()):
            expected = eval(CONSTANTS[macro]) * scale  # Annex text, a Python number
            got = row[col]
            self.assertTrue(abs(got - expected) <= RTOL * abs(expected),
                            "%s = %.12g, expected %.12g (%s)" % (col, got, expected, macro))


if __name__ == "__main__":
    unittest.main(verbosity=2)
