"""Parameter defaults are Verilog-A constant expressions, evaluated with the
language's semantics, per instance (PYMS_BUGS.md P1, 2026-10-24).

THE BUG: PyMS read a parameter default as a number, as another parameter's
name, or as a Python ``eval`` of its tokens (xyce_device_gen._resolve_default
and GiNaCEmitter.__init__), then overlaid the device shell's NAME=value file
with the names IN CAPITALS. On the reference divider (V1 = 1 V into the module
over R2 = 1 k, out = 1k / (r + 1k)) that gave, silently, exit 0:

  parameter real r = pow(10, 3);                  out = 0      (VACASK 0.5)
  parameter real r = (1 > 0) ? 1000.0 : 2000.0;   out = 0
  parameter real r = 2 ** 10;                     out = 0      (VACASK 0.494071)
  parameter real r = 1k * 2;  1_000;  'h3e8       out = 0
  parameter real g = 1 / 1000;                    out = 0.5    (Verilog-A 1/1000 = 0)
  parameter real r = 7 / 2 * 1000;                out = 0.2222 (Verilog-A r = 3000, 0.25)
  parameter real r = (3 > 2 > 1) * 1000.0 + 1.0;  out = 0.49975 (Verilog-A r = 1)
  parameter real RR = 500; parameter real R = 2 * RR;  out = 0 (R stayed the shell's 0)
  parameter integer b = 2; parameter real r = 2000 / b;  with b=4 on the card:
                                                  out = 0.5 (r baked from b's default; Verilog-A r = 500)
  parameter real r = 1k;  with r=2k on the card:  out = 0.5 (the value landed on a stray R)
  parameter real r = rx * 1000.0;  (rx undefined) out = 0, no message

THE FIX: vae/va_const.py evaluates a default with Verilog-A typing and
operators (integer division, the sign of %, left-associative comparisons that
yield 0/1, **, the ternary, the standard functions, scale factors on any
number, digit separators, based literals, $simparam from the shell), in
declaration order with the instance's given values (build_vae_so.py maps the
shell's NAME back to the module's spelling and bakes only the given set: the
P10 fix, test_hdl_param_name_case.py); the shell re-evaluates dependent defaults per instance in
processParams(); anything unevaluable is an error at .hdl time naming the file,
the line and the construct, and a build that fails at instance time is a Xyce
error, never a device that conducts nothing.

Two layers of tests: the evaluator alone (fast, no Xyce), and the forms
end-to-end through Xyce on the reference divider (private PYMS_CACHE /
PYMS_VAE_CACHE, PYMS_DIR = the PyMS copy this file sits in). Skips the Xyce
layer when there is no Xyce binary.

Run:
    cd utils/PyMS && python3 vae/tests/test_param_defaults.py
$XYCE names the binary (default /usr/local/src/xyce-build/src/Xyce);
PDTEST_KEEP=1 keeps the scratch directory.
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

from vae.parser import parse_verilog_a  # noqa: E402
from vae.preprocess import preprocess_source  # noqa: E402
from vae.va_const import (  # noqa: E402
    ConstExprError, evaluate_param_defaults, parse_const_expr, evaluate,
    XYCE_SIMPARAM_DEFAULTS,
)

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
RTOL = 1e-6

HEAD = '`include "disciplines.vams"\n`include "constants.vams"\n'


def module(decls, body="analog I(p, n) <+ V(p, n) / r;", pre=""):
    return (HEAD + pre + "module a1(p, n);\n  inout p, n;\n  electrical p, n;\n  "
            + decls.replace("\n", "\n  ") + "\n  " + body + "\nendmodule\n")


def defaults(decls, given=None, body="analog I(p, n) <+ V(p, n) / r;"):
    src = preprocess_source(module(decls, body), filename="a1.va")
    mod = parse_verilog_a(src, filename="a1.va")
    return evaluate_param_defaults(mod, given or {}, XYCE_SIMPARAM_DEFAULTS)


def ev(text):
    return evaluate(parse_const_expr(text), {}, lambda name: XYCE_SIMPARAM_DEFAULTS.get(name))


class TestEvaluator(unittest.TestCase):
    """va_const alone: Verilog-A semantics of the constant expressions."""

    def test_integer_arithmetic(self):
        self.assertEqual(ev("1 / 1000"), 0)
        self.assertEqual(ev("7 / 2 * 1000"), 3000)
        self.assertEqual(ev("1000 * (3 / 2)"), 1000)
        self.assertEqual(ev("-7 / 2"), -3)                 # truncation toward zero
        self.assertEqual(ev("-7 % 3"), -1)                 # the sign of the first operand
        self.assertEqual(ev("7 % -3"), 1)
        self.assertEqual(ev("2 ** 10"), 1024)
        self.assertEqual(ev("2 ** -1"), 0)                 # LRM table 4-6
        self.assertEqual(ev("(-1) ** 3"), -1)
        self.assertEqual(ev("-2 ** 2"), 4)                 # the unary minus binds tightest
        self.assertIsInstance(ev("7 / 2"), int)

    def test_real_arithmetic(self):
        self.assertAlmostEqual(ev("1000.0 / 3"), 333.3333333333333)
        self.assertAlmostEqual(ev("2.0 ** -1 * 2000"), 1000.0)
        self.assertAlmostEqual(ev("7.5 % 2 * 400"), 600.0)
        self.assertAlmostEqual(ev("-7.5 % 2"), -1.5)
        self.assertIsInstance(ev("2000.0 / 2"), float)

    def test_comparisons_and_logic(self):
        self.assertEqual(ev("(3 > 2 > 1)"), 0)            # left-associative, 0/1
        self.assertEqual(ev("(1 > 0) * 1000"), 1000)
        self.assertEqual(ev("(1 >= 0) + (1 <= 1) + (1 == 1) + (1 != 2) + (2 > 1) + (1 < 2)"), 6)
        self.assertEqual(ev("(1 && 1) + (1 || 0) + !0 + !5"), 3)
        self.assertEqual(ev("1.5 > 1"), 1)

    def test_bitwise(self):
        self.assertEqual(ev("(1016 & 1003) + (1000 | 0) + (1000 ^ 0) + ~(-1001) + (125 << 3) + (8000 >> 3)"), 6000)
        with self.assertRaisesRegex(ConstExprError, "needs integer operands"):
            ev("1000.0 | 0")
        with self.assertRaisesRegex(ConstExprError, "needs an integer operand"):
            ev("~1.0")

    def test_ternary(self):
        self.assertEqual(ev("(1 > 0) ? 1000.0 : 2000.0"), 1000.0)
        self.assertEqual(ev("0 ? 1.0 : 1 ? 1000.0 : 2.0"), 1000.0)
        self.assertIsInstance(ev("1 ? 7 : 2.0"), float)   # real if either branch is real
        self.assertEqual(ev("1 ? 7 : 1 / 0"), 7)          # the other branch is not evaluated

    def test_functions(self):
        self.assertAlmostEqual(ev("pow(10, 3)"), 1000.0)
        self.assertAlmostEqual(ev("sqrt(1e6)"), 1000.0)
        self.assertAlmostEqual(ev("ln(1.0) + 1000.0"), 1000.0)
        self.assertAlmostEqual(ev("log(1000.0)"), 3.0)                 # Verilog-A log is base 10
        self.assertAlmostEqual(ev("exp(0) * 1000"), 1000.0)
        self.assertEqual(ev("abs(-7) / 2"), 3)                          # abs keeps the integer type
        self.assertEqual(ev("min(7, 9) / 2"), 3)
        self.assertAlmostEqual(ev("max(1000.0, 1.0) + min(0.0, 2000.0)"), 1000.0)
        self.assertAlmostEqual(ev("floor(500.5) + ceil(499.5)"), 1000.0)
        self.assertAlmostEqual(ev("cos(0.0) + sin(0.0) + tan(0.0) + atan(0.0) + hypot(3.0, 4.0) + atan2(0.0, 1.0) + asin(0.0) + acos(1.0)"), 6.0)
        self.assertAlmostEqual(ev("cosh(0.0) + sinh(0.0) + tanh(0.0) + asinh(0.0) + acosh(1.0) + atanh(0.0)"), 1.0)
        with self.assertRaisesRegex(ConstExprError, "sqrt"):
            ev("sqrt(-1.0)")
        with self.assertRaisesRegex(ConstExprError, "not a standard function"):
            ev("foo(1.0)")

    def test_literals(self):
        self.assertEqual(ev("1_000"), 1000)
        self.assertAlmostEqual(ev("1_000.0"), 1000.0)
        self.assertAlmostEqual(ev("1k * 2"), 2000.0)
        self.assertAlmostEqual(ev("1000a * 1e18"), 1000.0)
        self.assertAlmostEqual(ev("1K / 1"), 1000.0)
        self.assertAlmostEqual(ev(".5k * 2"), 1000.0)
        self.assertAlmostEqual(ev("1m * 1"), 1e-3)
        self.assertEqual(ev("'h3e8"), 1000)
        self.assertEqual(ev("12'h3E8"), 1000)
        self.assertEqual(ev("8'shFF"), -1)
        self.assertEqual(ev("3000-1000-1000"), 1000)
        self.assertAlmostEqual(ev("+(+1000.0)"), 1000.0)
        self.assertAlmostEqual(ev("1000.0 /* c */ * 1"), 1000.0)

    def test_simparam(self):
        self.assertEqual(ev('$simparam("gmin", 1000.0)'), 1e-12)
        self.assertAlmostEqual(ev('$simparam("nosuchparam", 1000.0)'), 1000.0)
        with self.assertRaisesRegex(ConstExprError, "unknown to Xyce"):
            ev('$simparam("nosuchparam")')
        # 1.1.6: $temperature / $vt in a default are the instance's at run
        # time (a thermal default, test_temperature.py); without a temperature
        # given they have no constant value.
        with self.assertRaisesRegex(ConstExprError, "temperature at run time"):
            ev("$temperature")
        with self.assertRaisesRegex(ConstExprError, "not a constant"):
            ev("$abstime")

    def test_errors_name_the_construct(self):
        with self.assertRaisesRegex(ConstExprError, "integer division by zero"):
            ev("1000 / 0")
        with self.assertRaisesRegex(ConstExprError, "macro `RV is not defined"):
            ev("`RV")
        with self.assertRaisesRegex(ConstExprError, "not a parameter"):
            ev("rx * 1000.0")
        with self.assertRaisesRegex(ConstExprError, "malformed number"):
            ev("2abs")


class TestParamDefaults(unittest.TestCase):
    """evaluate_param_defaults on parsed modules: typing, dependency order, overrides."""

    def test_dependent_default_follows_an_override(self):
        d = "parameter integer b = 2;\nparameter real r = 2000 / b;"
        self.assertEqual(defaults(d), {"b": 2, "r": 1000.0})
        self.assertEqual(defaults(d, {"b": 4.0}), {"b": 4, "r": 500.0})
        d = "parameter real a = 250.0;\nparameter real b = 2 * a;\nparameter real r = 2 * b;"
        self.assertEqual(defaults(d, {"a": 500.0}), {"a": 500.0, "b": 1000.0, "r": 2000.0})

    def test_integer_parameter_rounds(self):
        self.assertEqual(defaults("parameter integer k = 2.5;\nparameter real r = 400.0 * k;"),
                         {"k": 3, "r": 1200.0})
        self.assertEqual(defaults("parameter integer k = -2.5;\nparameter real r = -400.0 * k;"),
                         {"k": -3, "r": 1200.0})
        self.assertEqual(defaults("parameter integer k = 7;\nparameter real r = k % 4 * 1000 - 2000;"),
                         {"k": 7, "r": 1000.0})
        self.assertEqual(defaults("parameter integer k = 1k;\nparameter real r = k;"), {"k": 1000, "r": 1000.0})

    def test_names_are_case_sensitive(self):
        self.assertEqual(defaults("parameter real RR = 500.0;\nparameter real R = 2 * RR;",
                                  body="analog I(p, n) <+ V(p, n) / R;"), {"RR": 500.0, "R": 1000.0})
        self.assertEqual(defaults("parameter real RR = 500.0;\nparameter real R = 2 * RR;", {"RR": 1000.0},
                                  body="analog I(p, n) <+ V(p, n) / R;"), {"RR": 1000.0, "R": 2000.0})

    def test_macros_and_strings(self):
        self.assertAlmostEqual(defaults("parameter real r = `M_PI * 1000.0 / `M_PI;")["r"], 1000.0)
        v = defaults('parameter string s = "x";\nparameter real r = 1k;')
        self.assertEqual(v, {"s": "x", "r": 1000.0})

    def test_forward_reference_self_reference_and_unknown_name_are_errors(self):
        with self.assertRaisesRegex(ConstExprError, r"parameter r of module a1: .* reads rr, a parameter declared after it"):
            defaults("parameter real r = 2 * rr;\nparameter real rr = 500.0;")
        with self.assertRaisesRegex(ConstExprError, "reads a, the parameter itself"):
            defaults("parameter real a = a;\nparameter real r = 1k;")
        with self.assertRaisesRegex(ConstExprError, "parameter r of module a1: .* the name rx is not a parameter"):
            defaults("parameter real r = rx * 1000.0;")

    def test_a_comment_that_eats_the_semicolon_is_an_error(self):
        with self.assertRaisesRegex(ConstExprError, "parameter r"):
            defaults("parameter real r = 1000.0 // c;")


# ---------------------------------------------------------------------------
# End to end through Xyce
# ---------------------------------------------------------------------------

DECK = """* parameter defaults (PYMS_BUGS P1): the reference divider
.hdl "a1.va"
V1 in 0 DC 1
.model m1 a1 %(card)s
Ya1 X1 in out m1 %(inst)s
R2 out 0 1k
%(options)s
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
    lines = [ln for ln in prn_text.splitlines() if ln.strip()]
    hdr = None
    for ln in lines:
        cols = ln.split()
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
    """The forms through Xyce: out = 1k / (r + 1k) on the reference divider."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="pdtest_")
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("PDTEST_KEEP"):
            print("kept", cls.work)
        else:
            shutil.rmtree(cls.work, ignore_errors=True)

    def run_deck(self, va, card="", inst="", options=""):
        type(self).n += 1
        d = os.path.join(self.work, "c%02d" % self.n)
        os.makedirs(d)
        with open(os.path.join(d, "a1.va"), "w") as f:
            f.write(va)
        with open(os.path.join(d, "a1.cir"), "w") as f:
            f.write(DECK % dict(card=card, inst=inst, options=options))
        r = subprocess.run([XYCE, "a1.cir"], capture_output=True, text=True,
                           timeout=600, cwd=d, env=_xyce_env(d))
        log = r.stdout + r.stderr
        prn = os.path.join(d, "a1.cir.prn")
        out = None
        if os.path.exists(prn):
            with open(prn) as f:
                out = _prn_vout(f.read())
        return r.returncode, out, log

    def check(self, va, expected, card="", inst="", options=""):
        rc, out, log = self.run_deck(va, card, inst, options)
        self.assertEqual(rc, 0, "Xyce exit %s:\n%s" % (rc, log[-3000:]))
        self.assertIsNotNone(out, "no V(OUT) in the .prn:\n%s" % log[-3000:])
        self.assertTrue(abs(out - expected) <= RTOL * max(abs(expected), 1e-3),
                        "out = %.12g, expected %.12g\n%s" % (out, expected, log[-3000:]))

    def test_forms_that_used_to_give_0(self):
        third = 1000.0 / 3000.0
        self.check(module("parameter real r = pow(10, 3);"), 0.5)
        self.check(module("parameter real r = (1 > 0) ? 1000.0 : 2000.0;"), 0.5)
        self.check(module("parameter real r = 2 ** 10;"), 1000.0 / 2024.0)
        self.check(module("parameter real r = 1k * 2;"), third)
        self.check(module("parameter real r = 1_000;"), 0.5)
        self.check(module("parameter real r = (1 >= 0) * 500.0 + (1 && 1) * 250.0 + !0 * 250.0;"), 0.5)
        self.check(module("parameter real r = 'h3e8;"), 0.5)
        self.check(module("parameter real RR = 500.0;\nparameter real R = 2 * RR;",
                          body="analog I(p, n) <+ V(p, n) / R;"), 0.5)

    def test_verilog_a_arithmetic_not_python_s(self):
        self.check(module("parameter real g = 1 / 1000;", body="analog I(p, n) <+ V(p, n) * g;"), 0.0)
        self.check(module("parameter real r = 7 / 2 * 1000;"), 0.25)
        self.check(module("parameter real r = (3 > 2 > 1) * 1000.0 + 1.0;"), 1000.0 / 1001.0)
        self.check(module("parameter real r = -7 % 3 * 1000 + 2000;"), 0.5)
        self.check(module("parameter integer k = 2.5;\nparameter real r = 400.0 * k;"), 1000.0 / 2200.0)

    def test_dependent_default_under_an_override(self):
        va = module("parameter integer b = 2;\nparameter real r = 2000 / b;")
        self.check(va, 0.5)
        self.check(va, 1000.0 / 1500.0, card="b=4")
        va = module("parameter real RR = 500.0;\nparameter real R = 2 * RR;",
                    body="analog I(p, n) <+ V(p, n) / R;")
        self.check(va, 1000.0 / 3000.0, card="RR=1000")
        va = module('(* type="instance" *) parameter real rr = 500.0;\nparameter real r = 2 * rr;')
        self.check(va, 1000.0 / 3000.0, inst="rr=1000")

    def test_card_value_reaches_a_lower_case_parameter(self):
        self.check(module("parameter real r = 1k;"), 1000.0 / 3000.0, card="r=2k")

    def test_simparam_is_the_simulator_s_value(self):
        va = module('parameter real r = 1e9 * $simparam("gmin", 1.0) * 1000.0;')
        self.check(va, 1000.0 / 1001.0)                                    # gmin 1e-12: r = 1
        self.check(va, 0.5, options=".options device gmin=1e-9")           # r = 1000

    def test_an_unevaluable_default_is_an_error_at_hdl_time(self):
        rc, out, log = self.run_deck(module("parameter real r = rx * 1000.0;"))
        self.assertNotEqual(rc, 0, "Xyce ran a module with an undefined name in a default:\n%s" % log[-2000:])
        flat = " ".join(log.split())                 # Xyce wraps its messages
        self.assertIn("PyMS compilation failed", flat)
        self.assertRegex(flat, r"a1\.va:6: parameter r of module a1: its default rx \* 1000\.0: the name rx is not a parameter")

    def test_a_build_that_fails_at_instance_time_is_an_error(self):
        va = module("parameter real rr = 1e6;\nparameter real r = sqrt(rr);")
        self.check(va, 0.5)                                                 # the form is fine
        rc, out, log = self.run_deck(va, card="rr=-1")                      # this instance is not
        self.assertNotEqual(rc, 0, "Xyce ran a device whose math library failed to build:\n%s" % log[-2000:])
        flat = " ".join(log.split())                 # Xyce wraps its messages
        self.assertIn("did not build or load", flat)
        self.assertIn("sqrt(-1.0) is not a real number", flat)


if __name__ == "__main__":
    unittest.main(verbosity=2)
