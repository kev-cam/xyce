"""A .model card's parameter must reach the PyMS .hdl module's math.
Recorded as a known failure on 2026-10-10 (design q.50); FIXED 2026-10-24
(P10 of PYMS_BUGS.md; the fix and its neighbours are measured by
test_hdl_param_name_case.py).

THE BUG (observed 2026-10-10 on DEVELOPMENT-202610100047 / 7cd78110; evidence
decks sv2ghdl/tests/vamos/fixtures/spectre/engine_facts/E63): a Verilog-A
module ``parameter real r = 1k`` driven by a card that sets r=2k is evaluated
with its DEFAULT. The card value is parsed and LISTED -- the .op listing shows
the model with ``R 2000`` -- yet 1 V across the device draws 1 mA, not 0.5 mA:
I(V1) = -1.0e-3 where -5.0e-4 is expected. Silent: no warning, no error. Both
card forms:

  (1) Y device:          .model rm myres r=2k        ymyres x1 a 0 rm
  (2) model-group level: module mydio (* xyceModelGroup="DIODE"
                           xyceLevelNumber="2002" *) with
                         .model dm d level=2002 r=2k  d1 a 0 dm

WHERE IT IS LOST: the generated device shell is right -- Model::Model runs
setModParams(mb.params) and Instance::processParams() serializes the card
value into the JIT params file (``R=2000`` / ``__GIVEN__=R,``).
build_vae_so.py then UPPERCASES every name (Xyce's convention) and
GiNaCEmitter overlays that dict on the .va's own, case-sensitive parameter
names: param_values ends up {'r': 1000.0, 'R': 2000.0} ("Resolved 2 model
parameters" for a one-parameter module) and every substitution of ``r`` reads
the default. Compact models with upper-case parameters never see it.

This test runs both decks end-to-end through Xyce (private PYMS_CACHE /
PYMS_VAE_CACHE, as test_hdl_cache_va_identity.py does). Per deck form:
  test_card_value_reaches_model  -- the module is bound on the card and the
                                    .op listing shows R 2000 (passes today:
                                    pins the symptom's first half)
  test_card_value_applied        -- I(V1) = -5.0e-4 (the card's r=2k);
                                    @unittest.expectedFailure from 2026-10-10
                                    until the fix of 2026-10-24

THE FIX: build_vae_so.py maps the shell's upper-case names back onto the
module's own parameter names and bakes only the values the netlist supplied
(the __GIVEN__ set) and the ones the shell set itself (__SET__); every other
default is the emitter's, evaluated from the .va text with those values in
place. Skips when there is no Xyce binary.

Run:
    cd utils/PyMS && python3 -m unittest vae.tests.test_hdl_model_card_params -v
$XYCE names the binary (default /usr/local/src/xyce-build/src/Xyce);
HDLTEST_KEEP=1 keeps the scratch directory.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
# The PyMS this test sits in (utils/PyMS): Xyce runs it through PYMS_DIR.
# Without it Xyce took an inherited PYMS_DIR or its own lookup (the installed
# /usr/local/share/xyce/PyMS first), so the test measured whichever copy
# that was: I(V1) = -1.0e-3 with the copy before the fix, while this tree
# was right (2026-10-25, PYMS1-G and -D saw it fail with PYMS_DIR unset).
PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

R_CARD = 2000.0                 # the card's r=2k
I_EXPECTED = -1.0 / R_CARD      # 1 V across r=2k, as .print dc i(v1) signs it
I_DEFAULT = -1.0 / 1000.0       # the module default r = 1k: the symptom
RTOL = 1e-6

# Module sources: E63's va/myres.va and mydio.va, verbatim.
VA_MYRES = """`include "disciplines.vams"
module myres(p, n);
  inout p, n; electrical p, n;
  parameter real r = 1k;
  analog I(p, n) <+ V(p, n) / r;
endmodule
"""

VA_MYDIO = """`include "disciplines.vams"
module mydio(p, n) (* xyceModelGroup="DIODE" xyceLevelNumber="2002" *);
  inout p, n; electrical p, n;
  parameter real r = 1k;
  analog I(p, n) <+ V(p, n) / r;
endmodule
"""

# E63 a.cir: the card parameter on a PyMS Y device
DECK_Y = """* model-card parameter on a PyMS Y device (E63 a.cir)
.hdl "myres.va"
v1 a 0 1
.model rm myres r=2k
ymyres x1 a 0 rm
.op
.print dc i(v1)
.end
"""

# E63 c.cir: the model-group route, .model d level=N with a card parameter
DECK_LEVEL = """* model-group route: .model d level=N with a card parameter (E63 c.cir)
.hdl "mydio.va"
v1 a 0 1
.model dm d level=2002 r=2k
d1 a 0 dm
.op
.print dc i(v1)
.end
"""


def _xyce_env(work):
    env = dict(os.environ)
    env["PYMS_DIR"] = PYMS_DIR
    env["PYMS_CACHE"] = os.path.join(work, "hdl_cache")
    env["PYMS_VAE_CACHE"] = os.path.join(work, "vae_cache")
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.pop("VAE_SO_PATH", None)
    env.pop("VAE_SO_DIR", None)
    return env


def _run_deck(work, deck_name, deck_text, va_name, va_text):
    """Run one deck in `work`; return (Xyce stdout+stderr, .prn text or '')."""
    with open(os.path.join(work, va_name), "w") as f:
        f.write(va_text)
    deck = os.path.join(work, deck_name)
    with open(deck, "w") as f:
        f.write(deck_text)
    r = subprocess.run([XYCE, deck_name], capture_output=True, text=True,
                       timeout=600, cwd=work, env=_xyce_env(work))
    log = r.stdout + r.stderr
    prn = deck + ".prn"
    prn_text = ""
    if os.path.exists(prn):
        with open(prn) as f:
            prn_text = f.read()
    return log, prn_text


def _prn_value(prn_text, col, row=0):
    """The value in column `col` of data row `row` of a Xyce .prn."""
    lines = [ln for ln in prn_text.splitlines() if ln.strip()]
    if not lines:
        raise AssertionError("empty .prn")
    header = lines[0].split()
    if col not in header:
        raise AssertionError("column %s not in .prn header %r" % (col, header))
    return float(lines[1 + row].split()[header.index(col)])


class _CardParamMixin(object):
    """One deck form, run once per class; see the module docstring."""

    DECK_NAME = VA_NAME = DECK = VA = MODULE = None

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(XYCE):
            raise unittest.SkipTest("no Xyce at %s" % XYCE)
        cls.work = tempfile.mkdtemp(prefix="hdlcardparam_")
        cls.log, cls.prn = _run_deck(cls.work, cls.DECK_NAME, cls.DECK,
                                     cls.VA_NAME, cls.VA)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("HDLTEST_KEEP"):
            sys.stderr.write("kept %s\n" % cls.work)
        else:
            shutil.rmtree(cls.work, ignore_errors=True)

    def test_card_value_reaches_model(self):
        """The .hdl module is bound on the card and the .op listing shows the
        card's R 2000: the value reaches the model object (passes today)."""
        self.assertIn("compiled and registered %s" % self.MODULE, self.log)
        self.assertIn("Number of %s models: 1" % self.MODULE, self.log)
        self.assertIn("Number of %s instances: 1" % self.MODULE, self.log)
        self.assertRegex(self.log, r"\bR\s+2000\b",
                         "the .op model listing does not show R 2000")
        self.assertNotIn("Netlist error", self.log)

    def test_card_value_applied(self):
        """I(V1) = -5.0e-4: the card's r=2k is what the module computes with.
        Known failure from 2026-10-10 (module evaluated with its default
        r = 1k, I(V1) = -1.0e-3, while the listing showed R 2000) until the
        fix of 2026-10-24."""
        i = _prn_value(self.prn, "I(V1)")
        self.assertTrue(
            abs(i - I_EXPECTED) <= RTOL * abs(I_EXPECTED),
            "I(V1) = %.9g, expected %.9g (card r=2k); %s" % (
                i, I_EXPECTED,
                "module evaluated with its default r = 1k"
                if abs(i - I_DEFAULT) <= RTOL * abs(I_DEFAULT) else "neither"))


class TestYCardParam(_CardParamMixin, unittest.TestCase):
    """E63 a.cir: .model rm myres r=2k on a Y device."""
    DECK_NAME, DECK = "ycard.cir", DECK_Y
    VA_NAME, VA = "myres.va", VA_MYRES
    MODULE = "myres"


class TestLevelCardParam(_CardParamMixin, unittest.TestCase):
    """E63 c.cir: .model dm d level=2002 r=2k binding a DIODE-group module."""
    DECK_NAME, DECK = "levelcard.cir", DECK_LEVEL
    VA_NAME, VA = "mydio.va", VA_MYDIO
    MODULE = "mydio"


if __name__ == "__main__":
    unittest.main(verbosity=2)
