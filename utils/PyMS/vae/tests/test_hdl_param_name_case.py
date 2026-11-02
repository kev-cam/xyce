"""A .model card's or instance line's parameter value must reach the
Verilog-A module's math whatever the case of the parameter's name.

THE BUG (P10 of PYMS_BUGS.md, 2026-10-24; the known failure recorded on
2026-10-10 by test_hdl_model_card_params.py): the generated device shell
registers every parameter with Xyce under its UPPER-CASE name (Xyce
parameter names are case-insensitive), receives the card's value there
(the .op listing shows ``R 2000``), and Instance::processParams() writes
``R=2000`` / ``__GIVEN__=R,`` into the JIT params file. build_vae_so.py
reads those names back as they are, and GiNaCEmitter overlays
``{'R': 2000.0}`` BESIDE the module's own case-sensitive ``r``: every
substitution of ``r`` reads the .va default, every ``$param_given(r)`` is
false, every runtime-callback parameter named in lower case stays baked.
The same happens to an instance parameter (``(* type="instance" *)``), to a
mixed-case name (``Rs``), to the nmos/pmos type variable of a module that
names it ``type`` or ``dtype``, and to a default that depends on an
overridden parameter (the shell wrote its own reading of the default, 0 for
anything but a plain number, on top of the emitter's). Silent: exit 0, no
message, wrong current. Measured on the pristine PyMS (1 V across the
device, ``.print dc i(v1)``): card r=2k on ``parameter real r`` gives
I(V1) = -1.0e-3 (the default 1k) where -5.0e-4 is right; the same card on
``parameter real R`` is right, which is why compact models with
upper-case parameters never showed it.

THE FIX (vae/build_vae_so.py, vae/ginac_emitter.py, vae/xyce_device_gen.py):
build_vae_so.py maps the shell's names onto the module's own parameter
names (case-insensitive inverse map), bakes only the values the netlist
supplied (the ``__GIVEN__`` set) and the ones the shell established itself
(``__SET__``: the nmos/pmos type variable), and leaves every other default
to the emitter, which evaluates it from the .va text with those values in
place; the emitter fetches a runtime-callback parameter through the
shell's upper-case name; the generator refuses two parameters that differ
only in case with an error naming them (Xyce used to abort with
"Parameter TEMP already added to class ..."); a math library that did not
build or load is a device error, not a device that conducts nothing.

Every deck below runs end to end through Xyce with PYMS_DIR pointing at
the PyMS copy this file sits in and private PYMS_CACHE / PYMS_VAE_CACHE
under a scratch directory (as test_hdl_cache_va_identity.py does), one
Xyce run per test. Skips when there is no Xyce binary.

Run:
    cd utils/PyMS && python3 vae/tests/test_hdl_param_name_case.py
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

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
# The PyMS copy this test belongs to: <PyMS>/vae/tests/<this file>
PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEVICE_GEN = os.path.join(PYMS_DIR, "vae", "xyce_device_gen.py")

RTOL = 1e-6
I_1K = -1.0 / 1000.0            # the module default r = 1k (the symptom)
I_2K = -1.0 / 2000.0            # the card's / instance's r = 2k
I_4K = -1.0 / 4000.0            # the $param_given else branch 4*r, r = 1k
I_500 = -1.0 / 500.0            # r = 2000 / b with b = 4 given

INC = '`include "disciplines.vams"\n'


def _module(name, params, body):
    return (INC + "module %s(p, n);\n  inout p, n; electrical p, n;\n%s\n"
            "  analog %s\nendmodule\n" % (name, params, body))


DIVIDE = "I(p, n) <+ V(p, n) / r;"
PARAM_GIVEN = ("begin\n    if ($param_given(r)) I(p, n) <+ V(p, n) / r;\n"
               "    else I(p, n) <+ V(p, n) / (4.0 * r);\n  end")


def _ydeck(mod, card_params="", inst_params=""):
    """1 V across a Y device of module `mod` bound on .model rm."""
    return ("* %s\n.hdl \"%s.va\"\nv1 a 0 1\n.model rm %s %s\ny%s x1 a 0 rm %s\n"
            ".op\n.print dc i(v1)\n.end\n"
            % (mod, mod, mod, card_params, mod, inst_params))


def _prn_value(prn_text, col):
    lines = [ln for ln in prn_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise AssertionError("no data row in .prn")
    header = lines[0].split()
    if col not in header:
        raise AssertionError("column %s not in .prn header %r" % (col, header))
    return float(lines[1].split()[header.index(col)])


class TestParamNameCase(unittest.TestCase):
    """One Xyce run per test; see the module docstring."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(XYCE):
            raise unittest.SkipTest("no Xyce at %s" % XYCE)
        cls.work = tempfile.mkdtemp(prefix="hdlparamcase_")
        cls.env = dict(os.environ)
        cls.env["PYTHONDONTWRITEBYTECODE"] = "1"
        cls.env["PYMS_DIR"] = PYMS_DIR
        cls.env["PYMS_CACHE"] = os.path.join(cls.work, "hdl_cache")
        cls.env["PYMS_VAE_CACHE"] = os.path.join(cls.work, "vae_cache")
        cls.env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (
            ":" + cls.env["LD_LIBRARY_PATH"] if cls.env.get("LD_LIBRARY_PATH") else "")
        for k in ("VAE_SO_PATH", "VAE_SO_DIR", "PYMS_CALLBACK_PARAMS"):
            cls.env.pop(k, None)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("HDLTEST_KEEP"):
            sys.stderr.write("kept %s\n" % cls.work)
        else:
            shutil.rmtree(cls.work, ignore_errors=True)

    def _run(self, tag, mod, va_text, deck_text, extra_env=None):
        """Run one deck in its own directory; return (log, prn text or '', rc)."""
        d = os.path.join(self.work, tag)
        os.makedirs(d)
        with open(os.path.join(d, mod + ".va"), "w") as f:
            f.write(va_text)
        with open(os.path.join(d, "deck.cir"), "w") as f:
            f.write(deck_text)
        env = dict(self.env)
        env.update(extra_env or {})
        r = subprocess.run([XYCE, "deck.cir"], capture_output=True, text=True,
                           timeout=600, cwd=d, env=env)
        prn = os.path.join(d, "deck.cir.prn")
        prn_text = ""
        if os.path.exists(prn):
            with open(prn) as f:
                prn_text = f.read()
        return r.stdout + r.stderr, prn_text, r.returncode

    def _assert_current(self, tag, mod, va_text, deck_text, expected, extra_env=None):
        log, prn, rc = self._run(tag, mod, va_text, deck_text, extra_env)
        self.assertNotIn("Netlist error", log)
        self.assertTrue(prn, "no .prn written; Xyce rc=%d\n%s" % (rc, log[-3000:]))
        i = _prn_value(prn, "I(V1)")
        hint = ""
        if abs(i - I_1K) <= RTOL * abs(I_1K) and expected != I_1K:
            hint = " (the module computed with its default r = 1k)"
        self.assertTrue(abs(i - expected) <= RTOL * abs(expected),
                        "%s: I(V1) = %.9g, expected %.9g%s" % (tag, i, expected, hint))

    # --- the card / instance value reaches the math whatever the name's case

    def test_card_lower_case(self):
        """.model rm m1 r=2k on ``parameter real r``: I(V1) = -5.0e-4."""
        self._assert_current("card_lower", "pc1",
                             _module("pc1", "  parameter real r = 1k;", DIVIDE),
                             _ydeck("pc1", "r=2k"), I_2K)

    def test_instance_lower_case(self):
        """ypc2 x1 a 0 rm r=2k on a type="instance" parameter r."""
        self._assert_current("inst_lower", "pc2",
                             _module("pc2", '  (* type="instance" *) parameter real r = 1k;', DIVIDE),
                             _ydeck("pc2", "", "r=2k"), I_2K)

    def test_card_mixed_case(self):
        """.model rm m rs=2k on ``parameter real Rs`` (the netlist's case is free)."""
        self._assert_current("card_mixed", "pc3",
                             _module("pc3", "  parameter real Rs = 1k;", "I(p, n) <+ V(p, n) / Rs;"),
                             _ydeck("pc3", "rs=2k"), I_2K)

    def test_card_upper_case_still_right(self):
        """The form that always worked: .model rm m R=2k on ``parameter real R``."""
        self._assert_current("card_upper", "pc4",
                             _module("pc4", "  parameter real R = 1k;", "I(p, n) <+ V(p, n) / R;"),
                             _ydeck("pc4", "R=2k"), I_2K)

    # --- defaults are the emitter's, evaluated with the given values in place

    def test_dependent_default_follows_card_override(self):
        """``parameter real b = 2; parameter real r = 2000 / b;`` with b=4 on
        the card: r = 500 (the default is re-evaluated per instance)."""
        self._assert_current("dep_default_card", "pc5",
                             _module("pc5", "  parameter real b = 2;\n  parameter real r = 2000 / b;", DIVIDE),
                             _ydeck("pc5", "b=4"), I_500)

    def test_dependent_default_nothing_given(self):
        """``parameter real rr = 500; parameter real r = 2 * rr;`` nothing on the
        card: r = 1000 (the shell's own reading of the default, 0, must not land)."""
        self._assert_current("dep_default_nogiven", "pc6",
                             _module("pc6", "  parameter real rr = 500;\n  parameter real r = 2 * rr;", DIVIDE),
                             _ydeck("pc6", ""), I_1K)

    # --- $param_given and the runtime callback see the module's names

    def test_param_given_lower_case(self):
        """$param_given(r) is true with r=2k on the card (I = -5.0e-4) and
        false without it (the 4*r branch, I = -2.5e-4)."""
        va = _module("pc7", "  parameter real r = 1k;", PARAM_GIVEN)
        self._assert_current("param_given_given", "pc7", va, _ydeck("pc7", "r=2k"), I_2K)
        self._assert_current("param_given_notgiven", "pc7", va, _ydeck("pc7", ""), I_4K)

    def test_callback_param_lower_case(self):
        """PYMS_CALLBACK_PARAMS=r keeps r symbolic in the math library and the
        shell's callback supplies the card's 2k at eval time."""
        self._assert_current("callback_lower", "pc8",
                             _module("pc8", "  parameter real r = 1k;", DIVIDE),
                             _ydeck("pc8", "r=2k"), I_2K,
                             extra_env={"PYMS_CALLBACK_PARAMS": "r"})

    # --- what PyMS cannot do is a clear error, never a silent zero

    def test_case_collision_is_a_clear_error(self):
        """Two parameters differing only in case (EKV 2.6's TEMP / Temp) are
        refused by the generator with both names, instead of Xyce's abort
        'Parameter TEMP already added to class ...'."""
        d = os.path.join(self.work, "collision")
        os.makedirs(d)
        va = _module("pc9", "  parameter real TEMP = 500;\n  parameter real Temp = 500;",
                     "I(p, n) <+ V(p, n) / (TEMP + Temp);")
        with open(os.path.join(d, "pc9.va"), "w") as f:
            f.write(va)
        r = subprocess.run([sys.executable, DEVICE_GEN, "pc9.va", "--output", d],
                           capture_output=True, text=True, timeout=120, cwd=d, env=self.env)
        out = r.stdout + r.stderr
        self.assertNotEqual(r.returncode, 0, "the generator accepted TEMP and Temp:\n" + out)
        self.assertNotIn("Traceback", out)
        self.assertRegex(out, r"'TEMP' and 'Temp'.*differ only in case")
        self.assertIn("pc9", out)

    def test_failed_math_build_is_an_error(self):
        """A math library that cannot be built (PYMS_VAE_CACHE unwritable) is a
        device error from Xyce (nonzero exit, the module named), not a
        device that silently conducts nothing."""
        log, prn, rc = self._run("nobuild", "pc10",
                                 _module("pc10", "  parameter real r = 1k;", DIVIDE),
                                 _ydeck("pc10", "r=2k"),
                                 extra_env={"PYMS_VAE_CACHE": "/proc/pyms_no_such_cache"})
        self.assertNotEqual(rc, 0, "Xyce exited 0 without a math library:\n" + log[-3000:])
        self.assertRegex(log, r"PyMS: the Verilog-A math library for module pc10")
        if prn:
            self.assertFalse(re.search(r"^\s*0\s", prn, re.M) and "0.00000000e+00" in prn,
                             "a zero current was printed for a device without math")


if __name__ == "__main__":
    unittest.main(verbosity=2)
