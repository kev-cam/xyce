"""The device temperature reaches a module (1.1.6, BUGS.md A17).

THE DEFECT (PYMS6 skeptic, 2026-11-01, on 1.1.5 and every earlier version):
the circuit temperature never reached a PyMS device.  The math library's
wrapper had ``static const double temperature = 300.15`` and the device
shell set ``state.Vt = 8.617087e-5 * 300.15``: at ``.options device
temp=127`` or ``.step temp`` a module computing V / (1000 * $temperature /
300.15) stayed at 0.5 (analytic 0.428602) while the built-ins followed the
temperature.  Every CMC model of the corpus reads $temperature.  Besides: a
parameter default reading $temperature was refused; a module variable named
Vt or temperature captured $vt / $temperature; $vt(x) stopped at the first
')' of x; a module parameter named TNOM or TEMP given on a card took
Xyce's Celsius-to-kelvin conversion (+273.15), and Xyce's device manager
wrote the circuit temperature into a module's TEMP at a .step temp (and
baked it: a library per temperature).

THE FIX: the instance's temperature is a run-time value.  The device shell
reads it at every evaluation through the instance's temperature source (a
pointer; today the instance's Xyce temperature, XYCEADMSINSTTEMP, which
Xyce's device manager sets at every temperature change) into the state
(VaeState::T); the eval code reads $temperature from it and computes $vt
with Xyce's k/q; a parameter default that reads them is computed in the
eval code's prologue; nothing temperature-dependent is in the library or
its cache key, so a .step temp builds one library.  A module's own TEMP /
TNOM are its own (no conversion, the device manager's write undone).

Each deck: the module X1 between in (V1 = 1 V dc, or dc 0 ac 1) and out,
R2 = 1k to ground; a module conducting k / 1000 gives V(out) = k / (k + 1).

Run:  cd utils/PyMS && python3 vae/tests/test_temperature.py
"""
from __future__ import annotations

import glob
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.tests._decks import mod, XYCE, XYCE_LIBDIR  # noqa: E402

T27, T127 = 300.15, 400.15


def kv(k):
    return k / (k + 1.0)


A127 = kv(T27 / T127)          # 0.428602...

# a conductance k/1000: I(p, n) <+ V(p, n) * (k) / 1000
def gmod(k_expr, pre="", decl=""):
    return mod("a1", decl + "  analog begin\n" + pre + "    I(p, n) <+ V(p, n) * (" + k_expr + ") / 1000;\n  end")


def run(va, card="", lines=".op\n.print dc v(out)", src="DC 1", inst=""):
    """(rc, values of V(out) per .step point (op) or VM(OUT) (ac), the math
    libraries built, log)"""
    work = tempfile.mkdtemp(prefix="pyms_temp_")
    env = dict(os.environ)
    env.update(PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(work, "hc"),
               PYMS_VAE_CACHE=os.path.join(work, "vc"), PYTHONDONTWRITEBYTECODE="1")
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    try:
        with open(os.path.join(work, "c.va"), "w") as fh:
            fh.write(va)
        with open(os.path.join(work, "c.cir"), "w") as fh:
            fh.write("* temperature\n.hdl \"c.va\"\nV1 in 0 %s\n.model m1 a1 %s\nYa1 X1 in out m1 %s\n"
                     "R2 out 0 1k\n%s\n.end\n" % (src, card, inst, lines))
        r = subprocess.run([XYCE, "c.cir"], cwd=work, env=env, capture_output=True, text=True,
                           timeout=1800)
        log = r.stdout + r.stderr
        ac = ".ac" in lines
        path = os.path.join(work, "c.cir.FD.prn" if ac else "c.cir.prn")
        vals = []
        try:
            with open(path) as fh:
                rows = [l.split() for l in fh.read().splitlines() if l.strip()]
            k = [c.upper() for c in rows[0]].index("VM(OUT)" if ac else "V(OUT)")
            for c in rows[1:]:
                if c[0].lower().startswith("end") or c[0].upper() == "INDEX":
                    continue
                vals.append(float(c[k]))
        except (OSError, IndexError, ValueError):
            pass
        libs = [x for x in glob.glob(os.path.join(work, "vc", "vae_*.so"))]
        return r.returncode, vals, len(libs), log
    finally:
        shutil.rmtree(work, ignore_errors=True)


class TestTemperature(unittest.TestCase):

    def check(self, va, lines, want, card="", src="DC 1", libs=None, tol=1e-6, inst=""):
        rc, got, nlib, log = run(va, card, lines, src, inst)
        print("want %s got %s libs %d" % (want, got, nlib), flush=True)
        self.assertEqual(rc, 0, log[-1500:])
        self.assertEqual(len(got), len(want), (got, log[-1500:]))
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, delta=tol * max(1.0, abs(w)), msg=(got, want))
        if libs is not None:
            self.assertEqual(nlib, libs, "math libraries built")

    # $temperature in the analog block
    def test_options_temp(self):
        va = gmod("300.15 / $temperature")
        self.check(va, ".op\n.print dc v(out)", [0.5])
        self.check(va, ".options device temp=127\n.op\n.print dc v(out)", [A127])

    def test_step_temp_one_library(self):
        va = gmod("300.15 / $temperature")
        self.check(va, ".step temp list 27 127 -23\n.op\n.print dc v(out)",
                   [0.5, A127, kv(T27 / 250.15)], libs=1)

    # $vt and $vt(T): kT/q at the instance temperature, Xyce's k/q
    def test_vt_and_vt_of_t(self):
        va = gmod("$vt(300.15) / $vt")
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [0.5, A127], libs=1)
        va = gmod("$vt($temperature) / $vt(T0)", decl="  parameter real T0 = 300.15;\n")
        self.check(va, ".options device temp=127\n.op\n.print dc v(out)", [kv(T127 / T27)])
        # the argument up to its own closing parenthesis (1.1.5: the first ')')
        va = gmod("$vt((300.15)) / $vt")
        self.check(va, ".options device temp=127\n.op\n.print dc v(out)", [A127])

    def test_kq_is_xyce_s(self):
        from vae.va_const import VAE_KOQ
        self.assertEqual(float(VAE_KOQ), 1.3806226e-23 / 1.6021918e-19)
        hdr = os.path.join(_PYMS_DIR, '..', '..', 'src', 'DeviceModelPKG', 'Core', 'N_DEV_Const.h')
        if os.path.exists(hdr):
            txt = open(hdr).read()
            kb = float(re.search(r'#define\s+CONSTboltz\s+\(([^)]+)\)', txt).group(1))
            q = float(re.search(r'#define\s+CONSTQ\s+\(([^)]+)\)', txt).group(1))
            self.assertEqual(float(VAE_KOQ), kb / q)

    # a module variable named Vt or temperature is the module's own
    def test_module_variables_named_vt_temperature(self):
        va = gmod("Vt / (2 * $vt)", pre="    Vt = 2 * $vt;\n", decl="  real Vt;\n")
        self.check(va, ".options device temp=127\n.op\n.print dc v(out)", [0.5])
        va = gmod("300.15 / temperature", pre="    temperature = $temperature;\n", decl="  real temperature;\n")
        self.check(va, ".options device temp=127\n.op\n.print dc v(out)", [A127])

    # parameter defaults that read the temperature: computed at run time
    def test_param_default_reads_temperature(self):
        va = gmod("300.15 / tr", decl="  parameter real tr = $temperature;\n")
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [0.5, A127], libs=1)
        # a chain of them, and one through $vt
        va = gmod("300.15 / t2", decl="  parameter real t1 = $temperature - 273.15;\n"
                                     "  parameter real t2 = t1 + 273.15;\n")
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [0.5, A127], libs=1)
        va = gmod("vr / $vt(300.15)", decl="  parameter real vr = $vt * 300.15 / $temperature * 300.15 / $temperature;\n")
        self.check(va, ".options device temp=127\n.op\n.print dc v(out)", [A127])
        # given on the card: the card's value
        va = gmod("300.15 / tr", decl="  parameter real tr = $temperature;\n")
        self.check(va, ".options device temp=127\n.op\n.print dc v(out)", [0.5], card="tr=300.15")
        # in a condition
        va = gmod("(tr > 350) ? 300.15 / tr : 1", decl="  parameter real tr = $temperature;\n")
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [0.5, A127])

    # initial_step / initial_model values derived from the temperature
    def test_initial_blocks_follow_the_temperature(self):
        va = gmod("300.15 / tk", pre="    @(initial_step) tk = $temperature;\n", decl="  real tk;\n")
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [0.5, A127], libs=1)
        # held: an @(initial_step) that reads a voltage
        va = gmod("300.15 / tk", pre="    @(initial_step) begin\n      if (V(p, n) > -100) tk = $temperature; else tk = 1;\n    end\n",
                  decl="  real tk;\n")
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [0.5, A127], libs=1)
        va = gmod("300.15 / tk", pre="    @(initial_model) tk = $temperature;\n", decl="  real tk;\n")
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [0.5, A127], libs=1)

    # 1.1.7 (BUGS.md A18): a .dc temp sweep moves the temperature between the
    # points of one analysis; a value captured in @(initial_step) keeps the
    # first point's temperature (1.1.6 folded the capture into the eval code
    # as `tk = _vae_T` and it followed the sweep), a direct read follows it
    def test_initial_step_capture_held_under_dc_temp(self):
        sweep = ".dc temp 27 127 100\n.print dc v(out)"
        self.check(gmod("300.15 / $temperature"), sweep, [0.5, A127], libs=1)
        va = gmod("300.15 / tk", pre="    @(initial_step) tk = $temperature;\n", decl="  real tk;\n")
        self.check(va, sweep, [0.5, 0.5], libs=1)
        # through $vt and through a thermal parameter
        va = gmod("$vt(300.15) / vk", pre="    @(initial_step) vk = $vt;\n", decl="  real vk;\n")
        self.check(va, sweep, [0.5, 0.5], libs=1)
        va = gmod("300.15 / tk", pre="    @(initial_step) tk = tr;\n",
                  decl="  parameter real tr = $temperature;\n  real tk;\n")
        self.check(va, sweep, [0.5, 0.5], libs=1)
        # a later .step point is a new analysis: captured again
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [0.5, A127], libs=1)

    # ac: the operating point's temperature, and the slope there
    def test_ac_at_temperature(self):
        # I = 1e-3 * V^2 * T / 300.15 from (1 V dc, ac 1) to out, 1k to ground
        def gain(T):
            a = T / 300.15
            lo, hi = 0.0, 1.0           # (1 - Vo)^2 * a = Vo
            for _ in range(200):
                m = 0.5 * (lo + hi)
                if (1 - m) ** 2 * a - m > 0:
                    lo = m
                else:
                    hi = m
            vo = 0.5 * (lo + hi)
            g = 2e-3 * a * (1 - vo)
            return g / (g + 1e-3)
        va = mod("a1", "  analog I(p, n) <+ 1e-3 * V(p, n) * V(p, n) * $temperature / 300.15;")
        self.check(va, ".step temp list 27 127\n.ac dec 1 1k 1k\n.print ac vm(out)",
                   [gain(T27), gain(T127)], src="DC 1 AC 1", libs=1)

    # the instance's own temperature, as Xyce's built-in devices have it:
    # TEMP (C) is the device temperature, else the circuit's + DTEMP
    def test_instance_temp_dtemp(self):
        va = gmod("300.15 / $temperature")
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [A127, A127], inst="temp=127", libs=1)
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [A127, kv(T27 / (T127 + 100))],
                   inst="dtemp=100", libs=1)
        self.check(va, ".options device temp=127\n.op\n.print dc v(out)", [0.5], inst="temp=27")

    # a module's own TEMP / TNOM: as the netlist gives them, never Xyce's
    def test_module_temp_tnom_are_its_own(self):
        va = gmod("(tnom + 273.15) / 300.15", decl="  parameter real tnom = 27;\n")
        self.check(va, ".op\n.print dc v(out)", [0.5], card="tnom=27")
        va = mod("a1", "  (*type=\"instance\"*) parameter real temp = 27;\n"
                       "  analog I(p, n) <+ V(p, n) * ((temp + 273.15) / 300.15) / 1000;")
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [0.5, 0.5], libs=1)
        self.check(va, ".step temp list 27 127\n.op\n.print dc v(out)", [kv(T127 / T27)] * 2,
                   inst="temp=127", libs=1)


if __name__ == "__main__":
    unittest.main()
