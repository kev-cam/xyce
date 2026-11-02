"""Branch currents as unknowns: a current read in a Verilog-A expression.

THE DEFECT (vamos spectre guide trials, round 5 item EI-1 and round 6 item 1;
PYMS_BUGS.md P6): PyMS built a device that contributes nothing for every
module that reads a branch current.  ``V(p, n) <+ r * I(p, n)`` left the text
``I(p, n)`` in the GiNaC metaprogram, g++ failed ("'p' was not declared in
this scope"), the math library never existed and Xyce ran the device as an
open circuit: V(out) = 0 with exit status 0 (the reference divider gives
0.5).  ``I(br)`` of a named branch was rewritten to ex(0) in
ginac_emitter._to_ginac_expr, so ``V(br) <+ r * I(br)`` was a 0 V source
(1.001 on the divider), and a voltage contribution was stamped as a Norton
source of 1 S with the sign of a negative resistance, so ``V(p, n) <+ 0.5``
gave 0.5005 and two voltage contributions on one branch 0.75 instead of 0.5.

THE FIX: the current of every voltage-contribution branch and of every probed
branch is an unknown of the device (MNA).  GiNaCEmitter.collect_current_unknowns
lists them from the whole analog block; the device shell (xyce_device_gen.py)
adds one internal solution variable per entry with the branch current in the
two KCL rows, and the math library forms the branch equation in its own row:
sum(e) - (V_a - V_b) = 0 for voltage contributions (and a probed branch with
no contribution: a short), i - sum(e) - d/dt(charge) = 0 for current
contributions; vae_n_currents / vae_current_label / vae_current_row export
the rows and the shell matches them by label.  A port current probe I(<p>)
is a clear error (CurrentProbeError) at .hdl time.

This module has two halves: generator-level tests (no Xyce) on the emitted
metaprogram and device shell, and end-to-end decks through Xyce (skipped
when there is no Xyce binary): the dc forms, the inductor ``V(p, n) <+
ddt(l * I(p, n))`` in ac (|H| = 1/sqrt(2) where omega L = R) and in tran (the
RL step, 1 - exp(-t/tau)).  Every value is compared with the analytic one.

Run:
    cd utils/PyMS && python3 vae/tests/test_current_probes.py
$XYCE names the binary (default /usr/local/src/xyce-build/src/Xyce); the
PyMS copy under test is the one this file sits in (PYMS_DIR is set to it).
"""
from __future__ import annotations

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

from vae.parser import parse_verilog_a  # noqa: E402
from vae.ginac_emitter import GiNaCEmitter, CurrentProbeError, emit_ginac_program, VAE_STATE_DECL  # noqa: E402
from vae.xyce_device_gen import generate_device_cpp  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
RTOL = 1e-6

HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'


def module(name, body, extra="", ports=("p", "n")):
    pl = ", ".join(ports)
    return (HDR + f"module {name}({pl});\n  inout {pl};\n  electrical {pl};\n"
            f"  parameter real r = 1k;\n{extra}  {body}\nendmodule\n")


# name -> (module text, analytic V(out) on the divider V1=1 V, R2=1k)
DC_MODULES = {
    "m_vfi":  (module("m_vfi", "analog V(p, n) <+ r * I(p, n);"), 0.5),
    "m_vbr":  (module("m_vbr", "analog V(br) <+ r * I(br);", "  branch (p, n) br;\n"), 0.5),
    "m_isel": (module("m_isel", "analog I(p, n) <+ V(p, n) / r + 0 * I(p, n);"), 0.5),
    "m_var":  (module("m_var", "analog begin\n    cur = I(p, n);\n    V(p, n) <+ r * cur;\n  end",
                      "  real cur;\n"), 0.5),
    "m_half": (module("m_half", "analog V(p, n) <+ r * I(p, n) * 0.5;"), 2.0 / 3.0),
    "m_ser":  (module("m_ser", "analog begin\n    V(p, m) <+ r * I(p, m);\n    I(m, n) <+ V(m, n) / r;\n  end",
                      "  electrical m;\n"), 1.0 / 3.0),
    "m_vc":   (module("m_vc", "analog V(p, n) <+ 0.5;"), 0.5),
    "m_two":  (module("m_two", "analog begin\n    V(p, n) <+ 0.25;\n    V(p, n) <+ 0.25;\n  end"), 0.5),
    "m_rev":  (module("m_rev", "analog V(p, n) <+ -r * I(n, p);"), 0.5),
    "m_ind":  (module("m_ind", "analog V(p, n) <+ ddt(l * I(p, n));", "  parameter real l = 1m;\n"), 1.0),
    # the static part and a ddt() term in one voltage contribution (D's split)
    "m_mix":  (module("m_mix", "analog V(p, n) <+ r * I(p, n) + ddt(0.0);"), 0.5),
    # the series RL: V = r*i + l*di/dt (a short plus r at dc)
    "m_rl":   (module("m_rl", "analog V(p, n) <+ r * I(p, n) + ddt(l * I(p, n));",
                      "  parameter real l = 1m;\n"), 0.5),
    # the named branch's current read in another contribution of that branch:
    # i = V/(2r) + i/2, so i = V/r
    "m_brrd": (module("m_brrd", "analog begin\n    I(br) <+ V(br) / (2 * r);\n"
                      "    I(br) <+ 0.5 * I(br);\n  end", "  branch (p, n) br;\n"), 0.5),
    # a gated voltage contribution that is not executed: an open branch (LRM)
    "m_gate": (module("m_gate", "analog if (V(p, n) > 2) V(p, n) <+ 0.5;"), 0.0),
    "m_gate_on": (module("m_gate_on", "analog if (V(p, n) < 2) V(p, n) <+ 0.5;"), 0.5),
    # a switch branch (exclusive arms of one voltage-dependent condition):
    # the V arm (r, V(p,n) = 0.5) is the consistent one for vt = 0.3, the I
    # arm (2r, V(p,n) = 2/3) for vt = 0.8
    "m_sw1":  (module("m_sw1", "analog begin\n    if (V(p, n) > 0.3) V(p, n) <+ r * I(p, n);\n"
                      "    else I(p, n) <+ V(p, n) / (2 * r);\n  end"), 0.5),
    "m_sw2":  (module("m_sw2", "analog begin\n    if (V(p, n) > 0.8) V(p, n) <+ r * I(p, n);\n"
                      "    else I(p, n) <+ V(p, n) / (2 * r);\n  end"), 1.0 / 3.0),
    # both kinds on one path: the kind contributed last decides (the LRM's
    # switch branch; VACASK agrees): a noise voltage source after a current
    # contribution is a 0 V source at dc, a current after a voltage wins
    "m_nzv":  (module("m_nzv", "analog begin\n    I(p, n) <+ V(p, n) / r;\n"
                      "    V(p, n) <+ white_noise(1e-20, \"x\");\n  end"), 1.0),
    "m_lastv": (module("m_lastv", "analog begin\n    I(p, n) <+ V(p, n) / r;\n    V(p, n) <+ 0.25;\n  end"), 0.75),
    "m_lasti": (module("m_lasti", "analog begin\n    V(p, n) <+ 0.25;\n    I(p, n) <+ V(p, n) / r;\n  end"), 0.5),
    "m_nzv0": (module("m_nzv0", "analog V(p, n) <+ white_noise(1e-20, \"x\");"), 1.0),
    "m_pi":   (module("m_pi", "analog begin\n    I(p, m) <+ V(p, m) / r;\n    I(m, n) <+ V(m, n) / r;\n"
                      "    I(p, n) <+ V(p, n) / r;\n  end", "  electrical m;\n"), 0.6),
}
# a current-controlled current source: the probe of one branch drives another
M_CCCS = module("m_cccs", "analog begin\n    I(p, n) <+ V(p, n) / r;\n    I(c, n) <+ 2 * I(p, n);\n  end",
                ports=("p", "n", "c"))
# a current-controlled current source through a NAMED branch
M_CCCS_BR = module("m_cccs_br", "analog begin\n    I(br) <+ V(br) / r;\n    I(c, n) <+ 2 * I(br);\n  end",
                   "  branch (p, n) br;\n", ports=("p", "n", "c"))
M_IND = DC_MODULES["m_ind"][0]
M_VFI = DC_MODULES["m_vfi"][0]
M_RL = DC_MODULES["m_rl"][0]
M_PORT = module("m_port", "analog V(p, n) <+ r * I(<p>);")


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
    return env


def _read_prn(path):
    """The columns and rows of a Xyce .prn: (header names, list of float rows)."""
    hdr, rows = None, []
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s:
                continue
            if s[0] == "Index":
                hdr = s
                continue
            if s[0].startswith("End"):
                break
            rows.append([float(x) for x in s])
    return hdr, rows


def _run_xyce(work, deck_text):
    deck = os.path.join(work, "deck.cir")
    with open(deck, "w") as f:
        f.write(deck_text)
    for stale in (deck + ".prn", deck + ".FD.prn"):   # outputs of an earlier deck here
        if os.path.exists(stale):
            os.remove(stale)
    r = subprocess.run([XYCE, "deck.cir"], capture_output=True, text=True,
                       timeout=900, cwd=work, env=_xyce_env(work))
    log = r.stdout + r.stderr
    prn = deck + ".prn"
    if not os.path.exists(prn) and os.path.exists(deck + ".FD.prn"):
        prn = deck + ".FD.prn"          # an .AC analysis prints here
    if not os.path.exists(prn):
        raise AssertionError("Xyce wrote no .prn (exit %d):\n%s" % (r.returncode, log[-3000:]))
    return _read_prn(prn), log


class TestGenerator(unittest.TestCase):
    """The emitter and the device shell, without Xyce."""

    def _records(self, va):
        mod = parse_verilog_a(va, filename="t.va")
        em = GiNaCEmitter(mod, param_values={})
        return em.collect_current_unknowns([p.name for p in mod.ports] + list(mod.internal_nodes))

    def test_voltage_contribution_and_probe_share_one_unknown(self):
        recs = self._records(DC_MODULES["m_vfi"][0])
        self.assertEqual([r['label'] for r in recs], ['BR(p,n)'])

    def test_named_branch_is_one_unknown(self):
        recs = self._records(DC_MODULES["m_vbr"][0])
        self.assertEqual([r['label'] for r in recs], ['BR(br)'])
        self.assertEqual((recs[0]['a'], recs[0]['b']), ('p', 'n'))

    def test_probed_current_branch_is_an_unknown(self):
        recs = self._records(DC_MODULES["m_isel"][0])
        self.assertEqual([r['label'] for r in recs], ['BR(p,n)'])

    def test_plain_divider_has_no_unknown(self):
        recs = self._records(module("m_plain", "analog I(p, n) <+ V(p, n) / r;"))
        self.assertEqual(recs, [])

    def test_probe_in_string_is_not_a_probe(self):
        recs = self._records(module("m_str", 'analog begin\n    $strobe("I(p, n) = %g", 1.0);\n'
                                             '    I(p, n) <+ V(p, n) / r;\n  end'))
        self.assertEqual(recs, [])

    def test_metaprogram_has_no_raw_probe_text(self):
        """The old failure: I(p, n) left verbatim in an expression for g++.
        Only the expression statements are checked (a branch label such as
        I(m,n) legitimately appears in comments and label strings)."""
        probe = re.compile(r'\bI\s*\(\s*\w+\s*(,\s*\w+\s*)?\)')
        for name, (va, _v) in DC_MODULES.items():
            mod = parse_verilog_a(va, filename=name + ".va")
            prog = emit_ginac_program(mod, param_values={})
            for line in prog.splitlines():
                if ('_contribs[br_idx]' in line or 'int_exprs.push_back' in line
                        or 'int_raw[' in line):
                    self.assertIsNone(probe.search(line),
                                      "%s: a raw current probe survives in: %s" % (name, line.strip()))
            self.assertIn('vae_n_currents', prog)

    def test_reversed_pair_reads_the_same_unknown_negated(self):
        mod = parse_verilog_a(DC_MODULES["m_rev"][0], filename="m_rev.va")
        prog = emit_ginac_program(mod, param_values={})
        self.assertIn('(-Ibr0_p_n)', prog)
        self.assertEqual(prog.count('symbol Ibr'), 1)

    def test_shell_allocates_the_unknowns(self):
        mod = parse_verilog_a(DC_MODULES["m_ser"][0], filename="m_ser.va")
        header, impl = generate_device_cpp(mod, '', va_path='m_ser.va')
        self.assertIn('static const int numCurrents = 1;', header)
        self.assertIn('numIntVars = 2;', impl)            # the internal node m and the current
        self.assertIn('"BR(p,m)"', impl)
        self.assertIn(VAE_STATE_DECL, header)
        self.assertIn('double I[', VAE_STATE_DECL)

    def test_port_current_probe_is_a_clear_error(self):
        mod = parse_verilog_a(M_PORT, filename="m_port.va")
        with self.assertRaises(CurrentProbeError) as cm:
            generate_device_cpp(mod, '', va_path='m_port.va')
        msg = str(cm.exception)
        self.assertIn('m_port.va:', msg)
        self.assertIn('I(<p>)', msg.replace(' ', ''))
        self.assertIn('port current probe', msg)


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce at %s" % XYCE)
class TestXyce(unittest.TestCase):
    """End to end through Xyce: every module of DC_MODULES in one deck (one
    divider each), the three-port CCCS, and the inductor in ac and tran."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="pyms_curprobe_")

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("HDLTEST_KEEP"):
            shutil.rmtree(cls.work, ignore_errors=True)

    def _write_modules(self, items):
        for name, va in items:
            with open(os.path.join(self.work, name + ".va"), "w") as f:
                f.write(va)

    def test_dc_forms(self):
        names = list(DC_MODULES)
        self._write_modules((n, DC_MODULES[n][0]) for n in names)
        lines = ["* current reads: every dc form, one divider each"]
        for i, n in enumerate(names):
            lines += ['.hdl "%s.va"' % n, "V%d in%d 0 DC 1.0" % (i, i),
                      ".model %s__m %s" % (n, n), "y%s X%d in%d out%d %s__m" % (n, i, i, i, n),
                      "R%d out%d 0 1000.0" % (i, i)]
        lines += [".options device temp=27.0 tnom=27.0 gmin=1e-12", ".OP",
                  ".PRINT DC " + " ".join("V(out%d)" % i for i in range(len(names))), ".end"]
        (hdr, rows), log = _run_xyce(self.work, "\n".join(lines) + "\n")
        self.assertEqual(len(rows), 1, log[-2000:])
        for i, n in enumerate(names):
            want = DC_MODULES[n][1]
            got = rows[0][1 + i]
            with self.subTest(module=n):
                self.assertAlmostEqual(got, want, delta=RTOL * abs(want),
                                       msg="%s: V(out)=%.12g, analytic %.12g" % (n, got, want))

    def test_cccs_three_port(self):
        self._write_modules([("m_cccs", M_CCCS)])
        deck = ("* a probe of one branch drives another\n"
                '.hdl "m_cccs.va"\nV1 in 0 DC 1.0\n.model m_cccs__m m_cccs\n'
                "ym_cccs X1 in out ctl m_cccs__m\nR2 out 0 1000.0\nR3 ctl 0 1000.0\n"
                ".options device temp=27.0 tnom=27.0 gmin=1e-12\n.OP\n.PRINT DC V(out) V(ctl)\n.end\n")
        (hdr, rows), log = _run_xyce(self.work, deck)
        self.assertEqual(len(rows), 1, log[-2000:])
        out, ctl = rows[0][1], rows[0][2]
        self.assertAlmostEqual(out, 0.75, delta=RTOL * 0.75, msg="V(out)=%.12g" % out)
        self.assertAlmostEqual(ctl, -0.5, delta=RTOL * 0.5, msg="V(ctl)=%.12g" % ctl)

    def test_cccs_named_branch(self):
        self._write_modules([("m_cccs_br", M_CCCS_BR)])
        deck = ("* I(br) of a named branch drives another branch\n"
                '.hdl "m_cccs_br.va"\nV1 in 0 DC 1.0\n.model m_cccs_br__m m_cccs_br\n'
                "ym_cccs_br X1 in out ctl m_cccs_br__m\nR2 out 0 1000.0\nR3 ctl 0 1000.0\n"
                ".options device temp=27.0 tnom=27.0 gmin=1e-12\n.OP\n.PRINT DC V(out) V(ctl)\n.end\n")
        (hdr, rows), log = _run_xyce(self.work, deck)
        self.assertEqual(len(rows), 1, log[-2000:])
        out, ctl = rows[0][1], rows[0][2]
        self.assertAlmostEqual(out, 0.75, delta=RTOL * 0.75, msg="V(out)=%.12g" % out)
        self.assertAlmostEqual(ctl, -0.5, delta=RTOL * 0.5, msg="V(ctl)=%.12g" % ctl)

    def test_two_instances_with_different_elaborations(self):
        """Two instances of one module whose parameter selects a voltage or a
        current contribution load two different math libraries (the library
        is keyed on the instance's parameters and culls the dead arm); each
        instance must map ITS library's rows: mode=1 is r (V <+ r*I), out 0.5;
        mode=0 is 2r (I <+ V/(2r)), out 1/3."""
        va = module("m_mode", "analog begin\n    if (mode == 1) V(p, n) <+ r * I(p, n);\n"
                    "    else I(p, n) <+ V(p, n) / (2 * r);\n  end", "  parameter integer mode = 0;\n")
        self._write_modules([("m_mode", va)])
        deck = ("* one module, two elaborations\n"
                '.hdl "m_mode.va"\nV1 in 0 DC 1.0\n.model m_mode1 m_mode mode=1\n.model m_mode0 m_mode mode=0\n'
                "ym_mode X1 in out1 m_mode1\nR2 out1 0 1000.0\n"
                "ym_mode X2 in out2 m_mode0\nR3 out2 0 1000.0\n"
                ".options device temp=27.0 tnom=27.0 gmin=1e-12\n.OP\n.PRINT DC V(out1) V(out2)\n.end\n")
        (hdr, rows), log = _run_xyce(self.work, deck)
        self.assertEqual(len(rows), 1, log[-2000:])
        o1, o2 = rows[0][1], rows[0][2]
        self.assertAlmostEqual(o1, 0.5, delta=RTOL * 0.5, msg="V(out1)=%.12g" % o1)
        self.assertAlmostEqual(o2, 1.0 / 3.0, delta=RTOL / 3.0, msg="V(out2)=%.12g" % o2)

    def test_series_resistor_ac(self):
        """V(p,n) <+ r*I(p,n) with R2 = 1k: the small-signal gain is 0.5 at
        every frequency; the series RL V <+ r*I + ddt(l*I) at omega = 2e6:
        H = R2 / (r + R2 + j omega l) = 1/(2+2j), |H| = 0.353553, -45 deg."""
        self._write_modules([("m_vfi", M_VFI), ("m_rl", M_RL)])
        f0 = 2e6 / (2 * math.pi)
        deck = ("* the series resistor and RL as voltage contributions, ac\n"
                '.hdl "m_vfi.va"\n.hdl "m_rl.va"\nV1 in 0 DC 0 AC 1.0\n'
                ".model m_vfi__m m_vfi\n.model m_rl__m m_rl\n"
                "ym_vfi X1 in out1 m_vfi__m\nR2 out1 0 1000.0\n"
                "ym_rl X2 in out2 m_rl__m\nR3 out2 0 1000.0\n"
                ".options device temp=27.0 tnom=27.0 gmin=1e-12\n"
                ".AC LIN 1 %.9g %.9g\n.PRINT AC VM(out1) VP(out1) VM(out2) VP(out2)\n.end\n" % (f0, f0))
        (hdr, rows), log = _run_xyce(self.work, deck)
        self.assertEqual(len(rows), 1, log[-2000:])
        vm1, vp1, vm2, vp2 = rows[0][2:6]
        self.assertAlmostEqual(vm1, 0.5, delta=RTOL, msg="VM(out1)=%.12g" % vm1)
        self.assertAlmostEqual(vp1, 0.0, delta=1e-6, msg="VP(out1)=%.12g" % vp1)
        self.assertAlmostEqual(vm2, 1 / (2 * math.sqrt(2)), delta=RTOL, msg="VM(out2)=%.12g" % vm2)
        self.assertAlmostEqual(vp2, -45.0, delta=1e-4, msg="VP(out2)=%.12g" % vp2)

    def test_series_resistor_tran(self):
        """V(p,n) <+ r*I(p,n) driven by a 1 kHz sine: V(out) = V(in)/2 at
        every time point."""
        self._write_modules([("m_vfi", M_VFI)])
        deck = ("* the series resistor as a voltage contribution, tran\n"
                '.hdl "m_vfi.va"\nV1 in 0 SIN(0 1 1k)\n.model m_vfi__m m_vfi\n'
                "ym_vfi X1 in out m_vfi__m\nR2 out 0 1000.0\n"
                ".options device temp=27.0 tnom=27.0 gmin=1e-12\n"
                ".TRAN 10u 2m\n.PRINT TRAN V(in) V(out)\n.end\n")
        (hdr, rows), log = _run_xyce(self.work, deck)
        self.assertGreater(len(rows), 50, log[-2000:])
        worst = max(abs(r[3] - 0.5 * r[2]) for r in rows)
        self.assertLess(worst, 1e-6, "max |V(out) - V(in)/2| = %.3g" % worst)

    def test_inductor_ac(self):
        """V(p,n) <+ ddt(l*I(p,n)) with R2 = 1k at omega = R/L = 1e6 rad/s:
        |V(out)/V(in)| = 1/sqrt(2), phase -45 degrees."""
        self._write_modules([("m_ind", M_IND)])
        f0 = 1e6 / (2 * math.pi)
        deck = ("* the inductor written as a voltage contribution, ac\n"
                '.hdl "m_ind.va"\nV1 in 0 DC 0 AC 1.0\n.model m_ind__m m_ind\n'
                "ym_ind X1 in out m_ind__m\nR2 out 0 1000.0\n"
                ".options device temp=27.0 tnom=27.0 gmin=1e-12\n"
                ".AC LIN 1 %.9g %.9g\n.PRINT AC VM(out) VP(out)\n.end\n" % (f0, f0))
        (hdr, rows), log = _run_xyce(self.work, deck)
        self.assertEqual(len(rows), 1, log[-2000:])
        vm, vp = rows[0][2], rows[0][3]
        self.assertAlmostEqual(vm, 1 / math.sqrt(2), delta=RTOL, msg="VM(out)=%.12g" % vm)
        self.assertAlmostEqual(vp, -45.0, delta=1e-4, msg="VP(out)=%.12g" % vp)

    def test_inductor_tran(self):
        """The RL step: V(out) = 1 - exp(-t/tau), tau = L/R = 1 us."""
        self._write_modules([("m_ind", M_IND)])
        deck = ("* the inductor written as a voltage contribution, tran\n"
                '.hdl "m_ind.va"\nV1 in 0 PULSE(0 1 0 1p 1p 1 2)\n.model m_ind__m m_ind\n'
                "ym_ind X1 in out m_ind__m\nR2 out 0 1000.0\n"
                ".options device temp=27.0 tnom=27.0 gmin=1e-12\n"
                ".options timeint reltol=1e-5 abstol=1e-12\n"
                ".TRAN 1n 2u\n.options output initial_interval=10n\n.PRINT TRAN V(out)\n.end\n")
        (hdr, rows), log = _run_xyce(self.work, deck)
        self.assertGreater(len(rows), 50, log[-2000:])
        ts = [r[1] for r in rows]
        vo = [r[2] for r in rows]
        for t_want in (0.5e-6, 1e-6, 2e-6):
            i = min(range(len(ts)), key=lambda k: abs(ts[k] - t_want))
            t = ts[i]
            want = 1 - math.exp(-t / 1e-6)
            with self.subTest(t=t):
                self.assertLess(abs(vo[i] - want), 1e-3,
                                "V(out)(%.4g)=%.6g, analytic %.6g" % (t, vo[i], want))


if __name__ == '__main__':
    unittest.main(verbosity=2)
