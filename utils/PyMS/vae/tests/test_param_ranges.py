"""Parameter value ranges: ``from`` and ``exclude`` (LRM 3.4.2).

THE DEFECT (PYMS1 skeptic, 2026-10-25; BUGS.md CHANGES 1.1.1): the parser read the
``from`` range into Param.from_range and threw ``exclude`` away, and nothing
ever checked either: ``parameter real r = 1k from (0:inf)`` with ``r=-5`` on
the card (or a default of -1) built and ran a negative resistor, rc 0.

THE FIX: vae/parser.py keeps every range of a declaration
(Param.value_ranges); vae/va_const.check_param_ranges enforces them on the
instance's final values (the given ones over the defaults evaluated with
them) with the LRM's semantics -- with from ranges the value must lie in one,
in no exclude range and on no excluded value; ( ) open, [ ] closed, inf
unbounded; a string parameter's '{...} set -- and build_vae_so.py refuses
the math library with one line naming the parameter, its value and the
range, which the device shell turns into an error of the device.

Run:  cd utils/PyMS && python3 vae/tests/test_param_ranges.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a  # noqa: E402
from vae import va_const  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'


def mod(decls, name="rg"):
    return (HDR + "module %s(p, n);\n  inout p, n;\n  electrical p, n;\n%s"
            "  analog I(p, n) <+ V(p, n) / r;\nendmodule\n" % (name, decls))


DECL = ("  parameter real r = 1k from (0:inf);\n"
        "  parameter integer k = 1 from [1:3] exclude 2;\n"
        "  parameter real t = 0.5 from [0:1) exclude (0.2:0.3];\n")


class TestCheck(unittest.TestCase):

    def _check(self, given, decl=DECL):
        m = parse_verilog_a(mod(decl), filename="rg.va")
        vals = va_const.evaluate_param_defaults(m, given)
        va_const.check_param_ranges(m, vals)

    def test_defaults_pass(self):
        self._check({})

    def test_open_lower_end(self):
        with self.assertRaises(va_const.ParamRangeError) as cm:
            self._check({"r": 0.0})
        msg = str(cm.exception)
        for part in ("rg.va:6", "parameter r", "= 0 ", "from (0:inf)"):
            self.assertIn(part, msg)

    def test_negative_card_value(self):
        with self.assertRaisesRegex(va_const.ParamRangeError, r"parameter r of module rg = -5 is outside"):
            self._check({"r": -5.0})

    def test_closed_and_half_open(self):
        self._check({"t": 0.0})
        self._check({"t": 0.2})
        with self.assertRaises(va_const.ParamRangeError):
            self._check({"t": 1.0})
        with self.assertRaisesRegex(va_const.ParamRangeError, r"excluded by exclude \(0\.2:0\.3\]"):
            self._check({"t": 0.3})

    def test_excluded_value(self):
        self._check({"k": 3})
        with self.assertRaisesRegex(va_const.ParamRangeError, r"parameter k of module rg = 2 is excluded by exclude 2"):
            self._check({"k": 2})
        with self.assertRaises(va_const.ParamRangeError):
            self._check({"k": 4})

    def test_default_outside_its_range(self):
        with self.assertRaisesRegex(va_const.ParamRangeError, "parameter r of module rg = -1 is outside"):
            self._check({}, "  parameter real r = -1 from (0:inf);\n")

    def test_range_reads_another_parameter(self):
        d = "  parameter real rmax = 2k;\n  parameter real r = 1k from (0:rmax];\n"
        self._check({}, d)
        self._check({"r": 2000.0}, d)
        with self.assertRaises(va_const.ParamRangeError):
            self._check({"r": 2001.0}, d)

    def test_string_value_set(self):
        d = '  parameter string mode = "fast" from \'{"fast", "slow"};\n  parameter real r = 1k;\n'
        self._check({}, d)
        self._check({"mode": "slow"}, d)
        with self.assertRaisesRegex(va_const.ParamRangeError, "parameter mode of module rg = 'medium'"):
            self._check({"mode": "medium"}, d)


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def _run(self, va, card):
        work = tempfile.mkdtemp(prefix="pyms_rng_")
        try:
            env = dict(os.environ)
            env.update(PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(work, "hc"),
                       PYMS_VAE_CACHE=os.path.join(work, "vc"), PYTHONDONTWRITEBYTECODE="1")
            env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            open(os.path.join(work, "rg.va"), "w").write(va)
            open(os.path.join(work, "d.cir"), "w").write(
                '* rg\n.hdl "rg.va"\nV1 in 0 DC 1\n.model m1 rg %s\nYrg X1 in out m1\nR2 out 0 1k\n'
                '.op\n.print dc v(out)\n.end\n' % card)
            r = subprocess.run([XYCE, "d.cir"], cwd=work, env=env, capture_output=True, text=True, timeout=900)
            got = None
            try:
                got = float(open(os.path.join(work, "d.cir.prn")).read().split("\n")[1].split()[1])
            except (OSError, IndexError, ValueError):
                pass
            return r.returncode, r.stdout + r.stderr, got
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_card_in_range_runs(self):
        rc, log, got = self._run(mod(DECL), "r=2k")
        self.assertEqual(rc, 0, log[-1500:])
        self.assertAlmostEqual(got, 1.0 / 3.0, places=6)

    def test_card_out_of_range_is_an_error(self):
        rc, log, got = self._run(mod(DECL), "r=-5")
        self.assertNotEqual(rc, 0, log[-1500:])
        self.assertIn("parameter r of module rg = -5 is outside its range from (0:inf)", log)

    def test_card_excluded_is_an_error(self):
        rc, log, got = self._run(mod(DECL), "k=2")
        self.assertNotEqual(rc, 0, log[-1500:])
        self.assertIn("parameter k of module rg = 2 is excluded by exclude 2", log)

    def test_default_out_of_range_is_an_error(self):
        rc, log, got = self._run(mod("  parameter real r = -1 from (0:inf);\n"), "")
        self.assertNotEqual(rc, 0, log[-1500:])
        self.assertIn("parameter r of module rg = -1 is outside its range from (0:inf)", log)


if __name__ == "__main__":
    unittest.main(verbosity=2)
