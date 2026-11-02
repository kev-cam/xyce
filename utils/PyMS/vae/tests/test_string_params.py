"""String parameters (LRM 3.4.6): defaults, cards and compares.

THE DEFECT (PYMS1 skeptic, 2026-10-25; BUGS.md CHANGES 1.1.1): a ``parameter string
mode = "fast";`` became the number 0.0 in GiNaCEmitter.param_values, so
``if (mode == "fast")`` evaluated ``0.0 == "fast"`` -- silently false, the
else arm built -- and the device shell registered the parameter as a double,
so ``.model m1 s1 mode=slow`` was Xyce's "Cannot convert MODE to numeric".

THE FIX: the emitter keeps the texts apart (GiNaCEmitter._string_values) and
folds every compare of a string parameter with a string literal to 1 or 0 at
elaboration (_fold_string_compares; the ternary resolvers see the texts
through _cond_ns); any other use in a condition is an EmitError.  The shell
registers a string parameter as a std::string member with its default, so a
card can give it, and passes NAME="text" to build_vae_so.py, which bakes it
like any given value; the from '{...} value set is checked
(va_const.check_param_ranges).

1.1.2: == and != on strings in a parameter default (va_const.evaluate, and
to_cxx for the shell's per-instance re-evaluation over its std::string
members), and two string parameters compared in a condition.

Run:  cd utils/PyMS && python3 vae/tests/test_string_params.py
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
from vae.ginac_emitter import emit_ginac_program, EmitError  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'


def mod(name, body, rng=" from '{\"fast\", \"slow\"}"):
    return (HDR + "module %s(p, n);\n  inout p, n;\n  electrical p, n;\n"
            "  parameter string mode = \"fast\"%s;\n  parameter real r = 1k;\n  real g;\n"
            "  analog begin\n%s\n  end\nendmodule\n" % (name, rng, body))


S1 = mod("s1", "    if (mode == \"fast\") g = 1 / r; else g = 1 / (3 * r);\n    I(p, n) <+ V(p, n) * g;")
# the reversed literal order and != in a ternary
S2 = mod("s2", "    g = (\"slow\" != mode) ? 1 / r : 1 / (3 * r);\n    I(p, n) <+ V(p, n) * g;")
# a compare beside a voltage in one condition
S3 = mod("s3", "    if (mode == \"fast\" && V(p, n) > 0.25) g = 1 / r; else g = 1 / (3 * r);\n"
         "    I(p, n) <+ V(p, n) * g;")

# 1.1.2 (PYMS2 skeptic, 2026-10-26): a string compare in a PARAMETER DEFAULT was
# refused ("an operand of == is a string where a number is needed"), and a
# compare of two string parameters (1.1.0: right) was a library build failure.
S5 = (HDR + "module s5(p, n);\n  inout p, n;\n  electrical p, n;\n"
      "  parameter string mode = \"fast\";\n  parameter string kind = \"fast\";\n"
      "  analog begin\n    if (mode == kind) I(p, n) <+ V(p, n) / 1000;\n"
      "    else I(p, n) <+ V(p, n) / 3000;\n  end\nendmodule\n")
S6 = (HDR + "module s6(p, n);\n  inout p, n;\n  electrical p, n;\n"
      "  parameter string mode = \"fast\";\n  parameter real r = (mode == \"slow\") ? 3k : 1k;\n"
      "  parameter integer k = (\"fast\" != mode);\n"
      "  analog I(p, n) <+ V(p, n) / (r * (1 + k));\nendmodule\n")


class TestEmitter(unittest.TestCase):

    def _f(self, va, pv):
        src = emit_ginac_program(parse_verilog_a(va, filename="s.va"), param_values=pv)
        return [l.strip() for l in src.split('\n') if 'F_contribs[br_idx] +=' in l]

    def test_default_and_given(self):
        self.assertIn('0.001', self._f(S1, {})[0])
        self.assertIn('0.000333333', self._f(S1, {'mode': 'slow'})[0])
        self.assertIn('0.001', self._f(S2, {})[0])
        self.assertIn('0.000333333', self._f(S2, {'mode': 'slow'})[0])

    def test_string_compare_in_a_default(self):
        from vae.va_const import evaluate_param_defaults, ConstExprError
        m = parse_verilog_a(S6, filename="s6.va")
        self.assertEqual(evaluate_param_defaults(m), {"mode": "fast", "r": 1000.0, "k": 0})
        self.assertEqual(evaluate_param_defaults(m, {"mode": "slow"}), {"mode": "slow", "r": 3000.0, "k": 1})
        bad = parse_verilog_a(S6.replace('(mode == "slow")', '(mode == 1)'), filename="s6.va")
        with self.assertRaisesRegex(ConstExprError, "compares a string with a number"):
            evaluate_param_defaults(bad)

    def test_two_string_parameters(self):
        self.assertIn('/ 1000;', self._f(S5, {})[0])
        self.assertIn('/ 3000;', self._f(S5, {'kind': 'slow'})[0])

    def test_other_use_is_an_error(self):
        bad = mod("s4", "    if (mode) g = 1 / r; else g = 1 / (3 * r);\n    I(p, n) <+ V(p, n) * g;", "")
        with self.assertRaisesRegex(EmitError, "string parameter mode"):
            emit_ginac_program(parse_verilog_a(bad, filename="s.va"), param_values={})


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def _run(self, va, name, card):
        work = tempfile.mkdtemp(prefix="pyms_str_")
        try:
            env = dict(os.environ)
            env.update(PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(work, "hc"),
                       PYMS_VAE_CACHE=os.path.join(work, "vc"), PYTHONDONTWRITEBYTECODE="1")
            env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            with open(os.path.join(work, "s.va"), "w") as f:
                f.write(va)
            with open(os.path.join(work, "d.cir"), "w") as f:
                f.write('* s\n.hdl "s.va"\nV1 in 0 DC 1\n.model m1 %s %s\nY%s X1 in out m1\nR2 out 0 1k\n'
                        '.op\n.print dc v(out)\n.end\n' % (name, card, name))
            r = subprocess.run([XYCE, "d.cir"], cwd=work, env=env, capture_output=True, text=True, timeout=900)
            got = None
            try:
                with open(os.path.join(work, "d.cir.prn")) as f:
                    got = float(f.read().split("\n")[1].split()[1])
            except (OSError, IndexError, ValueError):
                pass
            return r.returncode, r.stdout + r.stderr, got
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_values(self):
        bad = []
        for va, name, card, want in ((S1, "s1", "", 0.5), (S1, "s1", "mode=slow", 0.25),
                                     (S1, "s1", "mode=fast", 0.5), (S2, "s2", "mode=slow", 0.25),
                                     (S3, "s3", "", 0.5), (S3, "s3", "mode=slow", 0.25),
                                     (S5, "s5", "", 0.5), (S5, "s5", "kind=slow", 0.25),
                                     (S6, "s6", "", 0.5), (S6, "s6", "mode=slow", 1 / 7)):
            rc, log, got = self._run(va, name, card)
            print("%s %-10s want %g got %s rc %d" % (name, card, want, got, rc), flush=True)
            if rc != 0 or got is None or abs(got - want) > 1e-6:
                bad.append((name, card, want, got, rc, log[-800:]))
        self.assertEqual(bad, [])

    def test_value_outside_its_set(self):
        rc, log, got = self._run(S1, "s1", "mode=medium")
        self.assertNotEqual(rc, 0, log[-1500:])
        self.assertIn("parameter mode of module s1 = 'medium' is outside its range", log)


if __name__ == "__main__":
    unittest.main(verbosity=2)
