"""The GiNaC metaprogram prints the same eval code in every run.

THE DEFECT (BUGS.md D3, PYMS3 skeptic, 2026-10-26): GiNaC orders the terms of
a sum or product by hash values seeded from the address of each class's type
name (GiNaC hash_seed.h), so with address-space randomization one compiled
metaprogram printed its eval code in a different term order in every run:
Mextram bjtd504's eval.cpp differed by ~1800 lines between two builds of the
same source and its DC operating point converged in 2 runs of 4 (1.1.2).

THE FIX: the metaprogram re-executes itself once with randomization off
(personality ADDR_NO_RANDOMIZE), so the hash seeds, the term order and the
eval code are the same in every run.

Run:  cd utils/PyMS && python3 vae/tests/test_ginac_determinism.py
"""
from __future__ import annotations

import hashlib
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
from vae.ginac_emitter import emit_ginac_program  # noqa: E402
from vae.tests._decks import mod  # noqa: E402

# A diode-ish module with a few sums and products of node voltages and
# intermediates: enough terms that their order shows the hash seeds.
VA = mod("det", "  parameter real is = 1e-14;\n  parameter real nf = 1.5;\n  parameter real rs = 10;\n"
         "  parameter real cj = 1p;\n  electrical x;\n"
         "  real vd, id, q;\n  analog begin\n    vd = V(x, n);\n"
         "    id = is * (limexp(vd / (nf * $vt)) - 1) + 1e-12 * vd + 3e-3 * vd * vd * V(p, x);\n"
         "    q = cj * vd + 2 * cj * vd * vd + cj * V(p, n) * vd;\n"
         "    I(p, x) <+ V(p, x) / rs + 1e-3 * V(p, x) * V(x, n);\n"
         "    I(x, n) <+ id + ddt(q);\n  end")


def _have_ginac():
    return shutil.which("g++") is not None and os.path.exists("/usr/include/ginac/ginac.h")


@unittest.skipUnless(_have_ginac(), "no g++ / GiNaC")
class TestDeterminism(unittest.TestCase):

    def test_same_eval_code_every_run(self):
        work = tempfile.mkdtemp(prefix="pyms_det_")
        try:
            src = os.path.join(work, "g.cpp")
            exe = os.path.join(work, "g")
            open(src, "w").write("#define CONSTCtoK 273.15\n"
                                 + emit_ginac_program(parse_verilog_a(VA, filename="c.va")))
            subprocess.run(["g++", "-O1", "-std=c++17", "-o", exe, src, "-lginac", "-lcln"],
                           check=True, capture_output=True, timeout=900)
            outs = set()
            for _ in range(6):
                r = subprocess.run([exe], capture_output=True, text=True, timeout=600)
                self.assertEqual(r.returncode, 0, r.stderr[-500:])
                outs.add(hashlib.sha1(r.stdout.encode()).hexdigest())
            self.assertEqual(len(outs), 1, "the eval code differs between runs")
            # 1.1.4: a PYMS_GINAC_NOASLR the environment brings along is not
            # taken for the re-execution's marker (the PYMS4 skeptic: preset,
            # the program went on randomized, silently)
            env = dict(os.environ, PYMS_GINAC_NOASLR='1')
            preset = set()
            for _ in range(6):
                r = subprocess.run([exe], capture_output=True, text=True, timeout=600, env=env)
                self.assertEqual(r.returncode, 0, r.stderr[-500:])
                preset.add(hashlib.sha1(r.stdout.encode()).hexdigest())
            self.assertEqual(preset, outs, "PYMS_GINAC_NOASLR preset: the eval code differs")
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_warns_when_randomization_stays(self):
        """personality() refused (a seccomp filter) or ignored (a sandbox):
        the program says so on stderr ("PyMS warning: ... term order"), which
        build_vae_so.py passes on; before 1.1.4 it went on silently."""
        if shutil.which("gcc") is None:
            self.skipTest("no gcc")
        work = tempfile.mkdtemp(prefix="pyms_det2_")
        try:
            src = os.path.join(work, "g.cpp")
            exe = os.path.join(work, "g")
            open(src, "w").write("#define CONSTCtoK 273.15\n"
                                 + emit_ginac_program(parse_verilog_a(VA, filename="c.va")))
            subprocess.run(["g++", "-O1", "-std=c++17", "-o", exe, src, "-lginac", "-lcln"],
                           check=True, capture_output=True, timeout=900)
            shims = {
                "refused": "#include <errno.h>\nint personality(unsigned long p) { (void)p; errno = EPERM; return -1; }\n",
                "ignored": "int personality(unsigned long p) { (void)p; return 0; }\n",
            }
            for name, code in shims.items():
                c = os.path.join(work, name + ".c")
                so = os.path.join(work, name + ".so")
                open(c, "w").write(code)
                subprocess.run(["gcc", "-shared", "-fPIC", "-o", so, c], check=True, capture_output=True)
                r = subprocess.run([exe], capture_output=True, text=True, timeout=600,
                                   env=dict(os.environ, LD_PRELOAD=so))
                self.assertEqual(r.returncode, 0, r.stderr[-500:])
                self.assertIn("PyMS warning", r.stderr, name)
                self.assertIn("term order", r.stderr, name)
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
