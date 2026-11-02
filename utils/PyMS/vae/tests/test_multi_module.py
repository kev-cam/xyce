"""A .va file with several modules is several devices (1.1.4).

THE DEFECT (PYMS4 skeptic, g4_two_modules_ground_leak, 2026-10-27): the
generator built only the file's first module and N_DEV_PyMS.C registered
only the first "Module:" line, so a netlist naming the second module of the
file stopped with an unknown device.

THE FIX: xyce_device_gen.py writes a device shell for every module of the
file (the first as before; a further one as a Y device of its own name, its
math library built with build_vae_so.py --module) and prints one "Module:"
line each; N_DEV_PyMS.C compiles and registers every one.  A further module
PyMS cannot build is a note, not an error of the file.

Run:  cd utils/PyMS && python3 vae/tests/test_multi_module.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import shutil
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a  # noqa: E402
from vae.tests._decks import mod, run_decks, XYCE, HDR  # noqa: E402

SUB = "module sub(a);\n  inout a;\n  electrical a, g;\n  ground g;\n  analog I(a, g) <+ V(a, g) / 4000;\nendmodule\n"


def two(name, first_sub=True):
    body = mod(name, "  analog I(p, n) <+ V(p, n) / 1000;")
    return (HDR + SUB + body[len(HDR):]) if first_sub else (body + SUB)


GEN = os.path.join(_PYMS_DIR, 'vae', 'xyce_device_gen.py')


class TestParse(unittest.TestCase):

    def test_select_by_name(self):
        va = two("second")
        self.assertEqual(parse_verilog_a(va, filename="c.va").name, "sub")
        self.assertEqual(parse_verilog_a(va, filename="c.va", module="second").name, "second")

    def test_generator_writes_every_module(self):
        work = tempfile.mkdtemp(prefix="pyms_mm_")
        try:
            path = os.path.join(work, "c.va")
            with open(path, "w") as fh:
                fh.write(two("second"))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            r = subprocess.run([sys.executable, GEN, path, "--output", work],
                               capture_output=True, text=True, timeout=300, env=env)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            names = [l.split(',')[0][len('Module: '):] for l in r.stdout.splitlines()
                     if l.startswith('Module: ')]
            self.assertEqual(names, ['sub', 'second'])
            self.assertTrue(os.path.exists(os.path.join(work, 'N_DEV_PYMS_SECOND.C')))
            with open(os.path.join(work, 'N_DEV_PYMS_SECOND.C')) as fh:
                self.assertIn("--module 'second'", fh.read())
        finally:
            shutil.rmtree(work, ignore_errors=True)


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_second_module_runs(self):
        self.assertEqual(run_decks({"second": (two("second"), "", 0.5),
                                    "firstmod": (two("firstmod", first_sub=False), "", 0.5)},
                                   "pyms_mm_x_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
