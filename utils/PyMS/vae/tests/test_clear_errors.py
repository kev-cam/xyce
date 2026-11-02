"""One-line file:line errors for three forms that reached Python or g++.

THE DEFECTS (PYMS1 skeptic, 2026-10-25; BUGS.md CHANGES 1.1.1):
- a port current probe ``I(<p>)``: CurrentProbeError (a ParseError) escaped
  xyce_device_gen.py's main, which caught only ValueError and
  ConstExprError: a Python traceback under ".HDL: PyMS compilation failed";
- ``if (ddt(V(p, n)) > 0)``: the condition was lowered to C++ with ddt in
  it and the metaprogram's g++ said "'ddt' was not declared in this scope";
- an undeclared identifier (``V(p, n) / rzz``) went to g++ the same way.

THE FIX: main catches ParseError and EmitError too; ginac_emitter
.check_conditions refuses an analog operator in a condition with the
statement's file and line at .hdl time; build_vae_so.py turns g++'s "'X' was
not declared in this scope" into one line naming X and the file:line of the
first statement that reads it (ginac_emitter.find_identifier_use).

Run:  cd utils/PyMS && python3 vae/tests/test_clear_errors.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GEN = os.path.join(_PYMS_DIR, 'vae', 'xyce_device_gen.py')
XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'


def mod(body):
    return HDR + "module a1(p, n);\n  inout p, n;\n  electrical p, n;\n%s\nendmodule\n" % body


PORT = mod("  analog I(p, n) <+ V(p, n) / 1000 + 0 * I(<p>);")
DDTC = mod("  analog begin\n    if (ddt(V(p, n)) > 0) I(p, n) <+ V(p, n) / 1k;\n"
           "    else I(p, n) <+ V(p, n) / 1k;\n  end")
UNDECL = mod("  analog I(p, n) <+ V(p, n) / rzz;")


class TestGenerator(unittest.TestCase):

    def _gen(self, va):
        d = tempfile.mkdtemp(prefix="pyms_err_")
        try:
            with open(os.path.join(d, "a1.va"), "w") as f:
                f.write(va)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            r = subprocess.run([sys.executable, GEN, os.path.join(d, "a1.va"), "--output", d],
                               capture_output=True, text=True, env=env, timeout=300)
            return r.returncode, (r.stdout + r.stderr).strip()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_port_probe(self):
        rc, out = self._gen(PORT)
        self.assertNotEqual(rc, 0)
        self.assertNotIn("Traceback", out)
        self.assertEqual(len(out.splitlines()), 1, out)
        self.assertRegex(out, r"a1\.va:6: .*port current probe")

    def test_ddt_in_condition(self):
        rc, out = self._gen(DDTC)
        self.assertNotEqual(rc, 0)
        self.assertEqual(len(out.splitlines()), 1, out)
        self.assertRegex(out, r"a1\.va:7: ddt\(\) in the condition")


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_undeclared_identifier(self):
        work = tempfile.mkdtemp(prefix="pyms_err_")
        try:
            env = dict(os.environ)
            env.update(PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(work, "hc"),
                       PYMS_VAE_CACHE=os.path.join(work, "vc"), PYTHONDONTWRITEBYTECODE="1")
            env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            with open(os.path.join(work, "a1.va"), "w") as f:
                f.write(UNDECL)
            with open(os.path.join(work, "d.cir"), "w") as f:
                f.write('* u\n.hdl "a1.va"\nV1 in 0 DC 1\n.model m1 a1\nYa1 X1 in out m1\nR2 out 0 1k\n'
                        '.op\n.print dc v(out)\n.end\n')
            r = subprocess.run([XYCE, "d.cir"], cwd=work, env=env, capture_output=True, text=True, timeout=900)
            log = r.stdout + r.stderr
            self.assertNotEqual(r.returncode, 0, log[-1500:])
            self.assertRegex(log, r"build_vae_so: error: \S*a1\.va:6: 'rzz' is not declared in module a1")
            self.assertNotIn("was not declared in this scope", log)
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
