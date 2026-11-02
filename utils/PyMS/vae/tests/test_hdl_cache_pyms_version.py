"""The device-shell cache keys on the PyMS that generated the shell.

THE DEFECT (PYMS2 skeptic, 2026-10-26; BUGS.md CHANGES 1.1.2): N_DEV_PyMS.C
named the cache slot of a device shell (PYMS_CACHE/pyms_<module>_<hash>.so)
by the .va identity alone (path, content, PYMS_DEFINES).  Xyce through PyMS
1.1.0 and then through 1.1.1 with one PYMS_CACHE reused the 1.1.0 shell with
a 1.1.1 math library: 0 where 0.5 is right, exit status 0.

THE FIX: the slot is keyed by the .va identity AND the generated shell source
(.C and .h; the .C carries the PyMS version in its math-library key).  A
shell from another PyMS is never reused; an unchanged setup still hits.

The test runs one deck twice through one PYMS_CACHE: first through a copy of
this PyMS whose version reads 0.0.0+cachetest, then through this PyMS.  It
asserts two shell slots, each shell source naming its own version, and the
analytic value from both runs; and a third run (this PyMS again) reuses its
slot.  Needs the Xyce library with the fix (XYCE= selects the binary).

Run:  cd utils/PyMS && python3 vae/tests/test_hdl_cache_pyms_version.py
"""
from __future__ import annotations

import glob
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

import vae  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)

VA = """`include "disciplines.vams"
module cachever(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  analog I(p, n) <+ V(p, n) / r;
endmodule
"""
DECK = """* shell cache vs PyMS version
.hdl "cachever.va"
V1 in 0 DC 1
.model m1 cachever
Ycachever X1 in out m1
R2 out 0 1k
.op
.print dc v(out)
.end
"""
OTHER = "0.0.0+cachetest"


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestShellCacheVersion(unittest.TestCase):

    def test_other_pyms_never_reuses_the_shell(self):
        work = tempfile.mkdtemp(prefix="pyms_cachever_")
        try:
            other = os.path.join(work, "PyMS.other")
            shutil.copytree(_PYMS_DIR, other, ignore=shutil.ignore_patterns("__pycache__", "tests"))
            init = os.path.join(other, "vae", "__init__.py")
            text = open(init).read()
            text, n = re.subn(r"__version__ = '[^']*'", "__version__ = '%s'" % OTHER, text)
            self.assertEqual(n, 1)
            open(init, "w").write(text)
            open(os.path.join(work, "cachever.va"), "w").write(VA)
            open(os.path.join(work, "d.cir"), "w").write(DECK)
            hc = os.path.join(work, "hc")

            def run(pyms_dir):
                env = dict(os.environ)
                env.update(PYMS_DIR=pyms_dir, PYMS_CACHE=hc, PYMS_VAE_CACHE=os.path.join(work, "vc"),
                           PYTHONDONTWRITEBYTECODE="1")
                env.pop("PYMS_DEFINES", None)
                env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
                r = subprocess.run([XYCE, "d.cir"], cwd=work, env=env, capture_output=True, text=True, timeout=900)
                got = None
                prn = os.path.join(work, "d.cir.prn")
                try:
                    got = float([l.split() for l in open(prn)][1][1])
                    os.unlink(prn)
                except (OSError, IndexError, ValueError):
                    pass
                return r.returncode, got, (r.stdout + r.stderr)[-800:]

            for pyms_dir in (other, _PYMS_DIR):
                rc, got, log = run(pyms_dir)
                self.assertEqual(rc, 0, log)
                self.assertAlmostEqual(got, 0.5, delta=1e-6, msg=log)
            shells = sorted(glob.glob(os.path.join(hc, "pyms_cachever_*.so")))
            self.assertEqual(len(shells), 2, shells)
            versions = sorted(re.search(r"\|PyMS-([^\"]+)\"", open(s[:-3] + ".C").read()).group(1) for s in shells)
            self.assertEqual(versions, sorted([OTHER, vae.__version__]))
            # an unchanged setup reuses its slot
            before = {s: os.stat(s).st_mtime_ns for s in shells}
            rc, got, log = run(_PYMS_DIR)
            self.assertEqual(rc, 0, log)
            self.assertAlmostEqual(got, 0.5, delta=1e-6, msg=log)
            self.assertEqual({s: os.stat(s).st_mtime_ns for s in
                              glob.glob(os.path.join(hc, "pyms_cachever_*.so"))}, before)
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
