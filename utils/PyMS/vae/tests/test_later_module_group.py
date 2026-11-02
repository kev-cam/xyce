"""A later module's own xyceModelGroup registers (1.1.6, B21).

THE DEFECT (PYMS6 skeptic, 2026-11-01, on 1.1.5): 1.1.5 stopped a later
module's attributes from landing on the file's first module (B21,
m5_second_has_group_attr) by registering every later module as a Y device
only: a netlist using the later module through its own model group (``D1 in
out md`` with ``.model md d level=2001`` for the second module dd of a1.va)
stopped with "Model is required for device D1 and no valid model card
found" (m6b_dd_second; 1.1.4 ran a1 there, silently: 0.5 for 0.2).

THE FIX: xyce_device_gen.py -- a later module reads the attributes of its
own declaration (no sibling file lends it one) and registers its group and
level, unless they are the first module's (then a Y device of its own name,
with a note).

Run:  cd utils/PyMS && python3 vae/tests/test_later_module_group.py
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

from vae.tests._decks import HDR, XYCE, XYCE_LIBDIR, rv  # noqa: E402

A1 = "module a1(p, n);\n  inout p, n;\n  electrical p, n;\n  analog I(p, n) <+ V(p, n) / 1000;\nendmodule\n"
MID = "module mid(p, n);\n  inout p, n;\n  electrical p, n;\n  analog I(p, n) <+ V(p, n) / 2000;\nendmodule\n"


def dd(name, r, level="2001"):
    return ('(* xyceModelGroup="Diode" xyceLevelNumber="%s" *)\nmodule %s(p, n);\n  inout p, n;\n'
            '  electrical p, n;\n  analog I(p, n) <+ V(p, n) / %d;\nendmodule\n' % (level, name, r))


def deck(dev):
    return "* t\n.hdl \"a1.va\"\nV1 in 0 DC 1\n%s\nR2 out 0 1k\n.op\n.print dc v(out)\n.end\n" % dev


Y_A1 = ".model m1 a1\nYa1 X1 in out m1"
D_DD = ".model md d level=2001\nD1 in out md"
# name -> (file, deck, want, log text that must appear)
CASES = {
    "second_used_by_group": (HDR + A1 + dd("dd", 4000), deck(D_DD), rv(4000), None),
    "second_first_y": (HDR + A1 + dd("dd", 4000), deck(Y_A1), 0.5, None),
    "second_both": (HDR + A1 + dd("dd", 4000), deck(Y_A1 + "\n" + D_DD.replace("D1 in out", "D1 out 0")),
                    800.0 / 1800.0, None),
    "third_used_by_group": (HDR + A1 + MID + dd("dd", 4000), deck(D_DD), rv(4000), None),
    "group_of_the_first": (HDR + dd("d1", 4000) + dd("d2", 2000), deck(D_DD), rv(4000),
                           "of the file's first module d1"),
    "group_of_the_first_y": (HDR + dd("d1", 4000) + dd("d2", 2000), deck(".model m2 d2\nYd2 X1 in out m2"),
                             rv(2000), None),
}


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        work = tempfile.mkdtemp(prefix='pyms_lmg_')
        env = dict(os.environ, PYMS_DIR=_PYMS_DIR, PYTHONDONTWRITEBYTECODE='1')
        env['LD_LIBRARY_PATH'] = XYCE_LIBDIR + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        bad = []
        try:
            for name, (va, text, want, logwant) in CASES.items():
                d = os.path.join(work, name)
                os.makedirs(d)
                with open(os.path.join(d, 'a1.va'), 'w') as fh:
                    fh.write(va)
                with open(os.path.join(d, 'c.cir'), 'w') as fh:
                    fh.write(text)
                env.update(PYMS_CACHE=os.path.join(d, 'hc'), PYMS_VAE_CACHE=os.path.join(d, 'vc'))
                r = subprocess.run([XYCE, 'c.cir'], cwd=d, env=env, capture_output=True, text=True, timeout=1800)
                log = r.stdout + r.stderr
                got = None
                try:
                    with open(os.path.join(d, 'c.cir.prn')) as fh:
                        rows = [l.split() for l in fh.read().splitlines() if l.strip()]
                    got = float(rows[1][[c.upper() for c in rows[0]].index('V(OUT)')])
                except (OSError, IndexError, ValueError):
                    pass
                ok = r.returncode == 0 and got is not None and abs(got - want) <= 1e-6
                if logwant is not None:
                    os.makedirs(os.path.join(d, 'gen'), exist_ok=True)
                    # the generator's note reaches the .hdl output only on a
                    # rebuild of the shell; read the generator's own output
                    gen = subprocess.run([sys.executable, os.path.join(_PYMS_DIR, 'vae', 'xyce_device_gen.py'),
                                          os.path.join(d, 'a1.va'), '--output', os.path.join(d, 'gen')],
                                         env=env, capture_output=True, text=True, timeout=600)
                    ok = ok and logwant in gen.stdout + gen.stderr
                print("%-24s want %s got %s rc %d" % (name, want, got, r.returncode), flush=True)
                if not ok:
                    bad.append("%s: want %s got %s rc %d %s" % (name, want, got, r.returncode, log[-600:]))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(bad, [], "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
