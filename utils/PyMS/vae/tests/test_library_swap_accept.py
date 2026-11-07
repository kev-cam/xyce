"""The shell drops vae_needs_accept with the math library it came from (1.1.7).

THE DEFECT (PYMS7 code review F1, 2026-11-07, on 1.1.6; present since the
1.1.5 library swap): the generated Instance::acceptStep cached
dlsym(vae_dl_, "vae_needs_accept") in a function-static pointer keyed on the
dlopen handle. A .step of a baked parameter makes processParams dlclose the
old math library and dlopen the new one; glibc hands back the same handle
(the link_map is reused) while the code lies elsewhere, so the cached pointer
was stale: SIGSEGV (1.1.6) or an "Illegal instruction" (1.1.5) at the second
step point of a module with held state in a transient.

THE FIX: xyce_device_gen.py -- the pointer is a member of the instance
(pyms_needs_accept_), read together with the held state when a library is
loaded and cleared when the library is dropped; no function-static cache.

Run:  cd utils/PyMS && python3 vae/tests/test_library_swap_accept.py
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.tests._decks import mod, XYCE, XYCE_LIBDIR  # noqa: E402

# a held peak (pyms_nheld_ > 0) and, for mode=1, a much larger library so the
# second library maps its code at other addresses than the first
BIG = ("1e-9*sin(V(p,n))*cos(V(p,n))*exp(-V(p,n)*V(p,n)) + 1e-9*tanh(V(p,n)*3)*sinh(V(p,n))"
       " + 1e-9*atan(V(p,n))*ln(1+V(p,n)*V(p,n)) + 1e-9*sqrt(1+V(p,n)*V(p,n))*cosh(V(p,n)/7)")
STP = mod("stp", "  parameter real mode = 0;\n  parameter real r = 1k;\n  real pk;\n  analog begin\n"
                 "    @(initial_step) pk = 0;\n    if (V(p,n) > pk) pk = V(p,n);\n    if (mode > 0.5)\n"
                 "      I(p,n) <+ V(p,n)/r + 1e-4*pk + " + BIG + ";\n    else\n"
                 "      I(p,n) <+ V(p,n)/r + 1e-4*pk;\n  end")
TIMES = [0.1e-3, 0.6e-3, 1.25e-3, 1.9e-3]


def segments(path):
    with open(path) as fh:
        rows = [l.split() for l in fh.read().splitlines() if l.strip()]
    hdr = [c.upper() for c in rows[0]]
    tk, vk = hdr.index("TIME"), hdr.index("V(OUT)")
    segs, last = [], None
    for c in rows[1:]:
        try:
            t, v = float(c[tk]), float(c[vk])
        except (ValueError, IndexError):
            continue
        if last is None or t < last:
            segs.append([])
        segs[-1].append((t, v))
        last = t
    return segs


def interp(rows, t):
    for (t0, v0), (t1, v1) in zip(rows, rows[1:]):
        if t0 <= t <= t1:
            return v0 if t1 == t0 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return None


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_step_baked_param_tran(self):
        work = tempfile.mkdtemp(prefix='pyms_swap_')
        env = dict(os.environ, PYMS_DIR=_PYMS_DIR, PYTHONDONTWRITEBYTECODE='1',
                   PYMS_CACHE=os.path.join(work, 'hc'), PYMS_VAE_CACHE=os.path.join(work, 'vc'))
        env['LD_LIBRARY_PATH'] = XYCE_LIBDIR + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        try:
            with open(os.path.join(work, 'stp.va'), 'w') as fh:
                fh.write(STP)
            with open(os.path.join(work, 'c.cir'), 'w') as fh:
                fh.write('* library swap\n.hdl "stp.va"\nV1 in 0 SIN(0 1 1k)\n.model m1 stp mode=0\nYstp X1 in out m1\n'
                         'R2 out 0 1k\n.step m1:mode list 0 1 0\n'
                         '.tran 10u 2m\n.print tran v(out)\n.end\n')
            r = subprocess.run([XYCE, 'c.cir'], cwd=work, env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-800:])
            segs = segments(os.path.join(work, 'c.cir.prn'))
            self.assertEqual(len(segs), 3)
            got = [[interp(s, t) for t in TIMES] for s in segs]
            print('step values', got, flush=True)
            for g in got:
                self.assertNotIn(None, g)
            # mode=0 again after the mode=1 library: the same waveform as the first point
            for a, b in zip(got[0], got[2]):
                self.assertAlmostEqual(a, b, delta=1e-6)
            # two libraries built (mode 0 and mode 1)
            self.assertEqual(len(glob.glob(os.path.join(work, 'vc', '**', 'vae_*.so'), recursive=True)), 2)
            # no function-static dlsym cache left in the generated shell
            shells = [p for p in glob.glob(os.path.join(work, 'hc', '**', '*.C'), recursive=True)]
            self.assertTrue(shells)
            for p in shells:
                with open(p) as fh:
                    txt = fh.read()
                self.assertNotIn('static PymsNFn', txt)
                self.assertIn('pyms_needs_accept_', txt)
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
