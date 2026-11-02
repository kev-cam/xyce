"""Every analysis starts its held state from the declaration values (1.1.5).

THE DEFECT (PYMS5 skeptic, 2026-11-01, on 1.1.4; a regression against
1.1.3): with .step, the held state of an event variable (an @(initial_step)
or @(cross) value) went on from the values the previous step point ended
with, because the device shell's first evaluation of an analysis (phase 0)
read the committed values instead of the declaration values: ``integer cnt
= 0; @(initial_step) cnt = cnt + 1;`` counted 1, 2 over two .step points
(s5b_mstep_tran_is_accum / s5b_gstep_tran_is_accum: the second step 0.667
sin for 0.5 sin), and a cross hysteresis started step 2 in the state step 1
had left (s5_step_tran_cross_reset).

THE FIX: xyce_device_gen.py -- the shell keeps the declaration values
(vae_held_init) and restores them, and the crossing prediction, on the
first evaluation of every analysis (phase 0 after any other phase).

Run:  cd utils/PyMS && python3 vae/tests/test_held_step_reset.py
"""
from __future__ import annotations

import math
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

W = 2 * math.pi * 1e3
TIMES = [0.1e-3, 0.25e-3, 0.6e-3, 1.25e-3, 1.9e-3]
TJ = [10e-6, 0.1e-3, 0.6e-3, 1.25e-3]


def tr(g, times=TIMES):
    return [g * math.sin(W * t) for t in times]


ACCUM = mod("a1", "  parameter real rp = 1000;\n  integer cnt = 0;\n  analog begin\n    @(initial_step) cnt = cnt + 1;\n"
                  "    I(p, n) <+ V(p, n) * cnt / (rp * 0 + 1000);\n  end")
HYST = mod("a1", "  parameter real dm = 1;\n  real r = 4000;\n  analog begin\n    @(cross(V(p, n) - 0.1, +1)) r = 1000;\n"
                 "    @(cross(V(p, n) + 0.1, -1)) r = 4000;\n    I(p, n) <+ V(p, n) / (r * dm / dm);\n  end")
HYST_WANT = [0.2 * math.sin(W * 10e-6), 0.5 * math.sin(W * 0.1e-3), 0.2 * math.sin(W * 0.6e-3), 0.5 * math.sin(W * 1.25e-3)]
TRAN = ".options timeint reltol=1e-6 abstol=1e-12\n.tran 1u 2m\n.print tran v(out)"
TRANJ = ".options timeint reltol=1e-6 abstol=1e-12 erroption=1\n.tran 1u 1.3m 0 1u\n.print tran v(out)"

# name -> (module, .model card, the .step and analysis lines, times, want per step)
CASES = {
    "mstep_accum": (ACCUM, "rp=1000", ".step m1:rp list 1000 4000\n" + TRAN, TIMES, tr(0.5) + tr(0.5)),
    "gstep_accum": (ACCUM, "rp={rpv}", ".global_param rpv=1000\n.step rpv list 1000 4000\n" + TRAN, TIMES, tr(0.5) + tr(0.5)),
    "pstep_accum": (ACCUM, "", ".param dmy=1\n.step dmy list 1 2 3\n" + TRAN, TIMES, tr(0.5) * 3),
    "step_cross_reset": (HYST, "dm={dmy}", ".param dmy=1\n.step dmy list 1 2\n" + TRANJ, TJ, HYST_WANT + HYST_WANT),
}


def segments(path):
    with open(path) as fh:
        rows = [l.split() for l in fh.read().splitlines() if l.strip()]
    hdr = [c.upper() for c in rows[0]]
    tk, vk = hdr.index("TIME"), hdr.index("V(OUT)")
    segs, last = [], None
    for c in rows[1:]:
        if c[0].lower().startswith("end") or c[0].upper() == "INDEX":
            continue
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

    def test_decks(self):
        work = tempfile.mkdtemp(prefix='pyms_hstep_')
        env = dict(os.environ, PYMS_DIR=_PYMS_DIR, PYTHONDONTWRITEBYTECODE='1')
        env['LD_LIBRARY_PATH'] = XYCE_LIBDIR + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        bad = []
        try:
            for name, (va, card, ana, times, want) in CASES.items():
                d = os.path.join(work, name)
                os.makedirs(d)
                with open(os.path.join(d, 'a1.va'), 'w') as fh:
                    fh.write(va)
                with open(os.path.join(d, 'c.cir'), 'w') as fh:
                    fh.write('* %s\n.hdl "a1.va"\nV1 in 0 SIN(0 1 1k)\n.model m1 a1 %s\nYa1 X1 in out m1\n'
                             'R2 out 0 1k\n%s\n.end\n' % (name, card, ana))
                env.update(PYMS_CACHE=os.path.join(d, 'hc'), PYMS_VAE_CACHE=os.path.join(d, 'vc'))
                r = subprocess.run([XYCE, 'c.cir'], cwd=d, env=env, capture_output=True, text=True, timeout=1800)
                got = None
                try:
                    got = [interp(s, t) for s in segments(os.path.join(d, 'c.cir.prn')) for t in times]
                except (OSError, ValueError):
                    pass
                ok = (r.returncode == 0 and got is not None and len(got) == len(want) and None not in got
                      and all(abs(g - w) <= 1e-3 for g, w in zip(got, want)))
                print('%-18s want %s\n%-18s got  %s rc %d' % (name, ['%.4f' % w for w in want], '',
                      None if got is None else ['-' if g is None else '%.4f' % g for g in got], r.returncode), flush=True)
                if not ok:
                    bad.append((name, got, (r.stdout + r.stderr)[-500:] if got is None else ''))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
