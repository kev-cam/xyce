"""The model-group route (``.model m c level=6`` / ``C1 a b m``) and its siblings.

THE DEFECTS (PYMS1 skeptic, 2026-10-25; BUGS.md CHANGES 1.1.1):

- A9: toys/rlc.va has no xyce* attributes, but
  xyce_device_gen.scan_va_attributes fell back to ANY sibling .va of the
  directory that has them and gave rlc the Capacitor level-6 group of
  toys/capacitor.va.  A Capacitor-group module takes the simple R/L/C stamp
  (one capacitor), so rlc ran as a bare capacitor: |out| at 10 MHz 0.0627
  +86.4 deg against the analytic series RLC 0.0213 -87.6 deg, rc 0.  The
  same fallback made psp103_nqs.va a JUNCAP diode and Mextram's bjtd504.va
  a 4-terminal BJT.  Now only a variant of the same model inherits (one file
  name extends the other: psp102b.va / psp102.va).
- A10: on the simple R/L/C stamp a math library that did not build was
  ignored (rc 0: toys/capacitor.va, sg13g2 r3_cmc); it is a device error on
  every route now.
- B9: toys/capacitor.va's ``real Q`` collided with the eval function's
  charge vector ``double* Q`` ("declaration of 'double Q' shadows a
  parameter"); the eval's own names are ``_vae_*`` now.

Run:  cd utils/PyMS && python3 vae/tests/test_model_group_route.py
"""
from __future__ import annotations

import cmath
import glob
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

from vae.xyce_device_gen import scan_va_attributes  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'

CAPACITOR = HDR + '''
(*xyceModelGroup="Capacitor", xyceSpiceDeviceName="C", xyceLevelNumber="6"*)
module capacitor (p,n);
  electrical p,n;
  inout p,n;
  (*desc="Capacitance", type="instance" *) parameter real C=1p from (0:inf);
  real Q;
  analog
  begin
    Q=C*V(p,n);
    I(p,n) <+ ddt(Q);
  end
endmodule
'''
# toys/rlc.va (no attributes of its own)
RLC = HDR + '''
module rlc (p,n);
  electrical p,n;
  inout p,n;
  electrical internal1, internal2;
  (*desc="Inductance", type="instance"*) parameter real L=1e-3 from (0:inf);
  (*desc="Resistance", type="instance"*) parameter real R=1e3 from (0:inf);
  (*desc="Capacitance", type="instance"*) parameter real C=1e-12 from (0:inf);
  real InductorCurrent;
  real CapacitorCharge;
  analog
  begin
    I(p,internal1)<+ V(p,internal1)/R;
    CapacitorCharge = V(internal1,internal2)*C;
    I(internal1,internal2) <+ ddt(CapacitorCharge);
    InductorCurrent=I(internal2,n);
    V(internal2,n) <+ L*ddt(InductorCurrent);
  end
endmodule
'''
# a Capacitor-group module whose math library cannot build ($abstime: BUGS.md B14)
BADCAP = HDR + '''
(*xyceModelGroup="Capacitor", xyceSpiceDeviceName="C", xyceLevelNumber="6"*)
module badcap (p,n);
  electrical p,n;
  inout p,n;
  (*desc="Capacitance", type="instance" *) parameter real C=1p from (0:inf);
  analog I(p,n) <+ ddt(C*V(p,n)) + 0 * idt(V(p,n));   // idt: no math library (1.1.4: $abstime builds)
endmodule
'''


def _env(work):
    env = dict(os.environ)
    env.update(PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(work, "hc"),
               PYMS_VAE_CACHE=os.path.join(work, "vc"), PYTHONDONTWRITEBYTECODE="1")
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def _run(work, deck):
    open(os.path.join(work, "d.cir"), "w").write(deck)
    for f in glob.glob(os.path.join(work, "d.cir*prn")):
        os.remove(f)
    r = subprocess.run([XYCE, "d.cir"], cwd=work, env=_env(work), capture_output=True,
                       text=True, timeout=900)
    rows = []
    for fn in ("d.cir.FD.prn", "d.cir.prn"):
        p = os.path.join(work, fn)
        if os.path.exists(p):
            for l in open(p):
                s = l.split()
                if s and s[0][0].isdigit():
                    rows.append([float(x) for x in s])
            break
    return r.returncode, r.stdout + r.stderr, rows


class TestAttributes(unittest.TestCase):

    def test_unrelated_sibling_lends_no_group(self):
        d = tempfile.mkdtemp(prefix="pyms_sib_")
        try:
            open(os.path.join(d, "capacitor.va"), "w").write(CAPACITOR)
            open(os.path.join(d, "rlc.va"), "w").write(RLC)
            self.assertEqual(scan_va_attributes(os.path.join(d, "capacitor.va")).get('xyceModelGroup'),
                             'Capacitor')
            self.assertNotIn('xyceModelGroup', scan_va_attributes(os.path.join(d, "rlc.va")))
            # a variant of the same model still inherits
            open(os.path.join(d, "capacitor_b.va"), "w").write(RLC.replace("module rlc", "module capb"))
            self.assertEqual(scan_va_attributes(os.path.join(d, "capacitor_b.va")).get('xyceModelGroup'),
                             'Capacitor')
        finally:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="pyms_mgr_")

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def test_rlc_beside_capacitor_keeps_its_inductor(self):
        open(os.path.join(self.work, "capacitor.va"), "w").write(CAPACITOR)
        open(os.path.join(self.work, "rlc.va"), "w").write(RLC)
        rc, log, rows = _run(self.work, '* rlc ac\n.hdl "rlc.va"\nV1 in 0 DC 0 AC 1\n.model m1 rlc\n'
                             'Yrlc X1 in out m1\nR2 out 0 1k\n.ac dec 1 1e6 1e7\n.print ac vm(out) vp(out)\n.end\n')
        self.assertEqual(rc, 0, log[-2000:])
        self.assertEqual(len(rows), 2, log[-2000:])
        for row in rows:
            f = row[1]
            w = 2 * math.pi * f
            o = 1000 / (1000 + 1j * (w * 1e-3 - 1 / (w * 1e-12)) + 1000)
            self.assertAlmostEqual(row[2] / abs(o), 1.0, delta=1e-5, msg="f=%g" % f)
            # VP is printed in degrees by this Xyce build
            ph = row[3] if abs(row[3]) > 2 * math.pi else math.degrees(row[3])
            self.assertAlmostEqual(ph, math.degrees(cmath.phase(o)), delta=1e-3, msg="f=%g" % f)

    def test_capacitor_math_library_builds(self):
        open(os.path.join(self.work, "capacitor.va"), "w").write(CAPACITOR)
        rc, log, rows = _run(self.work, '* cap ac\n.hdl "capacitor.va"\nV1 in 0 DC 0 AC 1\n'
                             '.model m1 c level=6\nC1 in out m1 C=1n\nR2 out 0 1k\n'
                             '.ac dec 1 1e5 1e6\n.print ac vm(out)\n.end\n')
        self.assertEqual(rc, 0, log[-2000:])
        self.assertNotIn("shadows a parameter", log)
        self.assertTrue(glob.glob(os.path.join(self.work, "vc", "vae_capacitor_*.so")), log[-2000:])
        for row in rows:
            w = 2 * math.pi * row[1]
            o = 1000 / (1000 + 1 / (1j * w * 1e-9))
            self.assertAlmostEqual(row[2] / abs(o), 1.0, delta=1e-5)

    def test_failed_library_is_loud_on_the_model_group_route(self):
        open(os.path.join(self.work, "badcap.va"), "w").write(BADCAP)
        rc, log, rows = _run(self.work, '* badcap\n.hdl "badcap.va"\nV1 in 0 DC 1\n'
                             '.model m1 c level=6\nC1 in out m1 C=1n\nR2 out 0 1k\n'
                             '.op\n.print dc v(out)\n.end\n')
        self.assertNotEqual(rc, 0, log[-2000:])
        self.assertTrue("did not build or load" in log or "idt" in log, log[-2000:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
