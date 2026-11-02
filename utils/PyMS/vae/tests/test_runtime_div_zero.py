"""An integer / or % by a zero only the run time knows is not silent (1.1.4).

THE DEFECT (PYMS4 skeptic, i4_div_runtime_zero, 2026-10-27): ``k = (V(p, n)
> 100) ? 1 : 0; z = 5 / k;`` gave z = 0 with no word (the eval code's
_vae_idiv returned 0 for b == 0).

THE FIX: every integer division and modulus of the eval code carries the
statement's file:line (ginac_emitter.py _add_div_sites); the first time a
site divides by zero at run time the math library queues a message
(vae_take_message) and the device shell reports it with Xyce's UserWarning
for the device: ``<file>:<line>: an integer division by zero at run time
(Verilog-A leaves it undefined); PyMS gives 0``.  The value stays 0, once
per site per run.  (A zero divisor known at elaboration is an error since
1.1.3.)

Run:  cd utils/PyMS && python3 vae/tests/test_runtime_div_zero.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.tests._decks import mod, run_decks, XYCE  # noqa: E402

DIV = mod("div0", "  integer k, z;\n  analog begin\n    k = (V(p, n) > 100) ? 1 : 0;\n    z = 5 / k;\n    I(p, n) <+ V(p, n) / 1000 + z * 1e-3 * V(p, n);\n  end")
MODZ = mod("mod0", "  integer k, z;\n  analog begin\n    k = (V(p, n) > 100) ? 1 : 0;\n    z = 7 % k;\n    I(p, n) <+ V(p, n) / 1000 + z * 1e-3 * V(p, n);\n  end")


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_division_warns_with_file_line(self):
        logs = {}
        self.assertEqual(run_decks({"div0": (DIV, "", 0.5)}, "pyms_div0_", logs=logs), [])
        text = " ".join(logs["div0"].split())
        self.assertRegex(text, r"c\.va:9: an integer division by zero at run time")
        self.assertIn("PyMS gives 0", text)

    def test_modulus_warns(self):
        logs = {}
        self.assertEqual(run_decks({"mod0": (MODZ, "", 0.5)}, "pyms_mod0_", logs=logs), [])
        self.assertRegex(" ".join(logs["mod0"].split()), r"c\.va:9: an integer modulus by zero at run time")


if __name__ == "__main__":
    unittest.main(verbosity=2)
