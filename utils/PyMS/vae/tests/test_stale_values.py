"""The last reaching assignment decides a run-time condition (1.1.4).

THE DEFECT (PYMS4 skeptic, round b, 2026-10-27; every version since 1.1.0):
a variable assigned under a voltage-dependent condition and then assigned a
constant kept the gated symbol in a later run-time condition: the
condition's translator (ginac_emitter.py _cond_final_cpp._idsub) looked up
the variable's last SYMBOL before its known constant value, and a constant
assignment leaves the old symbol in place.  ``if (V > -1) k = 5; k = 0; if
(k < 1 && V > -1) g = 1m; else g = 0.25m;``: 0.2 for 0.5 (x5_stale_if_cond),
and every second pass of a loop that resets its counter (x5_two_vdep_loops_seq,
l4_nested_vdep_inner: 0.333 for 0.5).

THE FIX: the known constant first (var_values: present only while the last
assignment was that constant), as the expression translators have it.

Run:  cd utils/PyMS && python3 vae/tests/test_stale_values.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a  # noqa: E402
from vae.ginac_emitter import GiNaCEmitter  # noqa: E402
from vae.tests._decks import mod, run_decks, XYCE  # noqa: E402

H = "  integer k;\n  real g;\n  analog begin\n    k = 0; g = 0;\n"
CASES = {
    "stale_if_cond": (mod("stale_if_cond", H + "    if (V(p, n) > -1) k = 5;\n    k = 0;\n    if (k < 1 && V(p, n) > -1) g = 1m; else g = 0.25m;\n    I(p, n) <+ V(p, n) * g;\n  end"), "", 0.5),
    "real_stale_if": (mod("real_stale_if", "  real x, g;\n  analog begin\n    x = 0;\n    if (V(p, n) > -1) x = 5;\n    x = 0;\n    if (x < 1 && V(p, n) > -1) g = 1m; else g = 0.25m;\n    I(p, n) <+ V(p, n) * g;\n  end"), "", 0.5),
    "two_vdep_loops_seq": (mod("two_vdep_loops_seq", H + "    while (k < 2 && V(p, n) > -1) begin g = g + 0.25m; k = k + 1; end\n    k = 0;\n    while (k < 2 && V(p, n) > -1) begin g = g + 0.25m; k = k + 1; end\n    I(p, n) <+ V(p, n) * g;\n  end"), "", 0.5),
    "nested_vdep_inner": (mod("nested_vdep_inner", "  integer i, k;\n  real g;\n  analog begin\n    g = 0;\n    for (i = 0; i < 2; i = i + 1) begin\n      k = 0;\n      while (k < 3 && V(p, n) > -1) begin g = g + (1m / 6); k = k + 1; end\n    end\n    I(p, n) <+ V(p, n) * g;\n  end"), "", 0.5),
    "stale_ternary": (mod("stale_ternary", "  real x, y;\n  analog begin\n    x = 0;\n    if (V(p, n) > -1) x = 5;\n    x = 0;\n    y = (x < 1 && V(p, n) > -1) ? 1m : 0.25m;\n    I(p, n) <+ V(p, n) * y;\n  end"), "", 0.5),
}


class TestEmit(unittest.TestCase):

    def test_condition_reads_the_constant(self):
        prog = GiNaCEmitter(parse_verilog_a(CASES["stale_if_cond"][0], filename="c.va")).emit()
        cond = [l for l in prog.splitlines() if '_vae_latch' in l and '-1' in l and '< 1' in l]
        self.assertTrue(cond, "no indicator of the second condition")
        self.assertIn('0.0 < 1', cond[-1])
        self.assertNotIn('k__sel', cond[-1])


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_stale_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
