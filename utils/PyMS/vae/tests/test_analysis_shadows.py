"""Analysis-specific code: @(initial_step(list)), analysis(), $abstime,
@(final_step), @(cross) / @(above), the regime held for ac (1.1.4).

THE DEFECTS (PYMS4 skeptic, 2026-10-27, on 1.1.3; the 1 V / 1k divider):
  * an @(initial_step("ac")) or ("tran") body ran in a bare .op
    (is4_ac_only_op / is4_tran_only_op 0.2 for 0.5): the parser dropped the
    analysis list and the emitter ran every initial_step at every
    evaluation;
  * @(initial_step) v0 = V(p, n) was re-evaluated at every transient step
    (is4_vread_tran 0.5 sin for 1/3 sin: the block runs once, at the
    operating point, and v0 keeps its value);
  * $abstime in an initial_step (or anywhere) reached g++ (B14);
  * @(final_step) and @(cross(...)) bodies ran at every evaluation
    (is4_final_step_op, is4_cross_op 0.2 for 0.5);
  * analysis() reached g++ as an undeclared function.

THE DESIGN (the user's rule, 2026-10-30): initial_step("ac") and ("tran")
imply different analyses and may be entirely different code, so the math
library holds a shadow of the eval code per analysis context that needs one
(ginac_emitter.GiNaCEmitter.emit), chosen by the analysis and phase the
device shell passes (VaeState an, phase); a bare .op runs neither; a plain
initial_step runs once at the start of each analysis and holds its values;
the small-signal phase of an ac analysis linearises the operating point's
evaluation in the regime the operating point chose.

Run:  cd utils/PyMS && python3 vae/tests/test_analysis_shadows.py
"""
from __future__ import annotations

import math
import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a, NodeKind  # noqa: E402
from vae.ginac_emitter import GiNaCEmitter, EmitError  # noqa: E402
try:
    from vae.ginac_emitter import analysis_names  # noqa: E402
except ImportError:          # (PyMS before 1.1.4: the decks below still run)
    analysis_names = None
from vae.tests._decks import mod, rv, tran, run_decks, XYCE, TIMES  # noqa: E402

W = 2 * math.pi * 1e3


def trg(gains):
    return [g * math.sin(W * t) for g, t in zip(gains, TIMES)]


def r_is(lst, first, second):
    """r = first; @(initial_step(lst)) r = second; I <+ V / r"""
    head = '@(initial_step' + (('(' + ', '.join('"%s"' % a for a in lst) + ')') if lst else '') + ')'
    return mod("m", "  real r;\n  analog begin\n    r = %s;\n    %s r = %s;\n    I(p, n) <+ V(p, n) / r;\n  end"
               % (first, head, second))


CASES = {
    # the skeptic's decks
    "ac_only_op": (r_is(["ac"], 1000, 4000), "", 0.5),
    "tran_only_op": (r_is(["tran"], 1000, 4000), "", 0.5),
    "static_op": (r_is(["static"], 4000, 1000), "", 0.5),
    "vread_tran": (mod("m", "  real v0;\n  analog begin\n    @(initial_step) v0 = V(p, n);\n    I(p, n) <+ V(p, n) / 2000 + v0 / 2000;\n  end"), "", tran(1 / 3)),
    "vread_op": (mod("m", "  real v0;\n  analog begin\n    @(initial_step) v0 = V(p, n);\n    I(p, n) <+ V(p, n) / 2000 + v0 / 2000;\n  end"), "", 0.5),
    "abstime_tran": (mod("m", "  real t0;\n  analog begin\n    @(initial_step) t0 = $abstime;\n    I(p, n) <+ V(p, n) / (1000 + 1e9 * t0);\n  end"), "", tran(0.5)),
    "final_step_op": (mod("m", "  real r;\n  analog begin\n    r = 1000;\n    @(final_step) r = 4000;\n    I(p, n) <+ V(p, n) / r;\n  end"), "", 0.5),
    "cross_op": (mod("m", "  real r;\n  analog begin\n    r = 1000;\n    @(cross(V(p, n) - 10, 1)) r = 4000;\n    I(p, n) <+ V(p, n) / r;\n  end"), "", 0.5),
    # the lists and the other names
    "dc_op": (r_is(["dc"], 4000, 1000), "", 0.5),
    "noise_op": (r_is(["noise"], 1000, 4000), "", 0.5),
    "list_op": (r_is(["ac", "tran"], 1000, 4000), "", 0.5),
    "static_tran": (r_is(["static"], 4000, 1000), "", tran(0.2)),
    "tran_tran": (r_is(["tran"], 1000, 4000), "", tran(0.5)),
    "held_tran": (mod("m", "  real r;\n  analog begin\n    @(initial_step) r = 4000 * (1 + V(p, n));\n    I(p, n) <+ V(p, n) / r;\n  end"), "", tran(0.2)),
    "analysis_op": (mod("m", "  real r;\n  analog begin\n    if (analysis(\"tran\")) r = 4000; else r = 1000;\n    I(p, n) <+ V(p, n) / r;\n  end"), "", 0.5),
    "analysis_tran": (mod("m", "  real r;\n  analog begin\n    if (analysis(\"tran\")) r = 1000; else r = 4000;\n    I(p, n) <+ V(p, n) / r;\n  end"), "", tran(0.5)),
    "abstime_expr_tran": (mod("m", "  real r;\n  analog begin\n    r = ($abstime > 1m) ? 1000 : 4000;\n    I(p, n) <+ V(p, n) / r;\n  end"), "", trg([0.2, 0.2, 0.2, 0.5, 0.5])),
    "abstime_b14_op": (mod("m", "  analog V(p, n) <+ ($abstime > 1u) ? sqrt(-1.0 - V(p, n) * 0) : 0.5;"), "", 0.5),
    # ac: the operating point's evaluation, linearised in its regime
    "ac_is_ac": (r_is(["ac"], 1000, 4000), "", [0.2], "ac"),
    "ac_is_tran": (r_is(["tran"], 1000, 4000), "", [0.5], "ac"),
    "ac_analysis": (mod("m", "  real r;\n  analog begin\n    if (analysis(\"ac\")) r = 4000; else r = 1000;\n    I(p, n) <+ V(p, n) / r;\n  end"), "", [0.2], "ac"),
    "ac_regime_held": (mod("m", "  analog begin\n    if (V(p, n) > 0) I(p, n) <+ V(p, n) / 1000; else I(p, n) <+ V(p, n) / 4000;\n  end"), "", [0.2], "ac"),
    # cross with hysteresis: up through 0.1 -> 1k, down through -0.1 -> 4k
    "cross_hyst_tran": (mod("m", "  real r = 4000;\n  analog begin\n    @(cross(V(p, n) - 0.1, +1)) r = 1000;\n    @(cross(V(p, n) + 0.1, -1)) r = 4000;\n    I(p, n) <+ V(p, n) / r;\n  end"), "", trg([0.5, 0.5, 0.2, 0.5, 0.2]), "tranj"),
    "above_op": (mod("m", "  real r = 1000;\n  analog begin\n    @(above(V(p, n) - 10)) r = 4000;\n    I(p, n) <+ V(p, n) / r;\n  end"), "", 0.5),
    "timer_refused": (mod("m", "  real r;\n  analog begin\n    r = 1000;\n    @(timer(1m)) r = 4000;\n    I(p, n) <+ V(p, n) / r;\n  end"), "", "ERR"),
}
# run_decks names the module by the case
CASES = {k: (v[0].replace("module m(", "module %s(" % k),) + tuple(v[1:]) for k, v in CASES.items()}


class TestParse(unittest.TestCase):

    def test_event_list_is_kept(self):
        top = parse_verilog_a(r_is(["ac", "tran"], 1, 2), filename="c.va").analog_block
        ev = [n for n in top.children if n.kind == NodeKind.INITIAL_STEP][0]
        self.assertEqual([e['kind'] for e in ev.event], ['initial_step'])
        self.assertEqual(ev.event[0]['args'], ['"ac"', '"tran"'])

    def test_cross_or_final(self):
        va = mod("m", "  real r;\n  analog begin\n    @(cross(V(p, n) - 0.5, -1) or final_step) r = 1;\n    I(p, n) <+ V(p, n) / 1000;\n  end")
        top = parse_verilog_a(va, filename="c.va").analog_block
        ev = [n for n in top.children if n.kind == NodeKind.INITIAL_STEP][0]
        self.assertEqual([e['kind'] for e in ev.event], ['cross', 'final_step'])
        self.assertEqual(len(ev.event[0]['args']), 2)

    @unittest.skipIf(analysis_names is None, "no analysis_names (PyMS < 1.1.4)")
    def test_names(self):
        self.assertEqual(analysis_names(0, 0), {'dc', 'static'})
        self.assertEqual(analysis_names(1, 0), {'tran', 'ic', 'static'})
        self.assertEqual(analysis_names(1, 1), {'tran'})
        self.assertEqual(analysis_names(2, 2), analysis_names(2, 0))

    def test_shadows(self):
        e = GiNaCEmitter(parse_verilog_a(r_is(["tran"], 1000, 4000), filename="c.va"))
        prog = e.emit()
        self.assertGreater(len(e._shadows), 1)
        self.assertIn('_vae_eval_s1', prog)
        # a module without analysis-specific code has one body
        e = GiNaCEmitter(parse_verilog_a(mod("m", "  analog I(p, n) <+ V(p, n) / 1000;"), filename="c.va"))
        e.emit()
        self.assertEqual(len(e._shadows), 1)

    def test_invariant_initial_step_one_body(self):
        va = mod("m", "  parameter real rs = 250;\n  real r;\n  analog begin\n    @(initial_step) r = 4 * rs;\n    I(p, n) <+ V(p, n) / r;\n  end")
        e = GiNaCEmitter(parse_verilog_a(va, filename="c.va"))
        e.emit()
        self.assertEqual(len(e._shadows), 1)
        self.assertEqual(e._state_vars, [])

    def test_unsupported_event(self):
        with self.assertRaisesRegex(EmitError, r"c\.va:\d+: @\(timer"):
            GiNaCEmitter(parse_verilog_a(CASES["timer_refused"][0], filename="c.va")).emit()


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_shadows_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
