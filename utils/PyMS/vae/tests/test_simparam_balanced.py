"""$simparam("name", default) with nested parentheses in the default.

THE DEFECT (PYMS3 skeptic, 2026-10-26; BUGS.md CHANGES 1.1.3):
GiNaCEmitter._sub_simparam matched the call with a regular expression of ONE
level of parentheses, so ``$simparam("nosuch", max(1.0, min(2.0, 3.0)))`` was
not substituted and reached the C++ ("'$simparam' was not declared": no math
library).  Also _try_eval_expr evaluated the text from before the
substitution, so no $simparam expression was ever folded to a constant.

THE FIX: a balanced scan of the call (strings and any depth of parentheses);
_try_eval_expr evaluates the substituted text.

Run:  cd utils/PyMS && python3 vae/tests/test_simparam_balanced.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a  # noqa: E402
from vae.ginac_emitter import GiNaCEmitter, EmitError  # noqa: E402
from vae.tests._decks import mod, run_decks, XYCE  # noqa: E402

CASES = {
    "nested_cond": (mod("nested_cond", '  analog begin\n    if ($simparam("nosuch", max(1.0, min(2.0, 3.0))) > 1.5) I(p, n) <+ V(p, n) / 1000;\n    else I(p, n) <+ V(p, n) / 2000;\n  end'), "", 0.5),
    "nested_expr": (mod("nested_expr", '  analog I(p, n) <+ V(p, n) / (500 * $simparam("nosuch", (1 + (0.5 * (2.0)))));'), "", 0.5),
    "nested_assign": (mod("nested_assign", '  real r;\n  analog begin r = 250 * $simparam("nosuch", pow(2, (1 + 1))); I(p, n) <+ V(p, n) / r; end'), "", 0.5),
    "known_nested_default": (mod("known_nested_default", '  analog begin\n    if ($simparam("gmin", max(1.0, (2.0))) < 1e-6) I(p, n) <+ V(p, n) / 1000;\n    else I(p, n) <+ V(p, n) / 2000;\n  end'), "", 0.5),
}


class TestSubstitution(unittest.TestCase):

    def _em(self, simparams=None):
        return GiNaCEmitter(parse_verilog_a(CASES["nested_expr"][0], filename="c.va"), simparams=simparams)

    def test_nested_default(self):
        e = self._em()
        self.assertEqual(e._sub_simparam('$simparam ( "x" , max ( 1.0 , min ( 2.0 , 3.0 ) ) ) > 1').replace(' ', ''),
                         '(max(1.0,min(2.0,3.0)))>1')
        self.assertEqual(e._sub_simparam('a + $simparam("x", (1 + (2))) * $simparam("y", 3)').replace(' ', ''),
                         'a+((1+(2)))*(3)')

    def test_known_value_wins_and_no_default_is_an_error(self):
        e = self._em({'gmin': 1e-12})
        self.assertEqual(e._sub_simparam('$simparam("gmin", max(1, (2)))'), '(1e-12)')
        with self.assertRaises(EmitError):
            e._sub_simparam('$simparam("nosuch")')

    def test_folded_to_a_constant(self):
        self.assertEqual(self._em()._try_eval_expr('500 * $simparam ( "x" , ( 1 + ( 1 ) ) )'), 1000.0)


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_smpb_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
