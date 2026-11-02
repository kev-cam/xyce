"""Block-local variables are variables of their own block (LRM 6.8 scoping).

THE DEFECT (PYMS3 skeptic, 2026-10-26; BUGS.md CHANGES 1.1.3): every
block-local declaration merged into ONE module variable of the first
declaration's type: ``real x; x = 1000; begin : blk real x; x = 3000; end
I(p, n) <+ V(p, n) / x;`` gave 0.25 on the 1 V / 1k divider (0.5: the block's
x is another variable), an ``integer k`` shadowed by a block's ``real k``
truncated 0.5 to 0 (0.8 for 0.5), and two blocks declaring ``real t`` and
``integer t`` shared one real t (0.5357 for 0.5).  VACASK gives 0.5 for all.

THE FIX: the parser keeps a scope per begin...end; a block-local whose name
is taken (a module variable, parameter, node, an enclosing or an earlier
block's local) becomes a module variable of its own under a unique internal
name (``x__blk1``) with its own type, and the names inside the block are
renamed to it.  A fresh name keeps its name (no change for most models).

Run:  cd utils/PyMS && python3 vae/tests/test_block_scope.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a  # noqa: E402
from vae.tests._decks import mod, rv, run_decks, XYCE  # noqa: E402

CASES = {
    "shadow": (mod("shadow", "  real x;\n  analog begin\n    x = 1000;\n    begin : blk\n      real x;\n      x = 3000;\n    end\n    I(p, n) <+ V(p, n) / x;\n  end"), "", 0.5),
    "shadow_inner": (mod("shadow_inner", "  real x;\n  analog begin\n    x = 3000;\n    begin : blk\n      real x;\n      x = 1000;\n      I(p, n) <+ V(p, n) / x;\n    end\n  end"), "", 0.5),
    "shadow_type": (mod("shadow_type", "  integer k;\n  analog begin\n    k = 2;\n    begin : blk\n      real k;\n      k = 0.5;\n    end\n    I(p, n) <+ V(p, n) / (500 * k);\n  end"), "", 0.5),
    "shadow_type_inner": (mod("shadow_type_inner", "  real k;\n  analog begin\n    k = 0.5;\n    begin : blk\n      integer k;\n      k = 2.4;\n      I(p, n) <+ V(p, n) / (500 * k);\n    end\n  end"), "", 0.5),
    "two_blocks": (mod("two_blocks", "  analog begin\n    begin : b1\n      real t;\n      t = 2.5;\n    end\n    begin : b2\n      integer t;\n      t = 2.6;\n      I(p, n) <+ V(p, n) / (1000 * t / 3);\n    end\n  end"), "", 0.5),
    "nested3": (mod("nested3", "  real x;\n  analog begin\n    x = 1000;\n    begin : a\n      real x;\n      x = 2000;\n      begin : b\n        real x;\n        x = 3000;\n      end\n      I(p, n) <+ V(p, n) / x;\n    end\n  end"), "", rv(2000)),
    "param_shadow": (mod("param_shadow", "  parameter real r = 3000;\n  analog begin\n    begin : blk\n      real r;\n      r = 1000;\n      I(p, n) <+ V(p, n) / r;\n    end\n  end"), "", 0.5),
    "init_reads_outer": (mod("init_reads_outer", "  real x;\n  analog begin\n    x = 500;\n    begin : blk\n      real y = 2 * x;\n      I(p, n) <+ V(p, n) / y;\n    end\n  end"), "", 0.5),
}


class TestParse(unittest.TestCase):

    def test_shadow_is_its_own_variable(self):
        m = parse_verilog_a(CASES["shadow_type"][0], filename="c.va")
        types = {v.name: v.type for v in m.variables}
        self.assertEqual(types["k"], "integer")
        inner = [n for n in types if n.startswith("k__")]
        self.assertEqual(len(inner), 1)
        self.assertEqual(types[inner[0]], "real")

    def test_fresh_name_is_kept(self):
        m = parse_verilog_a(mod("f", "  analog begin\n    begin : b\n      real t;\n      t = 1000;\n      I(p, n) <+ V(p, n) / t;\n    end\n  end"), filename="c.va")
        self.assertEqual([v.name for v in m.variables], ["t"])


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_scope_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
