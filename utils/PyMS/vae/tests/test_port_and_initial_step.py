"""ANSI port declarations, ``inout electrical p;``, @(initial_step) blocks.

THE DEFECTS (PYMS3 skeptic, 2026-10-26: legal Verilog-A that PyMS refused or
got wrong; BUGS.md CHANGES 1.1.3):
  * ``module a1(inout electrical p, inout electrical n);`` -- the keywords
    became port names (g++ "redefinition of nodeID_inout");
  * ``inout electrical p;`` -- a parse error ("expected ',' or ';' after
    'electrical'");
  * @(initial_step) blocks were skipped by the emitter, so the variables they
    set read 0 (``@(initial_step) r = 4 * rs;`` then I <+ V / r: no solution).

THE FIX: the port list takes a direction and / or a discipline before names
(they hold until the next one); a direction declaration takes an optional
discipline; the emitter runs an initial_step block where it stands (at
every evaluation: the same as once for the parameter-only precomputation it
holds in practice).

Run:  cd utils/PyMS && python3 vae/tests/test_port_and_initial_step.py
"""
from __future__ import annotations

import os
import sys
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a, PortDir  # noqa: E402
from vae.tests._decks import HDR, mod, run_decks, XYCE  # noqa: E402

CASES = {
    "ansi": (HDR + "module ansi(inout electrical p, inout electrical n);\n  analog I(p, n) <+ V(p, n) / 1000;\nendmodule\n", "", 0.5),
    "ansi_shared": (HDR + "module ansi_shared(inout electrical p, n);\n  analog I(p, n) <+ V(p, n) / 1000;\nendmodule\n", "", 0.5),
    "inout_disc": (HDR + "module inout_disc(p, n);\n  inout electrical p;\n  inout electrical n;\n  analog I(p, n) <+ V(p, n) / 1000;\nendmodule\n", "", 0.5),
    "init_local": (mod("init_local", "  real r;\n  analog begin\n    @(initial_step) begin : init\n      real t;\n      t = 500;\n      r = 2 * t;\n    end\n    I(p, n) <+ V(p, n) / r;\n  end"), "", 0.5),
    "init_param": (mod("init_param", "  parameter real rs = 250;\n  real r;\n  analog begin\n    @(initial_step) r = 4 * rs;\n    I(p, n) <+ V(p, n) / r;\n  end"), "", 0.5),
}


class TestParse(unittest.TestCase):

    def test_ansi_ports(self):
        m = parse_verilog_a(HDR + "module a(inout electrical p, n, input thermal t);\n"
                            "  analog I(p, n) <+ V(p, n) / 1000;\nendmodule\n", filename="c.va")
        self.assertEqual([(p.name, p.direction, p.discipline) for p in m.ports],
                         [("p", PortDir.INOUT, "electrical"), ("n", PortDir.INOUT, "electrical"),
                          ("t", PortDir.IN, "thermal")])

    def test_direction_with_discipline(self):
        m = parse_verilog_a(CASES["inout_disc"][0], filename="c.va")
        self.assertEqual([p.name for p in m.ports], ["p", "n"])
        self.assertEqual(m.internal_nodes, [])


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def test_decks(self):
        self.assertEqual(run_decks(CASES, "pyms_portis_"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
