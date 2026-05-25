"""Companion to test_va_attributes.py.

Once xyceModelGroup is extracted correctly, the wrapper still has to
split the .va's declared ports into:

  numNodes()         — required terminals per SPICE convention (the
                       count the netlist's device-letter card supplies)
  numOptionalNodes() — extras the netlist may or may not include
                       (typically thermal nodes for self-heating)

Convention by group:
  MOSFET   4 (D G S B), one optional thermal makes 5
  BJT      4 (C B E S), one optional thermal makes 5
  Diode    2 (A C), one optional thermal makes 3
  Capacitor 2
  Resistor 2

At fa19579b the wrapper emits numNodes() = len(declared_ports) and
numOptionalNodes() = 0 unconditionally. That's wrong for any compact
model with a thermal port: BSIM-CMG 110/111 declare (d,g,s,e,t) so
numNodes=5, but the M-card has 4 nodes and a model name. Xyce parses
the model name as the 5th "node" and the next field as a parameter,
producing "Unrecognized parameter <whatever> for device M1".

This test asserts the (numNodes, numOptionalNodes) split the wrapper
WILL emit, for every entry that passes attribute extraction.
"""

from __future__ import annotations

import os
import re
import sys
import unittest

_VAE_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VAE_PKG_DIR not in sys.path:
    sys.path.insert(0, _VAE_PKG_DIR)

from vae.parser import parse_file  # noqa: E402
from vae.xyce_device_gen import scan_va_attributes, generate_device_cpp  # noqa: E402
from vae.tests.test_va_attributes import ENTRY_POINTS, GROUP_REQUIRED_TERMINALS  # noqa: E402


def _emit_and_extract_node_counts(va_path: str):
    """Parse the .va, run the wrapper generator, return (numNodes, optional)
    pulled from the emitted header. Raises on parse/codegen failure."""
    mod = parse_file(va_path)
    header, _impl = generate_device_cpp(mod, '', va_path=va_path)
    nn = re.search(r'numNodes\(\)\s*\{\s*return\s+(\d+)\s*;', header)
    no = re.search(r'numOptionalNodes\(\)\s*\{\s*return\s+(\d+)\s*;', header)
    if nn is None or no is None:
        raise RuntimeError(
            f'Header lacks numNodes() / numOptionalNodes() literals')
    return int(nn.group(1)), int(no.group(1)), len(mod.ports), \
        [p.name for p in mod.ports]


class TestTerminalSplit(unittest.TestCase):
    """For every entry whose xyceModelGroup is detected, the wrapper
    should emit numNodes() matching the SPICE device-letter convention
    and numOptionalNodes() = (declared ports) - (required).

    Where the .va declares more ports than the convention requires,
    the extras must end up in numOptionalNodes — otherwise the
    netlist's M-card / Q-card / D-card terminal-count check rejects
    correct netlists.
    """

    def _check_one(self, desc, va_path, want_group, want_level, want_req):
        if not os.path.exists(va_path):
            self.skipTest(f'{desc}: {va_path} not installed')

        # Skip entries where the upstream attribute test already fails —
        # they're tracked separately in test_va_attributes.py; piling on
        # here just makes the failure list noisy.
        attrs = scan_va_attributes(va_path)
        if attrs.get('xyceModelGroup') != want_group \
                or attrs.get('xyceLevelNumber') != want_level:
            self.skipTest(f'{desc}: attr extraction not working yet; '
                          f'fix that first (see test_va_attributes.py)')

        # Try to parse + generate. Some .va files trip the parser
        # itself — record that distinctly so it's clear the issue is
        # upstream of terminal-split.
        try:
            num_nodes, num_optional, n_declared, port_names = \
                _emit_and_extract_node_counts(va_path)
        except Exception as e:
            self.fail(f'{desc}: parser/codegen failure before terminal '
                      f'split check: {type(e).__name__}: {e}')

        expected_optional = max(0, n_declared - want_req)
        with self.subTest(model=desc, attr='numNodes'):
            self.assertEqual(num_nodes, want_req,
                f'{desc}: numNodes()={num_nodes}, want {want_req} '
                f'(SPICE-letter convention for {want_group}). Declared '
                f'ports: {port_names}. With numNodes()={num_nodes}, the '
                f'netlist M-card with {want_req} nodes + a model name '
                f'will have its model-name field parsed as a node and '
                f'the next field as an unrecognised parameter.')
        with self.subTest(model=desc, attr='numOptionalNodes'):
            self.assertEqual(num_optional, expected_optional,
                f'{desc}: numOptionalNodes()={num_optional}, want '
                f'{expected_optional}. Extras: {port_names[want_req:]}.')


def _add_per_entry_methods():
    for desc, va_path, want_group, want_level, want_req in ENTRY_POINTS:
        sanitised = desc.replace(' ', '_').replace('.', '_') \
                        .replace('/', '_').replace('-', '_') \
                        .replace('(', '').replace(')', '')
        def _make_test(_desc, _path, _g, _l, _r):
            def test(self):
                self._check_one(_desc, _path, _g, _l, _r)
            return test
        setattr(TestTerminalSplit, f'test_{sanitised}',
                _make_test(desc, va_path, want_group, want_level, want_req))


_add_per_entry_methods()


if __name__ == '__main__':
    unittest.main(verbosity=2)
