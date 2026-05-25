"""Validate PyMS's extraction of module-level Xyce attributes from every
top-level .va in the install tree.

The bug this catches:
  Most BSIM-CMG / MOSFET / BJT regression failures at fa19579b are
  "Unrecognized parameter X for device M..." where X is a node name
  (DRAIN, W, OUT, etc.) interpreted as a parameter. Root cause is the
  device wrapper's numNodes() returning the wrong terminal count, which
  in turn comes from xyceModelGroup not being extracted from the .va.

  scan_va_attributes() in xyce_device_gen.py uses a single regex over
  the top-level .va source. Several compact models — notably BSIM-CMG
  107 and 108 — put the module declaration (with the (*xyceModelGroup
  ... *) block) in an `include'd file (bsimcmg_main.va), so scan
  returns empty and the device class falls through to whatever default
  numNodes() / registerDevice path the generator uses.

The catalog below is the ground truth from a direct grep over the
install tree's .va sources. The test asserts scan_va_attributes returns
the right group + level for each entry point. Failing tests pinpoint
exactly which models need the include-following fix.

To regenerate ground truth:
    for d in /usr/local/share/xyce/verilog-a/*/; do
      grep -h 'xyceModelGroup\\|xyceLevelNumber' "$d"*.va "$d"*.include 2>/dev/null \\
        | head -1
    done
"""

from __future__ import annotations

import os
import sys
import unittest

# Make `vae.*` importable when run as a plain script.
_VAE_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VAE_PKG_DIR not in sys.path:
    sys.path.insert(0, _VAE_PKG_DIR)

from vae.xyce_device_gen import scan_va_attributes  # noqa: E402


# The .va file Xyce actually loads via .HDL (the "entry point"), along
# with the expected attributes discoverable from that file alone OR via
# its `include's. Required-terminal counts follow the SPICE convention
# Xyce uses for device-letter dispatch:
#   MOSFET: 4 (D G S B), one optional (thermal)
#   BJT:    4 (C B E S), one optional (thermal)
#   Diode:  2 (A C),     one optional (thermal)
#   Capacitor: 2
#
# Format: (description, va_path, expected_group, expected_level,
#          expected_required_terminals)
INSTALL = '/usr/local/share/xyce/verilog-a'
ENTRY_POINTS = [
    # BSIM-CMG family — 107 and 108 hide the module decl behind an
    # `include "bsimcmg_main.va"`; 110/111 have it directly in
    # bsimcmg.va. All four should resolve to MOSFET / corresponding level.
    ('BSIM-CMG 107', f'{INSTALL}/bsimcmg_107.0.0/code/bsimcmg.va',
        'MOSFET', '107', 4),
    ('BSIM-CMG 108', f'{INSTALL}/BSIMCMG108.0.0_20140822/code/bsimcmg.va',
        'MOSFET', '108', 4),
    ('BSIM-CMG 110', f'{INSTALL}/BSIMCMG110.0.0_20160101/code/bsimcmg.va',
        'MOSFET', '110', 4),
    ('BSIM-CMG 111', f'{INSTALL}/BSIM-CMG_111.2.1_06062022/code/bsimcmg.va',
        'MOSFET', '111', 4),
    # BSIM6 — bulk MOSFET.
    ('BSIM6',         f'{INSTALL}/BSIM6.1.1/code/BSIM6.1.1.va',
        'MOSFET', '77', 4),
    # PSP family.
    ('PSP 102',       f'{INSTALL}/psp102/psp102.va',
        'MOSFET', '102', 4),
    ('PSP 102b',      f'{INSTALL}/psp102/psp102b.va',
        'MOSFET', '102', 4),
    ('PSP 102e',      f'{INSTALL}/psp102/psp102e.va',
        'MOSFET', '102', 4),
    ('PSP 103',       f'{INSTALL}/psp103/psp103.va',
        'MOSFET', '103', 4),
    ('PSP 103t (self-heating)', f'{INSTALL}/psp103/psp103t.va',
        'MOSFET', '1031', 4),
    # BSIM-SOI family (silicon-on-insulator MOSFET, 5-terminal nominal:
    # D G S B E or similar — required terminals depends on version,
    # treat as MOSFET for device-letter dispatch).
    ('BSIM-SOI 4.5',  f'{INSTALL}/BSIM-SOI_4/bsimsoi4.5.0/bsimsoi450.va',
        'MOSFET', '70450', 4),
    ('BSIM-SOI 4.6.1', f'{INSTALL}/BSIM-SOI_4/bsimsoi4.6.1/bsimsoi461.va',
        'MOSFET', '70', 4),
    ('BSIM-SOI 4.7',  f'{INSTALL}/BSIM-SOI_4/bsimsoi4.7.0/bsimsoi.va',
        'MOSFET', '70470', 4),
    # BJT compact models.
    ('MEXTRAM bjt504', f'{INSTALL}/mextram_504.12.1/bjt504.va',
        'BJT', '504', 4),
    ('MEXTRAM bjt504t (self-heating)',
        f'{INSTALL}/mextram_504.12.1/bjt504t.va',
        'BJT', '505', 4),
    ('HICUM L2',      f'{INSTALL}/hicum/hicumL2V2p4p0.va',
        'BJT', '234', 4),
    ('HICUM L0',      f'{INSTALL}/hicum_l0/hicumL0V1p32.va',
        'BJT', '230', 4),
    ('FBH HBT 2.1',   f'{INSTALL}/fbh_hbt-2.1/fbhhbt-2.1.va',
        'BJT', '23', 4),
    # Diodes.
    ('Diode CMC 2.0', f'{INSTALL}/DIODE_CMC_2/diode_cmc_2.0.0/diode_cmc.va',
        'Diode', '2002', 2),
    # Other MOSFETs.
    ('L_UTSOI 102',   f'{INSTALL}/L_UTSOI/L_UTSOI_102/L_UTSOI_102.va',
        'MOSFET', '10240', 4),
    ('MVSG CMC 1.1',  f'{INSTALL}/mvsg_cmc_v1.1.0/vacode/mvsg_cmc_1.1.0.va',
        'MOSFET', '2002', 4),
    ('MVS 2.0',       f'{INSTALL}/mvs_2.0.0/mvs_2_0_0_etsoi.va',
        'MOSFET', '2000', 4),
    ('EKV 2.6',       f'{INSTALL}/EKV/ekv-2.6/ekv26_SDext_Verilog-A.va',
        'MOSFET', '260', 4),
    # Toys (simple test devices).
    ('toy capacitor', f'{INSTALL}/toys/capacitor.va',
        'Capacitor', '6', 2),
]


# Numeric mapping used by xyce_device_gen.py's _GROUP_REQUIRED for the
# numNodes() / numOptionalNodes() generation. Kept here so the test
# can also assert downstream behaviour will be correct.
GROUP_REQUIRED_TERMINALS = {
    'MOSFET': 4,
    'BJT':    4,
    'Diode':  2,
    'Resistor': 2,
    'Capacitor': 2,
    'Inductor': 2,
}


class TestVaAttributeExtraction(unittest.TestCase):
    """For each compact-model entry-point .va, scan_va_attributes
    should return the right xyceModelGroup and xyceLevelNumber. These
    drive the device-class registration (registerDevice("m", level),
    registerDevice("q", level), etc.) and the numNodes() value the
    netlist parser uses to split nodes-vs-params on the device card.
    """

    def _check_one(self, desc, va_path, want_group, want_level, want_req):
        if not os.path.exists(va_path):
            self.skipTest(f'{desc}: {va_path} not installed')
        attrs = scan_va_attributes(va_path)
        with self.subTest(model=desc, attr='xyceModelGroup'):
            self.assertEqual(attrs.get('xyceModelGroup'), want_group,
                f'{desc}: expected group={want_group!r}, '
                f'got attrs={attrs!r} — entry point may have the module '
                f'decl behind `include')
        with self.subTest(model=desc, attr='xyceLevelNumber'):
            self.assertEqual(attrs.get('xyceLevelNumber'), want_level,
                f'{desc}: expected level={want_level!r}, got {attrs!r}')
        with self.subTest(model=desc, attr='required_terminals'):
            mapped = GROUP_REQUIRED_TERMINALS.get(attrs.get('xyceModelGroup', ''))
            self.assertEqual(mapped, want_req,
                f'{desc}: numNodes() will dispatch to {mapped} '
                f'(want {want_req}). If group missing, the wrapper falls '
                f'through to len(mod.ports) which causes M-card node '
                f'fields to be parsed as instance parameters.')


def _add_per_entry_methods():
    """Build a separate test method per entry so failures are individually
    addressable (rather than one giant subTest dump)."""
    for desc, va_path, want_group, want_level, want_req in ENTRY_POINTS:
        sanitised = desc.replace(' ', '_').replace('.', '_').replace('/', '_').replace('-', '_')
        method_name = f'test_{sanitised}'
        def _make_test(_desc, _path, _g, _l, _r):
            def test(self):
                self._check_one(_desc, _path, _g, _l, _r)
            return test
        setattr(TestVaAttributeExtraction, method_name,
                _make_test(desc, va_path, want_group, want_level, want_req))


_add_per_entry_methods()


if __name__ == '__main__':
    unittest.main(verbosity=2)
