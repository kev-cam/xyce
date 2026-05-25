"""Tie each PARAM_M regression failure back to a specific PyMS bug class.

The two upstream tests in this directory characterise PyMS bugs in
isolation:

  test_va_attributes.py      — scan_va_attributes() can't find
                                xyceModelGroup for 11 of 24 .va entries
                                (`include`d modules, `\\`MACRO attr forms,
                                attrs before the module keyword, etc.)
  test_va_terminal_split.py  — generate_device_cpp() emits
                                numNodes()=len(declared_ports),
                                numOptionalNodes()=0 unconditionally,
                                wrong for any 5+-port MOSFET/BJT/diode

This test closes the loop: it walks an Xyce regression-suite results
file and asserts every PARAM_M failure (i.e. ``Unrecognized parameter X
for device Y``) on a recognised compact-model device can be attributed
to at least one PyMS bug class:

  (a) attr-extraction broken
  (b) terminal-split wrong
  (c) parser raises before yielding a module
  (d) parser yields a module with zero instance parameters

If a PARAM_M failure can't be attributed, the underlying bug is
separate from the four classes above and warrants investigation before
the codegen-rewrite work.

Without the bug-class attribution, "the param parsing is broken" is
unfalsifiable hand-waving; with it, fixing each bug class should clear
a known set of regression tests, and a regression that DOESN'T clear
points at a fifth bug class.

To regenerate the results file:
    run_ctest.pl -j 16 -o /tmp/regress_<rev>.txt /tmp/XyceTesting-dev

The test reads from $XYCE_REGRESSION_RESULTS or, if unset, the most
recent /tmp/regress_*.txt; it skips if no results file is found.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import unittest
from collections import defaultdict

_VAE_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VAE_PKG_DIR not in sys.path:
    sys.path.insert(0, _VAE_PKG_DIR)

from vae.parser import parse_file  # noqa: E402
from vae.xyce_device_gen import (  # noqa: E402
    scan_va_attributes, generate_device_cpp,
)
from vae.tests.test_va_attributes import (  # noqa: E402
    ENTRY_POINTS, GROUP_REQUIRED_TERMINALS,
)


# Maps an Xyce regression-suite directory to the ENTRY_POINTS desc(s)
# whose .va files it exercises. Built from inspecting the .HDL lines
# (and level= dispatch) of failing .cir files at fa19579b.
#
# Some directories exercise multiple models (notably VERILOG_MODEL_BINNING
# and Verilog_LEAD_CURRENTS — they sweep every BSIM-CMG level and BSIM6).
DIR_TO_ENTRIES = {
    'B4SOI':         ['BSIM-SOI 4.7'],
    'B4SOI_450':     ['BSIM-SOI 4.5'],
    'B4SOI_461':     ['BSIM-SOI 4.6.1'],
    'BSIM6':         ['BSIM6'],
    'BSIMCMG':       ['BSIM-CMG 107'],
    'BSIMCMG_107':   ['BSIM-CMG 107'],
    'BSIMCMG_108':   ['BSIM-CMG 108'],
    'BSIMCMG_110':   ['BSIM-CMG 110'],
    'BSIMCMG_111':   ['BSIM-CMG 111'],
    'PSP102':        ['PSP 102', 'PSP 102b', 'PSP 102e'],
    'PSP103':        ['PSP 103', 'PSP 103t (self-heating)'],
    'L_UTSOI':       ['L_UTSOI 102'],
    'EKV26':         ['EKV 2.6'],
    'MEXTRAM':       ['MEXTRAM bjt504', 'MEXTRAM bjt504t (self-heating)'],
    'HICUM':         ['HICUM L2', 'HICUM L0'],
    'DIODE_CMC':     ['Diode CMC 2.0'],
    'MVS2':          ['MVS 2.0'],
    'VERILOG_MODEL_BINNING': ['BSIM6', 'BSIM-CMG 107', 'BSIM-CMG 108',
                              'BSIM-CMG 110', 'BSIM-CMG 111'],
    'Verilog_LEAD_CURRENTS': ['BSIM6', 'BSIM-CMG 107', 'PSP 103'],
}

# Directories whose PARAM_M failures are on devices NOT backed by any
# .va in ENTRY_POINTS — out of scope for this test, but tracked so the
# "unattributed" bucket only contains genuinely-unexplained failures.
OUT_OF_SCOPE_DIRS = {
    # Y-device extension models (custom Verilog-A outside the catalog)
    'NEURON',
    # Built-in Xyce model levels (VBIC11 LEVEL=11 NPN, etc.)
    'VBIC13',
    # Mixed bag — some are VBIC level=11, some are diode level=1, etc.
    'Certification_Tests',
}


def _find_results_file():
    """Locate the regression-results TSV (skip-source for the test).
    Honours XYCE_REGRESSION_RESULTS, falls back to /tmp/regress_*.txt
    (most recent), then None."""
    env = os.environ.get('XYCE_REGRESSION_RESULTS')
    if env and os.path.exists(env):
        return env
    candidates = sorted(glob.glob('/tmp/regress_*.txt'),
                        key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


# Build the desc → bug-class map at module load. Each entry is checked
# against the four bug classes; an entry can be in multiple classes.
def _compute_bug_classes():
    """Return {entry_desc: set_of_bug_class_codes}.

    Codes:
      'a' attr_extraction_broken
      'b' terminal_split_wrong
      'c' parser_raises
      'd' parser_yields_zero_instance_params
    """
    result = {}
    for desc, va_path, want_group, want_level, want_req in ENTRY_POINTS:
        classes = set()
        if not os.path.exists(va_path):
            result[desc] = {'missing'}
            continue
        attrs = scan_va_attributes(va_path)
        if (attrs.get('xyceModelGroup') != want_group
                or attrs.get('xyceLevelNumber') != want_level):
            classes.add('a')
        try:
            mod = parse_file(va_path)
        except Exception:
            classes.add('c')
            result[desc] = classes
            continue
        n_inst = sum(1 for p in mod.params
                     if getattr(p, 'is_instance', False))
        if n_inst == 0:
            classes.add('d')
        try:
            header, _impl = generate_device_cpp(mod, '', va_path=va_path)
            nn = int(re.search(r'numNodes\(\)\s*\{\s*return\s+(\d+)',
                               header).group(1))
            no = int(re.search(r'numOptionalNodes\(\)\s*\{\s*return\s+(\d+)',
                               header).group(1))
            n_ports = len(mod.ports)
            want_opt = max(0, n_ports - want_req)
            if nn != want_req or no != want_opt:
                classes.add('b')
        except Exception:
            # Codegen blew up — count as 'c' (parser/codegen pipeline failed)
            classes.add('c')
        result[desc] = classes
    return result


BUG_CLASSES = _compute_bug_classes()


def _parse_param_m_failures(results_path):
    """Walk the results TSV; for each FAIL whose log contains
    'Unrecognized parameter X for device Y', return a list of dicts:

      {'name': ..., 'dir': ..., 'device_letter': 'M'/'Q'/'D'/'Y'/...,
       'params': set(rejected param tokens),
       'vas': list of .va paths mentioned in the log }
    """
    failures = []
    with open(results_path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4 or parts[0] != 'FAIL':
                continue
            name, log_path = parts[1], parts[2]
            if not os.path.exists(log_path):
                continue
            try:
                with open(log_path, errors='replace') as lf:
                    text = lf.read()
            except OSError:
                continue
            if 'Unrecognized parameter' not in text:
                continue
            # Skip if it's only "Unrecognized parameter" without "for device"
            offenders = re.findall(
                r'Unrecognized parameter (\w+) for device ([A-Za-z][\w!]*)',
                text)
            if not offenders:
                continue
            dirpart = name.split('/', 1)[0]
            params = {p for p, _ in offenders}
            device_letter = offenders[0][1][0].upper()
            vas = sorted(set(re.findall(
                r'(/usr/local/share/xyce/verilog-a/[^\s\'"]+\.va)\b', text)))
            failures.append({
                'name': name, 'dir': dirpart,
                'device_letter': device_letter,
                'params': params, 'vas': vas,
            })
    return failures


def _va_path_to_entry_desc(va_path):
    for desc, p, *_ in ENTRY_POINTS:
        if p == va_path:
            return desc
    return None


class TestParamMAttribution(unittest.TestCase):
    """Every PARAM_M regression failure on an in-catalog .va should be
    explained by at least one PyMS bug class (a/b/c/d). Failures on
    built-in or out-of-scope devices are reported but not asserted."""

    @classmethod
    def setUpClass(cls):
        cls.results_path = _find_results_file()
        if cls.results_path is None:
            raise unittest.SkipTest(
                'No regression-results file found '
                '(set XYCE_REGRESSION_RESULTS or run run_ctest.pl)')
        cls.failures = _parse_param_m_failures(cls.results_path)

    def test_every_in_catalog_failure_is_attributable(self):
        """For each PARAM_M failure whose log/dir maps to an
        ENTRY_POINTS entry: the entry must currently have a bug-class
        that explains the device-card rejection (a, b, c, or d).
        Failures with NO bug-class are the ones to investigate before
        any codegen rewrite."""
        unattributed = []
        per_entry_count = defaultdict(int)
        per_entry_classes = {}

        for f in self.failures:
            if f['dir'] in OUT_OF_SCOPE_DIRS:
                continue
            # Candidate entries: prefer .va paths from the log, fall
            # back to the directory map.
            cand_descs = []
            for va in f['vas']:
                d = _va_path_to_entry_desc(va)
                if d:
                    cand_descs.append(d)
            if not cand_descs:
                cand_descs = DIR_TO_ENTRIES.get(f['dir'], [])
            if not cand_descs:
                unattributed.append(
                    (f['name'], 'no entry mapping for dir + log'))
                continue
            # If ANY candidate entry has any bug class, the failure is
            # attributable.
            attributable_to = []
            for d in cand_descs:
                cls_set = BUG_CLASSES.get(d, set())
                per_entry_count[d] += 1
                per_entry_classes[d] = cls_set
                if cls_set & {'a', 'b', 'c', 'd'}:
                    attributable_to.append((d, sorted(cls_set)))
            if not attributable_to:
                unattributed.append(
                    (f['name'],
                     f'candidates {cand_descs!r} all have no bug class '
                     f'(params rejected: {sorted(f["params"])})'))

        # Diagnostic report — printed even on PASS so a successful run
        # documents the current attribution surface.
        sys.stderr.write(
            '\n=== PARAM_M attribution (results: %s) ===\n' %
            self.results_path)
        sys.stderr.write(
            '%-32s %-20s %s\n' % ('entry', 'bug classes', 'PARAM_M failures'))
        for d in sorted(per_entry_count):
            sys.stderr.write('%-32s %-20s %d\n' % (
                d, ','.join(sorted(per_entry_classes[d])) or '(none)',
                per_entry_count[d]))
        sys.stderr.write('Total in-catalog failures attributed: %d\n' %
                         sum(per_entry_count.values()))
        sys.stderr.write('Total unattributed: %d\n' % len(unattributed))

        if unattributed:
            msg = ['Unattributed PARAM_M failures (need new bug class):']
            for name, reason in unattributed[:25]:
                msg.append(f'  {name}: {reason}')
            if len(unattributed) > 25:
                msg.append(f'  ... and {len(unattributed) - 25} more')
            self.fail('\n'.join(msg))

    def test_out_of_scope_dirs_are_reported(self):
        """Sanity check: PARAM_M failures in OUT_OF_SCOPE_DIRS should
        exist (otherwise the directory probably moved into scope and
        the map is stale)."""
        seen = defaultdict(int)
        for f in self.failures:
            if f['dir'] in OUT_OF_SCOPE_DIRS:
                seen[f['dir']] += 1
        # Don't fail — just report. Empty buckets are fine if those
        # tests passed.
        if seen:
            sys.stderr.write('\n=== Out-of-scope PARAM_M (not asserted) ===\n')
            for d, n in sorted(seen.items()):
                sys.stderr.write('  %-32s %d\n' % (d, n))

    def test_bug_class_distribution(self):
        """Print which bug class(es) each entry currently triggers.
        This is the working surface for the fix work."""
        sys.stderr.write('\n=== ENTRY_POINTS bug-class status ===\n')
        sys.stderr.write('  a = attr_extraction_broken  '
                         'b = terminal_split_wrong\n'
                         '  c = parser_raises           '
                         'd = parser_yields_zero_instance_params\n')
        for desc, va, *_ in ENTRY_POINTS:
            cs = BUG_CLASSES.get(desc, set())
            status = ','.join(sorted(cs)) if cs else 'clean'
            sys.stderr.write('  %-32s %s\n' % (desc, status))


if __name__ == '__main__':
    unittest.main(verbosity=2)
