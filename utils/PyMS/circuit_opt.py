#!/usr/bin/env python3
"""
circuit_opt.py — Circuit-level device merging optimizer.

Detects common transistor-level patterns (Darlington pairs, current mirrors)
and replaces them with equivalent merged Verilog-A devices to reduce matrix
size and improve simulation speed.

Usage:
    circuit_opt.py input.cir -o output.cir [--report]
"""

import re
import os
import sys


def _parse_eng_val(s: str) -> float:
    """Parse SPICE engineering notation: 1k→1e3, 5p→5e-12, etc."""
    suffixes = {
        'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'K': 1e3,
        'M': 1e-3, 'U': 1e-6, 'N': 1e-9, 'P': 1e-12, 'F': 1e-15,
    }
    s = s.strip()
    m = re.match(r'^([+-]?[\d.]+(?:e[+-]?\d+)?)\s*(meg|[tgkmunpf])?', s, re.I)
    if m:
        num = float(m.group(1))
        suf = m.group(2)
        if suf:
            mult = suffixes.get(suf.upper())
            if mult:
                return num * mult
        return num
    return float(s)
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional


@dataclass
class BJT:
    name: str
    c: str    # collector
    b: str    # base
    e: str    # emitter
    bulk: str
    model: str
    params: str = ''


def parse_netlist(lines: List[str]) -> Tuple[Dict[str, BJT], List[str], List[str]]:
    """Parse netlist, extract BJTs and other lines."""
    bjts = {}
    resistors = {}  # name → (n1, n2, value)
    other_lines = []

    for line in lines:
        s = line.strip()
        # BJT: Q<name> collector base emitter [bulk] model [params]
        m = re.match(r'^(Q\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(.*)', s, re.I)
        if m:
            bjts[m.group(1)] = BJT(
                name=m.group(1), c=m.group(2), b=m.group(3),
                e=m.group(4), bulk=m.group(5), model=m.group(6),
                params=m.group(7).strip()
            )
            continue

        # Track resistors for Darlington base-emitter resistor detection
        m = re.match(r'^(R\S+)\s+(\S+)\s+(\S+)\s+(\S+)', s, re.I)
        if m:
            resistors[m.group(1)] = (m.group(2), m.group(3), m.group(4))

        other_lines.append(line)

    return bjts, resistors, other_lines


def find_darlington_pairs(bjts: Dict[str, BJT]) -> List[Tuple[str, str]]:
    """Find Darlington pairs: Q_driver emitter = Q_output base, same collector."""
    pairs = []
    found = set()
    names = list(bjts.keys())

    for i, na in enumerate(names):
        if na in found:
            continue
        a = bjts[na]
        for nb in names[i+1:]:
            if nb in found:
                continue
            b = bjts[nb]
            if a['type_'] != b['type_']:
                continue
            # a drives b: a.e = b.b, a.c = b.c
            if a.e == b.b and a.c == b.c:
                pairs.append((na, nb))  # driver, output
                found.add(na)
                found.add(nb)
                break
            # b drives a: b.e = a.b, b.c = a.c
            if b.e == a.b and b.c == a.c:
                pairs.append((nb, na))
                found.add(na)
                found.add(nb)
                break

    return pairs


def _node_connections(bjts: Dict[str, BJT], lines: List[str]) -> Dict[str, int]:
    """Count how many device terminals connect to each node."""
    counts: Dict[str, int] = {}
    for b in bjts.values():
        for n in (b.c, b.b, b.e):
            counts[n] = counts.get(n, 0) + 1
    # Also count non-BJT device connections from the netlist
    for line in lines:
        s = line.strip()
        if not s or s.startswith('*') or s.startswith('.'):
            continue
        if s[0].upper() == 'Q':
            continue  # already counted
        # Parse device line: first letter + name, then node names
        parts = s.split()
        if len(parts) >= 3 and parts[0][0].isalpha():
            for p in parts[1:]:
                if p.startswith('{') or '=' in p:
                    break
                # Skip model names / values (heuristic: if it looks like a node)
                if re.match(r'^[A-Za-z0-9_+\-]+$', p):
                    counts[p] = counts.get(p, 0) + 1
    return counts


def find_darlingtons(bjts: Dict[str, BJT],
                     node_conns: Dict[str, int] = None) -> List[Tuple[str, str]]:
    """Find Darlington pairs: driver.e = output.b, same collector, same type.

    Only merge if the shared node (driver emitter = output base) has no
    other connections besides the two BJTs being merged.
    """
    pairs = []
    used = set()
    names = list(bjts.keys())

    for i, na in enumerate(names):
        if na in used:
            continue
        a = bjts[na]
        for nb in names[i+1:]:
            if nb in used:
                continue
            b = bjts[nb]
            if a.model != b.model:
                continue
            # a is driver (input), b is output: a.e = b.b, same collector
            if a.e == b.b and a.c == b.c:
                # Check shared node has only 2 BJT connections
                shared = a.e
                if node_conns and node_conns.get(shared, 0) > 2:
                    continue  # external connections — can't internalize
                pairs.append((na, nb))
                used.update([na, nb])
                break
            # b is driver, a is output
            if b.e == a.b and b.c == a.c:
                shared = b.e
                if node_conns and node_conns.get(shared, 0) > 2:
                    continue
                pairs.append((nb, na))
                used.update([na, nb])
                break

    return pairs


def find_mirrors(bjts: Dict[str, BJT],
                 node_conns: Dict[str, int] = None) -> List[Tuple[str, str]]:
    """Find current mirrors: diode-connected ref + mirror with shared base+emitter.

    The shared base node (= ref collector for diode-connected) must have
    no external connections besides the BJTs being merged.
    """
    mirrors = []
    used = set()
    names = list(bjts.keys())

    for i, na in enumerate(names):
        if na in used:
            continue
        a = bjts[na]
        # Check if a is diode-connected (base = collector)
        if a.b != a.c:
            continue
        # Find mirror transistors sharing base and emitter
        for nb in names:
            if nb == na or nb in used:
                continue
            b = bjts[nb]
            if b.b == a.b and b.e == a.e and b.model == a.model:
                # Shared base node: count connections
                # For diode-connected, base=collector, so it has at least 3 BJT
                # terminal connections (a.b, a.c, b.b). Only merge if no others.
                shared = a.b
                bjt_conns = sum(1 for bj in bjts.values()
                               for n in (bj.c, bj.b, bj.e) if n == shared)
                if node_conns and node_conns.get(shared, 0) > bjt_conns:
                    continue  # external connections to shared base
                mirrors.append((na, nb))
                used.update([na, nb])

    return mirrors


def _bjt_analog_block(prefix: str, c: str, b: str, e: str, is_pnp: bool) -> str:
    """Generate Ebers-Moll BJT equations for one transistor.

    prefix: variable name prefix (e.g. 'q1_' or 'q2_')
    c, b, e: node names in the merged module
    """
    if is_pnp:
        vbe = f"V({e}, {b})"
        vbc = f"V({e}, {c})"
        vce = f"V({e}, {c})"
        ice = f"I({e}, {c})"
        ibe = f"I({e}, {b})"
        ibc = f"I({e}, {c})"
    else:
        vbe = f"V({b}, {e})"
        vbc = f"V({b}, {c})"
        vce = f"V({c}, {e})"
        ice = f"I({c}, {e})"
        ibe = f"I({b}, {e})"
        ibc = f"I({b}, {c})"

    return f"""
        // BJT {prefix} Ebers-Moll
        {prefix}Vbe = {vbe};
        {prefix}Vbc = {vbc};
        {prefix}If = {prefix}IS * (limexp({prefix}Vbe / $vt) - 1.0);
        {prefix}Ir = {prefix}IS * (limexp({prefix}Vbc / $vt) - 1.0);
        {prefix}Ic = {prefix}If * (1.0 + {vce} / {prefix}VAF) - {prefix}Ir;
        {ice} <+ {prefix}Ic;
        {ibe} <+ {prefix}If / {prefix}BF;
        {ibc} <+ {prefix}Ir / {prefix}BR;
        {ibe} <+ {prefix}CJE * ddt({prefix}Vbe);
        {ibc} <+ {prefix}CJC * ddt({prefix}Vbc);
"""


def generate_darlington_va(model_type: str) -> str:
    """Generate Verilog-A for a Darlington pair.

    Two full Ebers-Moll BJTs with internal node 'm' (driver emitter = output base).
    External: c (shared collector), b (driver base), e (output emitter).
    """
    is_pnp = model_type.upper() in ('PNP', 'PN')

    q1_block = _bjt_analog_block('q1_', 'c', 'b', 'm', is_pnp)
    q2_block = _bjt_analog_block('q2_', 'c', 'm', 'e', is_pnp)

    return f"""`include "disciplines.vams"
`include "constants.vams"
module darlington_{model_type.lower()}(c, b, e);
    inout c, b, e;
    electrical c, b, e;
    electrical m;  // internal: driver emitter = output base

    // Driver (Q1) parameters
    parameter real q1_IS  = 1e-14;
    parameter real q1_BF  = 100.0;
    parameter real q1_BR  = 1.0;
    parameter real q1_VAF = 1e10;
    parameter real q1_CJE = 0.0;
    parameter real q1_CJC = 0.0;

    // Output (Q2) parameters
    parameter real q2_IS  = 1e-14;
    parameter real q2_BF  = 100.0;
    parameter real q2_BR  = 1.0;
    parameter real q2_VAF = 1e10;
    parameter real q2_CJE = 0.0;
    parameter real q2_CJC = 0.0;

    real q1_Vbe, q1_Vbc, q1_If, q1_Ir, q1_Ic;
    real q2_Vbe, q2_Vbc, q2_If, q2_Ir, q2_Ic;

    analog begin
{q1_block}
{q2_block}
    end
endmodule
"""


def generate_mirror_va(model_type: str) -> str:
    """Generate Verilog-A for a current mirror.

    Two full Ebers-Moll BJTs: ref is diode-connected (base=collector),
    mirror shares base with ref. Internal node 'bp' (shared base point).
    External: cref (ref collector), cout (mirror collector), e (shared emitter).
    """
    is_pnp = model_type.upper() in ('PNP', 'PN')

    # Ref BJT: collector=cref, base=bp, emitter=e
    # Diode connection: V(bp, cref) = 0
    q1_block = _bjt_analog_block('q1_', 'cref', 'bp', 'e', is_pnp)
    # Mirror BJT: collector=cout, base=bp, emitter=e
    q2_block = _bjt_analog_block('q2_', 'cout', 'bp', 'e', is_pnp)

    # For the diode connection, short base to collector of ref
    if is_pnp:
        short = "V(bp, cref) <+ 0.0;"
    else:
        short = "V(bp, cref) <+ 0.0;"

    return f"""`include "disciplines.vams"
`include "constants.vams"
module mirror_{model_type.lower()}(cref, cout, e);
    inout cref, cout, e;
    electrical cref, cout, e;
    electrical bp;  // internal: shared base

    // Ref BJT parameters
    parameter real q1_IS  = 1e-14;
    parameter real q1_BF  = 100.0;
    parameter real q1_BR  = 1.0;
    parameter real q1_VAF = 1e10;
    parameter real q1_CJE = 0.0;
    parameter real q1_CJC = 0.0;

    // Mirror BJT parameters
    parameter real q2_IS  = 1e-14;
    parameter real q2_BF  = 100.0;
    parameter real q2_BR  = 1.0;
    parameter real q2_VAF = 1e10;
    parameter real q2_CJE = 0.0;
    parameter real q2_CJC = 0.0;

    real q1_Vbe, q1_Vbc, q1_If, q1_Ir, q1_Ic;
    real q2_Vbe, q2_Vbc, q2_If, q2_Ir, q2_Ic;

    analog begin
        // Diode connection: base tied to ref collector
        {short}
{q1_block}
{q2_block}
    end
endmodule
"""


def optimize_circuit(input_path: str, output_path: str,
                     report: bool = False) -> dict:
    """Optimize a circuit by merging device patterns."""
    with open(input_path, 'r') as f:
        lines = f.readlines()

    bjts = {}
    other_lines = []
    bjt_lines = {}  # name → original line
    model_params = {}  # model_name → {param: value}

    for line in lines:
        s = line.strip()
        # Parse .model cards for BJT parameters
        mm = re.match(r'^\.model\s+(\S+)\s+(?:NPN|PNP)\s*\(([^)]*)\)', s, re.I)
        if mm:
            mname = mm.group(1)
            params = {}
            for pm in re.finditer(r'(\w+)\s*=\s*([^\s,)]+)', mm.group(2)):
                try:
                    params[pm.group(1).upper()] = _parse_eng_val(pm.group(2))
                except (ValueError, TypeError):
                    pass
            model_params[mname] = params

        m = re.match(r'^(Q\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(.*)', s, re.I)
        if m:
            bjts[m.group(1)] = BJT(
                name=m.group(1), c=m.group(2), b=m.group(3),
                e=m.group(4), bulk=m.group(5), model=m.group(6),
                params=m.group(7).strip()
            )
            bjt_lines[m.group(1)] = line
        else:
            other_lines.append(line)

    # Compute node connectivity for safe merging
    node_conns = _node_connections(bjts, lines)

    # Find patterns (only merge if shared nodes have no external connections)
    darlingtons = find_darlingtons(bjts, node_conns)
    mirrors = find_mirrors(bjts, node_conns)

    stats = {
        'darlingtons': len(darlingtons),
        'mirrors': len(mirrors),
        'bjts_before': len(bjts),
        'nodes_saved': len(darlingtons) + len(mirrors),
    }

    if report:
        print(f"  BJTs: {len(bjts)}")
        print(f"  Darlington pairs: {len(darlingtons)}")
        for d, o in darlingtons:
            print(f"    {d} (driver) + {o} (output)")
        print(f"  Current mirrors: {len(mirrors)}")
        for r, m in mirrors:
            print(f"    {r} (ref) + {m} (mirror)")
        print(f"  Estimated nodes saved: {stats['nodes_saved']}")

    if not darlingtons and not mirrors:
        # Nothing to optimize — copy as-is
        with open(output_path, 'w') as f:
            f.writelines(lines)
        return stats

    # Generate VA files
    outdir = Path(output_path).parent
    va_files = set()
    merged = set()  # BJT names that were merged

    # Build output
    output = []
    hdl_added = set()

    # Process Darlington merges
    darlington_params = {}  # (driver, output) → param string
    for driver, output_q in darlingtons:
        d = bjts[driver]
        o = bjts[output_q]
        model_type = d.model
        va_name = f"darlington_{model_type.lower()}"

        if va_name not in va_files:
            va_content = generate_darlington_va(model_type)
            va_path = outdir / f"{va_name}.va"
            va_path.write_text(va_content)
            va_files.add(va_name)

        # Get model params for this BJT type
        mp = model_params.get(d.model, {})
        bf = mp.get('BF', 100.0)
        is_val = mp.get('IS', mp.get('Is', mp.get('is', 1e-16)))
        vaf = mp.get('VAF', 1e10)
        cje = mp.get('CJE', mp.get('Cje', mp.get('cje', 0.0)))
        cjc = mp.get('CJC', mp.get('Cjc', mp.get('cjc', 0.0)))
        darlington_params[(driver, output_q)] = (
            f"q1_BF={bf} q1_IS={is_val} q1_VAF={vaf} q1_CJE={cje} q1_CJC={cjc} "
            f"q2_BF={bf} q2_IS={is_val} q2_VAF={vaf} q2_CJE={cje} q2_CJC={cjc}"
        )

        merged.add(driver)
        merged.add(output_q)

    # Process mirror merges
    mirror_params = {}
    for ref, mir in mirrors:
        r = bjts[ref]
        m_bjt = bjts[mir]
        model_type = r.model
        va_name = f"mirror_{model_type.lower()}"

        if va_name not in va_files:
            va_content = generate_mirror_va(model_type)
            va_path = outdir / f"{va_name}.va"
            va_path.write_text(va_content)
            va_files.add(va_name)

        mp = model_params.get(r.model, {})
        bf = mp.get('BF', 100.0)
        is_val = mp.get('IS', mp.get('Is', mp.get('is', 1e-16)))
        vaf = mp.get('VAF', 1e10)
        cje = mp.get('CJE', mp.get('Cje', mp.get('cje', 0.0)))
        cjc = mp.get('CJC', mp.get('Cjc', mp.get('cjc', 0.0)))
        mirror_params[(ref, mir)] = (
            f"q1_BF={bf} q1_IS={is_val} q1_VAF={vaf} q1_CJE={cje} q1_CJC={cjc} "
            f"q2_BF={bf} q2_IS={is_val} q2_VAF={vaf} q2_CJE={cje} q2_CJC={cjc}"
        )

        merged.add(ref)
        merged.add(mir)

    # Write output netlist
    # Add .hdl directives after title line
    title_done = False
    for line in lines:
        s = line.strip()

        if not title_done and not s.startswith('.') and not s.startswith('*'):
            # First non-comment, non-directive line is probably past the title
            title_done = True

        # Add .hdl lines after first line
        if title_done and not hdl_added:
            for va_name in sorted(va_files):
                va_abs = (outdir / f"{va_name}.va").resolve()
                output.append(f'.hdl "{va_abs}"\n')
            hdl_added = True  # use truthy set as flag

        # Check if this is a merged BJT line
        m = re.match(r'^(Q\S+)\s', s, re.I)
        if m and m.group(1) in merged:
            # Skip — will be replaced below
            continue

        output.append(line)

    # Insert merged device lines before .end
    insert_lines = []
    for driver, output_q in darlingtons:
        d = bjts[driver]
        o = bjts[output_q]
        va_name = f"darlington_{d.model.lower()}"
        # Darlington: c=shared collector, b=driver base, e=output emitter
        inst_name = f"DARL_{driver}_{output_q}"
        mod_name = f"dmod_{driver}_{output_q}".lower()
        dp = darlington_params.get((driver, output_q), '')
        insert_lines.append(
            f"* [merged Darlington: {driver} + {output_q}]\n"
            f".model {mod_name} {va_name} {dp}\n"
            f"Y{va_name.upper()} {inst_name} {d.c} {d.b} {o.e} {mod_name}\n"
        )

    for ref, mir in mirrors:
        r = bjts[ref]
        m_bjt = bjts[mir]
        va_name = f"mirror_{r.model.lower()}"
        inst_name = f"MIR_{ref}_{mir}"
        mod_name = f"mmod_{ref}_{mir}".lower()
        mp = mirror_params.get((ref, mir), '')
        insert_lines.append(
            f"* [merged mirror: {ref} (ref) + {mir} (mirror)]\n"
            f".model {mod_name} {va_name} {mp}\n"
            f"Y{va_name.upper()} {inst_name} {r.c} {m_bjt.c} {r.e} {mod_name}\n"
        )

    # Insert before .end
    final = []
    for line in output:
        if line.strip().upper() == '.END':
            for il in insert_lines:
                final.append(il)
        final.append(line)

    with open(output_path, 'w') as f:
        f.writelines(final)

    stats['bjts_after'] = len(bjts) - len(merged)
    return stats


def main():
    p = argparse.ArgumentParser(description='Circuit-level device merger')
    p.add_argument('input', help='Input .cir file')
    p.add_argument('-o', '--output', required=True, help='Output .cir file')
    p.add_argument('--report', action='store_true', help='Print optimization report')
    args = p.parse_args()

    stats = optimize_circuit(args.input, args.output, report=args.report)
    print(f"Optimized: {stats['darlingtons']} Darlingtons, "
          f"{stats['mirrors']} mirrors merged "
          f"({stats.get('bjts_before', 0)} → {stats.get('bjts_after', stats.get('bjts_before', 0))} BJTs)")


if __name__ == '__main__':
    main()
