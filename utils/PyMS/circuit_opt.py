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


def find_darlingtons(bjts: Dict[str, BJT]) -> List[Tuple[str, str]]:
    """Find Darlington pairs: driver.e = output.b, same collector, same type."""
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
                pairs.append((na, nb))
                used.update([na, nb])
                break
            # b is driver, a is output
            if b.e == a.b and b.c == a.c:
                pairs.append((nb, na))
                used.update([na, nb])
                break

    return pairs


def find_mirrors(bjts: Dict[str, BJT]) -> List[Tuple[str, str]]:
    """Find current mirrors: diode-connected ref + mirror with shared base+emitter."""
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
                mirrors.append((na, nb))
                used.update([na, nb])

    return mirrors


def generate_darlington_va(model_type: str) -> str:
    """Generate Verilog-A for a Darlington pair."""
    is_pnp = model_type.upper() in ('PNP', 'PN')
    sign = "-" if is_pnp else ""
    vbe_sign = "V(e, b)" if is_pnp else "V(b, e)"
    vce_sign = "V(e, c)" if is_pnp else "V(c, e)"
    ice_dir = "I(e, c)" if is_pnp else "I(c, e)"
    ibe_dir = "I(e, b)" if is_pnp else "I(b, e)"

    return f"""`include "disciplines.vams"
`include "constants.vams"
module darlington_{model_type.lower()}(c, b, e);
    inout c, b, e;
    electrical c, b, e;
    parameter real BF1 = 100.0;
    parameter real BF2 = 100.0;
    parameter real IS1 = 1e-14;
    parameter real IS2 = 1e-14;
    real Vbe, Vce, Ib1, Ic1, Ib2, Ic2, Vbe1, Vbe2;
    analog begin
        Vbe = {vbe_sign};
        Vce = {vce_sign};
        // Driver: Q1 sees full Vbe, emitter drives Q2 base
        // Effective Vbe for Q1 = Vbe - Vbe2 (roughly Vbe - 0.65)
        Vbe2 = $vt * ln(1.0 + Vbe / (2.0 * $vt));
        if (Vbe2 > 0.8) Vbe2 = 0.8;
        Vbe1 = Vbe - Vbe2;
        // Q1 (driver)
        Ic1 = IS1 * (limexp(Vbe1 / $vt) - 1.0);
        Ib1 = Ic1 / BF1;
        // Q2 (output) — base current = Q1 emitter current
        Ic2 = IS2 * (limexp(Vbe2 / $vt) - 1.0);
        Ib2 = Ic2 / BF2;
        // Total: Ic = Ic1 + Ic2, Ib = Ib1
        {ice_dir} <+ Ic1 + Ic2;
        {ibe_dir} <+ Ib1;
    end
endmodule
"""


def generate_mirror_va(model_type: str) -> str:
    """Generate Verilog-A for a current mirror (no internal nodes)."""
    is_pnp = model_type.upper() in ('PNP', 'PN')

    if is_pnp:
        vbe = "V(e, cref)"  # diode-connected: Vbe = V(e,c) for ref
        i_ref = "I(e, cref)"
        i_out = "I(e, cout)"
        vce_out = "V(e, cout)"
    else:
        vbe = "V(cref, e)"
        i_ref = "I(cref, e)"
        i_out = "I(cout, e)"
        vce_out = "V(cout, e)"

    return f"""`include "disciplines.vams"
`include "constants.vams"
module mirror_{model_type.lower()}(cref, cout, e);
    inout cref, cout, e;
    electrical cref, cout, e;
    parameter real BF = 100.0;
    parameter real IS = 1e-14;
    parameter real VAF = 100.0;
    real Iref, Iout;
    analog begin
        // Diode-connected ref: base=collector, so Vbe=Vce for reference
        Iref = IS * (limexp({vbe} / $vt) - 1.0);
        {i_ref} <+ Iref + Iref / BF;
        // Mirror output with Early effect
        Iout = Iref * (1.0 + {vce_out} / VAF);
        {i_out} <+ Iout;
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
                    params[pm.group(1).upper()] = float(pm.group(2))
                except ValueError:
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

    # Find patterns
    darlingtons = find_darlingtons(bjts)
    mirrors = find_mirrors(bjts)

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
        is_val = mp.get('IS', 1e-14)
        darlington_params[(driver, output_q)] = f"BF1={bf} BF2={bf} IS1={is_val} IS2={is_val}"

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
        is_val = mp.get('IS', 1e-14)
        vaf = mp.get('VAF', 100.0)
        mirror_params[(ref, mir)] = f"BF={bf} IS={is_val} VAF={vaf}"

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
