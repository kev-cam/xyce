#!/usr/bin/env python3
"""
xyce_to_vhdl - Import Xyce SPICE subcircuits as VHDL entity wrappers

Parses a Xyce netlist, extracts .SUBCKT definitions, and generates
VHDL entities that carry the subcircuit text via spice_subckt().
Each subcircuit port becomes a std_logic signal on the VHDL entity.

Usage:
    xyce_to_vhdl.py <netlist.cir> [-o outdir] [--subckt NAME]

Output:
    One .vhd file per subcircuit (or one for a specific --subckt).
    Each file contains an entity with ports matching the SUBCKT ports
    and a concurrent spice_subckt() call carrying the full SPICE text.

The generated VHDL is meant to be analyzed by NVC alongside the
spice_subckt_pkg package.  At elaboration/run time, the cosim driver
reads the subcircuit text back and feeds it to Xyce.
"""

import argparse
import os
import re
import sys
import textwrap


def parse_subckt_params(param_str):
    """Parse PARAMS: key=val pairs from a .SUBCKT line."""
    params = {}
    for m in re.finditer(r'(\w+)\s*=\s*([^\s,]+)', param_str):
        params[m.group(1)] = m.group(2)
    return params


def parse_netlist(lines):
    """Extract .SUBCKT blocks and top-level preamble from netlist lines.

    Returns:
        subckts: list of dicts (name, ports, params, text)
        preamble: top-level lines outside subcircuits (.MODEL, .PARAM, etc.)
    """
    subckts = []
    preamble_lines = []
    current = None
    text_lines = []
    subckt_header = ''
    in_header = False

    for line in lines:
        stripped = line.rstrip('\n\r')
        upper = stripped.upper().lstrip()

        if in_header and stripped.lstrip().startswith('+'):
            subckt_header += ' ' + stripped.lstrip()[1:].lstrip()
            text_lines.append(stripped)
            continue
        else:
            in_header = False

        if upper.startswith('.SUBCKT'):
            text_lines = [stripped]
            subckt_header = stripped
            in_header = True

            current = {
                'name': None,
                'ports': [],
                'params': {},
                'text': None,
            }

        elif upper.startswith('.ENDS') and current is not None:
            text_lines.append(stripped)

            rest = subckt_header[subckt_header.upper().index('.SUBCKT') + 7:].strip()

            params_idx = rest.upper().find('PARAMS:')
            if params_idx >= 0:
                port_part = rest[:params_idx].strip()
                param_part = rest[params_idx + 7:]
                current['params'] = parse_subckt_params(param_part)
            else:
                port_part = rest

            tokens = port_part.split()
            if tokens:
                current['name'] = tokens[0]
                current['ports'] = tokens[1:]

            current['text'] = '\n'.join(text_lines)
            if current['name']:
                subckts.append(current)
            current = None
            text_lines = []
            subckt_header = None

        elif current is not None:
            text_lines.append(stripped)

        else:
            # Top-level line (outside any subcircuit)
            # Skip .end, blank lines, and title comments at line 1
            if upper.startswith('.END') and not upper.startswith('.ENDS'):
                continue
            if stripped:
                preamble_lines.append(stripped)

    preamble = '\n'.join(preamble_lines) if preamble_lines else ''
    return subckts, preamble


def guess_port_direction(name):
    """Guess port direction from name conventions.

    Returns 'in', 'out', or 'power'.
    """
    upper = name.upper()
    if upper in ('VDD', 'VCC', 'VSS', 'GND', 'AVDD', 'AVSS', 'DVDD', 'DVSS',
                 'SUB', 'SUBSTRATE', 'BULK'):
        return 'power'
    if upper in ('OUT', 'OUTPUT', 'Y', 'Q', 'QB', 'Z', 'DOUT'):
        return 'out'
    if upper.startswith('OUT') or upper.endswith('OUT'):
        return 'out'
    if upper in ('IN', 'INPUT', 'A', 'B', 'D', 'DIN', 'CLK', 'EN', 'RST',
                 'SET', 'RESET', 'GATE', 'G', 'S'):
        return 'in'
    if upper.startswith('IN') or upper.endswith('IN'):
        return 'in'
    return 'inout'


def vhdl_safe_name(name):
    """Make a SPICE name safe for VHDL identifiers."""
    # Replace non-alphanumeric with underscore
    s = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Must start with a letter
    if s and not s[0].isalpha():
        s = 'n_' + s
    # VHDL keywords
    if s.upper() in ('IN', 'OUT', 'SIGNAL', 'PORT', 'ENTITY', 'IS',
                      'BEGIN', 'END', 'ARCHITECTURE', 'COMPONENT',
                      'PROCESS', 'VARIABLE', 'GENERIC', 'MAP'):
        s = s + '_p'
    return s.lower()


def escape_vhdl_string(text):
    """Escape a string for VHDL string literal (double up quotes)."""
    return text.replace('"', '""')


def generate_vhdl(subckt, preamble=''):
    """Generate VHDL entity + architecture wrapping a Xyce subcircuit."""
    name = vhdl_safe_name(subckt['name'])
    ports = subckt['ports']
    params = subckt['params']
    # Combine preamble (models, params) with subcircuit text
    if preamble:
        spice_text = subckt['text'] + '\n' + preamble
    else:
        spice_text = subckt['text']

    lines = []
    lines.append(f'-- Auto-generated VHDL wrapper for Xyce subcircuit: {subckt["name"]}')
    lines.append(f'-- Source ports: {" ".join(ports)}')
    lines.append(f'-- Analog ports use logic3da (Thevenin V,R) for PWL bridge interface')
    lines.append('')
    lines.append('library ieee;')
    lines.append('use ieee.std_logic_1164.all;')
    lines.append('')
    lines.append('library sv2vhdl;')
    lines.append('use sv2vhdl.spice_subckt_pkg.all;')
    lines.append('use sv2vhdl.logic3da_pkg.all;')
    lines.append('')

    # Entity
    lines.append(f'entity {name} is')

    # Generics from PARAMS
    if params:
        lines.append('    generic (')
        param_lines = []
        for pname, pval in params.items():
            vname = vhdl_safe_name(pname)
            param_lines.append(f'        {vname} : real := {pval}')
        lines.append(';\n'.join(param_lines))
        lines.append('    );')

    # Ports use resolved_logic3da — Thevenin (V, R) with PWL flag
    # Power ports omitted from VHDL entity (handled as supply in Xyce)
    signal_ports = [(p, guess_port_direction(p)) for p in ports]
    vhdl_ports = [(p, d) for p, d in signal_ports if d != 'power']
    if vhdl_ports:
        lines.append('    port (')
        port_lines = []
        for p, d in vhdl_ports:
            vp = vhdl_safe_name(p)
            vhdl_dir = d if d in ('in', 'out', 'inout') else 'inout'
            port_lines.append(f'        {vp} : {vhdl_dir} resolved_logic3da')
        lines.append(';\n'.join(port_lines))
        lines.append('    );')

    lines.append(f'end entity {name};')
    lines.append('')

    # Architecture
    lines.append(f'architecture spice of {name} is')

    # Inherited power ports as internal signals with supply attribute
    power_ports = [(p, d) for p, d in signal_ports if d == 'power']
    if power_ports:
        lines.append('')
        lines.append('    -- Inherited power ports')
        lines.append('    attribute spice_supply : real;')
        for p, d in power_ports:
            vp = vhdl_safe_name(p)
            upper = p.upper()
            voltage = '0.0' if upper in ('VSS', 'GND') else '1.0'
            lines.append(f'    signal {vp} : resolved_logic3da;')
            lines.append(f'    attribute spice_supply of {vp} : signal is {voltage};')

    # Bridge signals for PWL interface
    lines.append('')
    lines.append('    -- Bridge signals for SPICE PWL interface')
    lines.append('    attribute spice_bridge : string;')
    for p, d in vhdl_ports:
        vp = vhdl_safe_name(p)
        if d == 'in':
            lines.append(f'    signal {vp}_rcv : logic3da;')
            lines.append(f'    attribute spice_bridge of {vp}_rcv : signal is "receiver:{p}";')
        elif d == 'out':
            lines.append(f'    signal {vp}_drv : logic3da;')
            lines.append(f'    attribute spice_bridge of {vp}_drv : signal is "driver:{p}";')
        else:
            lines.append(f'    signal {vp}_rcv : logic3da;')
            lines.append(f'    attribute spice_bridge of {vp}_rcv : signal is "receiver:{p}";')
            lines.append(f'    signal {vp}_drv : logic3da;')
            lines.append(f'    attribute spice_bridge of {vp}_drv : signal is "driver:{p}";')

    lines.append('')
    lines.append('begin')
    lines.append('')

    # Bridge assignments: connect entity ports to bridge signals
    for p, d in vhdl_ports:
        vp = vhdl_safe_name(p)
        if d == 'in':
            lines.append(f'    -- Receiver: SPICE reads input value from digital side')
            lines.append(f'    {vp}_rcv <= {vp}\'receiver;')
        elif d == 'out':
            lines.append(f'    -- Driver: SPICE drives output value to digital side')
            lines.append(f'    {vp}\'driver <= {vp}_drv;')
        else:
            lines.append(f'    -- Bidirectional bridge for {vp}')
            lines.append(f'    {vp}_rcv <= {vp}\'receiver;')
            lines.append(f'    {vp}\'driver <= {vp}_drv;')
    lines.append('')

    # SPICE subcircuit payload
    meta_parts = [f'DIALECT:xyce', f'SUBCKT:{subckt["name"]}']
    for p in ports:
        vp = vhdl_safe_name(p)
        d = guess_port_direction(p)
        meta_parts.append(f'PORT:{p}:{vp}:{d}')
    for pname, pval in params.items():
        vname = vhdl_safe_name(pname)
        meta_parts.append(f'PARAM:{pname}:{vname}')

    metadata = '|'.join(meta_parts)
    spice_flat = spice_text.replace('\n', '~NL~')
    full_payload = metadata + '||' + spice_flat
    escaped = escape_vhdl_string(full_payload)

    spice_parts = escaped.split('~NL~')
    if len(spice_parts) <= 1:
        lines.append(f'    spice_subckt("{escaped}");')
    else:
        lines.append('    spice_subckt(')
        for i, sp in enumerate(spice_parts):
            nl_delim = '~NL~' if i < len(spice_parts) - 1 else ''
            sep = ' &' if i < len(spice_parts) - 1 else ''
            lines.append(f'        "{sp}{nl_delim}"{sep}')
        lines.append('    );')

    lines.append('')
    lines.append('end architecture spice;')
    lines.append('')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Import Xyce SPICE subcircuits as VHDL entity wrappers')
    parser.add_argument('netlist', help='Xyce SPICE netlist file (.cir)')
    parser.add_argument('-o', '--outdir', default='.',
                        help='Output directory for .vhd files (default: .)')
    parser.add_argument('--subckt', default=None,
                        help='Only convert this subcircuit (by name)')
    parser.add_argument('--list', action='store_true',
                        help='List subcircuits in netlist and exit')
    args = parser.parse_args()

    with open(args.netlist, 'r') as f:
        lines = f.readlines()

    subckts, preamble = parse_netlist(lines)

    if not subckts:
        # No subcircuits found — wrap the entire netlist as a top-level block
        print(f'No .SUBCKT found in {args.netlist}', file=sys.stderr)
        # Create a synthetic subckt from the whole netlist
        basename = os.path.splitext(os.path.basename(args.netlist))[0]
        whole_text = ''.join(lines).rstrip()
        subckts = [{
            'name': basename,
            'ports': [],
            'params': {},
            'text': whole_text,
        }]
        print(f'Wrapping entire netlist as "{basename}"', file=sys.stderr)

    if args.list:
        for s in subckts:
            ports_str = ' '.join(s['ports']) if s['ports'] else '(no ports)'
            params_str = ', '.join(f'{k}={v}' for k, v in s['params'].items())
            print(f"  {s['name']}  ports: {ports_str}"
                  + (f'  params: {params_str}' if params_str else ''))
        return

    if args.subckt:
        subckts = [s for s in subckts if s['name'].upper() == args.subckt.upper()]
        if not subckts:
            print(f'Subcircuit "{args.subckt}" not found', file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    for s in subckts:
        vhdl = generate_vhdl(s, preamble)
        vname = vhdl_safe_name(s['name'])
        outpath = os.path.join(args.outdir, f'{vname}.vhd')
        with open(outpath, 'w') as f:
            f.write(vhdl)
        print(f'{outpath}: {s["name"]} ({len(s["ports"])} ports)')


if __name__ == '__main__':
    main()
