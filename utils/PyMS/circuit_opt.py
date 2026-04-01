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
import subprocess
import tempfile


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


def _compile_merged_so(cpp_source: str, output_path: str) -> bool:
    """Compile C++ source to a VAE .so."""
    with tempfile.NamedTemporaryFile(suffix='.cpp', mode='w', delete=False) as f:
        f.write(cpp_source)
        cpp = f.name
    try:
        r = subprocess.run(['g++', '-shared', '-fPIC', '-O2', '-o', output_path, cpp],
                          capture_output=True, timeout=30)
        return r.returncode == 0
    finally:
        os.unlink(cpp)


def _gen_darlington_so_src(IS=1e-16, BF=125.0, BR=1.0, CJE=0.0, CJC=0.0,
                           is_pnp=False) -> str:
    """Generate C++ for a Darlington VAE .so (node-indexed)."""
    # NPN: V[0]=c, V[1]=b, V[2]=e, V[3]=m (Q1.e=Q2.b)
    # PNP: same node layout, reversed Vbe/Vbc signs
    if is_pnp:
        q1_vbe = "s->V[3]-s->V[1]"  # V(m,b) = V(e1,b1)
        q1_vbc = "s->V[0]-s->V[1]"  # V(c,b)
        q2_vbe = "s->V[2]-s->V[3]"  # V(e,m) = V(e2,b2)
        q2_vbc = "s->V[0]-s->V[3]"  # V(c,m)
        sign = "-1.0"
    else:
        q1_vbe = "s->V[1]-s->V[3]"
        q1_vbc = "s->V[1]-s->V[0]"
        q2_vbe = "s->V[3]-s->V[2]"
        q2_vbc = "s->V[3]-s->V[0]"
        sign = "1.0"

    return f"""
#include <cstring>
#include <cmath>
struct VaeState {{ double V[16]; double Vt; }};
static inline double limexp(double x) {{
    return (x < 80.0) ? exp(x) : exp(80.0) * (1.0 + x - 80.0);
}}
extern "C" {{
void vae_eval(VaeState* s, double* F, double* Q) {{
    double vt=s->Vt, sgn={sign};
    double IS={IS}, BF={BF}, BR={BR}, CJE={CJE}, CJC={CJC};
    double q1_Vbe={q1_vbe}, q1_Vbc={q1_vbc};
    double q1_If=IS*(limexp(q1_Vbe/vt)-1.0), q1_Ir=IS*(limexp(q1_Vbc/vt)-1.0);
    double q1_Ic=q1_If-q1_Ir, q1_Ibe=q1_If/BF, q1_Ibc=q1_Ir/BR;
    double q2_Vbe={q2_vbe}, q2_Vbc={q2_vbc};
    double q2_If=IS*(limexp(q2_Vbe/vt)-1.0), q2_Ir=IS*(limexp(q2_Vbc/vt)-1.0);
    double q2_Ic=q2_If-q2_Ir, q2_Ibe=q2_If/BF, q2_Ibc=q2_Ir/BR;
    F[0]=sgn*((q1_Ic-q1_Ibc)+(q2_Ic-q2_Ibc));
    F[1]=sgn*(q1_Ibe+q1_Ibc);
    F[2]=sgn*(-q2_Ic-q2_Ibe);
    F[3]=sgn*((-q1_Ic-q1_Ibe)+(q2_Ibe+q2_Ibc));
    Q[0]=-CJC*q1_Vbc-CJC*q2_Vbc;
    Q[1]=CJC*q1_Vbc;
    Q[2]=-CJE*q2_Vbe;
    Q[3]=CJE*q1_Vbe+CJE*q2_Vbe+CJC*q2_Vbc;
}}
void vae_jacobian(VaeState* s, double* dFdV, double* dQdV) {{
    double dv=1e-6;double F0[4],Q0[4],Fp[4],Qp[4];VaeState sp;
    memset(dFdV,0,16*sizeof(double));memset(dQdV,0,16*sizeof(double));
    vae_eval(s,F0,Q0);
    for(int j=0;j<4;j++){{sp=*s;sp.V[j]+=dv;vae_eval(&sp,Fp,Qp);
    for(int i=0;i<4;i++){{dFdV[i*4+j]=(Fp[i]-F0[i])/dv;dQdV[i*4+j]=(Qp[i]-Q0[i])/dv;}}}}
}}
int vae_n_nodes(){{return 4;}}
int vae_n_branches(){{return 4;}}
}}
"""


def _gen_mirror_so_src(IS=1e-16, BF=125.0, BR=1.0, CJE=0.0, CJC=0.0,
                       is_pnp=False) -> str:
    """Generate C++ for a current mirror VAE .so (node-indexed).
    Nodes: V[0]=cref, V[1]=cout, V[2]=e, V[3]=bp (internal base)
    Diode connection: bp=cref via large conductance.
    """
    if is_pnp:
        q1_vbe = "s->V[2]-s->V[3]"  # V(e,bp)
        q1_vbc = "s->V[0]-s->V[3]"  # V(cref,bp)
        q2_vbe = "s->V[2]-s->V[3]"  # V(e,bp) same base
        q2_vbc = "s->V[1]-s->V[3]"  # V(cout,bp)
        sign = "-1.0"
    else:
        q1_vbe = "s->V[3]-s->V[2]"
        q1_vbc = "s->V[3]-s->V[0]"
        q2_vbe = "s->V[3]-s->V[2]"
        q2_vbc = "s->V[3]-s->V[1]"
        sign = "1.0"

    return f"""
#include <cstring>
#include <cmath>
struct VaeState {{ double V[16]; double Vt; }};
static inline double limexp(double x) {{
    return (x < 80.0) ? exp(x) : exp(80.0) * (1.0 + x - 80.0);
}}
extern "C" {{
void vae_eval(VaeState* s, double* F, double* Q) {{
    double vt=s->Vt, sgn={sign};
    double IS={IS}, BF={BF}, BR={BR}, CJE={CJE}, CJC={CJC};
    double Gshort=1e8;
    double q1_Vbe={q1_vbe}, q1_Vbc={q1_vbc};
    double q1_If=IS*(limexp(q1_Vbe/vt)-1.0), q1_Ir=IS*(limexp(q1_Vbc/vt)-1.0);
    double q1_Ic=q1_If-q1_Ir, q1_Ibe=q1_If/BF, q1_Ibc=q1_Ir/BR;
    double q2_Vbe={q2_vbe}, q2_Vbc={q2_vbc};
    double q2_If=IS*(limexp(q2_Vbe/vt)-1.0), q2_Ir=IS*(limexp(q2_Vbc/vt)-1.0);
    double q2_Ic=q2_If-q2_Ir, q2_Ibe=q2_If/BF, q2_Ibc=q2_Ir/BR;
    // Diode connection: bp shorted to cref
    double Vshort = s->V[3] - s->V[0];
    F[0]=sgn*(q1_Ic-q1_Ibc) - Gshort*Vshort;
    F[1]=sgn*(q2_Ic-q2_Ibc);
    F[2]=sgn*(-q1_Ic-q1_Ibe-q2_Ic-q2_Ibe);
    F[3]=sgn*(q1_Ibe+q1_Ibc+q2_Ibe+q2_Ibc) + Gshort*Vshort;
    Q[0]=-CJC*q1_Vbc; Q[1]=-CJC*q2_Vbc;
    Q[2]=-CJE*q1_Vbe-CJE*q2_Vbe;
    Q[3]=CJE*q1_Vbe+CJC*q1_Vbc+CJE*q2_Vbe+CJC*q2_Vbc;
}}
void vae_jacobian(VaeState* s, double* dFdV, double* dQdV) {{
    double dv=1e-6;double F0[4],Q0[4],Fp[4],Qp[4];VaeState sp;
    memset(dFdV,0,16*sizeof(double));memset(dQdV,0,16*sizeof(double));
    vae_eval(s,F0,Q0);
    for(int j=0;j<4;j++){{sp=*s;sp.V[j]+=dv;vae_eval(&sp,Fp,Qp);
    for(int i=0;i<4;i++){{dFdV[i*4+j]=(Fp[i]-F0[i])/dv;dQdV[i*4+j]=(Qp[i]-Q0[i])/dv;}}}}
}}
int vae_n_nodes(){{return 4;}}
int vae_n_branches(){{return 4;}}
}}
"""


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

    # Compile per-model VAE .so files with correct params
    vae_dir = outdir / 'vae_so'
    vae_dir.mkdir(exist_ok=True)

    for driver, output_q in darlingtons:
        d = bjts[driver]
        mod_name = f"dmod_{driver}_{output_q}".lower()
        mp = model_params.get(d.model, {})
        is_pnp = d.model.upper() in ('PNP', 'PN')
        src = _gen_darlington_so_src(
            IS=mp.get('IS', mp.get('Is', mp.get('is', 1e-16))),
            BF=mp.get('BF', 100.0), BR=mp.get('BR', 1.0),
            CJE=mp.get('CJE', mp.get('Cje', mp.get('cje', 0.0))),
            CJC=mp.get('CJC', mp.get('Cjc', mp.get('cjc', 0.0))),
            is_pnp=is_pnp)
        so_path = vae_dir / f"{mod_name.upper()}.so"
        if _compile_merged_so(src, str(so_path)):
            if report:
                print(f"  Compiled {so_path.name}")

    for ref, mir in mirrors:
        r = bjts[ref]
        mod_name = f"mmod_{ref}_{mir}".lower()
        mp = model_params.get(r.model, {})
        is_pnp = r.model.upper() in ('PNP', 'PN')
        src = _gen_mirror_so_src(
            IS=mp.get('IS', mp.get('Is', mp.get('is', 1e-16))),
            BF=mp.get('BF', 100.0), BR=mp.get('BR', 1.0),
            CJE=mp.get('CJE', mp.get('Cje', mp.get('cje', 0.0))),
            CJC=mp.get('CJC', mp.get('Cjc', mp.get('cjc', 0.0))),
            is_pnp=is_pnp)
        so_path = vae_dir / f"{mod_name.upper()}.so"
        if _compile_merged_so(src, str(so_path)):
            if report:
                print(f"  Compiled {so_path.name}")

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
    stats['vae_dir'] = str(vae_dir)
    return stats


def find_series_rc(lines: List[str]) -> List[Tuple[str, str, str, str, str, str, str]]:
    """Find series R+C or R+L pairs where the shared node has only 2 connections.

    Returns list of (r_name, cl_name, cl_type, ext_node1, ext_node2, r_val, cl_val).
    """
    devices = {}
    for l in lines:
        s = l.strip()
        if not s or s.startswith('*') or s.startswith('.'):
            continue
        parts = s.split()
        name = parts[0]
        prefix = name[0].upper()
        if prefix in ('R', 'C', 'L') and len(parts) >= 4:
            devices[name] = (prefix, (parts[1], parts[2]), parts[3])

    # Count node connections across ALL devices
    node_count: Dict[str, int] = {}
    for l in lines:
        s = l.strip()
        if not s or s.startswith('*') or s.startswith('.'):
            continue
        parts = s.split()
        if parts and parts[0][0].isalpha():
            for p in parts[1:]:
                if '=' in p or p.startswith('{'):
                    break
                if re.match(r'^[A-Za-z0-9_+\-]+$', p):
                    node_count[p] = node_count.get(p, 0) + 1

    pairs = []
    used = set()
    for rn, (rt, rnodes, rv) in devices.items():
        if rt != 'R' or rn in used:
            continue
        for cn, (ct, cnodes, cv) in devices.items():
            if ct not in ('C', 'L') or cn in used:
                continue
            shared = set(rnodes) & set(cnodes)
            if not shared:
                continue
            node = shared.pop()
            if node_count.get(node, 0) == 2:
                r_other = [n for n in rnodes if n != node][0]
                c_other = [n for n in cnodes if n != node][0]
                pairs.append((rn, cn, ct, r_other, c_other, rv, cv))
                used.add(rn)
                used.add(cn)
                break

    return pairs


def optimize_esr(input_path: str, output_path: str,
                 report: bool = False) -> dict:
    """Optimize by merging series R+C into C with ESR, R+L into L with R."""
    with open(input_path, 'r') as f:
        lines = f.readlines()

    pairs = find_series_rc(lines)

    stats = {'esr_merges': len(pairs)}

    if report:
        print(f"  Series R+C/R+L pairs: {len(pairs)}")
        for rn, cn, ct, n1, n2, rv, cv in pairs:
            print(f"    {rn}({rv}) + {cn}({cv}) → {ct} with ESR")

    if not pairs:
        with open(output_path, 'w') as f:
            f.writelines(lines)
        return stats

    merged_names = set()
    replacements = {}  # old device names → replacement lines

    for rn, cn, ct, n1, n2, rv, cv in pairs:
        merged_names.add(rn)
        merged_names.add(cn)
        if ct == 'C':
            # R+C → C with Rser parameter (Xyce supports this natively!)
            # Actually Xyce doesn't support Rser on C. Use a comment for clarity.
            # The real optimization: just keep both devices but Xyce handles them.
            # OR: for Verilog-A path, create a merged RC device.
            #
            # Simplest: Xyce doesn't need VA for this. Just reconnect:
            # Remove R, connect C directly between the two external nodes.
            # The R's effect is approximated by the C alone (for frequency response)
            # No — that changes the circuit! Keep both but mark as optimizable.
            #
            # Actually the simplest valid merge: replace R+C with a single
            # Xyce behavioral B-source or just note it.
            # For now: use the fact that this eliminates one node.
            # C(n1, n2) with ESR is modeled as: I = C*ddt(V) + V/R
            # But that's not a capacitor — it's a parallel R||C, not series R+C.
            #
            # Series R+C needs: V(a,b) = I*R + Q/C, or equivalently
            # keep both devices. The node reduction comes from internalizing
            # the shared node. Need a VA device.
            pass

    # For the ESR merge, generate a simple VA: series R + C as one device
    outdir = Path(output_path).parent
    va_written = set()
    merged_devices = {}

    for rn, cn, ct, n1, n2, rv, cv in pairs:
        if ct == 'C':
            va_name = 'esr_cap'
            if va_name not in va_written:
                va_path = outdir / f'{va_name}.va'
                va_path.write_text("""`include "disciplines.vams"
`include "constants.vams"
module esr_cap(a, b);
    inout a, b;
    electrical a, b;
    electrical mid;
    parameter real R = 1.0;
    parameter real C = 1e-12;
    analog begin
        I(a, mid) <+ V(a, mid) / R;
        I(mid, b) <+ C * ddt(V(mid, b));
    end
endmodule
""")
                va_written.add(va_name)

            mod_name = f"esrc_{rn}_{cn}".lower()
            merged_devices[(rn, cn)] = (va_name, mod_name, n1, n2, rv, cv)

        elif ct == 'L':
            va_name = 'esr_ind'
            if va_name not in va_written:
                va_path = outdir / f'{va_name}.va'
                va_path.write_text("""`include "disciplines.vams"
`include "constants.vams"
module esr_ind(a, b);
    inout a, b;
    electrical a, b;
    electrical mid;
    parameter real R = 1.0;
    parameter real L = 1e-6;
    analog begin
        I(a, mid) <+ V(a, mid) / R;
        V(mid, b) <+ L * ddt(I(mid, b));
    end
endmodule
""")
                va_written.add(va_name)

            mod_name = f"esrl_{rn}_{cn}".lower()
            merged_devices[(rn, cn)] = (va_name, mod_name, n1, n2, rv, cv)

    # Rewrite netlist
    output = []
    hdl_added = False

    for line in lines:
        s = line.strip()

        # Add .hdl after title
        if not hdl_added and not s.startswith('*') and not s.startswith('.'):
            for va_name in sorted(va_written):
                va_abs = (outdir / f'{va_name}.va').resolve()
                output.append(f'.hdl "{va_abs}"\n')
            hdl_added = True

        # Skip merged R and C/L devices
        parts = s.split()
        if parts:
            dname = parts[0]
            if dname in merged_names:
                output.append(f'* [merged ESR] {s}\n')
                continue

        output.append(line)

    # Add merged device instances before .end
    for (rn, cn), (va_name, mod_name, n1, n2, rv, cv) in merged_devices.items():
        insert = (
            f"* [ESR merge: {rn}({rv}) + {cn}({cv})]\n"
            f".model {mod_name} {va_name} R={rv} C={cv}\n"
            f"Y{va_name.upper()} {mod_name}_inst {n1} {n2} {mod_name}\n"
        )
        final = []
        for line in output:
            if line.strip().upper() == '.END':
                final.append(insert)
            final.append(line)
        output = final

    with open(output_path, 'w') as f:
        f.writelines(output)

    stats['nodes_saved'] = len(pairs)
    return stats


def find_passive_chains(lines: List[str]) -> Tuple[List[List[str]], Dict, Dict]:
    """Find series chains of R/L/C where intermediate nodes have exactly 2 connections."""
    devices = {}
    for l in lines:
        s = l.strip()
        if not s or s.startswith('*') or s.startswith('.'):
            continue
        parts = s.split()
        name = parts[0]
        prefix = name[0].upper()
        if prefix in ('R', 'C', 'L') and len(parts) >= 4:
            devices[name] = (prefix, parts[1], parts[2], parts[3])

    node_count: Dict[str, int] = {}
    for l in lines:
        s = l.strip()
        if not s or s.startswith('*') or s.startswith('.'):
            continue
        parts = s.split()
        if not parts or not parts[0][0].isalpha():
            continue
        for p in parts[1:]:
            if '=' in p or p.startswith('{'):
                break
            if re.match(r'^[A-Za-z0-9_+\-]+$', p):
                node_count[p] = node_count.get(p, 0) + 1

    node_to_devs: Dict[str, List[str]] = {}
    for name, (t, n1, n2, v) in devices.items():
        node_to_devs.setdefault(n1, []).append(name)
        node_to_devs.setdefault(n2, []).append(name)

    used = set()
    chains = []

    for start in devices:
        if start in used:
            continue
        # BFS/DFS to find chain
        chain = [start]
        used.add(start)
        for _ in range(100):  # safety limit
            extended = False
            for end_dev in [chain[0], chain[-1]]:
                t, n1, n2, v = devices[end_dev]
                for node in (n1, n2):
                    if node_count.get(node, 0) != 2:
                        continue
                    for nb in node_to_devs.get(node, []):
                        if nb not in used and nb in devices:
                            if end_dev == chain[-1]:
                                chain.append(nb)
                            else:
                                chain.insert(0, nb)
                            used.add(nb)
                            extended = True
                            break
                    if extended:
                        break
                if extended:
                    break
            if not extended:
                break

        if len(chain) >= 2:
            chains.append(chain)

    return chains, devices, node_count


def _gen_chain_va(chain_devices: List[Tuple[str, str, str, str]],
                  module_name: str) -> str:
    """Generate VA for a series chain of R/L/C devices.

    Each device: (type, n1, n2, value)
    Chain is series-connected: dev[i].n2 connects to dev[i+1].n1
    External: first device's n1 and last device's n2.
    Internal: all intermediate junction nodes.
    """
    n = len(chain_devices)
    # Node names: a (external), b (external), m0, m1, ... (internal)
    nodes = ['a'] + [f'm{i}' for i in range(n - 1)] + ['b']

    params = []
    analog_lines = []
    for i, (dtype, _, _, value) in enumerate(chain_devices):
        na = nodes[i]
        nb = nodes[i + 1]
        pname = f"{dtype.lower()}{i}"
        params.append(f"    parameter real {pname} = {value};")
        if dtype == 'R':
            analog_lines.append(f"        I({na}, {nb}) <+ V({na}, {nb}) / {pname};")
        elif dtype == 'C':
            analog_lines.append(f"        I({na}, {nb}) <+ {pname} * ddt(V({na}, {nb}));")
        elif dtype == 'L':
            analog_lines.append(f"        V({na}, {nb}) <+ {pname} * ddt(I({na}, {nb}));")

    internal = nodes[1:-1]
    int_decl = f"    electrical {', '.join(internal)};" if internal else ""

    return f"""`include "disciplines.vams"
`include "constants.vams"
module {module_name}(a, b);
    inout a, b;
    electrical a, b;
{int_decl}
{chr(10).join(params)}
    analog begin
{chr(10).join(analog_lines)}
    end
endmodule
"""


def optimize_spice(input_path: str, output_path: str,
                   report: bool = False) -> dict:
    """Merge series-connected passive chains into single VA devices."""
    with open(input_path, 'r') as f:
        lines = f.readlines()

    chains, devices, node_count = find_passive_chains(lines)

    stats = {'chains': len(chains), 'devices_merged': sum(len(c) for c in chains),
             'nodes_saved': sum(len(c) - 1 for c in chains)}

    if report:
        print(f"  Passive chains: {len(chains)}")
        for chain in chains:
            desc = ' → '.join(f"{d}({devices[d][0]}{devices[d][3]})" for d in chain)
            print(f"    {desc}")
        print(f"  Nodes saved: {stats['nodes_saved']}")

    if not chains:
        with open(output_path, 'w') as f:
            f.writelines(lines)
        return stats

    outdir = Path(output_path).parent
    merged_names = set()
    va_files = {}  # module_name → va_path
    chain_info = []  # (chain, module_name, ext_n1, ext_n2, param_str)

    for ci, chain in enumerate(chains):
        merged_names.update(chain)
        module_name = f"chain_{ci}"

        # Order chain so devices connect sequentially
        # Walk from first device through shared nodes
        ordered = []
        remaining = list(chain)
        current = remaining.pop(0)
        ordered.append(current)
        while remaining:
            t, n1, n2, v = devices[current]
            found = False
            for r in remaining:
                rt, rn1, rn2, rv = devices[r]
                if rn1 in (n1, n2) or rn2 in (n1, n2):
                    ordered.append(r)
                    remaining.remove(r)
                    current = r
                    found = True
                    break
            if not found:
                break

        # Determine external nodes (endpoints of the chain)
        all_chain_nodes = set()
        for d in ordered:
            t, n1, n2, v = devices[d]
            all_chain_nodes.update([n1, n2])
        # External nodes: those that connect to non-chain devices
        ext_nodes = [n for n in all_chain_nodes if node_count.get(n, 0) > 2 or
                     n not in all_chain_nodes or
                     sum(1 for d in ordered for nn in (devices[d][1], devices[d][2]) if nn == n) == 1]
        if len(ext_nodes) != 2:
            # Can't determine endpoints cleanly, skip
            for d in ordered:
                merged_names.discard(d)
            continue

        # Order devices from ext_nodes[0] to ext_nodes[1]
        chain_devs = [(devices[d][0], devices[d][1], devices[d][2],
                       _parse_eng_val(devices[d][3])) for d in ordered]

        va_content = _gen_chain_va(chain_devs, module_name)
        va_path = outdir / f"{module_name}.va"
        va_path.write_text(va_content)
        va_files[module_name] = va_path

        # Build param string for .model
        param_parts = []
        for i, (dtype, _, _, value) in enumerate(chain_devs):
            param_parts.append(f"{dtype.lower()}{i}={value}")

        chain_info.append((ordered, module_name, ext_nodes[0], ext_nodes[1],
                          ' '.join(param_parts)))

    # Rewrite netlist
    output = []
    hdl_added = False

    for line in lines:
        s = line.strip()
        if not hdl_added and s and not s.startswith('*') and not s.startswith('.'):
            for mn, vp in va_files.items():
                output.append(f'.hdl "{vp.resolve()}"\n')
            hdl_added = True

        parts = s.split()
        if parts and parts[0] in merged_names:
            output.append(f'* [chain merged] {s}\n')
            continue

        output.append(line)

    # Add merged devices before .end
    inserts = []
    for ordered, module_name, n1, n2, param_str in chain_info:
        mod_name = f"{module_name}_mod"
        inst_name = f"{module_name}_inst"
        inserts.append(
            f"* [chain: {' + '.join(ordered)}]\n"
            f".model {mod_name} {module_name} {param_str}\n"
            f"Y{module_name.upper()} {inst_name} {n1} {n2} {mod_name}\n"
        )

    final = []
    for line in output:
        if line.strip().upper() == '.END':
            for ins in inserts:
                final.append(ins)
        final.append(line)

    with open(output_path, 'w') as f:
        f.writelines(final)

    return stats


def main():
    p = argparse.ArgumentParser(description='Circuit-level device merger')
    p.add_argument('input', help='Input .cir file')
    p.add_argument('-o', '--output', required=True, help='Output .cir file')
    p.add_argument('--merge', default='all',
                   help='Merge mode: all, darlington, mirror, esr, none')
    p.add_argument('--report', action='store_true', help='Print optimization report')
    args = p.parse_args()

    mode = args.merge.lower()

    if mode == 'esr':
        stats = optimize_esr(args.input, args.output, report=args.report)
        print(f"ESR merge: {stats['esr_merges']} R+C/R+L pairs merged, "
              f"{stats.get('nodes_saved', 0)} nodes saved")
    elif mode == 'spice':
        stats = optimize_spice(args.input, args.output, report=args.report)
        print(f"SPICE merge: {stats['chains']} chains, "
              f"{stats['devices_merged']} devices merged, "
              f"{stats['nodes_saved']} nodes saved")
    elif mode == 'none':
        import shutil
        shutil.copy2(args.input, args.output)
        print("No optimization")
    else:
        stats = optimize_circuit(args.input, args.output, report=args.report)
        print(f"Optimized: {stats['darlingtons']} Darlingtons, "
              f"{stats['mirrors']} mirrors merged "
              f"({stats.get('bjts_before', 0)} → {stats.get('bjts_after', stats.get('bjts_before', 0))} BJTs)")


if __name__ == '__main__':
    main()
