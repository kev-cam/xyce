"""
VAE ADMS-compatible code generator.

Generates a C++ function body that reads from probeVars[] and writes to
staticContributions[]/dynamicContributions[]/d_staticContributions[][]/
d_dynamicContributions[][], matching the ADMS data layout so it can be
a drop-in replacement for updateIntermediateVars().

The ADMS layout:
  - probeVars[probeID]         : voltage differences (V(DI)-V(SI) etc.)
  - staticContributions[nodeID]: F contributions per KCL node
  - dynamicContributions[nodeID]: Q contributions per KCL node
  - d_staticContributions[nodeID][probeID]: dF/dV per node per probe
  - d_dynamicContributions[nodeID][probeID]: dQ/dV per node per probe
"""

import re
from dataclasses import dataclass


@dataclass
class ADMSLayout:
    """ADMS node/probe ID mapping extracted from a header file."""
    node_ids: dict  # name → int (e.g. 'DI' → 8)
    probe_ids: dict  # name → int (e.g. 'V_DI_SI' → 18)
    n_nodes: int
    n_probes: int

    # Derived: probe → (plus_node, minus_node)
    probe_nodes: dict = None

    def __post_init__(self):
        self.probe_nodes = {}
        for name, idx in self.probe_ids.items():
            # Parse 'V_DI_SI' → ('DI', 'SI'), 'V_DI_GND' → ('DI', None)
            parts = name.split('_')[1:]  # strip 'V_'
            if len(parts) == 2:
                p, n = parts[0], parts[1]
                self.probe_nodes[name] = (p, 'GND' if n == 'GND' else n)
            elif len(parts) == 1:
                self.probe_nodes[name] = (parts[0], 'GND')


def extract_adms_layout(header_path: str) -> ADMSLayout:
    """Extract node/probe ID mappings from an ADMS-generated header."""
    with open(header_path) as f:
        src = f.read()

    node_ids = {}
    probe_ids = {}

    for m in re.finditer(r'static const int (admsNodeID_(\w+))\s*=\s*([^;]+);', src):
        name = m.group(2)
        val = eval(m.group(3).strip())
        if val >= 0:
            node_ids[name] = val

    for m in re.finditer(r'static const int (admsProbeID_(V_\w+))\s*=\s*([^;]+);', src):
        name = m.group(2)
        val = eval(m.group(3).strip())
        probe_ids[name] = val

    return ADMSLayout(
        node_ids=node_ids,
        probe_ids=probe_ids,
        n_nodes=max(node_ids.values()) + 1 if node_ids else 0,
        n_probes=max(probe_ids.values()) + 1 if probe_ids else 0,
    )


def map_contribution(branch: tuple, node_ids: dict) -> list:
    """Map I(a,b) contribution to ADMS node IDs.

    I(a,b) <+ expr means:
      staticContributions[nodeID_a] += expr
      staticContributions[nodeID_b] -= expr

    Returns [(nodeID, sign), ...] pairs.
    """
    result = []
    if len(branch) >= 1:
        a = branch[0].upper()
        if a in node_ids:
            result.append((node_ids[a], +1))
    if len(branch) >= 2:
        b = branch[1].upper()
        if b in node_ids:
            result.append((node_ids[b], -1))
    return result


def map_jacobian_probe(node_a: str, node_b: str, probe_ids: dict) -> int:
    """Find the probe ID for V(node_a, node_b) = V_a - V_b.

    The ADMS Jacobian is indexed by probe voltages (voltage differences),
    not individual node voltages. Returns the probe index, or -1 if
    no matching probe exists.
    """
    # Direct match
    key = f'V_{node_a}_{node_b}'
    if key in probe_ids:
        return probe_ids[key]
    # Ground reference
    key = f'V_{node_a}_GND'
    if key in probe_ids and node_b == 'GND':
        return probe_ids[key]
    return -1


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: adms_compat.py <N_DEV_ADMS*.h>")
        sys.exit(1)

    layout = extract_adms_layout(sys.argv[1])
    print(f"Nodes ({layout.n_nodes}):")
    for name, idx in sorted(layout.node_ids.items(), key=lambda x: x[1]):
        print(f"  [{idx:2d}] {name}")
    print(f"\nProbes ({layout.n_probes}):")
    for name, idx in sorted(layout.probe_ids.items(), key=lambda x: x[1]):
        nodes = layout.probe_nodes.get(name, ('?', '?'))
        print(f"  [{idx:2d}] {name}  ({nodes[0]} - {nodes[1]})")
