#!/bin/bash
# Cosim test runner — sets up paths and runs NVC with Xyce co-simulation

NVC_BUILD=/usr/local/src/nvc-build
NVC=$NVC_BUILD/bin/nvc

export LD_LIBRARY_PATH=$NVC_BUILD:$NVC_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

WORK=gen/work
LIBS=$NVC_BUILD/lib
NETLIST=gen/cmos_inv.cir
BOUNDARY=gen/cosim.boundary
STOP=100ns

# Compile
$NVC --std=2040 --work=work:$WORK -L $LIBS -a gen/cmos_inv.vhd digital_inv.vhd cosim_tb.vhd || exit 1

# Elaborate
$NVC --std=2040 --work=work:$WORK -L $LIBS -e cosim_tb || exit 1

# Run with cosim
$NVC --std=2040 --work=work:$WORK -L $LIBS \
    -r --stop-time=$STOP \
    --xyce-netlist=$NETLIST \
    --xyce-config=$BOUNDARY \
    cosim_tb || exit 1

# Plot waveforms if --plot given
if [[ "$1" == "--plot" ]]; then
    PRN=$NETLIST.prn
    OUT=${2:-gen/cosim_waveforms.png}
    python3 - "$PRN" "$OUT" <<'PYEOF'
import sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

prn, out = sys.argv[1], sys.argv[2]
cols, data = [], []
with open(prn) as f:
    for line in f:
        parts = line.split()
        if not parts: continue
        if not cols:
            cols = parts[1:]  # skip Index
            continue
        try:
            data.append([float(p) for p in parts[1:]])
        except ValueError:
            pass

import numpy as np
data = np.array(data)
t = data[:,0] * 1e9
vcols = [i for i,c in enumerate(cols) if c.startswith('V(')]
icols = [i for i,c in enumerate(cols) if c.startswith('I(')]

npanels = (1 if vcols else 0) + (1 if icols else 0)
fig, axes = plt.subplots(npanels, 1, figsize=(12, 3*npanels+1), sharex=True)
if npanels == 1: axes = [axes]

ax = 0
if vcols:
    for i in vcols:
        axes[ax].plot(t, data[:,i], linewidth=1.5, label=cols[i])
    axes[ax].set_ylabel('Voltage (V)')
    axes[ax].legend(loc='best', fontsize=9)
    axes[ax].grid(True, alpha=0.3)
    axes[ax].set_title('NVC \u2194 Xyce Co-simulation Waveforms')
    ax += 1

if icols:
    for i in icols:
        axes[ax].plot(t, data[:,i]*1e6, linewidth=1.5, label=cols[i])
    axes[ax].set_ylabel('Current (\u00b5A)')
    axes[ax].legend(loc='best', fontsize=9)
    axes[ax].grid(True, alpha=0.3)
    ax += 1

axes[-1].set_xlabel('Time (ns)')
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Saved {out}')
PYEOF
fi
