#!/usr/bin/env python3
"""Plot VAE test circuit waveforms — diode clamp, RC filter, bridge rectifier."""

import csv
import math
import os
import sys

# Use Agg backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def read_csv(name):
    path = os.path.join(SCRIPT_DIR, name)
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def plot_diode_clamp():
    rows = read_csv('diode_clamp.csv')
    Vsup = [float(r['Vsup']) for r in rows]
    Vd   = [float(r['V_node1']) for r in rows]
    Ima  = [float(r['I_mA']) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle('Diode Clamp Circuit\nVsup → R(1k) → node1 → D → GND')

    ax1.plot(Vsup, Vd, 'b-', linewidth=1.5, label='V_node1')
    ax1.plot(Vsup, Vsup, 'k--', linewidth=0.8, alpha=0.5, label='V=Vsup')
    ax1.set_ylabel('Voltage (V)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(Vsup, Ima, 'r-', linewidth=1.5)
    ax2.set_xlabel('Vsup (V)')
    ax2.set_ylabel('Current (mA)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'diode_clamp.png')
    plt.savefig(out, dpi=150)
    print(f'Wrote {out}')
    plt.close()


def plot_rc_ac():
    rows = read_csv('rc_ac.csv')
    freq = [float(r['freq_Hz']) for r in rows]
    mag  = [float(r['mag_dB']) for r in rows]
    pha  = [float(r['phase_deg']) for r in rows]
    mag_a = [float(r['mag_dB_analytical']) for r in rows]
    pha_a = [float(r['phase_deg_analytical']) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle('RC Low-Pass Filter AC Sweep\nR=1kΩ, C=10nF, f_c=15.9kHz')

    ax1.semilogx(freq, mag, 'b-', linewidth=1.5, label='VAE numerical')
    ax1.semilogx(freq, mag_a, 'r--', linewidth=1.0, label='Analytical')
    ax1.axhline(-3, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(1.0 / (2 * math.pi * 1e3 * 10e-9), color='gray',
                linestyle=':', alpha=0.5, label='f_c')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(freq, pha, 'b-', linewidth=1.5, label='VAE numerical')
    ax2.semilogx(freq, pha_a, 'r--', linewidth=1.0, label='Analytical')
    ax2.axhline(-45, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'rc_ac.png')
    plt.savefig(out, dpi=150)
    print(f'Wrote {out}')
    plt.close()


def plot_bridge():
    rows = read_csv('bridge.csv')
    t    = [float(r['time_ms']) for r in rows]
    vn1  = [float(r['V_n1']) for r in rows]
    vn2  = [float(r['V_n2']) for r in rows]
    vop  = [float(r['V_out_p']) for r in rows]
    von  = [float(r['V_out_n']) for r in rows]
    vld  = [float(r['V_load']) for r in rows]
    ild  = [float(r['I_load_mA']) for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Full-Wave Bridge Rectifier\n4×Diode, R_load=100Ω, 5V peak 60Hz')

    ax1.plot(t, vn1, 'b-', linewidth=0.8, alpha=0.6, label='V_n1 (AC+)')
    ax1.plot(t, vn2, 'c-', linewidth=0.8, alpha=0.6, label='V_n2 (AC−)')
    ax1.set_ylabel('AC Input (V)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, vop, 'r-', linewidth=1.0, label='V_out+')
    ax2.plot(t, von, 'm-', linewidth=1.0, label='V_out−')
    ax2.plot(t, vld, 'k-', linewidth=1.5, label='V_load')
    ax2.set_ylabel('DC Output (V)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    ax3.plot(t, ild, 'g-', linewidth=1.0)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('I_load (mA)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'bridge.png')
    plt.savefig(out, dpi=150)
    print(f'Wrote {out}')
    plt.close()


if __name__ == '__main__':
    plot_diode_clamp()
    plot_rc_ac()
    plot_bridge()
