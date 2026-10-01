# Analog-on-top NVC↔Xyce cosim — working template

Demonstrates the SIMetrix-style runtime: **the analog integrator is the master
scheduler**, nvc runs the digital as a slave, and the analog/digital boundary is
a Xyce DPWL source with a `code:libcosim_bridge.so:nvc_bridge_init:{d2a,a2d}:<sig>`
URI. No `spice_subckt_pkg` needed (the digital is pure VHDL; the analog lives in
the `.cir`). This is what `simetrix_cosim.pl` emits.

## How it works

1. nvc runs Xyce with a single `simulateUntil(stop)`; there are no sync
   intervals, pauses or round trips.
2. At every converged candidate step tn→t (after Xyce's own error control,
   before the step is committed) `Transient::doLoopProcess` calls
   `Device::cosimCandidateStep`, which hands the step to the bridge
   (`xyce_bridge_step`). The bridge walks nvc from tn to t one digital time
   point at a time, depositing each A2D probe's linear trajectory V(tn)→V(t)
   in `COSIM_A2D_DV` increments (default 0.01 V), so thresholds and `'event`
   fire at the right digital time.
3. If the digital changes a D2A input at some t_evt < t, the step is **vetoed**:
   Xyce rejects it and redoes it to end exactly at t_evt; the D2A ramp (1 ns,
   per source) starts there. A change at t itself is accepted.
4. While nvc runs, the PWL tables of the D2A sources already describe how the
   analog inputs change (they extend to `TVVEC_END`), so the analog side never
   has to wait for a digital acceptance. Xyce stops only at the A/D points where
   an input actually changes.

- `min`    — D2A: an nvc square wave drives an analog node through the bridge;
             Xyce shows the RC-filtered response, tracking every edge.
- `a2d`    — A2D round trip: an analog stimulus is sampled into nvc, thresholded,
             and driven back out through a D2A source — analog→digital→analog.
- `glitch` — the digital reverses a D2A output 0.5 ns into its own 1 ns ramp;
             the ramp turns around from where it is (0.5 V at 50.5 ns), no jump.

Run:  `./run.sh min`, `./run.sh a2d` or `./run.sh glitch`

## Full pipeline from a SIMetrix netlist

`./gen_run.sh <design.net> [stop-time]` runs the whole flow: `.net` →
`simetrix_cosim.pl` → `{cir,boundary,vhd}` → cosim. `ENGINE=vacask` runs the
analog on VACASK instead (`cir2vacask.py` turns the Xyce deck into a `.sim`;
same `.boundary` and VHDL, see `VACASK/demo/cosim`).

- `hier.net` — minimal mixed-signal chain (adc_bridge → d_inverter → dac_bridge)
  with the digital buried in a `.subckt`. `aout` comes out the digital inverse.
  20 µs runs in 0.55 s / 176 time points on Xyce; the edges land 4-5 ns after
  the ideal instants (the adc_bridge and dac_bridge delays).
- `fly.net`  — the real SIMetrix flyback: a UC3844 controller (18 A-devices:
  nand/or/inv/tff/buffer/pullup/pulldown, two adc_schmitt, adc_bridge, four
  dac_bridge) co-simulating with the analog SMPS. Exercises the full translator:
  multi-input gates, bridge-generic propagation, the D2A series-R, SIMetrix
  syntactic cleanup, and the NMOS LEVEL=17 → VDMOS macromodel remap (IRFR420).
  Binds 7/7 boundaries. The controller comes alive at ~6 ms (UVLO on the slow
  Vcc charge) and switches at 183 kHz, Vout settling at 4.7 V. 7 ms run in
  53 s on Xyce (113 k time points, one per ms while nothing switches); the
  full 8 ms in about 12 s on VACASK. Both engines agree to <1 % on the 6-7 ms
  window.

## Tuning and tracing

- `COSIM_A2D_DV` — analog resolution of the A2D trajectory fed to nvc (V);
  smaller = more digital events per analog step, default 0.01.
- `COSIM_TRACE=1` — the bridge prints every D2A change and every veto
  (`step to X vetoed: input changed at Y`); nvc prints the simulateUntil cycle.
- `XYCE_COSIM_TRACE=1` — per-step trace from the Xyce glue.
