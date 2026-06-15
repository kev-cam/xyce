# Analog-on-top NVC↔Xyce cosim — working template

Demonstrates the SIMetrix-style runtime: **Xyce is the analog master**, nvc runs
the digital as a slave, and the analog/digital boundary is a Xyce DPWL source
with a `code:libcosim_bridge.so:nvc_bridge_init:{d2a,a2d}:<sig>` URI. No
`spice_subckt_pkg` needed (the digital is pure VHDL; the analog lives in the
`.cir`). This is what `simetrix_cosim.pl` emits.

- `min`  — D2A: an nvc square wave drives an analog node through the bridge;
           Xyce shows the RC-filtered response, tracking every edge.
- `a2d`  — A2D round trip: an analog stimulus is sampled into nvc, thresholded,
           and driven back out through a D2A source — analog→digital→analog.

Run:  `./run.sh min`  or  `./run.sh a2d`

## Full pipeline from a SIMetrix netlist

`./gen_run.sh <design.net> [stop-time]` runs the whole flow: `.net` →
`simetrix_cosim.pl` → `{cir,boundary,vhd}` → cosim.

- `hier.net` — minimal mixed-signal chain (adc_bridge → d_inverter → dac_bridge)
  with the digital buried in a `.subckt`. `aout` comes out the digital inverse.
- `fly.net`  — the real SIMetrix flyback: a UC3844 controller (18 A-devices:
  nand/or/inv/tff/buffer/pullup/pulldown, two adc_schmitt, adc_bridge, four
  dac_bridge) co-simulating with the analog SMPS. Exercises the full translator:
  multi-input gates, bridge-generic propagation, the D2A series-R, SIMetrix
  syntactic cleanup, and the NMOS LEVEL=17 → VDMOS macromodel remap (IRFR420).
  Binds 7/7 boundaries and converges; reaching steady state is impractical at
  the fixed 10ns cosim sync because the input ramps over 1ms.

Requires the two cosim.c fixes (kev-cam/nvc `fix/cosim-xyce-master-runtime`):
RTLD_LAZY for the Xyce dlopen, and +1fs inclusive digital event stepping.
