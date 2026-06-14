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

Requires the two cosim.c fixes (kev-cam/nvc `fix/cosim-xyce-master-runtime`):
RTLD_LAZY for the Xyce dlopen, and +1fs inclusive digital event stepping.
