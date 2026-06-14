#!/bin/bash
# Analog-on-top NVC<->Xyce cosim demo (Xyce master, nvc digital slave, boundary
# via Xyce DPWL "code:" sources). Proves both boundary directions end-to-end.
# Unlike test_inv_chain this needs NO spice_subckt_pkg (pure-digital VHDL).
#
# Prereqs:
#   - nvc built with cosim wired (src/cosim.c: --xyce-netlist/--xyce-config)
#   - libcosim_bridge.so built:  c++ -shared -fPIC -o $NVCB/lib/libcosim_bridge.so $NVCSRC/src/cosim_bridge.cpp
#   - Xyce built with libxycecinterface.so
set -e
NVCB=${NVCB:-/usr/local/src/nvc-build}
NVC=$NVCB/bin/nvc; LIBS=$NVCB/lib
XYCE_LIBDIR=${XYCE_LIBDIR:-$HOME/xyce-libs}
XCI=${XCI:-/usr/local/src/xyce-build/utils/XyceCInterface}
export LD_LIBRARY_PATH=$NVCB:$NVCB/lib:$XYCE_LIBDIR:$XCI:/usr/local/src/xyce-build/src

TEST=${1:-min}   # 'min' (D2A square wave) or 'a2d' (A2D round trip)
case $TEST in
  min) VHD=cosim_min.vhd; TOP=cosim_min; CIR=min.cir; BND=min.boundary;;
  a2d) VHD=cosim_a2d.vhd; TOP=cosim_a2d; CIR=a2d.cir; BND=a2d.boundary;;
  *) echo "usage: $0 [min|a2d]"; exit 1;;
esac
W=work_$TEST; rm -rf $W
$NVC --std=2040 --work=$W:$W -L $LIBS -a $VHD
$NVC --std=2040 --work=$W:$W -L $LIBS -e $TOP
$NVC --std=2040 --work=$W:$W -L $LIBS -r --stop-time=200ns \
     --xyce-netlist=$CIR --xyce-config=$BND $TOP
echo "--- $CIR.prn (boundary crossing result) ---"
column -t $CIR.prn | sed -n '1p;$p'
