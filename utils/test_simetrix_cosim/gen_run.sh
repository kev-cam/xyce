#!/bin/bash
# Full SIMetrix-cosim pipeline: a SIMetrix .net (digital A-devices, possibly in
# a .subckt) -> simetrix_cosim.pl -> {cir,boundary,vhd} -> run on the analog-on-
# top NVC<->Xyce runtime. Demonstrates a real mixed-signal cosim end to end.
#   usage: ./gen_run.sh <design.net> [stop-time]
# ENGINE=vacask runs the analog on VACASK instead: cir2vacask.py turns the
# Xyce deck into <design>.sim (same .boundary and VHDL) and nvc drives
# libvacaskcinterface (VCB = VACASK build dir, see VACASK demo/cosim).
set -e
NET=${1:?usage: gen_run.sh design.net [stop]}; STOP=${2:-2us}
base=$(basename "$NET" .net)
ENGINE=${ENGINE:-xyce}
NVCB=${NVCB:-/usr/local/src/nvc-build}; NVC=$NVCB/bin/nvc; LIBS=$NVCB/lib
XYCE_LIBDIR=${XYCE_LIBDIR:-$HOME/xyce-libs}
XCI=${XCI:-/usr/local/src/xyce-build/utils/XyceCInterface}
VCB=${VCB:-/opt/build.VACASK/cosim}
UTILS=/usr/local/src/xyce/utils
export LD_LIBRARY_PATH=$NVCB:$NVCB/lib:$XYCE_LIBDIR:$XCI:/usr/local/src/xyce-build/src:$VCB/cinterface
export SIM_MODULE_PATH=${SIM_MODULE_PATH:-$VCB/devices}
export SIM_OPENVAF=${SIM_OPENVAF:-/opt/openvaf-r-20260616/openvaf-r}   # behavioral sources compile via OpenVAF-r

perl $UTILS/simetrix_cosim.pl -o $base "$NET"
grep -q '^\.print\|^\.PRINT' $base.cir || sed -i 's/^\.end/.print tran V(*)\n.end/I' $base.cir
case $ENGINE in
  xyce)   ANALOG="--xyce-netlist=$base.cir";;
  vacask) python3 $UTILS/cir2vacask.py $base.cir -o $base.sim
          ANALOG="--vacask-netlist=$base.sim";;
  *) echo "ENGINE must be xyce or vacask"; exit 1;;
esac
rm -rf s2x work_$base
$NVC --std=2040 -L $LIBS --work=s2x -a $UTILS/simetrix_vhdl/xspice_digital.vhd
$NVC --std=2040 -L $LIBS -L . --work=work_$base -a $base.vhd
$NVC --std=2040 -L $LIBS -L . --work=work_$base -e ${base}_tb
$NVC --std=2040 -L $LIBS -L . --work=work_$base -r --stop-time=$STOP \
     $ANALOG --cosim-config=$base.boundary ${base}_tb
if [ $ENGINE = xyce ]; then
  echo "--- $base.cir.prn written ---"; [ -f $base.cir.prn ] && tail -2 $base.cir.prn
else
  echo "--- tran1.raw written (VACASK rawfile) ---"
fi
