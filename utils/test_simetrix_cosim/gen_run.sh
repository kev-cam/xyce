#!/bin/bash
# Full SIMetrix-cosim pipeline: a SIMetrix .net (digital A-devices, possibly in
# a .subckt) -> simetrix_cosim.pl -> {cir,boundary,vhd} -> run on the analog-on-
# top NVC<->Xyce runtime. Demonstrates a real mixed-signal cosim end to end.
#   usage: ./gen_run.sh <design.net> [stop-time]
set -e
NET=${1:?usage: gen_run.sh design.net [stop]}; STOP=${2:-2us}
base=$(basename "$NET" .net)
NVCB=${NVCB:-/usr/local/src/nvc-build}; NVC=$NVCB/bin/nvc; LIBS=$NVCB/lib
XYCE_LIBDIR=${XYCE_LIBDIR:-$HOME/xyce-libs}
XCI=${XCI:-/usr/local/src/xyce-build/utils/XyceCInterface}
UTILS=/usr/local/src/xyce/utils
export LD_LIBRARY_PATH=$NVCB:$NVCB/lib:$XYCE_LIBDIR:$XCI:/usr/local/src/xyce-build/src

perl $UTILS/simetrix_cosim.pl -o $base "$NET"
grep -q '^\.print\|^\.PRINT' $base.cir || sed -i 's/^\.end/.print tran V(*)\n.end/I' $base.cir
rm -rf s2x work_$base
$NVC --std=2040 -L $LIBS --work=s2x -a $UTILS/simetrix_vhdl/xspice_digital.vhd
$NVC --std=2040 -L $LIBS -L . --work=work_$base -a $base.vhd
$NVC --std=2040 -L $LIBS -L . --work=work_$base -e ${base}_tb
$NVC --std=2040 -L $LIBS -L . --work=work_$base -r --stop-time=$STOP \
     --xyce-netlist=$base.cir --xyce-config=$base.boundary ${base}_tb
echo "--- $base.cir.prn written ---"; [ -f $base.cir.prn ] && tail -2 $base.cir.prn
