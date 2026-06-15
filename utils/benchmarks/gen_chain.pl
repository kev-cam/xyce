#!/usr/bin/perl
# CMOS inverter chain of N stages: 2N MOSFETs + N load caps + ~N nodes.
# Driven by a pulse; short transient lets the edge propagate. Portable SPICE.
use strict; use warnings;
my $N = shift // 1000;
open my $f, '>', "chain$N.cir" or die;
print $f "* CMOS inverter chain, N=$N ($N*2 MOS, $N caps)\n";
print $f "Vdd vdd 0 1.8\n";
print $f "Vin n0 0 PULSE(0 1.8 1n 0.5n 0.5n 20n 40n)\n";
print $f ".model NM NMOS(LEVEL=1 VTO=0.4 KP=120u)\n";
print $f ".model PM PMOS(LEVEL=1 VTO=-0.4 KP=60u)\n";
for my $i (1 .. $N) {
    my ($in, $out) = ("n".($i-1), "n$i");
    print $f "MN$i $out $in 0 0 NM\n";
    print $f "MP$i $out $in vdd vdd PM\n";
    print $f "C$i $out 0 5f\n";
}
print $f ".save V(n$N)\n";
print $f ".tran 0.1n 50n\n";
print $f ".end\n";
close $f;
print "chain$N.cir: ", ($N*3+4), " lines, ", $N*2, " MOS, $N caps\n";
