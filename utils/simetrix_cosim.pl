#!/usr/bin/env perl
#
# simetrix_cosim.pl — emit the analog-on-top cosim triple from a SIMetrix
# (XSPICE mixed-signal) netlist: the Xyce analog deck (digital removed, PWL
# "code:" boundary sources added), the cosim.boundary map, and a VHDL
# testbench instantiating the digital code models from simetrix_vhdl.
#
# Domain split is EXPLICIT in the netlist: adc_bridge (A) marks analog->digital
# (its analog input net is an A2D boundary), dac_bridge marks digital->analog
# (its analog output net is a D2A boundary). Everything else `A`-device is pure
# digital and runs under nvc; everything non-`A` is analog and runs in Xyce.
#
# First cut: flat netlists (digital A-devices at top level). Digital buried in
# .subckt (e.g. the flyback's UC3844) needs flattening first -- a follow-on.
#
# CLI: simetrix_cosim.pl -o <basename> in.net   (writes <basename>.{cir,boundary,vhd})
#
use strict;
use warnings;
use Getopt::Long;

my $base = 'cosim';
GetOptions('o=s' => \$base) or die "usage: $0 -o base in.net\n";
my $in = shift @ARGV or die "usage: $0 -o base in.net\n";
(my $stem = $base) =~ s{.*/}{};          # VHDL entity name: basename, sanitized
$stem =~ s/[^A-Za-z0-9_]/_/g;

open my $fh, '<', $in or die "cannot read $in: $!\n";
my @lines = map { s/\r?\n$//r } <$fh>;
close $fh;

# map .model <name> <codemodel> [params]
my (%amodel, %aparams);
for my $l (@lines) {
    next unless $l =~ /^\s*\.model\s+(\S+)\s+(d_\w+|adc_\w+|dac_\w+)\b\s*(.*)$/i;
    $amodel{ lc $1 } = lc $2;
    $aparams{ lc $1 } = $3;
}

# XSPICE code model -> (VHDL entity, kind). kind: logic | a2d(adc) | d2a(dac)
my %MAP = (
    d_inverter => ['xsp_d_inverter', 'logic'],
    d_buffer   => ['xsp_d_buffer',   'logic'],
    d_nand     => ['xsp_d_nand',     'logic'],
    d_or       => ['xsp_d_or',       'logic'],
    d_tff      => ['xsp_d_tff',      'logic'],
    d_pullup   => ['xsp_d_pullup',   'logic'],
    d_pulldown => ['xsp_d_pulldown', 'logic'],
    adc_bridge => ['xsp_adc_bridge', 'a2d'],
    adc_schmitt=> ['xsp_adc_schmitt','a2d'],
    dac_bridge => ['xsp_dac_bridge', 'd2a'],
);

# Parse A-devices: A<name> <nodes...> <model>. Bracketed [..] groups one bus
# port. The model is the last bare token.
my @adev;
for my $l (@lines) {
    next unless $l =~ /^A\S*\s/i;
    (my $body = $l) =~ s/^(A\S*)\s+//; my $name = $1; $name =~ s/^A\$?/A/;
    my $model = ($body =~ s/\s+(\S+)\s*$//) ? $1 : '';
    my @nodes;
    while ($body =~ /\G\s*(?:\[([^\]]*)\]|(\S+))/gc) {
        push @nodes, defined $1 ? [ split ' ', $1 ] : $2;
    }
    my $type = $amodel{ lc $model } // lc $model;
    push @adev, { name => $name, nodes => \@nodes, model => $model, type => $type,
                  map => $MAP{$type}, src => $l };
}
sub _adline { my $a = shift; return $a->{src}; }

# Boundary nets: a2d device's analog input (first node) -> Xyce drives nvc;
# d2a device's analog output (last node) -> nvc drives Xyce.
my (%a2d, %d2a);   # net -> nvc signal name (== net here)
for my $a (@adev) {
    next unless $a->{map};
    my $kind = $a->{map}[1];
    if    ($kind eq 'a2d') { my $n = _flat($a->{nodes}[0]); $a2d{$n} = $n; }
    elsif ($kind eq 'd2a') { my $n = _flat($a->{nodes}[-1]); $d2a{$n} = $n; }
}
sub _flat { my $x = shift; ref $x ? $x->[0] : $x; }

# ---- emit Xyce deck: preserve structure (.subckt hierarchy intact -- no
# flattening). Replace each analog<->digital BRIDGE in-place with its code:
# PWL boundary source (matching "mark up the PWL sources in the subckt"); drop
# pure-digital A-devices (they live in nvc) and the code-model .model cards.
# Signal name == the analog boundary net (kept in sync with the nvc VHDL).
my $BR = 'libcosim_bridge.so:nvc_bridge_init';
my %ad = map { _adline($_) => $_ } @adev;
open my $cir, '>', "$base.cir" or die;
print $cir "* [s2x cosim] Xyce deck: analog + in-place code: PWL boundaries\n";
for my $l (@lines) {
    if (my $a = $ad{$l}) {                          # an A-device, in whatever scope
        my $kind = $a->{map} ? $a->{map}[1] : 'logic';
        if ($kind eq 'a2d') {                       # adc_bridge: Xyce node -> nvc
            my $n = _flat($a->{nodes}[0]);
            print $cir qq{I_$a->{name} $n 0 PWL FILE "code:$BR:a2d:$n"\n};
        } elsif ($kind eq 'd2a') {                  # dac_bridge: nvc -> Xyce node
            my $n = _flat($a->{nodes}[-1]);
            print $cir qq{V_$a->{name} $n 0 PWL FILE "code:$BR:d2a:$n"\n};
        } else {
            print $cir "* [s2x] digital (in nvc): $l\n";   # pure logic -> dropped
        }
        next;
    }
    next if $l =~ /^\s*\.model\s+\S+\s+(d_\w+|adc_\w+|dac_\w+)\b/i;
    if ($l =~ /^\.GRAPH\s+(\S+)/i) { print $cir ".PRINT TRAN V($1)\n"; }
    else                           { print $cir "$l\n"; }
}
close $cir;

# ---- emit cosim.boundary
# Field 1 is the NVC signal path: the VHDL testbench declares every net as
# "sig_<net>" (sig_ prefix avoids clashing with keywords/entity names), so the
# boundary path must carry that prefix to resolve. Field 2 is the bridge signal
# name, which must equal the net used in the .cir "code:...:{d2a,a2d}:<net>" URI.
open my $bnd, '>', "$base.boundary" or die;
print $bnd "# [s2x cosim] boundary map (analog-on-top)\n";
print $bnd "D2A .sig_$d2a{$_} $_\n" for sort keys %d2a;
print $bnd "A2D .sig_$a2d{$_} $_\n" for sort keys %a2d;
close $bnd;

# ---- emit VHDL testbench: instantiate the digital primitives, wire by net
my %sig;  # every net touched by a digital device becomes a logic3da signal
for my $a (@adev) { $sig{ _flat($_) } = 1 for map { ref $_ ? @$_ : $_ } @{$a->{nodes}} }
open my $vhd, '>', "$base.vhd" or die;
print $vhd <<"HDR";
-- [s2x cosim] digital testbench for $in (runs under nvc, cosim with Xyce)
library ieee; use ieee.std_logic_1164.all;
library sv2vhdl; use sv2vhdl.logic3d_types_pkg.all; use sv2vhdl.logic3da_pkg.all;
library s2x;

entity ${stem}_tb is end entity;
architecture cosim of ${stem}_tb is
HDR
print $vhd "    signal sig_$_ : resolved_logic3da;\n" for sort keys %sig;
print $vhd "begin\n";
my $idx = 0;
for my $a (@adev) {
    my $m = $a->{map} or do {
        print $vhd "    -- UNMAPPED code model: $a->{name} ($a->{type})\n"; next;
    };
    my $ent = $m->[0];
    my @flat = map { _flat($_) } @{$a->{nodes}};
    my $pm;
    if ($m->[1] eq 'logic' && $ent =~ /nand|_or$/) {
        # vector-input gates: map inputs onto i(0..), n_used, last node is output
        my @ins = @flat[0 .. $#flat-1]; my $out = $flat[-1];
        my $im = join(", ", map { "$_ => sig_$ins[$_]" } 0 .. $#ins);
        $pm = "i($im), n_used => " . scalar(@ins) . ", o => sig_$out";
    } else {
        # scalar entities: map ports positionally by entity convention
        $pm = _scalar_portmap($ent, \@flat);
    }
    printf $vhd "    u%d: entity s2x.%s(rtl) port map( %s );\n", $idx++, $ent, $pm;
}
print $vhd "end architecture;\n";
close $vhd;

print STDERR "wrote $base.cir, $base.boundary, $base.vhd  "
           . "(" . scalar(@adev) . " A-devices, "
           . scalar(keys %d2a) . " D2A, " . scalar(keys %a2d) . " A2D)\n";

sub _scalar_portmap {
    my ($ent, $n) = @_;
    return "i => sig_$n->[0], o => sig_$n->[1]"   if $ent =~ /inverter|buffer/;
    return "o => sig_$n->[0]"                      if $ent =~ /pullup|pulldown/;
    return "clk => sig_$n->[0], q => sig_$n->[1]"  if $ent =~ /tff/;
    return "a => sig_$n->[0], d => sig_$n->[1]"    if $ent =~ /adc/;   # analog in, digital out
    return "d => sig_$n->[0], a => sig_$n->[1]"    if $ent =~ /dac/;   # digital in, analog out
    return join(", ", map { "p$_ => sig_$n->[$_]" } 0 .. $#$n);
}
