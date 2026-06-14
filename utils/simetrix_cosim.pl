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
            # A real dac_bridge has finite output conductance (g_pullup/
            # g_pulldown); an IDEAL voltage source driving nonlinear analog
            # (e.g. a BJT base) collapses the transient timestep. Drive an
            # internal node and add a series output resistance R = 1/g_pullup
            # if the model gives one, else a nominal 50 ohm. n.b. multi-line
            # .model continuations aren't parsed, so most fall to the default.
            my $p = $aparams{ lc $a->{model} } // '';
            my $g = ($p =~ /g_pullup\s*=\s*([\d.eE+-]+)/i) ? $1 : 0;
            my $r = ($g > 0) ? 1.0 / $g : 50;
            print $cir qq{V_$a->{name} ${n}_drv 0 PWL FILE "code:$BR:d2a:$n"\n};
            printf $cir "R_%s %s_drv %s %.4g\n", $a->{name}, $n, $n, $r;
        } else {
            print $cir "* [s2x] digital (in nvc): $l\n";   # pure logic -> dropped
        }
        next;
    }
    next if $l =~ /^\s*\.model\s+\S+\s+(d_\w+|adc_\w+|dac_\w+)\b/i;
    if ($l =~ /^\.GRAPH\s+(\S+)/i) { print $cir ".PRINT TRAN V($1)\n"; }
    # SIMetrix allows a single-arg ".tran <tstop>" (auto step); Xyce's .TRAN
    # needs <tstep> <tstop>. Inject a nominal step = tstop/10000 (the cosim
    # loop governs the real stepping via simulateUntil anyway).
    elsif ($l =~ /^\s*\.tran\s+([\d.eE+-]+)([a-zA-Z]*)\s*$/i) {
        my ($n, $u) = ($1, $2);
        printf $cir ".tran %s%s %s%s\n", $n/10000, $u, $n, $u;
    }
    else {
        # SIMetrix syntactic cleanup on analog instance/subckt lines (mirrors
        # simetrix2xyce.pl): strip '$' in X-instance and subckt names, drop the
        # trailing "pinnames: ..." annotation, and turn a " : p=v" subckt param
        # clause into Xyce "PARAMS: p=v".
        if ($l =~ /^X/i) {
            $l =~ s/^X\$+/X/;
            $l =~ s/\$+//g;
            $l =~ s/\s+pinnames:.*$//i;
            $l =~ s/\s+:\s+/ PARAMS: /;
        }
        elsif ($l =~ /^\.subckt\s/i) {
            $l =~ s/^(\.subckt\s+)\$+/$1/i;
        }
        print $cir "$l\n";
    }
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
# Build the body first so we can also collect the extra signal declarations
# (vector-gate input buses) that must precede `begin`.
my (@busdecls, @body);
my $idx = 0;
for my $a (@adev) {
    my $m = $a->{map} or do {
        push @body, "    -- UNMAPPED code model: $a->{name} ($a->{type})";
        next;
    };
    my $ent = $m->[0];
    my @flat = map { _flat($_) } @{$a->{nodes}};
    my $pm;
    if ($m->[1] eq 'logic' && $ent =~ /nand|_or$/) {
        # vector-input gates: inputs are every node but the last, with any
        # bracketed bus group EXPANDED to its members (a 2-input nand is
        # written "[a b] o" -> nodes (a,b) then o); _flat collapses a bus to
        # its first element, so expand here instead. Drive the vector port
        # through an intermediate bus signal: passing a named aggregate
        # directly as a port actual trips an nvc inertial-actual codegen bug.
        # n_used tells the entity how many low elements are live.
        my @nodes = @{$a->{nodes}};
        my $out = _flat($nodes[-1]);
        my @ins = map { ref $_ ? @$_ : $_ } @nodes[0 .. $#nodes-1];
        my $bus = "u${idx}_ibus";
        push @busdecls, "    signal $bus : logic3da_vector(0 to 7);";
        my $im = join(", ", map { "$_ => sig_$ins[$_]" } 0 .. $#ins);
        push @body, "    $bus <= ($im, others => L3DA_0);";
        $pm = "i => $bus, n_used => " . scalar(@ins) . ", o => sig_$out";
    } else {
        # scalar entities: map ports positionally by entity convention
        $pm = _scalar_portmap($ent, \@flat);
    }
    push @body, sprintf("    u%d: entity s2x.%s(rtl) port map( %s );",
                        $idx++, $ent, $pm);
}
print $vhd "    signal sig_$_ : resolved_logic3da;\n" for sort keys %sig;
print $vhd "$_\n" for @busdecls;
print $vhd "begin\n";
print $vhd "$_\n" for @body;
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
