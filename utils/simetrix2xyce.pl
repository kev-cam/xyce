#!/usr/bin/env perl
#
# simetrix2xyce.pl — SIMetrix netlist (.net) -> Xyce, mixed-signal aware.
#
# SIMetrix's netlist (design.net, written next to the .sxsch on Run) is XSPICE
# mixed-signal: standard SPICE analog plus XSPICE digital `A` code-model
# devices (d_nand, d_inverter, adc_bridge, ...). Xyce has no XSPICE A device,
# so per dkc's architecture the analog goes to Xyce and each digital primitive
# becomes a Verilog/VHDL block co-simulated via nvc, with the analog/digital
# boundary handled by Xyce PWL "code:" source hooks (see test_inv_chain). The
# master flip (analog-on-top) is a separate later step.
#
# This pass: translate the SPICE analog + SIMetrix syntactic forms, and
# CATALOG the A-device code models (-> what the VHDL primitive library needs).
# CLI: simetrix2xyce.pl [-o out.cir] in.net
#
use strict;
use warnings;
use Getopt::Long;

my $opt_out;
my $verbose;
GetOptions('o=s' => \$opt_out, 'v|verbose' => \$verbose) or die "usage: $0 [-o out] in.net\n";
my $in = shift @ARGV or die "usage: $0 [-o out] in.net\n";

open my $fh, '<', $in or die "$0: cannot read $in: $!\n";
my @lines = <$fh>;
close $fh;
s/\r?\n$// for @lines;

my (@out, @changes, @warnings, @prints);
my %adev;             # code-model type -> count (the VHDL library scope)
my %amodel;           # .model name -> code-model type (for A-device resolution)

# Pass 1: map .model <name> <codemodel> so A-device model refs resolve to type
for my $l (@lines) {
    $amodel{ lc $1 } = lc $2
        if $l =~ /^\s*\.model\s+(\S+)\s+(d_\w+|adc_\w+|dac_\w+|a\w+_bridge)\b/i;
}

for my $raw (@lines) {
    (my $s = $raw) =~ s/^\s+|\s+$//g;

    if (!length $s)            { push @out, ""; next; }
    if ($s =~ /^\*#SIMETRIX/)  { push @out, "* [s2x] $s"; next; }   # tool marker
    if ($s =~ /^\*/)           { push @out, $s; next; }             # comment

    # .GRAPH <node> ... : SIMetrix probe directive. Collect the probed node
    # for a consolidated .PRINT; drop the directive.
    if ($s =~ /^\.GRAPH\s+(\S+)/i) {
        push @prints, $1;
        push @changes, ".GRAPH $1 -> .PRINT collection";
        push @out, "* [s2x] probe: $s";
        next;
    }

    # A-device: XSPICE digital code model. Model name is the LAST bare token;
    # resolve to its code-model type and tally. Emitted commented for now --
    # the VHDL-subckt + PWL-boundary rewrite is the next stage.
    if ($s =~ /^A/i) {
        my @t = split /\s+/, $s;
        my $model = $t[-1];
        my $type  = $amodel{ lc $model } // "?($model)";
        $adev{$type}++;
        push @warnings, "A-device $t[0] -> code model $type (needs VHDL block)";
        push @out, "* [s2x] DIGITAL A-device (cosim-pending): $s";
        next;
    }

    # X subckt instance: strip the SIMetrix '$' in the name, the trailing
    # "pinnames: ..." annotation, and turn a "... SUBCKT : p=v ..." param
    # clause into Xyce "PARAMS: p=v ...".
    if ($s =~ /^X/i) {
        $s =~ s/^X\$+/X/;                         # X$U1 -> XU1
        s/\$+//g for ($s);                        # $$subcktname (model token) -> subcktname
        my $note = ($s =~ s/\s+pinnames:.*$//i) ? 1 : 0;
        if ($s =~ s/\s+:\s+/ PARAMS: /) {        # subckt param clause
            push @changes, "subckt param ':' -> PARAMS:";
        }
        push @changes, "X-instance: stripped \$ / pinnames" if $note || $s =~ /PARAMS:/;
        push @out, $s;
        next;
    }

    # .subckt definitions: sanitize '$'/'$$' in the subckt name (SIMetrix uses
    # $$autogen names); the referencing X-instance trailing token is sanitized
    # the same way so the two still match.
    if ($s =~ /^\.subckt\s/i) {
        if ($s =~ s/^(\.subckt\s+)\$+/$1/i) { push @changes, "subckt name: stripped \$"; }
        push @out, $s;
        next;
    }

    # Everything else is standard SPICE -> pass through.
    push @out, $s;
}

# Consolidated .PRINT for the probed nodes, before the final .end
if (@prints) {
    my %seen; my @u = grep { !$seen{$_}++ } @prints;
    my $pr = ".PRINT TRAN " . join(' ', map { "V($_)" } @u);
    my ($end) = grep { $out[$_] =~ /^\s*\.end\s*$/i } reverse 0 .. $#out;
    if (defined $end) { splice @out, $end, 0, $pr; } else { push @out, $pr; }
}

my $ofh;
if (defined $opt_out) { open $ofh, '>', $opt_out or die "$0: cannot write $opt_out: $!\n"; }
else { $ofh = \*STDOUT; }
print {$ofh} "$_\n" for @out;
close $ofh if defined $opt_out;

# Catalog to stderr: the A-device code models = the VHDL primitive library scope
if (%adev) {
    print STDERR "\n== XSPICE digital code models referenced (VHDL library scope) ==\n";
    printf STDERR "  %-16s x%d\n", $_, $adev{$_} for sort keys %adev;
}
if ($verbose) {
    print STDERR "s2x: $_\n" for @changes;
    print STDERR "s2x: WARNING: $_\n" for @warnings;
}
