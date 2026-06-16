#!/usr/bin/env perl
#
# sxsch2net.pl — headless SIMetrix .sxsch schematic -> netlist (what the GUI
# writes as design.net on Run). Removes the GUI dependency for the SIMetrix
# pipeline (simetrix2xyce.pl / simetrix_cosim.pl).
#
# The .sxsch is ASCII. Structure:
#   .Symbol  blocks: a "TEMPLATE" property (e.g. "<ref> <nodelist> <value>")
#            and ordered Pins ("Pin name=.. order=N").
#   .Instance blocks: Property values (ref/model/value/params/...) plus a
#            "Netnames pinN=net" line giving each pin's net (connectivity is
#            EXPLICIT — no geometric net extraction needed).
#   Text value="..": the F11 command block — analysis + .model/.subckt cards.
#
# First cut: flat schematics with the standard device templates. Emits device
# lines (template-expanded) + the command/model/subckt text. Validated against
# the flyback's GUI design.net: 40/41 instances reproduced exactly.
#
# NOT handled yet (a SIMetrix script runs in the GUI to expand these):
#   - scripted/parameterised symbols, e.g. IdealTx transformers that synthesise
#     L+K from idealTx* properties (the flyback's TX2 -- the 1 missing instance);
#   - behavioural logic blocks in .sxldf files (d_logic_block, e.g. 4046 PLL).
#
# CLI: sxsch2net.pl file.sxsch [-o design.net]
#
use strict;
use warnings;

my ($in, $out);
while (@ARGV) { my $a = shift; if ($a eq '-o') { $out = shift } else { $in = $a } }
die "usage: $0 file.sxsch [-o out]\n" unless $in;

open my $fh, '<', $in or die "$0: cannot read $in: $!\n";
local $/; my $blob = <$fh>; close $fh;
$blob =~ s/\r//g;

# unescape a quoted SIMetrix value ("\n" -> newline, \" -> ", \\ -> \)
sub unesc { my $s = shift; $s =~ s/\\n/\n/g; $s =~ s/\\t/\t/g; $s =~ s/\\(["\\])/$1/g; return $s; }

# 1. Symbols: name -> { tmpl, pins => [names in pin order] }
my %sym;
while ($blob =~ /^\.Symbol\b(.*?)^\.EndSymbol/msg) {
    my $b = $1;
    my ($name) = $b =~ /^Attributes\b[^\n]*?\bname="([^"]*)"/m;
    next unless defined $name;
    my ($tmpl) = $b =~ /name="TEMPLATE"\s+value="((?:[^"\\]|\\.)*)"/;
    my @pins;
    while ($b =~ /^Pin\s+name="([^"]*)"\s+order=(\d+)/mg) { $pins[$2 - 1] = $1; }
    $sym{$name} = { tmpl => $tmpl, pins => \@pins };
}

# 2. Instances: symbol + properties + pin->net map
my @inst;
while ($blob =~ /^\.Instance\b(.*?)^\.EndInstance/msg) {
    my $b = $1;
    my ($sn) = $b =~ /^Attributes type=symbol\s+name="([^"]*)"/m;
    my %prop;
    while ($b =~ /^Property\s+name="([^"]*)"\s+value="((?:[^"\\]|\\.)*)"/mg) {
        $prop{ lc $1 } = unesc($2);
    }
    my %net;
    if ($b =~ /^Netnames\s+(.*)$/m) {
        my $nn = $1; while ($nn =~ /pin(\d+)="([^"]*)"/g) { $net{$1} = $2; }
    }
    push @inst, { sym => $sn, prop => \%prop, net => \%net };
}

# 2b. The command/model/subckt Text block(s) — pull them now so we know which
# instance "models" are actually .subckt names (those netlist as X<ref>).
my @text;
while ($blob =~ /^Text\s+value="((?:[^"\\]|\\.)*)"/mg) { push @text, unesc($1); }
my %subckt;
for my $t (@text) { $subckt{$1} = 1 while $t =~ /^\s*\.subckt\s+(\S+)/img; }
# XSPICE code-model names (.model <name> d_*/adc_*/dac_*) -> instances using
# them netlist as A-devices (A$<ref> <nodes> <model>), per SIMetrix.
my %codemodel;
for my $t (@text) {
    $codemodel{ lc $1 } = 1
        while $t =~ /^\s*\.model\s+(\S+)\s+(?:d_\w+|adc_\w+|dac_\w+|a\w+_bridge)\b/img;
}

# 3. Expand each instance's template into a netlist line.
# Connectivity comes from Netnames (pinN -> net); the symbol (if embedded)
# supplies the TEMPLATE, else we fall back by symbol/ref type. Standard library
# symbols (res/cap/ind/dio/...) aren't embedded in the .sxsch -- they all use
# "<ref> <nodelist> <value>", and ground/probe symbols are special-cased.
my (@dev, @graph);
for my $i (@inst) {
    my $p  = $i->{prop};
    my $sn = $i->{sym} // '';
    # nodes = pin nets in pin-number order, straight from Netnames
    my $maxpin = 0; for (keys %{ $i->{net} }) { $maxpin = $_ if $_ > $maxpin; }
    my @nodes = map { defined $i->{net}{$_} ? $i->{net}{$_} : '?' } 1 .. $maxpin;

    # ground/connector + text-annotation symbols emit no device
    next if $sn =~ /^(gnd|ground|0|node)$/i;
    next if $sn =~ /free_text|text|annotation|title|comment/i;

    # classify by the instance "value": a .subckt name -> X$<ref>; an XSPICE
    # code-model name -> A$<ref> (digital); else a plain device.
    my $mv = defined $p->{value} ? $p->{value} : '';
    my $is_sub = length $mv && $subckt{$mv};
    my $is_a   = length $mv && $codemodel{ lc $mv };

    my $s = $sym{$sn};
    my $t = ($s && defined $s->{tmpl} && length $s->{tmpl}) ? $s->{tmpl}
          : $is_a   ? 'A$<ref> <nodelist> <value>'   # XSPICE digital A-device
          : $is_sub ? 'X$<ref> <nodelist> <value>'   # subckt instance
          :           '<ref> <nodelist> <value>';     # default device template

    my $line = $t;
    $line =~ s/<ref>/defined $p->{ref}    ? $p->{ref}    : ''/ge;
    $line =~ s/<model>/defined $p->{model} ? $p->{model} : ''/ge;
    $line =~ s/<value>/defined $p->{value} ? $p->{value} : ''/ge;
    $line =~ s/<paramsvalue>/defined $p->{params} ? $p->{params} : ''/ge;
    $line =~ s/<nodelist>/join(' ', @nodes)/ge;
    $line =~ s/<node\[(\d+)\]>/defined $nodes[$1-1] ? $nodes[$1-1] : '?'/ge;
    $line =~ s/%VALUE%/defined $p->{value} ? $p->{value} : ''/ge;
    $line =~ s/<[^>]*>//g;            # drop unhandled <...> conditionals
    $line =~ s/%[^%]*%//g;            # drop unhandled %...% macros
    $line =~ s/\s+/ /g; $line =~ s/^\s+|\s+$//g;
    next unless length $line && $line !~ /^[A-Za-z]+$/;   # skip empty / token-only

    if ($line =~ /^\.GRAPH/i) { push @graph, $line; } else { push @dev, $line; }
}

# ---- emit
my $ofh; if (defined $out) { open $ofh, '>', $out or die "$0: cannot write $out: $!\n"; } else { $ofh = \*STDOUT; }
print {$ofh} "* [sxsch2net] $in\n";
print {$ofh} "$_\n" for @dev;
print {$ofh} "$_\n" for @graph;
print {$ofh} "$_\n" for @text;
print {$ofh} ".end\n";
close $ofh if defined $out;

printf STDERR "sxsch2net: %d devices, %d probes, %d text blocks\n",
    scalar(@dev), scalar(@graph), scalar(@text);
