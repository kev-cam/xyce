#!/usr/bin/env perl
#
# qspice2xyce.pl — QSPICE netlist (.cir) -> Xyce netlist translator.
#
# A streaming front-end in the same family as ltspice2xyce.pl, gnucap2xyce.pl
# and cadence2xyce.pl: read a QSPICE-flavour SPICE netlist (as written by the
# QSPICE GUI netlister from a .qsch), translate the QSPICE-isms Xyce doesn't
# accept, and emit a Xyce-runnable netlist.
#
# QSPICE dialect handled so far (grown test-first against the qspice-tests
# corpus, validated vs native QSPICE64.exe gold .qraw):
#   - BJT instance lines "Q<name> C B E [S] <model> <NPN|PNP> [params]":
#     unwrap the bracketed substrate node, drop the trailing type token.
#   - MOSFET/JFET trailing type tokens (NMOS/PMOS/NJF/PJF) likewise.
#   - .meas -> commented out (post-processing, not needed for the waveform;
#     same policy as ltspice2xyce.pl).
#   - .lib/.include with Windows paths: map into the QSPICE install or the
#     deck's own directory; 8.3 short names (PROGRA~1) expanded. Sectionless
#     .lib <file> -> .INCLUDE (Xyce treats 1-arg .LIB as sectioned).
#   - µ -> u, SINE( -> SIN( safety nets shared with the LTspice front-end.
#
# CLI: qspice2xyce.pl [-o out.cir] [--qspice-dir DIR] in.cir
#      --qspice-dir: where QSPICE's model .txt libs live, for .lib path
#      remapping (default: /mnt/c/Program Files/QSPICE, the WSL view).
#
use strict;
use warnings;
use Getopt::Long;
use File::Basename qw(dirname basename);

my $opt_out;
my $qspice_dir = '/mnt/c/Program Files/QSPICE';
my $verbose;
GetOptions('o=s' => \$opt_out, 'qspice-dir=s' => \$qspice_dir, 'v|verbose' => \$verbose)
    or die "usage: $0 [-o out.cir] [--qspice-dir DIR] in.cir\n";
my $in = shift @ARGV or die "usage: $0 [-o out.cir] [--qspice-dir DIR] in.cir\n";

open my $fh, '<', $in or die "$0: cannot read $in: $!\n";
my @lines = <$fh>;
close $fh;

my ($out, $changes, $warnings) = qcir_to_xyce(\@lines, dirname($in));

my $ofh;
if (defined $opt_out) {
    open $ofh, '>', $opt_out or die "$0: cannot write $opt_out: $!\n";
} else {
    $ofh = \*STDOUT;
}
print {$ofh} @$out;
close $ofh if defined $opt_out;

if ($verbose) {
    print STDERR "qspice2xyce: $_\n" for @$changes;
    print STDERR "qspice2xyce: WARNING: $_\n" for @$warnings;
}

# Map a Windows path from a QSPICE netlist to something Xyce-under-WSL can
# open. Expands the PROGRA~1 8.3 name, points QSPICE-install references at
# $qspice_dir, and falls back to the deck's own directory for bare names.
sub _map_win_path {
    my ($p, $deck_dir) = @_;
    $p =~ s/^["']|["']$//g;
    if ($p =~ m{^[A-Za-z]:\\}) {
        # QSPICE install (any spelling: 8.3 or long)
        if ($p =~ m{\\(?:PROGRA~1|Program Files)\\QSPICE\\(.+)$}i) {
            my $rest = $1; $rest =~ s{\\}{/}g;
            return "$qspice_dir/$rest";
        }
        # other absolute Windows path: translate drive to /mnt/<drive>
        $p =~ s{^([A-Za-z]):\\}{'/mnt/' . lc($1) . '/'}e;
        $p =~ s{\\}{/}g;
        return $p;
    }
    return $p;   # relative: leave for Xyce to resolve against cwd
}

sub qcir_to_xyce {
    my ($lines_ref, $deck_dir) = @_;
    my (@output, @changes, @warnings);
    my $lineno = 0;

    for my $line (@$lines_ref) {
        $lineno++;
        (my $stripped = $line) =~ s/\r?\n$//;
        $stripped =~ s/^\s+|\s+$//g;
        my $upper = uc($stripped);

        unless (length $stripped) { push @output, "\n"; next; }

        # Comments pass through
        if ($stripped =~ /^\*/) { push @output, "$stripped\n"; next; }

        # .meas: QSPICE op-point measures don't map onto Xyce .MEASURE; they are
        # post-processing and not needed for the waveform comparison.
        if ($upper =~ /^\.MEAS\b/) {
            push @changes, "L$lineno: .meas commented out";
            push @output, "* [qtz] .meas (not translated): $stripped\n";
            next;
        }

        # .lib / .include with a Windows path -> remap; sectionless .lib -> .INCLUDE
        if ($upper =~ /^\.(LIB|INCLUDE|INC)\b/) {
            my ($kw, $arg) = $stripped =~ /^(\S+)\s+(.*)$/;
            if (defined $arg && $arg =~ /\\|^[A-Za-z]:/) {
                my $mapped = _map_win_path($arg, $deck_dir);
                push @changes, "L$lineno: $kw windows path -> $mapped (.INCLUDE)";
                push @output, ".INCLUDE \"$mapped\"\n";
                next;
            }
            if ($upper =~ /^\.LIB\b/ && defined $arg) {
                my @t = split /\s+/, $arg;
                if (@t == 1 && $t[0] =~ m{[./]}) {
                    push @changes, "L$lineno: sectionless .LIB -> .INCLUDE";
                    push @output, ".INCLUDE $t[0]\n";
                    next;
                }
            }
            push @output, "$stripped\n";
            next;
        }

        # BJT: Q<name> C B E [S] <model> <NPN|PNP> [...] -> unwrap substrate,
        # drop the trailing type token QSPICE appends after the model name.
        if ($stripped =~ /^[Qq]\S*\s/) {
            my $l = $stripped;
            my $chg;
            $chg = 1 if $l =~ s/\[(\S+)\]/$1/;          # [0] -> 0
            $chg = 1 if $l =~ s/\s+(NPN|PNP|LPNP)\s*$//i;   # trailing type token
            if ($chg) {
                push @changes, "L$lineno: Q-line QSPICE form -> standard ($l)";
                push @output, "$l\n";
                next;
            }
        }

        # MOSFET/JFET trailing type tokens after the model name
        if ($stripped =~ /^[MJmj]\S*\s/) {
            my $l = $stripped;
            if ($l =~ s/\s+(NMOS|PMOS|NJF|PJF)\s*$//i) {
                push @changes, "L$lineno: dropped trailing device-type token";
                push @output, "$l\n";
                next;
            }
        }

        # Shared safety nets with the LTspice front-end
        if ($stripped =~ /\bSINE\s*\(/i) {
            (my $c = $stripped) =~ s/\bSINE\s*\(/SIN(/ig;
            push @changes, "L$lineno: SINE( -> SIN(";
            push @output, "$c\n";
            next;
        }
        if ($stripped =~ /\xb5|\xc2\xb5/) {
            (my $c = $stripped) =~ s/\xc2\xb5/u/g; $c =~ s/\xb5/u/g;
            push @changes, "L$lineno: micro sign -> u";
            push @output, "$c\n";
            next;
        }

        push @output, "$stripped\n";
    }

    return (\@output, \@changes, \@warnings);
}
