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

    my @extract_libs;          # QSPICE-install libs queued for model extraction
    my %models_inline;         # model names defined inline in the deck
    my %params_defined;        # .param/.step param names (for bare-ref bracing)

    # Pass 1: analysis type, for .plot/.print lines that omit it (QSPICE's
    # netlister writes ".plot Ic(Q1)" with no analysis keyword).
    my $analysis_type;
    for my $line (@$lines_ref) {
        my $u = uc($line);
        if    ($u =~ /^\s*\.TRAN\b/)  { $analysis_type = 'TRAN' }
        elsif ($u =~ /^\s*\.AC\b/)    { $analysis_type = 'AC' }
        elsif ($u =~ /^\s*\.DC\b/)    { $analysis_type = 'DC' }
        elsif ($u =~ /^\s*\.OP\b/)    { $analysis_type ||= 'DC' }
        elsif ($u =~ /^\s*\.NOISE\b/) { $analysis_type = 'NOISE' }
        $models_inline{ uc $1 } = 1 if $line =~ /^\s*\.model\s+(\S+)/i;
        $params_defined{ uc $1 } = 1 if $line =~ /^\s*\.param\s+(\w+)/i;
        $params_defined{ uc $1 } = 1 if $line =~ /^\s*\.step\s+(?:(?:oct|dec|lin)\s+)?param\s+(\w+)/i;
    }

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

        # .plot/.print: QSPICE's netlister omits the analysis keyword, separates
        # signals with commas, and allows bare expressions ("V(a)-V(b)"). Xyce
        # requires the analysis type, space separation, and {braces} around
        # anything that isn't a plain V(...)/I*(...) probe.
        if ($stripped =~ /^\.(?:PLOT|PRINT)\s+(.+)/i) {
            my $args = $1;
            my $kw = uc((split /\s+/, $stripped)[0]);
            my $type = '';
            if ($args =~ s/^\s*(TRAN|AC|DC|NOISE|HB)\b\s*//i) { $type = uc $1; }
            $type ||= $analysis_type || '';
            my @items = split /\s*,\s*/, $args;
            my @out_items;
            for my $it (@items) {
                $it =~ s/^\s+|\s+$//g;
                next unless length $it;
                # plain probe(s) (possibly several space-separated) pass through;
                # anything with operators outside the probe parens gets braces
                if ($it =~ /^(?:[VIP][a-z]*\(\s*[^()]+\s*\)\s*)+$/i) {
                    push @out_items, $it;
                } else {
                    push @out_items, "{$it}";
                }
            }
            my $sig = join ' ', @out_items;
            push @changes, "L$lineno: $kw -> .PRINT $type $sig";
            push @output, ".PRINT $type $sig\n";
            next;
        }

        # .tran single-arg "<Tstop>" -> ".TRAN <Tstop/1000> <Tstop>" (same rule
        # as ltspice2xyce.pl; Xyce needs an explicit step).
        if ($upper =~ /^\.TRAN\s/) {
            my $t = $stripped;
            my $has_uic = ($t =~ s/\s+(?:uic|startup)\s*$//i);
            my @f = split /\s+/, $t;
            if (@f == 2 && $f[1] =~ /^([\d.]+)([a-z]*)$/i) {
                push @changes, "L$lineno: .tran single-arg -> step added";
                my %mult = ('f'=>1e-15,'p'=>1e-12,'n'=>1e-9,'u'=>1e-6,'m'=>1e-3,
                            'ms'=>1e-3,'k'=>1e3,'meg'=>1e6,'g'=>1e9,''=>1,'s'=>1);
                my $m = $mult{ lc $2 } // 1;
                my $tstop = $1 * $m;
                my $tstep = $tstop / 1000;
                push @output, ".TRAN $tstep $tstop\n";
                next;
            }
            push @changes, "L$lineno: dropped uic/startup" if $has_uic;
            push @output, "$t\n";
            next;
        }

        # .step param NAME ... -> .STEP NAME ... (same rule as ltspice2xyce.pl)
        if ($upper =~ /^\.STEP\b/) {
            if ($stripped =~ /^\.step\s+(?:(oct|dec|lin)\s+)?param\s+(.+)/i) {
                my $mode = $1 ? uc($1) . ' ' : '';
                push @changes, "L$lineno: .step param -> .STEP";
                push @output, ".STEP $mode$2\n";
                next;
            }
            push @output, "$stripped\n";
            next;
        }

        # .lib / .include pointing into the QSPICE install: don't include the
        # whole lib (they mix plain models with QSPICE-level ones Xyce can't
        # parse, plus doc params); queue it for referenced-model extraction.
        if ($upper =~ /^\.(LIB|INCLUDE|INC)\b/) {
            my ($kw, $arg) = $stripped =~ /^(\S+)\s+(.*)$/;
            if (defined $arg && $arg =~ /\\|^[A-Za-z]:/) {
                my $mapped = _map_win_path($arg, $deck_dir);
                if ($mapped =~ m{\Q$qspice_dir\E}) {
                    push @changes, "L$lineno: $kw $arg -> deferred model extraction";
                    push @output, "* [qtz] models extracted from: $arg\n";
                    push @extract_libs, $mapped;
                    next;
                }
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

        # V/I source with a bare parameter name as its value: Xyce requires
        # braces ("V1 G 0 VGS" -> "V1 G 0 {VGS}" when VGS is a .param/.step name).
        if ($stripped =~ /^[VIvi]\S*\s/ && %params_defined) {
            my @tok = split /\s+/, $stripped;
            my $chg = 0;
            for my $i (3 .. $#tok) {
                if ($tok[$i] =~ /^\w+$/ && $params_defined{ uc $tok[$i] }) {
                    $tok[$i] = "{$tok[$i]}";
                    $chg = 1;
                }
            }
            if ($chg) {
                my $l = join ' ', @tok;
                push @changes, "L$lineno: braced bare param reference(s) on source line";
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

    # Post-pass: extract referenced models from the queued QSPICE-install libs.
    # A model is "referenced" if its name appears as a word in the deck and is
    # not already defined inline. Each extracted .model is sanitized; models of
    # types Xyce lacks (VDMOS, QSPICE numeric levels) are skipped with a warning.
    if (@extract_libs) {
        my $deck_text = join '', @output;
        my (@extracted, %seen);
        for my $lib (@extract_libs) {
            my $lfh;
            unless (open $lfh, '<', $lib) {
                push @warnings, "cannot read model lib $lib: $!";
                next;
            }
            while (my $ml = <$lfh>) {
                $ml =~ s/\r?\n$//;
                next unless $ml =~ /^\s*\.model\s+(\S+)\s+(\S+)(.*)$/i;
                my ($name, $type, $params) = ($1, $2, $3);
                next if $seen{ uc $name } || $models_inline{ uc $name };
                next unless $deck_text =~ /\b\Q$name\E\b/i;
                $seen{ uc $name } = 1;
                if ($type =~ /^VDMOS$/i || $params =~ /\blevel\s*=\s*(?:20\d\d)\b/i) {
                    push @warnings, "model $name ($type) uses a QSPICE/VDMOS level Xyce lacks -- needs a macromodel";
                    push @extracted, "* [qtz] UNSUPPORTED model skipped (needs macromodel): .model $name $type\n";
                    next;
                }
                my $clean = _sanitize_model_params($params);
                push @extracted, ".model $name $type$clean\n";
                push @changes, "extracted .model $name $type from " . basename($lib);
            }
            close $lfh;
        }
        if (@extracted) {
            # insert before the final .END (everything after .END is ignored)
            my ($end_idx) = grep { $output[$_] =~ /^\s*\.end\s*$/i } reverse 0 .. $#output;
            my @blk = ("* [qtz] models extracted from QSPICE libraries:\n", @extracted);
            if (defined $end_idx) { splice @output, $end_idx, 0, @blk; }
            else                  { push @output, @blk; }
        }
    }

    return (\@output, \@changes, \@warnings);
}

# Strip QSPICE documentation/extension params from a .model parameter list:
# mfg="..."/bare and Vrev=/Iave= datasheet annotations, Gp=/Tgp1=/Vp= QSPICE
# diode extensions Xyce lacks, "NAME= value" spacing, and µ -> u.
sub _sanitize_model_params {
    my ($p) = @_;
    $p =~ s/\xc2\xb5/u/g;  $p =~ s/\xb5/u/g;
    $p =~ s/\s+mfg\s*=\s*(?:"[^"]*"|\S+)//ig;
    $p =~ s/\s+(?:Vrev|Iave|Gp|Tgp1|Vp)\s*=\s*\S+//ig;
    $p =~ s/=\s+/=/g;
    return $p;
}
