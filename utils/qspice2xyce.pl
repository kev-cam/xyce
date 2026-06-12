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
    my %param_cards;           # names that have an actual .param card
    my %stepped;               # .step param names -> start value
    my %need_builtin;          # QTZ_* subckt definitions to emit
    my @appendix_models;       # generated .model cards (switches etc.)

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
        if ($line =~ /^\s*\.param\s+(.*)$/i) {
            my $body = $1;
            while ($body =~ /(\w+)\s*=/g) {
                $params_defined{ uc $1 } = 1;
                $param_cards{ uc $1 } = 1;
            }
        }
        if ($line =~ /^\s*\.step\s+(?:(?:oct|dec|lin)\s+)?param\s+(\w+)\s+(\S+)/i) {
            $params_defined{ uc $1 } = 1;
            $stepped{ uc $1 } = $2;   # start value, for the .param card Xyce needs
        }
    }

    for my $line (@$lines_ref) {
        $lineno++;
        (my $stripped = $line) =~ s/\r?\n$//;
        $stripped =~ s/^\s+|\s+$//g;
        my $upper = uc($stripped);

        unless (length $stripped) { push @output, "\n"; next; }

        # Comments pass through; QSPICE also writes C++-style // comments
        if ($stripped =~ /^\*/) { push @output, "$stripped\n"; next; }
        if ($stripped =~ m{^//}) {
            (my $c = $stripped) =~ s{^//}{*};
            push @changes, "L$lineno: // comment -> *";
            push @output, "$c\n";
            next;
        }

        # micro sign -> u, up front: every branch below may emit the line
        # directly, so the conversion must happen before any of them
        if ($stripped =~ /\xb5|\xc2\xb5/) {
            $stripped =~ s/\xc2\xb5/u/g;
            $stripped =~ s/\xb5/u/g;
            $upper = uc($stripped);
            push @changes, "L$lineno: micro sign -> u";
        }

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
                # dtmax: Xyce's adaptive stepper otherwise grows steps past
                # signal periods on low-amplitude startups (kills oscillators)
                my $dtmax = $tstop / 5000;
                push @output, ".TRAN $tstep $tstop 0 $dtmax\n";
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

        # V/I source fixups, applied together:
        #  - QSPICE's netlister writes transient specs as bare keywords
        #    ("V3 IN SGND sine 0 100m 1K"); Xyce needs SIN(0 100m 1K).
        #  - a bare (possibly negated) .param/.step name as a value needs
        #    braces: "V1 POS 0 X" -> {X}, "V2 NEG 0 -X" -> {-X}.
        if ($stripped =~ /^[VIvi]\S*\s/) {
            my @tok = split /\s+/, $stripped;
            my $chg = 0;
            for my $i (3 .. $#tok) {
                if ($tok[$i] =~ /^(sine|sin|pulse|exp|sffm|pwl)$/i) {
                    my @args = splice @tok, $i + 1;
                    my @tail;
                    while (@args && $args[-1] =~ /^\w+=/) { unshift @tail, pop @args; }
                    (my $kw = uc $tok[$i]) =~ s/^SINE$/SIN/;
                    $tok[$i] = "$kw(@args)";
                    push @tok, @tail;
                    push @changes, "L$lineno: bare $kw source spec -> $kw(...)";
                    $chg = 1;
                    last;
                }
            }
            if (%params_defined) {
                for my $i (3 .. $#tok) {
                    if ($tok[$i] =~ /^-?(\w+)$/ && $params_defined{ uc $1 }) {
                        $tok[$i] = "{$tok[$i]}";
                        $chg = 1;
                    }
                }
            }
            if ($chg) {
                push @changes, "L$lineno: source line normalized";
                push @output, join(' ', @tok) . "\n";
                next;
            }
        }

        # R/C/L value that's a bare parameter name -> braces ("R1 N01 0 R")
        if ($stripped =~ /^[RCLrcl]\S*\s/ && %params_defined) {
            my @tok = split /\s+/, $stripped;
            my $chg = 0;
            for my $i (3 .. $#tok) {
                if ($tok[$i] =~ /^-?(\w+)$/ && $params_defined{ uc $1 }) {
                    $tok[$i] = "{$tok[$i]}";
                    $chg = 1;
                }
            }
            if ($chg) {
                push @changes, "L$lineno: braced bare param reference(s)";
                push @output, join(' ', @tok) . "\n";
                next;
            }
        }

        # QSPICE built-in mixed-mode devices: instance prefix is 0xC3 (analog,
        # e.g. the RRopAmp ideal op-amp) or 0xA5 (logic/mixed: HMITT comparator,
        # OR gate, SR-FLOP). Unconnected pins netlist as bare 0xA5 tokens. The
        # model name is the LAST bare token; name=value params follow QSPICE
        # conventions. Pin maps (verified against the example schematics):
        #   RRopAmp:           vdd vss out in- in+
        #   HMITT/OR/SR-FLOP:  vdd vss out outbar in1 in2
        # Each maps to an X instance of a synthesized QTZ_* behavioral subckt.
        if ($stripped =~ /^(?:\xc3\x83|\xc3|\xc2\xa5|\xa5)(\S*)\s+(.*)$/s) {
            my ($iname, $rest) = ($1, $2);
            $rest =~ s/\xc2\xb5/u/g;  $rest =~ s/\xb5/u/g;   # micro in param values
            my (@nodes, @bare, %prm);
            for my $t (split /\s+/, $rest) {
                if ($t =~ /^(\w[\w-]*)=(\S+)$/) { $prm{ uc $1 } = $2; }
                elsif ($t eq "\xa5" || $t eq "\xc2\xa5") {
                    push @nodes, sprintf 'qtz_nc%d_%s', scalar @nodes, ($iname =~ /^\w+$/ ? $iname : 'x');
                } else { push @bare, scalar @nodes; push @nodes, $t; }
            }
            # model = last bare (non-NC, non-param) token
            my $model = @bare ? splice(@nodes, $bare[-1], 1) : '';
            my %builtin = (
                'RROPAMP' => { sub => 'QTZ_RROPAMP', npins => 5,
                               params => { AVOL => 'AVOL', GBW => 'GBW' } },
                'HMITT'   => { sub => 'QTZ_HMITT',   npins => 6, params => {} },
                'OR'      => { sub => 'QTZ_OR2',     npins => 6, params => {} },
                'SR-FLOP' => { sub => 'QTZ_SRFLOP',  npins => 6,
                               params => { TRISE => 'TRISE' } },
            );
            my $b = $builtin{ uc $model };
            if ($b && @nodes >= $b->{npins}) {
                my @pins = @nodes[0 .. $b->{npins} - 1];
                my (@p, %used);
                while (my ($qname, $sname) = each %{ $b->{params} }) {
                    if (defined $prm{$qname}) { push @p, "$sname=$prm{$qname}"; $used{$qname} = 1; }
                }
                for my $u (grep { !$used{$_} } sort keys %prm) {
                    push @warnings, "$model $iname: parameter $u=$prm{$u} not modeled";
                }
                my $params = @p ? ' PARAMS: ' . join(' ', @p) : '';
                push @changes, "L$lineno: QSPICE builtin $model -> X instance of $b->{sub}";
                push @output, "Xqtz${iname} @pins $b->{sub}$params\n";
                $need_builtin{ $b->{sub} } = 1;
                next;
            }
            push @warnings, "L$lineno: unrecognized QSPICE builtin '$model' left as-is";
            push @output, "$stripped\n";
            next;
        }

        # Device instance names may carry QSPICE glyph bytes (e.g. R<0xB4>F2 in
        # the AudioAmp example): sanitize the NAME token only and continue
        # processing the line normally. (0xC3/0xA5 builtins handled above.)
        if ($stripped !~ /^[.*]/) {
            my ($name) = $stripped =~ /^(\S+)/;
            if (defined $name && $name =~ /[\x80-\xff]/) {
                (my $clean = $name) =~ s/[\x80-\xff]/_/g;
                $stripped =~ s/^\Q$name\E/$clean/;
                $upper = uc($stripped);
                push @changes, "L$lineno: device name glyph byte(s) -> _ ($clean)";
            }
        }

        # Inline-parameter S switch (QSPICE/LTspice style): Xyce needs a .model.
        #   S1 a b c d Ron=10 Roff=1G Vt=2 Vh=-1  ->  S1 a b c d QTZ_SW_S1
        # Vh<0 means smooth (non-hysteretic): use a 1V transition window.
        if ($stripped =~ /^([Ss]\S*)\s+(\S+\s+\S+\s+\S+\s+\S+)\s+(.*\bVt=.*)$/i) {
            my ($sname, $snodes, $sparams) = ($1, $2, $3);
            my %sp;
            $sp{ uc $1 } = $2 while $sparams =~ /(\w+)=(\S+)/g;
            my $vt = $sp{VT} // 0;
            my $vh = ($sp{VH} // 0) > 0 ? $sp{VH} : 0.5;
            my $von  = $vt + $vh;
            my $voff = $vt - $vh;
            my $mname = "QTZ_SW_\U$sname";
            push @appendix_models,
                ".model $mname VSWITCH(RON=" . ($sp{RON} // 1) .
                " ROFF=" . ($sp{ROFF} // '1G') . " VON=$von VOFF=$voff)\n";
            push @changes, "L$lineno: inline switch params -> .model $mname";
            push @output, "$sname $snodes $mname\n";
            next;
        }

        # C or L with Rpar=/Rser= (QSPICE loss shorthands Xyce lacks):
        # Rpar -> companion parallel resistor; Rser -> series resistor through
        # an internal node.
        if ($stripped =~ /^[CcLl]\S*\s/ && $stripped =~ /\bR(?:par|ser)=/i) {
            my @tok = split /\s+/, $stripped;
            my ($cn, $n1, $n2, $val) = @tok[0 .. 3];
            my ($rpar, $rser, @keep);
            for my $t (@tok[4 .. $#tok]) {
                if    ($t =~ /^Rpar=(\S+)$/i) { $rpar = $1; }
                elsif ($t =~ /^Rser=(\S+)$/i) { $rser = $1; }
                else                          { push @keep, $t; }
            }
            my $bot = $n2;
            if (defined $rser) {
                $bot = "qtzser_$cn";
                push @output, "Rqtzser_$cn $bot $n2 $rser\n";
                push @changes, "L$lineno: C Rser= -> series resistor";
            }
            push @output, "$cn $n1 $bot $val @keep\n";
            if (defined $rpar) {
                push @output, "Rqtzpar_$cn $n1 $n2 $rpar\n";
                push @changes, "L$lineno: C Rpar= -> companion resistor";
            }
            next;
        }

        # Resistor flagged SHORTED (QSPICE shorted-component marker): a wire.
        if ($stripped =~ /^([Rr]\S*\s+\S+\s+\S+\s+)\S+\s+SHORTED\s*$/i) {
            push @changes, "L$lineno: SHORTED resistor -> 1m";
            push @output, "${1}1m\n";
            next;
        }

        # Identifier bytes >= 0x80 in subckt names / X-instance refs (QSPICE
        # embeds symbol-name glyphs, e.g. 0x95 in "X1\x95NE555"): map to '_'.
        if ($stripped =~ /[\x80-\xff]/ && $stripped =~ /^(?:\.subckt\b|\.ends\b|[Xx]\S*\s)/i) {
            (my $c = $stripped) =~ s/[\x80-\xff]/_/g;
            push @changes, "L$lineno: non-ASCII identifier byte(s) -> _";
            push @output, "$c\n";
            next;
        }

        # Shared safety nets with the LTspice front-end
        if ($stripped =~ /\bSINE\s*\(/i) {
            (my $c = $stripped) =~ s/\bSINE\s*\(/SIN(/ig;
            push @changes, "L$lineno: SINE( -> SIN(";
            push @output, "$c\n";
            next;
        }
        push @output, "$stripped\n";
    }

    my @appendix;              # lines to splice in before the final .END
    my %vdmos_models;          # VDMOS model names synthesized as subckts

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
                if ($type =~ /^VDMOS$/i) {
                    # synthesize a behavioral subckt; M instances referencing
                    # this model are rewritten to X instances below
                    push @extracted, _vdmos_subckt($name, $params, \@warnings);
                    $vdmos_models{ uc $name } = 1;
                    push @changes, "synthesized VDMOS macromodel QTZ_VDMOS_\U$name\E from " . basename($lib);
                    next;
                }
                if ($params =~ /\blevel\s*=\s*(?:20\d\d)\b/i) {
                    push @warnings, "model $name ($type) uses a QSPICE proprietary level Xyce lacks -- needs a macromodel";
                    push @extracted, "* [qtz] UNSUPPORTED model skipped (needs macromodel): .model $name $type\n";
                    next;
                }
                my $clean = _sanitize_model_params($params);
                push @extracted, ".model $name $type$clean\n";
                push @changes, "extracted .model $name $type from " . basename($lib);
            }
            close $lfh;
        }
        push @appendix, "* [qtz] models extracted from QSPICE libraries:\n", @extracted
            if @extracted;

        # Rewrite M instances of synthesized VDMOS models to X instances of the
        # 3-terminal macromodel subckt (M<name> d g s b <model> -> X... d g s),
        # and retarget .PRINT items that referenced the M device.
        if (%vdmos_models) {
            my %renamed;
            for my $l (@output) {
                next unless $l =~ /^([Mm]\S*)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)$/;
                my ($mn, $nd, $ng, $ns, $nb, $mdl, $extra) = ($1, $2, $3, $4, $5, $6, $7);
                next unless $vdmos_models{ uc $mdl };
                push @warnings, "VDMOS $mn: bulk node $nb != source $ns (3-terminal macromodel ties them)"
                    if uc($nb) ne uc($ns);
                push @warnings, "VDMOS $mn: instance params dropped: $extra" if length $extra;
                push @changes, "$mn -> Xqtzvd_$mn (VDMOS macromodel QTZ_VDMOS_\U$mdl\E)";
                $renamed{ uc $mn } = "Xqtzvd_$mn";
                $l = "Xqtzvd_$mn $nd $ng $ns QTZ_VDMOS_\U$mdl\E\n";
            }
            for my $l (grep { /^\.PRINT\b/i } @output) {
                # Id(M1)/I(M1) -> drain current = current through the macromodel's Rdd
                $l =~ s/\bI[dD]?\(\s*(\w+)\s*\)/exists $renamed{uc $1} ? "I($renamed{uc $1}:RDD)" : $&/ge;
            }
        }
    }

    # Xyce requires a .param card for every .STEP-swept parameter; QSPICE
    # decks often carry only the .step line. Supply the missing cards.
    for my $sname (sort grep { !$param_cards{$_} } keys %stepped) {
        push @appendix, ".param $sname=$stepped{$sname}\n";
        push @changes, "added .param $sname=$stepped{$sname} (required by Xyce for .STEP)";
    }

    # A synthesized latch makes the DC operating point ambiguous (Xyce DCOP
    # fails on the bistability); start the transient UIC, matching the
    # power-on-from-zero semantics of QSPICE's flop IC=0 default.
    if ($need_builtin{QTZ_SRFLOP}) {
        for my $l (@output) {
            if ($l =~ /^\.TRAN\b/i && $l !~ /\bUIC\b/i) {
                chomp $l; $l .= " UIC\n";
                push @changes, "added UIC to .TRAN (latch present, DCOP ambiguous)";
            }
        }
    }

    push @appendix, @appendix_models;
    my %gen = (QTZ_RROPAMP => \&_rropamp_subckt, QTZ_HMITT => \&_hmitt_subckt,
               QTZ_OR2 => \&_or2_subckt, QTZ_SRFLOP => \&_srflop_subckt);
    push @appendix, $gen{$_}->() for sort grep { $gen{$_} } keys %need_builtin;

    if (@appendix) {
        # insert before the final .END (everything after .END is ignored)
        my ($end_idx) = grep { $output[$_] =~ /^\s*\.end\s*$/i } reverse 0 .. $#output;
        if (defined $end_idx) { splice @output, $end_idx, 0, @appendix; }
        else                  { push @output, @appendix; }
    }

    return (\@output, \@changes, \@warnings);
}

# Behavioral equivalent of QSPICE's built-in rail-to-rail ideal op-amp
# (RRopAmp): single dominant pole giving DC gain AVOL and unity-gain bandwidth
# GBW, output hard-clamped to the rails. Slew/Rload/Phi not yet modeled.
sub _rropamp_subckt {
    return <<'EOS';
* [qtz] QSPICE RRopAmp behavioral equivalent. Ikick is a one-shot startup
* perturbation (zero at DC, so the operating point is unaffected) that knocks
* oscillator loops off the unstable zero equilibrium, which QSPICE's builtin
* leaves by itself; steady state stays clamp/network-determined.
.SUBCKT QTZ_RROPAMP vdd vss out inn inp PARAMS: AVOL=100K GBW=5MEG
G1 0 mid inp inn 1m
R1 mid 0 {AVOL/1m}
C1 mid 0 {1m/(6.283185307179586*GBW)}
Ikick 0 mid PULSE(0 1n 0 1u 1u 100u 1e6)
* rails buffered through E sources: outer node names may contain +/- (QSPICE
* allows "V+"), which Xyce expressions misparse; positional refs are safe.
Evp rp 0 vdd 0 1
Evn rn 0 vss 0 1
* anti-windup: hold mid within ~0.1V of the rails so the output leaves
* saturation as soon as the input reverses (QSPICE's builtin recovers
* instantly; an unbounded integrator node would lag and drop the frequency)
Bwind mid 0 I={(V(mid)>V(rp)+0.1)*(V(mid)-V(rp)-0.1) + (V(mid)<V(rn)-0.1)*(V(mid)-V(rn)+0.1)}
B1 out 0 V={max(min(V(mid),V(rp)),V(rn))}
.ENDS QTZ_RROPAMP
EOS
}

# QSPICE HMITT builtin: comparator with rail-to-rail complementary outputs.
# (The 555's references are static dividers, so hysteresis is not modeled yet.)
sub _hmitt_subckt {
    return <<'EOS';
* [qtz] QSPICE HMITT comparator behavioral equivalent
.SUBCKT QTZ_HMITT vdd vss out outb in1 in2
Evp rp 0 vdd 0 1
Evn rn 0 vss 0 1
Rin1 in1 rn 1G
Rin2 in2 rn 1G
B1 o 0 V={V(rn)+(V(rp)-V(rn))*(0.5+0.5*tanh(V(in1,in2)/5m))}
Ro o out 10
Co out rn 10n
Bb ob 0 V={V(rp)+V(rn)-V(out)}
Rb ob outb 10
Cb outb rn 10n
.ENDS QTZ_HMITT
EOS
}

# QSPICE OR builtin: 2-input OR with complementary outputs. Logic threshold is
# mid-rail; an unconnected input reads low via the 1G pulldown (QSPICE uses a
# one-input OR as an inverter through the outbar pin).
sub _or2_subckt {
    return <<'EOS';
* [qtz] QSPICE OR gate behavioral equivalent
.SUBCKT QTZ_OR2 vdd vss out outb in1 in2
Evp rp 0 vdd 0 1
Evn rn 0 vss 0 1
Rin1 in1 rn 1G
Rin2 in2 rn 1G
B1 o 0 V={V(rn)+(V(rp)-V(rn))*max(0.5+0.5*tanh((V(in1)-0.5*(V(rp)+V(rn)))/(0.05*(V(rp)-V(rn)))),0.5+0.5*tanh((V(in2)-0.5*(V(rp)+V(rn)))/(0.05*(V(rp)-V(rn)))))}
Ro o out 10
Co out rn 10n
Bb ob 0 V={V(rp)+V(rn)-V(out)}
Rb ob outb 10
Cb outb rn 10n
.ENDS QTZ_OR2
EOS
}

# QSPICE SR-FLOP builtin: set/reset latch with complementary outputs. State is
# an ODE on Cst (charge toward the asserted rail at TRISE rate, hold when
# neither input is asserted); Rleak settles the never-set latch low, matching
# QSPICE's IC=0 default. UVLO/Ttol not modeled.
sub _srflop_subckt {
    return <<'EOS';
* [qtz] QSPICE SR-FLOP behavioral equivalent
.SUBCKT QTZ_SRFLOP vdd vss out outb s r PARAMS: TRISE=1U
Evp rp 0 vdd 0 1
Evn rn 0 vss 0 1
Rs s rn 1G
Rr r rn 1G
Cst st rn 1n
Rleak st rn 100G
Bst 0 st I={1n/TRISE*((0.5+0.5*tanh((V(s)-0.5*(V(rp)+V(rn)))/(0.05*(V(rp)-V(rn)))))*(V(rp)-V(st))-(0.5+0.5*tanh((V(r)-0.5*(V(rp)+V(rn)))/(0.05*(V(rp)-V(rn)))))*(V(st)-V(rn)))}
B1 o 0 V={V(rn)+(V(rp)-V(rn))*(0.5+0.5*tanh((V(st)-0.5*(V(rp)+V(rn)))/(0.02*(V(rp)-V(rn)))))}
Ro o out 10
Co out rn 10n
Bb ob 0 V={V(rp)+V(rn)-V(out)}
Rb ob outb 10
Cb outb rn 10n
.ENDS QTZ_SRFLOP
EOS
}

# Engineering-notation value -> plain number (for baking into expressions,
# where Xyce suffix parsing can't be relied on). Returns undef if not numeric.
sub _eng2num {
    my ($v) = @_;
    return undef unless defined $v;
    $v =~ s/\xc2\xb5/u/; $v =~ s/\xb5/u/;
    return $1 * 1 if $v =~ /^([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)$/i;
    if ($v =~ /^([-+]?(?:\d+\.?\d*|\.\d+))(meg|mil|[tgkmunpf])[a-z]*$/i) {
        my %m = (t=>1e12, g=>1e9, meg=>1e6, k=>1e3, mil=>25.4e-6,
                 m=>1e-3, u=>1e-6, n=>1e-9, p=>1e-12, f=>1e-15);
        return $1 * $m{ lc $2 };
    }
    return undef;
}

# Synthesize a behavioral subckt for an LTspice/QSPICE VDMOS power-MOSFET
# model (Xyce has no VDMOS device). DC equations follow the LTspice VDMOS
# model (same analytical form devchar.py validated against LTspice): smooth
# subthreshold via Kp*Ks^2*ln(1+exp(Vov/Ks))^2, Mtriode-shaped triode region,
# Lambda channel-length modulation, body diode with Rb/Cjo, fixed Cgs and
# Cgdmin (the nonlinear Cgd and QSPICE extensions RonX/eta/tempcos are not
# modeled -- DC characteristics first).
sub _vdmos_subckt {
    my ($name, $params, $warnings) = @_;
    my %p;
    $p{ lc $1 } = $2 while $params =~ /(\w+)\s*=\s*(\S+)/g;
    my $pchan = ($params =~ /\bpchan\b/i) ? 1 : 0;
    my $pol = $pchan ? -1 : 1;

    my $num = sub { my ($k, $d) = @_; my $n = _eng2num($p{$k}); defined $n ? $n : $d };
    my $vto = abs($num->('vto', 2));
    my $kp  = $num->('kp', 10);
    my $lam = $num->('lambda', 0);
    my $rd  = $num->('rd', 0) || 1e-6;
    my $rs  = $num->('rs', 0) || 1e-6;
    my $rg  = $num->('rg', 0) || 1e-6;
    # default triode exponent 2 = the standard Kp*(Vov*Vds - Vds^2/2) form;
    # verified against QSPICE64 gold (mtriode=1 halves the linear region)
    my $mt  = $num->('mtriode', 2);
    # QSPICE RonX (reverse-engineered against gold): sharpens the knee --
    # triode transconductance scales by RonX and saturation starts at
    # Vov/RonX, keeping the (verified-exact) sat current continuous:
    #   Vds < Vov/RonX:  I = RonX*Kp*(Vov*Vds - RonX*Vds^2/2)*(1+lambda*Vds)
    my $ronx = $num->('ronx', 1);
    my $ks  = $num->('ksubthres', 0.1);
    my $cgs = $num->('cgs', 0);
    my $cgdmin = $num->('cgdmin', 0);
    my $is  = $num->('is', 1e-14);
    my $nd  = $num->('n', 1);
    my $rb  = $num->('rb', 0);
    my $cjo = $num->('cjo', 0);

    push @$warnings, "VDMOS $name: QSPICE extension eta=$p{eta} not modeled"
        if defined $p{eta};

    my $U = uc $name;
    my $S = $pol > 0 ? '' : '-';                       # current sign
    my $vgs = $pol > 0 ? 'V(gi,si)' : 'V(si,gi)';      # polarity-folded senses
    my $vds = $pol > 0 ? 'V(di,si)' : 'V(si,di)';
    my $vov = sprintf '(%s-%.6g)', $vgs, $vto;
    my $kpks2 = sprintf '%.6g', $kp * $ks * $ks;
    my $sub_t = sprintf '%s*ln(1+exp(%s/%.6g))*ln(1+exp(%s/%.6g))', $kpks2, $vov, $ks, $vov, $ks;
    my $tri_t = sprintf '%.6g*(%s*%s-%.6g*pow(max(%s,0),%.6g)*pow(max(%s,0),%.6g))*(1+%.6g*%s)',
                        $kp * $ronx, $vov, $vds, 0.5 * $ronx, $vds, $mt, $vov, 2 - $mt, $lam, $vds;
    my $sat_t = sprintf '0.5*%.6g*%s*%s*(1+%.6g*%s)', $kp, $vov, $vov, $lam, $vds;
    my $ich = sprintf 'IF(%s<=0,%s,IF(%s<%s/%.6g,%s,%s))',
                      $vov, $sub_t, $vds, $vov, $ronx, $tri_t, $sat_t;

    my ($ba, $bk) = $pol > 0 ? ('si', 'di') : ('di', 'si');   # body diode anode/cathode
    my $bd_rs = $rb > 0 ? sprintf(' RS=%.6g', $rb) : '';
    my $bd_cj = $cjo > 0 ? sprintf(' CJO=%.6g', $cjo) : '';

    my $txt = "* [qtz] VDMOS $name synthesized macromodel (" . ($pchan ? 'P' : 'N') . "-channel)\n";
    $txt .= ".SUBCKT QTZ_VDMOS_$U d g s\n";
    $txt .= sprintf "Rdd d di %.6g\n", $rd;
    $txt .= sprintf "Rgg g gi %.6g\n", $rg;
    $txt .= sprintf "Rss si s %.6g\n", $rs;
    $txt .= "Bch di si I={$S($ich)}\n";
    $txt .= ".model QTZ_VDMOS_${U}_BD D(IS=" . sprintf('%.6g', $is) . " N=" . sprintf('%.6g', $nd) . "$bd_rs$bd_cj)\n";
    $txt .= "Dbd $ba $bk QTZ_VDMOS_${U}_BD\n";
    $txt .= sprintf "Cq_gs gi si %.6g\n", $cgs    if $cgs > 0;
    $txt .= sprintf "Cq_gd gi di %.6g\n", $cgdmin if $cgdmin > 0;
    $txt .= ".ENDS QTZ_VDMOS_$U\n";
    return $txt;
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
