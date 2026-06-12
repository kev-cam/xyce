#!/usr/bin/env perl
#
# ltspice2xyce.pl — LTspice netlist (.cir/.net) -> Xyce netlist translator.
#
# A streaming front-end in the same family as gnucap2xyce.pl and
# cadence2xyce.pl: read an LTspice-flavour SPICE netlist, translate the
# LTspice-isms Xyce doesn't accept, and emit a Xyce-runnable netlist.
# Recurses into .include/.inc/.lib targets (translating each), since Xyce
# resolves relative includes against cwd and would otherwise read the original
# LTspice files untranslated.
#
# Handled: .param space-form -> name=value, .meas -> commented out, .PROBE/.PLOT
# removal, .PLOT->.PRINT, empty diode models, .func ^->**, SINE->SIN, .tran
# single-arg/uic, British notation (4n7), Rser=, ';' comments, micro sign, etc.
#
# Usage:
#   ltspice2xyce.pl circuit.cir                 # -> stdout
#   ltspice2xyce.pl -o out.cir circuit.cir
#   ltspice2xyce.pl --lib /path/standard.lib circuit.cir   # splice a model lib
#
# This is the canonical LTspice->Xyce translator; `ltz` calls it (so ltz and any
# LTspice-replacement shim share one implementation).
#
use strict;
use warnings;
use File::Basename qw(dirname basename);
use File::Spec;
use Getopt::Long qw(:config no_ignore_case);

my ($opt_output, $opt_lib, $opt_no_lib, $opt_verbose, $opt_help);
GetOptions(
    'o|output=s' => \$opt_output,
    'lib=s'      => \$opt_lib,
    'no-lib'     => \$opt_no_lib,
    'v|verbose'  => \$opt_verbose,
    'h|help'     => \$opt_help,
) or usage();
usage() if $opt_help || !@ARGV;

my $input = shift @ARGV;
translate_file($input, $opt_output, ($opt_no_lib ? undef : $opt_lib), 0);
exit 0;

sub usage {
    print STDERR
        "usage: ltspice2xyce.pl [-o out.cir] [--lib FILE] [--no-lib] [-v] input.cir\n";
    exit 2;
}
sub verbose { print STDERR "ltspice2xyce: @_\n" if $opt_verbose }

# Translate one LTspice netlist -> Xyce, recursing into its includes.
sub translate_file {
    my ($inpath, $outpath, $lib, $is_include) = @_;
    verbose("translate $inpath" . ($is_include ? " (include)" : ""));

    open my $fh, '<:raw', $inpath or die "ltspice2xyce: cannot read $inpath: $!\n";
    local $/; my $blob = <$fh>;
    close $fh;
    $blob =~ s/\x00//g;                           # LTspice 26 UTF-16LE libs
    $blob =~ s/\r\n?/\n/g;                        # strip Windows CR
    my @lines = map { "$_\n" } split /\n/, $blob;

    my ($out_ref) = cir_to_xyce(\@lines, $lib);

    # Splice the model library into the top-level netlist only.
    if ($lib && !$is_include) {
        splice @$out_ref, 1, 0, ".INCLUDE $lib\n";
    }

    # Recursively translate each relative .include/.inc/.lib target into the
    # output dir and repoint the directive at the translated copy (abs path).
    my $indir  = dirname($inpath);
    my $outdir = (defined $outpath) ? dirname($outpath) : $indir;
    for my $ln (@$out_ref) {
        next unless $ln =~ /^(\s*\.(?:include|inc|lib)\s+)(\S+)(.*)$/i;
        my ($pre, $f, $post) = ($1, $2, $3);
        $f =~ s/^["']|["']$//g;
        next if $f =~ /[\\]/ || File::Spec->file_name_is_absolute($f);
        my $src = File::Spec->catfile($indir, $f);
        if (!-f $src) {
            # not beside the deck: LTspice resolves bare lib names against its
            # install (lib\sub for subckt libs) -- do the same locally
            $src = _lt_find_lib(basename($f));
            next unless defined $src && -f $src;
            verbose("  resolved $f -> $src");
        }
        my $tgt = File::Spec->catfile($outdir, basename($f) . ".xyce");
        translate_file($src, $tgt, undef, 1);
        $ln = "$pre\"$tgt\"$post\n";
    }

    if (defined $outpath) {
        my $od = dirname($outpath);
        if ($od && !-d $od) { require File::Path; File::Path::make_path($od); }
        open my $o, '>', $outpath or die "ltspice2xyce: cannot write $outpath: $!\n";
        print $o @$out_ref;
        close $o;
    } else {
        print @$out_ref;
    }
    return $outpath;
}

# ===========================================================================
#  Translation core — moved verbatim from ltz/bin/ltz (the canonical home).
# ===========================================================================

sub _ltspice_param_to_xyce {
    my ($args) = @_;
    my @out;
    while (1) {
        $args =~ s/^[\s,]+//;
        last unless length $args;
        my $name;
        if    ($args =~ s/^([A-Za-z_]\w*)\s*=\s*//) { $name = $1; }   # name= value
        elsif ($args =~ s/^([A-Za-z_]\w*)\s+//)     { $name = $1; }   # name  value
        elsif ($args =~ s/^([A-Za-z_]\w*)\s*$//)    { push @out, $1; last; }  # trailing lone name
        else { return undef; }
        my $val = _take_param_value(\$args);
        return undef unless defined $val && length $val;
        push @out, "$name=$val";
    }
    return @out ? join(' ', @out) : undef;
}

# Pull one value token off the front of $$ref: a balanced {…} expression or a
# whitespace-delimited bareword.
sub _take_param_value {
    my ($ref) = @_;
    $$ref =~ s/^\s+//;
    return undef unless length $$ref;
    if ($$ref =~ /^\{/) {
        my ($depth, $i) = (0, 0);
        while ($i < length $$ref) {
            my $c = substr($$ref, $i, 1);
            $depth++ if $c eq '{';
            $depth-- if $c eq '}';
            $i++;
            last if $depth == 0;
        }
        my $v = substr($$ref, 0, $i, '');
        return $v;
    }
    $$ref =~ s/^(\S+)//;
    return $1;
}


# Locate LTspice's component database lib by basename (standard.mos etc.).
# Works from Cygwin or WSL; $LTSPICE_CMP overrides.
sub _lt_find_cmp_lib {
    my ($base) = @_;
    my @cands = grep { defined && length }
        ($ENV{LTSPICE_CMP} ? "$ENV{LTSPICE_CMP}/$base" : undef),
        glob("/mnt/c/Users/*/AppData/Local/LTspice/lib/cmp/$base"),
        glob("/cygdrive/c/Users/*/AppData/Local/LTspice/lib/cmp/$base");
    for my $p (@cands) { return $p if -f $p; }
    return undef;
}

# Locate any LTspice install lib by basename: subckt libs (lib/sub --
# UniversalOpAmp2.lib, LTC.lib, TowTom2.sub...) then the cmp database.
sub _lt_find_lib {
    my ($base) = @_;
    my @cands = grep { defined && length }
        ($ENV{LTSPICE_SUB} ? "$ENV{LTSPICE_SUB}/$base" : undef),
        glob("/mnt/c/Users/*/AppData/Local/LTspice/lib/sub/$base"),
        glob("/cygdrive/c/Users/*/AppData/Local/LTspice/lib/sub/$base");
    for my $p (@cands) { return $p if -f $p; }
    return _lt_find_cmp_lib($base);
}

# Slurp an LTspice lib file. LTspice 26 ships these UTF-16LE without BOM;
# strip NULs (content is ASCII) and CRs, join + continuation lines.
sub _lt_slurp {
    my ($path) = @_;
    open my $fh, '<:raw', $path or return undef;
    local $/; my $t = <$fh>; close $fh;
    $t =~ s/\x00//g;
    $t =~ s/\r//g;
    $t =~ s/\n\+/ /g;          # fold continuations for one-line .model scan
    return $t;
}

# Strip LTspice component-picker doc params from a .model parameter list
# (mfg=, Vds=, Ron=, Qg=, Iave=, Vrev=, type= ...; bare or quoted) and µ -> u.
sub _lt_sanitize_model_params {
    my ($p) = @_;
    $p =~ s/\xc2\xb5/u/g;  $p =~ s/\xb5/u/g;
    $p =~ s/\s+(?:mfg|Vds|Ron|Qg|Iave|Vrev|type)\s*=\s*(?:"[^"]*"|\S+)//ig;
    $p =~ s/=\s+/=/g;
    return $p;
}

# Engineering-notation value -> plain number (for baking into expressions).
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

# Synthesize a behavioral subckt for an LTspice VDMOS model (Xyce has no
# VDMOS device). Same analytical form validated against QSPICE64 gold in
# qspice2xyce.pl (QSPICE inherited LTspice's VDMOS): smooth subthreshold
# Kp*Ks^2*ln(1+exp(Vov/Ks))^2, standard triode (default exponent 2 -- the
# QSPICE-gold finding; LTspice native gold via ltz will confirm), Lambda CLM,
# body diode, fixed Cgs/Cgdmin, Rd/Rs/Rg, pchan folding. The interim
# destination is PyMS Verilog-A (see qspice_va/, blocked on the class-'e'
# linkage); this keeps parity with the QSPICE path meanwhile.
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
    my $mt  = $num->('mtriode', 2);
    my $ks  = $num->('ksubthres', 0.1);
    my $cgs = $num->('cgs', 0);
    my $cgdmin = $num->('cgdmin', 0);
    my $is  = $num->('is', 1e-14);
    my $nd  = $num->('n', 1);
    my $rb  = $num->('rb', 0);
    my $cjo = $num->('cjo', 0);

    my $U = uc $name;
    my $S = $pol > 0 ? '' : '-';
    my $vgs = $pol > 0 ? 'V(gi,si)' : 'V(si,gi)';
    my $vds = $pol > 0 ? 'V(di,si)' : 'V(si,di)';
    my $vov = sprintf '(%s-%.6g)', $vgs, $vto;
    my $kpks2 = sprintf '%.6g', $kp * $ks * $ks;
    my $sub_t = sprintf '%s*ln(1+exp(%s/%.6g))*ln(1+exp(%s/%.6g))', $kpks2, $vov, $ks, $vov, $ks;
    my $tri_t = sprintf '%.6g*(%s*%s-0.5*pow(max(%s,0),%.6g)*pow(max(%s,0),%.6g))*(1+%.6g*%s)',
                        $kp, $vov, $vds, $vds, $mt, $vov, 2 - $mt, $lam, $vds;
    my $sat_t = sprintf '0.5*%.6g*%s*%s*(1+%.6g*%s)', $kp, $vov, $vov, $lam, $vds;
    my $ich = sprintf 'IF(%s<=0,%s,IF(%s<%s,%s,%s))', $vov, $sub_t, $vds, $vov, $tri_t, $sat_t;

    my ($ba, $bk) = $pol > 0 ? ('si', 'di') : ('di', 'si');
    my $bd_rs = $rb > 0 ? sprintf(' RS=%.6g', $rb) : '';
    my $bd_cj = $cjo > 0 ? sprintf(' CJO=%.6g', $cjo) : '';

    my $txt = "* [ltz] VDMOS $name synthesized macromodel (" . ($pchan ? 'P' : 'N') . "-channel)\n";
    $txt .= ".SUBCKT LTZ_VDMOS_$U d g s\n";
    $txt .= sprintf "Rdd d di %.6g\n", $rd;
    $txt .= sprintf "Rgg g gi %.6g\n", $rg;
    $txt .= sprintf "Rss si s %.6g\n", $rs;
    $txt .= "Bch di si I={$S($ich)}\n";
    $txt .= ".model LTZ_VDMOS_${U}_BD D(IS=" . sprintf('%.6g', $is) . " N=" . sprintf('%.6g', $nd) . "$bd_rs$bd_cj)\n";
    $txt .= "Dbd $ba $bk LTZ_VDMOS_${U}_BD\n";
    $txt .= sprintf "Cq_gs gi si %.6g\n", $cgs    if $cgs > 0;
    $txt .= sprintf "Cq_gd gi di %.6g\n", $cgdmin if $cgdmin > 0;
    $txt .= ".ENDS LTZ_VDMOS_$U\n";
    return $txt;
}

# Decide whether a .PRINT/.PLOT body's leading analysis-type token matches an
# analysis actually present in the deck. LTspice decks sometimes carry a
# `.PRINT TRAN` alongside an AC-only analysis (and vice-versa); Xyce aborts with
# "Analysis type X and print type Y are inconsistent". Returns true to keep the
# line, false to drop it (its type has no matching analysis). A .PRINT with no
# leading type token, or when no analyses were detected, is always kept.
sub _print_type_matches {
    my ($body, $analyses) = @_;
    return 1 unless %$analyses;
    if ($body =~ /^\s*(TRAN|AC|DC|NOISE|HB|SENS|TRANADJOINT)\b/i) {
        return 0 unless $analyses->{ uc $1 };
    }
    return 1;
}

sub cir_to_xyce {
    my ($lines_ref, $lib_path) = @_;
    my @output;
    my @changes;
    my @warnings;
    my $has_end   = 0;
    my $has_print = 0;
    my $analysis_type;
    my %analyses;              # set of analysis types present (TRAN/AC/DC/NOISE)
    my %diode_models_used;     # model names referenced by D devices
    my %models_defined;        # model names defined by .model statements
    my @extract_libs;          # LTspice cmp libs queued for model extraction
    my %vdmos_models;          # VDMOS model names synthesized as subckts

    # First pass: detect analysis type, track models
    for my $line (@$lines_ref) {
        my $u = uc($line);
        if    ($u =~ /^\s*\.TRAN\b/) { $analysis_type = 'TRAN' }
        elsif ($u =~ /^\s*\.AC\b/)   { $analysis_type = 'AC' }
        elsif ($u =~ /^\s*\.DC\b/)   { $analysis_type = 'DC' }
        elsif ($u =~ /^\s*\.OP\b/)   { $analysis_type = 'DC' }
        elsif ($u =~ /^\s*\.TF\b/)   { $analysis_type = 'DC' }

        # Set of analysis types present, to reconcile .PRINT/.PLOT types against.
        $analyses{TRAN}  = 1 if $u =~ /^\s*\.TRAN\b/;
        $analyses{AC}    = 1 if $u =~ /^\s*\.AC\b/;
        $analyses{DC}    = 1 if $u =~ /^\s*\.(?:DC|OP|TF)\b/;
        $analyses{NOISE} = 1 if $u =~ /^\s*\.NOISE\b/;

        # Track diode instance model references: D<name> <n+> <n-> <model>
        if ($line =~ /^\s*D\S*\s+\S+\s+\S+\s+(\S+)/i) {
            $diode_models_used{$1} = 1;
        }
        # Track .model definitions
        if ($line =~ /^\s*\.MODEL\s+(\S+)\s/i) {
            $models_defined{$1} = 1;
        }
    }

    # Second pass: convert
    my $lineno = 0;
    for my $line (@$lines_ref) {
        $lineno++;
        (my $stripped = $line) =~ s/^\s+|\s+$//g;
        my $upper = uc($stripped);

        # Preserve empty lines
        unless (length $stripped) {
            push @output, $line;
            next;
        }

        # micro sign -> u, up front: branches below (.tran parsing, C/L
        # expansion) emit lines directly and must see converted values
        if ($stripped =~ /\xb5|\xc2\xb5/) {
            $stripped =~ s/\xc2\xb5/u/g;
            $stripped =~ s/\xb5/u/g;
            $upper = uc($stripped);
            $line  = "$stripped\n";   # branches that emit $line see it too
            push @changes, "L$lineno: micro sign -> u";
        }

        # .PROBE removal
        if ($upper eq '.PROBE' || $upper =~ /^\.PROBE\s/) {
            push @changes, "L$lineno: Removed .PROBE";
            push @output, "* [ltz] removed: $stripped\n";
            next;
        }

        # Bare .PLOT removal
        if ($upper eq '.PLOT') {
            push @changes, "L$lineno: Removed bare .PLOT";
            push @output, "* [ltz] removed: $stripped\n";
            next;
        }

        # .BACKANNO removal
        if ($upper =~ /^\.BACKANNO/) {
            push @changes, "L$lineno: Removed .BACKANNO";
            push @output, "* [ltz] removed: $stripped\n";
            next;
        }

        # .LIB with Windows paths. LTspice's own component database
        # (lib\cmp\standard.*) resolves locally when LTspice is installed:
        # queue it for referenced-model extraction (VDMOS models become
        # synthesized subckts; plain models are copied through). Anything
        # else Windows-pathed is dropped as before.
        if ($upper =~ /^\.LIB\b/ && $stripped =~ /\\/) {
            if ($stripped =~ /\\lib\\cmp\\(standard\.\w+)\s*$/i
                    && (my $local = _lt_find_cmp_lib($1))) {
                push @changes, "L$lineno: LTspice cmp lib -> deferred model extraction ($1)";
                push @output, "* [ltz] models extracted from: $1\n";
                push @extract_libs, $local;
                next;
            }
            push @warnings, "L$lineno: Windows .lib path removed: $stripped";
            push @output, "* [ltz] removed Windows path: $stripped\n";
            next;
        }


        # .LIB <file>  (sectionless whole-file include): LTspice/PSpice mean
        # "include the file", but Xyce reads a one-arg .LIB <file> as a sectioned
        # library and aborts ("Could not find .ENDL ... '.INC' was intended").
        # Rewrite to .INCLUDE so the recursive include-translator picks it up.
        # Keep ".LIB <section>" (section open, paired with .ENDL) and the
        # sectioned ".LIB <file> <section>" form untouched.
        if ($upper =~ /^\.LIB\b/) {
            my @t = split /\s+/, $stripped;     # .LIB  arg  [section]
            shift @t;
            if (@t == 1 && $t[0] =~ m{[./]} ) {  # single arg that looks like a file
                (my $file = $t[0]) =~ s/^["']|["']$//g;
                push @changes, "L$lineno: .LIB $file -> .INCLUDE (sectionless file)";
                push @output, ".INCLUDE $file\n";
                next;
            }
            push @output, $line;                 # section open / sectioned include
            next;
        }

        # .INCLUDE/.INC handling
        if ($upper =~ /^\.(?:INCLUDE|INC)\b/) {
            my ($inc_file) = $stripped =~ /^\S+\s+(.*)/;
            $inc_file //= '';
            if ($inc_file =~ /\\/) {
                push @warnings, "L$lineno: Windows .include path: $stripped";
                push @output, "* [ltz] removed Windows path: $stripped\n";
            } else {
                push @warnings, "L$lineno: .include dependency: $inc_file";
                push @output, $line;
            }
            next;
        }

        # Empty diode model -> add default params
        if ($stripped =~ /^\.model\s+(\S+)\s+D\s*$/i) {
            my $mname = $1;
            push @warnings, "L$lineno: Empty diode model '$mname' -- Xyce needs parameters";
            push @changes, "L$lineno: Added default diode params to .model $mname";
            push @output, ".model $mname D(IS=2.52e-9 RS=0.568 N=1.752 BV=100 IBV=100u)\n";
            next;
        }

        # .param: LTspice accepts space-form ".param name value [name value...]";
        # Xyce requires "name=value". Normalize (brace-expression aware).
        if ($stripped =~ /^\.param\s+(.+)/i) {
            my $args = $1;
            my $comment = ($args =~ s/\s*;(.*)$//) ? " ;$1" : '';   # strip inline ; comment
            my $conv = _ltspice_param_to_xyce($args);
            if (defined $conv) {
                push @changes, "L$lineno: .param -> name=value form" if $conv ne $args;
                push @output, ".param $conv$comment\n";
                next;
            }
            push @output, $line;   # unparseable — leave as-is
            next;
        }

        # .PLOT with args -> .PRINT
        if ($stripped =~ /^\.PLOT\s+(.+)/i) {
            my $args = $1;
            if (!_print_type_matches($args, \%analyses)) {
                push @changes, "L$lineno: dropped .PLOT (print type has no matching analysis)";
                push @output, "* [ltz] dropped (no matching analysis): $stripped\n";
                next;
            }
            push @changes, "L$lineno: .PLOT -> .PRINT";
            push @output, ".PRINT $args\n";
            $has_print = 1;
            next;
        }

        # .meas: LTspice .meas syntax (WHEN mag()=, FIND..AT, PARAM{}) doesn't
        # map cleanly onto Xyce .MEASURE and aborts the run. These are
        # post-processing measurements, not needed for the waveform .raw, so
        # comment them out — the simulation proceeds and the .raw is produced.
        # (TODO: a real .meas->.MEASURE translator for the cases Xyce supports.)
        if ($upper =~ /^\.MEAS[\s\t]/) {
            push @changes, "L$lineno: .meas commented out (Xyce .MEASURE incompatible)";
            push @output, "* [ltz] .meas (not translated): $stripped\n";
            next;
        }

        # .func: ^ -> **
        if ($upper =~ /^\.FUNC/) {
            if ($stripped =~ /\^/) {
                (my $converted = $stripped) =~ s/\^/**/g;
                push @changes, "L$lineno: Replaced ^ with ** in .func";
                push @output, "$converted\n";
            } else {
                push @output, $line;
            }
            next;
        }

        # .tran: LTspice allows single-arg ".tran <Tstop>", Xyce needs ".TRAN <Tstep> <Tstop>"
        # Also handles .tran 0 <tstop> ... where step=0 means "auto" in LTspice
        # Also handles trailing "uic" keyword and "startup" keyword
        if ($upper =~ /^\.TRAN\s/) {
            # Strip trailing UIC/startup keywords (Xyce uses .IC separately)
            my $tran_line = $stripped;
            my $has_uic = ($tran_line =~ s/\s+(?:uic|startup)\s*$//i);
            my @fields = split /\s+/, $tran_line;
            if (@fields == 2) {
                # Single arg: .tran <Tstop> -> .TRAN <Tstop/1000> <Tstop>
                my $tstop = $fields[1];
                my $tstop_val = _parse_eng($tstop);
                if (defined $tstop_val && $tstop_val > 0) {
                    my $tstep = $tstop_val / 1000;
                    push @changes, "L$lineno: .tran $tstop -> .TRAN $tstep $tstop (added step size)";
                    push @output, ".TRAN $tstep $tstop\n";
                } else {
                    push @output, "$tran_line\n";
                }
            } elsif (@fields >= 3) {
                # Multi-arg: .tran <tstep> <tstop> [tstart] [tmax]
                # If tstep is 0, auto-calculate from tstop
                my $tstep_val = _parse_eng($fields[1]);
                my $tstop_val = _parse_eng($fields[2]);
                if (defined $tstep_val && $tstep_val == 0 && defined $tstop_val && $tstop_val > 0) {
                    my $new_tstep = $tstop_val / 1000;
                    my $new_line = ".TRAN $new_tstep $fields[2]";
                    push @changes, "L$lineno: .tran step=0 -> .TRAN step=$new_tstep";
                    push @output, "$new_line\n";
                } else {
                    push @output, "$tran_line\n";
                }
            } else {
                push @output, "$tran_line\n";
            }
            push @changes, "L$lineno: Removed UIC keyword" if $has_uic;
            next;
        }

        # .STEP: LTspice writes ".step [oct|dec|lin] param <name> <args>"; Xyce's
        # .STEP takes no 'param' keyword (".STEP [LIN|DEC|OCT] <name> <args>").
        # Strip the keyword so the parameter sweep is recognized.
        if ($upper =~ /^\.STEP\b/) {
            if ($stripped =~ /^\.step\s+(?:(oct|dec|lin)\s+)?param\s+(.+)/i) {
                my $mode = $1 ? uc($1) . ' ' : '';
                push @changes, "L$lineno: .step param -> .STEP (dropped LTspice 'param' keyword)";
                push @output, ".STEP $mode$2\n";
                next;
            }
            push @output, $line;
            next;
        }

        # C or L with LTspice loss shorthands Xyce's primitives lack:
        # Rser= -> series resistor through an internal node, Rpar= ->
        # parallel resistor, Cpar= -> parallel capacitor. Values may be
        # brace expressions ({R1/2}).
        if ($stripped =~ /^[CcLl]\S*\s/ && $stripped =~ /\bR(?:par|ser)=|\bCpar=/i) {
            my @tok = split /\s+/, $stripped;
            my ($dn, $n1, $n2, $val) = @tok[0 .. 3];
            my ($rpar, $rser, $cpar, @keep);
            for my $t (@tok[4 .. $#tok]) {
                if    ($t =~ /^Rpar=(\S+)$/i) { $rpar = $1; }
                elsif ($t =~ /^Rser=(\S+)$/i) { $rser = $1; }
                elsif ($t =~ /^Cpar=(\S+)$/i) { $cpar = $1; }
                else                          { push @keep, $t; }
            }
            my $bot = $n2;
            if (defined $rser) {
                $bot = "ltzser_$dn";
                push @output, "Rltzser_$dn $bot $n2 $rser\n";
                push @changes, "L$lineno: $dn Rser= -> series resistor";
            }
            push @output, "$dn $n1 $bot $val @keep\n";
            if (defined $rpar) {
                push @output, "Rltzpar_$dn $n1 $n2 $rpar\n";
                push @changes, "L$lineno: $dn Rpar= -> parallel resistor";
            }
            if (defined $cpar) {
                push @output, "Cltzpar_$dn $n1 $n2 $cpar\n";
                push @changes, "L$lineno: $dn Cpar= -> parallel capacitor";
            }
            next;
        }

        # .options: LTspice option names don't map onto Xyce's sectioned
        # .OPTIONS. Translate maxstep -> TIMEINT DELMAX; drop the rest.
        if ($upper =~ /^\.OPTIONS?\s/) {
            if ($stripped =~ /\bmaxstep\s*=\s*(\S+)/i) {
                push @changes, "L$lineno: .options maxstep -> .OPTIONS TIMEINT DELMAX";
                push @output, ".OPTIONS TIMEINT DELMAX=$1\n";
            } else {
                push @changes, "L$lineno: LTspice .options dropped";
                push @output, "* [ltz] dropped: $stripped\n";
            }
            next;
        }

        # Track .PRINT
        if ($upper =~ /^\.PRINT/) {
            my ($body) = $stripped =~ /^\.PRINT\s*(.*)/i;
            if (defined $body && !_print_type_matches($body, \%analyses)) {
                push @changes, "L$lineno: dropped .PRINT (print type has no matching analysis)";
                push @output, "* [ltz] dropped (no matching analysis): $stripped\n";
                next;
            }
            $has_print = 1;
            push @output, $line;
            next;
        }

        # .END <name>: a subcircuit terminator mis-written as ".END <subckt>"
        # (LTspice tolerates it). Xyce treats a bare .END as end-of-netlist and
        # would drop every following block, so rewrite to ".ENDS <subckt>".
        if ($upper =~ /^\.END\s+\S/) {
            my ($name) = $stripped =~ /^\.END\s+(.*)$/i;
            push @changes, "L$lineno: .END $name -> .ENDS $name (subckt terminator)";
            push @output, ".ENDS $name\n";
            next;
        }

        # Track .END
        if ($upper eq '.END') {
            $has_end = 1;
            push @output, $line;
            next;
        }

        # --- Non-exclusive pre-processing (modifies $stripped in-place) ---
        my $modified = 0;

        # Device instance names may carry glyph bytes -- LTspice 26 writes
        # autogenerated names like "M\xC2\xA7Q1" (UTF-8 section sign).
        # Sanitize the NAME token only; the rest of the line is untouched.
        if ($stripped !~ /^[.*;]/) {
            my ($iname) = $stripped =~ /^(\S+)/;
            if (defined $iname && $iname =~ /[\x80-\xff]/) {
                (my $clean = $iname) =~ s/\xc2[\x80-\xbf]/_/g;
                $clean =~ s/[\x80-\xff]/_/g;
                $stripped =~ s/^\Q$iname\E/$clean/;
                push @changes, "L$lineno: device name glyph byte(s) -> _ ($clean)";
                $modified = 1;
            }
        }

        # Strip Rser= inline parameter on source lines (LTspice-specific)
        if ($stripped =~ /^[VI]\S*\s/i && $stripped =~ /\bRser=/i) {
            $stripped =~ s/\s+Rser=\S+//ig;
            push @changes, "L$lineno: Stripped Rser= from source";
            $modified = 1;
        }

        # British notation: 4n7 -> 4.7n, 2k2 -> 2.2k, 1u5 -> 1.5u etc.
        if ($stripped =~ s/\b(\d+)([pnumkMG])(\d+)\b/$1.$3$2/g) {
            push @changes, "L$lineno: British notation -> decimal";
            $modified = 1;
        }

        # --- Exclusive transformations (with next) ---

        # ; line comments -> * comments
        if ($stripped =~ /^;(.*)/) {
            push @changes, "L$lineno: ; comment -> * comment";
            push @output, "* $1\n";
            next;
        }

        # SINE( -> SIN( (LTspice uses SINE, standard SPICE/Xyce uses SIN)
        if ($stripped =~ /\bSINE\s*\(/i) {
            (my $converted = $stripped) =~ s/\bSINE\s*\(/SIN(/ig;
            push @changes, "L$lineno: SINE( -> SIN(";
            push @output, "$converted\n";
            next;
        }

        # (micro sign handled in top-of-loop preprocessing)

        # Pass-through
        push @output, $modified ? "$stripped\n" : $line;
    }

    # Post-processing
    if (!$has_print && $analysis_type) {
        push @warnings,
            "No .PRINT statement -- Xyce needs explicit output specification";
    }
    # Referenced-model extraction from LTspice's component database libs
    # (lib\cmp\standard.*, queued above). VDMOS models become synthesized
    # behavioral subckts (Xyce has no VDMOS device) and their M instances are
    # rewritten to X instances; other model types are copied through with
    # LTspice doc-params stripped.
    if (@extract_libs) {
        my $deck_text = join '', @output;
        my (@extracted, %seen);
        for my $lib (@extract_libs) {
            my $text = _lt_slurp($lib);
            unless (defined $text) {
                push @warnings, "cannot read LTspice lib $lib";
                next;
            }
            for my $ml (split /\n/, $text) {
                next unless $ml =~ /^\s*\.model\s+(\S+)\s+(\w+)\s*\(?(.*?)\)?\s*$/i;
                my ($name, $type, $params) = ($1, $2, $3);
                next if $seen{ uc $name } || $models_defined{ $name } || $models_defined{ uc $name };
                next unless $deck_text =~ /\b\Q$name\E\b/i;
                $seen{ uc $name } = 1;
                if ($type =~ /^VDMOS$/i) {
                    push @extracted, _vdmos_subckt($name, $params, \@warnings);
                    $vdmos_models{ uc $name } = 1;
                    push @changes, "synthesized VDMOS macromodel LTZ_VDMOS_\U$name\E";
                    next;
                }
                my $clean = _lt_sanitize_model_params($params);
                push @extracted, ".model $name $type($clean)\n";
                push @changes, "extracted .model $name $type from LTspice cmp lib";
            }
        }
        if (@extracted) {
            # insert before the final .END (everything after .END is ignored)
            my ($end_idx) = grep { $output[$_] =~ /^\s*\.end\s*$/i } reverse 0 .. $#output;
            my @blk = ("* [ltz] models extracted from LTspice component libs:\n", @extracted);
            if (defined $end_idx) { splice @output, $end_idx, 0, @blk; }
            else                  { push @output, @blk; }
        }

        # M instances of VDMOS models -> X instances of the 3-terminal subckt
        if (%vdmos_models) {
            my %renamed;
            for my $l (@output) {
                next unless $l =~ /^([Mm]\S*)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*(.*)$/;
                my ($mn, $nd, $ng, $ns, $nb, $mdl, $extra) = ($1, $2, $3, $4, $5, $6, $7);
                next unless $vdmos_models{ uc $mdl };
                push @warnings, "VDMOS $mn: bulk node $nb != source $ns (3-terminal macromodel ties them)"
                    if uc($nb) ne uc($ns);
                push @warnings, "VDMOS $mn: instance params dropped: $extra" if length $extra;
                push @changes, "$mn -> Xltzvd_$mn (VDMOS macromodel LTZ_VDMOS_\U$mdl\E)";
                $renamed{ uc $mn } = "Xltzvd_$mn";
                $l = "Xltzvd_$mn $nd $ng $ns LTZ_VDMOS_\U$mdl\E\n";
            }
            for my $l (grep { /^\.PRINT\b/i } @output) {
                $l =~ s/\bI[dD]?\(\s*(\w+)\s*\)/exists $renamed{uc $1} ? "I($renamed{uc $1}:RDD)" : $&/ge;
            }
        }
    }

    # Add generic .model for diodes with no model definition
    # Skip if bundled library is being included (it provides standard models)
    my @missing_models;
    if (!$lib_path) {
        for my $mname (sort keys %diode_models_used) {
            next if $models_defined{$mname};
            next if $mname =~ /^D$/i;
            next if $mname =~ /^\{/;
            push @missing_models, $mname;
        }
    }
    if (@missing_models) {
        # Insert before .END
        my @pre_end;
        my $end_line;
        if ($has_end) {
            $end_line = pop @output;  # remove .END temporarily
        }
        for my $mname (@missing_models) {
            push @pre_end, "* [ltz] auto-generated model for $mname\n";
            push @pre_end, ".model $mname D(IS=2.52e-9 RS=0.568 N=1.752 BV=100 IBV=100u)\n";
            push @changes, "Added generic .model for diode $mname";
        }
        push @output, @pre_end;
        push @output, $end_line if defined $end_line;
    }

    if (!$has_end) {
        push @output, ".END\n";
        push @changes, "Added missing .END";
    }

    # Keep only the LAST .END: LTspice files sometimes concatenate several
    # circuit blocks, each terminated by its own .END. Xyce treats the first
    # .END as end-of-netlist and errors on the trailing blocks, so comment out
    # every .END but the final one.
    {
        my @ends = grep { $output[$_] =~ /^\s*\.END\s*$/i } 0 .. $#output;
        if (@ends > 1) {
            $output[$_] = "* [ltz] intermediate .END removed\n" for @ends[0 .. $#ends - 1];
            push @changes, "Removed " . (@ends - 1) . " intermediate .END statement(s)";
        }
    }

    return (\@output, \@changes, \@warnings);
}

# ===========================================================================
#  .asc -> netlist conversion  (LTspice schematic parser)
# ===========================================================================

# (Symbol pin table is at top of file with globals)


sub _parse_eng {
    my ($s) = @_;
    my %suffix = (
        t => 1e12, g => 1e9, meg => 1e6, k => 1e3,
        m => 1e-3, u => 1e-6, n => 1e-9, p => 1e-12, f => 1e-15,
        mil => 25.4e-6,
    );
    if ($s =~ /^([+-]?[\d.]+(?:e[+-]?\d+)?)\s*(meg|mil|[tgkmunpf])?(?:[a-z]*)?$/i) {
        my ($num, $suf) = ($1, $2);
        return $num unless defined $suf;
        my $mult = $suffix{lc $suf};
        return defined $mult ? $num * $mult : $num;
    }
    # Plain number
    return $s if $s =~ /^[+-]?[\d.]+(?:e[+-]?\d+)?$/i;
    return undef;
}

