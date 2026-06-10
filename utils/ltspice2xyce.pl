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

    open my $fh, '<', $inpath or die "ltspice2xyce: cannot read $inpath: $!\n";
    my @lines = <$fh>;
    close $fh;
    s/\r\n?/\n/ for @lines;                       # strip Windows CR

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
        next unless -f $src;
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


sub cir_to_xyce {
    my ($lines_ref, $lib_path) = @_;
    my @output;
    my @changes;
    my @warnings;
    my $has_end   = 0;
    my $has_print = 0;
    my $analysis_type;
    my %diode_models_used;     # model names referenced by D devices
    my %models_defined;        # model names defined by .model statements

    # First pass: detect analysis type, track models
    for my $line (@$lines_ref) {
        my $u = uc($line);
        if    ($u =~ /^\s*\.TRAN\b/) { $analysis_type = 'TRAN' }
        elsif ($u =~ /^\s*\.AC\b/)   { $analysis_type = 'AC' }
        elsif ($u =~ /^\s*\.DC\b/)   { $analysis_type = 'DC' }
        elsif ($u =~ /^\s*\.OP\b/)   { $analysis_type = 'DC' }
        elsif ($u =~ /^\s*\.TF\b/)   { $analysis_type = 'DC' }

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

        # .LIB with Windows paths
        if ($upper =~ /^\.LIB\b/ && $stripped =~ /\\/) {
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

        # Track .PRINT
        if ($upper =~ /^\.PRINT/) {
            $has_print = 1;
            push @output, $line;
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

        # µ (micro sign) -> u (ASCII) for engineering notation
        # Handle both Latin-1 (0xB5) and UTF-8 (0xC2 0xB5) encodings
        if ($stripped =~ /\xb5|\xc2\xb5/) {
            (my $converted = $stripped) =~ s/\xc2\xb5/u/g;
            $converted =~ s/\xb5/u/g;
            push @changes, "L$lineno: µ -> u";
            push @output, "$converted\n";
            next;
        }

        # Pass-through
        push @output, $modified ? "$stripped\n" : $line;
    }

    # Post-processing
    if (!$has_print && $analysis_type) {
        push @warnings,
            "No .PRINT statement -- Xyce needs explicit output specification";
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

