#!/usr/bin/env perl
#
# cadence2xyce.pl — Cadence Spectre/AMS netlist preprocessor for Xyce
#
# Translates Cadence-dialect SPICE (.sp) to Xyce-compatible format:
#   - .hdl "file.va" is preserved (Xyce PyMS handles it)
#   - X instances referencing .hdl models → Y instances
#   - dc=val → dc val
#   - .option post/ingold → stripped
#   - .probe → stripped or converted to .PRINT
#   - par'expr' → {expr}
#
# Usage:
#   cadence2xyce.pl input.sp > output.cir
#   cadence2xyce.pl input.sp -o output.cir
#   cadence2xyce.pl --inplace input.sp    # modifies in place
#

use strict;
use warnings;
use File::Basename qw(dirname basename);
use Getopt::Long;

my $opt_output  = '';
my $opt_inplace = 0;
my $opt_verbose = 0;
GetOptions(
    'o=s'     => \$opt_output,
    'inplace' => \$opt_inplace,
    'v'       => \$opt_verbose,
) or die "Usage: $0 [options] input.sp\n";

my $input = $ARGV[0] or die "Usage: $0 [options] input.sp\n";
my $input_dir = dirname($input);

# ---------------------------------------------------------------------------
# Pass 1: Scan for .hdl modules and .model mappings
# ---------------------------------------------------------------------------

my %hdl_modules;   # module_name => { ports => N, va_file => path }
my %model_to_type; # model_name => module_type (from .model lines)

open my $fh, '<', $input or die "Cannot read $input: $!\n";
my @lines = <$fh>;
close $fh;

# Join continuation lines
my @joined;
my $buf = '';
for my $line (@lines) {
    $line =~ s/\r//g;
    if ($line =~ /^\+/) {
        $buf .= ' ' . substr($line, 1);
    } else {
        push @joined, $buf if length $buf;
        $buf = $line;
    }
}
push @joined, $buf if length $buf;

for my $line (@joined) {
    chomp(my $l = $line);

    # .include "file" — scan included files for .model declarations
    if ($l =~ /^\s*\.inc(?:lude)?\s+"?([^"\s]+)"?/i) {
        my $inc = $1;
        my $inc_path = (-f $inc) ? $inc : "$input_dir/$inc";
        if (-f $inc_path) {
            open my $ifh, '<', $inc_path or next;
            while (<$ifh>) {
                s/\r//g;
                if (/^\s*\.model\s+(\S+)\s+(\S+)/i) {
                    $model_to_type{lc $1} = lc $2;
                    verbose("  .model $1 → $2 (from $inc)");
                }
            }
            close $ifh;
        }
    }

    # .hdl "file.va" — scan the VA file for module declarations
    if ($l =~ /^\s*\.hdl\s+"?([^"]+)"?/i) {
        my $va_file = $1;
        # Resolve relative to input directory
        my $va_path = $va_file;
        unless (-f $va_path) {
            $va_path = "$input_dir/$va_file";
        }
        # Also check parent directories (code/ etc.)
        unless (-f $va_path) {
            for my $try ("$input_dir/../code/$va_file",
                         "$input_dir/../../code/$va_file",
                         "$input_dir/../$va_file") {
                if (-f $try) { $va_path = $try; last; }
            }
        }
        if (-f $va_path) {
            my @mods = scan_va_modules($va_path);
            for my $m (@mods) {
                $hdl_modules{lc $m->{name}} = $m;
                verbose("  .hdl module: $m->{name} ($m->{ports} ports) from $va_path");
            }
        } else {
            warn "cadence2xyce: .hdl file not found: $va_file\n";
        }
    }

    # .model <name> <type> ... — map model name to device type
    if ($l =~ /^\s*\.model\s+(\S+)\s+(\S+)/i) {
        my ($mname, $mtype) = (lc $1, lc $2);
        $model_to_type{$mname} = $mtype;
        verbose("  .model $mname → $mtype");
    }
}

# ---------------------------------------------------------------------------
# Pass 2: Transform
# ---------------------------------------------------------------------------

my @output;

for my $line (@lines) {
    chomp(my $l = $line);
    $l =~ s/\r//g;

    # .option: strip unsupported Cadence options
    if ($l =~ /^\s*\.option/i) {
        $l =~ s/\b(?:post|ingold)\b//gi;
        $l =~ s/\babstol=\S+//gi;  # Xyce uses .OPTIONS NONLIN ABSTOL=
        $l =~ s/\breltol=\S+//gi;
        $l =~ s/^\s*\.option\s*$//i;  # remove if empty
        if ($l =~ /^\s*$/) {
            push @output, "* [cadence2xyce] removed empty .option\n";
            next;
        }
    }

    # .probe → comment out (Xyce uses .PRINT)
    if ($l =~ /^\s*\.probe\b/i) {
        push @output, "* [cadence2xyce] $l\n";
        next;
    }

    # dc=val → dc val (Cadence voltage source syntax)
    $l =~ s/\bdc\s*=\s*(\S+)/dc $1/gi;
    $l =~ s/\bac\s*=\s*(\S+)/ac $1/gi;

    # par'expr' → {expr} (Cadence expression syntax)
    $l =~ s/par'([^']+)'/{$1}/g;

    # .model name moduletype → .model name xyce_type level=N
    # when moduletype matches a .hdl-registered module with ADMS attributes
    if ($l =~ /^\s*\.model\s+(\S+)\s+(\S+)(.*)/i) {
        my ($mname, $mtype, $rest) = ($1, $2, $3);
        my $mod = $hdl_modules{lc $mtype};
        if ($mod && $mod->{level}) {
            my $group = uc($mod->{group} // '');
            my $xyce_type;
            if    ($group eq 'MOSFET') { $xyce_type = ($rest =~ /TYPE\s*=\s*0/i) ? 'pmos' : 'nmos' }
            elsif ($group eq 'DIODE')  { $xyce_type = 'd' }
            elsif ($group eq 'BJT')    { $xyce_type = ($rest =~ /TYPE\s*=\s*0/i) ? 'pnp' : 'npn' }
            else                       { $xyce_type = lc $mtype }
            $l = ".model $mname $xyce_type level=$mod->{level}$rest";
            verbose("  .model $mname: $mtype → $xyce_type level=$mod->{level}");
        }
    }

    # .include — also preprocess included files (write modified copy)
    if ($l =~ /^\s*\.inc(?:lude)?\s+"?([^"\s]+)"?/i) {
        my $inc = $1;
        my $inc_path = (-f "$input_dir/$inc") ? "$input_dir/$inc" : $inc;
        if (-f $inc_path) {
            # Process the included file in-place for model card conversion
            my $modified = 0;
            open my $ifh, '<', $inc_path or next;
            my @inc_lines = <$ifh>;
            close $ifh;
            for my $il (@inc_lines) {
                $il =~ s/\r//g;
                if ($il =~ /^\s*\.model\s+(\S+)\s+(\S+)(.*)/i) {
                    my ($mn, $mt, $r) = ($1, $2, $3);
                    my $imod = $hdl_modules{lc $mt};
                    if ($imod && $imod->{level}) {
                        my $ig = uc($imod->{group} // '');
                        my $xt;
                        if    ($ig eq 'MOSFET') { $xt = ($r =~ /TYPE\s*=\s*0/i) ? 'pmos' : 'nmos' }
                        elsif ($ig eq 'DIODE')  { $xt = 'd' }
                        elsif ($ig eq 'BJT')    { $xt = ($r =~ /TYPE\s*=\s*0/i) ? 'pnp' : 'npn' }
                        else                    { $xt = lc $mt }
                        $il = ".model $mn $xt level=$imod->{level}$r\n";
                        verbose("  $inc: .model $mn: $mt → $xt level=$imod->{level}");
                        $modified = 1;
                    }
                }
            }
            if ($modified) {
                open my $ofh, '>', $inc_path or warn "Cannot write $inc_path: $!\n";
                print $ofh @inc_lines;
                close $ofh;
            }
        }
    }

    # X instances → Y instances when model maps to .hdl module
    if ($l =~ /^\s*(X\S*)\s+(.+)/) {
        my $inst_name = $1;
        my $rest = $2;
        # First collapse "key = val" → "key=val" for param detection
        $rest =~ s/(\w+)\s*=\s*(\S+)/$1=$2/g;
        # Parse: X1 node1 node2 ... modelname [param=val ...]
        my @tokens = split /\s+/, $rest;
        my $model_name = '';
        my @nodes;
        my @params;
        my $in_params = 0;
        for my $tok (@tokens) {
            if ($tok =~ /=/) {
                $in_params = 1;
                push @params, $tok;
            } elsif ($in_params) {
                push @params, $tok;
            } else {
                push @nodes, $tok;
            }
        }
        # Last node is actually the model name
        $model_name = pop @nodes if @nodes;

        my $mtype = $model_to_type{lc $model_name} // '';
        my $mod = $hdl_modules{$mtype};

        if ($mod) {
            my $prefix = $mod->{prefix};
            my $iname = $inst_name;
            $iname =~ s/^X//i;  # strip X prefix
            my $node_str = join(' ', @nodes);
            my $param_str = @params ? ' ' . join(' ', @params) : '';
            if ($prefix eq 'Y') {
                # Generic Y device: YMODNAME name nodes model params
                my $ytype = uc $mod->{name};
                $l = "Y${ytype} ${iname} ${node_str} ${model_name}${param_str}";
            } else {
                # Standard SPICE device: Mname nodes model params
                $l = "${prefix}${iname} ${node_str} ${model_name}${param_str}";
            }
            verbose("  Rewrite: $inst_name → ${prefix}${iname}");
        }
    }

    # Ensure .PRINT exists (add if only .probe was present)
    # (handled in post-processing below)

    push @output, "$l\n";
}

# Post-processing: ensure .PRINT exists
my $has_print = grep { /^\s*\.print\b/i } @output;
unless ($has_print) {
    my $analysis = '';
    for (@output) {
        if (/^\s*\.(tran|ac|dc|op)\b/i) {
            $analysis = uc $1;
            last;
        }
    }
    if ($analysis) {
        # Insert before .end
        my @final;
        for my $line (@output) {
            if ($line =~ /^\s*\.end\s*$/i) {
                push @final, ".PRINT $analysis V(*) I(*)\n";
            }
            push @final, $line;
        }
        @output = @final;
    }
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

if ($opt_inplace) {
    open my $ofh, '>', $input or die "Cannot write $input: $!\n";
    print $ofh @output;
    close $ofh;
} elsif ($opt_output) {
    open my $ofh, '>', $opt_output or die "Cannot write $opt_output: $!\n";
    print $ofh @output;
    close $ofh;
} else {
    print @output;
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

sub scan_va_modules {
    my ($va_path) = @_;
    my @modules;
    open my $vfh, '<', $va_path or return ();
    my $text = do { local $/; <$vfh> };
    close $vfh;
    $text =~ s/\r//g;

    # Find module declarations with optional (* attributes *)
    while ($text =~ /\bmodule\s+(\w+)\s*\(([^)]*)\)\s*(?:\(\*([^*]*)\*\))?\s*;/g) {
        my $name = $1;
        my $ports_str = $2;
        my $attrs_str = $3 // '';
        my @ports = grep { /\S/ } split /\s*,\s*/, $ports_str;

        # Parse attributes
        my %attrs;
        while ($attrs_str =~ /(\w+)\s*=\s*"([^"]*)"/g) {
            $attrs{$1} = $2;
        }

        # Determine SPICE prefix from xyceModelGroup
        my $group = $attrs{xyceModelGroup} // '';
        my $level = $attrs{xyceLevelNumber} // '';
        my $prefix = 'Y';  # default: generic Y device
        if    (uc($group) eq 'MOSFET') { $prefix = 'M' }
        elsif (uc($group) eq 'DIODE')  { $prefix = 'D' }
        elsif (uc($group) eq 'BJT')    { $prefix = 'Q' }

        push @modules, {
            name   => $name,
            ports  => scalar @ports,
            prefix => $prefix,
            group  => $group,
            level  => $level,
        };
    }
    return @modules;
}

sub verbose {
    return unless $opt_verbose;
    print STDERR "cadence2xyce: $_[0]\n";
}
