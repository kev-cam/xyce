#!/usr/bin/env perl
#
# gnucap2xyce.pl — gnucap testbench-Verilog-A → Xyce netlist converter
#
# The IHP-Open-PDK gnucap-stats branch ships compact-model regression
# tests written for gnucap's testbench-style Verilog-A: each ``<name>.gc``
# driver file ``include``s a ``<name>.va`` testbench module which
# instantiates parametrised device subcircuits via SystemVerilog
# ``#(.p(v))`` syntax. None of that is Xyce-flavour SPICE — port them
# so the same regression suite can target Xyce/PyMS.
#
# Mapping summary:
#   gnucap                                       Xyce
#   ─────────────────────────────────────────    ──────────────────────────────
#   load <path>.so                                (stripped — PyMS auto-loads)
#   verilog                                       (stripped — top-of-file marker)
#   include <path>.va                            .HDL "<path>"  (compact models)
#   include <path>.va                            .INCLUDE "<path>"  (otherwise)
#   options itl6=250                              .OPTIONS NONLIN MAXSTEP=250
#   ground gnd;                                   (gnd → 0 substituted)
#   module name();                                top-level (no encapsulation)
#   module name(p1,p2,...);                       .SUBCKT name p1 p2 ...
#   parameter real X = Y;                         .PARAM X=Y  (top-level)
#   parameter real X = Y from [LO:HI];            .PARAM X=Y  (range dropped)
#   localparam real X = Y;                         (dropped — local to body)
#   vsource #(.dc(V)) Vname(n1,n2);              VVname n1 n2 DC V
#   vsine #(.mag(M),.dc(D)) Vname(n1,n2);        VVname n1 n2 DC D AC M  +  SIN(...)
#   idc #(.dc(I)) Iname(n1,n2);                  IIname n1 n2 DC I
#   resistor #(.r(R)) Rname(n1,n2);              RRname n1 n2 R
#   <mod> #(.p(v)...) <inst>(<nodes>);           X<inst> <nodes> <mod> p=v ...
#   print dc v(out) i(tb.vdd) ...                .PRINT DC V(out) I(Vvdd_src) ...
#   dc <var> <start> <stop> <step>                .DC <var> <start> <stop> <step>
#   options/status/tran/ac ...                    .OPTIONS / .TRAN / .AC
#
# Usage:
#   gnucap2xyce.pl tb_res_basic_typ.gc                  # writes tb_res_basic_typ.cir
#   gnucap2xyce.pl -o out.cir tb_res_basic_typ.gc
#   gnucap2xyce.pl tb_res_basic_typ.gc tb_res_basic.va  # both inputs explicit
#   gnucap2xyce.pl --inplace ...                        # rewrite in place
#

use strict;
use warnings;
use File::Basename qw(dirname basename);
use File::Spec;
use Getopt::Long;

my $opt_output  = '';
my $opt_inplace = 0;
my $opt_verbose = 0;
my $opt_keep_gnd = 0;
my $opt_paramset_lib = '';
my @opt_paramset_va;
my $opt_paramset_dir = '';
my @opt_corner_search;
GetOptions(
    'o=s'                => \$opt_output,
    'inplace'            => \$opt_inplace,
    'v|verbose'          => \$opt_verbose,
    'keep-gnd'           => \$opt_keep_gnd,
    'paramset-library=s' => \$opt_paramset_lib,
    'paramset-va=s'      => \@opt_paramset_va,
    'paramset-dir=s'     => \$opt_paramset_dir,
    'corner-search=s'    => \@opt_corner_search,
) or usage();

# Load the optional paramset library produced by paramset2xyce.pl
# --library — legacy inline-expansion mode kept for back-compat. The
# new default path is pass-through: paramsets get emitted as
# per-paramset .va files that PyMS parses (it knows ``paramset`` syntax
# now and resolves bindings into the underlying module's parameter
# defaults). The .cir emits .HDL for each and .MODEL/device cards at
# the instance sites.
our $PARAMSETS = {};
if ($opt_paramset_lib) {
    my $rc = do $opt_paramset_lib;
    die "gnucap2xyce: can't load paramset library $opt_paramset_lib: $@$!\n"
        unless $rc;
    verbose(scalar(keys %$PARAMSETS) . " paramsets loaded from "
          . $opt_paramset_lib);
}

# Pass-through mode bookkeeping. ``%PARAMSET_TBL`` indexes paramsets
# parsed from any --paramset-va inputs OR auto-discovered from the
# testbench's ``load`` directives: { name => { ... } }. Filled by
# discover_paramsets().
our %PARAMSET_TBL;
# Per-conversion auto-assigned device-level numbers (paramsets need
# UNIQUE xyceLevelNumbers so PyMS auto-discovery keys (R,9101) ≠
# (R,9102)). Counter advances from 9100 to keep clear of compact-
# model levels (BSIM-CMG 107..111, PSP 102/103, BSIM-SOI 70.., etc.).
our $LEVEL_COUNTER = 9100;
# Built-in defaults for the IHP gnucap-stats catalog. Override via
# the underlying-aware ``%UNDERLYING_INFO`` below if a paramset
# wants a different mapping.
our %UNDERLYING_INFO = (
    sp_resistor  => { letter => 'R', model_type => 'R'    },
    sp_capacitor => { letter => 'C', model_type => 'C'    },
    r3_cmc       => { letter => 'R', model_type => 'R'    },
    PSP103VA     => { letter => 'M', model_type => 'NMOS' },
    PSP103TVA    => { letter => 'M', model_type => 'NMOS' },
    psp103va     => { letter => 'M', model_type => 'NMOS' },
    PSP103_VA    => { letter => 'M', model_type => 'NMOS' },
    BSIM6        => { letter => 'M', model_type => 'NMOS' },
    bsim6        => { letter => 'M', model_type => 'NMOS' },
    bsimcmg_108  => { letter => 'M', model_type => 'NMOS' },
    bsimcmg_110  => { letter => 'M', model_type => 'NMOS' },
    bsimcmg_111  => { letter => 'M', model_type => 'NMOS' },
    sg13_hv_nmos => { letter => 'M', model_type => 'NMOS' },
    sg13_hv_pmos => { letter => 'M', model_type => 'PMOS' },
    sg13_lv_nmos => { letter => 'M', model_type => 'NMOS' },
    sg13_lv_pmos => { letter => 'M', model_type => 'PMOS' },
    cap_cmim     => { letter => 'C', model_type => 'C'    },
    cap_rfcmim   => { letter => 'C', model_type => 'C'    },
);

usage() unless @ARGV;
my $gc_input = $ARGV[0];
my $gc_dir   = dirname($gc_input);

sub verbose { print STDERR "gnucap2xyce: @_\n" if $opt_verbose }
sub usage {
    die "Usage: $0 [options] <input.gc> [input.va]\n"
      . "Options:\n"
      . "  -o FILE                 write .cir to FILE (default: <input>.cir)\n"
      . "  --inplace               write next to the .gc\n"
      . "  --paramset-va FILE      paramset .va to scan (repeatable)\n"
      . "  --paramset-dir DIR      write generated per-paramset .va files here\n"
      . "                          (default: same dir as the .cir output)\n"
      . "  --corner-search PATH    extra dirs to search for cornerXXX.va files\n"
      . "  --paramset-library FILE  legacy inline-expansion mode\n"
      . "  -v|--verbose            chatter to stderr\n"
      . "  --keep-gnd              don't rewrite the ``ground`` node to 0\n";
}

# ---------------------------------------------------------------------------
# Paramset pass-through helpers
# ---------------------------------------------------------------------------

# Parse every ``paramset NAME UNDERLYING ... endparamset`` declaration in
# a .va file. Comments are stripped first (matching the parser's order)
# so a trailing ``//`` doesn't eat the following statement.
sub parse_paramset_file {
    my ($path) = @_;
    my @lines  = slurp_file($path);
    my $text   = join("\n", map { strip_line_comment($_) } @lines);
    # /* ... */ block comments removed too — IHP sources use them.
    $text =~ s{/\*.*?\*/}{ }gs;
    my @out;
    while ($text =~ /\bparamset\s+(\w+)\s+(\w+)(.*?)endparamset/sg) {
        my ($name, $under, $body) = ($1, $2, $3);
        push @out, {
            name       => $name,
            underlying => $under,
            body_text  => $body,
            src_path   => $path,
        };
    }
    return @out;
}

# Build the global %PARAMSET_TBL from the explicit --paramset-va files
# and any auto-discovered paramset .va sitting next to the testbench
# (the gnucap testbenches do ``load …paramset.so`` to bring them in;
# the .so doesn't exist for us, but the .va alongside does).
sub discover_paramsets {
    my ($gc_path, $tb_va_path) = @_;
    my @files = @opt_paramset_va;
    my $base_dir = dirname($gc_path);

    # Auto-discovery: peek at the testbench .va's ``load`` directives.
    my @candidates;
    if ($tb_va_path && -f $tb_va_path) {
        open my $f, '<', $tb_va_path or last;
        while (my $line = <$f>) {
            $line = strip_line_comment($line);
            if ($line =~ m{^\s*load\s+(\S+)}i) {
                my $rel = $1;
                $rel =~ s/\.so$/.va/;     # ``foo_paramset.so`` → ``foo_paramset.va``
                # plugins/models/X.va → models/X.va (gnucap build layout)
                (my $fallback = $rel) =~ s{plugins/models/}{models/};
                push @candidates, $rel, $fallback;
            }
        }
        close $f;
    }
    for my $c (@candidates) {
        my $abs = File::Spec->file_name_is_absolute($c)
                ? $c : File::Spec->rel2abs($c, $base_dir);
        if (-f $abs) {
            push @files, $abs;
            verbose("auto-discovered paramset .va: $abs");
        }
    }

    # Plus walk the ../../../models/ tree relative to the .gc if the
    # IHP layout is detected (gnucap/tests/gnucap/<dut>/<test>.gc).
    my $auto_models = File::Spec->rel2abs("$base_dir/../../../models",
                                          $base_dir);
    if (-d $auto_models) {
        opendir my $d, $auto_models or last;
        for my $f (readdir $d) {
            if ($f =~ /_paramset\.va$/) {
                my $abs = "$auto_models/$f";
                push @files, $abs unless grep { $_ eq $abs } @files;
            }
        }
        closedir $d;
    }

    my %seen;
    for my $f (@files) {
        next if $seen{$f}++;
        next unless -f $f;
        for my $ps (parse_paramset_file($f)) {
            next if exists $PARAMSET_TBL{$ps->{name}};
            $PARAMSET_TBL{$ps->{name}} = $ps;
        }
    }
    verbose(scalar(keys %PARAMSET_TBL) . " paramsets known");
}

# Lookup chain: explicit --paramset-vas → testbench load directives →
# IHP convention. Returns the path of an underlying compact-model .va
# we can `\\`include from a generated paramset .va, or empty string.
sub locate_underlying_va {
    my ($name, $gc_path) = @_;
    my $base_dir = dirname($gc_path);

    # Common IHP layouts to try:
    my @search;
    push @search, "$base_dir/../../../models";              # gnucap/models/
    push @search, "$base_dir/../../../../verilog-a/$name"; # verilog-a/<name>/
    push @search, "$base_dir/../../../../verilog-a/$name"; # same with -1 level
    push @search, "/usr/local/share/xyce/verilog-a";        # install tree
    push @search, "/usr/local/src/IHP-Open-PDK/ihp-sg13g2/libs.tech/verilog-a";
    push @search, "/usr/local/src/IHP-Open-PDK/ihp-sg13g2/libs.tech/gnucap/models";
    for my $d (@search) {
        next unless -d $d;
        # Standard cases — file named directly after the module
        for my $f ("$d/$name.va", "$d/$name/$name.va") {
            return $f if -f $f;
        }
        # IHP layout: sp_resistor lives in resistor.va, sp_capacitor in
        # capacitor.va; map module-name → file-name when the obvious
        # name doesn't exist.
        my %alias = (
            sp_resistor  => 'resistor.va',
            sp_capacitor => 'capacitor.va',
        );
        if ($alias{$name} && -f "$d/$alias{$name}") {
            return "$d/$alias{$name}";
        }
    }
    return '';
}

# Extract a corner module's localparams as { name => value-expr }.
# When the .gc driver instantiates ``moshv_tt corner_moshv()``, the
# bindings inside the paramset refer to that corner's localparams
# through ``corner_moshv.foo``. We pull them out so we can either
# bake values into the generated paramset .va or emit them as
# top-level .PARAMs in the .cir.
sub extract_corner_localparams {
    my ($corner_va_path, $module_name) = @_;
    return () unless -f $corner_va_path;
    my @lines = slurp_file($corner_va_path);
    my %out;
    my $in_target = 0;
    for my $raw (@lines) {
        my $line = strip_line_comment($raw);
        if ($line =~ /^\s*module\s+(\w+)\s*\(/) {
            $in_target = ($1 eq $module_name);
            next;
        }
        if ($line =~ /^\s*endmodule\b/ && $in_target) { $in_target = 0; next }
        next unless $in_target;
        if ($line =~ /^\s*localparam\s+(?:(?:real|integer)\s+)?(\w+)\s*=\s*([^;]+?)\s*;/) {
            $out{$1} = $2;
        }
    }
    return %out;
}

# Generate a per-paramset .va file. The file:
#   - `\`include`s the underlying compact-model .va
#   - declares any corner-derived parameters needed by the paramset
#     bindings (so ``corner_res.foo`` references resolve to values
#     visible to PyMS inside the paramset's scope)
#   - re-emits the original paramset body with the ``corner_X.`` prefix
#     stripped from identifier references
#   - tags the paramset with xyceModelGroup + xyceLevelNumber so the
#     C++ auto-loader registers it under the right device family
sub emit_paramset_va {
    my ($ps, $info, $level, $out_path, $underlying_va,
        $corner_vals_ref) = @_;
    my %cv = %$corner_vals_ref;
    my $body = $ps->{body_text};

    # Strip ``corner_<x>.name`` hierarchical refs in the body, and
    # collect the set of distinct simple-name references they map to.
    my %needed_corner;
    $body =~ s{\b(\w+)\.(\w+)\b}{
        if (defined $cv{$2}) {
            $needed_corner{$2} = $cv{$2};
            $2;                   # rewrite to bare identifier
        } else {
            "$1.$2";              # leave alone (might be a real hierarchical ref)
        }
    }ge;

    # Strip aliasparam — PyMS doesn't model aliases and the binding
    # path doesn't need them.
    $body =~ s/^\s*aliasparam\s+\w+\s*=\s*\w+\s*;\s*$//mg;

    open my $fh, '>', $out_path
        or die "gnucap2xyce: cannot write $out_path: $!\n";
    print $fh "// Generated by gnucap2xyce.pl from paramset $ps->{name}\n";
    print $fh "// Original source: $ps->{src_path}\n";
    print $fh "// Underlying compact model: $ps->{underlying}\n\n";
    if ($underlying_va) {
        print $fh "`include \"$underlying_va\"\n\n";
    }
    # Emit corner-derived params as Verilog-A parameter decls so the
    # paramset body's references resolve.
    if (%needed_corner) {
        print $fh "// Corner-derived constants (from cornerXXX.va):\n";
        for my $k (sort keys %needed_corner) {
            print $fh "parameter real $k = $needed_corner{$k};\n";
        }
        print $fh "\n";
    }
    # Xyce attribute tag for auto-discovery.
    my $devname = "IHP $ps->{name} paramset";
    print $fh "(* xyceModelGroup=\""
            . _group_for_letter($info->{letter})
            . "\" xyceLevelNumber=\"$level\" "
            . "xyceDeviceName=\"$devname\" *)\n";
    print $fh "paramset $ps->{name} $ps->{underlying}\n";
    print $fh $body;
    print $fh "\nendparamset\n";
    close $fh;
    verbose("wrote $out_path (level=$level)");
}

sub _group_for_letter {
    my ($l) = @_;
    return 'MOSFET'    if $l eq 'M';
    return 'BJT'       if $l eq 'Q';
    return 'Diode'     if $l eq 'D';
    return 'Resistor'  if $l eq 'R';
    return 'Capacitor' if $l eq 'C';
    return 'Inductor'  if $l eq 'L';
    return '';
}

# ---------------------------------------------------------------------------
# Read a file with comment-aware line joining. Gnucap uses // for comments
# but no \-continuation (one logical statement per line).
# ---------------------------------------------------------------------------
sub slurp_file {
    my ($path) = @_;
    open my $fh, '<', $path or die "gnucap2xyce: can't read $path: $!\n";
    my @lines = <$fh>;
    close $fh;
    chomp @lines;
    return @lines;
}

sub strip_line_comment {
    my ($s) = @_;
    # Respect "..." string literals when stripping ``//`` comments.
    my $out = '';
    my $in_str = 0;
    my $i = 0;
    while ($i < length $s) {
        my $c = substr($s, $i, 1);
        if ($in_str) {
            $out .= $c;
            if ($c eq '\\' && $i + 1 < length $s) {
                $out .= substr($s, $i + 1, 1);
                $i += 2;
                next;
            }
            $in_str = 0 if $c eq '"';
        } else {
            if ($c eq '"') { $in_str = 1; $out .= $c }
            elsif ($c eq '/' && $i + 1 < length $s
                   && substr($s, $i + 1, 1) eq '/') { last }
            else { $out .= $c }
        }
        $i++;
    }
    $out =~ s/\s+$//;
    return $out;
}

# Resolve an ``include`` path. Gnucap accepts bare paths (no quotes);
# resolution is relative to the .gc file's directory.
sub resolve_include {
    my ($rel, $base_dir) = @_;
    return $rel if File::Spec->file_name_is_absolute($rel) && -f $rel;
    my $abs = File::Spec->rel2abs($rel, $base_dir);
    return $abs if -f $abs;
    return $rel;  # let downstream complain
}

# ---------------------------------------------------------------------------
# Parse the .va testbench. We want:
#   - module name + port list (top-level testbench has no ports;
#     subcircuit-style would have ports)
#   - parameter declarations (become Xyce .PARAM at top level)
#   - instances:  <type> [#(.k(v)...)] <name>(<nodes>);
#   - ``ground gnd;``  (suppress + rewrite gnd → 0 in nodes)
#   - ``electrical ...;``  (Xyce doesn't need explicit electrical decls)
#   - ``input ...;`` / ``output ...;`` / ``inout ...;``  (only matters
#     when emitting a .SUBCKT — top-level testbench: drop)
# ---------------------------------------------------------------------------
sub parse_tb_va {
    my ($path) = @_;
    my @lines = slurp_file($path);

    my $in_module = 0;
    my %tb = (name => '', ports => [], params => [], instances => [],
              ground_node => '', has_module_decl => 0);

    for my $raw (@lines) {
        my $line = strip_line_comment($raw);
        next if $line =~ /^\s*$/;

        # Outside-module: ``load``, ``include``, ``options`` go to header
        if (!$in_module) {
            if ($line =~ /^\s*module\s+(\w+)\s*\(([^)]*)\)\s*;/) {
                $tb{name} = $1;
                my $portlist = $2;
                $portlist =~ s/\s+//g;
                $tb{ports} = $portlist eq '' ? [] : [split /,/, $portlist];
                $tb{has_module_decl} = 1;
                $in_module = 1;
                next;
            }
            # Anything else outside the module — file-level directives
            # — is handled by the caller (the .gc driver carries them).
            next;
        }

        # Inside module body
        last if $line =~ /^\s*endmodule\b/;

        # ``ground gnd;`` — record the ground node name to rewrite later
        if ($line =~ /^\s*ground\s+(\w+)\s*;/) {
            $tb{ground_node} = $1;
            next;
        }
        # ``electrical foo, bar;`` / ``input ...`` / ``output ...`` / ``inout ...``
        # — declarative only, no Xyce equivalent at top level.
        next if $line =~ /^\s*(?:electrical|input|output|inout)\b/;

        # ``parameter [type] NAME = VAL [from ...];``  Type is optional
        # (real/integer); some gnucap sources reorder it as ``parameter
        # NAME type = VAL`` so accept that too. Only consume a leading
        # token as the type when it's specifically ``real`` or ``integer``.
        if ($line =~ /^\s*parameter\s+(?:(?:real|integer)\s+)?(\w+)\s*(?:real|integer)?\s*(?:=\s*([^;]+?))?\s*(?:from\s+[^;]+?)?\s*;/) {
            my ($pn, $pv) = ($1, $2 // '');
            $pv =~ s/^\s+|\s+$//g;
            $pv = 0 if $pv eq '';
            push @{$tb{params}}, { name => $pn, val => $pv };
            next;
        }
        # ``localparam`` — Xyce has no per-block local params; treat as .PARAM
        if ($line =~ /^\s*localparam\s+(?:(?:real|integer)\s+)?(\w+)\s*=\s*([^;]+?)\s*;/) {
            push @{$tb{params}}, { name => $1, val => $2, local => 1 };
            next;
        }

        # Instance: ``<type> [#(.k(v)...)] <inst>(<nodes>);``
        if ($line =~ /^\s*(\w+)\s*(?:\#\(\s*(.*?)\s*\))?\s*(\w+)\s*\(([^)]*)\)\s*;/) {
            my ($type, $params_blk, $iname, $nodelist) = ($1, $2 // '', $3, $4);
            my @nodes = split /\s*,\s*/, $nodelist;
            my @ipairs;
            while ($params_blk =~ /\.(\w+)\s*\(\s*([^)]+?)\s*\)/g) {
                push @ipairs, { name => $1, val => $2 };
            }
            push @{$tb{instances}}, {
                type   => $type,
                inst   => $iname,
                nodes  => \@nodes,
                params => \@ipairs,
            };
            next;
        }

        verbose("skipping unrecognised body line: $line");
    }
    return \%tb;
}

# ---------------------------------------------------------------------------
# Rewrite a node identifier per the gnucap-vs-Xyce conventions:
#   - ``gnd`` (or whatever the ``ground`` decl named) → 0
# ---------------------------------------------------------------------------
sub xyce_node {
    my ($name, $gnd) = @_;
    return ($gnd ne '' && $name eq $gnd) ? '0' : $name;
}

# ---------------------------------------------------------------------------
# Instance translation. The first three types are gnucap built-in
# primitives we map to native SPICE; everything else becomes an
# X-subckt call (assuming the module either expands inline via PyMS
# or has a matching .SUBCKT/.MODEL elsewhere).
# ---------------------------------------------------------------------------
sub emit_instance {
    my ($inst, $gnd, $paramset_resolved) = @_;
    my $type = $inst->{type};
    my $name = $inst->{inst};
    my @nodes = map { xyce_node($_, $gnd) } @{$inst->{nodes}};
    my %p = map { $_->{name} => $_->{val} } @{$inst->{params}};

    # Add a device-letter prefix only if the instance name doesn't
    # already start with one of the SPICE letters Xyce uses for the
    # corresponding primitive. Avoids ``IIr1`` from gnucap's ``Ir1``.
    # Also stash the final emitted name on the instance hash so the
    # .PRINT translator can use it.
    my $with_letter = sub {
        my ($letter, $n) = @_;
        my $emitted = ($n =~ /^\Q$letter\E/i) ? $n : $letter . $n;
        $inst->{emitted_name} = $emitted;
        return $emitted;
    };

    if ($type eq 'vsource') {
        my $dc = $p{dc} // 0;
        return sprintf("%s %s %s DC %s",
                       $with_letter->('V', $name), $nodes[0], $nodes[1], $dc);
    }
    if ($type eq 'vsine') {
        my $dc  = $p{dc}  // 0;
        my $mag = $p{mag} // 1;
        my $freq= $p{freq} // '1k';
        # SIN source: VxName n+ n- SIN(<offset> <amplitude> <freq>)
        return sprintf("%s %s %s DC %s AC %s SIN(%s %s %s)",
                       $with_letter->('V', $name),
                       $nodes[0], $nodes[1], $dc, $mag, $dc, $mag, $freq);
    }
    if ($type eq 'idc') {
        my $dc = $p{dc} // 0;
        return sprintf("%s %s %s DC %s",
                       $with_letter->('I', $name), $nodes[0], $nodes[1], $dc);
    }
    if ($type eq 'resistor') {
        my $r = $p{r} // 1;
        return sprintf("%s %s %s %s",
                       $with_letter->('R', $name), $nodes[0], $nodes[1], $r);
    }
    # Paramset pass-through. If a per-paramset .va was generated
    # for this type, just emit a device-letter card referencing the
    # already-emitted .MODEL. PyMS' paramset resolver handles the
    # bindings; here we only stamp instance params from the call.
    if ($paramset_resolved && exists $paramset_resolved->{$type}) {
        my $r = $paramset_resolved->{$type};
        my $letter = $r->{info}{letter};
        my $iname = $with_letter->($letter, $name);
        # Pad the node list to the underlying's full port count
        # before the model name. The PyMS wrapper's registerLIDs
        # indexes extLIDVec[0..n_ext-1] unconditionally, so missing
        # optional ports must be supplied as ``0`` (ground) rather
        # than omitted. Without padding, the underlying-thermal /
        # body-pickup index is past-end → uninitialised LID →
        # segfault during matrixGlobalToLocal.
        my $n_underlying = $r->{n_underlying_ports} // scalar @nodes;
        my @padded = @nodes;
        while (@padded < $n_underlying) { push @padded, '0' }
        my @inst_kvs = map { "$_->{name}=$_->{val}" } @{$inst->{params}};
        my $line = sprintf("%s %s %s",
                           $iname, join(' ', @padded), $r->{model_name});
        $line .= ' ' . join(' ', @inst_kvs) if @inst_kvs;
        return $line;
    }

    # Legacy inline-expansion path (kept for back-compat with
    # --paramset-library mode).
    if (exists $PARAMSETS->{$type}) {
        return expand_paramset_inline($PARAMSETS->{$type}, $inst, $gnd);
    }

    # Generic: X-subckt instance. Param list goes through verbatim;
    # PyMS-auto-loaded modules accept ``key=val`` instance params,
    # subcircuits accept them via PARAMS:.
    my $iname = $with_letter->('X', $name);
    my $params_str = join(' ', map { "$_->{name}=$_->{val}" } @{$inst->{params}});
    if ($params_str ne '') {
        return sprintf("%s %s %s PARAMS: %s",
                       $iname, join(' ', @nodes), $type, $params_str);
    }
    return sprintf("%s %s %s",
                   $iname, join(' ', @nodes), $type);
}

# Inline-expand a paramset call. Produces one or more netlist lines
# (joined with \n by the caller).
sub expand_paramset_inline {
    my ($ps, $inst, $gnd) = @_;
    my $iname = $inst->{inst};

    # Resolve the call-site value for each paramset-declared instance
    # parameter: testbench overrides win, otherwise the paramset's
    # default.
    my %callvals;
    for my $p (@{$ps->{params}}) {
        $callvals{$p->[0]} = $p->[1];
    }
    for my $cp (@{$inst->{params}}) {
        $callvals{$cp->{name}} = $cp->{val};
    }

    # Substitute paramset params (with their call-site values) into
    # an expression. Word-boundary so ``w`` doesn't eat into ``weff``.
    my $subst_params = sub {
        my ($e) = @_;
        for my $k (keys %callvals) {
            my $v = $callvals{$k};
            $e =~ s/\b\Q$k\E\b/($v)/g;
        }
        return $e;
    };

    # Substitute the paramset's localparam names with the
    # instance-prefixed names that the per-instance .PARAM lines emit.
    # Localparams reference earlier localparams; prefixing keeps them
    # disambiguated across multiple paramset instances in one netlist.
    my $subst_locals = sub {
        my ($e) = @_;
        for my $lp (@{$ps->{localparams}}) {
            my $n = $lp->[0];
            $e =~ s/\b\Q$n\E\b/${iname}_$n/g;
        }
        return $e;
    };

    my @lines;
    # Localparams: emit each as a .PARAM with paramset-param values
    # substituted and PRIOR localparams renamed to <inst>_<name>.
    # (For the current localparam being defined, we don't rename
    # references to itself — they only appear via earlier locals.)
    for my $lp (@{$ps->{localparams}}) {
        my $expr = $subst_params->($lp->[1]);
        # Replace references to *other* localparams (skip self).
        for my $other (@{$ps->{localparams}}) {
            next if $other->[0] eq $lp->[0];
            $expr =~ s/\b\Q$other->[0]\E\b/${iname}_$other->[0]/g;
        }
        push @lines, sprintf(".PARAM %s_%s={%s}",
                             $iname, $lp->[0], $expr);
    }

    # Map testbench nodes onto the underlying's full port list,
    # padding any missing tail ports with 0 (gnucap behaviour: a
    # paramset called with N nodes ties any remaining ports to 0).
    my @device_nodes = @{$ps->{device_nodes}};
    my @passed = map { xyce_node($_, $gnd) } @{$inst->{nodes}};
    my @final_nodes;
    for (my $i = 0; $i < @device_nodes; $i++) {
        if ($i < @passed) {
            push @final_nodes, $passed[$i];
        } else {
            # Use whatever the paramset's device_nodes carried as the
            # default (typically '0' for the trailing thermal port).
            push @final_nodes, $device_nodes[$i];
        }
    }

    if (defined $ps->{level} && $ps->{letter} ne 'X') {
        my $mname = "m_" . $iname;
        my @mbinds;
        for my $b (@{$ps->{model_bindings}}) {
            my $e = $subst_locals->($subst_params->($b->[1]));
            push @mbinds, sprintf("%s={%s}", $b->[0], $e);
        }
        my $mcard = sprintf(".MODEL %s %s level=%d",
                            $mname, $ps->{type}, $ps->{level});
        $mcard .= "\n+ " . join(" ", @mbinds) if @mbinds;
        push @lines, $mcard;

        my @ibinds;
        for my $b (@{$ps->{inst_bindings}}) {
            my $e = $subst_locals->($subst_params->($b->[1]));
            push @ibinds, sprintf("%s={%s}", $b->[0], $e);
        }
        my $dev_line = sprintf("%s%s %s %s",
                               $ps->{letter}, $iname,
                               join(' ', @final_nodes), $mname);
        $dev_line .= ' ' . join(' ', @ibinds) if @ibinds;
        push @lines, $dev_line;
    } else {
        # Fallback when no device family is known — emit an X-call
        # to a .SUBCKT named after the paramset (caller must
        # .INCLUDE a paramset2xyce.pl -o .cir for this to resolve).
        push @lines, sprintf("* paramset %s: no device-family mapping; "
                           . "X-call to a .SUBCKT", $ps->{underlying});
        my @ibinds;
        for my $b ((@{$ps->{inst_bindings}}, @{$ps->{model_bindings}})) {
            my $e = $subst_locals->($subst_params->($b->[1]));
            push @ibinds, sprintf("%s={%s}", $b->[0], $e);
        }
        push @lines, sprintf("X%s %s %s%s",
                             $iname, join(' ', @final_nodes), $ps->{name},
                             @ibinds ? " " . join(' ', @ibinds) : '');
    }
    return join("\n", @lines);
}

# ---------------------------------------------------------------------------
# Parse the .gc driver. We pull out:
#   - ``include`` directives (resolve relative to the .gc dir)
#   - ``options ...``
#   - top-level testbench instantiation ``tbtype #(.p(v)) tb(nodes);``
#     — used to merge the .va testbench into top-level netlist
#   - ``print <analysis> <expr1> <expr2> ...``
#   - analysis commands: ``dc`` / ``ac`` / ``tran``
#   - sweep parameters declared via ``parameter real X = Y;``
# ---------------------------------------------------------------------------
sub parse_gc {
    my ($path) = @_;
    my @lines = slurp_file($path);
    my %g = (
        includes  => [],     # absolute paths
        options   => [],     # raw option strings
        tb_inst   => undef,  # { type, inst, nodes, params }
        prints    => [],     # { kind => dc/ac/tran, exprs => [...] }
        analyses  => [],     # { kind, args }
        params    => [],     # top-level sweep params
        corners   => [],     # ``moshv_ff corner_moshv();`` etc.
    );
    my $dir = dirname($path);
    for my $raw (@lines) {
        my $line = strip_line_comment($raw);
        next if $line =~ /^\s*$/;

        if ($line =~ /^\s*load\b/)     { next }   # plugin loads — discard
        if ($line =~ /^\s*verilog\b/)  { next }   # mode marker  — discard
        if ($line =~ /^\s*status\b/)   { next }   # output toggle — discard

        if ($line =~ /^\s*include\s+(\S+)/) {
            push @{$g{includes}}, resolve_include($1, $dir);
            next;
        }
        if ($line =~ /^\s*options?\s+(.*)/) {
            push @{$g{options}}, $1;
            next;
        }
        if ($line =~ /^\s*parameter\s+(?:(?:real|integer)\s+)?(\w+)\s*(?:real|integer)?\s*=\s*([^;]+?)\s*;/) {
            push @{$g{params}}, { name => $1, val => $2 };
            next;
        }
        # Top-level testbench instance: ``<type> [#(.p(v))] <inst>(<nodes>);``
        if ($line =~ /^\s*(\w+)\s*(?:\#\(\s*(.*?)\s*\))?\s*(\w+)\s*\(([^)]*)\)\s*;/) {
            my ($type, $params_blk, $iname, $nodelist) = ($1, $2 // '', $3, $4);
            my @nodes = split /\s*,\s*/, $nodelist;
            my @ipairs;
            while ($params_blk =~ /\.(\w+)\s*\(\s*([^)]+?)\s*\)/g) {
                push @ipairs, { name => $1, val => $2 };
            }
            # First port-less instance is the corner-selection module
            # (e.g. ``moshv_ff corner_moshv();``); next one with no
            # ports either is the testbench at the top level.
            my $entry = { type => $type, inst => $iname,
                          nodes => \@nodes, params => \@ipairs };
            if (@nodes == 0 || ($nodelist =~ /^\s*$/)) {
                push @{$g{corners}}, $entry;
            } else {
                $g{tb_inst} = $entry;
            }
            next;
        }
        # ``print <analysis> <exprs...>``
        if ($line =~ /^\s*print\s+(\w+)\s+(.+?)\s*$/) {
            my ($kind, $exprs_str) = ($1, $2);
            # split exprs on whitespace but respect (...)
            my @exprs = parse_print_exprs($exprs_str);
            push @{$g{prints}}, { kind => $kind, exprs => \@exprs };
            next;
        }
        # Analyses: ``dc <var> <lo> <hi> <step> [basic]``
        if ($line =~ /^\s*(dc|ac|tran|op)\b\s*(.*)$/) {
            push @{$g{analyses}}, { kind => $1, args => $2 };
            next;
        }
        verbose("skipping unrecognised .gc line: $line");
    }
    return \%g;
}

# Locate the .va include that defines a given module name.
sub find_module_def {
    my ($module_name, $includes) = @_;
    for my $inc (@$includes) {
        next unless $inc =~ /\.va$/ && -f $inc;
        open my $f, '<', $inc or next;
        while (<$f>) {
            if (/^\s*module\s+\Q$module_name\E\s*\(/) {
                close $f;
                return $inc;
            }
        }
        close $f;
    }
    return '';
}

# A "testbench module" has at least one non-parameter, non-port
# statement (instance, ground decl, etc.) inside its body. A corner
# module has only localparam.
sub is_testbench_module {
    my ($path, $module_name) = @_;
    open my $f, '<', $path or return 0;
    my $in_target = 0;
    my $has_inst  = 0;
    while (my $raw = <$f>) {
        my $line = strip_line_comment($raw);
        if ($line =~ /^\s*module\s+(\w+)\s*\(/) {
            $in_target = ($1 eq $module_name);
            next;
        }
        if ($line =~ /^\s*endmodule\b/ && $in_target) { last }
        next unless $in_target;
        # Skip declarative lines
        next if $line =~ /^\s*$/;
        next if $line =~ /^\s*(?:electrical|input|output|inout|ground|parameter|localparam)\b/;
        # Anything else inside the module body counts as an instance
        if ($line =~ /\w+\s*(?:#\()?[\w\s,.(){}=]*\s*;/) {
            $has_inst = 1;
            last;
        }
    }
    close $f;
    return $has_inst;
}

# Whitespace-split, respecting parens.
sub parse_print_exprs {
    my ($s) = @_;
    my @out;
    my $cur = '';
    my $depth = 0;
    for (my $i = 0; $i < length $s; $i++) {
        my $c = substr($s, $i, 1);
        if ($c eq '(') { $depth++; $cur .= $c }
        elsif ($c eq ')') { $depth--; $cur .= $c }
        elsif ($c =~ /\s/ && $depth == 0) {
            if (length $cur) { push @out, $cur; $cur = '' }
        } else { $cur .= $c }
    }
    push @out, $cur if length $cur;
    return @out;
}

# Translate a gnucap probe expression to Xyce. The gnucap form supports
# hierarchical access ``v(tb.r1)`` which becomes ``V(r1)`` once the
# testbench is inlined at the top level. ``i(tb.vdd_src)`` becomes
# ``I(Vvdd_src)`` since the vsource was emitted as ``V<name>``.
# ``iter(0)`` is gnucap-specific — silently drop.
sub xlate_probe_expr {
    my ($e, $vsrcs, $isrcs, $dev_first_node) = @_;
    return undef if $e =~ /^iter\(/i;
    if ($e =~ /^([vi])\(\s*(.+?)\s*\)$/i) {
        my ($kind, $arg) = (lc $1, $2);
        # Strip ``tb.`` prefix — testbench is inlined at top level
        $arg =~ s/^tb\.//;
        if ($kind eq 'v') {
            # gnucap convention: ``v(<device>)`` is the voltage at the
            # device's first non-ground terminal. Xyce's V() takes a
            # NODE name, not a device name, so look up the device-to-
            # first-node map and substitute.
            if (defined $dev_first_node && exists $dev_first_node->{$arg}) {
                return "V($dev_first_node->{$arg})";
            }
            # Multiple comma-separated nodes inside V(): keep as-is
            return "V($arg)";
        } else {
            # I() through a named source. Gnucap names the instance
            # (``vdd_src``); Xyce wants the actual emitted device name
            # (``Vvdd_src`` for ``Ir1`` → ``Ir1`` since the letter was
            # already there). $vsrcs / $isrcs map original-name →
            # emitted-name.
            return "I($vsrcs->{$arg})" if exists $vsrcs->{$arg};
            return "I($isrcs->{$arg})" if exists $isrcs->{$arg};
            return "I($arg)";  # last-resort passthrough
        }
    }
    return $e;
}

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
sub emit {
    my ($gc_path, $tb_va_path) = @_;
    my $gc = parse_gc($gc_path);

    # Identify which port-less instance is the testbench: it's the
    # one whose module body has actual instances (vsource/idc/sg13_/
    # etc.), as opposed to the corner module which carries only
    # localparam declarations. The gnucap .gc convention names them
    # ``corner_<lib>`` and ``tb`` respectively, but we don't rely on
    # that — we inspect the included .va files.
    my $tb_inst;
    my @corner_insts;
    for my $entry (@{$gc->{corners}}) {
        my $inc = find_module_def($entry->{type}, $gc->{includes});
        if ($inc && is_testbench_module($inc, $entry->{type})) {
            $tb_inst = $entry;
            $tb_va_path ||= $inc;
        } else {
            push @corner_insts, $entry;
        }
    }
    $tb_inst ||= $gc->{tb_inst};
    $gc->{tb_inst}    = $tb_inst;
    $gc->{corners}    = \@corner_insts;

    # Ported testbench (``tb_moshv_inv #(...) tb(in, out);``) — the
    # earlier loop only finds port-less testbenches; look up the .va
    # for the ported case here.
    if (!$tb_va_path && $tb_inst) {
        $tb_va_path = find_module_def($tb_inst->{type}, $gc->{includes});
    }

    my $tb = $tb_va_path ? parse_tb_va($tb_va_path) : undef;

    # --- Paramset pass-through pre-pass ---
    # Discover paramset .va files referenced by the testbench (or
    # supplied explicitly), then for each paramset the testbench
    # actually instantiates, generate a per-paramset .va next to the
    # output .cir. Each gets a unique xyceLevelNumber so PyMS auto-
    # discovery registers (R/C/M/Q/D, <level>) → that paramset's .va.
    discover_paramsets($gc_path, $tb_va_path)
        unless $opt_paramset_lib;
    # Map: gnucap-instance-type → { level, info, va_path } for the
    # generated .va. Populated as we see paramset instances referenced.
    my %paramset_resolved;
    my $paramset_dir = $opt_paramset_dir;
    if (!$paramset_dir) {
        $paramset_dir = $opt_output ? dirname($opt_output) : $gc_dir;
    }
    # Extract corner localparams for substitution into paramset bodies.
    # ``corner_<x>.foo`` references inside paramsets are rewritten to
    # bare ``foo`` and the value is emitted as a parameter declaration
    # at the head of the generated paramset .va.
    my %corner_vals;
    for my $corner_inst (@corner_insts) {
        my $cmod_name = $corner_inst->{type};
        # Look up the corner module's .va by searching the include list.
        my $cpath;
        for my $inc (@{$gc->{includes}}) {
            next unless $inc =~ /\.va$/ && -f $inc;
            my @ls = slurp_file($inc);
            for my $l (@ls) {
                if ($l =~ /^\s*module\s+\Q$cmod_name\E\s*\(/) {
                    $cpath = $inc; last;
                }
            }
            last if $cpath;
        }
        if ($cpath) {
            my %vals = extract_corner_localparams($cpath, $cmod_name);
            for my $k (keys %vals) { $corner_vals{$k} //= $vals{$k} }
        }
    }

    # Pre-resolve every paramset the testbench instantiates so we
    # know the levels + paths when emitting the .HDL and .MODEL lines
    # below.
    if ($tb) {
        for my $i (@{$tb->{instances}}) {
            my $t = $i->{type};
            next unless $PARAMSET_TBL{$t};
            next if $paramset_resolved{$t};
            my $ps = $PARAMSET_TBL{$t};
            my $info = $UNDERLYING_INFO{$ps->{underlying}};
            unless ($info) {
                verbose("WARNING: no UNDERLYING_INFO for "
                      . "$ps->{underlying} (paramset $t); "
                      . "defaulting to letter=X");
                $info = { letter => 'X', model_type => $ps->{underlying} };
            }
            $LEVEL_COUNTER++;
            my $level = $LEVEL_COUNTER;
            my $out_file = File::Spec->catfile(
                $paramset_dir, "_ps_$ps->{name}.va");
            my $underlying_va = locate_underlying_va(
                $ps->{underlying}, $gc_path);
            emit_paramset_va($ps, $info, $level, $out_file,
                             $underlying_va, \%corner_vals);
            # Count the underlying's port list so we can pad missing
            # nodes on the device card. Cheap-and-cheerful: parse the
            # paramset body's ``\`include`` target for ``module
            # UNDERLYING(...)`` and count commas + 1.
            my $n_under_ports = 0;
            if ($underlying_va && -f $underlying_va) {
                open my $f, '<', $underlying_va or next;
                while (my $l = <$f>) {
                    if ($l =~ /^\s*module\s+\Q$ps->{underlying}\E\s*\(([^)]*)\)/) {
                        my $pl = $1;
                        $pl =~ s/\s+//g;
                        $n_under_ports = scalar(split /,/, $pl);
                        last;
                    }
                }
                close $f;
            }
            $paramset_resolved{$t} = {
                level     => $level,
                info      => $info,
                va_path   => $out_file,
                model_name => "m_$ps->{name}",
                n_underlying_ports => $n_under_ports,
            };
        }
    }

    my @out;
    push @out, "* Generated by gnucap2xyce.pl from " . basename($gc_path);
    push @out, "* Source testbench: " . basename($tb_va_path)
        if $tb_va_path;
    push @out, "";

    # Module include files (compact-model wrappers): emit as .HDL so
    # PyMS picks them up. Corner .va files contain only localparam
    # declarations — translate to .PARAM lines.
    my %seen_inc;
    for my $inc (@{$gc->{includes}}) {
        next if $seen_inc{$inc}++;
        next if $inc =~ /tb_\w+\.va$/;     # the testbench itself — inlined below
        # Heuristic: corner file → .PARAM emission; module file → .HDL.
        my $base = basename($inc);
        if ($base =~ /^corner/i) {
            push @out, emit_corner_params($inc, $gc);
        } elsif ($inc =~ /_paramset\.va$/) {
            # Paramset files don't get .HDL'd directly — we emit
            # per-paramset split files below. The original file is just
            # informational here.
            push @out, "* original paramset source: $inc";
        } elsif ($inc =~ /\.va$/) {
            push @out, qq{.HDL "$inc"};
        } else {
            push @out, qq{.INCLUDE "$inc"};
        }
    }

    # .HDL the generated per-paramset .va files.
    for my $t (sort keys %paramset_resolved) {
        my $r = $paramset_resolved{$t};
        push @out, qq{.HDL "$r->{va_path}"};
    }
    push @out, "";

    # Top-level sweep parameters
    my %seen_param;
    for my $p (@{$gc->{params}}) {
        next if $seen_param{$p->{name}}++;
        push @out, ".PARAM $p->{name}=$p->{val}";
    }

    # Testbench inlining
    my %vsrcs;  # name => emitted-name (for I() probe rewriting)
    my %isrcs;
    my %dev_first_node;   # gnucap inst name → first non-gnd node, for V() probes
    if ($tb) {
        my $gnd = $opt_keep_gnd ? '' : ($tb->{ground_node} // '');

        # Testbench-level parameters (top-level testbench has no ports).
        # If the .gc already declared a sweep parameter with the same
        # name, skip the redundant testbench-side .PARAM — its value
        # would be the override-from-.gc, which is just the .gc param
        # name (e.g. ``vd=vd``), a no-op self-reference that Xyce
        # warns about as a duplicate.
        for my $p (@{$tb->{params}}) {
            next if $seen_param{$p->{name}};
            my $val = $p->{val};
            if ($gc->{tb_inst}) {
                for my $ov (@{$gc->{tb_inst}{params}}) {
                    $val = $ov->{val} if $ov->{name} eq $p->{name};
                }
            }
            next if $val eq $p->{name};  # identity override after dedup
            push @out, ".PARAM $p->{name}=$val";
            $seen_param{$p->{name}} = 1;
        }
        push @out, "" if @{$tb->{params}};

        # .MODEL cards for every paramset the testbench instantiates.
        # All instances of a given paramset share its .MODEL — Xyce
        # then dispatches to the PyMS-compiled wrapper registered at
        # (letter, level) by the auto-loader scanning the per-paramset
        # .va files we emitted above.
        for my $t (sort keys %paramset_resolved) {
            my $r = $paramset_resolved{$t};
            push @out, sprintf(".MODEL %s %s level=%d",
                               $r->{model_name},
                               $r->{info}{model_type},
                               $r->{level});
        }
        push @out, "" if %paramset_resolved;

        # Instance emission. emit_instance stashes the final
        # device-letter-prefixed name on the instance hash so the
        # .PRINT rewriter can target it.
        for my $i (@{$tb->{instances}}) {
            push @out, emit_instance($i, $gnd, \%paramset_resolved);
            if ($i->{type} =~ /^vs(?:ource|ine)$/) {
                $vsrcs{$i->{inst}} = $i->{emitted_name} // ('V' . $i->{inst});
            }
            if ($i->{type} eq 'idc') {
                $isrcs{$i->{inst}} = $i->{emitted_name} // ('I' . $i->{inst});
            }
            # Record the first non-ground node for V() probe rewriting.
            # gnucap's ``v(r1)`` = voltage at r1's first non-gnd terminal.
            my $first_nongnd;
            for my $n (@{$i->{nodes}}) {
                my $xn = xyce_node($n, $gnd);
                if ($xn ne '0') { $first_nongnd = $xn; last }
            }
            $dev_first_node{$i->{inst}} = $first_nongnd if defined $first_nongnd;
        }
    }
    push @out, "";

    # Analyses + .PRINT directives. Xyce wants .PRINT to come with the
    # analysis kind, so emit one .PRINT per kind seen in ``print``.
    for my $a (@{$gc->{analyses}}) {
        my $k = uc $a->{kind};
        my $args = $a->{args} // '';
        # ``dc`` with no sweep is an operating-point analysis, which
        # in Xyce is ``.OP`` (``.DC`` alone produces a parse error).
        if ($k eq 'DC' && $args =~ /^\s*$/) {
            push @out, '.OP';
        } else {
            push @out, ".$k $args";
        }
    }
    for my $pr (@{$gc->{prints}}) {
        my $k = uc $pr->{kind};
        my @e;
        for my $x (@{$pr->{exprs}}) {
            my $t = xlate_probe_expr($x, \%vsrcs, \%isrcs, \%dev_first_node);
            push @e, $t if defined $t;
        }
        push @out, ".PRINT $k " . join(' ', @e) if @e;
    }

    # Carry over gnucap options as Xyce options where there's an
    # equivalent; otherwise leave a comment.
    for my $o (@{$gc->{options}}) {
        if ($o =~ /itl6\s*=\s*(\S+)/i) {
            push @out, ".OPTIONS NONLIN MAXSTEP=$1  ; was: options $o";
        } else {
            push @out, "* gnucap2xyce: unmapped option ``$o``";
        }
    }

    push @out, ".END";

    # Sanity pass: any identifier referenced inside ``{ ... }``
    # braced expressions OR as a bare value of a ``param=value`` token
    # that hasn't been defined as a ``.PARAM`` gets a 0-default
    # emitted near the top. Most common cause: paramset bindings that
    # reference a statistical-corner module the testbench doesn't
    # instantiate (e.g. ``res_stat_param.drsh_rsil`` flattens to
    # ``drsh_rsil`` but only the typical corner is included).
    my %defined;
    for my $ln (@out) {
        while ($ln =~ /^\s*\.PARAM\s+(\w+)\s*=/gmi) { $defined{lc $1} = 1 }
    }
    my %referenced;
    for my $ln (@out) {
        # ``{ expr }`` blocks
        while ($ln =~ /\{([^{}]*)\}/g) {
            my $expr = $1;
            while ($expr =~ /\b([A-Za-z_][A-Za-z0-9_]*)\b/g) {
                $referenced{lc $1} = 1;
            }
        }
    }
    # Known Xyce reserved/built-in identifiers — don't fallback-default
    # these even if they appear in expressions.
    my %reserved = map { $_ => 1 } qw(
        time temp tnom v i sqrt exp log ln pow abs min max
        sin cos tan atan asin acos sinh cosh tanh atan2
        if else inf
    );
    my @missing;
    for my $r (sort keys %referenced) {
        next if $defined{$r} || $reserved{$r};
        # Numeric constants picked up by the regex (rare — already
        # filtered by the leading letter, but ``e6`` etc. could slip
        # through after a digit).
        next if $r =~ /^\d/;
        push @missing, $r;
    }
    if (@missing) {
        my @prologue = ("* gnucap2xyce: 0-defaults for params referenced "
                      . "but not defined (probably statistical-corner "
                      . "bindings; typical-corner sim drops the variance):");
        push @prologue, ".PARAM $_=0" for @missing;
        # Insert just after the header comments (line 2) so PARAMs
        # are defined before any usage.
        splice @out, 2, 0, @prologue;
    }
    return join("\n", @out) . "\n";
}

# Corner files declare a single ``module foo();`` containing only
# ``localparam`` declarations. Emit them as ``.PARAM`` so the testbench
# (which references e.g. ``sg13g2_hv_nmos_vfbo``) sees the corner-typ
# values. Multi-corner selection (typ/ff/ss/...) is not yet handled —
# we emit whichever module appears first in the file.
sub emit_corner_params {
    my ($path, $gc) = @_;
    my @lines = slurp_file($path);
    my @out = ("* corner file: " . basename($path));
    # Which corner module did the .gc select?
    my $want;
    if ($gc && @{$gc->{corners}}) {
        $want = $gc->{corners}[0]{type};
    }
    my $in_target = 0;
    my $depth = 0;
    for my $raw (@lines) {
        my $line = strip_line_comment($raw);
        next if $line =~ /^\s*$/;
        if ($line =~ /^\s*module\s+(\w+)\s*\(/) {
            $in_target = (defined $want && $1 eq $want) ? 1 : 0;
            $in_target ||= !defined $want;  # if none specified, take first
            $want //= $1;
            next;
        }
        if ($line =~ /^\s*endmodule\b/) {
            $in_target = 0;
            next;
        }
        next unless $in_target;
        if ($line =~ /^\s*localparam\s+(?:\w+\s+)?(\w+)\s*=\s*([^;]+?)\s*;/) {
            push @out, ".PARAM $1=$2";
        }
    }
    return @out;
}

# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
my $tb_va = (@ARGV >= 2) ? $ARGV[1] : '';
my $output = emit($gc_input, $tb_va);

if ($opt_inplace) {
    (my $target = $gc_input) =~ s/\.gc$/.cir/;
    open my $ofh, '>', $target or die "Can't write $target: $!\n";
    print $ofh $output;
    close $ofh;
    verbose("wrote $target");
} elsif ($opt_output) {
    open my $ofh, '>', $opt_output or die "Can't write $opt_output: $!\n";
    print $ofh $output;
    close $ofh;
    verbose("wrote $opt_output");
} else {
    (my $target = basename($gc_input)) =~ s/\.gc$/.cir/;
    $target = File::Spec->catfile($gc_dir, $target);
    open my $ofh, '>', $target or die "Can't write $target: $!\n";
    print $ofh $output;
    close $ofh;
    verbose("wrote $target");
}
