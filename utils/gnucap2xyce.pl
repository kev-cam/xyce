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
GetOptions(
    'o=s'        => \$opt_output,
    'inplace'    => \$opt_inplace,
    'v|verbose'  => \$opt_verbose,
    'keep-gnd'   => \$opt_keep_gnd,
) or usage();

usage() unless @ARGV;
my $gc_input = $ARGV[0];
my $gc_dir   = dirname($gc_input);

sub verbose { print STDERR "gnucap2xyce: @_\n" if $opt_verbose }
sub usage   { die "Usage: $0 [-o out.cir] [--inplace] [-v] <input.gc> [input.va]\n" }

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
    my ($inst, $gnd) = @_;
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
    my ($e, $vsrcs, $isrcs) = @_;
    return undef if $e =~ /^iter\(/i;
    if ($e =~ /^([vi])\(\s*(.+?)\s*\)$/i) {
        my ($kind, $arg) = (lc $1, $2);
        # Strip ``tb.`` prefix — testbench is inlined at top level
        $arg =~ s/^tb\.//;
        if ($kind eq 'v') {
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

    my @out;
    push @out, "* Generated by gnucap2xyce.pl from " . basename($gc_path);
    push @out, "* Source testbench: " . basename($tb_va_path)
        if $tb_va_path;
    push @out, "";

    # Module include files (compact-model wrappers): emit as .HDL so
    # PyMS picks them up. Corner .va files contain only localparam
    # declarations — translate to .PARAM lines for now (TODO: support
    # corner selection via the gc driver's ``moshv_ff corner_moshv();``).
    my %seen_inc;
    for my $inc (@{$gc->{includes}}) {
        next if $seen_inc{$inc}++;
        next if $inc =~ /tb_\w+\.va$/;     # the testbench itself — inlined below
        # Heuristic: corner file → .PARAM emission; module file → .HDL.
        my $base = basename($inc);
        if ($base =~ /^corner/i) {
            push @out, emit_corner_params($inc, $gc);
        } elsif ($inc =~ /\.va$/) {
            push @out, qq{.HDL "$inc"};
        } else {
            push @out, qq{.INCLUDE "$inc"};
        }
    }
    push @out, "";

    # Top-level sweep parameters
    my %seen_param;
    for my $p (@{$gc->{params}}) {
        next if $seen_param{$p->{name}}++;
        push @out, ".PARAM $p->{name}=$p->{val}";
    }

    # Testbench inlining
    my %vsrcs;  # name => 1 (for I() probe rewriting)
    my %isrcs;
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

        # Instance emission. emit_instance stashes the final
        # device-letter-prefixed name on the instance hash so the
        # .PRINT rewriter can target it.
        for my $i (@{$tb->{instances}}) {
            push @out, emit_instance($i, $gnd);
            if ($i->{type} =~ /^vs(?:ource|ine)$/) {
                $vsrcs{$i->{inst}} = $i->{emitted_name} // ('V' . $i->{inst});
            }
            if ($i->{type} eq 'idc') {
                $isrcs{$i->{inst}} = $i->{emitted_name} // ('I' . $i->{inst});
            }
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
            my $t = xlate_probe_expr($x, \%vsrcs, \%isrcs);
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
