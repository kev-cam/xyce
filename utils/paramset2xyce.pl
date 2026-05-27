#!/usr/bin/env perl
#
# paramset2xyce.pl — gnucap-style ``paramset NAME UNDERLYING ... endparamset``
# → Xyce ``.SUBCKT`` translator.
#
# Verilog-A's paramset construct (gnucap-flavour as shipped with the
# IHP-Open-PDK SG13G2 PDK) is essentially a partial parameter binding
# on an existing compact model: the paramset declares some instance
# parameters of its own, computes a few localparams from them, then
# names the model parameters it overrides via dotted assignments.
# That maps cleanly onto Xyce's ``.SUBCKT`` with ``PARAMS:`` plus
# inline ``.PARAM`` expressions for the localparams.
#
# Example
# -------
#   paramset Rparasitic sp_resistor
#       parameter real R = 0;
#       parameter real tc1 = 0.00353;
#       .resistance = R * corner_res.res_rpara;
#       .tc = tc1;
#   endparamset
#
# becomes
#
#   .SUBCKT Rparasitic n1 n2 PARAMS: R=0 tc1=0.00353
#   Xinternal n1 n2 sp_resistor resistance={R * res_rpara} tc={tc1}
#   .ENDS Rparasitic
#
# Usage:
#   paramset2xyce.pl <input>.va [<input>.va ...] -o models.cir
#   paramset2xyce.pl --underlying-ports r3_cmc=n1,nc,n2,dt <inputs> -o ...
#
# Notes
# -----
# - Gnucap allows two ``paramset`` declarations with the same name
#   differing only in a range constraint (e.g. ``rsil`` for ``mm_ok=0``
#   vs. ``rsil`` for ``mm_ok=1``). Xyce has no overloading, so only the
#   first occurrence wins. That gives the typical-corner / no-mismatch
#   variant, which matches the gnucap2xyce.pl test invocations.
#
# - Hierarchical names in expressions (``corner_res.foo``) reference the
#   instantiated corner module from the testbench. gnucap2xyce.pl
#   flattens corner localparams into top-level ``.PARAM foo=…``; this
#   converter strips the ``<instance>.`` prefix so the references
#   resolve to the flattened names.
#
# - Underlying-module port lists aren't visible from a paramset file —
#   the paramset just names the module. Built-in defaults for the IHP
#   models are baked in; pass ``--underlying-ports name=p1,p2,...`` to
#   override or extend.
#
# - ``aliasparam x = y;`` is dropped. Xyce ``.SUBCKT`` parameters don't
#   have aliases; the canonical name passes through.
#
# - Range constraints (``from [lo:hi]``) are stripped. Xyce will accept
#   the default value or whatever the caller passes, no range checking.
#
# - ``$rdist_normal(seed, mean, sigma, ...)`` (Monte Carlo) is replaced
#   by its mean argument. Monte Carlo support would need a separate pass.
#

use strict;
use warnings;
use Getopt::Long;
use File::Basename qw(basename);

my $opt_output  = '';
my $opt_library = '';
my $opt_verbose = 0;
my @opt_ports;
GetOptions(
    'o=s'                 => \$opt_output,
    'library=s'           => \$opt_library,
    'v|verbose'           => \$opt_verbose,
    'underlying-ports=s'  => \@opt_ports,
) or usage();
usage() unless @ARGV;

sub usage {
    die "Usage: $0 [-o out.cir] [--library out.pl] "
      . "[--underlying-ports name=p1,p2,...] <input.va>...\n"
      . "  -o out.cir     emit .SUBCKT translations (full wrappers)\n"
      . "  --library X.pl emit a Perl library for gnucap2xyce.pl to\n"
      . "                 inline-expand paramset instances at call sites\n"
}
sub verbose { print STDERR "paramset2xyce: @_\n" if $opt_verbose }

# Underlying-module table. For each known underlying:
#   ports     — port list passed to the .SUBCKT (matches the underlying's
#               VA module decl)
#   letter    — Xyce device letter (R / C / M / Q / D)
#   type      — Xyce .MODEL type token (R / C / NMOS / PMOS / NPN / D / ...)
#   level     — xyceLevelNumber registered by the PyMS auto-loader for
#               this device's .va
#   va_path   — installed wrapper .va, used to classify binding targets
#               as instance- vs model-side parameters
my %UNDERLYING = (
    sp_resistor  => { ports=>[qw(pos neg)],     letter=>'R', type=>'R',    level=>9001,
                      va_path=>'/usr/local/share/xyce/verilog-a/sg13g2/resistor.va' },
    sp_capacitor => { ports=>[qw(pos neg)],     letter=>'C', type=>'C',    level=>9001,
                      va_path=>'/usr/local/share/xyce/verilog-a/sg13g2/capacitor.va' },
    r3_cmc       => { ports=>[qw(n1 nc n2 dt)], letter=>'R', type=>'R',    level=>9002,
                      va_path=>'/usr/local/share/xyce/verilog-a/sg13g2/r3_cmc.va' },
    PSP103VA     => { ports=>[qw(d g s b)],     letter=>'M', type=>'NMOS', level=>103  },
    psp103va     => { ports=>[qw(d g s b)],     letter=>'M', type=>'NMOS', level=>103  },
    PSP103_VA    => { ports=>[qw(d g s b)],     letter=>'M', type=>'NMOS', level=>103  },
    PSP103TVA    => { ports=>[qw(d g s b dt)],  letter=>'M', type=>'NMOS', level=>1031 },
    mosvar       => { ports=>[qw(d g s b)],     letter=>'M', type=>'NMOS', level=>9003 },
    mextram      => { ports=>[qw(c b e s)],     letter=>'Q', type=>'NPN',  level=>504  },
);

# Cache of underlying → { param_name => 'instance'|'model' }. Filled on
# demand by scanning the underlying's .va for ``\`IPR*`` / ``\`IPI*``
# (instance) versus ``\`MPR*`` / ``\`MPI*`` (model) macro calls. Also
# recognises ``(* type="instance" *)`` attributes on plain parameter
# declarations.
my %_param_class_cache;
sub classify_underlying_params {
    my ($underlying) = @_;
    return $_param_class_cache{$underlying} if $_param_class_cache{$underlying};
    my $u = $UNDERLYING{$underlying};
    return {} unless $u && $u->{va_path} && -f $u->{va_path};
    my %map;
    # Read the underlying .va plus any sibling `.include`s in the same
    # directory (the CMC macro idiom puts the parameter macros there).
    my %seen;
    my @queue = ($u->{va_path});
    my $dir = $u->{va_path}; $dir =~ s|/[^/]*$||;
    while (my $p = shift @queue) {
        next if $seen{$p}++;
        open my $fh, '<', $p or next;
        local $/;
        my $t = <$fh>;
        close $fh;
        # Follow `\`include "..."` recursively (same dir only).
        while ($t =~ /`include\s+"([^"]+)"/g) {
            my $inc = "$dir/$1";
            push @queue, $inc if -f $inc && !$seen{$inc};
        }
        # `\`IPRcc(name, ...)` and friends → instance
        while ($t =~ /`IP[RI]\w*\s*\(\s*(\w+)/g) { $map{lc $1} = 'instance' }
        # `\`MPRcc(name, ...)` and friends → model
        while ($t =~ /`MP[RI]\w*\s*\(\s*(\w+)/g) { $map{lc $1} ||= 'model' }
        # Accellera form: ``(* type="instance" *) parameter ... NAME``
        while ($t =~ /\(\*[^*]*type\s*=\s*"instance"[^*]*\*\)\s*parameter\s+(?:\w+\s+)?(\w+)/g) {
            $map{lc $1} = 'instance';
        }
        # Plain ``parameter`` decls without type=instance → model side
        # (for sp_resistor/sp_capacitor the model params are bare).
        while ($t =~ /^\s*parameter\s+(?:\w+\s+)?(\w+)\s*=/mg) {
            $map{lc $1} ||= 'model';
        }
        # Sibling .include files alongside the .va
        opendir my $d, $dir or last;
        for my $f (readdir $d) {
            next unless $f =~ /\.(include|vams|h)$/;
            my $fp = "$dir/$f";
            push @queue, $fp if -f $fp && !$seen{$fp};
        }
        closedir $d;
    }
    $_param_class_cache{$underlying} = \%map;
    return \%map;
}
for my $ov (@opt_ports) {
    if ($ov =~ /^(\w+)\s*=\s*(.+)$/) {
        my ($name, $list) = ($1, $2);
        $UNDERLYING{$name}{ports} = [split /[,\s]+/, $list];
    }
}

# ---------------------------------------------------------------------------
# Comment-aware single-line strip. Matches gnucap2xyce.pl's version.
# ---------------------------------------------------------------------------
sub strip_line_comment {
    my ($s) = @_;
    my $out = '';
    my $in_str = 0;
    my $i = 0;
    while ($i < length $s) {
        my $c = substr($s, $i, 1);
        if ($in_str) {
            $out .= $c;
            if ($c eq '\\' && $i + 1 < length $s) {
                $out .= substr($s, $i + 1, 1);
                $i += 2; next;
            }
            $in_str = 0 if $c eq '"';
        } else {
            if    ($c eq '"') { $in_str = 1; $out .= $c }
            elsif ($c eq '/' && $i + 1 < length $s
                              && substr($s, $i + 1, 1) eq '/') { last }
            else  { $out .= $c }
        }
        $i++;
    }
    $out =~ s/\s+$//;
    return $out;
}

# Strip ``/* ... */`` block comments from a multi-line string (within
# the paramset body where we read across lines).
sub strip_block_comments {
    my ($s) = @_;
    $s =~ s{/\*.*?\*/}{ }gs;
    return $s;
}

# ---------------------------------------------------------------------------
# Expression rewriting: gnucap → Xyce-acceptable.
#
# - Strip ``<inst>.<name>``  → ``<name>``  (hierarchical access into
#   corner modules that we've flattened to top-level .PARAMs).
# - Replace ``$rdist_normal(seed, mean, sigma[, scope])`` → ``mean``
#   (drop the Monte Carlo randomisation; nominal corner stays).
# - Strip outer parentheses if redundant. Otherwise pass through —
#   Xyce expression syntax accepts most C/Verilog operators inside
#   ``{ ... }``.
# ---------------------------------------------------------------------------
sub rewrite_expr {
    my ($e) = @_;
    $e =~ s/^\s+|\s+$//g;
    # $rdist_normal(seed, mean, ...) → mean.
    while ($e =~ /\$rdist_normal\s*\(([^)]*)\)/) {
        my $args = $1;
        my @a = split /\s*,\s*/, $args;
        my $repl = (@a >= 2) ? $a[1] : '0';
        $e =~ s/\$rdist_normal\s*\([^)]*\)/$repl/;
    }
    # ``corner_res.foo`` → ``foo`` (allow $a.b.c too, take the last seg).
    $e =~ s/\b([A-Za-z_]\w*)\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)/$2/g;
    # ``\`define``d backtick references — leave alone, Xyce expression
    # parser doesn't speak ``\`name`` but anything that survives this
    # point is the caller's problem.
    return $e;
}

# ---------------------------------------------------------------------------
# Parse a .va file into a list of paramset hashes. Each hash has:
#   name       — paramset name
#   underlying — name of the wrapped module
#   ports      — port list (from %UNDERLYING_PORTS lookup)
#   params     — list of [name, default]  (instance parameters)
#   localparams — list of [name, expr]
#   bindings   — list of [model_param, expr]  (the ``.X = expr;`` lines)
# ---------------------------------------------------------------------------
sub parse_va {
    my ($path) = @_;
    open my $fh, '<', $path or die "paramset2xyce: cannot read $path: $!\n";
    my $text = do { local $/; <$fh> };
    close $fh;
    $text = strip_block_comments($text);
    # Strip ``// ...`` line comments BEFORE splitting bodies on ``;``.
    # Otherwise a comment trailing one statement extends across the
    # ``;`` boundary and eats the following statement on the next line.
    $text = join("\n", map { strip_line_comment($_) } split /\n/, $text);

    my @sets;
    my @seen;     # dedupe by name; first wins

    while ($text =~ /paramset\s+(\w+)\s+(\w+)\s*(.*?)endparamset/sg) {
        my ($name, $underlying, $body) = ($1, $2, $3);
        if (grep { $_ eq $name } @seen) {
            verbose("dropping duplicate paramset $name (second occurrence)");
            next;
        }
        push @seen, $name;

        my $ul = $UNDERLYING{$underlying} // {};
        my $ps = {
            name => $name, underlying => $underlying,
            ports => $ul->{ports} // [],
            letter => $ul->{letter} // 'X',
            type   => $ul->{type}   // $underlying,
            level  => $ul->{level},
            params => [], localparams => [], bindings => [],
        };

        for my $raw (split /;/, $body) {
            my $line = strip_line_comment($raw);
            $line =~ s/^\s+|\s+$//g;
            next if $line eq '';

            # ``parameter [type] NAME = VAL [from ...]``
            if ($line =~ /^parameter\s+(?:(?:real|integer)\s+)?(\w+)\s*(?:real|integer)?\s*=\s*(.+?)\s*(?:from\s+.+)?$/) {
                push @{$ps->{params}}, [$1, rewrite_expr($2)];
                next;
            }
            # ``aliasparam x = y`` — skip
            if ($line =~ /^aliasparam\b/) {
                next;
            }
            # ``localparam [type] NAME = EXPR``
            if ($line =~ /^localparam\s+(?:(?:real|integer)\s+)?(\w+)\s*=\s*(.+)$/) {
                push @{$ps->{localparams}}, [$1, rewrite_expr($2)];
                next;
            }
            # ``.MODELPARAM = expression``
            if ($line =~ /^\.\s*(\w+)\s*=\s*(.+)$/) {
                push @{$ps->{bindings}}, [lc $1, rewrite_expr($2)];
                next;
            }
            verbose("skipping unrecognised paramset line in $name: $line");
        }
        push @sets, $ps;
    }
    return @sets;
}

# ---------------------------------------------------------------------------
# Finalise a parsed paramset: classify each binding into instance- or
# model-side based on the underlying's .va, compute the thermal-strip
# port lists used by both .SUBCKT and device-card emitters.
# ---------------------------------------------------------------------------
sub finalize_paramset {
    my ($ps) = @_;
    my $class = classify_underlying_params($ps->{underlying});
    my (@inst_b, @model_b);
    for my $b (@{$ps->{bindings}}) {
        my $kind = $class->{lc $b->[0]} // 'instance';
        if ($kind eq 'model') { push @model_b, $b }
        else                  { push @inst_b,  $b }
    }
    $ps->{inst_bindings}  = \@inst_b;
    $ps->{model_bindings} = \@model_b;

    # Thermal-port handling: gnucap testbenches call paramsets with
    # the thermal port omitted; the .SUBCKT mirrors that by dropping
    # the trailing dt/tnode/t port and binding it to 0 on the device
    # card. ``sub_ports`` is the externally-visible port list;
    # ``device_nodes`` is what we pass to the device card (with the
    # missing port replaced by '0').
    my @sub_ports    = @{$ps->{ports}};
    my @device_nodes = @{$ps->{ports}};
    if (@sub_ports && $sub_ports[-1] =~ /^(?:dt|tnode|t)$/) {
        pop @sub_ports;
        $device_nodes[-1] = '0';
    }
    $ps->{sub_ports}    = \@sub_ports;
    $ps->{device_nodes} = \@device_nodes;
    return $ps;
}

# ---------------------------------------------------------------------------
# Emit a Perl library that gnucap2xyce.pl loads via ``do``. The library
# defines our ``$PARAMSETS`` hash keyed by paramset name with all the
# fields the call-site inliner needs.
# ---------------------------------------------------------------------------
sub emit_library {
    my (@paramsets) = @_;
    my @lines;
    push @lines, "# Generated by paramset2xyce.pl --library";
    push @lines, "# " . scalar(@paramsets) . " paramsets";
    push @lines, "our \$PARAMSETS = {";
    for my $ps (@paramsets) {
        push @lines, "  '$ps->{name}' => {";
        push @lines, "    underlying  => '$ps->{underlying}',";
        push @lines, "    letter      => '$ps->{letter}',";
        push @lines, "    type        => '$ps->{type}',";
        push @lines, "    level       => "
                   . (defined $ps->{level} ? $ps->{level} : 'undef') . ",";
        push @lines, "    sub_ports   => [" . _qlist($ps->{sub_ports}) . "],";
        push @lines, "    device_nodes=> [" . _qlist($ps->{device_nodes}) . "],";
        push @lines, "    params      => [";
        push @lines, "      " . _kv_entry($_) for @{$ps->{params}};
        push @lines, "    ],";
        push @lines, "    localparams => [";
        push @lines, "      " . _kv_entry($_) for @{$ps->{localparams}};
        push @lines, "    ],";
        push @lines, "    inst_bindings  => [";
        push @lines, "      " . _kv_entry($_) for @{$ps->{inst_bindings}};
        push @lines, "    ],";
        push @lines, "    model_bindings => [";
        push @lines, "      " . _kv_entry($_) for @{$ps->{model_bindings}};
        push @lines, "    ],";
        push @lines, "  },";
    }
    push @lines, "};";
    push @lines, "1;";   # so ``do`` returns true on success
    return join("\n", @lines) . "\n";
}

sub _qlist {
    my ($list) = @_;
    return join(', ', map { _qstr($_) } @$list);
}
sub _qstr {
    my ($s) = @_;
    $s =~ s/\\/\\\\/g; $s =~ s/'/\\'/g;
    return "'$s'";
}
sub _kv_entry {
    my ($pair) = @_;
    return "[" . _qstr($pair->[0]) . ", " . _qstr($pair->[1]) . "],";
}

# ---------------------------------------------------------------------------
# Emit a single .SUBCKT for a paramset.
# ---------------------------------------------------------------------------
sub emit_subckt {
    my ($ps) = @_;
    my @out;
    if (!@{$ps->{ports}}) {
        push @out, "* WARNING: paramset $ps->{name}: no port list known for "
                 . "underlying $ps->{underlying}; using ``n1 n2`` as a guess.";
    }
    my @sub_ports    = @{$ps->{sub_ports}};
    my @device_nodes = @{$ps->{device_nodes}};

    my $params_str = '';
    if (@{$ps->{params}}) {
        $params_str = ' PARAMS: '
            . join(' ', map { "$_->[0]=$_->[1]" } @{$ps->{params}});
    }
    push @out, sprintf(".SUBCKT %s %s%s",
                       $ps->{name}, join(' ', @sub_ports), $params_str);

    # Localparams as inline .PARAM lines (order matters — earlier
    # locals are visible to later ones).
    for my $lp (@{$ps->{localparams}}) {
        push @out, sprintf(".PARAM %s={%s}", $lp->[0], $lp->[1]);
    }

    my @inst_binds  = map { sprintf("%s={%s}", $_->[0], $_->[1]) }
                      @{$ps->{inst_bindings}};
    my @model_binds = map { sprintf("%s={%s}", $_->[0], $_->[1]) }
                      @{$ps->{model_bindings}};
    if (defined $ps->{level} && $ps->{letter} ne 'X') {
        my $mname = "m_" . $ps->{name};
        my $mcard = sprintf(".MODEL %s %s level=%d",
                            $mname, $ps->{type}, $ps->{level});
        if (@model_binds) {
            # Break long .MODEL lines with ``+`` continuations.
            $mcard .= "\n+ " . join(" ", @model_binds);
        }
        push @out, $mcard;
        my $bind = @inst_binds ? (' ' . join(' ', @inst_binds)) : '';
        push @out, sprintf("%s1 %s %s%s",
                           $ps->{letter}, join(' ', @device_nodes), $mname, $bind);
    } else {
        push @out, "* WARNING: paramset $ps->{name}: no device-letter "
                 . "mapping for underlying $ps->{underlying}; emitting "
                 . "X-call (assumes a .SUBCKT $ps->{underlying} exists).";
        # Without a known device-letter, the instance/model split is
        # meaningless — pass everything as subckt instance params.
        my @all = (@inst_binds, @model_binds);
        my $bind = @all ? (' ' . join(' ', @all)) : '';
        push @out, sprintf("Xinternal %s %s%s",
                           join(' ', @device_nodes), $ps->{underlying}, $bind);
    }
    push @out, ".ENDS $ps->{name}";
    push @out, '';
    return @out;
}

# ---------------------------------------------------------------------------
# Top-level: walk every input, accumulate, emit.
# ---------------------------------------------------------------------------
my @sets;
for my $in (@ARGV) {
    my @s = parse_va($in);
    finalize_paramset($_) for @s;
    push @sets, @s;
    verbose(sprintf("%s: %d paramsets", basename($in), scalar @s));
}

# --library mode: emit a Perl data structure for gnucap2xyce.pl to
# load and use for inline expansion at instance call sites.
if ($opt_library) {
    open my $fh, '>', $opt_library or die "paramset2xyce: $opt_library: $!\n";
    print $fh "# Generated by paramset2xyce.pl --library from "
            . join(', ', map { basename($_) } @ARGV) . "\n";
    print $fh emit_library(@sets);
    close $fh;
    verbose("wrote $opt_library");
}

# -o file.cir or stdout: emit .SUBCKT wrappers (still useful when
# inline expansion isn't applicable, e.g. for hand-written netlists
# that already use the paramset names).
if ($opt_output || !$opt_library) {
    my @all = (
        "* Generated by paramset2xyce.pl from "
        . join(', ', map { basename($_) } @ARGV),
        "* " . scalar(@sets) . " paramsets translated",
        '',
    );
    for my $ps (@sets) {
        push @all, emit_subckt($ps);
    }
    my $text = join("\n", @all) . "\n";
    if ($opt_output) {
        open my $fh, '>', $opt_output or die "paramset2xyce: $opt_output: $!\n";
        print $fh $text;
        close $fh;
        verbose("wrote $opt_output");
    } else {
        print $text;
    }
}
