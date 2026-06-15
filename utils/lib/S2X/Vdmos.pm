package S2X::Vdmos;
#
# Shared power-MOSFET macromodel synthesis for the *2xyce translators.
#
# SIMetrix/LTspice power MOSFETs use NMOS/PMOS LEVEL=17 (e.g. the IRFR420);
# Xyce can't build a LEVEL=17 model card, and its native VDMOS (LEVEL=18) is a
# different academic short-channel model with none of the power-MOS parameters
# (KP/CGDMAX/CGDMIN/...). So both simetrix2xyce.pl (analog) and
# simetrix_cosim.pl (analog-on-top cosim) remap a LEVEL=17 model to the same
# behavioral subckt macromodel synthesised here. The analytical form is the one
# ltspice2xyce.pl uses and that was validated against QSPICE64 gold: smooth
# subthreshold Kp*Ks^2*ln(1+exp(Vov/Ks))^2, standard triode, Lambda CLM, body
# diode, fixed Cgs/Cgd, Rd/Rs/Rg, p-channel folding.
#
use strict;
use warnings;
use Exporter 'import';
our @EXPORT_OK = qw(spice_num vdmos_subckt);

# SPICE number (with engineering suffix) -> plain float, or undef. Matches
# ltspice2xyce.pl's _eng2num so the macromodel is bit-identical on either path:
# micro sign (\xb5 / UTF-8 \xc2\xb5) -> u, the SPICE multipliers incl meg and
# mil(=25.4e-6), and trailing letters after the suffix are ignored.
sub spice_num {
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

# ($name, $params, $pchan) -> ".SUBCKT LTZ_VDMOS_<NAME> d g s ... .ENDS" text.
# $pchan: 1/0 to force polarity; omit (undef) to auto-detect the LTspice
# "pchan" keyword in the params (the simetrix path passes NMOS/PMOS explicitly).
sub vdmos_subckt {
    my ($name, $params, $pchan) = @_;
    my %p; $p{ lc $1 } = $2 while $params =~ /(\w+)\s*=\s*([^\s)]+)/g;
    $pchan = ($params =~ /\bpchan\b/i) ? 1 : 0 unless defined $pchan;
    my $pol = $pchan ? -1 : 1;
    my $num = sub { my ($k, $d) = @_; my $n = spice_num($p{$k}); defined $n ? $n : $d };
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
    my $txt = "* [s2x] VDMOS $name macromodel (" . ($pchan ? 'P' : 'N') . "-channel)\n";
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

1;
