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

# SPICE number (with engineering suffix) -> plain float, or undef.
sub spice_num {
    my $s = shift;
    return undef unless defined $s
        && $s =~ /^\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*([a-zA-Z]*)\s*$/;
    my ($v, $suf) = ($1, lc $2);
    my $mul = 1;
    if    ($suf =~ /^meg/)          { $mul = 1e6; }
    elsif ($suf =~ /^([fpnumkgt])/) {
        my %M = (f=>1e-15, p=>1e-12, n=>1e-9, u=>1e-6, m=>1e-3, k=>1e3, g=>1e9, t=>1e12);
        $mul = $M{$1};
    }
    return $v * $mul;
}

# ($name, $params, $pchan) -> ".SUBCKT LTZ_VDMOS_<NAME> d g s ... .ENDS" text.
sub vdmos_subckt {
    my ($name, $params, $pchan) = @_;
    my %p; $p{ lc $1 } = $2 while $params =~ /(\w+)\s*=\s*([^\s)]+)/g;
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
    my $txt = "* [s2x] VDMOS $name macromodel (" . ($pchan ? 'P' : 'N') . "-channel, from LEVEL=17)\n";
    $txt .= ".SUBCKT LTZ_VDMOS_$U d g s\n";
    $txt .= sprintf "Rdd d di %.6g\n", $rd;
    $txt .= sprintf "Rgg g gi %.6g\n", $rg;
    $txt .= sprintf "Rss si s %.6g\n", $rs;
    $txt .= "Bch di si I={$S($ich)}\n";
    $txt .= ".model LTZ_VDMOS_${U}_BD D(IS=" . sprintf('%.6g', $is) . " N=" . sprintf('%.6g', $nd) . ")\n";
    $txt .= "Dbd $ba $bk LTZ_VDMOS_${U}_BD\n";
    $txt .= sprintf "Cq_gs gi si %.6g\n", $cgs    if $cgs > 0;
    $txt .= sprintf "Cq_gd gi di %.6g\n", $cgdmin if $cgdmin > 0;
    $txt .= ".ENDS LTZ_VDMOS_$U\n";
    return $txt;
}

1;
