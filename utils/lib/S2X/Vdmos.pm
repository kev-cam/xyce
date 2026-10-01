package S2X::Vdmos;
#
# Shared power-MOSFET macromodel synthesis for the *2xyce translators.
#
# SIMetrix/LTspice power MOSFETs use NMOS/PMOS LEVEL=17 (e.g. the IRFR420);
# Xyce can't build a LEVEL=17 model card, and its native VDMOS (LEVEL=18) is a
# different academic short-channel model with none of the power-MOS parameters
# (KP/CGDMAX/CGDMIN/...). So both simetrix2xyce.pl (analog) and
# simetrix_cosim.pl (analog-on-top cosim) remap a LEVEL=17 model to the same
# behavioral subckt macromodel synthesised here: the channel current of the
# LTspice-compatible VDMOS as ngspice implements it (vdmosload.c), body diode,
# fixed Cgs/Cgd, Rd/Rs/Rg, p-channel folding.
#
# Channel (forward, Vds>=0; for Vds<0 Vgd replaces Vgs and the current is
# negated, the device is symmetric):
#   Vgst  = Ks*ln(1+exp((Vgs-Vto-Subshift)/Ks))   smooth overdrive, all regions
#   Vdss  = Vds*Mtriode
#   Betap = Kp*(1+Lambda*Vds)/(1+Theta*max(Vgs-Vto,0))
#   Id    = Betap*Vgst^2/2                  Vgst <= Vdss   (saturation)
#         = Betap*Vdss*(Vgst-Vdss/2)        otherwise      (triode)
# It is continuous with continuous derivatives and Id(Vds=0)=0. (The previous
# form switched to a separate subthreshold term below Vto, which jumped by
# Kp*Ks^2*ln(2)^2 at Vgs=Vto and did not vanish at Vds=0, and took Mtriode as
# an exponent; with default parameters above threshold the two agree.)
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
    my $mt  = $num->('mtriode', 1);
    my $ks  = $num->('ksubthres', 0.1);
    my $th  = $num->('theta', 0);
    my $ssh = $num->('subshift', 0);
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
    # Forward/reverse: Vgs or Vgd, |Vds|, sign of the result (see the header)
    my $vgx  = sprintf 'IF(%s>=0,%s,%s-%s)', $vds, $vgs, $vgs, $vds;
    my $x    = sprintf '(%s-%.6g)', $vgx, $vto + $ssh;
    # Ks*ln(1+exp(x/Ks)) written so that exp() cannot overflow
    my $vgst = sprintf '(max(%s,0)+%.6g*ln(1+exp(-abs(%s)/%.6g)))', $x, $ks, $x, $ks;
    my $vdss = sprintf '(abs(%s)*%.6g)', $vds, $mt;
    my $betap = sprintf '(%.6g*(1+%.6g*%s)', $kp, $lam, $vds;
    $betap .= $th != 0 ? sprintf('/(1+%.6g*max(%s-%.6g,0)))', $th, $vgx, $vto) : ')';
    my $id = sprintf 'IF(%s<=%s,0.5*%s*%s*%s,%s*%s*(%s-0.5*%s))',
                     $vgst, $vdss, $betap, $vgst, $vgst, $betap, $vdss, $vgst, $vdss;
    my $ich = sprintf 'IF(%s>=0,1,-1)*%s', $vds, $id;
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
