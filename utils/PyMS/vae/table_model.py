"""
PyMS table-driven device models -- the general table fallback.

When a Verilog-A device will not converge through the GiNaC-compiled analytical
eval -- e.g. an exponential junction whose `limexp` is mapped to plain `exp`,
losing the current limiting that keeps Newton in range -- PyMS can fall back to a
TABLE-driven model: the device characteristic is sampled into a lookup table and
evaluated by interpolation.  An interpolated table is smooth, BOUNDED, and clamps
at its edges, so it converges where the bare exponential does not.

A table model is just another `.so` implementing the SAME VAE ABI
(`vae_eval` / `vae_jacobian` / `vae_n_nodes`) as the JIT'd regimes (cf.
`vae/pmos_table.cpp`), so it drops straight into the generic Xyce wrapper device.

This module owns the C++ EMISSION only; callers supply the sampled characteristic
(the I-V curve).  Two sources feed it:
  - PyMS itself, sampling the VA model it is compiling (the --merge fallback), or
  - device characterization (ltz/devchar, from LTspice/Xyce/ngspice sweeps).
"""
from __future__ import annotations
import os
import subprocess
from typing import Sequence

_HEADER = """// {name} -- table-driven device model (PyMS/Xyce VAE ABI)
// Auto-generated. {desc}
#include <cstring>
#include <cmath>

struct VaeState {{ double V[16]; double Vt; }};
"""

# Uniform-grid table: O(1) index from (x-x0)/dx -- no breakpoint search. The
# sampler emits a uniform Vd grid, so we never scan. Clamps flat at the edges
# (the bounded behaviour that makes this converge where bare exp does not).
_INTERP1D = """
static inline double interp1d(double x, double x0, double dx, int nx, const double* t)
{
    double fi = (x - x0) / dx;
    int ix = (int)fi;
    if (ix < 0) ix = 0; else if (ix > nx-2) ix = nx-2;
    double fx = fi - (double)ix;
    if (fx < 0) fx = 0; else if (fx > 1) fx = 1;
    return t[ix]*(1.0-fx) + t[ix+1]*fx;
}
"""


def _uniform_grid(vd_bp):
    """(x0, dx); assert the grid is uniform (the sampler guarantees it)."""
    x0 = vd_bp[0]
    dx = (vd_bp[-1] - vd_bp[0]) / (len(vd_bp) - 1)
    dev = max(abs((vd_bp[i] - x0) - i*dx) for i in range(len(vd_bp)))
    if dev > 1e-9 * abs(dx):
        raise ValueError("table_model: non-uniform grid (%.3g) needs the scan interp" % dev)
    return x0, dx


def emit_diode_table_so(model_name: str, vd: Sequence[float], idv: Sequence[float],
                        cjo: float = 0.0, max_points: int = 256) -> str:
    """C++ source for a table-driven DIODE (VAE ABI .so).

    vd, idv : the diode I(Vd) characteristic (Vd = anode - cathode, forward +).
    cjo     : zero-bias junction capacitance (linear Q = cjo*Vd; 0 to omit).

    Nodes: V[0]=anode, V[1]=cathode.  F[0]=+Id, F[1]=-Id.  Jacobian: finite
    differences on the table.  The fallback for exponential diodes (and the
    bfit --merge'd models that contain them).
    """
    pts = sorted(zip(vd, idv))
    if len(pts) > max_points:                       # uniform subsample
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
    n = len(pts)
    vd_bp = [p[0] for p in pts]
    id_tbl = [p[1] for p in pts]
    vd0, vdstep = _uniform_grid(vd_bp)

    cpp = _HEADER.format(name=model_name,
                         desc=f"{n}-point I(Vd) linear-interpolation diode table") + f"""
static const int N_VD = {n};
static const double VD0 = {vd0:.8e}, VDSTEP = {vdstep:.8e};
static const double id_tbl[{n}] = {{
    {', '.join(f'{v:.8e}' for v in id_tbl)}
}};
static const double CJO = {cjo:.6e};
{_INTERP1D}
extern "C" {{

int vae_n_nodes() {{ return 2; }}              // anode, cathode
int vae_n_branches() {{ return 2; }}

void vae_eval(VaeState* s, double* F, double* Q)
{{
    double Vd = s->V[0] - s->V[1];              // anode - cathode
    double Id = interp1d(Vd, VD0, VDSTEP, N_VD, id_tbl);
    F[0] =  Id;                                  // anode
    F[1] = -Id;                                  // cathode
    Q[0] =  CJO * Vd;
    Q[1] = -CJO * Vd;
}}

void vae_jacobian(VaeState* s, double* dFdV, double* dQdV)
{{
    const double dv = 1e-6;
    VaeState sp; double F0[2], Q0[2], Fp[2], Qp[2];
    memset(dFdV, 0, 2*2*sizeof(double));
    memset(dQdV, 0, 2*2*sizeof(double));
    vae_eval(s, F0, Q0);
    for (int j = 0; j < 2; j++) {{
        sp = *s; sp.V[j] += dv;
        vae_eval(&sp, Fp, Qp);
        for (int i = 0; i < 2; i++) {{
            dFdV[i*2 + j] = (Fp[i] - F0[i]) / dv;
            dQdV[i*2 + j] = (Qp[i] - Q0[i]) / dv;
        }}
    }}
}}

}} // extern "C"
"""
    return cpp


def emit_bridge_table_so(model_name: str, vd: Sequence[float], idv: Sequence[float],
                         rbleed: float = None, cjo: float = 0.0,
                         max_points: int = 256) -> str:
    """C++ source for a table-driven full-bridge rectifier (VAE ABI .so).

    The bfit --merge'd bridge (D1:a->p, D2:b->p, D3:n->a, D4:n->b) rendered with
    its four diodes sharing ONE interpolation table instead of inlined exp bodies
    -- the merged-model table fallback.  4 terminals: V[0]=a V[1]=b V[2]=p V[3]=n.
    Optional rbleed (across the AC input a-b) is folded in as the merge does.
    """
    pts = sorted(zip(vd, idv))
    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
    n = len(pts)
    vd_bp = [p[0] for p in pts]
    id_tbl = [p[1] for p in pts]
    vd0, vdstep = _uniform_grid(vd_bp)
    gbleed = (1.0 / rbleed) if rbleed else 0.0

    cpp = _HEADER.format(name=model_name,
                         desc=f"{n}-pt table full-bridge (4 diodes share one I(Vd) table)") + f"""
static const int N_VD = {n};
static const double VD0 = {vd0:.8e}, VDSTEP = {vdstep:.8e};
static const double id_tbl[{n}] = {{
    {', '.join(f'{v:.8e}' for v in id_tbl)}
}};
static const double CJO    = {cjo:.6e};
static const double GBLEED = {gbleed:.6e};   // 1/Rbleed across a-b (0 if none)
{_INTERP1D}
extern "C" {{

int vae_n_nodes() {{ return 4; }}              // a, b, p, n
int vae_n_branches() {{ return 4; }}

void vae_eval(VaeState* s, double* F, double* Q)
{{
    double Va=s->V[0], Vb=s->V[1], Vp=s->V[2], Vn=s->V[3];
    double i1 = interp1d(Va-Vp, VD0, VDSTEP, N_VD, id_tbl);   // D1 a->p
    double i2 = interp1d(Vb-Vp, VD0, VDSTEP, N_VD, id_tbl);   // D2 b->p
    double i3 = interp1d(Vn-Va, VD0, VDSTEP, N_VD, id_tbl);   // D3 n->a
    double i4 = interp1d(Vn-Vb, VD0, VDSTEP, N_VD, id_tbl);   // D4 n->b
    double ir = GBLEED*(Va-Vb);                         // bleed a-b
    F[0] =  i1 - i3 + ir;        // a
    F[1] =  i2 - i4 - ir;        // b
    F[2] = -i1 - i2;             // p (+out)
    F[3] =  i3 + i4;             // n (-out)
    double q1=CJO*(Va-Vp), q2=CJO*(Vb-Vp), q3=CJO*(Vn-Va), q4=CJO*(Vn-Vb);
    Q[0] = q1 - q3; Q[1] = q2 - q4; Q[2] = -q1 - q2; Q[3] = q3 + q4;
}}

void vae_jacobian(VaeState* s, double* dFdV, double* dQdV)
{{
    const double dv = 1e-6;
    VaeState sp; double F0[4], Q0[4], Fp[4], Qp[4];
    memset(dFdV, 0, 4*4*sizeof(double));
    memset(dQdV, 0, 4*4*sizeof(double));
    vae_eval(s, F0, Q0);
    for (int j = 0; j < 4; j++) {{
        sp = *s; sp.V[j] += dv;
        vae_eval(&sp, Fp, Qp);
        for (int i = 0; i < 4; i++) {{
            dFdV[i*4 + j] = (Fp[i] - F0[i]) / dv;
            dQdV[i*4 + j] = (Qp[i] - Q0[i]) / dv;
        }}
    }}
}}

}} // extern "C"
"""
    return cpp


def _smooth_mos_id(vgs, vds, kp, vth, lam, w, l, vtsm=0.05):
    """Level-1 MOSFET id with a softplus-smoothed threshold (no hard cutoff)."""
    import math
    beta = kp * w / l
    vov = vgs - vth
    vovs = vtsm * math.log1p(math.exp(min(vov / vtsm, 30.0)))   # softplus(max(vov,0))
    vd = vds if vds > 0.0 else 0.0
    if vd < vovs:
        return beta * (vovs * vd - 0.5 * vd * vd) * (1.0 + lam * vd)   # triode
    return 0.5 * beta * vovs * vovs * (1.0 + lam * vd)                  # saturation


def emit_diffpair_table_so(model_name: str, kp=150e-6, vth=0.6, lam=0.02,
                           w=20e-6, l=1e-6, cgs=2e-15, cgd=1e-15,
                           vg=(-1.0, 3.5, 46), vd=(-0.5, 3.5, 41)) -> str:
    """Table fall-back .so for a bfit --merge'd diff-pair (6 ports d1,g1,d2,g2,s,b).

    A 2-D MOSFET id(vgs,vds) lookup shared by both halves. The threshold is
    softplus-smoothed so the Jacobian has a NON-ZERO gradient everywhere -- which is
    what lets Xyce's DC operating point settle the high-gain feedback loop, where the
    analytical hard-cutoff square-law (zero gradient below threshold) sticks on the
    degenerate output~0 solution. Loaded into Xyce via VAE_SO_PATH (overrides the
    JIT'd analytical eval of the same .va wrapper). Validated: op-amp follower that
    collapsed analytically (timestep -> 0) converges to 0.10% with this table.
    """
    vg0, vg1, ng = vg
    vd0, vd1, nd = vd
    dvg = (vg1 - vg0) / (ng - 1)
    dvd = (vd1 - vd0) / (nd - 1)
    tbl = [_smooth_mos_id(vg0 + i*dvg, vd0 + j*dvd, kp, vth, lam, w, l)
           for i in range(ng) for j in range(nd)]
    cpp = _HEADER.format(name=model_name,
                         desc=f"{ng}x{nd} 2-D id(vgs,vds) diff-pair table fall-back") + f"""
static const int NG={ng}, ND={nd};
static const double VG0={vg0:.6e}, DVG={dvg:.8e}, VD0={vd0:.6e}, DVD={dvd:.8e};
static const double CGS={cgs:.4e}, CGD={cgd:.4e};
static const double TBL[{ng*nd}] = {{
    {', '.join('%.7e' % v for v in tbl)}
}};
static inline double interp2d(double vgs, double vds) {{
    double fi=(vgs-VG0)/DVG, fj=(vds-VD0)/DVD;
    int i=(int)fi, j=(int)fj;
    if(i<0)i=0; else if(i>NG-2)i=NG-2;
    if(j<0)j=0; else if(j>ND-2)j=ND-2;
    double a=fi-i, b=fj-j; if(a<0)a=0; if(a>1)a=1; if(b<0)b=0; if(b>1)b=1;
    double t00=TBL[i*ND+j], t01=TBL[i*ND+j+1], t10=TBL[(i+1)*ND+j], t11=TBL[(i+1)*ND+j+1];
    return t00*(1-a)*(1-b)+t01*(1-a)*b+t10*a*(1-b)+t11*a*b;
}}
extern "C" {{
int vae_n_nodes() {{ return 6; }}              // d1,g1,d2,g2,s,b
int vae_n_branches() {{ return 6; }}
void vae_eval(VaeState* s, double* F, double* Q) {{
    double Vd1=s->V[0],Vg1=s->V[1],Vd2=s->V[2],Vg2=s->V[3],Vs=s->V[4];
    double id1=interp2d(Vg1-Vs, Vd1-Vs), id2=interp2d(Vg2-Vs, Vd2-Vs);
    for(int k=0;k<6;k++){{ F[k]=0; Q[k]=0; }}
    F[0]=id1; F[2]=id2; F[4]=-(id1+id2);                 // KCL: drains +, shared source -
    Q[1]=CGS*(Vg1-Vs)+CGD*(Vg1-Vd1); Q[0]=-CGD*(Vg1-Vd1);
    Q[3]=CGS*(Vg2-Vs)+CGD*(Vg2-Vd2); Q[2]=-CGD*(Vg2-Vd2);
}}
void vae_jacobian(VaeState* s, double* dF, double* dQ) {{
    const double dv=1e-6; VaeState sp; double F0[6],Q0[6],Fp[6],Qp[6];
    memset(dF,0,36*sizeof(double)); memset(dQ,0,36*sizeof(double));
    vae_eval(s,F0,Q0);
    for(int j=0;j<6;j++){{ sp=*s; sp.V[j]+=dv; vae_eval(&sp,Fp,Qp);
        for(int i=0;i<6;i++){{ dF[i*6+j]=(Fp[i]-F0[i])/dv; dQ[i*6+j]=(Qp[i]-Q0[i])/dv; }} }}
}}
}} // extern "C"
"""
    return cpp


def compile_table_so(cpp_source: str, output_path: str, cxx: str = None) -> bool:
    """Compile a table-model .so (PyMS ABI). Returns True on success."""
    cxx = cxx or os.environ.get("CXX", "g++")
    src = output_path + ".cpp"
    open(src, "w").write(cpp_source)
    r = subprocess.run([cxx, "-O2", "-shared", "-fPIC", src, "-o", output_path],
                       capture_output=True, text=True)
    return r.returncode == 0
