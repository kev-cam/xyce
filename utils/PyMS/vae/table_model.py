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

_INTERP1D = """
static inline double interp1d(double x, const double* xbp, int nx, const double* t)
{
    int ix = 0;
    if (x <= xbp[0]) ix = 0;
    else if (x >= xbp[nx-1]) ix = nx - 2;          // clamp/extrapolate flat at the edge
    else { while (ix < nx-2 && xbp[ix+1] < x) ix++; }
    double fx = (x - xbp[ix]) / (xbp[ix+1] - xbp[ix] + 1e-30);
    if (fx < 0) fx = 0; if (fx > 1) fx = 1;
    return t[ix]*(1-fx) + t[ix+1]*fx;
}
"""


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

    cpp = _HEADER.format(name=model_name,
                         desc=f"{n}-point I(Vd) linear-interpolation diode table") + f"""
static const int N_VD = {n};
static const double vd_bp[{n}] = {{
    {', '.join(f'{v:.8e}' for v in vd_bp)}
}};
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
    double Id = interp1d(Vd, vd_bp, N_VD, id_tbl);
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


def compile_table_so(cpp_source: str, output_path: str, cxx: str = None) -> bool:
    """Compile a table-model .so (PyMS ABI). Returns True on success."""
    cxx = cxx or os.environ.get("CXX", "g++")
    src = output_path + ".cpp"
    open(src, "w").write(cpp_source)
    r = subprocess.run([cxx, "-O2", "-shared", "-fPIC", src, "-o", output_path],
                       capture_output=True, text=True)
    return r.returncode == 0
