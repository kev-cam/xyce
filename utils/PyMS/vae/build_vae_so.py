#!/usr/bin/env python3
"""JIT per-instance VAE math .so builder.

Invoked by the generated device shell's processParams() (see xyce_device_gen.py)
to build the GiNaC vae_eval/vae_jacobian shared library for THIS instance, with
all model+instance parameters baked in as constants (only node voltages remain
symbolic). Writes <out_so>; exits nonzero on failure.

Usage: build_vae_so.py <va_path> <out_so> <params_file>
  params_file: one 'NAME=value' per line (uppercase names).
"""
import sys, os, subprocess

# Make 'vae' importable whether we're run from the package dir or elsewhere.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))          # .../PyMS  (so `import vae`)
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))

from vae.parser import parse_file
from vae.ginac_emitter import emit_ginac_program


class BuildError(Exception):
    pass


def _run(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise BuildError("%s\n%s" % (cmd, r.stderr[:4000]))
    return r


def main():
    if len(sys.argv) < 4:
        sys.stderr.write("usage: build_vae_so.py <va> <out_so> <params_file>\n")
        sys.exit(2)
    va, out_so, pfile = sys.argv[1], sys.argv[2], sys.argv[3]

    params = {}
    # The device shell serializes ALL params (given + defaulted) as NAME=value,
    # plus a single __GIVEN__=NAME1,NAME2,... line naming ONLY the params the
    # user explicitly supplied (from Xyce's given()). That set drives
    # $param_given() in the emitter; without it every baked default would read
    # as "given" and select the wrong model branch (e.g. nVtm = NVTM = 0).
    given = None  # None => legacy shells with no marker => keys()-based fallback
    try:
        with open(pfile) as f:
            for line in f:
                line = line.strip()
                if not line or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip()
                if k == '__GIVEN__':
                    given = {n.strip().upper() for n in v.split(',') if n.strip()}
                    continue
                try:
                    params[k.upper()] = float(v.strip())
                except ValueError:
                    pass
    except OSError:
        pass

    mod = parse_file(va)

    def _attempt(pvals, tag):
        wd = out_so + ".build"
        os.makedirs(wd, exist_ok=True)
        g_cpp = os.path.join(wd, "ginac.cpp")
        g_bin = os.path.join(wd, "ginac")
        eval_cpp = os.path.join(wd, "eval.cpp")
        wrap_cpp = os.path.join(wd, "wrap.cpp")
        # CONSTCtoK (Celsius->Kelvin): define if the preprocessor didn't expand
        # it (include not found). Harmless when already a literal.
        src = "#define CONSTCtoK 273.15\n" + emit_ginac_program(
            mod, param_values=pvals, given_params=given)
        with open(g_cpp, "w") as f:
            f.write(src)
        # 1. compile GiNaC metaprogram; 2. run it to emit the specialized eval.
        _run("g++ -O2 -std=c++17 -o %s %s -lginac -lcln" % (g_bin, g_cpp))
        _run("%s > %s" % (g_bin, eval_cpp), timeout=900)
        # 3. wrap (VaeState ABI + C exports); compile the loadable .so.
        wrapper = (
            '#include <cmath>\n#include <cstdio>\n#include <cstring>\n'
            'struct VaeState { double V[16]; double Vt; };\n'
            'inline double conjugate(double x){ return x; }\n'
            '#define vae_eval _vae_eval_impl\n'
            '#define vae_jacobian _vae_jacobian_impl\n'
            'static const double temperature = 300.15;\n'
            '#include "%s"\n'
            '#undef vae_eval\n#undef vae_jacobian\n'
            'extern "C" void vae_eval(VaeState* s, double* F, double* Q){ _vae_eval_impl(s,F,Q); }\n'
            'extern "C" void vae_jacobian(VaeState* s, double* dFdV, double* dQdV){ _vae_jacobian_impl(s,dFdV,dQdV); }\n'
            % eval_cpp
        )
        with open(wrap_cpp, "w") as f:
            f.write(wrapper)
        _run("g++ -O2 -std=c++17 -shared -fPIC -o %s %s -lm" % (out_so, wrap_cpp))
        sys.stderr.write("build_vae_so: built %s (%d params, %s)\n" % (out_so, len(pvals), tag))

    try:
        _attempt(params, "full")
    except BuildError as e:
        # Expression-valued .va defaults the device shell couldn't evaluate get
        # passed as 0, which can make GiNaC hit pow(0, -n) / x/0 at codegen.
        # Retry letting the emitter resolve those (drop zero-valued params) so
        # its .va-default resolution applies instead.
        sys.stderr.write("build_vae_so: full build failed (%s); retrying without zero-valued params\n"
                         % str(e).splitlines()[0][:200])
        nz = {k: v for k, v in params.items() if v != 0.0}
        _attempt(nz, "nonzero-only")


if __name__ == "__main__":
    main()
