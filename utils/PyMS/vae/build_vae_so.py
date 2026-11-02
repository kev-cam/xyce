#!/usr/bin/env python3
"""JIT per-instance VAE math .so builder.

Invoked by the generated device shell's processParams() (see xyce_device_gen.py)
to build the GiNaC vae_eval/vae_jacobian shared library for THIS instance, with
all model+instance parameters baked in as constants (only node voltages remain
symbolic). Writes <out_so>; exits nonzero on failure.

Usage: build_vae_so.py <va_path> <out_so> <params_file> [--define NAME[=VALUE]]...
  --define: the command-line macros the device shell was generated with
  (xyce_device_gen.py --define / PYMS_DEFINES); the shell passes them on.
  params_file: one 'NAME=value' per line in Xyce's names (upper case), plus
  the marker lines __GIVEN__=A,B,... (the parameters the netlist supplied),
  __SET__=T,... (the ones the shell set itself, e.g. the nmos/pmos type
  variable), __CALLBACK__=C,... (kept symbolic, fetched at eval time) and
  __SIMPARAM__name=value (the simulator's gmin and tnom, for $simparam).
  Xyce parameter names are case-insensitive, Verilog-A names are not: every
  name is mapped back onto the module's own parameter name before it
  reaches the emitter (see main()).
"""
import sys, os, re, shutil, subprocess

# Make 'vae' importable whether we're run from the package dir or elsewhere.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))          # .../PyMS  (so `import vae`)
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))

from vae.parser import parse_file
from vae.ginac_emitter import emit_ginac_program, EmitError, VAE_STATE_DECL, find_identifier_use
from vae.va_const import ConstExprError, check_param_ranges, evaluate_param_defaults
from vae import dual_ad


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
    defines = []
    module = None
    rest = sys.argv[4:]
    while rest:
        a = rest.pop(0)
        if a == '--define' and rest:
            defines.append(rest.pop(0))
        elif a.startswith('--define='):
            defines.append(a[len('--define='):])
        elif a == '--module' and rest:
            # the module of a file with several (the device shell's own)
            module = rest.pop(0)
        else:
            sys.stderr.write("build_vae_so.py: unknown argument %r\n" % a)
            sys.exit(2)

    params = {}
    # The device shell serializes ALL params (given + defaulted) as NAME=value,
    # plus a single __GIVEN__=NAME1,NAME2,... line naming ONLY the params the
    # user explicitly supplied (from Xyce's given()). That set drives
    # $param_given() in the emitter; without it every baked default would read
    # as "given" and select the wrong model branch (e.g. nVtm = NVTM = 0).
    given = None  # None => legacy shells with no marker => keys()-based fallback
    # Callback (runtime) params: kept symbolic and fetched via _pcb() at eval
    # time, so Xyce .SAMPLING/AGAUSS can vary them (e.g. PSP103 DELVTO for MC
    # mismatch) through ONE .so build. The shell names them with a __CALLBACK__
    # marker; PYMS_CALLBACK_PARAMS (comma-separated) adds to that set for
    # manual/experimental builds. Empty by default => byte-identical to before.
    callback = set()
    # Values the shell established outside the netlist (the nmos/pmos type
    # variable, set in Model::Model from the .model keyword): baked like given
    # values, but not "given" for $param_given().
    set_by_shell = set()
    # __SIMPARAM__name=value lines carry the simulator's $simparam values (gmin,
    # tnom) for the parameter defaults that read them.
    simparams = {}
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
                if k == '__SET__':
                    set_by_shell |= {n.strip().upper() for n in v.split(',') if n.strip()}
                    continue
                if k == '__CALLBACK__':
                    callback |= {n.strip().upper() for n in v.split(',') if n.strip()}
                    continue
                if k == '__DEFINES__':
                    continue  # the cache key's record of the macros; they come as --define
                if k.startswith('__SIMPARAM__'):
                    try:
                        simparams[k[len('__SIMPARAM__'):]] = float(v.strip())
                    except ValueError:
                        pass
                    continue
                v = v.strip()
                if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                    # a string parameter's text (NAME="text"), unescaped
                    params[k.upper()] = v[1:-1].replace('\\"', '"').replace('\\\\', '\\')
                    continue
                try:
                    params[k.upper()] = float(v)
                except ValueError:
                    pass
    except OSError:
        pass
    _env_cb = os.environ.get('PYMS_CALLBACK_PARAMS', '')
    callback |= {n.strip().upper() for n in _env_cb.split(',') if n.strip()}

    mod = parse_file(va, defines=defines, module=module)

    # The shell speaks Xyce's names: it registers every parameter in capitals
    # and the netlist parser folds a card's names the same way, so the file
    # above says R=2000 for the module's ``parameter real r``. Verilog-A names
    # are case-sensitive, and the emitter substitutes the module's own names:
    # an overlay {'R': 2000.0} beside its 'r' left every card and instance
    # value unused (test_hdl_model_card_params.py, 2026-10-10). Map every name
    # back onto the module's parameter it stands for. Array parameters are
    # never registered with Xyce and never appear here.
    by_upper = {}
    for p in mod.params:
        if getattr(p, 'array_size', None):
            continue
        by_upper.setdefault(p.name.upper(), []).append(p.name)

    def module_name(xyce_name):
        names = by_upper.get(xyce_name.upper())
        if not names:
            raise BuildError("%s: '%s' from the device shell's params file %s is not a "
                             "parameter of module %s (a stale device shell in PYMS_CACHE?)"
                             % (va, xyce_name, pfile, mod.name))
        if len(names) > 1:
            raise BuildError("%s: '%s' matches the parameters %s of module %s, which differ "
                             "only in case; Xyce parameter names are case-insensitive"
                             % (va, xyce_name, " and ".join(repr(n) for n in names), mod.name))
        return names[0]

    try:
        # Only the values the netlist supplied and the ones the shell set itself
        # are baked. Every other parameter keeps its .va default, which the
        # emitter evaluates with these values in place -- so a default that
        # depends on an overridden parameter follows the override. The shell's
        # own value for a defaulted parameter is only its reading of the
        # default (0 for anything but a plain number) and must not land on top.
        # A legacy shell without a __GIVEN__ marker cannot tell: bake all.
        baked = params if given is None else {
            k: v for k, v in params.items() if k in given or k in set_by_shell}
        pvals = {module_name(k): v for k, v in baked.items()}
        given_mod = None if given is None else {module_name(k) for k in given}
        # PYMS_CALLBACK_PARAMS applies to every module of a deck: a name this
        # module does not declare is simply not one of its parameters.
        callback_mod = {module_name(k) for k in callback if k.upper() in by_upper}
    except BuildError as e:
        sys.stderr.write("build_vae_so: error: %s\n" % e)
        sys.exit(1)

    def _attempt(pvals, tag):
        # Build in a directory private to this process and install the .so by
        # an atomic rename: a concurrent Xyce run of the same (va, params)
        # identity (a parallel sweep through one PYMS_VAE_CACHE) then sees
        # either no .so or a complete one, never a half-written file or this
        # process's intermediates (the shared ginac/eval.cpp/wrap.cpp of the
        # old <out_so>.build raced the same way). The first process to finish
        # keeps its intermediates as <out_so>.build for inspection.
        wd = "%s.build.%d" % (out_so, os.getpid())
        os.makedirs(wd, exist_ok=True)
        g_cpp = os.path.join(wd, "ginac.cpp")
        g_bin = os.path.join(wd, "ginac")
        eval_cpp = os.path.join(wd, "eval.cpp")
        wrap_cpp = os.path.join(wd, "wrap.cpp")
        # CONSTCtoK (Celsius->Kelvin): define if the preprocessor didn't expand
        # it (include not found). Harmless when already a literal.
        src = "#define CONSTCtoK 273.15\n" + emit_ginac_program(
            mod, param_values=pvals, given_params=given_mod,
            callback_params=callback_mod, simparams=simparams)
        with open(g_cpp, "w") as f:
            f.write(src)
        # 1. compile GiNaC metaprogram; 2. run it to emit the specialized eval.
        # -O0: the metaprogram runs once and, since 1.1.4 (no analytic
        # Jacobian), for milliseconds; its compile was the larger part of a
        # big model's build (L_UTSOI 102: 158 s at -O2, 22 s at -O0, the same
        # eval code byte for byte; BSIM6.1.1 62 s / 12 s, HiCUM L2 33 s / 7 s)
        _run("g++ -O0 -std=c++17 -o %s %s -lginac -lcln" % (g_bin, g_cpp))
        r = subprocess.run("%s > %s" % (g_bin, eval_cpp), shell=True, capture_output=True,
                           text=True, timeout=900)
        # The metaprogram's own warnings (the determinism guard: address-space
        # randomization could not be switched off) reach the user.
        for line in (r.stderr or '').splitlines():
            if line.startswith('PyMS warning'):
                sys.stderr.write(line + '\n')
        if r.returncode == 3:
            # The program refused a construct (a non-linear ddt(), see
            # ginac_emitter's _vae_split_ddt): its message says which, and no
            # retry with other parameter values can help.
            raise EmitError(r.stderr.strip())
        if r.returncode != 0:
            raise BuildError("%s\n%s" % (g_bin, r.stderr[:4000]))
        # 3. wrap (VaeState ABI + C exports); compile the loadable .so.
        # vae_jacobian (1.1.5) is the EXACT Jacobian: forward-mode automatic
        # differentiation of this eval code (vae/dual_ad.py).  The eval is
        # compiled a second time, in namespace _vae_ad, with every double a
        # dual number carrying its derivatives with respect to all the
        # library's unknowns; one dual evaluation gives every column, and it
        # differentiates exactly the value path the eval computes (the same
        # ternaries, run-time loops, held values and regime latches), so the
        # Jacobian is consistent with F/Q by construction -- the reason 1.1.0
        # had moved to difference quotients, whose truncation error (a step
        # of 1e-4*(|V|+1) on the absolute node potential) made AC wrong by up
        # to 9 % at bias and stalled a 45 V rectifier's transient.
        # The state carries the node voltages V and, after Vt, the branch-
        # current unknowns I (vae_n_currents of them, the device shell's
        # internal variables); the Jacobian has one column per unknown,
        # nodes first: (n_nodes + n_currents) columns.
        with open(eval_cpp) as f:
            eval_text = f.read()
        eval_ad_cpp = os.path.join(wd, "eval_ad.cpp")
        so_tmp = os.path.join(wd, "vae.so")

        def _wrap(P):
            with open(eval_ad_cpp, "w") as f:
                f.write(dual_ad.transform(eval_text, P))
            with open(wrap_cpp, "w") as f:
                f.write(
                    '#include <cmath>\n#include <cstdio>\n#include <cstring>\n'
                    + VAE_STATE_DECL + '\n'
                    'inline double conjugate(double x){ return x; }\n'
                    '// Runtime (callback) params: the eval fetches these via _pcb("NAME")\n'
                    '// instead of a baked literal, so the compiler cannot const-fold them\n'
                    '// and Xyce .SAMPLING/AGAUSS reaches them live (no per-sample rebuild).\n'
                    '// The device shell installs _pms_cb via vae_set_param_cb; it returns\n'
                    '// the value for whichever instance is currently being evaluated. Until\n'
                    '// installed (or for an unknown name) we return 0.0, the Verilog-A\n'
                    '// default for a shift param like DELVTO.\n'
                    'static double (*_pms_cb)(const char*) = 0;\n'
                    'extern "C" void vae_set_param_cb(double (*f)(const char*)){ _pms_cb = f; }\n'
                    'static inline double _pcb(const char* n){ return _pms_cb ? _pms_cb(n) : 0.0; }\n'
                    '#define vae_eval _vae_eval_impl\n'
                    '#define vae_jacobian _vae_jacobian_impl\n'
                    '// (1.1.6: no temperature here; the eval code reads the\n'
                    '// instance temperature from the state, VaeState::T)\n'
                    '#include "eval.cpp"\n'
                    '#undef vae_eval\n#undef vae_jacobian\n'
                    'extern "C" void vae_eval(VaeState* s, double* F, double* Q){ _vae_eval_impl(s,F,Q); }\n'
                    + dual_ad.header(P) +
                    '// the eval code on dual numbers\n'
                    'namespace _vae_ad {\n'
                    '#define vae_eval _vae_eval_ad\n'
                    '#include "eval_ad.cpp"\n'
                    '#undef vae_eval\n'
                    '}  // namespace _vae_ad\n'
                    + dual_ad.jacobian_cpp(P))
            _run("g++ -O2 -std=c++17 -shared -fPIC -o %s %s -lm" % (so_tmp, wrap_cpp))

        # The sparse plan (dual_ad.plan: the dependency analysis, the
        # colouring, the variables kept double); should g++ refuse it (a
        # dual assigned to a variable the analysis found free of every
        # unknown), the full plan, exact as well.  PYMS_AD_FULL=1 asks for
        # the full plan (a check of the sparse one: the same Jacobian).
        P = dual_ad.plan(eval_text, full=os.environ.get('PYMS_AD_FULL') == '1')
        try:
            _wrap(P)
        except BuildError as e:
            if P.mode == 'full':
                raise
            sys.stderr.write("build_vae_so: note: the sparse Jacobian plan did not compile (%s); "
                             "building the full one\n" % str(e).splitlines()[-1][:200])
            _wrap(P.full('the sparse plan did not compile'))
        os.replace(so_tmp, out_so)
        try:
            os.rename(wd, out_so + ".build")
        except OSError:
            shutil.rmtree(wd, ignore_errors=True)
        sys.stderr.write("build_vae_so: built %s (%d params, %s)\n" % (out_so, len(pvals), tag))

    # The LRM's value ranges (from / exclude) on this instance's final values:
    # the given ones over the defaults evaluated with them. A value outside
    # its range is an error naming the parameter, the value and the range,
    # never a library built with it (BUGS.md CHANGES 1.1.1).
    # A thermal parameter (its default reads $temperature / $vt, 1.1.6) has
    # its value at run time only: its range is not checked here (BUGS.md A17).
    try:
        _pn = {p.name for p in mod.params}
        check_param_ranges(mod, evaluate_param_defaults(
            mod, {k: v for k, v in pvals.items() if k in _pn}, simparams, thermal=set()),
            simparams)
    except ConstExprError as e:
        sys.stderr.write("build_vae_so: error: %s\n" % e)
        sys.exit(1)

    try:
        try:
            _attempt(pvals, "full")
        except BuildError as e:
            if given is not None:
                raise   # every baked value is one the netlist or the shell set
            # Legacy shell (no __GIVEN__ marker): expression-valued .va defaults
            # it couldn't evaluate get passed as 0, which can make GiNaC hit
            # pow(0, -n) / x/0 at codegen. Retry letting the emitter resolve
            # those (drop zero-valued params) so its .va-default resolution
            # applies instead.
            sys.stderr.write("build_vae_so: full build failed (%s); retrying without zero-valued params\n"
                             % str(e).splitlines()[0][:200])
            nz = {k: v for k, v in pvals.items() if v != 0.0}
            _attempt(nz, "nonzero-only")
    except BuildError as e:
        # A name g++ does not know is a name the module never declared (or a
        # construct PyMS lowers to plain C++ that has no such function): say
        # so in one line with the statement's file and line, not g++'s page.
        m = re.search("error: ['\u2018]([^'\u2019]+)['\u2019] was not declared in this scope", str(e))
        if m:
            name = m.group(1)
            loc = find_identifier_use(mod, name)
            where = "%s: " % loc if loc is not None else "%s: " % va
            sys.stderr.write("build_vae_so: error: %s'%s' is not declared in module %s: not a "
                             "parameter, variable, node, branch or function PyMS supports\n"
                             % (where, name, mod.name))
            sys.exit(1)
        sys.stderr.write("build_vae_so: error: %s did not build:\n%s\n" % (out_so, e))
        sys.exit(1)
    except EmitError as e:
        # A construct the device form cannot hold (ddt() used non-linearly,
        # idt()): the message names file, line and construct.
        sys.stderr.write("build_vae_so: error: %s\n" % e)
        sys.exit(1)
    except ConstExprError as e:
        # A parameter default this instance's values make unevaluable (the
        # forms themselves are checked at .hdl time by xyce_device_gen.py);
        # the shell turns the missing library into an error of the device.
        sys.stderr.write("build_vae_so: error: %s: %s\n" % (va, e))
        sys.exit(1)


if __name__ == "__main__":
    main()
