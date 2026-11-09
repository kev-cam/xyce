"""
VAE GiNaC emitter — translates parsed Verilog-A AST into a GiNaC C++ program.

The emitted program:
  1. Declares GiNaC symbols for node voltages and instance parameters
  2. Mirrors the Verilog-A analog block structure using GiNaC expressions
  3. Differentiates contributions w.r.t. all node voltages
  4. Prints the final C++ eval and Jacobian code via print_csrc_double

Elaboration-time parameter values (from the model card) are used to resolve
parameter-dependent conditionals. Only instance parameters (W, L, NF, etc.)
remain as GiNaC symbols for symbolic differentiation.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from .parser import (
    Module, ASTNode, NodeKind, ContribKind, Port, PortDir, Param, Var,
    ParseError
)
from .va_const import evaluate_param_defaults, to_cxx, param_kind, VAE_KOQ, T_SYM, VT_SYM


# ---------------------------------------------------------------------------
# Verilog-A builtins → GiNaC C++ equivalents
# ---------------------------------------------------------------------------

_VA_TO_GINAC = {
    'ln': 'log',
    'limexp': 'exp',   # treat as exp for differentiation; limexp wrapper added to output
    'lexp': 'exp',     # clamped exp — treat as exp for differentiation
    'lln': 'log',      # clamped log — treat as log for differentiation
    'abs': 'abs',
    'pow': 'pow',
    'exp': 'exp',
    'log': 'log',
    'sqrt': 'sqrt',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'tanh': 'tanh',
    'atan': 'atan',
    'atan2': 'atan2',
    'hypot': 'hypot',
    # ddt(q) stays an opaque GiNaC function while a right-hand side is built;
    # _vae_split_ddt (emitted into the program) then moves its linear terms
    # to the charge, so a static term beside it stays a current.
    'ddt': 'vae_ddt',
}

# Functions that need special handling for GiNaC (not direct mappings)
_SPECIAL_FUNCS = {'min', 'max', 'hypsmooth', 'hypmax', 'Tempdep'}

# A ddt() call in translated text (the parser joins tokens with blanks)
# The state a device shell hands its math library (vae_eval / vae_jacobian):
# the node voltages, Vt, the branch-current unknowns. The shell
# (xyce_device_gen) and the library wrapper (build_vae_so) declare it from
# this one text; VAE_STATE_ABI is part of the math-library cache key, so a
# library built for another layout is never loaded.
VAE_STATE_CAP = 64
# Since 1.1.4 the state also carries the simulation time ($abstime), the
# analysis (an: 0 dc/op, 1 tran, 2 ac, 3 noise) and its phase (0 the first
# point -- the operating point the analysis starts from, where
# @(initial_step) runs; 1 the later points of a dc sweep or a transient; 2
# the small-signal sweep of an ac or noise analysis), and the device's held
# state: Hr the values held from the last accepted point (read), Hw the
# values of this evaluation (written; null when this evaluation must not
# write, e.g. a finite-difference perturbation), Hl (phase 2 only) the
# values of the last operating-point evaluation, whose regime the
# small-signal phase keeps.  See GiNaCEmitter.emit.
# Since 1.1.6 the state carries the device temperature T (kelvin): the
# instance's temperature at this evaluation, which the device shell reads
# through the instance's temperature source (a pointer: today the instance's
# Xyce temperature -- the circuit TEMP, .options device temp, .step temp --;
# later a thermal-network node or a self-heating state).  The eval code
# reads $temperature from it and computes $vt = VAE_KOQ * T itself, so
# nothing temperature-dependent is baked into the library or its cache key
# (BUGS.md CHANGES 1.1.6, A17).  Vt stays in the layout (the shell still
# sets it, = VAE_KOQ * T) for the libraries' ABI; the eval no longer reads it.
VAE_STATE_DECL = ('struct VaeState { double V[%d]; double Vt; double I[%d]; double t; '
                  'int an; int phase; const double* Hr; double* Hw; const double* Hl; double T; };'
                  % (VAE_STATE_CAP, VAE_STATE_CAP))
VAE_STATE_ABI = 'VaeState-V%d-Vt-I%d-t-an-ph-H3-T' % (VAE_STATE_CAP, VAE_STATE_CAP)
# k/q of Xyce's own devices (N_DEV_Const.h: CONSTboltz / CONSTQ =
# 1.3806226e-23 / 1.6021918e-19, CONSTKoverQ), the correctly rounded double;
# $vt and $vt(T) use it, so a PyMS device and a built-in agree.  1.1.0-1.1.5
# used 8.617087e-5 (9.5e-9 relative larger); VACASK / openvaf use the 2019
# SI values (1.380649e-23 / 1.602176634e-19 = 8.617333262e-5, 2.86e-5
# relative larger than Xyce's).
# (VAE_KOQ, and the eval code's -- and the metaprogram's GiNaC symbols' --
# names of $temperature and $vt, T_SYM and VT_SYM, live in va_const, which
# the parameter defaults that read them share.)


def sub_thermal(text: str) -> str:
    """$vt(x) -> (VAE_KOQ * (x)) with x up to its own closing parenthesis
    (nested ones too: $vt(T + (1)); 1.1.0-1.1.5 stopped at the first ')'),
    $vt() and $vt -> _vae_vt, $temperature -> _vae_T: the eval code's names
    of kT/q and of the instance's temperature (1.1.0-1.1.5: Vt and
    temperature, which a module variable of that name -- ``real Vt;`` is
    common -- captured)."""
    out, i, n = [], 0, len(text)
    pat = re.compile(r'\$vt\b')
    while i < n:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        j = m.end()
        while j < n and text[j] in ' \t\n':
            j += 1
        if j < n and text[j] == '(':
            depth, k = 0, j
            while k < n:
                if text[k] == '(':
                    depth += 1
                elif text[k] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            arg = text[j + 1:k].strip()
            if k >= n:          # unbalanced: leave the rest to the parser's error
                out.append(text[m.start():])
                break
            out.append(VT_SYM if not arg else '(%s * (%s))' % (VAE_KOQ, sub_thermal(arg)))
            i = k + 1
        else:
            out.append(VT_SYM)
            i = m.end()
    s = ''.join(out)
    return re.sub(r'\$temperature\b', T_SYM, s)

# The analysis contexts a math library may be evaluated in: (an, phase).
# phase 3: the evaluation of a transient's accepted point (the device shell's
# acceptStep), where a cross / above event that was detected fires.
VAE_CONTEXTS = ((0, 0), (0, 1), (1, 0), (1, 1), (1, 3), (2, 0), (2, 2), (3, 0), (3, 2))
_AN_NAME = {0: 'dc', 1: 'tran', 2: 'ac', 3: 'noise'}


def analysis_names(an: int, phase: int) -> frozenset:
    """The names analysis() is true for, and the analyses an
    @(initial_step(...)) list matches, in the context (an, phase) (LRM 4.6.1,
    table "analysis argument"): the operating point an analysis starts from
    is "static" and the analysis itself ("ic" too before a transient); the
    later points of a dc sweep are "dc" and "static"; the transient steps
    "tran".  The small-signal phase (2) of ac / noise is the linearisation
    of the operating point's evaluation: the same code, the same names."""
    base = _AN_NAME[an]
    if phase in (0, 2):
        return frozenset({base, 'static'} | ({'ic'} if an == 1 else set()))
    return frozenset({base} | ({'static'} if an == 0 else set()))   # phases 1 and 3

_VAE_DDT_RE = re.compile(r'\bvae_ddt\s*\(')

# The ground node inside the emitter (a branch end, a short class).  Not a
# Verilog-A identifier: before 1.1.3 it was the string 'gnd', so a port or an
# internal node merely NAMED gnd was taken for ground (a short to it collapsed
# into the reference node).  A net declared ``ground`` never reaches the
# emitter as a name (parser.Parser._apply_grounds).
GND = '<ground>'

# Helpers of integer arithmetic (int_ops.lower_integer_ops writes them into
# the expressions): Python for elaboration-time evaluation.
def _py_idiv(a, b, *_site):
    a, b = float(a), float(b)
    if b == 0.0:
        raise ZeroDivisionError('integer division by zero')
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _py_imod(a, b, *_site):
    return float(a) - float(b) * _py_idiv(a, b)


def _py_rint(x):
    x = float(x)
    import math
    r = math.floor(abs(x) + 0.5)
    return r if x >= 0 else -r


_INT_HELPERS_PY = {'_vae_idiv': _py_idiv, '_vae_imod': _py_imod, '_vae_rint': _py_rint}
# ... and C++ for the generated eval code (doubles holding integer values).
# The run-time messages of the eval code (since 1.1.4): a warning is queued
# (vae_take_message) and an error kept (vae_take_error); the device shell
# reports each after the evaluation with Xyce's UserWarning / UserFatal, the
# device's name and the source file:line in the text.
_MSG_HELPERS_CPP = (
    'static const char* _vae_msgq[32]; static int _vae_msgn = 0; static const char* _vae_err = 0;',
    'static inline void _vae_note(const char* m){ if (_vae_msgn < 32) _vae_msgq[_vae_msgn++] = m; }',
    'static inline void _vae_fatal(const char* m){ if (!_vae_err) _vae_err = m; }',
    'extern "C" const char* vae_take_message(){ return _vae_msgn > 0 ? _vae_msgq[--_vae_msgn] : 0; }',
    'extern "C" const char* vae_take_error(){ const char* m = _vae_err; _vae_err = 0; return m; }',
    'static inline double _vae_loop_over(bool over, const char* where){ if (over) _vae_fatal(where); return over ? 1.0 : 0.0; }',
    # an indicator of a voltage-dependent condition: the regime of the
    # operating point is held for the small-signal phase (2) of an ac / noise
    # analysis, which linearises there and never decides the regime again:
    # it reads the indicator the last operating-point evaluation wrote (Hl,
    # the shell's pending values; the shell commits nothing on the way into
    # phase 2, so Hr still holds what the operating point started from and
    # the phase-2 evaluation is the operating point's evaluation again)
    'static inline double _vae_latch(VaeState* s, int k, bool c){ if (s->phase == 2 && s->Hl) return s->Hl[k]; '
    'double v = c ? 1.0 : 0.0; if (s->Hw && s->phase != 2) s->Hw[k] = v; return v; }',
    'static inline double _vae_held(const VaeState* s, int k, double init){ return s->Hr ? s->Hr[k] : init; }',
    'static inline void _vae_hold(VaeState* s, int k, double v){ if (s->Hw) s->Hw[k] = v; }',
    # cross(e, dir) / above(e): did e cross 0 since the last accepted point
    'static inline double _vae_cross(double prev, double now, int dir){ return ((dir >= 0 && prev < 0.0 && now >= 0.0) '
    '|| (dir <= 0 && prev > 0.0 && now <= 0.0)) ? 1.0 : 0.0; }',
)


def _int_helpers_cpp(n_sites: int) -> tuple:
    """The integer helpers of the eval code.  A division or modulus by a
    zero that only the run time knows (an elaboration-time zero is an
    EmitError) gives 0, and the first time per site a warning names the
    source file:line (site: the index into _vae_div_msg; -1 none).  Before
    1.1.4 the 0 was silent."""
    return (
        'static inline void _vae_div0(int site){ static unsigned char said[%d] = {0}; '
        'int k = (site >= 0 && site < %d) ? site : %d; if (!said[k]) { said[k] = 1; '
        '_vae_note(_vae_div_msg[k]); } }' % (n_sites + 1, n_sites, n_sites),
        'static inline double _vae_idiv(double a,double b,int site=-1){ if (b==0.0) { _vae_div0(site); return 0.0; } return std::trunc(a/b); }',
        'static inline double _vae_imod(double a,double b,int site=-1){ if (b==0.0) { _vae_div0(site); return 0.0; } return a-b*std::trunc(a/b); }',
        'static inline double _vae_rint(double x){ return x>=0.0 ? std::floor(x+0.5) : -std::floor(-x+0.5); }',
    )


# Kept for the regime.py / codegen.py walkers (not on the Xyce route).
_INT_HELPERS_CPP = (
    'static inline double _vae_idiv(double a,double b,int site=-1){ return b==0.0 ? 0.0 : std::trunc(a/b); }',
    'static inline double _vae_imod(double a,double b,int site=-1){ return b==0.0 ? 0.0 : a-b*std::trunc(a/b); }',
    'static inline double _vae_rint(double x){ return x>=0.0 ? std::floor(x+0.5) : -std::floor(-x+0.5); }',
)

# A current probe in Verilog-A expression text: I(a, b), I(br) or the port
# probe I(<p>).  The parser joins tokens with blanks, so blanks are allowed
# anywhere.  Group 1 is a string literal (left alone: "I(p, n)" inside a
# $strobe format is not a probe); 2/4 the port-probe brackets; 3 and 5 the
# names.
_STRING_OR_PROBE_RE = re.compile(
    r'("[^"]*")|\bI\s*\(\s*(<)?\s*(\w+)\s*(>)?\s*(?:,\s*(\w+)\s*)?\)')
# A Verilog-A string literal.
_STRING_LIT_RE = re.compile(r'"(?:[^"\\]|\\.)*"')


class CurrentProbeError(ParseError):
    """A current probe PyMS cannot build (a port current I(<p>)) or a branch
    whose current cannot be an unknown (a switch branch).  The message names
    file, line and construct.  Raised while the device shell is generated,
    before any C++ compile, so Xyce reports it as a netlist error."""


def _ast_walk(node):
    """Every statement node of an analog block, depth first (IF arms too)."""
    if node is None:
        return
    yield node
    for c in getattr(node, 'children', None) or ():
        yield from _ast_walk(c)
    yield from _ast_walk(getattr(node, 'else_body', None))


_ANALOG_OP_RE = re.compile(r'\b(ddt|idt|idtmod)\s*\(')


def check_conditions(module):
    """An analog operator (ddt, idt, ...) inside the condition of an if is
    refused with the statement's file and line before any C++ is written:
    PyMS lowers a condition to plain C++ and there is no ddt there (the
    metaprogram's g++ said "'ddt' was not declared in this scope")."""
    for n in _ast_walk(getattr(module, 'analog_block', None)):
        cond = getattr(n, 'condition', None)
        if cond:
            m = _ANALOG_OP_RE.search(_STRING_LIT_RE.sub('""', cond))
            if m:
                raise EmitError(f'{n.loc}: {m.group(1)}() in the condition "{" ".join(cond.split())}": '
                                f'PyMS does not support an analog operator in a condition '
                                f'(assign it to a variable outside the if, or rewrite the test)')


_ACCESS_RE = re.compile(r'\b([VI])\s*\(\s*(<)?\s*(\w+)\s*>?\s*(?:,\s*(\w+)\s*)?\)')


def check_declarations(module):
    """Every name a statement of the analog block writes or reads as a
    node or branch is declared: the target of a contribution and every
    V(..) / I(..) argument is a port, a node, a ground or a named branch,
    and the target of an assignment is a declared variable (or a parameter:
    the Xyce/ADMS dialect, below).  An undeclared one is a file:line error before any C++ is written; before,
    ``I(p, q) <+ ...`` with q undeclared, ``I(bq) <+ ...`` and ``zz = 1;``
    ran with exit status 0 (the contribution went nowhere, zz became a
    local)."""
    nets = ({p.name for p in module.ports} | set(module.internal_nodes)
            | set(getattr(module, 'grounds', ()) or ()))
    branches = set(module.branch_map)
    variables = {v.name for v in module.variables} | set(getattr(module, 'genvars', ()) or ())
    params = {p.name for p in module.params}

    def net(n, loc, what):
        if n not in nets and n not in branches:
            raise EmitError(f"{loc}: '{n}' in {what} is not declared in module "
                            f"{module.name}: not a port, node, ground or branch")

    # The bodies of the analog functions (LRM 4.7.1): an assignment target is
    # an argument (output or inout; an input is read-only), a local or the
    # function's name; a function accesses no node or branch.  Before 1.1.3
    # the bodies were not parsed and a typo there passed generation.
    for f in getattr(module, 'functions', ()) or ():
        fvars = {v.name for v in f.variables} | {f.name}
        args = dict(f.args)
        for a in args:
            if a not in {v.name for v in f.variables}:
                fvars.add(a)
        for st in _ast_walk(f.body):
            if st.kind == NodeKind.CONTRIB:
                raise EmitError(f"{st.loc}: a contribution in analog function {f.name}: "
                                f"an analog function cannot contribute to a branch")
            if st.kind == NodeKind.ASSIGN and not st.lhs.startswith('_vae_t'):
                if st.lhs not in fvars:
                    raise EmitError(f"{st.loc}: assignment to '{st.lhs}' in analog function {f.name}: "
                                    f"not an argument, a local variable or the function name")
                if args.get(st.lhs) == 'input':
                    raise EmitError(f"{st.loc}: assignment to the input argument '{st.lhs}' of "
                                    f"analog function {f.name}")
            for text in (st.expr, st.condition, (st.loop or {}).get('cond')):
                m = _ACCESS_RE.search(_STRING_LIT_RE.sub('""', text or ''))
                if m:
                    raise EmitError(f"{st.loc}: {m.group(0).strip()} in analog function {f.name}: "
                                    f"an analog function cannot access a node or branch")

    for st in _ast_walk(getattr(module, 'analog_block', None)):
        if st.kind == NodeKind.CONTRIB:
            names = list(st.branch or ())
            what = f"the contribution target ({', '.join(names)})"
            if len(names) == 1:
                net(names[0], st.loc, what)     # a node or a named branch
            else:
                for n in names:
                    if n not in nets:
                        raise EmitError(f"{st.loc}: '{n}' in {what} is not declared in module "
                                        f"{module.name}: not a port, node or ground")
        elif (st.kind == NodeKind.ASSIGN and st.lhs not in variables and st.lhs not in params
              and not st.lhs.startswith('_vae_t')):   # parser._lower_ternaries' temporaries
            # (an assignment to a parameter is the Xyce/ADMS dialect of a
            # hidden type parameter, VBIC 1.3's VBICtype: accepted, BUGS.md C11)
            raise EmitError(f"{st.loc}: assignment to '{st.lhs}': not declared in module "
                            f"{module.name} (an assignment needs a real or integer variable)")
        for text in (st.expr, st.condition, (st.loop or {}).get('cond')):
            for m in _ACCESS_RE.finditer(_STRING_LIT_RE.sub('""', text or '')):
                names = [n for n in (m.group(3), m.group(4)) if n is not None]
                for n in names:
                    net(n, st.loc, f"{m.group(1)}({', '.join(names)})")


def find_identifier_use(module, name):
    """The SourceLoc of the first statement whose text reads `name` as an
    identifier (outside string literals), or None."""
    pat = re.compile(r'(?<![\w$`])' + re.escape(name) + r'\b')
    for n in _ast_walk(getattr(module, 'analog_block', None)):
        for field in ('condition', 'expr', 'text'):
            t = getattr(n, field, None)
            if isinstance(t, str) and pat.search(_STRING_LIT_RE.sub('""', t)):
                return getattr(n, 'loc', None)
    return None


class EmitError(Exception):
    """A construct the emitter cannot build into a Xyce device, named with its
    file and line. Raised while the analog block is walked, so a construct in
    a branch the parameters switch off costs nothing."""


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

class GiNaCEmitter:
    """Emit a GiNaC C++ program from a parsed Verilog-A module.

    At elaboration time, process/model parameters are known constants.
    Only instance parameters (W, L, NFIN, etc.) remain symbolic.

    The emitter resolves all parameter-dependent if/else branches using
    the supplied param_values dict, producing a single branch-free code
    path per unique parameter set.
    """

    def __init__(self, module: Module,
                 param_values: Optional[dict[str, float]] = None,
                 line_directives: Optional[bool] = None,
                 forced_conditions: Optional[dict[int, bool]] = None,
                 forced_nodes: Optional[dict[int, bool]] = None,
                 assume_true: Optional[set[str]] = None,
                 collapse_nodes: bool = True,
                 given_params: Optional[set[str]] = None,
                 callback_params: Optional[set[str]] = None,
                 simparams: Optional[dict[str, float]] = None):
        """
        Args:
            module: Parsed Verilog-A module AST.
            param_values: The parameter values to take as supplied (the netlist's
                          model card + instance values, and what the shell sets),
                          by the module's own (case-sensitive) names. Every other
                          parameter takes its Verilog-A default, evaluated here —
                          only node voltages remain symbolic for differentiation.
            simparams: The simulator's $simparam values (gmin, tnom) for the
                          defaults that read them; None = not known.
            line_directives: None=off, True=#line, False=//line comment.
            forced_conditions: (deprecated) Map of condition index → True/False.
            forced_nodes: Map of id(ast_node) → True/False for regime
                          compilation.  When provided, voltage-dependent
                          conditions are resolved deterministically instead
                          of emitting runtime C++ if/else blocks.
            assume_true: Set of condition substrings that should always
                         resolve to True. E.g. {'DevTemp', 'TEMPMOD'}
                         for models with varying temperature.
        """
        self.mod = module
        self.port_names = [p.name for p in module.ports]
        self.internal_nodes = list(module.internal_nodes)
        self.all_nodes = self.port_names + self.internal_nodes
        self.branch_map = dict(module.branch_map)  # branch_name → node_name
        self.collapse_nodes = collapse_nodes
        self.line_directives = line_directives

        # Build parameter value map: name → numeric value.  The supplied values
        # (the netlist's, by the module's own names: build_vae_so.py passes the
        # given ones and what the shell sets) overlay the Verilog-A defaults,
        # which va_const evaluates in declaration order with the language's
        # semantics (integer division, **, the functions, the ternary, $simparam
        # from simparams) — per instance, so an override of b re-evaluates
        # r = 2000 / b.  A default PyMS cannot evaluate raises ConstExprError
        # naming the parameter; it is never 0.
        self.param_values: dict[str, float] = {}
        given_values: dict[str, float] = dict(param_values) if param_values else {}
        self._simparams = dict(simparams) if simparams else {}
        pnames = {p.name for p in module.params}
        # Thermal parameters (1.1.6): a default that reads $temperature / $vt,
        # or another thermal parameter, the netlist did not give.  Its value is
        # the instance's at run time: never baked, a symbol here, computed
        # from the state's temperature in the eval code's prologue
        # (_thermal_cxx), so the library and its cache key stay free of the
        # temperature.
        self._thermal_params: set[str] = set()
        resolved = evaluate_param_defaults(
            module, {k: v for k, v in given_values.items() if k in pnames}, self._simparams,
            thermal=self._thermal_params)
        # String parameters (LRM 3.4.6) keep their text apart from the
        # numbers: a condition or ternary that compares one ("mode ==
        # "fast"") is resolved at elaboration with Python's string equality
        # (_cond_ns); any other use of one is refused (_refuse_strings).
        self._string_values: dict[str, str] = {}
        for p in module.params:
            if p.name in self._thermal_params:
                continue
            val = resolved.get(p.name)
            if isinstance(val, str):
                self._string_values[p.name] = val
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                self.param_values[p.name] = float(val)
            else:
                self.param_values[p.name] = 0.0   # arrays, as before
        for k, v in given_values.items():
            if k not in pnames:
                self.param_values[k] = v
            elif isinstance(v, str):
                self._string_values[k] = v

        # No instance params — everything is constant at compile time
        self.instance_params: set[str] = set()

        # Callback (runtime) params: kept SYMBOLIC (never baked as a literal), and
        # fetched at eval time through a shell-provided callback `_pcb("NAME")`.
        # A compiler cannot const-fold an opaque external call, so the value stays
        # live — Xyce .SAMPLING/AGAUSS on such a param (e.g. PSP103 DELVTO) flows
        # through a SINGLE .so build without a per-sample rebuild, and the shell's
        # callback returns the value for whichever instance is currently being
        # evaluated (per-device local mismatch). Move them out of the baked
        # param_values and into instance_params so every substitution path leaves
        # them as bare symbols (defined via _pcb in the eval preamble).
        self._callback_params: set[str] = set(callback_params) if callback_params else set()
        for cp in self._callback_params:
            self.instance_params.add(cp)
            self.param_values.pop(cp, None)
        # the thermal parameters' C++ (declaration order = dependency order):
        # the other parameters' values baked, thermal and callback ones by name
        self._thermal_cxx: list = []
        if self._thermal_params:
            from .va_const import parse_param_default, ConstExprError as _CE
            _kinds = {q.name: param_kind(q) for q in module.params}

            def _tref(nm):
                if nm in self._thermal_params or nm in self._callback_params:
                    return nm
                if nm in self._string_values:
                    raise _CE(f"the string parameter {nm}")
                return repr(float(self.param_values.get(nm, 0.0)))
            for q in module.params:
                if q.name not in self._thermal_params:
                    continue
                self.instance_params.add(q.name)
                self.param_values.pop(q.name, None)
                try:
                    code, kind = to_cxx(parse_param_default(q), _kinds, _tref)
                except _CE as e:
                    raise _CE(f"parameter {q.name} of module {module.name}: its default "
                              f"{q.default}: {e}")
                if _kinds.get(q.name) == 'int' and kind != 'int':
                    code = f'_vae_rint({code})'
                self._thermal_cxx.append((q.name, code))

        # Track explicitly given params (for $param_given). The JIT builder
        # bakes ALL params (given + defaulted) into param_values, so
        # param_values.keys() OVER-reports given-ness — e.g. NVTM=0 (a default)
        # would make $param_given(NVTM) true and select `nVtm = NVTM = 0`,
        # collapsing every /nVtm term. When the caller knows the true
        # user-given set (from Xyce's given()), it passes given_params and we
        # use exactly that; otherwise fall back to the legacy keys() behavior.
        self._given_params: set[str] = set()
        if given_params is not None:
            self._given_params = set(given_params)
        elif param_values:
            self._given_params = set(param_values.keys())

        # Variable values resolved during elaboration
        self.var_values: dict[str, float] = {}
        # Track declared variables: name → type ('ex' or 'double')
        self._declared_vars: dict[str, str] = {}
        # All names the module declares as real/integer locals. A declared var
        # referenced with no value on the reached path is 0 in Verilog-A
        # (reals zero-init). Its assignment may have been compile-time skipped
        # (e.g. binning guarded by `if(BULKMOD!=0)`, or `qinv` behind an
        # undefined `ifdef`) — emit 0 rather than leaking a bare identifier the
        # generated C++ can't resolve (K1_i, ETA0R_i, qinv, ...).
        self._declared_var_names: set[str] = {v.name for v in module.variables}
        # Nesting depth of unresolved runtime conditions
        self._runtime_cond_depth: int = 0
        # CSE: track symbol versions for non-constant variables
        self._var_versions: dict[str, int] = {}
        self._var_sym_name: dict[str, str] = {}
        # Variables whose value carries ddt(): name -> translated GiNaC
        # expression text. They get no CSE symbol; the text is inlined where
        # the variable is read, so the ddt() terms reach the contribution and
        # are split off there (a charge cannot hide inside an intermediate).
        self._ddt_vars: dict[str, str] = {}

        # Regime compilation: forced condition outcomes
        self._forced_conditions = forced_conditions  # by index (deprecated)
        self._forced_nodes = forced_nodes  # by id(ast_node)
        self._assume_true = assume_true or set()
        # Registry of voltage-dependent conditions encountered during walk
        self._condition_registry: list[str] = []
        # Voltage-dependent conditional handling via indicator-select.
        # _cond_stack holds metaprogram ex-strings (indicator factors, e.g.
        # "_sel_3" or "(1 - _sel_3)"); an assignment inside is gated by their
        # product. _sel_counter names the runtime 0/1 indicator symbols.
        self._cond_stack: list[str] = []
        self._sel_counter: int = 0
        # Branch-current unknowns (MNA): one per voltage-contribution branch
        # and per branch whose current is probed; see
        # collect_current_unknowns().  A record holds key (the branch
        # identity), a / b (its end nodes after resolution, GND for
        # ground), label ('BR(p,n)' / 'BR(br)': the equation row's label,
        # shared with the device shell), sym (the GiNaC symbol and eval
        # variable), has_V / has_I (contribution kinds emitted on it),
        # probed (its current read in an emitted expression) and loc.
        self._cur_records: list[dict] = []
        self._cur_by_key: dict[tuple, dict] = {}
        # The constant values a variable may hold at this point of the walk
        # whatever the runtime indicators chose (a gated constant assignment
        # adds its value to the previous ones; anything else not constant
        # drops the variable): the start values of a loop counter assigned
        # under a voltage-dependent condition (HiCUM L2's l_it), for
        # _loop_trip_bound.
        self._possible: dict[str, set] = {}

        # --- analysis contexts and held state (1.1.4) ---
        # The context of the walk: (an, phase) (VAE_CONTEXTS) and the names
        # analysis() / @(initial_step(...)) match in it.
        self._ctx = (0, 0)
        self._ctx_names = analysis_names(0, 0)
        # Held state slots, shared by every analysis shadow of the library:
        # key -> index into VaeState Hr / Hw.  Keys: ('var', name) a variable
        # an event statement assigns (its value held from one point to the
        # next: Verilog-A variables are static), ('cross', id, occ) the value
        # of a cross/above expression at the last accepted point, ('ind',
        # ...) the indicator of a voltage-dependent condition (the regime the
        # operating point established, held for the small-signal phase).
        self._held: dict = {}
        self._held_init: dict = {}
        self._state_vars: list = []
        self._state_set: set = set()
        self._state_init: dict = {}
        self._shadows: list = [list(VAE_CONTEXTS)]
        self._ctx_shadow: dict = {c: 0 for c in VAE_CONTEXTS}
        self._event_info: dict = {}
        self._cross_exports: list = []      # (slot, dir) of every cross / above
        # the statement being walked (its key for the held indicators, its
        # location for the run-time messages)
        self._stmt_stack: list = []
        self._stmt_occ: dict = {}
        self._cur_loc = None
        # the enclosing run-time loops (_walk_runtime_loop): their
        # contribution accumulators
        self._rt_loops: list = []
        # sites of an integer / or % whose divisor only the run time knows
        self._div_sites: list = []
        self._div_site_of: dict = {}
        # the indicator keys of the walks that hold an operating-point phase
        # (only those are held for the small-signal phase)
        self._op_keys: set = set()
        self._walk_has_op_phase = True

    def emit(self) -> str:
        """Generate the GiNaC C++ program source.

        Analysis-specific code (1.1.4).  `@(initial_step("ac"))` and
        `@(initial_step("tran"))` imply different analyses, so the library
        holds a SHADOW of the eval code per analysis context that needs one:
        the analog block is walked once per context (an, phase) of
        VAE_CONTEXTS whose event behaviour differs -- which
        @(initial_step(list)) statements run (in the first-point phase of an
        analysis the list names, LRM 4.6.1 names; a bare .op runs neither
        "ac" nor "tran"), whether a cross / above event fires (at an accepted
        point of a transient), the values of analysis() -- and vae_eval
        dispatches on the state's an and phase (no run-time test of the
        analysis inside one body).  A module without such code has one body,
        as before.  The operating point / transient code establishes the
        regime; the small-signal phase of ac / noise evaluates the operating
        point's code again (the same shadow: what initial_step("ac") set at
        the operating point holds) and reads the regime of the operating
        point's last evaluation (_vae_latch), so it linearises there and never
        decides the regime again.  A variable an event statement assigns is
        held from point to point (Verilog-A variables are static): the
        evaluation starts it from the held value, and its value at the end
        is the new held one; a plain @(initial_step) therefore runs once at
        the start of each analysis and its variables keep their values."""
        self._analyse_events()
        return self._emit_ginac_program()

    # --- events, contexts and shadows (1.1.4) ---

    _EVENTS_SUPPORTED = ('initial_step', 'final_step', 'cross', 'above', 'initial_model',
                         'initial_instance', 'noise')
    # the blocks of the compact-modeling dialect (ADMS's partitions) that run
    # where they stand, as every PyMS before 1.1.4 ran them
    _EVENTS_ALWAYS = frozenset(('initial_model', 'initial_instance', 'noise'))
    _ANALYSIS_CALL_RE = re.compile(r'\banalysis\s*\(\s*((?:"[^"]*"\s*,?\s*)+)\)')
    # what an invariant @(initial_step) body may read besides parameters and
    # its own earlier assignments: it then computes the same at every point
    # and runs where it stands in every context (toys/diode2.va).  Not the
    # temperature: $temperature, $vt without an argument and the thermal
    # parameters read the instance's temperature at each evaluation (which a
    # .dc temp sweep -- later a thermal node -- moves between points), so a
    # body that reads them captures the value of the initial step and is
    # held like a bias-dependent one (1.1.7, BUGS.md A18).
    _INVARIANT_OK = frozenset((
        'ln', 'log', 'exp', 'sqrt', 'pow', 'abs', 'min', 'max', 'floor', 'ceil', 'sin', 'cos',
        'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh',
        'atanh', 'hypot', 'limexp', 'lexp', 'lln', 'hypsmooth', 'hypmax', 'Tempdep',
        '_vae_idiv', '_vae_imod', '_vae_rint', '$simparam', '$mfactor',
        '$param_given', '$port_connected', '$strobe', '$display', '$write', '$fstrobe',
        '$fwrite', '$fdisplay', '$warning', '$debug', '$finish', '$stop'))

    def _events_of(self, node) -> list:
        return node.event or [{'kind': 'initial_step', 'args': []}]

    @staticmethod
    def _assigned_vars(node) -> list:
        """The variables assigned anywhere below node, in order (IF arms,
        blocks, loops, event bodies)."""
        out = []
        for n in _ast_walk(node):
            if n.kind == NodeKind.ASSIGN and n.lhs:
                out.append(n.lhs)
        return list(dict.fromkeys(out))

    def _invariant_body(self, node, params: set) -> bool:
        known = set(params) - self._thermal_params
        for st in _ast_walk(node.children[0] if node.children else None):
            if st.kind == NodeKind.CONTRIB or st.kind == NodeKind.INITIAL_STEP:
                return False
            texts = [st.expr, st.condition, (st.loop or {}).get('cond')]
            for text in texts:
                if not text:
                    continue
                s = _STRING_LIT_RE.sub('""', text)
                if re.search(r'\b[VI]\s*\(', s) or re.search(r'\banalysis\s*\(', s):
                    return False
                for m in re.finditer(r"(?<![\w.'`])([$]?[A-Za-z_]\w*)", s):
                    nm = m.group(1)
                    if nm in known or nm in self._INVARIANT_OK:
                        continue
                    # $vt(T) of an invariant argument (scanned on) is invariant
                    if nm == '$vt' and re.match(r'\s*\(', s[m.end():]):
                        continue
                    return False
            if st.kind == NodeKind.ASSIGN and st.lhs:
                known.add(st.lhs)
        return True

    def _analyse_events(self):
        """Classify the event statements, the held variables and the analysis
        shadows (self._shadows: [(contexts)], self._ctx_shadow: context ->
        shadow index) before the walks."""
        block = self.mod.analog_block
        nodes = [n for n in _ast_walk(block) if n.kind == NodeKind.INITIAL_STEP]
        params = {p.name for p in self.mod.params}
        info = {}
        for n in nodes:
            evs = self._events_of(n)
            for e in evs:
                if e['kind'] not in self._EVENTS_SUPPORTED:
                    raise EmitError(f"{self._where(n)}: @({e['kind']}(...)): PyMS supports the events "
                                    f"initial_step, final_step, cross, above (and the blocks "
                                    f"initial_model, initial_instance, noise), not {e['kind']}")
                if e['kind'] in ('initial_step', 'final_step'):
                    for a in e['args']:
                        if not re.fullmatch(r'"[^"]*"', a.strip()):
                            raise EmitError(f"{self._where(n)}: @({e['kind']}({a})): the analysis list of "
                                            f"{e['kind']} must be string literals")
                if e['kind'] in ('cross', 'above') and not e['args']:
                    raise EmitError(f"{self._where(n)}: @({e['kind']}()) without an expression")
            kinds = [e['kind'] for e in evs]
            cand = kinds == ['initial_step'] and not evs[0]['args'] and self._invariant_body(n, params)
            # @(initial_model) / @(initial_instance) (the compact-modeling
            # extension the CMC models use: BSIM-CMG, PSP, HiCUM, Mextram,
            # VBIC, ...) run once per model / instance before any analysis,
            # on parameters: the same values at every point; @(noise) holds
            # the noise sources (0 to the eval code).  They run where they
            # stand, as every PyMS before 1.1.4 ran them.
            always = bool(set(kinds) & self._EVENTS_ALWAYS)
            info[id(n)] = dict(node=n, evs=evs, trivial=cand or always, always=always)
        # an invariant initial_step whose variables are assigned elsewhere too
        # is no longer invariant from point to point: it is held
        changed = True
        while changed:
            changed = False
            inside = set()
            for i in info.values():
                if i['trivial']:
                    for n in _ast_walk(i['node']):
                        inside.add(id(n))
            outside = {n.lhs for n in _ast_walk(block)
                       if n.kind == NodeKind.ASSIGN and n.lhs and id(n) not in inside}
            for i in info.values():
                if i.get('always'):
                    continue
                if i['trivial'] and set(self._assigned_vars(i['node'])) & outside:
                    i['trivial'] = False
                    changed = True
        self._event_info = info
        # held variables: assigned in an event statement that does not run
        # at every point (a final_step body never runs during an analysis)
        held = []
        for i in info.values():
            if i['trivial']:
                continue
            if all(e['kind'] == 'final_step' for e in i['evs']):
                continue
            held.extend(self._assigned_vars(i['node']))
        self._state_vars = [v for v in dict.fromkeys(held) if not v.startswith('_vae_t')]
        self._state_set = set(self._state_vars)
        # a held variable's declaration value (``real cnt = 0;``) is its
        # initial held value
        self._state_init = {}
        for n in _ast_walk(block):
            if (n.kind == NodeKind.ASSIGN and n.decl_init and n.lhs in self._state_set
                    and n.lhs not in self._state_init):
                v = self._try_eval_expr(n.expr)
                if v is None:
                    # (va_const reads every constant form: 1e9, 'h5, ...)
                    try:
                        from .va_const import parse_const_expr, evaluate
                        env = dict(self.param_values)
                        env.update(self._string_values)
                        v = float(evaluate(parse_const_expr(n.expr), env,
                                           (lambda nm: self._simparams.get(nm)),
                                           {p.name for p in self.mod.params}))
                    except Exception:
                        v = None
                if v is None:
                    raise EmitError(f'{self._where(n)}: the declaration value of {n.lhs} (a variable an '
                                    f'event statement assigns, held from point to point) must be a '
                                    f'constant expression')
                self._state_init[n.lhs] = float(v)
        uses_analysis = any(
            re.search(r'\banalysis\s*\(', _STRING_LIT_RE.sub('""', t or ''))
            for n in _ast_walk(block) for t in (n.expr, n.condition, (n.loop or {}).get('cond')))
        sigs, self._shadows, self._ctx_shadow = {}, [], {}
        for ctx in VAE_CONTEXTS:
            parts = [self._event_mode(i, ctx) for i in info.values() if not i['trivial']]
            if uses_analysis:
                parts.append(analysis_names(*ctx))
            sig = tuple(parts)
            if sig not in sigs:
                sigs[sig] = len(self._shadows)
                self._shadows.append([])
            self._shadows[sigs[sig]].append(ctx)
            self._ctx_shadow[ctx] = sigs[sig]

    @staticmethod
    def _initial_step_runs(ev, ctx) -> bool:
        """In the first-point phase of an analysis the list names (and in
        the small-signal phase of ac / noise, which linearises that point's
        evaluation); never at the later points."""
        an, phase = ctx
        if phase in (1, 3):
            return False
        names = [a.strip()[1:-1] for a in ev['args']]
        return not names or bool(set(names) & analysis_names(an, 0))

    def _event_mode(self, info, ctx) -> tuple:
        out = []
        for e in info['evs']:
            k = e['kind']
            if k == 'initial_step':
                out.append('run' if self._initial_step_runs(e, ctx) else 'skip')
            elif k == 'final_step':
                out.append('skip')
            elif k in self._EVENTS_ALWAYS:
                out.append('run')
            elif k == 'cross':
                out.append('detect' if ctx == (1, 3) else 'track')
            else:  # above: at the first point when e >= 0, then on a rising crossing
                out.append('detect' if ctx == (1, 3) else ('above0' if ctx[1] in (0, 2) else 'track'))
        return tuple(out)

    def _sub_analysis(self, text: str) -> str:
        """analysis("name", ...) -> 1 when the walk's context has one of the
        names, else 0 (LRM 4.6.1)."""
        if not text or 'analysis' not in text:
            return text

        def one(m):
            names = re.findall(r'"([^"]*)"', m.group(1))
            return '1' if set(names) & self._ctx_names else '0'
        return self._ANALYSIS_CALL_RE.sub(one, text)

    def _held_slot(self, key, init: float = 0.0) -> int:
        if key not in self._held:
            self._held[key] = len(self._held)
            self._held_init[self._held[key]] = init
        return self._held[key]

    def _ind_key(self):
        """A key of the indicator being allocated that is the same in every
        shadow walk: the statement (its node and the how-many-th time the walk
        enters it) and the indicator's number within the statement."""
        if not self._stmt_stack:
            return None
        top = self._stmt_stack[-1]
        top[2] += 1
        return ('ind', top[0], top[1], top[2])

    # --- Condition evaluation ---

    def _preprocess_sys_funcs(self, cond: str) -> str:
        """Replace $port_connected and $param_given with constants."""
        # $port_connected(x) → 1 (all ports connected, temp is a node)
        cond = re.sub(r'\$port_connected\s*\(\s*\w+\s*\)', '1', cond)
        # $param_given(X) → 1 if X was explicitly supplied, 0 otherwise
        def _pg(m):
            return '1' if m.group(1) in self._given_params else '0'
        cond = re.sub(r'\$param_given\s*\(\s*(\w+)\s*\)', _pg, cond)
        # $simparam("name"[, default]) -> the simulator's value, as in
        # expressions (a condition reached g++ with it: "'$simparam' was not
        # declared in this scope")
        cond = self._sub_simparam(cond)
        # analysis("name", ...): a constant of the walk's analysis context
        cond = self._sub_analysis(cond)
        return self._fold_string_compares(cond)

    def _fold_string_compares(self, text: str) -> str:
        """Fold every compare of a string parameter with a string literal
        (``mode == "fast"``, ``"slow" != mode``) to 1 or 0: the value is
        known at elaboration (the math library is built per parameter set).
        Any other use of a string parameter in a condition is an EmitError
        (a string never reaches the generated C++)."""
        if not self._string_values:
            return text
        names = '|'.join(re.escape(n) for n in sorted(self._string_values, key=len, reverse=True))
        lit = r'"((?:[^"\\]|\\.)*)"'

        def fold(name, op, literal):
            eq = self._string_values[name] == literal.encode().decode('unicode_escape')
            return '1' if (eq if op.replace(' ', '') == '==' else not eq) else '0'
        text = re.sub(r'\b(' + names + r')\s*(=\s*=|!\s*=)\s*' + lit,
                      lambda m: fold(m.group(1), m.group(2), m.group(3)), text)
        text = re.sub(lit + r'\s*(=\s*=|!\s*=)\s*\b(' + names + r')\b',
                      lambda m: fold(m.group(3), m.group(2), m.group(1)), text)
        # two string parameters (mode == kind)
        def fold2(m):
            eq = self._string_values[m.group(1)] == self._string_values[m.group(3)]
            return '1' if (eq if m.group(2).replace(' ', '') == '==' else not eq) else '0'
        text = re.sub(r'\b(' + names + r')\s*(=\s*=|!\s*=)\s*\b(' + names + r')\b', fold2, text)
        left = re.search(r'\b(' + names + r')\b', _STRING_LIT_RE.sub('""', text))
        if left:
            raise EmitError(f'{text.strip()}: the string parameter {left.group(1)} is used other than in '
                            f'== or != against a string literal or another string parameter; PyMS '
                            f'supports only those compares of a string parameter')
        return text

    def _sub_simparam(self, text: str) -> str:
        """``$simparam("name"[, default])`` in analog code: the simulator's
        value when the shell passed one (gmin, tnom: the __SIMPARAM__ lines
        build_vae_so.py reads), else the default, else an EmitError.  (Before
        1.1.1 the default was taken whatever the simulator said, and
        ``$simparam("gmin")`` without one was 0: a silent 0 S conductance.)"""
        def _one(name, default):
            if name in self._simparams:
                return '(' + repr(float(self._simparams[name])) + ')'
            if default is not None:
                return '(' + default.strip() + ')'
            raise EmitError(f'$simparam("{name}") has no value in PyMS (it knows gmin and tnom '
                            f'from Xyce) and no default')
        # A balanced scan of the call (1.1.2 matched the default with a
        # regular expression of one level of parentheses: a default such as
        # max(1.0, min(2.0, 3.0)) was not substituted and reached the C++).
        out, pos = [], 0
        head = re.compile(r'\$simparam\s*\(\s*"([^"]*)"\s*')
        while True:
            m = head.search(text, pos)
            if m is None:
                out.append(text[pos:])
                return ''.join(out)
            i, depth, default = m.end(), 0, None
            if i < len(text) and text[i] == ',':
                j = i + 1
                k, in_str = j, False
                while k < len(text):
                    ch = text[k]
                    if in_str:
                        if ch == '\\':
                            k += 1
                        elif ch == '"':
                            in_str = False
                    elif ch == '"':
                        in_str = True
                    elif ch in '([':
                        depth += 1
                    elif ch in ')]':
                        if depth == 0:
                            break
                        depth -= 1
                    k += 1
                default, i = text[j:k], k
            if i >= len(text) or text[i] != ')':
                raise EmitError(f'{" ".join(text.split())}: a malformed $simparam call')
            out.append(text[pos:m.start()])
            out.append(_one(m.group(1), default))
            pos = i + 1

    def _cond_ns(self, with_vars: bool = True) -> dict:
        """The namespace a compile-time condition is evaluated in: the
        numeric parameter values, the string parameters' texts and (unless
        with_vars is false) the variables known at this point."""
        ns = dict(self.param_values)
        ns.update(self._string_values)
        if with_vars:
            ns.update(self.var_values)
        ns.update(_INT_HELPERS_PY)
        return ns

    def _eval_condition(self, cond: str) -> Optional[bool]:
        """Try to evaluate a condition using known parameter values.

        Returns True/False if fully resolvable, None if it depends on
        instance params or node voltages.
        """
        if cond is None:
            return None

        cond = self._preprocess_sys_funcs(cond)

        # Extract identifiers from condition (not the words of a string literal)
        idents = set(re.findall(r'[A-Za-z_]\w*', _STRING_LIT_RE.sub('""', cond)))
        idents -= {'if', 'else', 'begin', 'end'}
        idents -= set(_INT_HELPERS_PY)

        # Check if all identifiers are known constants (not instance params)
        param_names = {p.name for p in self.mod.params}
        for ident in idents:
            if ident in self.instance_params:
                return None  # Depends on instance param — can't resolve
            if ident in self._string_values:
                continue
            if ident not in self.param_values and ident in param_names:
                return None  # Parameter without known value
            if ident not in param_names and ident not in self.var_values:
                # Could be a variable — check if it's a node voltage
                if ident in self.all_nodes:
                    return None
                # Unknown identifier — might be a local variable
                if ident not in self.var_values:
                    return None

        # Build evaluation namespace
        ns = self._cond_ns()

        # Normalize tokenizer-split operators: = = → ==, ! = → !=, etc.
        py_cond = cond
        py_cond = py_cond.replace('= =', '==')
        py_cond = py_cond.replace('! =', '!=')
        py_cond = py_cond.replace('> =', '>=')
        py_cond = py_cond.replace('< =', '<=')
        py_cond = py_cond.replace('& &', ' and ')
        py_cond = py_cond.replace('| |', ' or ')
        py_cond = py_cond.replace('&&', ' and ')
        py_cond = py_cond.replace('||', ' or ')
        # Standalone ! (not part of !=) → not
        py_cond = re.sub(r'(?<!=)!(?!=)', ' not ', py_cond)

        try:
            result = eval(py_cond, {"__builtins__": {}}, ns)
            return bool(result)
        except Exception:
            return None

    def _eval_condition_with_vars(self, cond: str) -> Optional[bool]:
        """Evaluate a condition using both param_values and var_values.

        Unlike _eval_condition(), this skips the strict identifier pre-checks
        and just attempts evaluation directly. This catches cases where
        computed variables (like Isbs) determine the branch but weren't
        tracked through _eval_condition()'s conservative checks.
        """
        if cond is None:
            return None

        cond = self._preprocess_sys_funcs(cond)

        # If it references node voltages, can't resolve
        if re.search(r'\bV\s*\(', cond) or re.search(r'\bTemp\s*\(', cond):
            return None
        if '$vt' in cond or '$temperature' in cond or '$abstime' in cond or '$realtime' in cond:
            return None

        # Normalize operators
        py_cond = cond
        py_cond = py_cond.replace('= =', '==')
        py_cond = py_cond.replace('! =', '!=')
        py_cond = py_cond.replace('> =', '>=')
        py_cond = py_cond.replace('< =', '<=')
        py_cond = py_cond.replace('& &', ' and ')
        py_cond = py_cond.replace('| |', ' or ')
        py_cond = py_cond.replace('&&', ' and ')
        py_cond = py_cond.replace('||', ' or ')
        py_cond = re.sub(r'(?<!=)!(?!=)', ' not ', py_cond)

        # Build namespace from all known values
        import math
        ns = self._cond_ns()
        ns.update({
            'ln': math.log, 'log': math.log, 'exp': math.exp,
            'sqrt': math.sqrt, 'pow': math.pow, 'abs': abs,
            'min': min, 'max': max, 'hypot': math.hypot,
        })

        try:
            result = eval(py_cond, {"__builtins__": {}}, ns)
            return bool(result)
        except Exception:
            return None

    @staticmethod
    def _strip_balanced_parens(s: str) -> str:
        """Strip outermost balanced parentheses."""
        s = s.strip()
        while s.startswith('(') and s.endswith(')'):
            depth = 0
            for i, ch in enumerate(s):
                if ch == '(': depth += 1
                elif ch == ')': depth -= 1
                if depth == 0 and i < len(s) - 1:
                    return s  # Not a balanced outer pair
            s = s[1:-1].strip()
        return s

    def _condition_to_cpp(self, cond: str) -> str:
        """Translate a VA condition to C++ GiNaC runtime check."""
        c = self._preprocess_sys_funcs(cond)
        c = c.replace('= =', '==').replace('! =', '!=')
        c = c.replace('> =', '>=').replace('< =', '<=')
        c = c.replace('& &', '&&').replace('| |', '||')

        # Compound conditions
        if '||' in c:
            parts = c.split('||')
            return ' || '.join(f'({self._condition_to_cpp(p.strip())})' for p in parts)
        if '&&' in c:
            parts = c.split('&&')
            return ' && '.join(f'({self._condition_to_cpp(p.strip())})' for p in parts)

        c = self._strip_balanced_parens(c)
        neg = False
        if c.startswith('!'):
            neg = True
            c = self._strip_balanced_parens(c[1:].strip())

        # Simple comparison: expr OP value
        for op, func in [('>=', 'cond_ge'), ('<=', 'cond_le'), ('!=', 'cond_ne'),
                         ('==', 'cond_eq'), ('>', 'cond_gt'), ('<', 'cond_lt')]:
            if op in c:
                parts = c.split(op, 1)
                lhs = self._strip_balanced_parens(parts[0])
                rhs = self._strip_balanced_parens(parts[1])
                lhs = self._subst_known_cpp(lhs)
                rhs = self._subst_known_cpp(rhs)
                try:
                    rhs_val = float(eval(rhs, {"__builtins__": {}}, {}))
                    result = f'{func}({lhs}, {rhs_val})'
                except Exception:
                    # If either side is symbolic, assume true (branch-set binning)
                    result = (f'[&](){{ ex _a = ({lhs}); ex _b = ({rhs}); '
                              f'if (!is_a<numeric>(_a) || !is_a<numeric>(_b)) return true; '
                              f'return ex_to<numeric>(_a).to_double() {op} '
                              f'ex_to<numeric>(_b).to_double(); }}()')
                return f'!({result})' if neg else result

        return f'!({self._subst_known_cpp(c)})' if neg else self._subst_known_cpp(c)

    def _subst_known_cpp(self, expr: str) -> str:
        """Substitute known constant params/vars with their values in a C++ expression."""
        def _sub(m):
            name = m.group(0)
            if name in self.param_values:
                return repr(self.param_values[name])
            if name in self.var_values:
                return repr(self.var_values[name])
            if name in self._ddt_vars:
                raise EmitError(
                    f'module {self.mod.name}: the variable {name} carries ddt() '
                    f'and is read in a condition `{expr}`: a time derivative can '
                    f'only enter a contribution linearly; PyMS cannot build this')
            # Use versioned symbol name for non-constant variables
            if name in self._var_sym_name:
                return self._var_sym_name[name]
            # Declared local with no value on this path → 0 (VA zero-init).
            if name in self._declared_var_names:
                return '0.0'
            return name
        return re.sub(r'\b[A-Za-z_]\w*\b', _sub, expr)

    def _cond_final_cpp(self, cond: str, active_nodes: list[str]) -> str:
        """Lower a Verilog-A condition to a C++ boolean over the FINAL-eval
        doubles: node voltages V(a,b) -> (V_a - V_b), current variable symbol
        names, and resolved parameter values. Used as the runtime test of an
        indicator-select. (This is where the guard's V() gets lowered — the old
        _condition_to_cpp path leaked raw V(p,n) into the emitted code.)"""
        c = self._preprocess_sys_funcs(cond)
        for a, b in (('= =', '=='), ('! =', '!='), ('> =', '>='), ('< =', '<='),
                     ('& &', '&&'), ('| |', '||')):
            c = c.replace(a, b)
        # I(a,b) / I(br) -> the branch-current unknown (a double of the eval)
        c = self._lower_probes(c, active_nodes, ginac=False)
        branch_neg = getattr(self.mod, 'branch_neg_map', {}) or {}
        def repl_v(m):
            p1 = m.group(1).strip()
            p2 = m.group(2).strip() if m.group(2) else None
            if p2 is None and p1 in self.branch_map and p1 in branch_neg:
                p2 = branch_neg[p1]
            if p1 in self.branch_map:
                p1 = self.branch_map[p1]
            if p2 and p2 in self.branch_map:
                p2 = self.branch_map[p2]
            if p1 not in active_nodes and p1 in self.internal_nodes:
                p1 = self._resolve_shorted_node(p1, active_nodes)
            if p2 and p2 not in active_nodes and p2 in self.internal_nodes:
                p2 = self._resolve_shorted_node(p2, active_nodes)
            a = '0.0' if p1 is None else f'V_{p1}'
            b = None if not p2 else ('0.0' if p2 is None else f'V_{p2}')
            return f'({a} - {b})' if b is not None else a
        c = re.sub(r'V\s*\(\s*(\w+)\s*(?:,\s*(\w+)\s*)?\)', repl_v, c)
        c = sub_thermal(c)
        c = re.sub(r'\$(abstime|realtime)\b', '_vae_abstime', c)
        # Verilog-A math funcs -> C++ (double) equivalents for the runtime test
        for va, cpp in (('limexp', 'std::exp'), ('lexp', 'std::exp'),
                        ('lln', 'std::log'), ('ln', 'std::log'),
                        ('sqrt', 'std::sqrt'), ('abs', 'std::fabs'),
                        ('min', 'std::fmin'), ('max', 'std::fmax')):
            c = re.sub(r'\b' + va + r'\b', cpp, c)
        # Substitute params/vars (leave V_x, std::*, and already-lowered tokens)
        def _idsub(m):
            nm = m.group(0)
            if nm.startswith('V_') or nm in ('V_', VT_SYM, T_SYM, 'std', '_vae_abstime'):
                return nm
            if nm in self.param_values and nm not in self.instance_params:
                return repr(self.param_values[nm])
            if nm in self.instance_params and nm in self.param_values:
                return repr(self.param_values[nm])
            if nm in self._ddt_vars:
                raise EmitError(
                    f'module {self.mod.name}: the variable {nm} carries ddt() '
                    f'and is read in a condition `{cond}`: a time derivative can '
                    f'only enter a contribution linearly; PyMS cannot build this')
            # The LAST reaching assignment decides: a constant assignment
            # leaves the variable in var_values (and an older symbol in
            # _var_sym_name, which it does not drop), a non-constant one
            # removes it from var_values.  So var_values first, as
            # _subst_known_cpp, _var_current_ex_str and _to_ginac_expr have
            # it.  (Before 1.1.4 the symbol came first: `if (V > -1) k = 5;
            # k = 0; if (k < 1 && ...)` read the stale gated k, x5_stale_if_cond
            # 0.2 for 0.5, and so did the second pass of a loop, which resets
            # its counter: x5_two_vdep_loops_seq, l4_nested_vdep_inner.)
            if nm in self.var_values:
                return repr(self.var_values[nm])
            if nm in self._var_sym_name:
                return self._var_sym_name[nm]
            # Declared local with no value on this path → 0 (VA zero-init).
            if nm in self._declared_var_names:
                return '0.0'
            return nm
        c = re.sub(r'\b[A-Za-z_]\w*\b', _idsub, c)
        return self._add_div_sites(c)

    def _div_site(self, kind: str) -> int:
        loc = self._cur_loc
        where = f'{loc.file}:{loc.line}' if loc is not None and getattr(loc, 'line', 0) else f'module {self.mod.name}'
        key = (where, kind)
        if key not in self._div_site_of:
            self._div_site_of[key] = len(self._div_sites)
            what = 'division' if kind == 'idiv' else 'modulus'
            self._div_sites.append(f'{where}: an integer {what} by zero at run time (Verilog-A '
                                   f'leaves it undefined); PyMS gives 0')
        return self._div_site_of[key]

    _DIV_CALL_RE = re.compile(r'\b(_?vae_(idiv|imod))\s*\(')

    def _add_div_sites(self, text: str) -> str:
        """Give every integer division / modulus call of a translated text
        (``vae_idiv(a, b)`` of the metaprogram, ``_vae_idiv ( a , b )`` of
        the C++) its run-time message site as a third argument: the
        statement's file:line (a constant divisor folds before)."""
        if not text or 'vae_i' not in text:
            return text
        out, pos = [], 0
        for m in list(self._DIV_CALL_RE.finditer(text)):
            if m.start() < pos:
                continue
            depth, j, commas = 1, m.end(), 0
            while j < len(text) and depth:
                ch = text[j]
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif ch == ',' and depth == 1:
                    commas += 1
                j += 1
            if depth or commas != 1:
                continue
            inner = self._add_div_sites(text[m.end():j - 1])
            site = self._div_site(m.group(2))
            arg = f'numeric({site})' if not m.group(1).startswith('_') else str(site)
            out.append(text[pos:m.start()])
            out.append(f'{m.group(1)}({inner}, {arg})')
            pos = j
        out.append(text[pos:])
        return ''.join(out)

    def _alloc_indicator(self, lines: list[str], cond_cpp: str, pfx: str) -> str:
        """Emit a runtime 0/1 indicator symbol for a voltage-dependent condition.
        The symbol is declared in the FINAL eval/jacobian as a raw C++ ternary
        (via int_raw), and participates in GiNaC expressions as an independent
        symbol so forward-AD gives dS/dV = 0 -> the correct branch-selected
        derivative for the selects that reference it."""
        self._sel_counter += 1
        name = f'_sel_{self._sel_counter}'
        # Outside a run-time loop the indicator is held (_vae_latch): the
        # small-signal phase of ac / noise reads the regime of the operating
        # point instead of deciding it again.  Inside a run-time loop it is
        # evaluated afresh every pass.
        key = self._ind_key()
        if self._walk_has_op_phase and key is not None:
            self._op_keys.add(key)
        if key is not None and not self._rt_loops and (self._walk_has_op_phase or key in self._op_keys):
            raw = f'_vae_latch(_vae_s, {self._held_slot(key)}, ({cond_cpp}))'
        else:
            raw = f'({cond_cpp}) ? 1.0 : 0.0'
        esc = raw.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'{pfx}symbol {name}("{name}");')
        lines.append(f'{pfx}int_raw[int_syms.size()] = "{esc}";')
        lines.append(f'{pfx}int_syms.push_back({name});')
        lines.append(f'{pfx}int_exprs.push_back({name});')
        lines.append(f'{pfx}int_names.push_back("{name}");')
        return name

    # GiNaC/C++ identifiers a Verilog-A variable name must not collide with when
    # used verbatim as a symbol (e.g. PSP103 has `real ex, inv_ex, ...`; a bare
    # `symbol ex` collides with the GiNaC `ex` type -> compile errors).
    _GINAC_RESERVED = frozenset({
        'ex', 'symbol', 'numeric', 'lst', 'matrix', 'function', 'relational',
        'I', 'Pi', 'Euler', 'Catalan', 'conjugate', 'N',
        'pow', 'sqrt', 'exp', 'log', 'abs', 'sin', 'cos', 'tan',
        'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh',
        'step', 'csgn', 'double', 'int', 'float', 'return', 'const',
    })

    def _safe_sym(self, name: str) -> str:
        """Mangle a VA variable name that would collide with a GiNaC/C++ identifier."""
        return name + '_va' if name in self._GINAC_RESERVED else name

    def _var_current_ex_str(self, name: str) -> str:
        """Metaprogram ex-string for a variable's current value (its value
        BEFORE a gated reassignment). Precedence matches _subst_known_cpp:
        constant value, then current symbol, then 0 (unset VA var reads as 0)."""
        if name in self._ddt_vars:
            return self._ddt_vars[name]
        if name in self.var_values:
            return repr(self.var_values[name])
        if name in self._var_sym_name:
            return self._var_sym_name[name]
        return 'ex(0)'

    # --- ddt()/idt(): what the F(V) + d/dt Q(V) form of a Xyce device holds ---

    _TIME_OP_RE = re.compile(r'\b(ddt|idt|idtmod)\s*\(')

    def _where(self, node: ASTNode) -> str:
        loc = getattr(node, 'loc', None)
        if loc is not None and getattr(loc, 'line', None):
            return f'{loc.file}:{loc.line}'
        return f'module {self.mod.name}'

    def _expr_carries_ddt(self, expr: str) -> bool:
        """True when expr calls ddt() or reads a variable that carries ddt()."""
        if not expr:
            return False
        if re.search(r'\bddt\s*\(', expr):
            return True
        return any(ident in self._ddt_vars
                   for ident in re.findall(r'\b[A-Za-z_]\w*\b', expr))

    def _check_time_operators(self, expr: str, node: ASTNode, what: str):
        """Refuse, naming file, line and construct, what cannot be built:
        idt()/idtmod() (a time integral needs a state of its own) and ddt()
        under another function (a non-linear use of the derivative). The
        linear uses of ddt() are split off by _vae_split_ddt at the
        contribution."""
        if not expr:
            return
        text = ' '.join(expr.split())
        for m in self._TIME_OP_RE.finditer(expr):
            if m.group(1) != 'ddt':
                raise EmitError(
                    f'{self._where(node)}: {m.group(1)}() in {what} `{text}`: '
                    f'PyMS evaluates a contribution as a static part plus the '
                    f'time derivative of a charge; a time integral has no place '
                    f'in that form (it needs an internal state node) and is '
                    f'not supported')
            fname = self._enclosing_call(expr, m.start())
            if fname:
                raise EmitError(
                    f'{self._where(node)}: ddt() inside {fname}(...) in {what} '
                    f'`{text}`: a time derivative must enter a contribution '
                    f'linearly, as a sum of ddt() terms with constant factors; '
                    f'PyMS cannot build a non-linear use of ddt()')

    @staticmethod
    def _enclosing_call(expr: str, pos: int) -> Optional[str]:
        """The function whose argument list encloses expr[pos], or None when
        only grouping parentheses (or nothing) do."""
        depth = 0
        i = pos - 1
        while i >= 0:
            ch = expr[i]
            if ch == ')':
                depth += 1
            elif ch == '(':
                if depth:
                    depth -= 1
                else:
                    j = i - 1
                    while j >= 0 and expr[j] in ' \t':
                        j -= 1
                    k = j
                    while k >= 0 and (expr[k].isalnum() or expr[k] in '_$`'):
                        k -= 1
                    if k < j:
                        return expr[k + 1:j + 1]
            i -= 1
        return None

    @staticmethod
    def _drop_ddt_tolerance(s: str) -> str:
        """vae_ddt(q, tol) -> vae_ddt(q): the LRM's absolute tolerance is a
        solver hint the device interface has no use for."""
        pos = 0
        while True:
            m = _VAE_DDT_RE.search(s, pos)
            if not m:
                return s
            depth, j, comma = 1, m.end(), -1
            while j < len(s) and depth:
                c = s[j]
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif c == ',' and depth == 1 and comma < 0:
                    comma = j
                j += 1
            if comma >= 0 and depth == 0:
                s = s[:comma] + s[j - 1:]
            pos = m.end()

    def _find_call2(self, expr: str, fname: str):
        """Find an innermost 2-arg fname(A,B) call (args free of nested
        min/max). Returns (start, end, A, B) or None."""
        i = 0
        while True:
            m = re.search(r'\b' + fname + r'\s*\(', expr[i:])
            if not m:
                return None
            popen = i + m.end() - 1
            depth = 0
            args = []
            argstart = popen + 1
            j = popen
            end = None
            while j < len(expr):
                ch = expr[j]
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        args.append(expr[argstart:j])
                        end = j + 1
                        break
                elif ch == ',' and depth == 1:
                    args.append(expr[argstart:j])
                    argstart = j + 1
                j += 1
            if end is None:
                return None
            inner = expr[popen + 1:end - 1]
            if len(args) == 2 and not re.search(r'\b(min|max)\s*\(', inner):
                return (i + m.start(), end, args[0].strip(), args[1].strip())
            i = popen + 1  # search deeper for a truly-innermost call

    def _find_ternary(self, expr: str):
        """Find a top-level (depth-0) ternary  cond ? A : B  in expr; returns
        (cond, A, B) or None. Handles nested ?: when locating the matching ':'."""
        depth = 0
        q = -1
        for i, ch in enumerate(expr):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == '?' and depth == 0:
                q = i
                break
        if q < 0:
            return None
        depth = 0
        nest = 0
        for j in range(q + 1, len(expr)):
            ch = expr[j]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif depth == 0:
                if ch == '?':
                    nest += 1
                elif ch == ':':
                    if nest == 0:
                        return (expr[:q].strip(), expr[q + 1:j].strip(), expr[j + 1:].strip())
                    nest -= 1
        return None

    def _emit_select_temp(self, lines, pfx, base_g, alt_g, cond_cpp, alt_cpp, base_cpp, active_nodes) -> str:
        """Emit a select intermediate  temp = (cond) ? alt : base. The VALUE is a
        C++ ternary (via int_raw) so the non-taken branch is NOT evaluated — this
        preserves domain guards like max(x,tiny) that avoid 1/0, log(<=0), etc.
        (a branchless base+(alt-base)*S would compute inf and hit inf*0 = NaN).
        The int_expr carries the arithmetic form base+(alt-base)*S purely so
        forward-AD produces the branch-selected derivative (S has dS/dV = 0)."""
        if _VAE_DDT_RE.search(base_g) or _VAE_DDT_RE.search(alt_g):
            raise EmitError(
                f'module {self.mod.name}: ddt() inside a min()/max()/ternary '
                f'select `{cond_cpp} ? {alt_cpp} : {base_cpp}`: a time derivative '
                f'must enter a contribution linearly, as a sum of ddt() terms '
                f'with constant factors; PyMS cannot build a selected ddt()')
        S = self._alloc_indicator(lines, cond_cpp, pfx)
        temp = f'_mm_{self._sel_counter}'
        sel_g = f'(({base_g}) + ((({alt_g})) - ({base_g})) * {S})'
        # the value selects by the indicator (the same test, or the regime
        # held for the small-signal phase)
        tern = f'(({S} != 0.0) ? ({alt_cpp}) : ({base_cpp}))'
        esc = tern.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'{pfx}symbol {temp}("{temp}");')
        lines.append(f'{pfx}int_raw[int_syms.size()] = "{esc}";')
        lines.append(f'{pfx}int_syms.push_back({temp});')
        lines.append(f'{pfx}int_exprs.push_back(_safe_ex([&]() -> ex {{ return ({sel_g}); }}));')
        lines.append(f'{pfx}int_names.push_back("{temp}");')
        return temp

    def _hoist_selects(self, expr: str, lines: list[str], active_nodes: list[str], pfx: str) -> str:
        """Lower voltage-dependent min()/max()/ternary in an expression to
        indicator selects, hoisted to intermediate symbols so the surrounding
        expression stays symbolic (autodiff-friendly) while the choice is a
        genuine runtime decision. min(A,B)=B+(A-B)*[A<B], max(A,B)=B+(A-B)*[A>B],
        (cond?A:B)=B+(A-B)*[cond]. One innermost site per pass, rescanning."""
        while True:
            progressed = False
            # min/max anywhere (innermost first), before ternary so a ternary's
            # branch min/max becomes a symbol first.
            for fname, op in (('min', '<'), ('max', '>')):
                found = self._find_call2(expr, fname)
                if not found:
                    continue
                start, end, A, B = found
                Acpp = self._cond_final_cpp(A, active_nodes)
                Bcpp = self._cond_final_cpp(B, active_nodes)
                temp = self._emit_select_temp(
                    lines, pfx,
                    self._to_ginac_expr(B, active_nodes), self._to_ginac_expr(A, active_nodes),
                    f'({Acpp}) {op} ({Bcpp})', Acpp, Bcpp, active_nodes)
                expr = expr[:start] + temp + expr[end:]
                progressed = True
                break
            if progressed:
                continue
            tern = self._find_ternary(expr)
            if tern and not re.search(r'\b(min|max)\s*\(', tern[1] + ' ' + tern[2]):
                cond, A, B = tern
                temp = self._emit_select_temp(
                    lines, pfx,
                    self._to_ginac_expr(B, active_nodes), self._to_ginac_expr(A, active_nodes),
                    self._cond_final_cpp(cond, active_nodes),
                    self._cond_final_cpp(A, active_nodes), self._cond_final_cpp(B, active_nodes),
                    active_nodes)
                expr = temp
                progressed = True
            if not progressed:
                break
        return expr

    def _maybe_hoist(self, expr: str, lines: list[str], active_nodes: list[str], pfx: str) -> str:
        """Hoist min/max/ternary selects only when the expression is
        voltage-dependent (constant cases fold numerically via _try_eval_expr)."""
        if expr and (re.search(r'\b(min|max)\s*\(', expr) or '?' in expr) \
                and self._expr_uses_voltage(expr):
            return self._hoist_selects(expr, lines, active_nodes, pfx)
        return expr

    # --- GiNaC C++ emission ---

    def _emit_ginac_program(self) -> str:
        lines = []
        lines.append('// AUTO-GENERATED by VAE — GiNaC code generator')
        lines.append(f'// Source model: {self.mod.name}')
        lines.append(f'// Resolved {len(self.param_values)} model parameters at elaboration time')
        lines.append(f'// Instance parameters (symbolic): {sorted(self.instance_params)}')
        lines.append('#include <cmath>')
        lines.append('#include <cstdlib>')
        lines.append('#include <ginac/ginac.h>')
        lines.append('#include <iostream>')
        lines.append('#include <sstream>')
        lines.append('#include <string>')
        lines.append('#include <set>')
        lines.append('#include <vector>')
        lines.append('#include <map>')
        lines.append('#include <functional>')
        lines.append('#include <sys/personality.h>')
        lines.append('#include <cstdio>')
        lines.append('#include <cstring>')
        lines.append('#include <cerrno>')
        lines.append('#include <unistd.h>')
        lines.append('using namespace GiNaC;')
        lines.append('using namespace std;')
        lines.append('')
        lines.append('string to_cpp(const ex& e) {')
        lines.append('    ostringstream os;')
        lines.append('    e.print(print_csrc_double(os));')
        lines.append('    return os.str();')
        lines.append('}')
        lines.append('')
        # Helper: min/max for GiNaC expressions (used when args may be symbolic)
        lines.append('// min/max that work with GiNaC ex and plain doubles')
        lines.append('ex _vae_min(const ex& a, const ex& b) {')
        lines.append('    if (is_a<numeric>(a) && is_a<numeric>(b))')
        lines.append('        return ex_to<numeric>(a) < ex_to<numeric>(b) ? a : b;')
        lines.append('    return a;')
        lines.append('}')
        lines.append('ex _vae_max(const ex& a, const ex& b) {')
        lines.append('    if (is_a<numeric>(a) && is_a<numeric>(b))')
        lines.append('        return ex_to<numeric>(a) > ex_to<numeric>(b) ? a : b;')
        lines.append('    return a;')
        lines.append('}')
        lines.append('')
        # hypsmooth(x, c) = 0.5*(x + sqrt(x*x + 4*c*c))
        lines.append('double hypsmooth(double x, double c) {')
        lines.append('    return (x + std::sqrt(x*x + 4*c*c)) / 2;')
        lines.append('}')
        lines.append('ex hypsmooth(const ex& x, const ex& c) {')
        lines.append('    return (x + sqrt(x*x + 4*c*c)) / 2;')
        lines.append('}')
        # hypmax(x, xmin, c) = xmin + 0.5*(x-xmin-c + sqrt((x-xmin-c)^2 - 4*xmin*c))
        lines.append('double hypmax(double x, double xmin, double c) {')
        lines.append('    double d = x - xmin - c;')
        lines.append('    return xmin + (d + std::sqrt(d*d - 4*xmin*c)) / 2;')
        lines.append('}')
        lines.append('ex hypmax(const ex& x, const ex& xmin, const ex& c) {')
        lines.append('    ex d = x - xmin - c;')
        lines.append('    return xmin + (d + sqrt(d*d - 4*xmin*c)) / 2;')
        lines.append('}')
        lines.append('')
        # Tempdep(PARAML, PARAMT, DELTEMP, TEMPMOD) — temp dependence
        lines.append('double Tempdep(double PARAML, double PARAMT, double DELTEMP, double TEMPMOD) {')
        lines.append('    if (TEMPMOD != 0.0)')
        lines.append('        return PARAML + hypmax(PARAMT * DELTEMP, -PARAML, 1e-6);')
        lines.append('    return PARAML * hypsmooth(1 + PARAMT * DELTEMP - 1e-6, 1e-3);')
        lines.append('}')
        lines.append('ex Tempdep(const ex& PARAML, const ex& PARAMT, const ex& DELTEMP, const ex& TEMPMOD) {')
        lines.append('    if (is_a<numeric>(TEMPMOD) && !ex_to<numeric>(TEMPMOD).is_zero())')
        lines.append('        return PARAML + hypmax(PARAMT * DELTEMP, -PARAML, numeric("1e-6"));')
        lines.append('    return PARAML * hypsmooth(1 + PARAMT * DELTEMP - numeric("1e-6"), numeric("1e-3"));')
        lines.append('}')
        lines.append('')
        # Safe wrappers to avoid GiNaC exact-arithmetic pole errors
        lines.append('// Use double-precision numerics to get IEEE behavior (inf, not throw)')
        lines.append('ex N(double v) { return numeric(v); }')
        lines.append('// Build an intermediate expression, tolerating GiNaC pole errors')
        lines.append('// (e.g. IMAX/(idsatsti*LS) when idsatsti is baked to 0 inside a')
        lines.append('// guarded branch): the VALUE comes from the int_raw ternary, and the')
        lines.append('// derivative of a guard-false constant branch is correctly 0.')
        lines.append('ex _safe_ex(const std::function<ex()>& f) { try { return f(); } catch (...) { return ex(0); } }')
        lines.append('')
        # ddt(): kept as an opaque GiNaC function while a right-hand side is
        # built, then split off by _vae_split_ddt. A contribution's expression
        # F + sum_k c_k * ddt(q_k) stamps F as the current and sum_k c_k * q_k
        # as the charge. Only a constant factor c_k can be carried along, since
        # Xyce evaluates a device as F(V) + d/dt Q(V); anything else (ddt under
        # a function or a power, ddt times ddt, a voltage-dependent factor)
        # stops the build with a message naming the contribution.
        lines.append('DECLARE_FUNCTION_1P(vae_ddt)')
        lines.append('REGISTER_FUNCTION(vae_ddt, dummy())')
        lines.append('')
        # Integer arithmetic (int_ops.lower_integer_ops): truncating / and the
        # % with the sign of the first operand on two integer operands, and
        # the rounding of a real assigned to an integer.  A numeric argument
        # folds; a symbolic one stays a function, piecewise constant
        # (derivative 0), printed as the eval code's _vae_idiv / _vae_imod /
        # _vae_rint (_INT_HELPERS_CPP).
        lines.append('static double _vae_trunc_div(double a, double b) { return b == 0.0 ? 0.0 : std::trunc(a / b); }')
        # (the third argument: the run-time message site of the statement)
        lines.append('DECLARE_FUNCTION_3P(vae_idiv)')
        lines.append('DECLARE_FUNCTION_3P(vae_imod)')
        lines.append('DECLARE_FUNCTION_1P(vae_rint)')
        lines.append('static ex vae_idiv_eval(const ex& a, const ex& b, const ex& s) {')
        lines.append('    if (is_a<numeric>(a) && is_a<numeric>(b) && !ex_to<numeric>(b).is_zero()) return numeric(_vae_trunc_div(ex_to<numeric>(a).to_double(), ex_to<numeric>(b).to_double()));')
        lines.append('    return vae_idiv(a, b, s).hold();')
        lines.append('}')
        lines.append('static ex vae_imod_eval(const ex& a, const ex& b, const ex& s) {')
        lines.append('    if (is_a<numeric>(a) && is_a<numeric>(b) && !ex_to<numeric>(b).is_zero()) { double x = ex_to<numeric>(a).to_double(), y = ex_to<numeric>(b).to_double(); return numeric(x - y * std::trunc(x / y)); }')
        lines.append('    return vae_imod(a, b, s).hold();')
        lines.append('}')
        lines.append('static ex vae_rint_eval(const ex& x) {')
        lines.append('    if (is_a<numeric>(x)) { double v = ex_to<numeric>(x).to_double(); return numeric(v >= 0.0 ? std::floor(v + 0.5) : -std::floor(-v + 0.5)); }')
        lines.append('    return vae_rint(x).hold();')
        lines.append('}')
        lines.append('static ex vae_int_deriv3(const ex&, const ex&, const ex&, unsigned) { return 0; }')
        lines.append('static ex vae_int_deriv1(const ex&, unsigned) { return 0; }')
        lines.append('static void vae_idiv_print(const ex& a, const ex& b, const ex& s, const print_context& c) { c.s << "_vae_idiv("; a.print(c); c.s << ","; b.print(c); c.s << ",(int)"; s.print(c); c.s << ")"; }')
        lines.append('static void vae_imod_print(const ex& a, const ex& b, const ex& s, const print_context& c) { c.s << "_vae_imod("; a.print(c); c.s << ","; b.print(c); c.s << ",(int)"; s.print(c); c.s << ")"; }')
        lines.append('static void vae_rint_print(const ex& x, const print_context& c) { c.s << "_vae_rint("; x.print(c); c.s << ")"; }')
        lines.append('REGISTER_FUNCTION(vae_idiv, eval_func(vae_idiv_eval).derivative_func(vae_int_deriv3).print_func<print_csrc_double>(vae_idiv_print))')
        lines.append('REGISTER_FUNCTION(vae_imod, eval_func(vae_imod_eval).derivative_func(vae_int_deriv3).print_func<print_csrc_double>(vae_imod_print))')
        lines.append('REGISTER_FUNCTION(vae_rint, eval_func(vae_rint_eval).derivative_func(vae_int_deriv1).print_func<print_csrc_double>(vae_rint_print))')
        lines.append('')
        lines.append('// The opaque intermediates whose value moves with the voltages (a run-time')
        lines.append('// loop\'s variables, a held or event value): never a constant factor.')
        lines.append('static vector<symbol> _vae_vdep_opaque;')
        lines.append('// The shared branch-label table of the analysis shadows: the row of a label')
        lines.append('// (appended when new), the shadow\'s F/Q grown to hold it.')
        lines.append('static int _vae_find(const vector<string>& v, const string& s) {')
        lines.append('    for (int i = 0; i < (int)v.size(); i++) if (v[i] == s) return i;')
        lines.append('    return -1;')
        lines.append('}')
        lines.append('static int _vae_br(vector<string>& labels, vector<ex>& F, vector<ex>& Q, const string& s) {')
        lines.append('    int i = _vae_find(labels, s);')
        lines.append('    if (i < 0) { i = labels.size(); labels.push_back(s); }')
        lines.append('    while ((int)F.size() <= i) { F.push_back(ex(0)); Q.push_back(ex(0)); }')
        lines.append('    return i;')
        lines.append('}')
        lines.append('// True when no node voltage reaches e, directly or through the CSE')
        lines.append('// intermediates (an indicator symbol, recorded as its own expression,')
        lines.append('// is a constant to forward-AD and counts as one here too).')
        lines.append('static bool _vae_voltage_free(const ex& e, const vector<symbol>& nodes,')
        lines.append('                              const vector<symbol>& int_syms, const vector<ex>& int_exprs,')
        lines.append('                              std::set<string>& seen) {')
        lines.append('    for (auto& n : nodes) if (e.has(n)) return false;')
        lines.append('    for (auto& n : _vae_vdep_opaque) if (e.has(n)) return false;')
        lines.append('    for (size_t i = 0; i < int_syms.size(); i++) {')
        lines.append('        if (!e.has(int_syms[i]) || int_exprs[i].is_equal(int_syms[i])) continue;')
        lines.append('        if (!seen.insert(int_syms[i].get_name()).second) continue;')
        lines.append('        for (size_t j = i; j < int_syms.size(); j++)')
        lines.append('            if (int_syms[j].is_equal(int_syms[i]) && !int_exprs[j].is_equal(int_syms[j])')
        lines.append('                && !_vae_voltage_free(int_exprs[j], nodes, int_syms, int_exprs, seen)) return false;')
        lines.append('    }')
        lines.append('    return true;')
        lines.append('}')
        lines.append('// rhs = F + sum_k c_k * vae_ddt(q_k): F to the current, sum_k c_k * q_k to the charge.')
        lines.append('static void _vae_split_ddt(const ex& rhs, const vector<symbol>& nodes,')
        lines.append('                           const vector<symbol>& int_syms, const vector<ex>& int_exprs,')
        lines.append('                           ex& F, ex& Q, const char* where) {')
        lines.append('    const ex pat = vae_ddt(wild());')
        lines.append('    ex e = rhs.expand();')
        lines.append('    vector<ex> terms;')
        lines.append('    if (is_a<GiNaC::add>(e)) for (size_t i = 0; i < e.nops(); i++) terms.push_back(e.op(i));')
        lines.append('    else terms.push_back(e);')
        lines.append('    for (auto& t : terms) {')
        lines.append('        if (!t.has(pat)) { F += t; continue; }')
        lines.append('        vector<ex> factors;')
        lines.append('        if (is_a<GiNaC::mul>(t)) for (size_t i = 0; i < t.nops(); i++) factors.push_back(t.op(i));')
        lines.append('        else factors.push_back(t);')
        lines.append('        ex coef = 1, arg; int n_ddt = 0; bool nonlinear = false;')
        lines.append('        for (auto& f : factors) {')
        lines.append('            if (!f.has(pat)) { coef *= f; continue; }')
        lines.append('            if (is_a<GiNaC::function>(f) && ex_to<GiNaC::function>(f).get_name() == "vae_ddt" && !f.op(0).has(pat)) { n_ddt++; arg = f.op(0); }')
        lines.append('            else nonlinear = true;')
        lines.append('        }')
        lines.append('        if (nonlinear || n_ddt != 1) {')
        lines.append('            cerr << "PyMS: " << where << ": the term " << t << " uses ddt() non-linearly (under a function or a power, in a quotient, or ddt times ddt); a time derivative must enter a contribution as a sum of ddt() terms with constant factors" << endl;')
        lines.append('            exit(3);')
        lines.append('        }')
        lines.append('        std::set<string> seen;')
        lines.append('        if (!_vae_voltage_free(coef, nodes, int_syms, int_exprs, seen)) {')
        lines.append('            cerr << "PyMS: " << where << ": the factor " << coef << " of ddt(" << arg << ") depends on a node voltage or a branch current; Xyce evaluates a device as F(x) + d/dt Q(x), so the factor of ddt() must be constant (write ddt(" << coef << " * (" << arg << ")) if that is what is meant)" << endl;')
        lines.append('            exit(3);')
        lines.append('        }')
        lines.append('        Q += coef * arg;')
        lines.append('    }')
        lines.append('}')
        lines.append('')
        # Condition helpers for runtime checks on GiNaC expressions
        lines.append('// Runtime condition checks — if numeric, compare; if symbolic, return default')
        # Runtime condition helpers for GiNaC symbolic expressions.
        # When the value is numeric, evaluate directly.
        # When symbolic, default to false — this selects the "normal" regime
        # (Vds >= 0, no source-drain swap) for MOSFET models.
        # For full regime support, use forced_conditions to override.
        lines.append('bool cond_gt(const ex& a, double b) {')
        lines.append('    if (is_a<numeric>(a)) return ex_to<numeric>(a).to_double() > b;')
        lines.append('    return false;')
        lines.append('}')
        lines.append('bool cond_lt(const ex& a, double b) {')
        lines.append('    if (is_a<numeric>(a)) return ex_to<numeric>(a).to_double() < b;')
        lines.append('    return false;')
        lines.append('}')
        lines.append('bool cond_ge(const ex& a, double b) {')
        lines.append('    if (is_a<numeric>(a)) return ex_to<numeric>(a).to_double() >= b;')
        lines.append('    return false;')
        lines.append('}')
        lines.append('bool cond_le(const ex& a, double b) {')
        lines.append('    if (is_a<numeric>(a)) return ex_to<numeric>(a).to_double() <= b;')
        lines.append('    return false;')
        lines.append('}')
        lines.append('bool cond_eq(const ex& a, double b) {')
        lines.append('    if (is_a<numeric>(a)) return ex_to<numeric>(a).to_double() == b;')
        lines.append('    return false;')
        lines.append('}')
        lines.append('bool cond_ne(const ex& a, double b) {')
        lines.append('    if (is_a<numeric>(a)) return ex_to<numeric>(a).to_double() != b;')
        lines.append('    return true;')
        lines.append('}')
        lines.append('')
        lines.append('int main(int argc, char** argv) {')
        # Deterministic output (BUGS.md D3): GiNaC orders the terms of a sum
        # or product by hash values seeded from the ADDRESS of each class's
        # type name (hash_seed.h), so with address-space randomization every
        # run of the same program printed its eval code in another order (the
        # same values, summed in another order: Mextram bjtd504 converged in
        # 2 runs of 4).  The program re-executes itself once with
        # randomization off; then the order is the same in every run.
        # Since 1.1.4 the marker of the re-execution is the process id (execv
        # keeps it), so a PYMS_GINAC_NOASLR the environment brings along is
        # not taken for one and the program re-executes anyway; and when the
        # randomization cannot be switched off (personality() fails, or is
        # ignored) the program says so -- "PyMS warning: ... term order" on
        # stderr, which build_vae_so.py passes on -- instead of going on
        # silently nondeterministic.
        lines.append('    {')
        lines.append('        char _me[32]; snprintf(_me, sizeof(_me), "%ld", (long)getpid());')
        lines.append('        const char* _mk = getenv("PYMS_GINAC_NOASLR");')
        lines.append('        bool _reexec = _mk && !strcmp(_mk, _me);')
        lines.append('        int pers = personality(0xffffffff);')
        lines.append('        if (pers == -1) {')
        lines.append('            fprintf(stderr, "PyMS warning: personality() failed (%s): address-space randomization stays on, so GiNaC\'s term order -- the summation order of the eval code -- may differ from run to run\\n", strerror(errno));')
        lines.append('        } else if (!(pers & ADDR_NO_RANDOMIZE)) {')
        lines.append('            if (_reexec) {')
        lines.append('                fprintf(stderr, "PyMS warning: the system ignored ADDR_NO_RANDOMIZE (personality): GiNaC\'s term order -- the summation order of the eval code -- may differ from run to run\\n");')
        lines.append('            } else if (personality(pers | ADDR_NO_RANDOMIZE) == -1) {')
        lines.append('                fprintf(stderr, "PyMS warning: personality(ADDR_NO_RANDOMIZE) failed (%s): GiNaC\'s term order -- the summation order of the eval code -- may differ from run to run\\n", strerror(errno));')
        lines.append('            } else {')
        lines.append('                setenv("PYMS_GINAC_NOASLR", _me, 1);')
        lines.append('                execv("/proc/self/exe", argv);')
        lines.append('                fprintf(stderr, "PyMS warning: re-executing with ADDR_NO_RANDOMIZE failed (%s): GiNaC\'s term order -- the summation order of the eval code -- may differ from run to run\\n", strerror(errno));')
        lines.append('            }')
        lines.append('        }')
        lines.append('    }')

        # Declare symbols for node voltages
        lines.append('    // Node voltage symbols')
        for node in self.all_nodes:
            lines.append(f'    symbol V_{node}("V_{node}");')

        # No parameter symbols — all params are constants at compile time,
        # EXCEPT callback (runtime) params, which stay symbolic so they are not
        # folded into constants and are fetched via _pcb() in the eval.
        if self._callback_params:
            lines.append('    // Callback (runtime) parameter symbols')
            for cp in sorted(self._callback_params):
                lines.append(f'    symbol {cp}("{cp}");')
        if self._thermal_params:
            lines.append('    // Thermal parameters (defaults reading $temperature / $vt)')
            for tp, _code in self._thermal_cxx:
                lines.append(f'    symbol {tp}("{tp}");')

        # Special symbols — skip if the model declares them as variables
        _model_vars = {v.name for v in self.mod.variables}
        lines.append(f'    symbol {VT_SYM}("{VT_SYM}");  // $vt: kT/q at the instance temperature')
        lines.append(f'    symbol {T_SYM}("{T_SYM}");  // $temperature: the instance temperature (K)')
        lines.append('    symbol _vae_abstime("_vae_abstime");  // $abstime: the state\'s t')
        lines.append('    _vae_vdep_opaque.push_back(_vae_abstime);  // never a constant factor of ddt()')
        lines.append('    (void)argc;')
        lines.append('')

        # Determine active nodes by walking AST with resolved conditions
        active_nodes = list(self.port_names)
        if self.collapse_nodes:
            shorted = self._find_shorted_nodes(self.mod.analog_block)
        else:
            shorted = set()  # keep all nodes active for Xyce integration
        for n in self.internal_nodes:
            if n not in shorted:
                active_nodes.append(n)

        n_nodes = len(active_nodes)

        # Emit shorting annotations
        for n in sorted(shorted):
            target = self._find_short_target(self.mod.analog_block, n)
            if target:
                lines.append(f'    // SHORT: {n} → {target}')

        # Branch-current unknowns (MNA): a symbol for the current of every
        # voltage-contribution branch and every probed branch; their equation
        # rows are completed after the walk (_finalize_current_rows).
        self.collect_current_unknowns(active_nodes)
        lines.append('    // Branch-current unknown symbols')
        for rec in self._cur_records:
            lines.append(f'    symbol {rec["sym"]}("{rec["sym"]}");')
        lines.append('    vector<symbol> cur_syms;    // the unknowns alive in this elaboration')
        lines.append('    vector<string> cur_labels;  // their row labels (shared with the device shell)')
        lines.append('    vector<int> cur_rows;       // the row holding each one\'s equation')
        lines.append('')

        sym_list = ", ".join(f"V_{n}" for n in active_nodes)
        name_list = ", ".join('"' + n + '"' for n in active_nodes)
        lines.append(f'    vector<symbol> nodes = {{{sym_list}}};')
        # Every unknown a ddt() factor must be free of: the node voltages and
        # the branch currents (_vae_split_ddt).
        lines.append('    vector<symbol> unk_syms = nodes;')
        for rec in self._cur_records:
            lines.append(f'    unk_syms.push_back({rec["sym"]});')
        lines.append(f'    vector<string> node_names = {{{name_list}}};')
        lines.append(f'    int n_nodes = {n_nodes};')
        lines.append('')

        # The analysis shadows (emit(): one per group of contexts whose event
        # behaviour differs; one for a module without such code).  Each is a
        # walk of the analog block into its own intermediates and F/Q rows;
        # the branch-label table and the list of branch-current unknowns are
        # shared, so a row means the same in every shadow.
        lines.append('    // the analysis shadows: intermediates (symbol, expression, output name;')
        lines.append('    // raw C++ value or statement by index) and F/Q rows of each')
        lines.append('    struct _Shadow { vector<symbol> int_syms; vector<ex> int_exprs; vector<string> int_names;')
        lines.append('                     std::map<int,std::string> int_raw; std::map<int,std::string> int_stmt;')
        lines.append('                     vector<ex> F, Q; std::set<string> alive; };')
        lines.append(f'    vector<_Shadow> _sh({len(self._shadows)});')
        lines.append(f'    vector<string> branch_labels;')
        lines.append('')
        if not hasattr(self, '_shadows') or not self._shadows:
            self._shadows = [list(VAE_CONTEXTS)]
            self._ctx_shadow = {c: 0 for c in VAE_CONTEXTS}
        for k, ctxs in enumerate(self._shadows):
            self._reset_walk_state()
            self._ctx = ctxs[0]
            self._ctx_names = analysis_names(*ctxs[0])
            self._walk_has_op_phase = any(c[1] == 0 for c in ctxs) or len(self._shadows) == 1
            lines.append(f'    {{ // shadow {k}: contexts {ctxs}')
            lines.append(f'    auto& int_syms = _sh[{k}].int_syms; auto& int_exprs = _sh[{k}].int_exprs;')
            lines.append(f'    auto& int_names = _sh[{k}].int_names; auto& int_raw = _sh[{k}].int_raw;')
            lines.append(f'    auto& int_stmt = _sh[{k}].int_stmt; auto& _alive = _sh[{k}].alive;')
            lines.append(f'    auto& F_contribs = _sh[{k}].F; auto& Q_contribs = _sh[{k}].Q;')
            self._held_preamble(lines, '    ')
            # Walk analog block, emitting GiNaC expressions
            self._walk_analog_block(lines, self.mod.analog_block,
                                   active_nodes, indent=4)
            self._held_epilogue(lines, '    ')
            self._finalize_current_rows(lines, '    ')
            lines.append('    (void)int_stmt; (void)_alive;')
            lines.append('    }')

        # Print C++ eval code
        lines.append('')
        lines.append(f'    int n_branches = branch_labels.size();')
        lines.append('')

        self._emit_code_printer(lines, active_nodes)

        lines.append('    return 0;')
        lines.append('}')
        return '\n'.join(lines) + '\n'

    def _reset_walk_state(self):
        """A fresh walk (an analysis shadow): every variable unknown again."""
        self.var_values = {}
        self._declared_vars = {}
        self._var_versions = {}
        self._var_sym_name = {}
        self._ddt_vars = {}
        self._possible = {}
        self._cond_stack = []
        self._runtime_cond_depth = 0
        self._condition_registry = []
        self._stmt_stack = []
        self._stmt_occ = {}
        self._rt_loops = []
        for rec in self._cur_records:
            rec['has_V'] = rec['has_I'] = rec['probed'] = False
            rec['v_gates'] = []
            rec['order'] = []

    @staticmethod
    def _cout(lines: list, text: str, I: str = '    '):
        """A line of the printed eval code (text verbatim)."""
        esc = text.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'{I}cout << "{esc}" << endl;')

    def _emit_code_printer(self, lines: list[str], active_nodes: list[str]):
        """Emit the code that prints the eval code: one function per analysis
        shadow, the dispatcher vae_eval, and the exports the device shell and
        build_vae_so.py's wrapper read (branch and node labels, the
        branch-current unknowns, the held state, the cross events, the
        run-time messages).

        No Jacobian is printed here (since 1.1.4): build_vae_so.py
        differentiates this eval code itself, exactly, by forward-mode
        automatic differentiation on dual numbers (vae/dual_ad.py, 1.1.5;
        1.1.0-1.1.4 took difference quotients).  The forward-AD vae_jacobian
        the metaprogram printed before 1.1.4 differentiated the arithmetic
        form of an indicator select while the value came from its ternary,
        and was never called; its GiNaC work and its g++ time were the
        larger part of a big model's build (HiCUM L2, PSP103)."""
        n_nodes = len(active_nodes)
        I = '    '  # base indent
        cout = lambda t: self._cout(lines, t, I)
        cout('static inline double _san(double x){ return (x==x && x*0.0==0.0) ? x : 0.0; }')
        # VA smoothing helpers may appear verbatim in ternary/select values that
        # were lowered as C++ text (not expanded through GiNaC); define them.
        cout('static inline double hypsmooth(double x,double c){ return (x+std::sqrt(x*x+4*c*c))/2; }')
        cout('static inline double hypmax(double x,double xmin,double c){ double d=x-xmin-c; return xmin+(d+std::sqrt(d*d-4*xmin*c))/2; }')
        cout('static inline double Tempdep(double PARAML,double PARAMT,double DELTEMP,double TEMPMOD){ return (TEMPMOD!=0.0) ? (PARAML+hypmax(PARAMT*DELTEMP,-PARAML,1e-6)) : (PARAML*hypsmooth(1+PARAMT*DELTEMP-1e-6,1e-3)); }')
        # ddx(expr, V_node) = small-signal OP-output derivative (gm/gds/caps).
        # These are report-only quantities, never in the large-signal I<+/Q<+
        # contributions, so stub to 0 for the DC/transient solve.
        cout('static inline double ddx(double,double){ return 0.0; }')
        cout('#include <cstdio>')
        cout('#include <cstring>')
        for h in _MSG_HELPERS_CPP:
            cout(h)
        sites = list(self._div_sites) + [f'module {self.mod.name}: an integer division or modulus by '
                                         f'zero at run time; PyMS gives 0']
        cout('static const char* _vae_div_msg[] = {'
             + ', '.join('"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"' for s in sites) + '};')
        for h in _int_helpers_cpp(len(self._div_sites)):
            cout(h)
        # the held state: its size, initial values and the cross events
        nh = len(self._held)
        cout(f'extern "C" int vae_n_held() {{ return {nh}; }}')
        init = ' '.join(f'H[{s}] = {v!r};' for s, v in sorted(self._held_init.items()) if v != 0.0)
        cout(f'extern "C" void vae_held_init(double* H) {{ for (int i = 0; i < {nh}; i++) H[i] = 0.0; {init} }}')
        cx = self._cross_exports
        # an accepted point is evaluated again, to hold its values, only for
        # held variables and cross events (the held regime of the indicators
        # needs no accepted point: it is read by the small-signal phase only)
        cout(f'extern "C" int vae_needs_accept() {{ return {1 if (self._state_vars or cx) else 0}; }}')
        cout(f'extern "C" int vae_n_cross() {{ return {len(cx)}; }}')
        cout('extern "C" int vae_cross_slot(int i) { static const int s[] = {'
             + ''.join(f'{s}, ' for s, _p, _d, _k in cx) + '-1}; return (i >= 0 && i < '
             + f'{len(cx)}) ? s[i] : -1; }}')
        cout('extern "C" int vae_cross_pending_slot(int i) { static const int s[] = {'
             + ''.join(f'{p}, ' for _s, p, _d, _k in cx) + '-1}; return (i >= 0 && i < '
             + f'{len(cx)}) ? s[i] : -1; }}')
        cout('extern "C" int vae_cross_dir(int i) { static const int d[] = {'
             + ''.join(f'{d}, ' for _s, _p, d, _k in cx) + '0}; return (i >= 0 && i < '
             + f'{len(cx)}) ? d[i] : 0; }}')
        _mvars = {v.name for v in self.mod.variables}
        single = len(self._shadows) == 1
        for k, ctxs in enumerate(self._shadows):
            fname = 'vae_eval' if single else f'_vae_eval_s{k}'
            cout(f'// analysis shadow {k}: (an, phase) {ctxs}')
            cout(('void ' if single else 'static void ') + fname
                 + '(VaeState* _vae_s, double* _vae_F, double* _vae_Q) {')
            for i, n in enumerate(active_nodes):
                cout(f'    double V_{n} = _vae_s->V[{i}];')
            lines.append(f'{I}for (int k = 0; k < (int)cur_syms.size(); k++)')
            lines.append(f'{I}    cout << "    double " << cur_syms[k].get_name() << " = _vae_s->I[" << k << "];" << endl;')
            # $temperature: the instance's temperature at this evaluation,
            # from the state (the shell reads it through the instance's
            # temperature source); $vt from it with Xyce's k/q (1.1.6; before,
            # a constant 300.15 K of the wrapper and the shell's fixed Vt).
            # AD hook: _vae_T is a double -- no unknown reaches it, so the
            # dual copy keeps it and everything only it reaches on double.  A
            # thermal unknown (a self-heating node) would be one more seed
            # of dual_ad (_vae_s->T beside _vae_s->V / ->I): _vae_T and $vt
            # become duals and their column is dF/dT.
            cout(f'    double {T_SYM} = _vae_s->T;')
            cout(f'    double {VT_SYM} = {VAE_KOQ} * {T_SYM};')
            cout('    double _vae_abstime = _vae_s->t; (void)_vae_abstime;')
            # Runtime (callback) params — fetched live so .SAMPLING reaches them.
            # The shell's callback knows the parameter by Xyce's (upper-case) name.
            for cp in sorted(self._callback_params):
                cout(f'    double {cp} = _pcb("{cp.upper()}");')
            # thermal parameters: their defaults at this evaluation's
            # temperature (1.1.6), so a .step temp follows without a rebuild
            for tp, code in self._thermal_cxx:
                cout(f'    double {tp} = {code};')
            lines.append(f'{I}cout << endl;')
            lines.append(f'{I}{{')
            lines.append(f'{I}    _Shadow& S = _sh[{k}];')
            lines.append(f'{I}    vector<set<string>> dst(1);')
            lines.append(f'{I}    for (int i = 0; i < (int)S.int_syms.size(); i++) {{')
            lines.append(f'{I}        if (S.int_stmt.count(i)) {{')
            lines.append(f'{I}            const string& t = S.int_stmt[i];')
            lines.append(f'{I}            cout << "    " << t << endl;')
            lines.append(f'{I}            if (!t.empty() && t.back() == \'{{\') dst.push_back(dst.back());')
            lines.append(f'{I}            else if (!t.empty() && t[0] == \'}}\' && dst.size() > 1) dst.pop_back();')
            lines.append(f'{I}            continue;')
            lines.append(f'{I}        }}')
            lines.append(f'{I}        string sn = S.int_syms[i].get_name();')
            lines.append(f'{I}        string val = S.int_raw.count(i) ? S.int_raw[i] : to_cpp(S.int_exprs[i]);')
            lines.append(f'{I}        if (dst.back().count(sn) == 0) {{')
            lines.append(f'{I}            cout << "    double " << sn << " = " << val << ";" << endl;')
            lines.append(f'{I}            dst.back().insert(sn);')
            lines.append(f'{I}        }} else {{')
            lines.append(f'{I}            cout << "    " << sn << " = " << val << ";" << endl;')
            lines.append(f'{I}        }}')
            lines.append(f'{I}    }}')
            # Every row is written (the shadows share the row table; the
            # wrapper's arrays are reused between calls): a row this shadow
            # has no contribution to is 0, and the row of a branch-current
            # unknown dead in this shadow pins it to 0.
            lines.append(f'{I}    for (int i = 0; i < n_branches; i++) {{')
            lines.append(f'{I}        ex f = i < (int)S.F.size() ? S.F[i] : ex(0);')
            lines.append(f'{I}        ex q = i < (int)S.Q.size() ? S.Q[i] : ex(0);')
            lines.append(f'{I}        int ci = _vae_find(cur_labels, branch_labels[i]);')
            lines.append(f'{I}        if (ci >= 0 && cur_rows[ci] == i && !S.alive.count(branch_labels[i])) f = cur_syms[ci];')
            lines.append(f'{I}        cout << "    _vae_F[" << i << "] = " << to_cpp(f) << ";  // " << branch_labels[i] << endl;')
            lines.append(f'{I}        cout << "    _vae_Q[" << i << "] = " << to_cpp(q) << ";" << endl;')
            lines.append(f'{I}    }}')
            lines.append(f'{I}}}')
            cout('}')
            lines.append(f'{I}cout << endl;')
        if not single:
            cout('void vae_eval(VaeState* _vae_s, double* _vae_F, double* _vae_Q) {')
            cout('    switch (_vae_s->an * 4 + _vae_s->phase) {')
            for (an, ph), k in sorted(self._ctx_shadow.items()):
                cout(f'    case {an * 4 + ph}: _vae_eval_s{k}(_vae_s, _vae_F, _vae_Q); return;')
            cout(f'    default: _vae_eval_s{self._ctx_shadow[(0, 0)]}(_vae_s, _vae_F, _vae_Q); return;')
            cout('    }')
            cout('}')

        # Branch / node label exports — let the Xyce-side wrapper match
        # its own contribution walk against the GiNaC walk's branch
        # ordering. Without this, the wrapper's index-i for branch I
        # may not be the GiNaC's index-i (the GiNaC walk does
        # parameter-resolution to skip dead conditional branches; the
        # wrapper's walk doesn't). Mismatch → F-values stamp into the
        # wrong KCL row.
        lines.append(f'{I}cout << "extern \\"C\\" int vae_n_branches() {{ return " << n_branches << "; }}" << endl;')
        lines.append(f'{I}cout << "extern \\"C\\" int vae_n_nodes() {{ return " << n_nodes << "; }}" << endl;')
        lines.append(f'{I}cout << "extern \\"C\\" const char* vae_branch_label(int i) {{" << endl;')
        lines.append(f'{I}cout << "    static const char* labels[] = {{";')
        lines.append(f'{I}for (int i = 0; i < n_branches; i++) {{')
        lines.append(f'{I}    if (i) cout << ", ";')
        lines.append(f'{I}    cout << "\\"" << branch_labels[i] << "\\"";')
        lines.append(f'{I}}}')
        lines.append(f'{I}if (n_branches == 0) cout << "\\"\\"";')
        lines.append(f'{I}cout << "}};" << endl;')
        lines.append(f'{I}cout << "    return (i >= 0 && i < " << n_branches << ") ? labels[i] : \\"\\";" << endl;')
        lines.append(f'{I}cout << "}}" << endl;')
        lines.append(f'{I}cout << "extern \\"C\\" const char* vae_node_label(int i) {{" << endl;')
        lines.append(f'{I}cout << "    static const char* labels[] = {{";')
        lines.append(f'{I}for (int i = 0; i < (int)node_names.size(); i++) {{')
        lines.append(f'{I}    if (i) cout << ", ";')
        lines.append(f'{I}    cout << "\\"" << node_names[i] << "\\"";')
        lines.append(f'{I}}}')
        lines.append(f'{I}cout << "}};" << endl;')
        lines.append(f'{I}cout << "    return (i >= 0 && i < " << n_nodes << ") ? labels[i] : \\"\\";" << endl;')
        lines.append(f'{I}cout << "}}" << endl;')
        # Branch-current unknowns: how many (the Jacobian has n_nodes + this
        # many columns, in that order; the eval reads them from s->I[k]), the
        # row label of each (the device shell matches it against its own
        # list from collect_current_unknowns) and the row holding its
        # equation.  The trailing sentinel keeps the arrays non-empty.
        lines.append(f'{I}cout << "extern \\"C\\" int vae_n_currents() {{ return " << cur_syms.size() << "; }}" << endl;')
        lines.append(f'{I}cout << "extern \\"C\\" const char* vae_current_label(int i) {{" << endl;')
        lines.append(f'{I}cout << "    static const char* labels[] = {{";')
        lines.append(f'{I}for (int i = 0; i < (int)cur_labels.size(); i++) cout << "\\"" << cur_labels[i] << "\\", ";')
        lines.append(f'{I}cout << "\\"\\"}};" << endl;')
        lines.append(f'{I}cout << "    return (i >= 0 && i < " << cur_labels.size() << ") ? labels[i] : \\"\\";" << endl;')
        lines.append(f'{I}cout << "}}" << endl;')
        lines.append(f'{I}cout << "extern \\"C\\" int vae_current_row(int i) {{" << endl;')
        lines.append(f'{I}cout << "    static const int rows[] = {{";')
        lines.append(f'{I}for (int i = 0; i < (int)cur_rows.size(); i++) cout << cur_rows[i] << ", ";')
        lines.append(f'{I}cout << "-1}};" << endl;')
        lines.append(f'{I}cout << "    return (i >= 0 && i < " << cur_rows.size() << ") ? rows[i] : -1;" << endl;')
        lines.append(f'{I}cout << "}}" << endl;')

        # Metadata
        lines.append(f'{I}cout << "// n_nodes = " << n_nodes << endl;')
        lines.append(f'{I}cout << "// n_branches = " << n_branches << endl;')
        lines.append(f'{I}cout << "// n_currents = " << cur_syms.size() << endl;')
        lines.append(f'{I}cout << "// analysis shadows = {len(self._shadows)}" << endl;')
        lines.append(f'{I}cout << "// held = {len(self._held)}" << endl;')
        lines.append(f'{I}{{ size_t ni = 0; for (auto& s : _sh) ni += s.int_syms.size(); cout << "// intermediates = " << ni << endl; }}')
        lines.append(f'{I}cout << "// nodes:";')
        lines.append(f'{I}for (auto& n : node_names) cout << " " << n;')
        lines.append(f'{I}cout << endl;')
        lines.append(f'{I}cout << "// currents:";')
        lines.append(f'{I}for (auto& l : cur_labels) cout << " " << l;')
        lines.append(f'{I}cout << endl;')

    def _emit_line_directive(self, lines: list[str], node: ASTNode, pfx: str):
        """Emit a #line or //line directive."""
        if self.line_directives is None or not node.loc or not node.loc.line:
            return
        prefix = '#' if self.line_directives else '//'
        filename = node.loc.file
        lineno = node.loc.line
        escaped = filename.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'{pfx}cout << "{prefix}line {lineno} \\"{escaped}\\"" << endl;')

    # A message that already starts with a source location ``file:line: ``.
    _LOCATED_RE = re.compile(r'^\S+:\d+: ')

    def _walk_analog_block(self, lines: list[str], node: ASTNode,
                           active_nodes: list[str], indent: int):
        """_walk_statement, with the source file:line of the statement put in
        front of an EmitError or CurrentProbeError raised without one (the
        expression translators do not know the statement)."""
        if node is not None:
            occ = self._stmt_occ.get(id(node), 0) + 1
            self._stmt_occ[id(node)] = occ
            self._stmt_stack.append([id(node), occ, 0])
        saved_loc = self._cur_loc
        if node is not None and getattr(node, 'loc', None) is not None and getattr(node.loc, 'line', 0):
            self._cur_loc = node.loc
        try:
            self._walk_statement(lines, node, active_nodes, indent)
        except (EmitError, CurrentProbeError) as e:
            msg = str(e)
            if self._LOCATED_RE.match(msg) or node is None or node.kind == NodeKind.BLOCK:
                raise
            where = self._where(node)
            if where.startswith('module '):
                raise
            raise type(e)(f'{where}: {msg}') from None
        finally:
            self._cur_loc = saved_loc
            if node is not None:
                self._stmt_stack.pop()

    def _walk_statement(self, lines: list[str], node: ASTNode,
                        active_nodes: list[str], indent: int):
        """Walk AST and emit GiNaC expression-building code.

        Parameter-dependent conditions are resolved using self.param_values.
        Only instance-param or voltage-dependent conditions remain as runtime code.
        """
        if node is None:
            return
        pfx = ' ' * indent

        if node.kind == NodeKind.BLOCK and node.loop is not None:
            self._walk_loop(lines, node, active_nodes, indent)

        elif node.kind == NodeKind.BLOCK:
            for child in node.children:
                self._walk_analog_block(lines, child, active_nodes, indent)

        elif node.kind == NodeKind.ASSIGN:
            if node.decl_init and node.lhs in self._state_set:
                # the declaration value of a held variable is its initial
                # held value (vae_held_init), not an assignment of every point
                return
            self._emit_line_directive(lines, node, pfx)
            # an integer variable takes a real value rounded (LRM 4.2.1.1)
            expr = node.expr if not node.int_lhs else f'_vae_rint ( {node.expr} )'
            self._check_time_operators(expr, node, 'the assignment')

            # A value that carries ddt() (written here, or read from a variable
            # that carries it) gets no CSE symbol: its translated text is
            # inlined where the variable is read, so the ddt() terms reach the
            # contribution and _vae_split_ddt there. A gated assignment keeps
            # the select form lhs_prev + (rhs - lhs_prev) * Sprod, whose
            # indicator factors are constants to the split.
            if self._expr_carries_ddt(expr) or \
                    (self._cond_stack and node.lhs in self._ddt_vars):
                rhs_expr = self._maybe_hoist(expr, lines, active_nodes, pfx)
                rhs = self._to_ginac_expr(rhs_expr, active_nodes)
                if self._cond_stack:
                    prev = self._var_current_ex_str(node.lhs)
                    sprod = ' * '.join(f'({s})' for s in self._cond_stack)
                    rhs = f'(({prev}) + ((({rhs})) - ({prev})) * ({sprod}))'
                self.var_values.pop(node.lhs, None)
                self._possible.pop(node.lhs, None)
                self._ddt_vars[node.lhs] = f'({rhs})'
                lines.append(f'{pfx}// {node.lhs} carries ddt(): inlined where it is read')
                return
            self._ddt_vars.pop(node.lhs, None)

            # Gated assignment inside a voltage-dependent conditional: emit as
            # an arithmetic select  lhs = lhs_prev + (rhs - lhs_prev) * Sprod
            # (Sprod = product of enclosing indicator factors). The branch is a
            # genuine runtime decision, and forward-AD gives the branch-selected
            # derivative because each indicator has dS/dV = 0.
            if self._cond_stack:
                v = None if self._expr_uses_voltage(expr) else self._try_eval_expr(expr)
                if node.lhs in self._possible:
                    before = self._possible[node.lhs]
                elif node.lhs in self.var_values:
                    before = {self.var_values[node.lhs]}
                elif node.lhs not in self._declared_vars:
                    before = {0.0}                    # never assigned: a variable starts at 0
                else:
                    before = None
                if v is not None and before is not None:
                    self._possible[node.lhs] = before | {v}
                else:
                    self._possible.pop(node.lhs, None)
                prev = self._var_current_ex_str(node.lhs)
                rhs_expr = self._maybe_hoist(expr, lines, active_nodes, pfx)
                rhs = self._to_ginac_expr(rhs_expr, active_nodes)
                sprod = ' * '.join(f'({s})' for s in self._cond_stack)
                # arithmetic form for forward-AD (indicators have dS/dV=0 ->
                # branch-selected derivative); ternary form for the VALUE so the
                # non-taken RHS is not evaluated (preserves domain guards like
                # `if (Isbs>0) x = .../Isbs`).
                gated = f'(({prev}) + ((({rhs})) - ({prev})) * ({sprod}))'
                rhs_cpp = self._cond_final_cpp(rhs_expr, active_nodes)
                prev_cpp = '0.0' if prev == 'ex(0)' else prev
                tern = f'((({sprod}) != 0.0) ? ({rhs_cpp}) : ({prev_cpp}))'
                esc = tern.replace('\\', '\\\\').replace('"', '\\"')
                self._var_versions[node.lhs] = self._var_versions.get(node.lhs, 0) + 1
                sym_name = f'{self._safe_sym(node.lhs)}__sel{self._var_versions[node.lhs]}'
                self.var_values.pop(node.lhs, None)
                self._var_sym_name[node.lhs] = sym_name
                self._declared_vars[node.lhs] = 'sym'
                lines.append(f'{pfx}symbol {sym_name}("{sym_name}");')
                lines.append(f'{pfx}int_raw[int_syms.size()] = "{esc}";')
                lines.append(f'{pfx}int_syms.push_back({sym_name});')
                lines.append(f'{pfx}int_exprs.push_back(_safe_ex([&]() -> ex {{ return ({gated}); }}));')
                lines.append(f'{pfx}int_names.push_back("{node.lhs}");')
                return

            uses_voltage = self._expr_uses_voltage(expr)

            # Try constant evaluation
            val = None if uses_voltage else self._try_eval_expr(expr)
            if val is not None:
                self._possible[node.lhs] = {val}
            else:
                self._possible.pop(node.lhs, None)
            if val is not None and self._runtime_cond_depth == 0:
                # Outside runtime conditions: pure constant, no GiNaC symbol needed
                self.var_values[node.lhs] = val
                return

            # Non-constant (or a constant inside a runtime condition, which
            # also keeps its value for downstream condition resolution):
            # create a GiNaC symbol for CSE.  The right-hand side is
            # translated BEFORE the left-hand side's known value is updated
            # or dropped: `x = a; x = x + tc * ($temperature - 300.15);`
            # reads the OLD x (1k).  Dropping it first turned the second
            # assignment into the self-reference x = x + ... -- emitted as
            # `double x = 0.0;` (or `double y = y;` for a block-local), which
            # zeroed BSIM-SOI 4.x's junction potential PhiBSWG (NaN charge,
            # no DC convergence; 2026-10-25).
            expr_h = self._maybe_hoist(expr, lines, active_nodes, pfx)
            ginac_expr = self._to_ginac_expr(expr_h, active_nodes)
            if val is not None:
                self.var_values[node.lhs] = val
            else:
                self.var_values.pop(node.lhs, None)

            if node.lhs not in self._declared_vars:
                # First assignment — create new symbol
                sym_name = self._safe_sym(node.lhs)
                self._var_versions[node.lhs] = 0
                self._var_sym_name[node.lhs] = sym_name
                self._declared_vars[node.lhs] = 'sym'
                lines.append(f'{pfx}symbol {sym_name}("{sym_name}");')
            elif self._runtime_cond_depth > 0:
                # Inside runtime condition — reuse existing symbol, update expression
                sym_name = self._var_sym_name[node.lhs]
            else:
                # Reassignment outside condition — new version
                self._var_versions[node.lhs] += 1
                sym_name = f'{self._safe_sym(node.lhs)}__{self._var_versions[node.lhs]}'
                self._var_sym_name[node.lhs] = sym_name
                lines.append(f'{pfx}symbol {sym_name}("{sym_name}");')

            lines.append(f'{pfx}int_syms.push_back({sym_name});')
            lines.append(f'{pfx}int_exprs.push_back(_safe_ex([&]() -> ex {{ return ({ginac_expr}); }}));')
            lines.append(f'{pfx}int_names.push_back("{node.lhs}");')

        elif node.kind == NodeKind.CONTRIB:
            self._emit_contribution(lines, node, active_nodes, pfx)

        elif node.kind == NodeKind.IF:
            resolved = self._eval_condition(node.condition)
            if resolved is True:
                # Take true branch only
                for child in node.children:
                    self._walk_analog_block(lines, child, active_nodes, indent)
            elif resolved is False:
                # Take else branch only
                if node.else_body:
                    self._walk_analog_block(lines, node.else_body,
                                           active_nodes, indent)
            else:
                # Can't resolve statically — try evaluating with var_values
                resolved_dyn = self._eval_condition_with_vars(node.condition)
                if resolved_dyn is True:
                    lines.append(f'{pfx}// DYNAMIC CONDITION (true): {node.condition}')
                    self._runtime_cond_depth += 1
                    for child in node.children:
                        self._walk_analog_block(lines, child, active_nodes, indent)
                    self._runtime_cond_depth -= 1
                elif resolved_dyn is False:
                    lines.append(f'{pfx}// DYNAMIC CONDITION (false): {node.condition}')
                    if node.else_body:
                        self._runtime_cond_depth += 1
                        self._walk_analog_block(lines, node.else_body,
                                               active_nodes, indent)
                        self._runtime_cond_depth -= 1
                else:
                    # Check assume_true patterns
                    if any(pat in node.condition for pat in self._assume_true):
                        lines.append(f'{pfx}// ASSUMED TRUE: {node.condition}')
                        self._runtime_cond_depth += 1
                        for child in node.children:
                            self._walk_analog_block(lines, child, active_nodes, indent)
                        self._runtime_cond_depth -= 1
                    else:
                        # Voltage-dependent condition — register it
                        cond_idx = len(self._condition_registry)
                        self._condition_registry.append(node.condition)
                        nid = id(node)

                        # Check forced_nodes (by AST node id) or forced_conditions (by index)
                        forced = None
                        if self._forced_nodes is not None and nid in self._forced_nodes:
                            forced = self._forced_nodes[nid]
                        elif self._forced_conditions is not None and cond_idx in self._forced_conditions:
                            forced = self._forced_conditions[cond_idx]

                        if forced is not None:
                            lines.append(f'{pfx}// REGIME CONDITION [{cond_idx}] = {forced}: {node.condition}')
                            if forced:
                                for child in node.children:
                                    self._walk_analog_block(lines, child, active_nodes, indent)
                            else:
                                if node.else_body:
                                    self._walk_analog_block(lines, node.else_body,
                                                           active_nodes, indent)
                        else:
                            # Voltage-dependent condition -> indicator-select.
                            # Emit one runtime 0/1 indicator S; each assignment
                            # inside the branch is gated by the product of
                            # enclosing indicators (see NodeKind.ASSIGN). True
                            # children gate by S, else children by (1 - S). This
                            # keeps everything symbolic (so forward-AD works) while
                            # the clamp/branch is a genuine runtime decision.
                            cond_cpp = self._cond_final_cpp(node.condition, active_nodes)
                            S = self._alloc_indicator(lines, cond_cpp, pfx)
                            lines.append(f'{pfx}// SELECT [{cond_idx}] via {S}: {node.condition}')
                            self._cond_stack.append(S)
                            self._runtime_cond_depth += 1
                            for child in node.children:
                                self._walk_analog_block(lines, child, active_nodes, indent)
                            self._runtime_cond_depth -= 1
                            self._cond_stack.pop()
                            if node.else_body:
                                self._cond_stack.append(f'(1 - {S})')
                                self._runtime_cond_depth += 1
                                self._walk_analog_block(lines, node.else_body, active_nodes, indent)
                                self._runtime_cond_depth -= 1
                                self._cond_stack.pop()

        elif node.kind == NodeKind.INITIAL_STEP:
            self._walk_event(lines, node, active_nodes, indent)

        elif node.kind == NodeKind.EXPR:
            pass  # Skip raw expressions ($strobe, $finish, etc.)

    # --- events (LRM 5.10.3), held state (1.1.4) ---

    def _emit_stmt(self, lines: list, text: str, pfx: str):
        """A C++ statement of the eval code, verbatim, at this point of the
        intermediate list (the printer's int_stmt: a text ending in '{' opens
        a scope, one starting with '}' closes it)."""
        self._sel_counter += 1
        name = f'_st_{self._sel_counter}'
        esc = text.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'{pfx}symbol {name}("{name}");')
        lines.append(f'{pfx}int_stmt[int_syms.size()] = "{esc}";')
        lines.append(f'{pfx}int_syms.push_back({name});')
        lines.append(f'{pfx}int_exprs.push_back({name});')
        lines.append(f'{pfx}int_names.push_back("{name}");')

    def _raw_sym(self, lines: list, base: str, raw: str, pfx: str, vdep: bool = True) -> str:
        """An intermediate of the eval code whose value is the C++ text raw,
        opaque to the metaprogram (a symbol; with vdep it counts as voltage-
        dependent where a ddt() factor must be constant)."""
        self._sel_counter += 1
        name = f'{base}_{self._sel_counter}'
        esc = raw.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'{pfx}symbol {name}("{name}");')
        lines.append(f'{pfx}int_raw[int_syms.size()] = "{esc}";')
        lines.append(f'{pfx}int_syms.push_back({name});')
        lines.append(f'{pfx}int_exprs.push_back({name});')
        lines.append(f'{pfx}int_names.push_back("{name}");')
        if vdep:
            lines.append(f'{pfx}_vae_vdep_opaque.push_back({name});')
        return name

    def _cur_cpp(self, name: str) -> str:
        """The C++ text of a variable's value at this point of the walk."""
        if name in self._ddt_vars:
            raise EmitError(f'{self._where_cur()}: the variable {name} carries ddt() and must keep '
                            f'its value from one pass or point to the next; PyMS cannot build this')
        if name in self.var_values:
            return repr(float(self.var_values[name]))
        if name in self._var_sym_name:
            return self._var_sym_name[name]
        return '0.0'

    def _where_cur(self) -> str:
        loc = self._cur_loc
        if loc is not None and getattr(loc, 'line', 0):
            return f'{loc.file}:{loc.line}'
        return f'module {self.mod.name}'

    def _bind_symbol(self, var: str, sym: str):
        """From here on the walk reads var as the eval-code variable sym."""
        self.var_values.pop(var, None)
        self._possible.pop(var, None)
        self._ddt_vars.pop(var, None)
        self._var_sym_name[var] = sym
        self._declared_vars[var] = 'sym'
        self._var_versions.setdefault(var, 0)

    def _held_preamble(self, lines: list, pfx: str):
        """Every held variable starts the evaluation from its value held at
        the last accepted point (the initial value before the first)."""
        for v in self._state_vars:
            init = self._state_init.get(v, 0.0)
            slot = self._held_slot(('var', v), init)
            # (a held value is constant in time between two points: a ddt()
            # factor may be one)
            sym = self._raw_sym(lines, f'{self._safe_sym(v)}__h', f'_vae_held(_vae_s, {slot}, {init!r})', pfx,
                                vdep=False)
            self._bind_symbol(v, sym)

    def _held_epilogue(self, lines: list, pfx: str):
        """... and its value at the end of the evaluation is the new held one."""
        for v in self._state_vars:
            slot = self._held_slot(('var', v))
            self._emit_stmt(lines, f'_vae_hold(_vae_s, {slot}, {self._cur_cpp(v)});', pfx)

    def _emit_cross(self, lines, node, idx, ev, active_nodes, pfx) -> Optional[str]:
        """cross(e[, dir, ...]) / above(e[, ...]) (LRM 5.10.3.3-4): the value
        of e is held from one accepted point of a transient to the next.  The
        evaluation of an accepted point (phase 3, the device shell's
        acceptStep) that finds e crossed 0 since the last one, in the
        direction dir (above: upwards), marks the event pending; the shell
        then puts a breakpoint just after that point, and the evaluation of
        that breakpoint's accepted point fires it: the body runs there and
        its values are held from there on, so the change of the circuit
        comes at a breakpoint, where Xyce restarts the integration (a body
        run inside a step would change the circuit within the step, a jump
        no step size resolves).  The shell also puts a breakpoint where the
        last two accepted values predict the crossing, so that the event
        lands close to it.  The trial evaluations of a step never fire it.
        above() fires at the first point of an analysis too when e >= 0.
        Returns the indicator of the event in this context, or None when it
        cannot fire here (a dc point, an ac sweep, a trial evaluation)."""
        args = ev['args']
        expr = args[0]
        self._check_time_operators(expr, node, f'the {ev["kind"]}() expression')
        if ev['kind'] == 'cross':
            d = 0
            if len(args) > 1 and args[1].strip():
                v = self._try_eval_expr(args[1])
                if v is None:
                    raise EmitError(f'{self._where(node)}: cross({" ".join(expr.split())}, {args[1].strip()}): '
                                    f'the direction must be a constant (-1, 0 or +1)')
                d = 1 if v > 0 else (-1 if v < 0 else 0)
        else:
            d = 1
        occ = self._stmt_occ.get(id(node), 1)
        slot = self._held_slot(('cross', id(node), occ, idx), 0.0)
        pslot = self._held_slot(('crossp', id(node), occ, idx), 0.0)
        if not any(s == slot for s, _p, _d, _k in self._cross_exports):
            self._cross_exports.append((slot, pslot, d, ev['kind']))
        now = self._raw_sym(lines, '_cx', self._cond_final_cpp(expr, active_nodes), pfx)
        an, phase = self._ctx
        fire = None
        if (an, phase) == (1, 3):
            pend = self._raw_sym(lines, '_evp', f'_vae_held(_vae_s, {pslot}, 0.0)', pfx)
            crossed = self._raw_sym(lines, '_evc', f'_vae_cross(_vae_held(_vae_s, {slot}, {now}), {now}, {d})', pfx)
            self._emit_stmt(lines, f'_vae_hold(_vae_s, {pslot}, ({pend} != 0.0) ? 0.0 : {crossed});', pfx)
            fire = pend
        elif ev['kind'] == 'above' and phase in (0, 2):
            fire = self._raw_sym(lines, '_ev', f'({now} >= 0.0) ? 1.0 : 0.0', pfx)
        self._emit_stmt(lines, f'_vae_hold(_vae_s, {slot}, {now});', pfx)
        return fire

    def _walk_event(self, lines: list, node: ASTNode, active_nodes: list, indent: int):
        """An event-controlled statement @(events) body (1.1.4).

        initial_step[(list)]: runs at the first point of an analysis (the
        operating point it starts from) when the list is empty or names one
        of the analyses (analysis_names), in the shadow of that context; it
        does not run anywhere else -- in a bare .op neither
        initial_step("ac") nor initial_step("tran") runs -- and the variables
        it set keep their values (held).  An invariant initial_step (it reads
        parameters only and its variables are assigned nowhere else) gives
        the same at every point and runs where it stands, as before.
        final_step: its body runs after the last point, when no evaluation
        follows: it changes no value PyMS computes (system tasks are not
        executed by PyMS), so it is not built.  cross / above: _emit_cross.
        Before 1.1.4 every event body ran at every evaluation."""
        pfx = ' ' * indent
        info = self._event_info.get(id(node))
        if info is None:
            info = dict(node=node, evs=self._events_of(node), trivial=True)
        if info['trivial']:
            for child in node.children:
                self._walk_analog_block(lines, child, active_nodes, indent)
            return
        kinds = ' or '.join(e['kind'] for e in info['evs'])
        runs, gates = False, []
        for idx, e in enumerate(info['evs']):
            k = e['kind']
            if k in self._EVENTS_ALWAYS:
                runs = True
            elif k == 'initial_step':
                runs = runs or self._initial_step_runs(e, self._ctx)
            elif k in ('cross', 'above'):
                g = self._emit_cross(lines, node, idx, e, active_nodes, pfx)
                if g is not None:
                    gates.append(g)
        if runs:
            lines.append(f'{pfx}// EVENT @({kinds}): runs in this analysis context')
            for child in node.children:
                self._walk_analog_block(lines, child, active_nodes, indent)
            return
        if not gates:
            lines.append(f'{pfx}// EVENT @({kinds}): does not run in this analysis context')
            return
        if len(gates) == 1:
            S = gates[0]
        else:
            S = self._raw_sym(lines, '_ev', ' || '.join(f'({g} != 0.0)' for g in gates) + ' ? 1.0 : 0.0', pfx)
        lines.append(f'{pfx}// EVENT @({kinds}) via {S}')
        self._cond_stack.append(S)
        self._runtime_cond_depth += 1
        try:
            for child in node.children:
                self._walk_analog_block(lines, child, active_nodes, indent)
        finally:
            self._runtime_cond_depth -= 1
            self._cond_stack.pop()

    # --- loops (LRM 5.10) ---

    # Iterations a loop may take at elaboration (its condition known there),
    # and iterations PyMS unrolls under runtime indicators (a condition on
    # the node voltages, bounded by a counter).
    _LOOP_EXACT_LIMIT = 100000
    _LOOP_RUNTIME_LIMIT = 1000
    # Iterations unrolled under runtime indicators at most (only a loop whose
    # body has ddt(), a voltage contribution or an event statement; every
    # other loop of the run time is a C++ loop of the eval code since 1.1.4):
    # every one is a copy of the body in the metaprogram, and g++ time grows
    # faster than the copies (HiCUM L2's 101-iteration Q_pT loop: a 2 MB
    # program, 1607 s and 6.7 GB to compile at -O2).  A loop that may need
    # more is guarded: a hard error of the simulation.
    _LOOP_RUNTIME_UNROLL = 16

    @staticmethod
    def _loop_parts(node):
        lp, ch = node.loop, node.children
        init = ch[0] if lp['has_init'] else None
        step = ch[-1] if lp['has_step'] else None
        body = ch[1 if lp['has_init'] else 0]
        return lp, init, body, step

    def _loop_cond_value(self, cond: str) -> Optional[bool]:
        r = self._eval_condition(cond)
        if r is None:
            r = self._eval_condition_with_vars(cond)
        return r

    @staticmethod
    def _split_top(text: str, sep: str) -> list:
        out, depth, cur, i = [], 0, [], 0
        while i < len(text):
            ch = text[i]
            if ch in '([':
                depth += 1
            elif ch in ')]':
                depth -= 1
            if depth == 0 and text.startswith(sep, i):
                out.append(''.join(cur))
                cur = []
                i += len(sep)
                continue
            cur.append(ch)
            i += 1
        out.append(''.join(cur))
        return [o.strip() for o in out if o.strip()]

    def _loop_trip_bound(self, cond: str, body: ASTNode, step: Optional[ASTNode]) -> Optional[int]:
        """An upper bound of the iterations left for a loop whose condition is
        not known at elaboration: a conjunct of the condition that reads one
        variable x (and constants), x known now and changed only by one
        top-level ``x = x + K`` / ``x = x - K`` of the body or the step, K
        constant.  The bound is the number of iterations that conjunct allows
        (HiCUM L2's ``l_it <= l_itmax``).  None: no such conjunct."""
        known_fn = set(_INT_HELPERS_PY) | {'abs', 'min', 'max', 'sqrt', 'exp', 'ln', 'log', 'pow'}
        body_top = body.children if (body.kind == NodeKind.BLOCK and body.loop is None) else [body]
        best = None
        for c in self._split_top(cond, '&&'):
            ids = set(re.findall(r'\b[A-Za-z_]\w*\b', _STRING_LIT_RE.sub('""', c)))
            ids -= known_fn
            ids -= set(self.param_values) - self.instance_params
            ids -= set(self._string_values)
            if len(ids) != 1:
                continue
            x = ids.pop()
            starts = ({self.var_values[x]} if x in self.var_values else self._possible.get(x))
            if not starts:
                continue
            assigns = [n for n in _ast_walk(body) if n.kind == NodeKind.ASSIGN and n.lhs == x]
            if step is not None and step.lhs == x:
                assigns.append(step)
            if len(assigns) != 1:
                continue
            a = assigns[0]
            if not (a is step or a in body_top):
                continue
            e = ' '.join(a.expr.split())
            # (the tokenizer reads a signed number as one token: 'k +1')
            m = re.fullmatch(re.escape(x) + r'\s*([+-])\s*(.+)', e)
            sign, ktext = (m.group(1), m.group(2)) if m else (None, None)
            if m is None:
                m = re.fullmatch(r'(.+?)\s*\+\s*' + re.escape(x), e)
                if m is None:
                    continue
                sign, ktext = '+', m.group(1)
            k = self._try_eval_expr(ktext)
            if not k:
                continue
            delta = k if sign == '+' else -k
            had, saved = x in self.var_values, self.var_values.get(x)
            ok, most = True, 0
            try:
                for v0 in starts:
                    v, n = v0, 0
                    while True:
                        self.var_values[x] = v
                        r = self._eval_condition_with_vars(c)
                        if r is None:
                            ok = False
                            break
                        if not r:
                            break
                        n += 1
                        v += delta
                        if n > self._LOOP_RUNTIME_LIMIT:
                            break
                    most = max(most, n)
            finally:
                if had:
                    self.var_values[x] = saved
                else:
                    self.var_values.pop(x, None)
            if ok:
                best = most if best is None else min(best, most)
        return best

    def _walk_loop(self, lines: list, node: ASTNode, active_nodes: list, indent: int):
        """for / while / repeat (LRM 5.10).  A loop whose condition is known
        at elaboration (the parameters, the variables folded so far) is
        unrolled exactly: the body runs as often as the condition says, a
        zero-trip loop not at all.  A condition PyMS cannot fold (it reads
        the node voltages) makes the rest of the loop a C++ loop of the eval
        code, exact (_walk_runtime_loop, 1.1.4).  Only a body with ddt(), a
        voltage contribution or an event statement is unrolled under a chain
        of runtime indicators, iteration k running while iterations 1..k-1
        ran and the condition holds, up to the bound a counter conjunct gives
        (_loop_trip_bound) and at most _LOOP_RUNTIME_UNROLL iterations; when
        the loop may need more (no bound, or a larger one) an evaluation
        whose loop would go on is a hard error of the simulation naming the
        loop (1.1.3: a warning, and the values after 16).  (Before 1.1.3 a
        loop's header was dropped and its body ran exactly once, silently.)"""
        lp, init, body, step = self._loop_parts(node)
        where = self._where(node)
        kind = lp['kind']
        pfx = ' ' * indent

        def run_once():
            self._walk_analog_block(lines, body, active_nodes, indent)
            if step is not None:
                self._walk_analog_block(lines, step, active_nodes, indent)

        if kind == 'repeat':
            n = self._try_eval_expr(lp['cond'])
            if n is None:
                raise EmitError(f"{where}: repeat ({' '.join(lp['cond'].split())}): PyMS needs the "
                                f"repeat count at elaboration (parameters and folded variables)")
            n = int(_py_rint(n))
            if n > self._LOOP_EXACT_LIMIT:
                raise EmitError(f"{where}: repeat count {n} is more than PyMS unrolls "
                                f"({self._LOOP_EXACT_LIMIT})")
            lines.append(f'{pfx}// LOOP repeat x{max(n, 0)}')
            for _ in range(max(n, 0)):
                self._walk_analog_block(lines, body, active_nodes, indent)
            return
        if init is not None:
            self._walk_analog_block(lines, init, active_nodes, indent)
        trips = 0
        while True:
            c = self._loop_cond_value(lp['cond'])
            if c is False:
                lines.append(f'{pfx}// LOOP {kind} ran {trips} times at elaboration: {lp["cond"]}')
                return
            if c is None:
                break
            trips += 1
            if trips > self._LOOP_EXACT_LIMIT:
                raise EmitError(f"{where}: the {kind} loop ({' '.join(lp['cond'].split())}) runs more "
                                f"than {self._LOOP_EXACT_LIMIT} times at elaboration (an endless loop?)")
            run_once()
        why = self._rt_loop_unfit(body, step)
        if why is None:
            self._walk_runtime_loop(lines, node, lp, body, step, active_nodes, indent, trips)
            return
        bound = self._loop_trip_bound(lp['cond'], body, step)
        unroll = self._LOOP_RUNTIME_UNROLL if bound is None else min(bound, self._LOOP_RUNTIME_UNROLL)
        lines.append(f'{pfx}// LOOP {kind}: {trips} iterations at elaboration, then up to {unroll} '
                     f'under runtime indicators (bound {bound}; no run-time loop: {why}): {lp["cond"]}')
        # The first indicator includes the enclosing runtime gates: a loop
        # inside a voltage-dependent branch that is not taken runs no
        # iteration (and its guard stays quiet).
        prev = ' * '.join(f'({g})' for g in self._cond_stack) if self._cond_stack else None
        for k in range(unroll):
            cond_cpp = self._cond_final_cpp(lp['cond'], active_nodes)
            if prev is not None:
                cond_cpp = f'({prev} != 0.0) && ({cond_cpp})'
            S = self._alloc_indicator(lines, cond_cpp, pfx)
            self._condition_registry.append(lp['cond'])
            self._cond_stack.append(S)
            self._runtime_cond_depth += 1
            try:
                run_once()
            finally:
                self._runtime_cond_depth -= 1
                self._cond_stack.pop()
            prev = S
        if bound is None or bound > unroll:
            cond_cpp = self._cond_final_cpp(lp['cond'], active_nodes)
            if prev is not None:
                cond_cpp = f'({prev} != 0.0) && ({cond_cpp})'
            # More iterations than were unrolled is a hard error of the
            # simulation naming the loop (1.1.3 printed a warning and went on
            # with the values after 16: silently wrong).
            msg = (f'{where}: this {kind} loop needs more than the {unroll} iterations PyMS '
                   f'unrolls for it (its body has {why}, so it cannot run as a loop of the '
                   f'eval code); PyMS stops the simulation')
            self._sel_counter += 1
            name = f'_loop_guard_{self._sel_counter}'
            raw = f'_vae_loop_over(({cond_cpp}), "{msg}")'
            esc = raw.replace('\\', '\\\\').replace('"', '\\"')
            lines.append(f'{pfx}symbol {name}("{name}");')
            lines.append(f'{pfx}int_raw[int_syms.size()] = "{esc}";')
            lines.append(f'{pfx}int_syms.push_back({name});')
            lines.append(f'{pfx}int_exprs.push_back({name});')
            lines.append(f'{pfx}int_names.push_back("{name}");')

    # A run-time loop that would go on past this many passes is a hard error
    # of the simulation (an endless loop).
    _LOOP_RUNTIME_MAX = 100000

    def _rt_loop_unfit(self, body: ASTNode, step: Optional[ASTNode]) -> Optional[str]:
        """Why a loop whose condition only the run time knows cannot be a C++
        loop of the eval code (None: it can).  A voltage contribution, a
        ddt() or an event statement in the body needs the walk's static form
        (a branch-current row, the charge split, the analysis shadows) and is
        unrolled instead."""
        nodes = list(_ast_walk(body)) + (list(_ast_walk(step)) if step is not None else [])
        for n in nodes:
            if n.kind == NodeKind.CONTRIB and n.contrib_kind == ContribKind.V:
                return 'a voltage contribution'
            if n.kind == NodeKind.INITIAL_STEP:
                return 'an event statement'
            for t in (n.expr, n.condition, (n.loop or {}).get('cond')):
                if t and (re.search(r'\b(ddt|idt|idtmod)\s*\(', t)
                          or any(v in self._ddt_vars for v in re.findall(r'\b[A-Za-z_]\w*\b', t))):
                    return 'ddt()'
            if n.kind == NodeKind.ASSIGN and n.lhs in self._ddt_vars:
                return 'ddt()'
        return None

    def _walk_runtime_loop(self, lines: list, node: ASTNode, lp: dict, body: ASTNode,
                           step: Optional[ASTNode], active_nodes: list, indent: int, trips: int):
        """A loop whose condition reads the run-time state becomes a C++ loop
        of the eval code (1.1.4): its body is walked ONCE, with every variable
        the body or the step assigns read from a loop variable of the eval
        code (``k__lc7``), and the condition is tested before every pass, so
        the loop runs exactly as often as it says -- no unroll limit.  The
        Jacobian differentiates the eval code itself (dual_ad.py), so
        no derivative of the loop is needed.  A current contribution in the
        body adds to an accumulator that joins the branch after the loop.
        More than _LOOP_RUNTIME_MAX passes is a hard error of the simulation
        naming the loop's file:line (an endless loop).  (1.1.3 unrolled at
        most 16 passes and went on silently with the values after 16.)"""
        pfx = ' ' * indent
        where = self._where(node)
        kind = lp['kind']
        self._sel_counter += 1
        lid = self._sel_counter
        carried = self._assigned_vars(body)
        if step is not None:
            carried = list(dict.fromkeys(carried + self._assigned_vars(step)))
        lines.append(f'{pfx}// LOOP {kind}: {trips} iterations at elaboration, then a run-time loop '
                     f'of the eval code: {lp["cond"]}')
        loop_syms = {}
        for v in carried:
            sym = self._raw_sym(lines, f'{self._safe_sym(v)}__lc{lid}', self._cur_cpp(v), pfx)
            loop_syms[v] = sym
            self._bind_symbol(v, sym)
        acc_pos = len(lines)
        gate = ' * '.join(f'({g})' for g in self._cond_stack) if self._cond_stack else None
        cond_cpp = self._cond_final_cpp(lp['cond'], active_nodes)
        if gate is not None:
            cond_cpp = f'({gate} != 0.0) && ({cond_cpp})'
        msg = (f'{where}: this {kind} loop did not end after {self._LOOP_RUNTIME_MAX} passes '
               f'(an endless loop?); PyMS stops the simulation')
        it = f'_it{lid}'
        self._emit_stmt(lines, f'for (int {it} = 0; ; ++{it}) {{', pfx)
        self._emit_stmt(lines, f'if (!({cond_cpp})) break;', pfx)
        self._emit_stmt(lines, f'if ({it} >= {self._LOOP_RUNTIME_MAX}) {{ _vae_fatal("{msg}"); break; }}', pfx)
        saved = (self._cond_stack, self._runtime_cond_depth)
        loop = dict(lid=lid, acc={}, outer=list(self._cond_stack))
        self._rt_loops.append(loop)
        self._cond_stack, self._runtime_cond_depth = [], 0
        try:
            self._walk_analog_block(lines, body, active_nodes, indent)
            if step is not None:
                self._walk_analog_block(lines, step, active_nodes, indent)
            for v in carried:
                cur = self._cur_cpp(v)
                if cur != loop_syms[v]:
                    self._emit_stmt(lines, f'{loop_syms[v]} = {cur};', pfx)
        finally:
            self._rt_loops.pop()
            self._cond_stack, self._runtime_cond_depth = saved
        self._emit_stmt(lines, '}', pfx)
        for v in carried:
            self._bind_symbol(v, loop_syms[v])
        # the accumulators of the body's current contributions: declared
        # before the loop, added to their branches after it
        decl = []
        for label, acc in loop['acc'].items():
            decl.append(f'{pfx}symbol {acc}("{acc}");')
            decl.append(f'{pfx}int_raw[int_syms.size()] = "0.0";')
            decl.append(f'{pfx}int_syms.push_back({acc});')
            decl.append(f'{pfx}int_exprs.push_back({acc});')
            decl.append(f'{pfx}int_names.push_back("{acc}");')
            decl.append(f'{pfx}_vae_vdep_opaque.push_back({acc});')
            lines.append(f'{pfx}{{ // {label}: the contributions of the loop at {where}')
            lines.append(f'{pfx}    int br_idx = _vae_br(branch_labels, F_contribs, Q_contribs, "{label}");')
            lines.append(f'{pfx}    F_contribs[br_idx] += {acc};')
            lines.append(f'{pfx}}}')
        lines[acc_pos:acc_pos] = decl

    def _count_voltage_conditions(self, node: ASTNode) -> int:
        """Count voltage-dependent conditions in a subtree without emitting code.

        Used to keep condition indices synchronized when a forced condition
        causes a branch to be skipped.  Mirrors the condition resolution
        logic of _walk_analog_block.
        """
        if node is None:
            return 0
        count = 0
        if node.kind == NodeKind.BLOCK:
            for child in node.children:
                count += self._count_voltage_conditions(child)
        elif node.kind == NodeKind.IF:
            resolved = self._eval_condition(node.condition)
            if resolved is True:
                for child in node.children:
                    count += self._count_voltage_conditions(child)
            elif resolved is False:
                if node.else_body:
                    count += self._count_voltage_conditions(node.else_body)
            else:
                resolved_dyn = self._eval_condition_with_vars(node.condition)
                if resolved_dyn is True:
                    for child in node.children:
                        count += self._count_voltage_conditions(child)
                elif resolved_dyn is False:
                    if node.else_body:
                        count += self._count_voltage_conditions(node.else_body)
                else:
                    # This IS a voltage-dependent condition
                    count += 1
                    # Count nested conditions in both branches
                    for child in node.children:
                        count += self._count_voltage_conditions(child)
                    if node.else_body:
                        count += self._count_voltage_conditions(node.else_body)
        return count

    def _register_skipped_conditions(self, node: ASTNode):
        """Register placeholder entries for voltage-dependent conditions
        in a subtree that won't be walked (skipped branch)."""
        if node is None:
            return
        if node.kind == NodeKind.BLOCK:
            for child in node.children:
                self._register_skipped_conditions(child)
        elif node.kind == NodeKind.IF:
            resolved = self._eval_condition(node.condition)
            if resolved is True:
                for child in node.children:
                    self._register_skipped_conditions(child)
            elif resolved is False:
                if node.else_body:
                    self._register_skipped_conditions(node.else_body)
            else:
                resolved_dyn = self._eval_condition_with_vars(node.condition)
                if resolved_dyn is True:
                    for child in node.children:
                        self._register_skipped_conditions(child)
                elif resolved_dyn is False:
                    if node.else_body:
                        self._register_skipped_conditions(node.else_body)
                else:
                    self._condition_registry.append(node.condition)
                    for child in node.children:
                        self._register_skipped_conditions(child)
                    if node.else_body:
                        self._register_skipped_conditions(node.else_body)

    def _predeclare_vars(self, lines: list[str], node: ASTNode, pfx: str):
        """Pre-declare variables that are first assigned inside a runtime condition block."""
        assigned = self._collect_assigned_vars(node)
        for var in assigned:
            if var not in self._declared_vars:
                # Pre-declare as symbol with initial zero value
                self._var_versions[var] = 0
                sym_name = var
                self._var_sym_name[var] = sym_name
                self._declared_vars[var] = 'sym'
                lines.append(f'{pfx}symbol {sym_name}("{sym_name}");')
                lines.append(f'{pfx}int_syms.push_back({sym_name});')
                lines.append(f'{pfx}int_exprs.push_back(ex(0));')
                lines.append(f'{pfx}int_names.push_back("{var}");')

    def _collect_assigned_vars(self, node: ASTNode) -> list[str]:
        """Collect all variable names assigned in this subtree."""
        result = []
        if node is None:
            return result
        if node.kind == NodeKind.ASSIGN:
            result.append(node.lhs)
        if node.kind in (NodeKind.BLOCK, NodeKind.IF):
            for child in node.children:
                result.extend(self._collect_assigned_vars(child))
            if node.kind == NodeKind.IF and node.else_body:
                result.extend(self._collect_assigned_vars(node.else_body))
        return result

    def _emit_contribution(self, lines: list[str], node: ASTNode,
                          active_nodes: list[str], pfx: str):
        """Emit a GiNaC contribution statement."""
        expr_str = node.expr.strip()
        self._check_time_operators(expr_str, node, 'the contribution')
        expr_h = self._maybe_hoist(expr_str, lines, active_nodes, pfx)
        ginac_expr = self._to_ginac_expr(expr_h, active_nodes)
        # ddt() terms, written here or carried in by an inlined variable, are
        # split off below: the static part is the current, each linear term
        # c * ddt(q) the charge c * q (c constant), anything else an error.
        has_ddt = bool(_VAE_DDT_RE.search(ginac_expr))
        # Gate contributions inside a voltage-dependent conditional, exactly like
        # assignments: e.g. `if(sigvds>0) I(di,si)<+ids; else I(si,di)<+ids;`
        # must put ids on ONE branch at runtime. Without this both branches get
        # the full ids and cancel at the shared node -> zero terminal current.
        # (With ddt() the gate is applied to the two parts after the split.)
        sprod = ' * '.join(f'({s})' for s in self._cond_stack) if self._cond_stack else None
        if sprod and not has_ddt:
            # (1.1.9, BUGS.md B24) A select, not a product: the gated value is
            # an intermediate whose VALUE is the C++ ternary (S != 0 ? e : 0),
            # printed from e when the metaprogram runs, so the dual-number
            # Jacobian takes the slope of the taken arm only.  The product
            # e * S was NaN wherever the untaken arm is not finite (1/rs * V
            # * 0 with rs = 0: every point of a V/I switch failed).  The
            # arithmetic form stays the intermediate's expression (its
            # voltage dependence for the analyses that ask).
            self._sel_counter += 1
            gc = f'_gc_{self._sel_counter}'
            lines.append(f'{pfx}symbol {gc}("{gc}");')
            lines.append(f'{pfx}{{ ex _gce = _safe_ex([&]() -> ex {{ return ({ginac_expr}); }});')
            lines.append(f'{pfx}  int_raw[int_syms.size()] = string("((({sprod}) != 0.0) ? (") + to_cpp(_gce) + ") : 0.0)";')
            lines.append(f'{pfx}  int_syms.push_back({gc});')
            lines.append(f'{pfx}  int_exprs.push_back(_safe_ex([&]() -> ex {{ return _gce * ({sprod}); }}));')
            lines.append(f'{pfx}  int_names.push_back("{gc}"); }}')
            ginac_expr = gc
        branch_label = f'{node.contrib_kind.name}({",".join(node.branch)})'

        # A branch with a current unknown (every voltage-contribution
        # branch and every probed branch: collect_current_unknowns) has ONE
        # equation row, labelled by the branch; its contributions accumulate
        # there with the sign of their node order and
        # _finalize_current_rows completes the equation after the walk.
        # Any other current contribution keeps its own KCL row, as before.
        ends = self._branch_endpoints(node.branch, active_nodes)
        rec, sign = None, 1
        if ends is not None:
            rec, sign = self._find_record(ends[0])
        if node.contrib_kind == ContribKind.V:
            if ends is None:
                return  # V(c,c) <+ 0: the short that collapsed an internal node
            if rec is None:
                raise CurrentProbeError(
                    f'{node.loc}: V({",".join(node.branch)}) <+ ...: no branch-current '
                    f'unknown was collected for this contribution (internal error)')
        if rec is not None:
            # The voltage contributions of the branch accumulate in its row
            # BR(..), its current contributions in a side row BRI(..) (no
            # shell label matches it); _finalize_current_rows combines the
            # two into the branch equation by the kinds that were executed.
            if node.contrib_kind == ContribKind.V:
                branch_label = rec['label']
                rec['has_V'] = True
                # the runtime gate of this voltage contribution (None: always
                # executed): the branch is a voltage source while one of its
                # voltage contributions runs
                rec['v_gates'].append(sprod)
            else:
                branch_label = 'BRI' + rec['label'][2:]
                rec['has_I'] = True
            stack = list(self._cond_stack)
            if self._rt_loops:
                # inside a run-time loop: the loop's own gate is part of the path
                stack = (list(self._rt_loops[0]['outer'])
                         + [f'_loop{l["lid"]}' for l in self._rt_loops] + stack)
            rec['order'].append((node.contrib_kind.name, stack))
            if sign < 0:
                ginac_expr = f'(-({ginac_expr}))'

        if self._rt_loops:
            # a current contribution in a run-time loop: this pass's value adds
            # to the loop's accumulator of the branch (_walk_runtime_loop)
            loop = self._rt_loops[-1]
            acc = loop['acc'].get(branch_label)
            if acc is None:
                self._sel_counter += 1
                acc = f'_acc_{self._sel_counter}'
                loop['acc'][branch_label] = acc
            self._sel_counter += 1
            ctr = f'_ctr_{self._sel_counter}'
            lines.append(f'{pfx}symbol {ctr}("{ctr}");')
            lines.append(f'{pfx}int_syms.push_back({ctr});')
            lines.append(f'{pfx}int_exprs.push_back(_safe_ex([&]() -> ex {{ return ({ginac_expr}); }}));')
            lines.append(f'{pfx}int_names.push_back("{ctr}");')
            self._emit_stmt(lines, f'{acc} += {ctr};', pfx)
            return

        self._emit_line_directive(lines, node, pfx)
        lines.append(f'{pfx}{{ // {branch_label}')
        lines.append(f'{pfx}    int br_idx = _vae_br(branch_labels, F_contribs, Q_contribs, "{branch_label}");')
        if has_ddt:
            where = f'{self._where(node)}: {branch_label} <+ {" ".join(expr_str.split())}'
            esc = where.replace('\\', '\\\\').replace('"', '\\"')
            lines.append(f'{pfx}    ex _st = ex(0), _ch = ex(0);  // static part, charge')
            lines.append(f'{pfx}    _vae_split_ddt({ginac_expr}, unk_syms, int_syms, int_exprs, _st, _ch, "{esc}");')
            charge_expr = f'((_ch) * ({sprod}))' if sprod else '_ch'
            ginac_expr = f'((_st) * ({sprod}))' if sprod else '_st'
            lines.append(f'{pfx}    Q_contribs[br_idx] += {charge_expr};')
        # The static part. On a branch with a current unknown it is a term of
        # the branch equation (completed by _finalize_current_rows: for a
        # voltage contribution sum(e) - (V_a - V_b) = 0, the charge parts
        # above in the same row as d/dt); otherwise the KCL current.
        lines.append(f'{pfx}    F_contribs[br_idx] += {ginac_expr};')
        lines.append(f'{pfx}}}')

    # --- Branch currents as unknowns ---

    def _branch_endpoints(self, names, active_nodes: list[str]):
        """Resolve a branch reference — the (p,) / (p, n) tuple of a
        contribution target or the names inside a probe I(p, n) / I(br) —
        to (key, a, b): key is the branch identity, ('br', name) for a named
        branch and ('pair', a, b) for an unnamed one; a and b are the end
        nodes after named-branch and shorted-node resolution (GND for
        ground).  Returns None for a degenerate branch (both ends on one
        node: the V(c,c) <+ 0 left by a collapsed internal node)."""
        branch_neg = getattr(self.mod, 'branch_neg_map', {}) or {}
        p1 = names[0]
        p2 = names[1] if len(names) > 1 else None
        key = None
        if p2 is None and p1 in self.branch_map:
            key = ('br', p1)
            p2 = branch_neg.get(p1)
            p1 = self.branch_map[p1]
        else:
            if p1 in self.branch_map:
                p1 = self.branch_map[p1]
            if p2 and p2 in self.branch_map:
                p2 = self.branch_map[p2]

        def _res(n):
            if n is None or n == GND:
                return GND
            if n not in active_nodes and n in self.internal_nodes:
                t = self._resolve_shorted_node(n, active_nodes)
                return GND if t is None else t
            return n
        a, b = _res(p1), _res(p2)
        if a == b:
            return None
        if key is None:
            key = ('pair', a, b)
        return key, a, b

    def _find_record(self, key):
        """The record of a branch identity and the sign of a reference
        through `key`: -1 when the key names the record's node pair in the
        reverse order (I(n, p) is -I(p, n) on the same unnamed branch).
        (None, 1) when the branch has no current unknown."""
        rec = self._cur_by_key.get(key)
        if rec is not None:
            return rec, 1
        if key[0] == 'pair':
            rec = self._cur_by_key.get(('pair', key[2], key[1]))
            if rec is not None:
                return rec, -1
        return None, 1

    def _add_record(self, key, a, b, raw, loc):
        rec, _sign = self._find_record(key)
        if rec is not None:
            return rec
        inner = ','.join(raw)
        rec = dict(key=key, a=a, b=b, label=f'BR({inner})',
                   sym='Ibr%d_%s' % (len(self._cur_records),
                                      re.sub(r'\W+', '_', inner)),
                   loc=loc, has_V=False, has_I=False, probed=False, v_gates=[],
                   order=[])
        self._cur_records.append(rec)
        self._cur_by_key[key] = rec
        return rec

    def _scan_probes(self, text: Optional[str], active_nodes: list[str], loc):
        if not text:
            return
        for m in _STRING_OR_PROBE_RE.finditer(text):
            if m.group(1) is not None:
                continue
            if m.group(2) or m.group(4):
                raise CurrentProbeError(
                    f'{loc}: {m.group(0).strip()}: a port current probe I(<port>) '
                    f'is not supported by PyMS; the current of a branch, '
                    f'I(a, b) or I(br), is')
            names = (m.group(3),) + ((m.group(5),) if m.group(5) else ())
            ends = self._branch_endpoints(names, active_nodes)
            if ends is not None:
                self._add_record(ends[0], ends[1], ends[2], names, loc)

    def _collect_cur(self, node: Optional[ASTNode], active_nodes: list[str]):
        if node is None:
            return
        if node.kind == NodeKind.CONTRIB:
            if node.contrib_kind == ContribKind.V:
                ends = self._branch_endpoints(node.branch, active_nodes)
                if ends is not None:
                    self._add_record(ends[0], ends[1], ends[2], node.branch, node.loc)
            self._scan_probes(node.expr, active_nodes, node.loc)
        elif node.kind == NodeKind.ASSIGN:
            self._scan_probes(node.expr, active_nodes, node.loc)
        elif node.kind == NodeKind.IF:
            self._scan_probes(node.condition, active_nodes, node.loc)
            for c in node.children:
                self._collect_cur(c, active_nodes)
            self._collect_cur(node.else_body, active_nodes)
        elif node.kind in (NodeKind.BLOCK, NodeKind.INITIAL_STEP):
            for c in node.children:
                self._collect_cur(c, active_nodes)

    def collect_current_unknowns(self, active_nodes: list[str]) -> list[dict]:
        """Walk the WHOLE analog block (every arm of every conditional) and
        record each branch whose current is an unknown of the device: the
        target of a voltage contribution V(a,b) <+ e (the MNA form of a
        voltage source: V_a - V_b - e = 0, the branch current in the two KCL
        rows) and every branch whose current is probed in an emitted
        expression, I(a, b) or I(br) (the current of a branch with current
        contributions is the unknown i with i - sum(e) = 0; a branch with no
        contribution is a short carrying the unknown).  The device shell
        (xyce_device_gen.py) and the math library call this on the same
        module, so both hold the same ordered list with the same row labels;
        the library exports the row of each branch's equation.  A port
        current probe I(<p>) raises CurrentProbeError."""
        self._cur_records = []
        self._cur_by_key = {}
        self._collect_cur(self.mod.analog_block, active_nodes)
        return self._cur_records

    def _lower_probes(self, text: str, active_nodes: list[str], ginac: bool) -> str:
        """Replace every current probe in `text` by its branch-current
        symbol (negated for the reverse node order); a degenerate probe
        I(c,c) is 0.  `ginac` selects the metaprogram form of that zero
        (ex(0)) over the plain C++ one (0.0)."""
        def _repl(m):
            if m.group(1) is not None:
                return m.group(0)
            if m.group(2) or m.group(4):
                raise CurrentProbeError(
                    f'{m.group(0).strip()}: a port current probe I(<port>) is not '
                    f'supported by PyMS; the current of a branch, I(a, b) or I(br), is')
            names = (m.group(3),) + ((m.group(5),) if m.group(5) else ())
            ends = self._branch_endpoints(names, active_nodes)
            if ends is None:
                return 'ex(0)' if ginac else '0.0'
            rec, sign = self._find_record(ends[0])
            if rec is None:
                raise CurrentProbeError(
                    f'{m.group(0).strip()}: no branch-current unknown was collected '
                    f'for this probe (internal error: collect_current_unknowns did '
                    f'not see this expression)')
            rec['probed'] = True
            return f'(-{rec["sym"]})' if sign < 0 else rec['sym']
        return _STRING_OR_PROBE_RE.sub(_repl, text)

    def _finalize_current_rows(self, lines: list[str], pfx: str):
        """After the walk: complete the equation row of every branch-current
        unknown alive in this elaboration (a contribution emitted on it or
        its current read).  The walk left the branch's voltage contributions
        (static FV, charge QV) in its row BR(..) and its current
        contributions (FI, QI) in the side row BRI(..); with i the unknown,
        V = V_a - V_b and s the 0/1 'a voltage contribution runs' indicator:

          voltage contributions only (or a probed branch without any: a
          short):              FV + dQV/dt - s*V - (1 - s)*i = 0
                               (s = 1 when a contribution is ungated; a
                               gated one not executed leaves an open branch)
          current only:        i - FI - dQI/dt = 0
          both, in exclusive arms of one voltage-dependent condition (a
          switch branch):      FV + FI + d(QV + QI)/dt - s*V - (1 - s)*i = 0
          both on one path:    the LRM's switch-branch rule, the kind
                               contributed LAST decides and the other kind's
                               contributions are discarded

        Anything else (both kinds on paths that may overlap) is a clear
        error.  Registers the unknown (symbol, row label, row) for the eval
        preamble, the Jacobian columns and the vae_n_currents /
        vae_current_label / vae_current_row exports the device shell
        reads."""
        for rec in self._cur_records:
            if not (rec['has_V'] or rec['has_I'] or rec['probed']):
                continue  # dead in this elaboration: the shell pins its unknown to 0
            kind = 'V'
            if rec['has_V'] and rec['has_I']:
                vst = [st for k, st in rec['order'] if k == 'V']
                ist = [st for k, st in rec['order'] if k == 'I']
                if all(self._exclusive_stacks(a, b) for a in vst for b in ist):
                    kind = 'switch'
                elif all(st == rec['order'][0][1] for _k, st in rec['order']):
                    kind = rec['order'][-1][0]
                else:
                    raise CurrentProbeError(
                        f'{rec["loc"]}: branch {rec["label"][3:-1]} has both a voltage '
                        f'and a current contribution on paths that are neither one '
                        f'straight path nor exclusive arms of one condition; PyMS does '
                        f'not support this switch branch')
            elif rec['has_I']:
                kind = 'I'
            label, sym = rec['label'], rec['sym']
            ilabel = 'BRI' + label[2:]
            lines.append(f'{pfx}{{ // branch current {label}')
            lines.append(f'{pfx}    int br_idx = _vae_br(branch_labels, F_contribs, Q_contribs, "{label}");')
            lines.append(f'{pfx}    int bri_idx = _vae_find(branch_labels, "{ilabel}");')
            lines.append(f'{pfx}    if (bri_idx >= (int)F_contribs.size()) bri_idx = -1;')
            lines.append(f'{pfx}    ex FV = F_contribs[br_idx], QV = Q_contribs[br_idx], FI = ex(0), QI = ex(0);')
            lines.append(f'{pfx}    if (bri_idx >= 0) {{ FI = F_contribs[bri_idx]; QI = Q_contribs[bri_idx];')
            lines.append(f'{pfx}        F_contribs[bri_idx] = ex(0); Q_contribs[bri_idx] = ex(0); }}')
            if kind == 'I':
                lines.append(f'{pfx}    F_contribs[br_idx] = {sym} - FI;')
                lines.append(f'{pfx}    Q_contribs[br_idx] = -QI;')
            else:
                terms = []
                if rec['a'] != GND:
                    terms.append(f'V_{rec["a"]}')
                if rec['b'] != GND:
                    terms.append(f'- V_{rec["b"]}' if terms else f'(-V_{rec["b"]})')
                vab = f'({" ".join(terms)})'
                gates = rec['v_gates']
                if gates and all(g is not None for g in gates):
                    s = '(1 - ' + ' * '.join(f'(1 - ({g}))' for g in gates) + ')'
                    tail = f' - {s} * {vab} - (1 - {s}) * {sym}'
                else:
                    tail = f' - {vab}'
                if kind == 'switch':
                    lines.append(f'{pfx}    F_contribs[br_idx] = FV + FI{tail};')
                    lines.append(f'{pfx}    Q_contribs[br_idx] = QV + QI;')
                else:
                    lines.append(f'{pfx}    F_contribs[br_idx] = FV{tail};')
            # (one list of the unknowns over the shadows: a row index is the
            # same in every shadow, the branch-label table being shared)
            lines.append(f'{pfx}    if (_vae_find(cur_labels, "{label}") < 0) {{')
            lines.append(f'{pfx}        cur_syms.push_back({sym});')
            lines.append(f'{pfx}        cur_labels.push_back("{label}");')
            lines.append(f'{pfx}        cur_rows.push_back(br_idx);')
            lines.append(f'{pfx}    }}')
            lines.append(f'{pfx}    _alive.insert("{label}");')
            lines.append(f'{pfx}}}')

    @staticmethod
    def _exclusive_stacks(a: list, b: list) -> bool:
        """True when two condition stacks (lists of indicator factors, see
        _cond_stack) cannot both be 1: after a common prefix one holds S and
        the other (1 - S)."""
        for x, y in zip(a, b):
            if x == y:
                continue
            return x == f'(1 - {y})' or y == f'(1 - {x})'
        return False

    # --- Expression analysis ---

    def _expr_uses_voltage(self, expr: str) -> bool:
        """Check if expression references node voltages (V(x,y)) or voltage-dependent vars."""
        if expr is None:
            return False
        # Direct voltage reference
        if re.search(r'[VI]\s*\(', expr):
            return True
        if 'Temp' in expr and re.search(r'Temp\s*\(', expr):
            return True  # Temp(t) is a node voltage
        if '$vt' in expr or '$temperature' in expr:
            return True
        if '$abstime' in expr or '$realtime' in expr:
            return True   # the simulation time: a value of the run time
        # Check if any referenced variable is voltage-dependent
        # (declared as symbol AND not known as a constant in var_values)
        idents = set(re.findall(r'\b[A-Za-z_]\w*\b', expr))
        for ident in idents:
            if ident in self._declared_vars and ident not in self.var_values:
                return True
            if ident in self._ddt_vars:
                return True  # a value with ddt() in it moves with the voltages
        return False

    # --- Expression translation ---

    def _to_cpp_expr(self, expr: str, active_nodes: list[str]) -> str:
        """Translate Verilog-A expression to plain C++ (double arithmetic)."""
        if expr is None:
            return '0.0'

        result = expr

        # Normalize operators
        result = result.replace('= =', '==').replace('! =', '!=')
        result = result.replace('> =', '>=').replace('< =', '<=')
        result = result.replace('& &', '&&').replace('| |', '||')

        # V(a,b) → shouldn't appear here (checked by _expr_uses_voltage)
        # $vt, $temperature, Temp() — same

        # Function mapping (VA → C++ with std:: prefix to avoid GiNaC overloads)
        # Apply longer names first, then general ones; use negative lookbehind
        # to avoid re-prefixing already-prefixed names
        _cpp_funcs = [
            ('limexp', 'std::exp'), ('lexp', 'std::exp'), ('lln', 'std::log'),
            ('ln', 'std::log'), ('abs', 'std::fabs'),
            ('sqrt', 'std::sqrt'), ('pow', 'std::pow'),
            ('exp', 'std::exp'), ('log', 'std::log'),
            ('sin', 'std::sin'), ('cos', 'std::cos'),
            ('tan', 'std::tan'), ('tanh', 'std::tanh'),
            ('atan2', 'std::atan2'), ('atan', 'std::atan'),
            ('hypot', 'std::hypot'),
            ('min', 'std::fmin'), ('max', 'std::fmax'),
        ]
        for va_name, cpp_name in _cpp_funcs:
            result = re.sub(rf'(?<!:)\b{va_name}\b(?=\s*\()', cpp_name, result)

        # `DEFINE → identifier
        result = re.sub(r'`(\w+)', r'\1', result)

        # Noise / ddx → 0
        def _strip_call(name, s):
            pattern = rf'\b{name}\s*\('
            while True:
                m = re.search(pattern, s)
                if not m: break
                start = m.end(); depth = 1; i = start
                while i < len(s) and depth > 0:
                    if s[i] == '(': depth += 1
                    elif s[i] == ')': depth -= 1
                    i += 1
                s = s[:m.start()] + '0.0' + s[i:]
            return s
        result = _strip_call('white_noise', result)
        result = _strip_call('flicker_noise', result)
        result = _strip_call('ddx', result)

        # Ternary evaluation
        def _eval_ternary(s):
            pattern = r'\(([^?]+)\?\s*([^:]+):\s*([^)]+)\)'
            while True:
                m = re.search(pattern, s)
                if not m: break
                cond_str = m.group(1).strip()
                py_cond = re.sub(r'(?<!=)!(?!=)', ' not ', cond_str.replace('&&', ' and ').replace('||', ' or '))
                try:
                    if eval(py_cond, {"__builtins__": {}}, self._cond_ns(with_vars=False)):
                        s = s[:m.start()] + f'({m.group(2).strip()})' + s[m.end():]
                    else:
                        s = s[:m.start()] + f'({m.group(3).strip()})' + s[m.end():]
                except Exception:
                    break
            return s
        result = _eval_ternary(result)

        # Substitute known constant params with values
        param_set = set(self.param_values.keys())
        var_set = set(self.var_values.keys())
        _skip_cpp = {'fabs', 'fmin', 'fmax', 'pow', 'exp', 'log', 'sqrt',
                     'sin', 'cos', 'tan', 'tanh', 'atan', 'atan2', 'hypot',
                     'hypsmooth', 'hypmax', 'Tempdep'}
        def _subst(m):
            name = m.group(0)
            if name in _skip_cpp:
                return name
            if name in self.param_values:
                return repr(self.param_values[name])
            if name in self.var_values:
                return repr(self.var_values[name])
            return name
        result = re.sub(r'\b[A-Za-z_]\w*\b', _subst, result)

        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def _to_ginac_expr(self, expr: str, active_nodes: list[str]) -> str:
        """Translate Verilog-A expression to GiNaC C++ expression."""
        if expr is None:
            return 'ex(0)'

        result = expr

        # Normalize tokenizer-split operators in expressions too
        result = result.replace('= =', '==')
        result = result.replace('! =', '!=')
        result = result.replace('> =', '>=')
        result = result.replace('< =', '<=')
        result = result.replace('& &', '&&')
        result = result.replace('| |', '||')

        # V(a,b) → (V_a - V_b), V(a) → V_a, grounded nodes → 0.
        # Verilog-A also allows V(named_branch) for a branch declared
        # with two endpoints (``branch (a, b) name;``). The parser
        # stores those endpoints in branch_map (primary) +
        # branch_neg_map (secondary); fold both in so the contribution
        # actually depends on the differential voltage rather than
        # just the primary node — the singular-Jacobian symptom on
        # internal nodes traces back to this drop.
        branch_neg = getattr(self.mod, 'branch_neg_map', {}) or {}
        def repl_v(m):
            p1 = m.group(1).strip()
            p2 = m.group(2).strip() if m.group(2) else None
            # Resolve branch names to their nodes. If a single-arg V()
            # references a named branch with a secondary endpoint,
            # synthesize the second argument from branch_neg_map.
            if p2 is None and p1 in self.branch_map and p1 in branch_neg:
                p2 = branch_neg[p1]
            if p1 in self.branch_map:
                p1 = self.branch_map[p1]
            if p2 and p2 in self.branch_map:
                p2 = self.branch_map[p2]
            if p1 not in active_nodes and p1 in self.internal_nodes:
                p1 = self._resolve_shorted_node(p1, active_nodes)
            if p2 and p2 not in active_nodes and p2 in self.internal_nodes:
                p2 = self._resolve_shorted_node(p2, active_nodes)
            # Handle grounded nodes (resolved to None)
            p1_expr = 'ex(0)' if p1 is None else f'V_{p1}'
            p2_expr = 'ex(0)' if p2 is None else f'V_{p2}' if p2 else None
            if p2_expr:
                return f'({p1_expr} - {p2_expr})'
            return p1_expr
        result = re.sub(r'V\s*\(\s*(\w+)\s*(?:,\s*(\w+)\s*)?\)', repl_v, result)

        # I(a,b) / I(br) probes → the branch-current unknown's symbol (a
        # branch whose only contribution is noise has i = 0 at dc/tran)
        result = self._lower_probes(result, active_nodes, ginac=True)

        # $limit(expr, ...) → just expr
        result = re.sub(r'\$limit\s*\(\s*([^,]+),.*?\)', r'\1', result)

        # Temp(node-or-branch) → V_<resolved-node>. Verilog-A allows
        # passing a branch label here (e.g. ``Temp(b_rth)`` where
        # ``branch (dt) b_rth;``); resolve via branch_map first so we
        # don't emit an undefined ``V_b_rth`` symbol.
        def _repl_temp(m):
            name = m.group(1)
            if name in self.branch_map:
                name = self.branch_map[name]
            return f'V_{name}'
        result = re.sub(r'Temp\s*\(\s*(\w+)\s*\)', _repl_temp, result)

        # Pwr(branch) → 0 in eval context (thermal-power contribution
        # acts as an internal source the wrapper doesn't model
        # directly). Same shape as the I(branch) probe rule above.
        result = re.sub(r'Pwr\s*\(\s*\w+\s*\)', 'ex(0)', result)

        # $vt(expr) -> (k/q * (expr)), $vt -> _vae_vt, $temperature -> _vae_T
        result = sub_thermal(result)
        # $abstime: the simulation time the device shell passes in the state
        # (0 at a dc operating point); a symbol of its own, constant to the
        # node voltages (BUGS.md B14: it reached g++ verbatim before 1.1.4)
        result = re.sub(r'\$(abstime|realtime)\b', '_vae_abstime', result)
        # analysis("name", ...): a constant of the walk's analysis context
        result = self._sub_analysis(result)

        # $mfactor — Verilog-A parallel-multiplier system parameter.
        # Default to 1.0 (single device); multi-instance scaling would
        # need a runtime hook the wrapper doesn't expose.
        result = re.sub(r'\$mfactor\b', '1.0', result)

        # $port_connected(x) / $param_given(x) in expression context.
        # Conditions handle these via _preprocess_sys_funcs; the same
        # rewrite has to apply when they appear inside ordinary
        # expressions (some compact models gate scaling math on
        # ``$param_given(W) ? ... : ...``).
        result = re.sub(r'\$port_connected\s*\(\s*\w+\s*\)', '1', result)
        def _pg_expr(m):
            return '1' if m.group(1) in self._given_params else '0'
        result = re.sub(r'\$param_given\s*\(\s*(\w+)\s*\)', _pg_expr, result)

        # $simparam("name"[, default]) -> the simulator's value (the old BUGS.md B10)
        result = self._sub_simparam(result)

        # Function mapping
        for va_name, ginac_name in _VA_TO_GINAC.items():
            result = re.sub(rf'\b{va_name}\b(?=\s*\()', ginac_name, result)
        # integer / % and the integer rounding -> the metaprogram's GiNaC functions
        result = re.sub(r'\b_vae_(idiv|imod|rint)\b(?=\s*\()', r'vae_\1', result)
        # min/max → _vae_min/_vae_max (GiNaC-compatible helpers, avoid name collision)
        result = re.sub(r'\bmin\b(?=\s*\()', '_vae_min', result)
        result = re.sub(r'\bmax\b(?=\s*\()', '_vae_max', result)
        # ddt(q, tol): keep the derivative's argument alone
        result = self._drop_ddt_tolerance(result)

        # `DEFINE references → identifier
        result = re.sub(r'`(\w+)', r'\1', result)

        # Noise → 0 (handle nested parens and string args like "shot")
        def _strip_noise_call(name, s):
            pattern = rf'\b{name}\s*\('
            while True:
                m = re.search(pattern, s)
                if not m:
                    break
                # Find matching close paren
                start = m.end()
                depth = 1
                i = start
                while i < len(s) and depth > 0:
                    if s[i] == '(':
                        depth += 1
                    elif s[i] == ')':
                        depth -= 1
                    i += 1
                s = s[:m.start()] + 'ex(0)' + s[i:]
            return s
        result = _strip_noise_call('white_noise', result)
        result = _strip_noise_call('flicker_noise', result)

        # ddx(expr, V_x) → 0 (derivative operator — not needed for eval)
        result = _strip_noise_call('ddx', result)

        # Evaluate ternary (cond ? a : b) where cond is constant
        def _find_ternary(s):
            """Find a paren-balanced ternary (cond ? true : false) and return
            (start, end, cond_str, true_str, false_str) or None."""
            # Find '(' followed eventually by '?' — scan for balanced match
            idx = 0
            while idx < len(s):
                start = s.find('(', idx)
                if start < 0:
                    break
                # Scan forward for '?' at depth 0 (relative to this open paren)
                depth = 1
                i = start + 1
                q_pos = -1
                colon_pos = -1
                while i < len(s) and depth > 0:
                    ch = s[i]
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth == 0:
                            break
                    elif ch == '?' and depth == 1 and q_pos < 0:
                        q_pos = i
                    elif ch == ':' and depth == 1 and q_pos >= 0 and colon_pos < 0:
                        colon_pos = i
                    i += 1
                if depth == 0 and q_pos > 0 and colon_pos > 0:
                    end = i + 1  # past the closing ')'
                    cond_str = s[start+1:q_pos].strip()
                    true_str = s[q_pos+1:colon_pos].strip()
                    false_str = s[colon_pos+1:i].strip()
                    return (start, end, cond_str, true_str, false_str)
                idx = start + 1
            return None

        def _eval_ternary(s):
            while True:
                t = _find_ternary(s)
                if t is None:
                    break
                start, end, cond_str, true_val, false_val = t
                # Try to evaluate the condition
                py_cond = cond_str.replace('&&', ' and ').replace('||', ' or ')
                py_cond = re.sub(r'(?<!=)!(?!=)', ' not ', py_cond)
                # Replace numeric("X") with X for eval
                py_cond = re.sub(r'numeric\("([^"]+)"\)', r'\1', py_cond)
                try:
                    ns = self._cond_ns()
                    result_val = eval(py_cond, {"__builtins__": {}}, ns)
                    replacement = true_val if result_val else false_val
                    s = s[:start] + f'({replacement})' + s[end:]
                except Exception:
                    break  # Can't evaluate — leave as-is
            return s
        result = _eval_ternary(result)

        # Single-pass substitution of known constant params and vars
        # Build the substitutable set (exclude funcs, keywords, node voltages)
        _skip = set(_VA_TO_GINAC.values()) | {'numeric', 'ex', VT_SYM, T_SYM}
        _skip |= {f'V_{n}' for n in self.all_nodes}
        _skip |= {'_vae_min', '_vae_max', 'hypsmooth', 'hypmax', 'Tempdep',
                  'vae_idiv', 'vae_imod', 'vae_rint'}
        _skip |= self.instance_params
        def _subst_known(m):
            name = m.group(0)
            is_call = m.string[m.end():m.end() + 1] == '('
            # A reserved GiNaC/C++ identifier used as a CALL — ex(0), exp(x),
            # _vae_min(...) — stays verbatim; but the SAME name used as a
            # VARIABLE (e.g. PSP103's `real ex`) must map to its (mangled)
            # symbol, else `1.0 / ex` collides with the GiNaC `ex` type.
            if name in _skip and is_call:
                return name
            if name in self.param_values:
                return repr(self.param_values[name])
            if name in self.var_values:
                return repr(self.var_values[name])
            # A variable carrying ddt(): its translated text, inlined
            if name in self._ddt_vars and not is_call:
                return self._ddt_vars[name]
            # Use versioned symbol name for non-constant variables
            if name in self._var_sym_name:
                return self._var_sym_name[name]
            # Declared local with no value on any reached path → 0 (VA reals
            # zero-init). Its assignment was compile-time skipped (guarded by a
            # false `if`, or behind an undefined `ifdef); emitting the bare name
            # would leave a dangling identifier the eval C++ can't resolve.
            if name in self._declared_var_names:
                return '0'
            return name
        result = re.sub(r'\b[A-Za-z_]\w*\b', _subst_known, result)

        # Re-run ternary evaluation after substitution (catches cases where
        # condition identifiers were substituted to constants)
        result = _eval_ternary(result)

        # Convert remaining ternaries to gmax/gmin:
        #   (a > b ? a : b) → gmax(a, b)     i.e. max(a,b)
        #   (a < b ? a : b) → gmin(a, b)     i.e. min(a,b)
        #   (a > b ? b : a) → gmin(a, b)     i.e. min(a,b)
        #   (a < b ? b : a) → gmax(a, b)     i.e. max(a,b)
        # These come from clip()/max()/min() macros in compact models.
        def _ternary_to_minmax(s):
            while True:
                t = _find_ternary(s)
                if t is None:
                    break
                start, end, cond_str, true_val, false_val = t
                # Normalize whitespace for matching
                c = ' '.join(cond_str.split())
                tv = ' '.join(true_val.split())
                fv = ' '.join(false_val.split())
                # Try to match (a) > (b) or (a) < (b)
                m = re.match(r'^(.+?)\s*([><])\s*(.+)$', c)
                if m:
                    lhs = ' '.join(m.group(1).strip().strip('()').strip().split())
                    op = m.group(2)
                    rhs = ' '.join(m.group(3).strip().strip('()').strip().split())
                    tv_n = ' '.join(tv.strip().strip('()').strip().split())
                    fv_n = ' '.join(fv.strip().strip('()').strip().split())
                    if op == '>' and lhs == tv_n and rhs == fv_n:
                        s = s[:start] + f'_vae_max({true_val}, {false_val})' + s[end:]
                        continue
                    if op == '>' and lhs == fv_n and rhs == tv_n:
                        s = s[:start] + f'_vae_min({true_val}, {false_val})' + s[end:]
                        continue
                    if op == '<' and lhs == tv_n and rhs == fv_n:
                        s = s[:start] + f'_vae_min({true_val}, {false_val})' + s[end:]
                        continue
                    if op == '<' and lhs == fv_n and rhs == tv_n:
                        s = s[:start] + f'_vae_max({true_val}, {false_val})' + s[end:]
                        continue
                # Not a simple min/max — leave for ex() wrapping below
                break
            return s

        # Nested clip(x, lo, hi) produces nested ternaries — iterate
        for _ in range(5):
            prev = result
            result = _ternary_to_minmax(result)
            if result == prev:
                break

        # Wrap bare numeric literals adjacent to ? and : with ex() for
        # GiNaC type compatibility in any remaining ternaries.
        # Numbers may be wrapped in parens from macro expansion: ( 1.0e26 )
        _num = r'-?[\d.]+[eE]?[+-]?\d*'
        def _wrap_ternary_numerics(s):
            s = re.sub(rf'\?\s*\(?\s*({_num})\s*\)?\s*:', lambda m: f'? ex({m.group(1)}) :', s)
            s = re.sub(rf':\s*\(?\s*({_num})\s*\)?\s*\)', lambda m: f': ex({m.group(1)}) )', s)
            return s
        result = _wrap_ternary_numerics(result)
        result = _ternary_to_minmax(result)

        # Guard: division by constant zero → replace expression with ex(0)
        # Models like BSIM6 divide by params that can be zero (e.g. NJSD=0)
        # but multiply the result by zero later. GiNaC crashes on the division.
        result = re.sub(r'/\s*0\.0\b', '/ ex(1) * ex(0) ', result)
        result = re.sub(r'/\s*\(\s*0\.0\s*\)', '/ (ex(1)) * ex(0) ', result)

        result = self._real_numeric_call_args(result)
        # the run-time message site of each integer / and % (1.1.4)
        result = self._add_div_sites(result)

        result = re.sub(r'\s+', ' ', result).strip()
        return result

    # Math functions both <cmath> and GiNaC declare: with an integer literal
    # among all-numeric arguments neither overload is better and g++ refuses
    # the call ("call of overloaded 'pow(int, double)' is ambiguous",
    # DIODE_CMC 2.0's pow(2, -1.0 / MJ) once MJ is substituted).
    _NUMERIC_CALL_RE = re.compile(
        r'\b(pow|exp|log|sqrt|sin|cos|tan|tanh|atan|atan2|hypot|abs)\s*\(')
    _NUMERIC_CALL_NAMES = frozenset(
        {'pow', 'exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'tanh', 'atan',
         'atan2', 'hypot', 'abs', 'double'})
    _NUMBER_RE = re.compile(r'(?<![\w.])(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')

    @classmethod
    def _real_numeric_call_args(cls, s: str) -> str:
        """Give every argument of a math call whose arguments are all numeric
        (no symbol, no variable) the type double, so that <cmath>'s real
        function is the one called: pow ( 2 , -1.0 / 0.5 ) -> pow ( double(2)
        , double(-1.0 / 0.5) ). The cast keeps C++'s value of each argument
        (an integer quotient stays truncated, as Verilog-A has it); a call
        with a symbolic argument is left as it is. Innermost calls first, so
        a constant call inside a constant call counts as numeric."""
        starts = [m for m in cls._NUMERIC_CALL_RE.finditer(s)]
        for m in reversed(starts):
            depth, j, args, a0 = 1, m.end(), [], m.end()
            while j < len(s) and depth:
                c = s[j]
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        args.append((a0, j))
                elif c == ',' and depth == 1:
                    args.append((a0, j))
                    a0 = j + 1
                j += 1
            if depth or not args:
                continue
            texts = [s[a:b] for a, b in args]
            if any(not t.strip() or
                   set(re.findall(r'\b[A-Za-z_]\w*\b', t)) - cls._NUMERIC_CALL_NAMES
                   for t in texts):
                continue
            # Only an integer literal makes the call ambiguous: with real
            # literals alone <cmath>'s double overload is chosen already.
            if not any(re.fullmatch(r'\d+', num)
                       for t in texts for num in cls._NUMBER_RE.findall(t)):
                continue
            new = ','.join(f' double({t.strip()}) ' for t in texts)
            s = s[:m.end()] + new + s[args[-1][1]:]
        return s

    def _try_eval_expr(self, expr: str) -> Optional[float]:
        """Try to evaluate expression as a constant float.

        Returns float value if all referenced params/vars are known constants,
        None otherwise.
        """
        if expr is None:
            return None

        s = expr
        # $simparam("name"[, default]) -> the simulator's value (before $ check)
        s = self._sub_simparam(s)
        s = self._sub_analysis(s)
        # Skip if contains V(), I(), $vt, etc.
        if re.search(r'[VI]\s*\(', s) or '$' in s:
            return None

        # Check all identifiers are resolvable
        idents = set(re.findall(r'[A-Za-z_]\w*', s))
        # Remove function names
        idents -= set(_VA_TO_GINAC.keys())
        idents -= {'ln', 'limexp', 'abs', 'pow', 'exp', 'log', 'sqrt',
                    'sin', 'cos', 'tan', 'tanh', 'atan', 'atan2', 'hypot',
                    'min', 'max', 'ddt', 'e', 'E',
                    'hypsmooth', 'hypmax', 'Tempdep', 'lexp', 'lln',
                    'floor', 'ceil', 'fabs'}
        idents -= set(_INT_HELPERS_PY)

        ns = {}
        for ident in idents:
            if ident in self.param_values and ident not in self.instance_params:
                ns[ident] = self.param_values[ident]
            elif ident in self._string_values:
                ns[ident] = self._string_values[ident]
            elif ident in self.var_values:
                ns[ident] = self.var_values[ident]
            else:
                return None  # Unknown or instance-dependent

        # Add math functions
        import math
        ns.update({
            'ln': math.log, 'log': math.log, 'exp': math.exp,
            'sqrt': math.sqrt, 'pow': math.pow, 'abs': abs, 'fabs': abs,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'tanh': math.tanh, 'atan': math.atan, 'atan2': math.atan2,
            'hypot': math.hypot, 'min': min, 'max': max,
            'limexp': math.exp, 'lexp': math.exp, 'lln': math.log,
            'floor': math.floor, 'ceil': math.ceil,
        })
        # Add model-defined functions
        def _hypsmooth(x, c):
            return (x + math.sqrt(x*x + 4*c*c)) / 2
        def _hypmax(x, xmin, c):
            d = x - xmin - c
            return xmin + (d + math.sqrt(d*d - 4*xmin*c)) / 2
        def _tempdep(PARAML, PARAMT, DELTEMP, TEMPMOD):
            if TEMPMOD != 0:
                return PARAML + _hypmax(PARAMT * DELTEMP, -PARAML, 1e-6)
            return PARAML * _hypsmooth(1 + PARAMT * DELTEMP - 1e-6, 1e-3)
        ns['hypsmooth'] = _hypsmooth
        ns['hypmax'] = _hypmax
        ns['Tempdep'] = _tempdep
        ns.update(_INT_HELPERS_PY)

        # Translate operators (of the text with $simparam substituted: the
        # original still had the $simparam call, which Python cannot read, so
        # a $simparam expression was never folded to a constant)
        py = s.replace('&&', ' and ').replace('||', ' or ')

        # Convert C ternary (cond ? a : b) to Python (a if cond else b)
        def _c_ternary_to_py(s):
            while True:
                # Find paren-balanced ternary
                idx = 0
                found = False
                while idx < len(s):
                    start = s.find('(', idx)
                    if start < 0:
                        break
                    depth = 1
                    i = start + 1
                    q_pos = -1
                    colon_pos = -1
                    while i < len(s) and depth > 0:
                        ch = s[i]
                        if ch == '(':
                            depth += 1
                        elif ch == ')':
                            depth -= 1
                            if depth == 0:
                                break
                        elif ch == '?' and depth == 1 and q_pos < 0:
                            q_pos = i
                        elif ch == ':' and depth == 1 and q_pos >= 0 and colon_pos < 0:
                            colon_pos = i
                        i += 1
                    if depth == 0 and q_pos > 0 and colon_pos > 0:
                        cond = s[start+1:q_pos].strip()
                        true_v = s[q_pos+1:colon_pos].strip()
                        false_v = s[colon_pos+1:i].strip()
                        s = s[:start] + f'(({true_v}) if ({cond}) else ({false_v}))' + s[i+1:]
                        found = True
                        break
                    idx = start + 1
                if not found:
                    break
            return s
        py = _c_ternary_to_py(py)

        try:
            result = eval(py, {"__builtins__": {}}, ns)
            return float(result)
        except ZeroDivisionError as e:
            if '_vae_idiv' in py or '_vae_imod' in py:
                raise EmitError(f'{" ".join(expr.split())}: {e} (integer arithmetic, LRM 4.2)')
            return None
        except Exception:
            return None

    # --- Node resolution ---

    def _short_classes(self) -> dict:
        """The node collapse of the module: internal node -> the node it
        merges into (GND for ground), for every internal node that does
        not survive.

        Only a short ``V(a, b) <+ 0`` (or ``V(a) <+ 0``) that runs
        UNCONDITIONALLY collapses: it is the same in every elaboration, so
        the device shell (which decides its node list once per module, from
        no instance) and the math library (built per parameter set) agree on
        the node list (the old BUGS.md D4).  A short inside a conditional -- a
        parameter-dependent ``if (RCBLI > 0) ... else V(c4, c1) <+ 0`` -- is
        a 0 V source on its branch-current unknown instead
        (collect_current_unknowns / _finalize_current_rows).  The nodes of
        one class of shorts merge into ONE survivor: the first port of the
        class, else ground, else the first internal node declared.  (The
        old rule removed both internal ends of a short, and the contributions
        of the class all went to ground: the silent 0 of BUGS.md CHANGES 1.1.1.)"""
        cached = getattr(self, '_short_map_cache', None)
        if cached is not None:
            return cached
        branch_neg = getattr(self.mod, 'branch_neg_map', {}) or {}
        parent: dict = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def norm(n):
            if n is None or n == GND:
                return GND
            return n

        def ends(names):
            p1 = names[0]
            p2 = names[1] if len(names) > 1 else None
            if p2 is None and p1 in self.branch_map:
                return norm(self.branch_map[p1]), norm(branch_neg.get(p1))
            return (norm(self.branch_map.get(p1, p1)),
                    norm(self.branch_map.get(p2, p2) if p2 is not None else None))

        # A short whose current is read anywhere (I(br), I(a, b) on its node
        # pair, either order) is no collapse: it stays a 0 V source on its
        # branch-current unknown, which the probe reads (an ammeter).
        # Collapsing it left the probe on a degenerate branch, read as 0.
        probed = set()
        for st in _ast_walk(self.mod.analog_block):
            for text in (st.expr, st.condition, st.text):
                for m in _STRING_OR_PROBE_RE.finditer(text or ''):
                    if m.group(1) is None and not (m.group(2) or m.group(4)):
                        probed.add(frozenset(ends((m.group(3),) + ((m.group(5),) if m.group(5) else ()))))

        def walk(node):
            if node is None:
                return
            if node.kind == NodeKind.BLOCK and node.loop is None:
                for c in node.children:
                    walk(c)
            elif (node.kind == NodeKind.CONTRIB and node.contrib_kind == ContribKind.V
                  and node.expr.strip() in ('0', '0.0')):
                p1, p2 = ends(node.branch)
                if frozenset((p1, p2)) in probed:
                    return
                ra, rb = find(p1), find(p2)
                if ra != rb:
                    parent[ra] = rb
            # an IF, CASE or loop body runs conditionally: no collapse there

        walk(self.mod.analog_block)
        members: dict = {}
        for n in list(parent):
            members.setdefault(find(n), []).append(n)
        result = {}
        for group in members.values():
            ports = [p for p in self.port_names if p in group]
            if ports:
                surv = ports[0]
            elif GND in group:
                surv = GND
            else:
                surv = next(n for n in self.internal_nodes if n in group)
            for n in group:
                if n != surv and n in self.internal_nodes:
                    result[n] = surv
        self._short_map_cache = result
        return result

    def _resolve_shorted_node(self, node: str, active_nodes: list[str]) -> Optional[str]:
        """The node a collapsed internal node merges into (_short_classes);
        None when it merges into ground.  A node that did not collapse is
        returned unchanged."""
        t = self._short_classes().get(node) if self.collapse_nodes else None
        if t is None:
            return node if node in active_nodes else None
        return None if t == GND else t

    def _find_short_target(self, ast_node: ASTNode, internal: str) -> Optional[str]:
        """The node an internal node is shorted to (for the SHORT comments)."""
        t = self._short_classes().get(internal)
        return t

    def _find_shorted_nodes(self, node: ASTNode) -> set[str]:
        """The internal nodes that collapse into another node (_short_classes).
        `node` is kept for the callers; the analysis covers the module's
        whole analog block."""
        return set(self._short_classes())

    # --- Condition evaluator (plain C++ for regime key computation) ---

    def emit_condition_evaluator(self, active_nodes: list[str],
                                 analysis: 'RegimeAnalysis') -> str:
        """Generate a standalone C++ function that evaluates regime conditions.

        Walks the analog block exactly like _walk_analog_block, but instead
        of emitting GiNaC expressions, emits plain C++ double arithmetic.
        For parameter-resolved IF nodes, takes the resolved branch.
        For assume_true IF nodes, takes the true branch.
        For voltage-dependent IF nodes (regime conditions), emits a real
        C++ `if` that sets a bit in the regime key.

        Args:
            active_nodes: List of active node names (ports + non-shorted internals).
            analysis: RegimeAnalysis with condition-to-index mapping.

        Returns:
            C++ source code for a standalone evaluator function.
        """
        lines = []
        lines.append('// AUTO-GENERATED by VAE — regime condition evaluator')
        lines.append(f'// Source model: {self.mod.name}')
        lines.append(f'// {analysis.n_conditions} voltage-dependent conditions')
        lines.append('#include <cmath>')
        lines.append('#include <cstdint>')
        lines.append('#include <cstdio>')
        lines.append('')

        # Inline helper functions
        lines.append('static inline double hypsmooth(double x, double c) {')
        lines.append('    return 0.5 * (x + std::sqrt(x * x + 4.0 * c * c));')
        lines.append('}')
        lines.append('')
        lines.append('static inline double hypmax(double x, double xmin, double c) {')
        lines.append('    double d = x - xmin - c;')
        lines.append('    return xmin + 0.5 * (d + std::sqrt(d * d - 4.0 * xmin * c));')
        lines.append('}')
        lines.append('')
        lines.append('static inline double Tempdep(double PARAML, double PARAMT, double DELTEMP, double TEMPMOD) {')
        lines.append('    if (TEMPMOD != 0.0)')
        lines.append('        return PARAML + hypmax(PARAMT * DELTEMP, -PARAML, 1e-6);')
        lines.append('    return PARAML * hypsmooth(1.0 + PARAMT * DELTEMP - 1e-6, 1e-3);')
        lines.append('}')
        lines.append('')
        lines.append('static inline double lexp(double x) {')
        lines.append('    return (x < 80.0) ? std::exp(x) : std::exp(80.0) * (1.0 + x - 80.0);')
        lines.append('}')
        lines.extend(_INT_HELPERS_CPP)
        lines.append('')
        lines.append('static inline double lln(double x) {')
        lines.append('    return (x > 1e-80) ? std::log(x) : std::log(1e-80) + (x - 1e-80) / 1e-80;')
        lines.append('}')
        lines.append('')

        # Condition name table
        lines.append(f'static const int N_CONDITIONS = {analysis.n_conditions};')
        lines.append('static const char* condition_names[] = {')
        for c in analysis.conditions:
            escaped = c.condition.replace('\\', '\\\\').replace('"', '\\"')
            lines.append(f'    "{escaped}",  // [{c.index}]')
        lines.append('};')
        lines.append('')

        # Function signature
        # Only inject Vt from argument if the model doesn't define it internally
        _model_vars = {v.name for v in self.mod.variables}
        lines.append('extern "C" uint64_t vae_eval_regime(double* V, int n_nodes, double _Vt_arg) {')
        lines.append(f'    double {VT_SYM} = _Vt_arg;')
        lines.append('    uint64_t key = 0;')
        lines.append('')

        # Node voltage unpacking
        lines.append('    // Unpack node voltages')
        for i, n in enumerate(active_nodes):
            lines.append(f'    double V_{n} = V[{i}];')

        # Temperature: find the temperature node by name ('t' in BSIM-CMG)
        temp_idx = None
        for i, n in enumerate(active_nodes):
            if n == 't':
                temp_idx = i
                break
        if temp_idx is not None:
            lines.append(f'    double {T_SYM} = V[{temp_idx}];  // temperature node (V_{active_nodes[temp_idx]})')
        else:
            lines.append(f'    double {T_SYM} = _Vt_arg / {VAE_KOQ};  // from the Vt it is given (no thermal node)')
        lines.append('')

        # Reset var_values for the walk (keep param_values intact)
        saved_var_values = dict(self.var_values)
        saved_declared = dict(self._declared_vars)
        saved_runtime_depth = self._runtime_cond_depth
        saved_versions = dict(self._var_versions)
        saved_sym_name = dict(self._var_sym_name)
        saved_registry = list(self._condition_registry)
        saved_ddt_vars = dict(self._ddt_vars)

        self.var_values = {}
        self._declared_vars = {}
        self._runtime_cond_depth = 0
        self._var_versions = {}
        self._var_sym_name = {}
        self._condition_registry = []
        self._ddt_vars = {}

        # Walk the AST
        self._walk_condition_eval(lines, self.mod.analog_block,
                                  active_nodes, analysis, indent=4)

        # Restore state
        self.var_values = saved_var_values
        self._declared_vars = saved_declared
        self._runtime_cond_depth = saved_runtime_depth
        self._var_versions = saved_versions
        self._var_sym_name = saved_sym_name
        self._condition_registry = saved_registry
        self._ddt_vars = saved_ddt_vars

        lines.append('')
        lines.append('    return key;')
        lines.append('}')
        lines.append('')

        # Convenience: print routine
        lines.append('extern "C" void vae_print_regime(uint64_t key) {')
        lines.append(f'    for (int i = 0; i < {analysis.n_conditions}; i++) {{')
        lines.append('        if (key & (1ULL << i))')
        lines.append('            printf("  [%2d] TRUE:  %s\\n", i, condition_names[i]);')
        lines.append('        else')
        lines.append('            printf("  [%2d] false: %s\\n", i, condition_names[i]);')
        lines.append('    }')
        lines.append('}')

        return '\n'.join(lines) + '\n'

    def _walk_condition_eval(self, lines: list[str], node: 'ASTNode',
                             active_nodes: list[str],
                             analysis: 'RegimeAnalysis', indent: int):
        """Walk AST emitting plain C++ double code for regime evaluation.

        Mirrors _walk_analog_block logic but:
        - Emits plain C++ doubles instead of GiNaC expressions
        - For voltage-dependent IFs, emits `if (cond) key |= (1ULL << idx);`
        - Skips contributions (not needed for condition evaluation)
        """
        if node is None:
            return
        pfx = ' ' * indent

        if node.kind == NodeKind.BLOCK:
            for child in node.children:
                self._walk_condition_eval(lines, child, active_nodes,
                                          analysis, indent)

        elif node.kind == NodeKind.ASSIGN:
            uses_voltage = self._expr_uses_voltage(node.expr)

            if not uses_voltage:
                val = self._try_eval_expr(node.expr)
                if val is not None:
                    self.var_values[node.lhs] = val
                    if self._runtime_cond_depth == 0:
                        # Only skip emit if the variable hasn't been declared
                        # in C++ yet; if it was already declared (e.g. with a
                        # voltage-dependent value), we must emit the update to
                        # avoid leaving a stale value in the C++ variable.
                        if node.lhs not in self._declared_vars:
                            return
                else:
                    self.var_values.pop(node.lhs, None)
            else:
                self.var_values.pop(node.lhs, None)

            # Emit as plain C++ double assignment
            cpp_expr = self._to_eval_cpp_expr(node.expr, active_nodes)
            if node.lhs not in self._declared_vars:
                lines.append(f'{pfx}double {node.lhs} = {cpp_expr};')
                self._declared_vars[node.lhs] = 'double'
            else:
                lines.append(f'{pfx}{node.lhs} = {cpp_expr};')

        elif node.kind == NodeKind.CONTRIB:
            pass  # Skip contributions — not needed for condition evaluation

        elif node.kind == NodeKind.IF:
            nid = id(node)

            # Check if this is a cataloged voltage-dependent condition
            if nid in analysis._node_map:
                cond_idx = analysis._node_map[nid]
                # Pre-declare all vars assigned in either branch
                assigned = self._collect_assigned_vars(node)
                for var in assigned:
                    if var not in self._declared_vars:
                        lines.append(f'{pfx}double {var} = 0.0;')
                        self._declared_vars[var] = 'double'
                cpp_cond = self._condition_to_eval_cpp(node.condition, active_nodes)
                lines.append(f'{pfx}// Regime condition [{cond_idx}]: {node.condition}')
                lines.append(f'{pfx}if ({cpp_cond}) {{')
                lines.append(f'{pfx}    key |= (1ULL << {cond_idx});')
                self._runtime_cond_depth += 1
                for child in node.children:
                    self._walk_condition_eval(lines, child, active_nodes,
                                             analysis, indent + 4)
                self._runtime_cond_depth -= 1
                lines.append(f'{pfx}}} else {{')
                self._runtime_cond_depth += 1
                if node.else_body:
                    self._walk_condition_eval(lines, node.else_body,
                                             active_nodes, analysis,
                                             indent + 4)
                self._runtime_cond_depth -= 1
                lines.append(f'{pfx}}}')
            else:
                # Not a regime condition — resolve statically
                resolved = self._eval_condition(node.condition)
                if resolved is True:
                    for child in node.children:
                        self._walk_condition_eval(lines, child, active_nodes,
                                                  analysis, indent)
                elif resolved is False:
                    if node.else_body:
                        self._walk_condition_eval(lines, node.else_body,
                                                  active_nodes, analysis,
                                                  indent)
                else:
                    resolved_dyn = self._eval_condition_with_vars(node.condition)
                    if resolved_dyn is True:
                        self._runtime_cond_depth += 1
                        for child in node.children:
                            self._walk_condition_eval(lines, child,
                                                      active_nodes, analysis,
                                                      indent)
                        self._runtime_cond_depth -= 1
                    elif resolved_dyn is False:
                        if node.else_body:
                            self._runtime_cond_depth += 1
                            self._walk_condition_eval(lines, node.else_body,
                                                      active_nodes, analysis,
                                                      indent)
                            self._runtime_cond_depth -= 1
                    elif any(pat in node.condition for pat in self._assume_true):
                        # Assumed true
                        self._runtime_cond_depth += 1
                        for child in node.children:
                            self._walk_condition_eval(lines, child,
                                                      active_nodes, analysis,
                                                      indent)
                        self._runtime_cond_depth -= 1
                    else:
                        # Unresolvable non-regime condition — emit runtime C++ if
                        cpp_cond = self._condition_to_eval_cpp(
                            node.condition, active_nodes)
                        # Pre-declare variables assigned in both branches
                        assigned = self._collect_assigned_vars(node)
                        for var in assigned:
                            if var not in self._declared_vars:
                                lines.append(f'{pfx}double {var} = 0.0;')
                                self._declared_vars[var] = 'double'
                        lines.append(f'{pfx}if ({cpp_cond}) {{')
                        self._runtime_cond_depth += 1
                        for child in node.children:
                            self._walk_condition_eval(lines, child,
                                                      active_nodes, analysis,
                                                      indent + 4)
                        self._runtime_cond_depth -= 1
                        if node.else_body:
                            lines.append(f'{pfx}}} else {{')
                            self._runtime_cond_depth += 1
                            self._walk_condition_eval(lines, node.else_body,
                                                      active_nodes, analysis,
                                                      indent + 4)
                            self._runtime_cond_depth -= 1
                        lines.append(f'{pfx}}}')

        elif node.kind == NodeKind.INITIAL_STEP:
            pass

        elif node.kind == NodeKind.EXPR:
            pass

    def _to_eval_cpp_expr(self, expr: str, active_nodes: list[str]) -> str:
        """Translate VA expression to plain C++ double arithmetic.

        Similar to _to_cpp_expr but handles V(a,b), $vt, $temperature,
        Temp(node), and model-defined functions for the evaluator context.
        """
        if expr is None:
            return '0.0'

        result = expr

        # Normalize operators
        result = result.replace('= =', '==').replace('! =', '!=')
        result = result.replace('> =', '>=').replace('< =', '<=')
        result = result.replace('& &', '&&').replace('| |', '||')

        # V(a,b) → (V_a - V_b), V(a) → V_a, V(named_branch) → both
        # endpoints when the branch was declared with two nodes.
        branch_neg = getattr(self.mod, 'branch_neg_map', {}) or {}
        def repl_v(m):
            p1 = m.group(1).strip()
            p2 = m.group(2).strip() if m.group(2) else None
            if p2 is None and p1 in self.branch_map and p1 in branch_neg:
                p2 = branch_neg[p1]
            if p1 in self.branch_map:
                p1 = self.branch_map[p1]
            if p2 and p2 in self.branch_map:
                p2 = self.branch_map[p2]
            if p1 not in active_nodes and p1 in self.internal_nodes:
                p1 = self._resolve_shorted_node(p1, active_nodes)
            if p2 and p2 not in active_nodes and p2 in self.internal_nodes:
                p2 = self._resolve_shorted_node(p2, active_nodes)
            p1_expr = '0.0' if p1 is None else f'V_{p1}'
            p2_expr = '0.0' if p2 is None else f'V_{p2}' if p2 else None
            if p2_expr:
                return f'({p1_expr} - {p2_expr})'
            return p1_expr
        result = re.sub(r'V\s*\(\s*(\w+)\s*(?:,\s*(\w+)\s*)?\)', repl_v, result)

        # I(x,y) → 0.0 (current probes not available in evaluator)
        def _strip_call(name, s, replacement='0.0'):
            pattern = rf'\b{name}\s*\('
            while True:
                m = re.search(pattern, s)
                if not m:
                    break
                start = m.end(); depth = 1; i = start
                while i < len(s) and depth > 0:
                    if s[i] == '(':
                        depth += 1
                    elif s[i] == ')':
                        depth -= 1
                    i += 1
                s = s[:m.start()] + replacement + s[i:]
            return s

        # $limit(expr, ...) → just expr
        result = re.sub(r'\$limit\s*\(\s*([^,]+),.*?\)', r'\1', result)

        # Temp(node) → V_node
        result = re.sub(r'Temp\s*\(\s*(\w+)\s*\)',
                        lambda m: f'V_{m.group(1)}', result)

        # $vt(expr) -> (k/q * (expr)), $vt -> _vae_vt, $temperature -> _vae_T
        result = sub_thermal(result)

        # $simparam("name"[, default]) -> the simulator's value
        result = self._sub_simparam(result)

        # Function mapping (VA → C++ std:: calls)
        _cpp_funcs = [
            ('limexp', 'lexp'), ('lexp', 'lexp'), ('lln', 'lln'),
            ('ln', 'std::log'), ('abs', 'std::fabs'),
            ('sqrt', 'std::sqrt'), ('pow', 'std::pow'),
            ('exp', 'std::exp'), ('log', 'std::log'),
            ('sin', 'std::sin'), ('cos', 'std::cos'),
            ('tan', 'std::tan'), ('tanh', 'std::tanh'),
            ('atan2', 'std::atan2'), ('atan', 'std::atan'),
            ('hypot', 'std::hypot'),
            ('min', 'std::fmin'), ('max', 'std::fmax'),
        ]
        for va_name, cpp_name in _cpp_funcs:
            result = re.sub(rf'(?<!:)\b{va_name}\b(?=\s*\()', cpp_name, result)

        # `DEFINE → identifier
        result = re.sub(r'`(\w+)', r'\1', result)

        # Noise / ddx → 0
        result = _strip_call('white_noise', result)
        result = _strip_call('flicker_noise', result)
        result = _strip_call('ddx', result)

        # ddt(x) → 0.0 (not needed for DC condition eval)
        result = _strip_call('ddt', result)

        # Ternary evaluation
        def _eval_ternary(s):
            # Match innermost ternary: (cond ? true : false) where cond has
            # no nested parens.  [^?()] prevents matching across parens.
            pattern = r'\(([^?()]+)\?\s*([^:()]+):\s*([^()]+)\)'
            while True:
                m = re.search(pattern, s)
                if not m:
                    break
                cond_str = m.group(1).strip()
                true_val = m.group(2).strip()
                false_val = m.group(3).strip()
                py_cond = cond_str.replace('&&', ' and ').replace('||', ' or ')
                py_cond = re.sub(r'(?<!=)!(?!=)', ' not ', py_cond)
                try:
                    ns = self._cond_ns()
                    result_val = eval(py_cond, {"__builtins__": {}}, ns)
                    replacement = true_val if result_val else false_val
                    s = s[:m.start()] + f'({replacement})' + s[m.end():]
                except Exception:
                    # Can't resolve — emit as C++ ternary and stop
                    # (further replacements would loop on the new parens)
                    s = s[:m.start()] + f'(({cond_str}) ? ({true_val}) : ({false_val}))' + s[m.end():]
                    break
            return s
        result = _eval_ternary(result)

        # Substitute known constant params/vars with numeric values
        _skip_cpp = {'fabs', 'fmin', 'fmax', 'pow', 'exp', 'log', 'sqrt',
                     'sin', 'cos', 'tan', 'tanh', 'atan', 'atan2', 'hypot',
                     'hypsmooth', 'hypmax', 'Tempdep', 'lexp', 'lln',
                     VT_SYM, T_SYM, 'key'}
        _skip_cpp |= {f'V_{n}' for n in self.all_nodes}
        # Also skip variables that have been declared as C++ doubles
        declared = set(self._declared_vars.keys())

        def _subst(m):
            name = m.group(0)
            if name in _skip_cpp:
                return name
            if name in declared:
                return name  # Already a C++ local variable
            if name in self.param_values:
                return repr(self.param_values[name])
            if name in self.var_values:
                return repr(self.var_values[name])
            return name
        result = re.sub(r'\b[A-Za-z_]\w*\b', _subst, result)

        # Second pass ternary after substitution
        result = _eval_ternary(result)

        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def _condition_to_eval_cpp(self, cond: str, active_nodes: list[str]) -> str:
        """Translate a VA condition to plain C++ for the evaluator."""
        c = self._preprocess_sys_funcs(cond)
        c = c.replace('= =', '==').replace('! =', '!=')
        c = c.replace('> =', '>=').replace('< =', '<=')
        c = c.replace('& &', '&&').replace('| |', '||')

        # V(a,b) → (V_a - V_b), V(a) → V_a, V(named_branch) using both
        # endpoints when the branch declaration named two nodes.
        branch_neg = getattr(self.mod, 'branch_neg_map', {}) or {}
        def repl_v(m):
            p1 = m.group(1).strip()
            p2 = m.group(2).strip() if m.group(2) else None
            if p2 is None and p1 in self.branch_map and p1 in branch_neg:
                p2 = branch_neg[p1]
            if p1 in self.branch_map:
                p1 = self.branch_map[p1]
            if p2 and p2 in self.branch_map:
                p2 = self.branch_map[p2]
            if p1 not in active_nodes and p1 in self.internal_nodes:
                p1 = self._resolve_shorted_node(p1, active_nodes)
            if p2 and p2 not in active_nodes and p2 in self.internal_nodes:
                p2 = self._resolve_shorted_node(p2, active_nodes)
            if p2:
                return f'(V_{p1} - V_{p2})'
            return f'V_{p1}'
        c = re.sub(r'V\s*\(\s*(\w+)\s*(?:,\s*(\w+)\s*)?\)', repl_v, c)

        # Temp(node) → V_node
        c = re.sub(r'Temp\s*\(\s*(\w+)\s*\)',
                   lambda m: f'V_{m.group(1)}', c)

        # $vt(expr) -> (k/q * (expr)), $vt -> _vae_vt, $temperature -> _vae_T
        c = sub_thermal(c)

        # Function mapping
        _cpp_funcs = [
            ('limexp', 'lexp'), ('lexp', 'lexp'), ('lln', 'lln'),
            ('ln', 'std::log'), ('abs', 'std::fabs'),
            ('sqrt', 'std::sqrt'), ('pow', 'std::pow'),
            ('exp', 'std::exp'), ('log', 'std::log'),
            ('min', 'std::fmin'), ('max', 'std::fmax'),
            ('tanh', 'std::tanh'),
        ]
        for va_name, cpp_name in _cpp_funcs:
            c = re.sub(rf'(?<!:)\b{va_name}\b(?=\s*\()', cpp_name, c)

        # Substitute known params/vars
        _skip = {'fabs', 'fmin', 'fmax', 'pow', 'exp', 'log', 'sqrt',
                 'sin', 'cos', 'tan', 'tanh', 'atan', 'atan2', 'hypot',
                 'hypsmooth', 'hypmax', 'Tempdep', 'lexp', 'lln',
                 VT_SYM, T_SYM, 'key'}
        _skip |= {f'V_{n}' for n in self.all_nodes}
        declared = set(self._declared_vars.keys())

        def _subst(m):
            name = m.group(0)
            if name in _skip:
                return name
            if name in declared:
                return name
            if name in self.param_values:
                return repr(self.param_values[name])
            if name in self.var_values:
                return repr(self.var_values[name])
            return name
        c = re.sub(r'\b[A-Za-z_]\w*\b', _subst, c)

        return c


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def emit_ginac_program(module: Module,
                       param_values: Optional[dict[str, float]] = None,
                       line_directives: Optional[bool] = None,
                       given_params: Optional[set[str]] = None,
                       callback_params: Optional[set[str]] = None,
                       simparams: Optional[dict[str, float]] = None) -> str:
    """Generate GiNaC C++ source from a parsed Verilog-A module.

    The netlist's parameter values (model card + instance) are supplied as
    constants; every other parameter takes its Verilog-A default, evaluated
    per instance. Only node voltages remain symbolic for differentiation.
    Each unique parameter set gets its own compiled code; common geometries
    share the same .so.

    Args:
        module: Parsed Verilog-A module.
        param_values: The netlist's parameter values (model + instance), by the
                      module's own names; the given ones overlay the defaults.
        simparams: The simulator's $simparam values (gmin, tnom), if known.
        line_directives: None=off, True=#line (active), False=//line (inactive).
        given_params: The set of parameter names the USER explicitly supplied
                      (from Xyce's given()). Drives $param_given(). If None,
                      falls back to param_values.keys() (legacy). Must be passed
                      by the JIT builder, which bakes defaults into param_values
                      and would otherwise mis-report every default as "given".
        callback_params: Parameter names to keep SYMBOLIC and fetch at eval time
                      via the shell callback _pcb("NAME") instead of baking a
                      literal. A compiler cannot const-fold an opaque call, so
                      the value stays live and Xyce .SAMPLING/AGAUSS can vary it
                      (e.g. PSP103 DELVTO for Monte-Carlo Vt mismatch) through a
                      single .so build. Empty by default (byte-identical output).
    """
    return GiNaCEmitter(module, param_values=param_values,
                        line_directives=line_directives,
                        given_params=given_params,
                        callback_params=callback_params,
                        simparams=simparams).emit()
