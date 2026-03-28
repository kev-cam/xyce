"""
VAE regime analysis and JIT compilation cache.

Walks a parsed Verilog-A AST with resolved parameters to identify
voltage-dependent conditions.  Each unique combination of condition
outcomes defines a "regime" (operating region).  Regimes are compiled
on demand to branch-free eval/jacobian shared libraries and cached
on disk keyed by (model_hash, instance_hash, regime_key).

Condition synchronization uses AST node identity (id(node)) so that
the analyzer and emitter always agree on which condition is which,
regardless of how var_values evolve along different branch paths.

Usage:
    cache = RegimeCache(va_path, param_values)
    cache.elaborate()                    # once per instance
    regime = cache.get_regime(voltages)  # per timestep
    regime.eval(state, F, Q)
    regime.jacobian(state, dFdV, dQdV)
"""

from __future__ import annotations
import hashlib
import os
import re
import subprocess
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .parser import Module, ASTNode, NodeKind, parse_file
from .ginac_emitter import GiNaCEmitter


# ---------------------------------------------------------------------------
# Condition analysis
# ---------------------------------------------------------------------------

@dataclass
class ConditionInfo:
    """A single voltage-dependent condition in the analog block."""
    index: int              # sequential index (bit position in regime key)
    condition: str          # original VA condition text
    node_id: int            # id(ast_node) — for mapping to emitter
    parent_index: int       # index of enclosing condition (-1 if top-level)
    parent_branch: bool     # True = in parent's true-branch, False = else-branch
    depth: int              # nesting depth (0 = top-level)


@dataclass
class RegimeAnalysis:
    """Result of analyzing a model for operating regimes."""
    conditions: list[ConditionInfo]
    # Map from AST node id to condition index
    _node_map: dict[int, int] = field(default_factory=dict, repr=False)

    @property
    def n_conditions(self) -> int:
        return len(self.conditions)

    def _resolve_key(self, key: int) -> dict[int, bool]:
        """Resolve a regime bitmask to {condition_index: True/False}.

        A child condition is reachable only when its parent's outcome
        matches the child's parent_branch:
        - parent_branch=True  → child is in the 'if' body → reachable when parent is True
        - parent_branch=False → child is in the 'else' body → reachable when parent is False
        If unreachable, the child is forced False.
        """
        by_index = {}
        for c in self.conditions:
            bit = bool(key & (1 << c.index))
            if c.parent_index >= 0:
                parent_val = by_index.get(c.parent_index, True)
                reachable = (parent_val == c.parent_branch)
                by_index[c.index] = bit if reachable else False
            else:
                by_index[c.index] = bit
        return by_index

    def forced_nodes(self, key: int) -> dict[int, bool]:
        """Convert a regime bitmask to {node_id: True/False}.

        Returns a dict keyed by AST node id(node) suitable for passing
        to GiNaCEmitter.
        """
        by_index = self._resolve_key(key)
        return {c.node_id: by_index[c.index] for c in self.conditions}

    def normalize_key(self, key: int) -> int:
        """Normalize a regime key by clearing unreachable condition bits."""
        by_index = self._resolve_key(key)
        normalized = 0
        for idx, val in by_index.items():
            if val:
                normalized |= (1 << idx)
        return normalized

    def key_to_index_map(self, key: int) -> dict[int, bool]:
        """Convert regime key to {condition_index: True/False}."""
        by_index = {}
        for c in self.conditions:
            bit = bool(key & (1 << c.index))
            if c.parent_index >= 0 and not by_index.get(c.parent_index, True):
                by_index[c.index] = False
            else:
                by_index[c.index] = bit
        return by_index


class RegimeAnalyzer:
    """Walks the AST to find voltage-dependent conditions.

    Uses the same condition resolution logic as GiNaCEmitter (tiers 1+2)
    to identify which IF nodes cannot be resolved at elaboration time.
    These are the conditions that define operating regimes.
    """

    def __init__(self, module: Module, param_values: dict[str, float],
                 given_params: Optional[set[str]] = None,
                 assume_true: Optional[set[str]] = None):
        self._emitter = GiNaCEmitter(module, param_values=param_values,
                                     assume_true=assume_true)
        if given_params:
            self._emitter._given_params = given_params
        self._module = module
        self._assume_true = assume_true or set()
        self._conditions: list[ConditionInfo] = []
        self._node_map: dict[int, int] = {}
        self._runtime_cond_depth = 0

    def analyze(self) -> RegimeAnalysis:
        """Walk the AST and collect voltage-dependent conditions."""
        if self._module.analog_block:
            self._walk(self._module.analog_block, parent_index=-1,
                       parent_branch=True, depth=0)
        analysis = RegimeAnalysis(
            conditions=self._conditions,
            _node_map=self._node_map,
        )
        return analysis

    def _walk(self, node: ASTNode, parent_index: int,
              parent_branch: bool, depth: int):
        if node is None:
            return

        if node.kind == NodeKind.BLOCK:
            for child in node.children:
                self._walk(child, parent_index, parent_branch, depth)

        elif node.kind == NodeKind.ASSIGN:
            e = self._emitter
            uses_voltage = e._expr_uses_voltage(node.expr)
            if not uses_voltage:
                val = e._try_eval_expr(node.expr)
                if val is not None:
                    e.var_values[node.lhs] = val
                    if self._runtime_cond_depth == 0:
                        return
                else:
                    e.var_values.pop(node.lhs, None)
            else:
                e.var_values.pop(node.lhs, None)

        elif node.kind == NodeKind.IF:
            e = self._emitter
            resolved = e._eval_condition(node.condition)
            if resolved is True:
                for child in node.children:
                    self._walk(child, parent_index, parent_branch, depth)
            elif resolved is False:
                if node.else_body:
                    self._walk(node.else_body, parent_index,
                               parent_branch, depth)
            else:
                resolved_dyn = e._eval_condition_with_vars(node.condition)
                if resolved_dyn is True:
                    self._runtime_cond_depth += 1
                    for child in node.children:
                        self._walk(child, parent_index, parent_branch, depth)
                    self._runtime_cond_depth -= 1
                elif resolved_dyn is False:
                    if node.else_body:
                        self._runtime_cond_depth += 1
                        self._walk(node.else_body, parent_index,
                                   parent_branch, depth)
                        self._runtime_cond_depth -= 1
                else:
                    # Check assume_true patterns
                    if any(pat in node.condition for pat in self._assume_true):
                        # Assumed true — walk true branch, skip else
                        self._runtime_cond_depth += 1
                        for child in node.children:
                            self._walk(child, parent_index, parent_branch, depth)
                        self._runtime_cond_depth -= 1
                        return

                    # Voltage-dependent condition
                    cond_idx = len(self._conditions)
                    nid = id(node)
                    info = ConditionInfo(
                        index=cond_idx,
                        condition=node.condition,
                        node_id=nid,
                        parent_index=parent_index,
                        parent_branch=parent_branch,
                        depth=depth,
                    )
                    self._conditions.append(info)
                    self._node_map[nid] = cond_idx

                    # Walk both branches to find nested conditions
                    self._runtime_cond_depth += 1
                    for child in node.children:
                        self._walk(child, cond_idx, True, depth + 1)
                    if node.else_body:
                        self._walk(node.else_body, cond_idx, False, depth + 1)
                    self._runtime_cond_depth -= 1

        elif node.kind == NodeKind.INITIAL_STEP:
            pass


# ---------------------------------------------------------------------------
# Compiled regime handle
# ---------------------------------------------------------------------------

_EvalFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_double),
                           ctypes.POINTER(ctypes.c_double))
_JacobFn = _EvalFn


@dataclass
class CompiledRegime:
    """A compiled regime with eval/jacobian function pointers."""
    regime_key: int
    so_path: str
    _lib: ctypes.CDLL = field(repr=False, default=None)
    eval_fn: Optional[_EvalFn] = field(repr=False, default=None)
    jacobian_fn: Optional[_JacobFn] = field(repr=False, default=None)


# ---------------------------------------------------------------------------
# Regime cache — JIT compilation and caching
# ---------------------------------------------------------------------------

def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _hash_params(params: dict[str, float]) -> str:
    items = sorted(params.items())
    h = hashlib.sha256()
    for k, v in items:
        h.update(f"{k}={v!r}\n".encode())
    return h.hexdigest()[:16]


_SPICE_SUFFIX = {
    'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'K': 1e3,
    'M': 1e-3, 'MIL': 25.4e-6,
    'U': 1e-6, 'N': 1e-9, 'P': 1e-12, 'F': 1e-15, 'A': 1e-18,
}

def _parse_spice_float(s: str) -> float:
    """Parse a SPICE numeric value with optional engineering suffix."""
    s = s.strip()
    # Try plain float first
    try:
        return float(s)
    except ValueError:
        pass
    # Try with SPICE suffix (case-insensitive)
    su = s.upper()
    for suffix, mult in sorted(_SPICE_SUFFIX.items(), key=lambda x: -len(x[0])):
        if su.endswith(suffix):
            num_part = s[:len(s) - len(suffix)]
            try:
                return float(num_part) * mult
            except ValueError:
                continue
    raise ValueError(f"Cannot parse SPICE value: {s!r}")


def parse_modelcard(path: str) -> dict[str, float]:
    """Parse a SPICE .model card and return parameter name->value dict."""
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('*') or line.startswith('.model'):
                continue
            if line.startswith('+'):
                line = line[1:].strip()
            if '//' in line:
                line = line[:line.index('//')]
            for m in re.finditer(r'(\w+)\s*=\s*([^\s,]+)', line):
                name, val = m.group(1), m.group(2)
                try:
                    params[name] = _parse_spice_float(val)
                except ValueError:
                    pass
    return params


def _try_eval_full(expr: str, emitter: GiNaCEmitter,
                    volt_ns: dict) -> Optional[float]:
    """Evaluate an expression with all known values + voltages.

    Unlike the emitter's _try_eval_expr (which only uses param_values and
    var_values), this also substitutes node voltage values — used for
    runtime regime key evaluation where all voltages are known numbers.
    """
    import math
    # Build combined namespace
    ns = dict(emitter.param_values)
    ns.update(emitter.var_values)
    ns.update(volt_ns)
    ns.update({
        'ln': math.log, 'log': math.log, 'exp': math.exp,
        'sqrt': math.sqrt, 'pow': math.pow, 'abs': abs,
        'min': min, 'max': max, 'hypot': math.hypot,
        'tanh': math.tanh, 'cosh': math.cosh, 'sinh': math.sinh,
        'floor': math.floor, 'ceil': math.ceil, 'fabs': abs,
    })
    # Also add model-defined functions if available
    def hypsmooth(x, c):
        return 0.5 * (x + math.sqrt(x * x + 4 * c * c))
    def lexp(x):
        return math.exp(x) if x < 80 else math.exp(80) * (1 + x - 80)
    def lln(x):
        return math.log(x) if x > 1e-80 else math.log(1e-80) + (x - 1e-80) / 1e-80
    def hypmax(x, xmin, c):
        return xmin + 0.5 * (x - xmin - c + math.sqrt((x - xmin - c)**2 - 4.0 * xmin * c))
    def Tempdep(PARAML, PARAMT, DELTEMP, TEMPMOD):
        if TEMPMOD != 0:
            return PARAML + hypmax(PARAMT * DELTEMP, -PARAML, 1e-6)
        else:
            return PARAML * hypsmooth(1.0 + PARAMT * DELTEMP - 1e-6, 1e-3)
    ns.update({'hypsmooth': hypsmooth, 'hypmax': hypmax, 'Tempdep': Tempdep,
               'lexp': lexp, 'lln': lln})

    # $simparam("name", default) → default value
    e = expr
    e = re.sub(r'\$simparam\s*\(\s*"[^"]*"\s*,\s*([^)]+)\)', r'\1', e)
    e = re.sub(r'\$simparam\s*\(\s*"[^"]*"\s*\)', '0', e)

    # Substitute V(a,b) → (V_a - V_b) and V(a) → V_a
    e = re.sub(r'V\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)',
               r'(V_\1 - V_\2)', e)
    e = re.sub(r'V\s*\(\s*(\w+)\s*\)', r'V_\1', e)
    e = re.sub(r'Temp\s*\(\s*(\w+)\s*\)', r'V_\1', e)
    e = e.replace('$temperature', 'temperature')
    e = e.replace('$vt', 'Vt')

    # Convert C-style ternary (cond ? a : b) → _ternary(cond, a, b)
    # Handles nested ternaries (CLIP_BOTH) by iterating inner-to-outer
    def _ternary_to_func(s):
        for _ in range(10):
            prev = s
            idx = 0
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
                    s = s[:start] + f'_ternary({cond},{true_v},{false_v})' + s[i+1:]
                    break  # restart after substitution
                idx = start + 1
            if s == prev:
                break
        return s
    e = _ternary_to_func(e)

    ns['_ternary'] = lambda c, a, b: a if c else b

    try:
        result = eval(e, {"__builtins__": {}}, ns)
        return float(result)
    except Exception:
        return None


class RegimeCache:
    """JIT compilation cache for operating regimes.

    Elaboration (once per instance):
      - Parse .va model, resolve parameter-dependent conditions
      - Analyze voltage-dependent conditions and their nesting

    Runtime (per timestep):
      - Evaluate conditions with current voltages -> regime key
      - Cache lookup: hit -> return compiled function pointers
      - Cache miss -> JIT compile: emit GiNaC, compile, run, compile .so, cache
      - Detect regime change -> signal timestep limiting
    """

    def __init__(self, va_path: str, param_values: dict[str, float],
                 given_params: Optional[set[str]] = None,
                 assume_true: Optional[set[str]] = None,
                 cache_dir: str = "~/.vae/cache"):
        self.va_path = va_path
        self.param_values = dict(param_values)
        self.given_params = given_params or set(param_values.keys())
        self.assume_true = assume_true or set()
        self.cache_dir = Path(os.path.expanduser(cache_dir))

        self.module: Optional[Module] = None
        self.analysis: Optional[RegimeAnalysis] = None
        self._regimes: dict[int, CompiledRegime] = {}
        self._instance_dir: Optional[Path] = None
        self._prev_key: Optional[int] = None
        self._eval_regime_fn = None  # compiled C++ evaluator

    def elaborate(self):
        """Parse model and analyze voltage-dependent conditions."""
        self.module = parse_file(self.va_path)
        analyzer = RegimeAnalyzer(self.module, self.param_values,
                                  self.given_params, self.assume_true)
        self.analysis = analyzer.analyze()

        # Set up cache directory
        with open(self.va_path) as f:
            model_hash = _hash_str(f.read())
        instance_hash = _hash_params(self.param_values)
        self._instance_dir = self.cache_dir / model_hash / instance_hash
        self._instance_dir.mkdir(parents=True, exist_ok=True)

        n = self.analysis.n_conditions
        print(f"Elaborated {self.module.name}: {n} voltage-dependent conditions")
        for c in self.analysis.conditions:
            indent = "  " * (c.depth + 1)
            print(f"  [{c.index:2d}]{indent}{c.condition}")

        # Build compiled C++ condition evaluator for regime key computation
        self._eval_regime_fn, _, _ = self.load_condition_evaluator()

    def get_regime(self, voltages: list[float], vt: float = None,
                   temperature: float = None) -> tuple[CompiledRegime, bool]:
        """Get compiled regime for current operating point.

        Uses the compiled C++ condition evaluator for accurate regime
        key computation. Falls back to Python evaluator if C++ is not
        available.

        Returns (regime, changed) where changed=True if the regime
        differs from the previous call (timestep should be limited).
        """
        # Use compiled C++ evaluator if available
        if self._eval_regime_fn is not None:
            if vt is None:
                vt = 8.617087e-5 * (temperature or 300.15)
            key = self._eval_regime_fn(voltages, vt)
        else:
            key = self._eval_regime_key(voltages, vt, temperature)
        key = self.analysis.normalize_key(key)
        changed = (self._prev_key is not None and key != self._prev_key)
        self._prev_key = key

        if key not in self._regimes:
            self._regimes[key] = self._compile_regime(key)

        return self._regimes[key], changed

    def _eval_regime_key(self, voltages: list[float],
                         vt: float = None,
                         temperature: float = None) -> int:
        """Evaluate all conditions with current voltages to get regime key.

        Walks the AST, computing intermediates numerically and evaluating
        each voltage-dependent condition at the current operating point.
        """
        # Build a fresh emitter for variable tracking
        e = GiNaCEmitter(self.module, param_values=self.param_values,
                         assume_true=self.assume_true)
        e._given_params = self.given_params

        # Build voltage namespace
        port_names = [p.name for p in self.module.ports]
        internal_nodes = list(self.module.internal_nodes)
        all_nodes = port_names + internal_nodes
        volt_ns = {}
        for i, n in enumerate(all_nodes):
            if i < len(voltages):
                volt_ns[f'V_{n}'] = voltages[i]
        if vt is not None:
            volt_ns['Vt'] = vt
        if temperature is not None:
            volt_ns['temperature'] = temperature

        node_map = self.analysis._node_map
        key = 0

        def walk(node, runtime_depth=0):
            nonlocal key
            if node is None:
                return
            if node.kind == NodeKind.BLOCK:
                for child in node.children:
                    walk(child, runtime_depth)
            elif node.kind == NodeKind.ASSIGN:
                # Try to evaluate with all known values + voltages
                val = _try_eval_full(node.expr, e, volt_ns)
                if val is not None:
                    e.var_values[node.lhs] = val
                else:
                    e.var_values.pop(node.lhs, None)
            elif node.kind == NodeKind.IF:
                nid = id(node)
                # Check if this is a cataloged voltage-dependent condition
                if nid in node_map:
                    idx = node_map[nid]
                    # Evaluate condition with all known values + voltages
                    result = self._eval_condition_numeric(
                        e, node.condition, volt_ns)
                    if result:
                        key |= (1 << idx)
                        for child in node.children:
                            walk(child, runtime_depth + 1)
                    else:
                        if node.else_body:
                            walk(node.else_body, runtime_depth + 1)
                else:
                    # Param/var-resolved or assumed-true condition
                    resolved = e._eval_condition(node.condition)
                    if resolved is True:
                        for child in node.children:
                            walk(child, runtime_depth)
                    elif resolved is False:
                        if node.else_body:
                            walk(node.else_body, runtime_depth)
                    else:
                        resolved_dyn = e._eval_condition_with_vars(
                            node.condition)
                        if resolved_dyn is True:
                            for child in node.children:
                                walk(child, runtime_depth + 1)
                        elif resolved_dyn is False:
                            if node.else_body:
                                walk(node.else_body, runtime_depth + 1)
                        elif any(pat in node.condition
                                 for pat in self.assume_true):
                            # Assumed true — walk true branch
                            for child in node.children:
                                walk(child, runtime_depth + 1)
                        else:
                            # Unresolvable — try numeric eval with voltages
                            result = self._eval_condition_numeric(
                                e, node.condition, volt_ns)
                            if result:
                                for child in node.children:
                                    walk(child, runtime_depth + 1)
                            else:
                                if node.else_body:
                                    walk(node.else_body, runtime_depth + 1)

        walk(self.module.analog_block)
        return key

    def _eval_condition_numeric(self, emitter: GiNaCEmitter,
                                cond: str, volt_ns: dict) -> bool:
        """Evaluate a condition with actual voltage values."""
        import math
        cond = emitter._preprocess_sys_funcs(cond)
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

        ns = dict(emitter.param_values)
        ns.update(emitter.var_values)
        ns.update(volt_ns)
        ns.update({
            'ln': math.log, 'log': math.log, 'exp': math.exp,
            'sqrt': math.sqrt, 'pow': math.pow, 'abs': abs,
            'min': min, 'max': max, 'hypot': math.hypot,
        })
        try:
            result = eval(py_cond, {"__builtins__": {}}, ns)
            return bool(result)
        except Exception:
            return False  # Default to False on eval failure

    def _compile_regime(self, regime_key: int) -> CompiledRegime:
        """JIT compile a regime: GiNaC emit -> compile -> run -> compile .so."""
        so_path = self._instance_dir / f"regime_{regime_key:016x}.so"

        # Check disk cache
        if so_path.exists():
            return self._load_regime(regime_key, str(so_path))

        print(f"JIT compiling regime 0x{regime_key:x}...")

        # 1. Emit GiNaC program with forced conditions (keyed by node id)
        forced = self.analysis.forced_nodes(regime_key)
        emitter = GiNaCEmitter(self.module, param_values=self.param_values,
                               forced_nodes=forced,
                               assume_true=self.assume_true)
        emitter._given_params = self.given_params
        ginac_src = emitter.emit()

        # Verify all conditions were resolved
        n_runtime = ginac_src.count('// RUNTIME CONDITION')
        if n_runtime > 0:
            print(f"  WARNING: {n_runtime} unresolved runtime conditions remain")

        # 2. Compile GiNaC program
        ginac_cpp = self._instance_dir / f"regime_{regime_key:016x}_ginac.cpp"
        ginac_bin = self._instance_dir / f"regime_{regime_key:016x}_ginac"
        eval_cpp = self._instance_dir / f"regime_{regime_key:016x}_eval.cpp"

        ginac_cpp.write_text(ginac_src)
        _run_cmd(f"g++ -O2 -std=c++17 -o {ginac_bin} {ginac_cpp} -lginac -lcln")

        # 3. Run GiNaC program to generate eval/jacobian code
        _run_cmd(f"{ginac_bin} > {eval_cpp}", timeout=600)

        # 4. Compile eval code to shared library
        wrapper_cpp = self._instance_dir / f"regime_{regime_key:016x}_wrapper.cpp"
        wrapper_cpp.write_text(self._make_wrapper(eval_cpp))
        _run_cmd(f"g++ -O2 -std=c++17 -shared -fPIC -o {so_path} {wrapper_cpp} -lm")

        print(f"  Compiled to {so_path}")
        return self._load_regime(regime_key, str(so_path))

    def _make_wrapper(self, eval_cpp: Path) -> str:
        """Generate wrapper C++ that includes the eval code with proper setup."""
        port_names = [p.name for p in self.module.ports]
        internal_nodes = list(self.module.internal_nodes)
        n_nodes = len(port_names) + len(internal_nodes)
        return f"""\
#include <cmath>
#include <cstdio>
#include <cstring>

struct VaeState {{
    double V[{n_nodes}];
    double Vt;
}};

inline double conjugate(double x) {{ return x; }}

#define vae_eval _vae_eval_impl
#define vae_jacobian _vae_jacobian_impl
static const double temperature = 300.15; // nominal (DTA=0)
#include "{eval_cpp}"
#undef vae_eval
#undef vae_jacobian

extern "C" void vae_eval(VaeState* s, double* F, double* Q) {{
    _vae_eval_impl(s, F, Q);
}}
extern "C" void vae_jacobian(VaeState* s, double* dFdV, double* dQdV) {{
    _vae_jacobian_impl(s, dFdV, dQdV);
}}
"""

    def compile_condition_evaluator(self) -> str:
        """Generate, compile, and cache the condition evaluator shared library.

        Returns the path to the compiled .so file.  The evaluator exports:
          - uint64_t vae_eval_regime(double* V, int n_nodes, double Vt)
          - void vae_print_regime(uint64_t key)
        """
        assert self.module is not None, "Call elaborate() first"
        assert self.analysis is not None, "Call elaborate() first"

        so_path = self._instance_dir / "condition_eval.so"
        cpp_path = self._instance_dir / "condition_eval.cpp"

        if so_path.exists():
            return str(so_path)

        # Build emitter with same config used for analysis
        emitter = GiNaCEmitter(self.module, param_values=self.param_values,
                               assume_true=self.assume_true)
        emitter._given_params = self.given_params

        # Determine active nodes (same logic as emit())
        active_nodes = list(emitter.port_names)
        shorted = emitter._find_shorted_nodes(self.module.analog_block)
        for n in emitter.internal_nodes:
            if n not in shorted:
                active_nodes.append(n)

        # Use the analyzer's emitter's AST nodes so id() matches analysis._node_map
        # We must re-run analysis to get matching node ids
        analyzer = RegimeAnalyzer(self.module, self.param_values,
                                  self.given_params, self.assume_true)
        analysis = analyzer.analyze()

        # Rebuild emitter so it walks the same AST nodes
        emitter2 = analyzer._emitter
        emitter2.var_values = {}
        emitter2._declared_vars = {}
        emitter2._runtime_cond_depth = 0
        emitter2._var_versions = {}
        emitter2._var_sym_name = {}
        emitter2._condition_registry = []

        src = emitter2.emit_condition_evaluator(active_nodes, analysis)

        cpp_path.write_text(src)
        _run_cmd(f"g++ -O2 -std=c++17 -shared -fPIC -o {so_path} {cpp_path} -lm")
        print(f"Compiled condition evaluator to {so_path}")
        return str(so_path)

    def load_condition_evaluator(self):
        """Compile (if needed) and load the condition evaluator.

        Returns a callable: eval_regime(voltages, vt) -> uint64_t
        """
        so_path = self.compile_condition_evaluator()
        lib = ctypes.CDLL(so_path)

        # uint64_t vae_eval_regime(double* V, int n_nodes, double Vt)
        _eval_fn = lib.vae_eval_regime
        _eval_fn.restype = ctypes.c_uint64
        _eval_fn.argtypes = [ctypes.POINTER(ctypes.c_double),
                             ctypes.c_int, ctypes.c_double]

        _print_fn = lib.vae_print_regime
        _print_fn.restype = None
        _print_fn.argtypes = [ctypes.c_uint64]

        n_active = len([p.name for p in self.module.ports])
        shorted = GiNaCEmitter(self.module, param_values=self.param_values,
                               assume_true=self.assume_true) \
                      ._find_shorted_nodes(self.module.analog_block)
        for n in self.module.internal_nodes:
            if n not in shorted:
                n_active += 1

        def eval_regime(voltages, vt=0.02585):
            arr = (ctypes.c_double * len(voltages))(*voltages)
            return _eval_fn(arr, len(voltages), ctypes.c_double(vt))

        def print_regime(key):
            _print_fn(ctypes.c_uint64(key))

        return eval_regime, print_regime, n_active

    def _load_regime(self, regime_key: int, so_path: str) -> CompiledRegime:
        """Load a compiled regime .so and extract function pointers."""
        lib = ctypes.CDLL(so_path)
        eval_fn = _EvalFn(('vae_eval', lib))
        jac_fn = _JacobFn(('vae_jacobian', lib))
        regime = CompiledRegime(
            regime_key=regime_key,
            so_path=so_path,
            _lib=lib,
            eval_fn=eval_fn,
            jacobian_fn=jac_fn,
        )
        self._regimes[regime_key] = regime
        return regime


def _run_cmd(cmd: str, timeout: int = 120):
    """Run a shell command, raising on failure."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result
