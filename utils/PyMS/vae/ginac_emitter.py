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
    Module, ASTNode, NodeKind, ContribKind, Port, PortDir, Param, Var
)


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
}

# Functions that need special handling for GiNaC (not direct mappings)
_SPECIAL_FUNCS = {'min', 'max', 'hypsmooth', 'hypmax', 'Tempdep'}


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
                 assume_true: Optional[set[str]] = None):
        """
        Args:
            module: Parsed Verilog-A module AST.
            param_values: All parameter values (model card + instance).
                          Every parameter should have a known value — only
                          node voltages remain symbolic for differentiation.
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
        self.line_directives = line_directives

        # Build parameter value map: name → numeric value
        # Start with AST defaults, overlay with caller-supplied values
        self.param_values: dict[str, float] = {}
        # First pass: numeric defaults
        for p in module.params:
            try:
                self.param_values[p.name] = float(p.default)
            except (ValueError, TypeError):
                pass
        # Apply caller-supplied values (model card + instance)
        if param_values:
            self.param_values.update(param_values)
        # Second pass: resolve param-referencing defaults (e.g. DVT1SS = DVT1)
        # Iterate until stable — handles chains like A = B, B = C
        for _ in range(5):
            changed = False
            for p in module.params:
                if p.name in self.param_values:
                    continue  # already resolved
                # Try to evaluate default using known param values
                default = p.default
                if default is None:
                    self.param_values[p.name] = 0.0
                    changed = True
                    continue
                try:
                    val = eval(default, {"__builtins__": {}}, self.param_values)
                    self.param_values[p.name] = float(val)
                    changed = True
                except Exception:
                    pass
            if not changed:
                break
        # Any still-unresolved params default to 0
        for p in module.params:
            if p.name not in self.param_values:
                self.param_values[p.name] = 0.0

        # No instance params — everything is constant at compile time
        self.instance_params: set[str] = set()

        # Track explicitly given params (for $param_given)
        self._given_params: set[str] = set()
        if param_values:
            self._given_params = set(param_values.keys())

        # Variable values resolved during elaboration
        self.var_values: dict[str, float] = {}
        # Track declared variables: name → type ('ex' or 'double')
        self._declared_vars: dict[str, str] = {}
        # Nesting depth of unresolved runtime conditions
        self._runtime_cond_depth: int = 0
        # CSE: track symbol versions for non-constant variables
        self._var_versions: dict[str, int] = {}
        self._var_sym_name: dict[str, str] = {}

        # Regime compilation: forced condition outcomes
        self._forced_conditions = forced_conditions  # by index (deprecated)
        self._forced_nodes = forced_nodes  # by id(ast_node)
        self._assume_true = assume_true or set()
        # Registry of voltage-dependent conditions encountered during walk
        self._condition_registry: list[str] = []

    def emit(self) -> str:
        """Generate the GiNaC C++ program source."""
        return self._emit_ginac_program()

    # --- Condition evaluation ---

    def _preprocess_sys_funcs(self, cond: str) -> str:
        """Replace $port_connected and $param_given with constants."""
        # $port_connected(x) → 1 (all ports connected, temp is a node)
        cond = re.sub(r'\$port_connected\s*\(\s*\w+\s*\)', '1', cond)
        # $param_given(X) → 1 if X was explicitly supplied, 0 otherwise
        def _pg(m):
            return '1' if m.group(1) in self._given_params else '0'
        cond = re.sub(r'\$param_given\s*\(\s*(\w+)\s*\)', _pg, cond)
        return cond

    def _eval_condition(self, cond: str) -> Optional[bool]:
        """Try to evaluate a condition using known parameter values.

        Returns True/False if fully resolvable, None if it depends on
        instance params or node voltages.
        """
        if cond is None:
            return None

        cond = self._preprocess_sys_funcs(cond)

        # Extract identifiers from condition
        idents = set(re.findall(r'[A-Za-z_]\w*', cond))
        idents -= {'if', 'else', 'begin', 'end'}

        # Check if all identifiers are known constants (not instance params)
        param_names = {p.name for p in self.mod.params}
        for ident in idents:
            if ident in self.instance_params:
                return None  # Depends on instance param — can't resolve
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
        ns = dict(self.param_values)
        ns.update(self.var_values)

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
        if '$vt' in cond or '$temperature' in cond:
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
        ns = dict(self.param_values)
        ns.update(self.var_values)
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
            # Use versioned symbol name for non-constant variables
            if name in self._var_sym_name:
                return self._var_sym_name[name]
            return name
        return re.sub(r'\b[A-Za-z_]\w*\b', _sub, expr)

    # --- GiNaC C++ emission ---

    def _emit_ginac_program(self) -> str:
        lines = []
        lines.append('// AUTO-GENERATED by VAE — GiNaC code generator')
        lines.append(f'// Source model: {self.mod.name}')
        lines.append(f'// Resolved {len(self.param_values)} model parameters at elaboration time')
        lines.append(f'// Instance parameters (symbolic): {sorted(self.instance_params)}')
        lines.append('#include <cmath>')
        lines.append('#include <ginac/ginac.h>')
        lines.append('#include <iostream>')
        lines.append('#include <sstream>')
        lines.append('#include <string>')
        lines.append('#include <set>')
        lines.append('#include <vector>')
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
        lines.append('ex gmin(const ex& a, const ex& b) {')
        lines.append('    // If both numeric, evaluate directly')
        lines.append('    if (is_a<numeric>(a) && is_a<numeric>(b))')
        lines.append('        return ex_to<numeric>(a) < ex_to<numeric>(b) ? a : b;')
        lines.append('    return a;  // fallback: return first arg (imprecise but safe for constants)')
        lines.append('}')
        lines.append('ex gmax(const ex& a, const ex& b) {')
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
        lines.append('int main() {')

        # Declare symbols for node voltages
        lines.append('    // Node voltage symbols')
        for node in self.all_nodes:
            lines.append(f'    symbol V_{node}("V_{node}");')

        # No parameter symbols — all params are constants at compile time

        # Special symbols
        lines.append('    symbol Vt("Vt");')
        lines.append('    symbol temperature("temperature");')
        lines.append('')

        # Determine active nodes by walking AST with resolved conditions
        active_nodes = list(self.port_names)
        shorted = self._find_shorted_nodes(self.mod.analog_block)
        for n in self.internal_nodes:
            if n not in shorted:
                active_nodes.append(n)

        n_nodes = len(active_nodes)

        # Emit shorting annotations
        for n in sorted(shorted):
            target = self._find_short_target(self.mod.analog_block, n)
            if target:
                lines.append(f'    // SHORT: {n} → {target}')

        sym_list = ", ".join(f"V_{n}" for n in active_nodes)
        name_list = ", ".join('"' + n + '"' for n in active_nodes)
        lines.append(f'    vector<symbol> nodes = {{{sym_list}}};')
        lines.append(f'    vector<string> node_names = {{{name_list}}};')
        lines.append(f'    int n_nodes = {n_nodes};')
        lines.append('')

        # Intermediates for CSE — each variable assignment records (symbol, expression)
        lines.append(f'    // CSE intermediates: symbol, expression, output name')
        lines.append(f'    vector<symbol> int_syms;')
        lines.append(f'    vector<ex> int_exprs;')
        lines.append(f'    vector<string> int_names;')
        lines.append('')

        # Branch/contribution vectors
        lines.append(f'    vector<ex> F_contribs;')
        lines.append(f'    vector<ex> Q_contribs;')
        lines.append(f'    vector<string> branch_labels;')
        lines.append('')

        # Walk analog block, emitting GiNaC expressions
        self._walk_analog_block(lines, self.mod.analog_block,
                               active_nodes, indent=4)

        # Print C++ eval/jacobian code
        lines.append('')
        lines.append(f'    int n_branches = F_contribs.size();')
        lines.append('')

        self._emit_code_printer(lines, active_nodes)

        lines.append('    return 0;')
        lines.append('}')
        return '\n'.join(lines) + '\n'

    def _emit_code_printer(self, lines: list[str], active_nodes: list[str]):
        """Emit code that prints eval/jacobian using forward-mode AD through intermediates."""
        n_nodes = len(active_nodes)
        I = '    '  # base indent

        # --- vae_eval ---
        lines.append(f'{I}cout << "void vae_eval(VaeState* s, double* F, double* Q) {{" << endl;')
        for i, n in enumerate(active_nodes):
            lines.append(f'{I}cout << "    double V_{n} = s->V[{i}];" << endl;')
        lines.append(f'{I}cout << "    double Vt = s->Vt;" << endl;')
        lines.append(f'{I}cout << endl;')

        # Print intermediate computations using symbol names
        lines.append(f'{I}set<string> declared;')
        lines.append(f'{I}for (int i = 0; i < (int)int_syms.size(); i++) {{')
        lines.append(f'{I}    string sn = int_syms[i].get_name();')
        lines.append(f'{I}    if (declared.count(sn) == 0) {{')
        lines.append(f'{I}        cout << "    double " << sn << " = " << to_cpp(int_exprs[i]) << ";" << endl;')
        lines.append(f'{I}        declared.insert(sn);')
        lines.append(f'{I}    }} else {{')
        lines.append(f'{I}        cout << "    " << sn << " = " << to_cpp(int_exprs[i]) << ";" << endl;')
        lines.append(f'{I}    }}')
        lines.append(f'{I}}}')

        # Print contributions
        lines.append(f'{I}for (int i = 0; i < n_branches; i++) {{')
        lines.append(f'{I}    cout << "    F[" << i << "] = " << to_cpp(F_contribs[i]) << ";  // " << branch_labels[i] << endl;')
        lines.append(f'{I}}}')
        lines.append(f'{I}for (int i = 0; i < n_branches; i++) {{')
        lines.append(f'{I}    if (!Q_contribs[i].is_zero())')
        lines.append(f'{I}        cout << "    Q[" << i << "] = " << to_cpp(Q_contribs[i]) << ";" << endl;')
        lines.append(f'{I}}}')
        lines.append(f'{I}cout << "}}" << endl << endl;')

        # --- vae_jacobian using forward-mode AD ---
        lines.append(f'{I}cout << "void vae_jacobian(VaeState* s, double* dFdV, double* dQdV) {{" << endl;')
        for i, n in enumerate(active_nodes):
            lines.append(f'{I}cout << "    double V_{n} = s->V[{i}];" << endl;')
        lines.append(f'{I}cout << "    double Vt = s->Vt;" << endl;')
        lines.append(f'{I}cout << endl;')

        # Print intermediate computations (same as eval, with dedup)
        lines.append(f'{I}declared.clear();')
        lines.append(f'{I}for (int i = 0; i < (int)int_syms.size(); i++) {{')
        lines.append(f'{I}    string sn = int_syms[i].get_name();')
        lines.append(f'{I}    if (declared.count(sn) == 0) {{')
        lines.append(f'{I}        cout << "    double " << sn << " = " << to_cpp(int_exprs[i]) << ";" << endl;')
        lines.append(f'{I}        declared.insert(sn);')
        lines.append(f'{I}    }} else {{')
        lines.append(f'{I}        cout << "    " << sn << " = " << to_cpp(int_exprs[i]) << ";" << endl;')
        lines.append(f'{I}    }}')
        lines.append(f'{I}}}')
        lines.append(f'{I}cout << endl;')

        # Forward-mode AD: for each node voltage, propagate derivatives through intermediates
        lines.append(f'{I}int n_ints = int_syms.size();')
        lines.append(f'{I}cerr << "Computing Jacobian via forward-mode AD..." << endl;')
        lines.append(f'{I}cerr << "  " << n_ints << " intermediates, " << n_nodes << " node voltages" << endl;')
        lines.append(f'{I}for (int col = 0; col < n_nodes; col++) {{')
        lines.append(f'{I}    cerr << "  d/d" << node_names[col] << "..." << endl;')

        # Create derivative symbols and compute chain rule
        lines.append(f'{I}    // Forward sweep: compute d(intermediate)/d(node_voltage)')
        lines.append(f'{I}    vector<pair<symbol, symbol>> deriv_pairs;  // (int_sym, deriv_sym)')
        lines.append(f'{I}    for (int i = 0; i < n_ints; i++) {{')
        lines.append(f'{I}        ex d = int_exprs[i].diff(nodes[col]);')
        lines.append(f'{I}        // Chain rule: add contributions from prior intermediates')
        lines.append(f'{I}        for (auto& [dep_sym, dep_dsym] : deriv_pairs) {{')
        lines.append(f'{I}            ex pd = int_exprs[i].diff(dep_sym);')
        lines.append(f'{I}            if (!pd.is_zero()) d += pd * dep_dsym;')
        lines.append(f'{I}        }}')
        lines.append(f'{I}        if (!d.is_zero()) {{')
        lines.append(f'{I}            string dname = "d_" + int_syms[i].get_name() + "_d" + node_names[col];')
        lines.append(f'{I}            symbol ds(dname);')
        lines.append(I + '            deriv_pairs.push_back({int_syms[i], ds});')
        lines.append(f'{I}            if (declared.count(dname) == 0) {{')
        lines.append(f'{I}                cout << "    double " << dname << " = " << to_cpp(d) << ";" << endl;')
        lines.append(f'{I}                declared.insert(dname);')
        lines.append(f'{I}            }} else {{')
        lines.append(f'{I}                cout << "    " << dname << " = " << to_cpp(d) << ";" << endl;')
        lines.append(f'{I}            }}')
        lines.append(f'{I}        }}')
        lines.append(f'{I}    }}')

        # Contribution derivatives
        lines.append(f'{I}    // Contribution derivatives')
        lines.append(f'{I}    for (int br = 0; br < n_branches; br++) {{')
        lines.append(f'{I}        ex dF = F_contribs[br].diff(nodes[col]);')
        lines.append(f'{I}        for (auto& [dep_sym, dep_dsym] : deriv_pairs) {{')
        lines.append(f'{I}            ex pd = F_contribs[br].diff(dep_sym);')
        lines.append(f'{I}            if (!pd.is_zero()) dF += pd * dep_dsym;')
        lines.append(f'{I}        }}')
        lines.append(f'{I}        if (!dF.is_zero())')
        lines.append(f'{I}            cout << "    dFdV[" << br << " * " << n_nodes << " + " << col')
        lines.append(f'{I}                 << "] = " << to_cpp(dF) << ";" << endl;')
        lines.append(f'{I}        ex dQ = Q_contribs[br].diff(nodes[col]);')
        lines.append(f'{I}        for (auto& [dep_sym, dep_dsym] : deriv_pairs) {{')
        lines.append(f'{I}            ex pd = Q_contribs[br].diff(dep_sym);')
        lines.append(f'{I}            if (!pd.is_zero()) dQ += pd * dep_dsym;')
        lines.append(f'{I}        }}')
        lines.append(f'{I}        if (!dQ.is_zero())')
        lines.append(f'{I}            cout << "    dQdV[" << br << " * " << n_nodes << " + " << col')
        lines.append(f'{I}                 << "] = " << to_cpp(dQ) << ";" << endl;')
        lines.append(f'{I}    }}')
        lines.append(f'{I}}}')
        lines.append(f'{I}cout << "}}" << endl << endl;')

        # Metadata
        lines.append(f'{I}cout << "// n_nodes = " << n_nodes << endl;')
        lines.append(f'{I}cout << "// n_branches = " << n_branches << endl;')
        lines.append(f'{I}cout << "// intermediates = " << n_ints << endl;')
        lines.append(f'{I}cout << "// nodes:";')
        lines.append(f'{I}for (auto& n : node_names) cout << " " << n;')
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

    def _walk_analog_block(self, lines: list[str], node: ASTNode,
                          active_nodes: list[str], indent: int):
        """Walk AST and emit GiNaC expression-building code.

        Parameter-dependent conditions are resolved using self.param_values.
        Only instance-param or voltage-dependent conditions remain as runtime code.
        """
        if node is None:
            return
        pfx = ' ' * indent

        if node.kind == NodeKind.BLOCK:
            for child in node.children:
                self._walk_analog_block(lines, child, active_nodes, indent)

        elif node.kind == NodeKind.ASSIGN:
            self._emit_line_directive(lines, node, pfx)
            uses_voltage = self._expr_uses_voltage(node.expr)

            # Try constant evaluation
            if not uses_voltage:
                val = self._try_eval_expr(node.expr)
                if val is not None:
                    self.var_values[node.lhs] = val
                    if self._runtime_cond_depth == 0:
                        # Outside runtime conditions: pure constant, no GiNaC symbol needed
                        return
                    # Inside runtime conditions: still track value for downstream
                    # condition resolution, but also emit GiNaC symbol for output
                else:
                    self.var_values.pop(node.lhs, None)
            else:
                self.var_values.pop(node.lhs, None)

            # Non-constant: create a GiNaC symbol for CSE
            ginac_expr = self._to_ginac_expr(node.expr, active_nodes)

            if node.lhs not in self._declared_vars:
                # First assignment — create new symbol
                sym_name = node.lhs
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
                sym_name = f'{node.lhs}__{self._var_versions[node.lhs]}'
                self._var_sym_name[node.lhs] = sym_name
                lines.append(f'{pfx}symbol {sym_name}("{sym_name}");')

            lines.append(f'{pfx}int_syms.push_back({sym_name});')
            lines.append(f'{pfx}int_exprs.push_back({ginac_expr});')
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
                            # No forced outcome — emit C++ if/else with GiNaC runtime check
                            self._predeclare_vars(lines, node, pfx)
                            cpp_cond = self._condition_to_cpp(node.condition)
                            lines.append(f'{pfx}// RUNTIME CONDITION [{cond_idx}]: {node.condition}')
                            lines.append(f'{pfx}if ({cpp_cond}) {{')
                            self._runtime_cond_depth += 1
                            for child in node.children:
                                self._walk_analog_block(lines, child, active_nodes, indent + 4)
                            self._runtime_cond_depth -= 1
                            if node.else_body:
                                lines.append(f'{pfx}}} else {{')
                                self._runtime_cond_depth += 1
                                self._walk_analog_block(lines, node.else_body, active_nodes, indent + 4)
                                self._runtime_cond_depth -= 1
                            lines.append(f'{pfx}}}')

        elif node.kind == NodeKind.INITIAL_STEP:
            pass  # Skip initial_step

        elif node.kind == NodeKind.EXPR:
            pass  # Skip raw expressions ($strobe, $finish, etc.)

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
        is_ddt = False
        inner_expr = expr_str
        # Check if expression contains ddt() — may be wrapped: 1.0 * ddt(q)
        if 'ddt' in expr_str:
            # Extract ddt argument and treat as charge contribution
            # ddt may contain a complex expression: ddt(MULT_i * Qg)
            m = re.search(r'ddt\s*\(', expr_str)
            if m:
                is_ddt = True
                # Find matching close paren
                start = m.end()
                depth = 1
                i = start
                while i < len(expr_str) and depth > 0:
                    if expr_str[i] == '(':
                        depth += 1
                    elif expr_str[i] == ')':
                        depth -= 1
                    i += 1
                ddt_arg = expr_str[start:i-1]
                inner_expr = expr_str[:m.start()] + ddt_arg + expr_str[i:]

        ginac_expr = self._to_ginac_expr(inner_expr, active_nodes)
        branch_label = f'{node.contrib_kind.name}({",".join(node.branch)})'

        br_p = node.branch[0]
        br_n = node.branch[1] if len(node.branch) > 1 else 'gnd'

        # Resolve branch names to their node
        if br_p in self.branch_map:
            br_p = self.branch_map[br_p]
        if br_n in self.branch_map:
            br_n = self.branch_map[br_n]

        # Resolve shorted nodes (returns None if shorted to ground)
        if br_p not in active_nodes and br_p in self.internal_nodes:
            br_p = self._resolve_shorted_node(br_p, active_nodes)
        if br_n != 'gnd' and br_n not in active_nodes and br_n in self.internal_nodes:
            br_n = self._resolve_shorted_node(br_n, active_nodes)

        # Nodes resolved to ground become 'gnd'
        if br_p is None:
            br_p = 'gnd'
        if br_n is None:
            br_n = 'gnd'

        # Skip tautological V(c,c) <+ 0 or V(gnd) <+ 0
        if node.contrib_kind == ContribKind.V and br_p == br_n:
            return
        if node.contrib_kind == ContribKind.V and br_p == 'gnd' and br_n == 'gnd':
            return

        self._emit_line_directive(lines, node, pfx)
        lines.append(f'{pfx}{{ // {branch_label}')
        lines.append(f'{pfx}    int br_idx = -1;')
        lines.append(f'{pfx}    for (int i = 0; i < (int)branch_labels.size(); i++)')
        lines.append(f'{pfx}        if (branch_labels[i] == "{branch_label}") {{ br_idx = i; break; }}')
        lines.append(f'{pfx}    if (br_idx < 0) {{')
        lines.append(f'{pfx}        br_idx = F_contribs.size();')
        lines.append(f'{pfx}        F_contribs.push_back(ex(0));')
        lines.append(f'{pfx}        Q_contribs.push_back(ex(0));')
        lines.append(f'{pfx}        branch_labels.push_back("{branch_label}");')
        lines.append(f'{pfx}    }}')
        if is_ddt:
            lines.append(f'{pfx}    Q_contribs[br_idx] += {ginac_expr};')
        elif node.contrib_kind == ContribKind.V:
            if br_n != 'gnd' and br_n in active_nodes:
                lines.append(f'{pfx}    F_contribs[br_idx] += {ginac_expr} - (V_{br_p} - V_{br_n});')
            elif br_p != 'gnd':
                lines.append(f'{pfx}    F_contribs[br_idx] += {ginac_expr} - V_{br_p};')
            else:
                # V(gnd,...) — expr contributes directly
                lines.append(f'{pfx}    F_contribs[br_idx] += {ginac_expr};')
        else:
            lines.append(f'{pfx}    F_contribs[br_idx] += {ginac_expr};')
        lines.append(f'{pfx}}}')

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
        # Check if any referenced variable is voltage-dependent
        # (declared as symbol AND not known as a constant in var_values)
        idents = set(re.findall(r'\b[A-Za-z_]\w*\b', expr))
        for ident in idents:
            if ident in self._declared_vars and ident not in self.var_values:
                return True
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
                    if eval(py_cond, {"__builtins__": {}}, self.param_values):
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

        # V(a,b) → (V_a - V_b), V(a) → V_a, grounded nodes → 0
        def repl_v(m):
            p1 = m.group(1).strip()
            p2 = m.group(2).strip() if m.group(2) else None
            # Resolve branch names to their node
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

        # I(branch) probe → 0 for DC/transient (noise branches carry no signal current)
        result = re.sub(r'I\s*\(\s*\w+\s*\)', 'ex(0)', result)

        # $limit(expr, ...) → just expr
        result = re.sub(r'\$limit\s*\(\s*([^,]+),.*?\)', r'\1', result)

        # Temp(node) → V_node (thermal node temperature access)
        result = re.sub(r'Temp\s*\(\s*(\w+)\s*\)', lambda m: f'V_{m.group(1)}', result)

        # $vt → Vt, $temperature → temperature
        result = re.sub(r'\$vt\b', 'Vt', result)
        result = re.sub(r'\$temperature\b', 'temperature', result)

        # $simparam("name", default) → default value
        result = re.sub(r'\$simparam\s*\(\s*"[^"]*"\s*,\s*([^)]+)\)', r'\1', result)
        result = re.sub(r'\$simparam\s*\(\s*"[^"]*"\s*\)', '0', result)

        # Function mapping
        for va_name, ginac_name in _VA_TO_GINAC.items():
            result = re.sub(rf'\b{va_name}\b(?=\s*\()', ginac_name, result)
        # min/max → gmin/gmax (GiNaC-compatible helpers)
        result = re.sub(r'\bmin\b(?=\s*\()', 'gmin', result)
        result = re.sub(r'\bmax\b(?=\s*\()', 'gmax', result)

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
                    ns = dict(self.param_values)
                    ns.update(self.var_values)
                    result_val = eval(py_cond, {"__builtins__": {}}, ns)
                    replacement = true_val if result_val else false_val
                    s = s[:start] + f'({replacement})' + s[end:]
                except Exception:
                    break  # Can't evaluate — leave as-is
            return s
        result = _eval_ternary(result)

        # Single-pass substitution of known constant params and vars
        # Build the substitutable set (exclude funcs, keywords, node voltages)
        _skip = set(_VA_TO_GINAC.values()) | {'numeric', 'ex', 'Vt', 'temperature'}
        _skip |= {f'V_{n}' for n in self.all_nodes}
        _skip |= {'gmin', 'gmax', 'hypsmooth', 'hypmax', 'Tempdep'}
        _skip |= self.instance_params
        def _subst_known(m):
            name = m.group(0)
            if name in _skip:
                return name
            if name in self.param_values:
                return repr(self.param_values[name])
            if name in self.var_values:
                return repr(self.var_values[name])
            # Use versioned symbol name for non-constant variables
            if name in self._var_sym_name:
                return self._var_sym_name[name]
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
                        s = s[:start] + f'gmax({true_val}, {false_val})' + s[end:]
                        continue
                    if op == '>' and lhs == fv_n and rhs == tv_n:
                        s = s[:start] + f'gmin({true_val}, {false_val})' + s[end:]
                        continue
                    if op == '<' and lhs == tv_n and rhs == fv_n:
                        s = s[:start] + f'gmin({true_val}, {false_val})' + s[end:]
                        continue
                    if op == '<' and lhs == fv_n and rhs == tv_n:
                        s = s[:start] + f'gmax({true_val}, {false_val})' + s[end:]
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

        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def _try_eval_expr(self, expr: str) -> Optional[float]:
        """Try to evaluate expression as a constant float.

        Returns float value if all referenced params/vars are known constants,
        None otherwise.
        """
        if expr is None:
            return None

        s = expr
        # $simparam("name", default) → default value (before $ check)
        s = re.sub(r'\$simparam\s*\(\s*"[^"]*"\s*,\s*([^)]+)\)', r'\1', s)
        s = re.sub(r'\$simparam\s*\(\s*"[^"]*"\s*\)', '0', s)
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

        ns = {}
        for ident in idents:
            if ident in self.param_values and ident not in self.instance_params:
                ns[ident] = self.param_values[ident]
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

        # Translate operators
        py = expr.replace('&&', ' and ').replace('||', ' or ')

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
        except Exception:
            return None

    # --- Node resolution ---

    def _resolve_shorted_node(self, node: str, active_nodes: list[str]) -> Optional[str]:
        """Resolve a shorted internal node to its target.

        Returns the target node name, or None if the node is shorted to ground
        (e.g. single-node V(N) <+ 0).
        """
        target = self._find_short_target(self.mod.analog_block, node)
        if target and target in active_nodes:
            return target
        # No partner found — node is shorted to ground
        return None

    def _find_short_target(self, ast_node: ASTNode, internal: str) -> Optional[str]:
        """Find which node an internal node is shorted to via V(x,y) <+ 0."""
        if ast_node is None:
            return None
        if ast_node.kind == NodeKind.CONTRIB and ast_node.contrib_kind == ContribKind.V:
            expr = ast_node.expr.strip()
            if expr in ('0', '0.0'):
                branch = ast_node.branch
                if internal in branch:
                    for b in branch:
                        if b != internal:
                            return b
        if ast_node.kind in (NodeKind.BLOCK, NodeKind.IF):
            for c in ast_node.children:
                r = self._find_short_target(c, internal)
                if r:
                    return r
            if ast_node.kind == NodeKind.IF and ast_node.else_body:
                r = self._find_short_target(ast_node.else_body, internal)
                if r:
                    return r
        return None

    def _find_shorted_nodes(self, node: ASTNode) -> set[str]:
        """Find internal nodes shorted by V(x,y) <+ 0, respecting resolved conditions."""
        shorted = set()
        if node is None:
            return shorted
        if node.kind == NodeKind.BLOCK:
            for child in node.children:
                shorted |= self._find_shorted_nodes(child)
        elif node.kind == NodeKind.CONTRIB:
            if node.contrib_kind == ContribKind.V:
                expr = node.expr.strip()
                if expr == '0' or expr == '0.0':
                    for b in node.branch:
                        if b in self.internal_nodes:
                            shorted.add(b)
        elif node.kind == NodeKind.IF:
            resolved = self._eval_condition(node.condition)
            if resolved is True:
                for child in node.children:
                    shorted |= self._find_shorted_nodes(child)
            elif resolved is False:
                if node.else_body:
                    shorted |= self._find_shorted_nodes(node.else_body)
            else:
                # Can't resolve — include both branches
                for child in node.children:
                    shorted |= self._find_shorted_nodes(child)
                if node.else_body:
                    shorted |= self._find_shorted_nodes(node.else_body)
        return shorted

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
        lines.append('extern "C" uint64_t vae_eval_regime(double* V, int n_nodes, double Vt) {')
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
            lines.append(f'    double temperature = V[{temp_idx}];  // temperature node (V_{active_nodes[temp_idx]})')
        else:
            lines.append(f'    double temperature = 300.15;  // nominal (no thermal node)')
        lines.append('')

        # Reset var_values for the walk (keep param_values intact)
        saved_var_values = dict(self.var_values)
        saved_declared = dict(self._declared_vars)
        saved_runtime_depth = self._runtime_cond_depth
        saved_versions = dict(self._var_versions)
        saved_sym_name = dict(self._var_sym_name)
        saved_registry = list(self._condition_registry)

        self.var_values = {}
        self._declared_vars = {}
        self._runtime_cond_depth = 0
        self._var_versions = {}
        self._var_sym_name = {}
        self._condition_registry = []

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

        # V(a,b) → (V_a - V_b), V(a) → V_a, grounded nodes → 0.0
        def repl_v(m):
            p1 = m.group(1).strip()
            p2 = m.group(2).strip() if m.group(2) else None
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

        # $vt → Vt, $temperature → temperature
        result = re.sub(r'\$vt\b', 'Vt', result)
        result = re.sub(r'\$temperature\b', 'temperature', result)

        # $simparam("name", default) → default value
        result = re.sub(r'\$simparam\s*\(\s*"[^"]*"\s*,\s*([^)]+)\)', r'\1', result)
        result = re.sub(r'\$simparam\s*\(\s*"[^"]*"\s*\)', '0.0', result)

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
                    ns = dict(self.param_values)
                    ns.update(self.var_values)
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
                     'Vt', 'temperature', 'key'}
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

        # V(a,b) → (V_a - V_b), V(a) → V_a
        def repl_v(m):
            p1 = m.group(1).strip()
            p2 = m.group(2).strip() if m.group(2) else None
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

        # $vt → Vt, $temperature → temperature
        c = re.sub(r'\$vt\b', 'Vt', c)
        c = re.sub(r'\$temperature\b', 'temperature', c)

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
                 'Vt', 'temperature', 'key'}
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
                       line_directives: Optional[bool] = None) -> str:
    """Generate GiNaC C++ source from a parsed Verilog-A module.

    All parameters (model card + instance) must be supplied as constants.
    Only node voltages remain symbolic for differentiation.
    Each unique parameter set gets its own compiled code; common geometries
    share the same .so.

    Args:
        module: Parsed Verilog-A module.
        param_values: All parameter values (model + instance). Merged over AST defaults.
        line_directives: None=off, True=#line (active), False=//line (inactive).
    """
    return GiNaCEmitter(module, param_values=param_values,
                        line_directives=line_directives).emit()
