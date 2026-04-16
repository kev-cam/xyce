"""
VAE code generator — transforms parsed Verilog-A AST into C++ .so source.

Generates:
  - vae_eval(): stamps current/charge contributions into output arrays
  - vae_jacobian(): analytic Jacobian via sympy differentiation
  - vae_model_name(), vae_port_count(), etc.: metadata

The generated code uses the VaeState struct for node voltages and parameters.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
import sympy

from .parser import (
    Module, ASTNode, NodeKind, ContribKind, Port, PortDir, Param, Var
)


# ---------------------------------------------------------------------------
# Expression translation: Verilog-A expression string → sympy / C++ string
# ---------------------------------------------------------------------------

# Verilog-A built-in functions → C++ equivalents
_VA_BUILTINS = {
    'ln': 'log',
    'limexp': 'vae_limexp',
    'abs': 'fabs',
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
    'min': 'fmin',
    'max': 'fmax',
    'floor': 'floor',
    'ceil':  'ceil',
}

# Verilog-A → sympy function mapping
_VA_TO_SYMPY = {
    'ln': 'log',
    'limexp': 'exp',  # limexp is exp with limiting — treat as exp for differentiation
    'abs': 'Abs',
    'pow': 'Pow',
}


@dataclass
class BranchRef:
    """Reference to a branch voltage V(a,b) or current I(a,b)."""
    kind: str          # 'V' or 'I'
    ports: tuple[str, ...]

    @property
    def symbol_name(self) -> str:
        return f"{self.kind}_{'_'.join(self.ports)}"


@dataclass
class Contribution:
    """A current or charge contribution to a branch."""
    kind: ContribKind   # I or V
    branch: tuple[str, ...]
    expr_str: str       # C++ expression string
    sympy_expr: object  # sympy expression for differentiation
    is_ddt: bool = False  # True if this is a ddt() contribution (charge)


class CodeGen:
    """Generate C++ source from a parsed Verilog-A module."""

    def __init__(self, module: Module):
        self.mod = module
        self.port_names = [p.name for p in module.ports]
        self.internal_nodes = list(module.internal_nodes)
        self.all_nodes = self.port_names + self.internal_nodes
        self.n_ports = len(self.port_names)
        self.n_internal = len(self.internal_nodes)
        self.n_nodes = self.n_ports + self.n_internal

        # Sympy symbols for node voltages
        self.v_syms: dict[str, sympy.Symbol] = {}
        for node in self.all_nodes:
            self.v_syms[node] = sympy.Symbol(f'V_{node}', real=True)

        # Sympy symbols for parameters
        self.p_syms: dict[str, sympy.Symbol] = {}
        for p in module.params:
            self.p_syms[p.name] = sympy.Symbol(p.name, real=True, positive=True)

        # Sympy symbols for variables
        self.var_syms: dict[str, sympy.Symbol] = {}
        for v in module.variables:
            self.var_syms[v.name] = sympy.Symbol(v.name, real=True)

        # Special symbols
        self.sym_temperature = sympy.Symbol('temperature', real=True, positive=True)
        self.sym_vt = sympy.Symbol('Vt', real=True, positive=True)

        # Collected contributions
        self.contributions: list[Contribution] = []
        # Variable assignments (for substitution)
        self.var_assignments: dict[str, sympy.Expr] = {}
        # Initial step assignments
        self.initial_assignments: dict[str, str] = {}
        # C++ statements for eval body
        self.eval_stmts: list[str] = []
        # C++ statements for jacobian body (emitted inline with contributions)
        self.jac_stmts: list[str] = []

    def generate(self) -> str:
        """Generate complete C++ source file."""
        self._process_analog_block(self.mod.analog_block)
        return self._emit_cpp()

    # --- AST walking ---

    def _process_analog_block(self, node: ASTNode):
        if node is None:
            return
        if node.kind == NodeKind.BLOCK:
            for child in node.children:
                self._process_analog_block(child)
        elif node.kind == NodeKind.ASSIGN:
            self._process_assign(node)
        elif node.kind == NodeKind.CONTRIB:
            self._process_contrib(node)
        elif node.kind == NodeKind.IF:
            self._process_if(node)
        elif node.kind == NodeKind.INITIAL_STEP:
            self._process_initial_step(node)
        elif node.kind == NodeKind.EXPR:
            pass  # Skip unknown expressions

    def _process_assign(self, node: ASTNode):
        cpp_expr = self._translate_expr(node.expr)
        sympy_expr = self._to_sympy(node.expr)
        self.var_assignments[node.lhs] = sympy_expr
        self.eval_stmts.append(f'    double {node.lhs} = {cpp_expr};')
        self.jac_stmts.append(f'    double {node.lhs} = {cpp_expr};')

    def _process_contrib(self, node: ASTNode):
        expr_str = node.expr.strip()

        # Check for ddt()
        is_ddt = False
        inner_expr = expr_str
        ddt_match = re.match(r'^ddt\s*\((.*)\)$', expr_str)
        if ddt_match:
            is_ddt = True
            inner_expr = ddt_match.group(1).strip()

        cpp_expr = self._translate_expr(inner_expr)
        sympy_expr = self._to_sympy(inner_expr)

        contrib = Contribution(
            kind=node.contrib_kind,
            branch=node.branch,
            expr_str=cpp_expr,
            sympy_expr=sympy_expr,
            is_ddt=is_ddt,
        )
        self.contributions.append(contrib)

        # Emit eval contribution statement
        idx = self._contrib_index(node.contrib_kind, node.branch)
        if is_ddt:
            self.eval_stmts.append(f'    Q[{idx}] += {cpp_expr};')
        elif node.contrib_kind == ContribKind.V:
            p, q = node.branch[0], node.branch[1] if len(node.branch) > 1 else 'gnd'
            self.eval_stmts.append(f'    // V({",".join(node.branch)}) <+ {inner_expr}')
            self.eval_stmts.append(f'    F[{idx}] += {cpp_expr} - V_{p}_{q};')
        else:
            self.eval_stmts.append(f'    F[{idx}] += {cpp_expr};')

        # Emit inline Jacobian stamps for this contribution
        full_expr = self._substitute_vars(sympy_expr)

        # For V contributions: F = expr - V_branch, so adjust
        if node.contrib_kind == ContribKind.V and not is_ddt:
            p, q = node.branch[0], node.branch[1] if len(node.branch) > 1 else 'gnd'
            vp = self.v_syms.get(p)
            vq = self.v_syms.get(q)
            if vp:
                full_expr = full_expr - vp
            if vq:
                full_expr = full_expr + vq

        target = 'dQdV' if is_ddt else 'dFdV'
        for col, nd in enumerate(self.all_nodes):
            vsym = self.v_syms[nd]
            deriv = sympy.diff(full_expr, vsym)
            if deriv != 0:
                cpp_deriv = self._sympy_to_cpp(deriv)
                self.jac_stmts.append(
                    f'    {target}[{idx} * {self.n_nodes} + {col}] += {cpp_deriv};')

    def _process_if(self, node: ASTNode):
        cond_cpp = self._translate_expr(node.condition)
        self.eval_stmts.append(f'    if ({cond_cpp}) {{')
        self.jac_stmts.append(f'    if ({cond_cpp}) {{')
        for child in node.children:
            self._process_analog_block(child)
        if node.else_body:
            self.eval_stmts.append(f'    }} else {{')
            self.jac_stmts.append(f'    }} else {{')
            self._process_analog_block(node.else_body)
        self.eval_stmts.append(f'    }}')
        self.jac_stmts.append(f'    }}')

    def _process_initial_step(self, node: ASTNode):
        # Collect initial step assignments — these go in a separate init function
        for child in node.children:
            if child.kind == NodeKind.BLOCK:
                for c in child.children:
                    if c.kind == NodeKind.ASSIGN:
                        self.initial_assignments[c.lhs] = self._translate_expr(c.expr)

    # --- Branch/contribution indexing ---

    def _branch_pairs(self) -> list[tuple[str, str]]:
        """Get unique branch pairs from contributions."""
        seen = []
        for c in self.contributions:
            pair = (c.branch[0], c.branch[1] if len(c.branch) > 1 else 'gnd')
            if pair not in seen:
                seen.append(pair)
        return seen

    def _contrib_index(self, kind: ContribKind, branch: tuple) -> int:
        """Map a contribution to its index in F[] / Q[] arrays."""
        pair = (branch[0], branch[1] if len(branch) > 1 else 'gnd')
        pairs = self._branch_pairs()
        if pair not in pairs:
            pairs.append(pair)
        return pairs.index(pair)

    def _node_index(self, name: str) -> int:
        """Map node name to index in voltage array."""
        if name in self.all_nodes:
            return self.all_nodes.index(name)
        return -1

    # --- Expression translation ---

    def _translate_expr(self, expr: str) -> str:
        """Translate Verilog-A expression to C++."""
        if expr is None:
            return "0.0"

        result = expr

        # V(a,b) → (s->V[idx_a] - s->V[idx_b])
        def repl_v(m):
            p1 = m.group(1).strip()
            p2 = m.group(2).strip() if m.group(2) else None
            if p2:
                i1 = self._node_index(p1)
                i2 = self._node_index(p2)
                return f'(s->V[{i1}] - s->V[{i2}])'
            else:
                i1 = self._node_index(p1)
                return f's->V[{i1}]'
        result = re.sub(r'V\s*\(\s*(\w+)\s*(?:,\s*(\w+)\s*)?\)', repl_v, result)

        # $limit(expr, "func", args...) → just the first argument (limiting handled by wrapper)
        result = re.sub(r'\$limit\s*\(\s*', 'vae_limit(s, ', result)

        # $vt → s->Vt
        result = re.sub(r'\$vt\b', 's->Vt', result)

        # $temperature → s->temperature
        result = re.sub(r'\$temperature\b', 's->temperature', result)

        # Function name mapping
        for va_name, cpp_name in _VA_BUILTINS.items():
            result = re.sub(rf'\b{va_name}\b(?=\s*\()', cpp_name, result)

        # `DEFINE references → their values (handled as constants)
        result = re.sub(r'`(\w+)', r'\1', result)

        # white_noise(...) → 0.0 (noise not evaluated in DC/tran)
        result = re.sub(r'white_noise\s*\([^)]*\)', '0.0', result)
        result = re.sub(r'flicker_noise\s*\([^)]*\)', '0.0', result)

        # Parameter references → s->params[idx]
        # Skip array parameters — those are emitted as namespace-scope
        # static const double[] and referenced directly by name + subscript.
        for i, p in enumerate(self.mod.params):
            if getattr(p, 'array_size', None):
                continue
            result = re.sub(rf'\b{re.escape(p.name)}\b', f's->params[{i}]', result)

        # Clean up spacing
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def _to_sympy(self, expr: str) -> sympy.Expr:
        """Convert Verilog-A expression to sympy for differentiation."""
        if expr is None:
            return sympy.Integer(0)

        s = expr

        # V(a,b) → V_a_b symbol
        def repl_v(m):
            p1 = m.group(1).strip()
            p2 = m.group(2).strip() if m.group(2) else 'gnd'
            if p2 and p2 != 'gnd':
                return f'(V_{p1} - V_{p2})'
            return f'V_{p1}'
        s = re.sub(r'V\s*\(\s*(\w+)\s*(?:,\s*(\w+)\s*)?\)', repl_v, s)

        # $limit(expr, ...) → just expr (for differentiation purposes)
        s = re.sub(r'\$limit\s*\(\s*([^,]+),.*?\)', r'\1', s)

        # $vt → Vt
        s = re.sub(r'\$vt\b', 'Vt', s)
        # $temperature → temperature
        s = re.sub(r'\$temperature\b', 'temperature', s)
        # `DEFINE → constant name
        s = re.sub(r'`(\w+)', r'\1', s)

        # limexp → exp (for differentiation)
        s = re.sub(r'\blimexp\b', 'exp', s)
        s = re.sub(r'\bln\b', 'log', s)

        # white_noise/flicker_noise → 0
        s = re.sub(r'white_noise\s*\([^)]*\)', '0', s)
        s = re.sub(r'flicker_noise\s*\([^)]*\)', '0', s)

        # Build local namespace for sympy parsing
        ns = {}
        for name, sym in self.v_syms.items():
            ns[f'V_{name}'] = sym
        for name, sym in self.p_syms.items():
            ns[name] = sym
        for name, sym in self.var_syms.items():
            ns[name] = sym
        ns['Vt'] = self.sym_vt
        ns['temperature'] = self.sym_temperature

        # Add standard math functions
        ns['exp'] = sympy.exp
        ns['log'] = sympy.log
        ns['sqrt'] = sympy.sqrt
        ns['sin'] = sympy.sin
        ns['cos'] = sympy.cos
        ns['tan'] = sympy.tan
        ns['tanh'] = sympy.tanh
        ns['pow'] = sympy.Pow
        ns['abs'] = sympy.Abs
        ns['fabs'] = sympy.Abs

        # Define placeholder constants
        for define_name in re.findall(r'\b[A-Z_][A-Z_0-9]+\b', s):
            if define_name not in ns:
                ns[define_name] = sympy.Symbol(define_name, real=True, positive=True)

        try:
            return sympy.sympify(s, locals=ns)
        except Exception:
            # Can't parse — return a dummy symbol
            return sympy.Symbol(f'__unparsed_{hash(s) % 10000}')

    # --- Jacobian computation ---

    def _compute_jacobian(self) -> list[tuple[int, int, str]]:
        """Compute analytic Jacobian entries: (row, col, C++ expr).

        For each contribution F[row], differentiate w.r.t. each node voltage V[col].
        Also handles Q (charge) contributions for dF/dV of reactive elements.
        """
        entries = []
        pairs = self._branch_pairs()

        for ci, contrib in enumerate(self.contributions):
            if contrib.is_ddt:
                continue  # dQ/dV handled separately

            row = self._contrib_index(contrib.kind, contrib.branch)

            # Get the full expression with variable substitutions
            expr = contrib.sympy_expr
            expr = self._substitute_vars(expr)

            # Handle V contribution: F = expr - V_branch
            if contrib.kind == ContribKind.V:
                p, q = contrib.branch[0], contrib.branch[1] if len(contrib.branch) > 1 else 'gnd'
                vp = self.v_syms.get(p)
                vq = self.v_syms.get(q)
                if vp:
                    expr = expr - vp
                if vq:
                    expr = expr + vq

            for col, node in enumerate(self.all_nodes):
                vsym = self.v_syms[node]
                deriv = sympy.diff(expr, vsym)
                if deriv != 0:
                    cpp = self._sympy_to_cpp(deriv)
                    entries.append((row, col, cpp))

        # dQ/dV entries (for reactive contributions)
        for ci, contrib in enumerate(self.contributions):
            if not contrib.is_ddt:
                continue
            row = self._contrib_index(contrib.kind, contrib.branch)
            expr = contrib.sympy_expr
            expr = self._substitute_vars(expr)
            for col, node in enumerate(self.all_nodes):
                vsym = self.v_syms[node]
                deriv = sympy.diff(expr, vsym)
                if deriv != 0:
                    cpp = self._sympy_to_cpp(deriv)
                    entries.append((row, col, f'/* dQ/dV */ {cpp}'))

        return entries

    def _substitute_vars(self, expr: sympy.Expr) -> sympy.Expr:
        """Substitute variable assignments into expression."""
        # Do multiple passes to handle chained assignments
        for _ in range(5):
            changed = False
            for name, val in self.var_assignments.items():
                sym = self.var_syms.get(name)
                if sym is not None and expr.has(sym):
                    expr = expr.subs(sym, val)
                    changed = True
            if not changed:
                break
        return expr

    def _sympy_to_cpp(self, expr: sympy.Expr) -> str:
        """Convert sympy expression to C++ string."""
        from sympy.printing.c import C99CodePrinter
        printer = C99CodePrinter()
        result = printer.doprint(expr)
        # Replace symbols — longest first to avoid partial matches
        nodes_sorted = sorted(self.all_nodes, key=len, reverse=True)
        for node in nodes_sorted:
            idx = self._node_index(node)
            result = re.sub(rf'\bV_{re.escape(node)}\b', f's->V[{idx}]', result)
        params_sorted = sorted(self.mod.params, key=lambda p: len(p.name), reverse=True)
        for p in params_sorted:
            result = re.sub(rf'\b{re.escape(p.name)}\b', f's->params[{self._param_index(p.name)}]', result)
        result = re.sub(r'\bVt\b', 's->Vt', result)
        result = re.sub(r'\btemperature\b', 's->temperature', result)
        return result

    def _param_index(self, name: str) -> int:
        for i, p in enumerate(self.mod.params):
            if p.name == name:
                return i
        return -1

    # --- C++ emission ---

    def _emit_cpp(self) -> str:
        pairs = self._branch_pairs()
        n_branches = len(pairs)

        lines = []
        lines.append('// AUTO-GENERATED by VAE — do not edit')
        lines.append(f'// Source: {self.mod.name}')
        lines.append('#include <cmath>')
        lines.append('#include <cstring>')
        lines.append('')
        lines.append('// Limiting exponential — prevents overflow')
        lines.append('static inline double vae_limexp(double x) {')
        lines.append('    if (x > 80.0) return exp(80.0) * (1.0 + x - 80.0);')
        lines.append('    if (x < -80.0) return exp(-80.0);')
        lines.append('    return exp(x);')
        lines.append('}')
        lines.append('')
        lines.append('// $limit stub — limiting handled by device wrapper')
        lines.append('struct VaeState;')
        lines.append('static inline double vae_limit(VaeState*, double v, ...) { return v; }')
        lines.append('')

        # VaeState struct — pointer-based ABI for variable-size models
        lines.append('struct VaeState {')
        lines.append(f'    double* V;            // node voltages (caller-allocated)')
        lines.append(f'    double* params;       // model/instance parameters (caller-allocated)')
        lines.append(f'    double temperature;   // device temperature (K)')
        lines.append(f'    double Vt;            // thermal voltage kT/q')
        lines.append(f'    double dt;            // timestep')
        lines.append(f'    double time;          // current time')
        lines.append('};')
        lines.append('')

        # Preprocessor defines
        for name, value in [('CONSTroot2', '1.41421356237309504880'),
                           ('X_K', '1.3806226e23'),
                           ('CONSTboltz', '1.3806226e-23'),
                           ('CONSTq', '1.602176634e-19')]:
            lines.append(f'#ifndef {name}')
            lines.append(f'#define {name} {value}')
            lines.append(f'#endif')
        lines.append('')

        # --- vae_eval ---
        lines.append('extern "C" {')
        lines.append('')
        lines.append(f'void vae_eval(VaeState* s, double* F, double* Q) {{')

        # Define branch voltages as locals for readability
        for i, (p, q) in enumerate(pairs):
            pi = self._node_index(p)
            qi = self._node_index(q)
            if q == 'gnd':
                lines.append(f'    double V_{p} = s->V[{pi}];')
            else:
                lines.append(f'    double V_{p}_{q} = s->V[{pi}] - s->V[{qi}];')
        lines.append('')

        # Zero output arrays
        lines.append(f'    memset(F, 0, {n_branches} * sizeof(double));')
        lines.append(f'    memset(Q, 0, {n_branches} * sizeof(double));')
        lines.append('')

        # Emit eval body
        for stmt in self.eval_stmts:
            lines.append(stmt)
        lines.append('}')
        lines.append('')

        # --- vae_jacobian ---
        lines.append(f'void vae_jacobian(VaeState* s, double* dFdV, double* dQdV) {{')
        lines.append(f'    // dFdV[row * {self.n_nodes} + col], dQdV[row * {self.n_nodes} + col]')

        # Branch voltage locals
        for i, (p, q) in enumerate(pairs):
            pi = self._node_index(p)
            qi = self._node_index(q)
            if q == 'gnd':
                lines.append(f'    double V_{p} = s->V[{pi}];')
            else:
                lines.append(f'    double V_{p}_{q} = s->V[{pi}] - s->V[{qi}];')

        lines.append(f'    memset(dFdV, 0, {n_branches} * {self.n_nodes} * sizeof(double));')
        lines.append(f'    memset(dQdV, 0, {n_branches} * {self.n_nodes} * sizeof(double));')
        lines.append('')

        # Emit inline Jacobian body (mirrors eval structure with if/else)
        for stmt in self.jac_stmts:
            lines.append(stmt)
        lines.append('}')
        lines.append('')

        # --- Metadata ---
        lines.append(f'const char* vae_model_name() {{ return "{self.mod.name}"; }}')
        lines.append(f'int vae_port_count() {{ return {self.n_ports}; }}')
        lines.append(f'int vae_internal_node_count() {{ return {self.n_internal}; }}')
        lines.append(f'int vae_branch_count() {{ return {n_branches}; }}')
        lines.append('')

        # Port/node name accessors
        lines.append(f'const char* vae_port_name(int i) {{')
        lines.append(f'    static const char* names[] = {{')
        for n in self.all_nodes:
            lines.append(f'        "{n}",')
        lines.append(f'    }};')
        lines.append(f'    return (i >= 0 && i < {self.n_nodes}) ? names[i] : "";')
        lines.append(f'}}')
        lines.append('')

        # Branch info
        lines.append(f'void vae_branch_info(int i, int* node_p, int* node_n, int* is_vsrc) {{')
        for i, (p, q) in enumerate(pairs):
            pi = self._node_index(p)
            qi = self._node_index(q) if q != 'gnd' else -1
            # Check if any V contribution exists for this branch
            is_v = any(c.kind == ContribKind.V and
                      (c.branch[0], c.branch[1] if len(c.branch)>1 else 'gnd') == (p,q)
                      for c in self.contributions if not c.is_ddt)
            lines.append(f'    if (i == {i}) {{ *node_p = {pi}; *node_n = {qi}; *is_vsrc = {int(is_v)}; return; }}')
        lines.append(f'}}')
        lines.append('')

        # Parameter info
        lines.append(f'int vae_param_count() {{ return {len(self.mod.params)}; }}')
        lines.append(f'const char* vae_param_name(int i) {{')
        lines.append(f'    static const char* names[] = {{')
        for p in self.mod.params:
            lines.append(f'        "{p.name}",')
        lines.append(f'    }};')
        lines.append(f'    return (i >= 0 && i < {len(self.mod.params)}) ? names[i] : "";')
        lines.append(f'}}')
        lines.append(f'double vae_param_default(int i) {{')
        lines.append(f'    static const double defaults[] = {{')
        for p in self.mod.params:
            lines.append(f'        {p.default},')
        lines.append(f'    }};')
        lines.append(f'    return (i >= 0 && i < {len(self.mod.params)}) ? defaults[i] : 0.0;')
        lines.append(f'}}')

        lines.append('')
        lines.append('} // extern "C"')

        return '\n'.join(lines) + '\n'


def generate(module: Module) -> str:
    """Generate C++ source from parsed Module."""
    return CodeGen(module).generate()
