"""
VAE GiNaC emitter — translates parsed Verilog-A AST into a GiNaC C++ program.

The emitted program:
  1. Declares GiNaC symbols for node voltages and parameters
  2. Mirrors the Verilog-A analog block structure using GiNaC expressions
  3. Differentiates contributions w.r.t. all node voltages
  4. Prints the final C++ eval and Jacobian code via print_csrc_double

Parameter-dependent conditionals are split into separate variants at
elaboration time (when parameter values are known). Each variant is a
complete, branch-free device implementation.
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
    'abs': 'abs',
    'pow': 'pow',
    'exp': 'exp',
    'log': 'log',
    'sqrt': 'sqrt',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'tanh': 'tanh',
}


# ---------------------------------------------------------------------------
# Variant: one code path with all parameter-dependent branches resolved
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    """One elaboration-time variant of the model."""
    label: str
    param_conditions: dict[str, object]  # param_name → value for condition eval
    nodes: list[str]                      # active node names (ports + active internals)
    branches: list[tuple[str, str]]       # (p, q) branch pairs
    contributions: list[dict]             # {branch_idx, expr_ginac, is_ddt, kind}
    assignments: list[dict]               # {lhs, expr_ginac}


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------

class GiNaCEmitter:
    """Emit a GiNaC C++ program from a parsed Verilog-A module."""

    def __init__(self, module: Module, line_directives: Optional[bool] = None):
        """
        Args:
            module: Parsed Verilog-A module AST.
            line_directives: None = off, True = active (#line), False = inactive (//line comment).
        """
        self.mod = module
        self.port_names = [p.name for p in module.ports]
        self.internal_nodes = list(module.internal_nodes)
        self.all_nodes = self.port_names + self.internal_nodes
        self.line_directives = line_directives

    def emit(self) -> str:
        """Generate the GiNaC C++ program source."""
        variants = self._extract_variants()
        return self._emit_ginac_program(variants)

    # --- Variant extraction ---

    def _extract_variants(self) -> list[dict]:
        """Identify parameter-dependent branches and enumerate variants."""
        param_conds = self._find_param_conditions(self.mod.analog_block)

        if not param_conds:
            # No parameter-dependent branches — single variant
            return [{'label': 'default', 'conditions': {}}]

        # Generate variant for each combination of parameter conditions
        variants = []
        self._enumerate_variants(param_conds, 0, {}, variants)
        return variants

    def _find_param_conditions(self, node: ASTNode) -> list[dict]:
        """Find if/else conditions that depend only on parameters."""
        if node is None:
            return []
        conds = []
        if node.kind == NodeKind.IF:
            if self._is_param_condition(node.condition):
                conds.append({
                    'condition': node.condition,
                    'true_value': True,
                    'false_value': False,
                })
            # Recurse into children
            for c in node.children:
                conds.extend(self._find_param_conditions(c))
            if node.else_body:
                conds.extend(self._find_param_conditions(node.else_body))
        elif node.kind == NodeKind.BLOCK:
            for c in node.children:
                conds.extend(self._find_param_conditions(c))
        return conds

    def _is_param_condition(self, cond: str) -> bool:
        """Check if condition depends only on parameters (not node voltages)."""
        if cond is None:
            return False
        param_names = {p.name for p in self.mod.params}
        # Tokenize condition and check all identifiers are parameters or constants
        idents = set(re.findall(r'[A-Za-z_]\w*', cond))
        # Remove known non-variable tokens
        idents -= {'if', 'else', 'begin', 'end'}
        return len(idents) > 0 and idents.issubset(param_names)

    def _enumerate_variants(self, conds, idx, current, results):
        if idx >= len(conds):
            label_parts = []
            for c in conds:
                val = current[c['condition']]
                label_parts.append(f"{c['condition']}={'true' if val else 'false'}")
            results.append({
                'label': ' && '.join(label_parts) if label_parts else 'default',
                'conditions': dict(current),
            })
            return
        c = conds[idx]
        for val in [True, False]:
            current[c['condition']] = val
            self._enumerate_variants(conds, idx + 1, current, results)

    # --- GiNaC C++ emission ---

    def _emit_ginac_program(self, variants: list[dict]) -> str:
        lines = []
        lines.append('// AUTO-GENERATED by VAE — GiNaC code generator')
        lines.append(f'// Source model: {self.mod.name}')
        lines.append('#include <ginac/ginac.h>')
        lines.append('#include <iostream>')
        lines.append('#include <sstream>')
        lines.append('#include <string>')
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
        lines.append('int main() {')

        # Declare symbols for all node voltages
        lines.append('    // Node voltage symbols')
        for node in self.all_nodes:
            lines.append(f'    symbol V_{node}("V_{node}");')

        # Declare symbols for parameters
        lines.append('    // Parameter symbols')
        for p in self.mod.params:
            lines.append(f'    symbol {p.name}("{p.name}");')

        # Special symbols
        lines.append('    symbol Vt("Vt");')
        lines.append('    symbol temperature("temperature");')
        lines.append('')
        # Preprocessor-defined constants as GiNaC numerics
        lines.append('    // Constants from `define directives')
        lines.append('    ex CONSTroot2 = numeric("1.41421356237309504880");')
        lines.append('    ex X_K = numeric("1.3806226e23");')
        lines.append('    ex CONSTboltz = numeric("1.3806226e-23");')
        lines.append('    ex CONSTq = numeric("1.602176634e-19");')
        lines.append('')

        # Emit each variant
        for vi, variant in enumerate(variants):
            lines.append(f'    // --- Variant {vi}: {variant["label"]} ---')
            lines.append(f'    cout << "// ======================================" << endl;')
            lines.append(f'    cout << "// Variant {vi}: {variant["label"]}" << endl;')
            lines.append(f'    cout << "// ======================================" << endl << endl;')
            lines.append('')
            self._emit_variant(lines, variant)
            lines.append('')

        lines.append('    return 0;')
        lines.append('}')
        return '\n'.join(lines) + '\n'

    def _emit_variant(self, lines: list[str], variant: dict):
        """Emit GiNaC code for one variant."""
        conds = variant['conditions']

        # Determine which nodes are active in this variant
        active_nodes = list(self.port_names)
        shorted = set()
        if self.internal_nodes:
            shorted = self._find_shorted_nodes(self.mod.analog_block, conds)
            for n in self.internal_nodes:
                if n not in shorted:
                    active_nodes.append(n)

        # Emit shorting annotations
        for n in sorted(shorted):
            target = self._find_short_target(self.mod.analog_block, n)
            if target:
                lines.append(f'    // SHORT: {n} → {target}  (V({n},{target}) <+ 0)')
                lines.append(f'    cout << "// SHORT: {n} shorted to {target}" << endl;')

        n_nodes = len(active_nodes)

        # Collect branches and expressions by walking the AST
        sym_list = ", ".join(f"V_{n}" for n in active_nodes)
        name_list = ", ".join('"' + n + '"' for n in active_nodes)
        lines.append(f'    {{ // variant scope')
        lines.append(f'        vector<symbol> nodes = {{{sym_list}}};')
        lines.append(f'        vector<string> node_names = {{{name_list}}};')
        lines.append(f'        int n_nodes = {n_nodes};')
        lines.append('')

        # Walk analog block, emitting GiNaC expressions
        # Collect contributions as we go
        lines.append(f'        // Analog block expressions')
        lines.append(f'        vector<ex> F_contribs;  // current contributions per branch')
        lines.append(f'        vector<ex> Q_contribs;  // charge contributions per branch')
        lines.append(f'        vector<string> branch_labels;')
        lines.append('')

        self._walk_analog_block(lines, self.mod.analog_block, conds,
                               active_nodes, indent=8)

        # Emit the C++ code printer
        lines.append('')
        lines.append(f'        int n_branches = F_contribs.size();')
        lines.append('')

        # Print eval function
        lines.append(f'        cout << "void vae_eval(VaeState* s, double* F, double* Q) {{" << endl;')
        for i, n in enumerate(active_nodes):
            lines.append(f'        cout << "    double V_{n} = s->V[{i}];" << endl;')
        for i, p in enumerate(self.mod.params):
            if not self._param_is_const_in_variant(p.name, conds):
                lines.append(f'        cout << "    double {p.name} = s->params[{i}];" << endl;')
        lines.append(f'        cout << "    double Vt = s->Vt;" << endl;')
        lines.append(f'        cout << endl;')

        lines.append(f'        for (int i = 0; i < n_branches; i++) {{')
        lines.append(f'            cout << "    F[" << i << "] = " << to_cpp(F_contribs[i]) << ";  // " << branch_labels[i] << endl;')
        lines.append(f'        }}')
        lines.append(f'        for (int i = 0; i < n_branches; i++) {{')
        lines.append(f'            if (!Q_contribs[i].is_zero())')
        lines.append(f'                cout << "    Q[" << i << "] = " << to_cpp(Q_contribs[i]) << ";" << endl;')
        lines.append(f'        }}')
        lines.append(f'        cout << "}}" << endl << endl;')

        # Print Jacobian function
        lines.append(f'        cout << "void vae_jacobian(VaeState* s, double* dFdV, double* dQdV) {{" << endl;')
        for i, n in enumerate(active_nodes):
            lines.append(f'        cout << "    double V_{n} = s->V[{i}];" << endl;')
        for i, p in enumerate(self.mod.params):
            if not self._param_is_const_in_variant(p.name, conds):
                lines.append(f'        cout << "    double {p.name} = s->params[{i}];" << endl;')
        lines.append(f'        cout << "    double Vt = s->Vt;" << endl;')
        lines.append(f'        cout << endl;')

        lines.append(f'        for (int br = 0; br < n_branches; br++) {{')
        lines.append(f'            for (int col = 0; col < n_nodes; col++) {{')
        lines.append(f'                ex dF = F_contribs[br].diff(nodes[col]);')
        lines.append(f'                if (!dF.is_zero())')
        lines.append(f'                    cout << "    dFdV[" << br << " * " << n_nodes << " + " << col')
        lines.append(f'                         << "] = " << to_cpp(dF) << ";  // dF[" << br << "]/d"')
        lines.append(f'                         << node_names[col] << endl;')
        lines.append(f'                ex dQ = Q_contribs[br].diff(nodes[col]);')
        lines.append(f'                if (!dQ.is_zero())')
        lines.append(f'                    cout << "    dQdV[" << br << " * " << n_nodes << " + " << col')
        lines.append(f'                         << "] = " << to_cpp(dQ) << ";  // dQ[" << br << "]/d"')
        lines.append(f'                         << node_names[col] << endl;')
        lines.append(f'            }}')
        lines.append(f'        }}')
        lines.append(f'        cout << "}}" << endl << endl;')

        # Print metadata
        lines.append(f'        cout << "// n_nodes = " << n_nodes << endl;')
        lines.append(f'        cout << "// n_branches = " << n_branches << endl;')
        lines.append(f'        cout << "// nodes:";')
        lines.append(f'        for (auto& n : node_names) cout << " " << n;')
        lines.append(f'        cout << endl;')

        lines.append(f'    }} // end variant scope')

    def _emit_line_directive(self, lines: list[str], node: ASTNode, pfx: str):
        """Emit a #line or //line directive into the generated output code.

        line_directives=True  → #line N "file"   (real preprocessor directive, GDB mapping)
        line_directives=False → //line N "file"   (comment, no debugger impact)
        line_directives=None  → nothing emitted
        """
        if self.line_directives is None or not node.loc or not node.loc.line:
            return
        prefix = '#' if self.line_directives else '//'
        filename = node.loc.file
        lineno = node.loc.line
        escaped = filename.replace('\\', '\\\\').replace('"', '\\"')
        # Emit a cout that prints the line directive into the generated code
        lines.append(f'{pfx}cout << "{prefix}line {lineno} \\"{escaped}\\"" << endl;')

    def _walk_analog_block(self, lines: list[str], node: ASTNode,
                          conds: dict, active_nodes: list[str], indent: int):
        """Walk AST and emit GiNaC expression-building code."""
        if node is None:
            return
        pfx = ' ' * indent

        if node.kind == NodeKind.BLOCK:
            for child in node.children:
                self._walk_analog_block(lines, child, conds, active_nodes, indent)

        elif node.kind == NodeKind.ASSIGN:
            self._emit_line_directive(lines, node, pfx)
            ginac_expr = self._to_ginac_expr(node.expr, active_nodes)
            lines.append(f'{pfx}ex {node.lhs} = {ginac_expr};')

        elif node.kind == NodeKind.CONTRIB:
            expr_str = node.expr.strip()
            is_ddt = False
            inner_expr = expr_str
            ddt_match = re.match(r'^ddt\s*\((.*)\)$', expr_str)
            if ddt_match:
                is_ddt = True
                inner_expr = ddt_match.group(1).strip()

            ginac_expr = self._to_ginac_expr(inner_expr, active_nodes)
            branch_label = f'{node.contrib_kind.name}({",".join(node.branch)})'

            # Find or create branch index
            br_p = node.branch[0]
            br_n = node.branch[1] if len(node.branch) > 1 else 'gnd'

            # Resolve shorted nodes to their active equivalents
            if br_p not in active_nodes and br_p in self.internal_nodes:
                br_p = self._resolve_shorted_node(br_p, active_nodes)
            if br_n != 'gnd' and br_n not in active_nodes and br_n in self.internal_nodes:
                br_n = self._resolve_shorted_node(br_n, active_nodes)

            # Skip tautological V contributions where both nodes merged (V(c,c) <+ 0)
            if node.contrib_kind == ContribKind.V and br_p == br_n:
                return

            self._emit_line_directive(lines, node, pfx)
            # Check if branch already exists
            lines.append(f'{pfx}{{ // contribution {branch_label}')
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
                # V contribution: F = expr - V_branch
                if br_n != 'gnd' and br_n in active_nodes:
                    lines.append(f'{pfx}    F_contribs[br_idx] += {ginac_expr} - (V_{br_p} - V_{br_n});')
                else:
                    lines.append(f'{pfx}    F_contribs[br_idx] += {ginac_expr} - V_{br_p};')
            else:
                lines.append(f'{pfx}    F_contribs[br_idx] += {ginac_expr};')
            lines.append(f'{pfx}}}')

        elif node.kind == NodeKind.IF:
            if self._is_param_condition(node.condition):
                # Parameter-dependent: resolve at elaboration time
                cond_val = conds.get(node.condition)
                if cond_val is True or cond_val is None:
                    # Take true branch (or both if not split)
                    for child in node.children:
                        self._walk_analog_block(lines, child, conds,
                                               active_nodes, indent)
                elif cond_val is False:
                    if node.else_body:
                        self._walk_analog_block(lines, node.else_body, conds,
                                               active_nodes, indent)
            else:
                # Runtime condition — emit as-is (not split)
                # This shouldn't happen for simple models
                lines.append(f'{pfx}// WARNING: runtime condition not split: {node.condition}')
                for child in node.children:
                    self._walk_analog_block(lines, child, conds,
                                           active_nodes, indent)

        elif node.kind == NodeKind.INITIAL_STEP:
            pass  # Skip initial_step for now

    def _to_ginac_expr(self, expr: str, active_nodes: list[str]) -> str:
        """Translate Verilog-A expression to GiNaC C++ expression."""
        if expr is None:
            return 'ex(0)'

        result = expr

        # V(a,b) → (V_a - V_b)
        def repl_v(m):
            p1 = m.group(1).strip()
            p2 = m.group(2).strip() if m.group(2) else None
            # If a node is shorted (not in active_nodes), substitute the node it's shorted to
            if p1 not in active_nodes and p1 in self.internal_nodes:
                # Internal node shorted to its port — find which port
                p1 = self._resolve_shorted_node(p1, active_nodes)
            if p2 and p2 not in active_nodes and p2 in self.internal_nodes:
                p2 = self._resolve_shorted_node(p2, active_nodes)
            if p2:
                return f'(V_{p1} - V_{p2})'
            return f'V_{p1}'
        result = re.sub(r'V\s*\(\s*(\w+)\s*(?:,\s*(\w+)\s*)?\)', repl_v, result)

        # $limit(expr, ...) → just expr
        result = re.sub(r'\$limit\s*\(\s*([^,]+),.*?\)', r'\1', result)

        # $vt → Vt
        result = re.sub(r'\$vt\b', 'Vt', result)
        result = re.sub(r'\$temperature\b', 'temperature', result)

        # Function mapping
        for va_name, ginac_name in _VA_TO_GINAC.items():
            result = re.sub(rf'\b{va_name}\b(?=\s*\()', ginac_name, result)

        # `DEFINE references
        result = re.sub(r'`(\w+)', r'\1', result)

        # Noise → 0
        result = re.sub(r'white_noise\s*\([^)]*\)', 'ex(0)', result)
        result = re.sub(r'flicker_noise\s*\([^)]*\)', 'ex(0)', result)

        # Numeric literals need ex() wrapping for GiNaC
        # But simple numbers in arithmetic work fine with GiNaC operator overloading

        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def _resolve_shorted_node(self, node: str, active_nodes: list[str]) -> str:
        """Find which active node a shorted internal node maps to.

        Determined by the V(x,y) <+ 0 constraint in the analog block.
        The shorted internal node is connected to the other node in the branch.
        """
        target = self._find_short_target(self.mod.analog_block, node)
        if target and target in active_nodes:
            return target
        # Fallback: last port
        for p in reversed(self.port_names):
            if p in active_nodes:
                return p
        return node

    def _find_short_target(self, ast_node: ASTNode, internal: str) -> Optional[str]:
        """Find which node an internal node is shorted to via V(x,y) <+ 0."""
        if ast_node is None:
            return None
        if ast_node.kind == NodeKind.CONTRIB and ast_node.contrib_kind == ContribKind.V:
            expr = ast_node.expr.strip()
            if expr in ('0', '0.0'):
                branch = ast_node.branch
                if internal in branch:
                    # Return the other node in the branch
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

    def _find_shorted_nodes(self, node: ASTNode, conds: dict) -> set[str]:
        """Find internal nodes shorted by V(x,y) <+ 0 in this variant."""
        shorted = set()
        if node is None:
            return shorted
        if node.kind == NodeKind.BLOCK:
            for child in node.children:
                shorted |= self._find_shorted_nodes(child, conds)
        elif node.kind == NodeKind.CONTRIB:
            if node.contrib_kind == ContribKind.V:
                # V(ci,c) <+ 0 → ci shorted to c
                expr = node.expr.strip()
                if expr == '0' or expr == '0.0':
                    for b in node.branch:
                        if b in self.internal_nodes:
                            shorted.add(b)
        elif node.kind == NodeKind.IF:
            if self._is_param_condition(node.condition):
                cond_val = conds.get(node.condition)
                if cond_val is True:
                    for child in node.children:
                        shorted |= self._find_shorted_nodes(child, conds)
                elif cond_val is False and node.else_body:
                    shorted |= self._find_shorted_nodes(node.else_body, conds)
            else:
                for child in node.children:
                    shorted |= self._find_shorted_nodes(child, conds)
                if node.else_body:
                    shorted |= self._find_shorted_nodes(node.else_body, conds)
        return shorted

    def _param_is_const_in_variant(self, param_name: str, conds: dict) -> bool:
        """Check if parameter is used only in resolved conditions (not in expressions)."""
        # For now, assume all parameters are needed
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def emit_ginac_program(module: Module, line_directives: Optional[bool] = None) -> str:
    """Generate GiNaC C++ source from a parsed Verilog-A module.

    Args:
        module: Parsed Verilog-A module.
        line_directives: None=off, True=#line (active), False=//line (inactive comment).
    """
    return GiNaCEmitter(module, line_directives=line_directives).emit()
