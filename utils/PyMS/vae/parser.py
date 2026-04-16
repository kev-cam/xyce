"""
VAE Verilog-A parser — parses module declarations and analog blocks.

Parses enough of Verilog-A to extract:
  - Module name, ports, port directions, disciplines
  - Parameter declarations with defaults, ranges, attributes
  - Variable declarations
  - analog begin...end block as an AST

Does NOT parse: discipline/nature definitions, `include files.
Those are provided by the elaboration context.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

class NodeKind(Enum):
    MODULE = auto()
    PARAM = auto()
    VAR = auto()
    BLOCK = auto()            # begin...end
    CONTRIB = auto()          # I(a,b) <+ expr  or  V(a,b) <+ expr
    IF = auto()
    ASSIGN = auto()           # variable = expr
    CALL = auto()             # function call (standalone)
    INITIAL_STEP = auto()     # @(initial_step) begin...end
    EXPR = auto()             # raw expression string (leaf)


class PortDir(Enum):
    IN = auto()
    OUT = auto()
    INOUT = auto()


class ContribKind(Enum):
    I = auto()   # current contribution
    V = auto()   # voltage contribution
    Q = auto()   # charge contribution (via ddt)


@dataclass
class Port:
    name: str
    direction: PortDir = PortDir.INOUT
    discipline: str = "electrical"


@dataclass
class Param:
    name: str
    default: str = "0"
    type: str = "real"
    from_range: Optional[str] = None    # e.g. "(0:inf)"
    desc: Optional[str] = None
    is_instance: bool = False           # type="instance" attribute
    # Array-parameter support: when declared as `parameter real T[0:N-1] = '{...}`
    # array_size is N and elements is the parsed list of initialiser values.
    array_size: Optional[int] = None
    elements: Optional[list[str]] = None


@dataclass
class Var:
    name: str
    type: str = "real"


@dataclass
class SourceLoc:
    file: str = ""
    line: int = 0

    def __str__(self):
        return f"{self.file}:{self.line}" if self.file else f":{self.line}"


@dataclass
class ASTNode:
    kind: NodeKind
    text: str = ""
    children: list[ASTNode] = field(default_factory=list)
    # For CONTRIB:
    contrib_kind: Optional[ContribKind] = None
    branch: Optional[tuple[str, ...]] = None   # (p,) or (p, q)
    expr: Optional[str] = None
    # For IF:
    condition: Optional[str] = None
    else_body: Optional[ASTNode] = None
    # For ASSIGN:
    lhs: Optional[str] = None
    # Source location
    loc: Optional[SourceLoc] = None


@dataclass
class Module:
    name: str
    ports: list[Port] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    variables: list[Var] = field(default_factory=list)
    internal_nodes: list[str] = field(default_factory=list)
    branch_map: dict[str, str] = field(default_factory=dict)  # branch_name → node_name
    analog_block: Optional[ASTNode] = None
    attributes: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Token patterns — order matters (longer matches first)
_TOKEN_PATTERNS = [
    ('COMMENT_LINE', r'//[^\n]*'),
    ('COMMENT_BLOCK', r'/\*.*?\*/'),
    ('STRING', r'"[^"]*"'),
    ('ATTR_OPEN', r'\(\*'),
    ('ATTR_CLOSE', r'\*\)'),
    ('CONTRIB', r'<\+'),
    ('NUMBER', r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?:[fpnumkMGT])?'),
    ('IDENT', r'[A-Za-z_\$`][A-Za-z0-9_\$]*'),
    ('OP', r'[+\-*/(){},;:=<>!&|~\[\]@^?]'),
    ('WS', r'\s+'),
]

_TOKEN_RE = re.compile('|'.join(f'(?P<{n}>{p})' for n, p in _TOKEN_PATTERNS), re.DOTALL)


@dataclass
class Token:
    type: str
    value: str
    pos: int = 0


def tokenize(source: str) -> list[Token]:
    tokens = []
    for m in _TOKEN_RE.finditer(source):
        typ = m.lastgroup
        if typ in ('WS', 'COMMENT_LINE', 'COMMENT_BLOCK'):
            continue
        tokens.append(Token(typ, m.group(), m.start()))
    return tokens


# ---------------------------------------------------------------------------
# Scale factors
# ---------------------------------------------------------------------------

_SCALE = {
    'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6,
    'm': 1e-3, 'k': 1e3, 'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12,
}


def parse_number(s: str) -> str:
    """Convert Verilog-A number with scale suffix to numeric string."""
    if s and s[-1] in _SCALE:
        try:
            return str(float(s[:-1]) * _SCALE[s[-1]])
        except ValueError:
            return s
    return s


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: list[Token], source: str = "", filename: str = ""):
        self.tokens = tokens
        self.pos = 0
        self.filename = filename
        # Build line offset table for pos→line mapping
        self._line_offsets = [0]
        for i, ch in enumerate(source):
            if ch == '\n':
                self._line_offsets.append(i + 1)

    def _loc(self) -> SourceLoc:
        """Get source location of current token."""
        t = self.peek()
        if t is None:
            return SourceLoc(self.filename, 0)
        return self._loc_at(t.pos)

    def _loc_at(self, char_pos: int) -> SourceLoc:
        """Convert character position to SourceLoc."""
        # Binary search for line number
        lo, hi = 0, len(self._line_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._line_offsets[mid] <= char_pos:
                lo = mid
            else:
                hi = mid - 1
        return SourceLoc(self.filename, lo + 1)  # 1-based line numbers

    def peek(self, offset=0) -> Optional[Token]:
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else None

    def at(self, *values) -> bool:
        t = self.peek()
        return t is not None and t.value in values

    def at_type(self, typ: str) -> bool:
        t = self.peek()
        return t is not None and t.type == typ

    def expect(self, value: str) -> Token:
        t = self.peek()
        if t is None or t.value != value:
            got = t.value if t else 'EOF'
            raise ParseError(f"Expected '{value}', got '{got}' at pos {t.pos if t else 'EOF'}")
        self.pos += 1
        return t

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def consume_ident(self) -> str:
        t = self.peek()
        if t is None or t.type != 'IDENT':
            got = t.value if t else 'EOF'
            raise ParseError(f"Expected identifier, got '{got}'")
        self.pos += 1
        return t.value

    # --- Module-level parsing ---

    def parse_module(self) -> Module:
        # Skip preprocessor directives and nature/discipline blocks before module
        self._skip_preamble()

        # No module found (e.g. include-only / macro-definition file)
        if self.pos >= len(self.tokens):
            raise ParseError("No module found in file (macro/include-only file?)")

        # Parse optional attributes (* ... *)
        attrs = {}
        if self.at('(*'):
            attrs = self._parse_attributes()

        self.expect('module')
        name = self.consume_ident()

        mod = Module(name=name, attributes=attrs)

        # Port list
        if self.at('('):
            self.advance()  # (
            port_names = []
            while not self.at(')'):
                port_names.append(self.consume_ident())
                if self.at(','):
                    self.advance()
            self.expect(')')
            for pn in port_names:
                mod.ports.append(Port(name=pn))

        # Skip `attr(...) or (* ... *) after port list, before ;
        self._skip_inline_attrs()

        self.expect(';')

        # Module body — declarations and analog block
        while not self.at('endmodule'):
            t = self.peek()
            if t is None:
                break
            if self.at('inout'):
                self._parse_port_dir(mod, PortDir.INOUT)
            elif self.at('input'):
                self._parse_port_dir(mod, PortDir.IN)
            elif self.at('output'):
                self._parse_port_dir(mod, PortDir.OUT)
            elif self.at('electrical'):
                self._parse_electrical(mod)
            elif self.at('parameter'):
                mod.params.append(self._parse_param())
            elif self.at('real') or self.at('integer'):
                self._parse_variables(mod)
            elif self.at('analog'):
                t2 = self.peek(1)
                if t2 and t2.value == 'function':
                    self._skip_analog_function()
                else:
                    mod.analog_block = self._parse_analog()
            elif self.at('(*'):
                # Attribute before parameter or other declaration
                attrs = self._parse_attributes()
                if self.at('parameter'):
                    p = self._parse_param()
                    for k, v in attrs.items():
                        if k == 'desc':
                            p.desc = v
                        elif k == 'type' and v == 'instance':
                            p.is_instance = True
                    mod.params.append(p)
            elif self.at('branch'):
                # branch (node) name ; → record name→node mapping
                self.advance()  # 'branch'
                if self.at('('):
                    self.advance()
                    node_name = self.consume_ident()
                    self.expect(')')
                    branch_name = self.consume_ident()
                    mod.branch_map[branch_name] = node_name
                    self.expect(';')
                else:
                    self._skip_to_semi()
            elif self.at('genvar'):
                self._skip_to_semi()
            elif t.value.startswith('`'):
                self._skip_directive()
            else:
                # Skip unknown token
                self.advance()

        self.expect('endmodule')
        return mod

    def _skip_preamble(self):
        """Skip preprocessor directives, nature/discipline blocks before module."""
        while self.pos < len(self.tokens):
            t = self.peek()
            if t is None:
                break
            if t.value.startswith('`'):
                self._skip_directive()
            elif t.value in ('nature', 'discipline'):
                self._skip_block(t.value, 'end' + t.value)
            elif t.value == 'module':
                break
            elif t.value == '(*':
                # Could be attribute on module — check if 'module' follows
                break
            else:
                # Skip unknown preamble token
                self.advance()

    def _skip_directive(self):
        """Skip preprocessor directive — consume tokens on the same line."""
        start_loc = self._loc()
        t = self.peek()
        directive = t.value if t else ''
        self.advance()  # `include/`define/etc.

        if directive in ('`ifdef', '`ifndef'):
            # Skip condition identifier on same line
            while self.pos < len(self.tokens):
                tok = self.peek()
                tok_loc = self._loc_at(tok.pos)
                if tok_loc.line != start_loc.line:
                    break
                self.advance()
            return

        # For `else, `endif — just the directive token, already consumed
        if directive in ('`else', '`endif', '`undef'):
            return

        # For `define — may have multi-line continuation with backslash,
        # but our tokenizer strips comments. Just consume same line.
        # For `include and others — consume same line.
        while self.pos < len(self.tokens):
            t = self.peek()
            tok_loc = self._loc_at(t.pos)
            if tok_loc.line != start_loc.line:
                break
            self.advance()

    def _skip_block(self, start_kw: str, end_kw: str):
        """Skip a block from start_kw to end_kw (e.g. nature/endnature)."""
        self.advance()  # consume start keyword
        while self.pos < len(self.tokens):
            t = self.peek()
            if t.value == end_kw:
                self.advance()
                # consume optional ;
                if self.at(';'):
                    self.advance()
                return
            self.advance()

    def _skip_to_semi(self):
        """Skip tokens until and including the next semicolon."""
        while self.pos < len(self.tokens):
            t = self.advance()
            if t.value == ';':
                return

    def _skip_analog_function(self):
        """Skip 'analog function ... endfunction'."""
        self.advance()  # analog
        self.advance()  # function
        while self.pos < len(self.tokens):
            t = self.advance()
            if t.value == 'endfunction':
                return

    def _skip_inline_attrs(self):
        """Skip inline `attr(...) or (* ... *) between tokens."""
        while self.pos < len(self.tokens):
            t = self.peek()
            if t.value == '(*':
                self._parse_attributes()  # consume and discard
            elif t.value.startswith('`'):
                t2 = self.peek(1)
                if t2 and t2.value == '(':
                    # `attr(...) or similar macro call — skip to matching )
                    self.advance()  # `attr
                    self.advance()  # (
                    depth = 1
                    while self.pos < len(self.tokens) and depth > 0:
                        tt = self.advance()
                        if tt.value == '(':
                            depth += 1
                        elif tt.value == ')':
                            depth -= 1
                else:
                    break  # standalone `macro — not an attr
            else:
                break

    def _parse_attributes(self) -> dict[str, str]:
        """Parse (* key=value, key=value *) attributes."""
        attrs = {}
        self.expect('(*')
        while not self.at('*)'):
            key = self.consume_ident()
            if self.at('='):
                self.advance()
                t = self.peek()
                if t.type == 'STRING':
                    attrs[key] = t.value.strip('"')
                    self.advance()
                else:
                    attrs[key] = self.advance().value
            else:
                attrs[key] = "true"
            if self.at(','):
                self.advance()
        self.expect('*)')
        return attrs

    def _parse_port_dir(self, mod: Module, direction: PortDir):
        self.advance()  # inout/input/output
        names = []
        while not self.at(';'):
            if self.at_type('IDENT'):
                names.append(self.consume_ident())
            elif self.at(','):
                self.advance()
            else:
                self.advance()
        self.expect(';')
        for name in names:
            for p in mod.ports:
                if p.name == name:
                    p.direction = direction

    def _parse_electrical(self, mod: Module):
        self.advance()  # electrical
        names = []
        while not self.at(';'):
            if self.at_type('IDENT'):
                names.append(self.consume_ident())
            elif self.at(','):
                self.advance()
            else:
                self.advance()
        self.expect(';')
        # Mark any non-port names as internal nodes
        port_names = {p.name for p in mod.ports}
        for name in names:
            if name not in port_names:
                mod.internal_nodes.append(name)

    def _parse_param(self) -> Param:
        # Handle optional attributes before 'parameter'
        attrs = {}
        if self.at('(*'):
            attrs = self._parse_attributes()

        self.expect('parameter')
        typ = self.consume_ident()  # real or integer
        name = self.consume_ident()

        # Optional array bounds: `[lo:hi]` immediately after the name.
        array_size = None
        if self.at('['):
            self.advance()
            lo_str = self._collect_param_value_until(':', ']')
            if self.at(':'):
                self.advance()
            hi_str = self._collect_param_value_until(']')
            self.expect(']')
            try:
                lo = int(float(parse_number(lo_str.strip())))
                hi = int(float(parse_number(hi_str.strip())))
                array_size = hi - lo + 1
            except ValueError:
                array_size = 0

        default = "0"
        elements = None
        if self.at('='):
            self.advance()
            if array_size is not None and self._at_array_init():
                elements = self._parse_array_init()
            else:
                default = self._collect_param_value()

        from_range = None
        while self.at('from') or self.at('exclude'):
            if self.at('from'):
                self.advance()
                from_range = self._collect_param_value()
            elif self.at('exclude'):
                self.advance()
                self._collect_param_value()  # discard

        # Skip inline `attr(...) or (* ... *) before ;
        self._skip_inline_attrs()

        self.expect(';')

        p = Param(name=name, type=typ, default=parse_number(default), from_range=from_range)
        if array_size is not None:
            p.array_size = array_size
            p.elements = elements or []
        if attrs.get('type') == 'instance':
            p.is_instance = True
        if 'desc' in attrs:
            p.desc = attrs['desc']
        return p

    def _at_array_init(self) -> bool:
        """Peek for the start of `'{` or `{` aggregate initialiser."""
        t = self.peek()
        if t is None:
            return False
        if t.value == "'":
            t2 = self.peek(1)
            return t2 is not None and t2.value == '{'
        return t.value == '{'

    def _parse_array_init(self) -> list:
        """Parse `'{v0, v1, ...}` or `{v0, v1, ...}` aggregate, return list of
        string values (preserving literal form for codegen)."""
        # Consume optional `'`
        if self.peek() and self.peek().value == "'":
            self.advance()
        self.expect('{')
        elements = []
        while not self.at('}'):
            # Collect one element (may be signed number or expression);
            # stops at top-level `,` or `}`.
            val = self._collect_array_element()
            if val.strip():
                elements.append(val.strip())
            if self.at(','):
                self.advance()
        self.expect('}')
        return elements

    def _collect_array_element(self) -> str:
        """Collect tokens until a top-level `,` or `}`."""
        parts = []
        depth = 0
        while self.pos < len(self.tokens):
            t = self.peek()
            if depth == 0 and t.value in (',', '}'):
                break
            if t.value in ('(', '[', '{'):
                depth += 1
            elif t.value in (')', ']', '}'):
                depth -= 1
            parts.append(t.value)
            self.advance()
        return ' '.join(parts)

    def _collect_param_value_until(self, *stops) -> str:
        """Like _collect_param_value but stops on arbitrary given tokens at depth 0.
        Used for parsing `[lo:hi]` array bounds."""
        parts = []
        depth = 0
        while self.pos < len(self.tokens):
            t = self.peek()
            if depth == 0 and t.value in stops:
                break
            if t.value in ('(', '['):
                depth += 1
            elif t.value in (')', ']'):
                depth -= 1
            parts.append(t.value)
            self.advance()
        return ' '.join(parts)

    def _parse_variables(self, mod: Module):
        typ = self.advance().value  # real or integer
        while not self.at(';'):
            if self.at_type('IDENT'):
                mod.variables.append(Var(name=self.consume_ident(), type=typ))
            elif self.at(','):
                self.advance()
            else:
                self.advance()
        self.expect(';')

    # --- Analog block parsing ---

    def _parse_analog(self) -> ASTNode:
        self.expect('analog')
        return self._parse_statement()

    def _parse_statement(self) -> ASTNode:
        if self.at('begin'):
            return self._parse_begin_end()
        elif self.at('if'):
            return self._parse_if()
        elif self.at('@'):
            return self._parse_event()
        elif self.at('case'):
            return self._parse_case()
        elif self.at('for'):
            return self._parse_for()
        elif self.at('while'):
            return self._parse_while()
        else:
            return self._parse_simple_statement()

    def _parse_begin_end(self) -> ASTNode:
        loc = self._loc()
        self.expect('begin')
        # Optional named block: begin : label
        if self.at(':'):
            self.advance()
            self.consume_ident()  # block label
        block = ASTNode(kind=NodeKind.BLOCK, loc=loc)
        while not self.at('end'):
            block.children.append(self._parse_statement())
        self.expect('end')
        return block

    def _parse_if(self) -> ASTNode:
        loc = self._loc()
        self.expect('if')
        self.expect('(')
        cond = self._collect_balanced('(', ')')
        self.expect(')')

        body = self._parse_statement()

        else_body = None
        if self.at('else'):
            self.advance()
            else_body = self._parse_statement()

        return ASTNode(kind=NodeKind.IF, condition=cond,
                       children=[body], else_body=else_body, loc=loc)

    def _parse_event(self) -> ASTNode:
        self.expect('@')
        self.expect('(')
        event_name = self.consume_ident()
        # Optional arguments — may themselves contain nested parens (e.g.
        # @(timer(dt, dt)) or @(cross(V(a,b)-vt, +1))). Track depth so we
        # stop on the `)` that matches the opening `(` we consumed above.
        depth = 0
        while self.pos < len(self.tokens):
            t = self.peek()
            if t is None:
                break
            if t.value == '(':
                depth += 1
                self.advance()
            elif t.value == ')':
                if depth == 0:
                    break
                depth -= 1
                self.advance()
            else:
                self.advance()
        self.expect(')')
        body = self._parse_statement()
        if event_name == 'initial_step':
            return ASTNode(kind=NodeKind.INITIAL_STEP, children=[body])
        return body  # Other events: just parse body for now

    def _parse_case(self) -> ASTNode:
        """Parse case(expr) ... endcase as a sequence of if/else branches."""
        loc = self._loc()
        self.expect('case')
        self.expect('(')
        case_expr = self._collect_balanced('(', ')')
        self.expect(')')

        # Collect case items: each is value[, value]: statement(s)
        # Treat as if/else chain for the emitter
        block = ASTNode(kind=NodeKind.BLOCK, loc=loc)
        while not self.at('endcase'):
            if self.at('default'):
                self.advance()
                self.expect(':')
                while not self.at('endcase') and not self._is_case_label():
                    block.children.append(self._parse_statement())
            else:
                # case value(s)
                values = []
                while not self.at(':'):
                    if self.at(','):
                        self.advance()
                    else:
                        values.append(self.advance().value)
                self.expect(':')
                # Build condition: (case_expr == v1 || case_expr == v2 ...)
                cond_parts = [f'{case_expr} == {v}' for v in values]
                cond = ' || '.join(cond_parts)
                body = ASTNode(kind=NodeKind.BLOCK, loc=self._loc())
                while not self.at('endcase') and not self._is_case_label():
                    body.children.append(self._parse_statement())
                ifnode = ASTNode(kind=NodeKind.IF, condition=cond,
                                children=[body], loc=loc)
                block.children.append(ifnode)
        self.expect('endcase')
        return block

    def _is_case_label(self) -> bool:
        """Check if current position is a case label (value :) or 'default'."""
        if self.at('default'):
            return True
        # Look ahead for pattern: value [, value] :
        save = self.pos
        try:
            depth = 0
            while self.pos < len(self.tokens):
                t = self.peek()
                if t is None:
                    break
                if t.value == ':' and depth == 0:
                    self.pos = save
                    return True
                if t.value in (';', 'begin', 'end', 'if', 'case', 'endcase'):
                    break
                if t.value in ('(', '['):
                    depth += 1
                elif t.value in (')', ']'):
                    depth -= 1
                self.pos += 1
        finally:
            self.pos = save
        return False

    def _parse_for(self) -> ASTNode:
        """Parse for(init; cond; step) body — treat body as block."""
        loc = self._loc()
        self.expect('for')
        self.expect('(')
        # Skip init; cond; step
        depth = 0
        while self.pos < len(self.tokens):
            t = self.peek()
            if t.value == '(' :
                depth += 1
            elif t.value == ')':
                if depth == 0:
                    break
                depth -= 1
            self.advance()
        self.expect(')')
        body = self._parse_statement()
        return body  # Just include the body; loop unrolling not needed for symbolic

    def _parse_while(self) -> ASTNode:
        """Parse while(cond) body — treat body as block."""
        loc = self._loc()
        self.expect('while')
        self.expect('(')
        depth = 0
        while self.pos < len(self.tokens):
            t = self.peek()
            if t.value == '(':
                depth += 1
            elif t.value == ')':
                if depth == 0:
                    break
                depth -= 1
            self.advance()
        self.expect(')')
        body = self._parse_statement()
        return body

    def _parse_simple_statement(self) -> ASTNode:
        # Look ahead to determine statement type
        # Could be: contribution (I/V <+), assignment (var = expr), or call

        # Check for contribution: I(a,b) <+ or V(a,b) <+
        t = self.peek()
        if t and t.value in ('I', 'V') and self.peek(1) and self.peek(1).value == '(':
            return self._parse_contribution()

        # Assignment: ident = expr ;
        if t and t.type == 'IDENT':
            t2 = self.peek(1)
            if t2 and t2.value == '=':
                loc = self._loc()
                lhs = self.consume_ident()
                self.expect('=')
                expr = self._collect_expr_until(';')
                self.expect(';')
                return ASTNode(kind=NodeKind.ASSIGN, lhs=lhs, expr=expr, loc=loc)

        # Fallback — collect to semicolon
        text = self._collect_expr_until(';')
        self.expect(';')
        return ASTNode(kind=NodeKind.EXPR, text=text)

    def _parse_contribution(self) -> ASTNode:
        loc = self._loc()
        kind_str = self.advance().value  # I or V
        ckind = ContribKind.I if kind_str == 'I' else ContribKind.V

        self.expect('(')
        branch_parts = []
        while not self.at(')'):
            if self.at(','):
                self.advance()
            elif self.at_type('IDENT'):
                branch_parts.append(self.consume_ident())
            else:
                self.advance()
        self.expect(')')

        self.expect('<+')

        expr = self._collect_expr_until(';')
        self.expect(';')

        return ASTNode(
            kind=NodeKind.CONTRIB,
            contrib_kind=ckind,
            branch=tuple(branch_parts),
            expr=expr,
            loc=loc,
        )

    # --- Expression collectors ---

    def _collect_param_value(self) -> str:
        """Collect parameter default or range value, stopping at ;, from, exclude, or `attr(...)."""
        parts = []
        depth = 0
        while self.pos < len(self.tokens):
            t = self.peek()
            if depth == 0:
                if t.value in (';', 'from', 'exclude'):
                    break
                if t.value == '(*':
                    break
                # Stop at `macro(...) attribute annotations (e.g. `attr(...))
                # but NOT value macros like `NMOS, `P_Q etc.
                if t.value.startswith('`'):
                    t2 = self.peek(1)
                    if t2 and t2.value == '(':
                        break  # `attr(...) style
            if t.value in ('(', '['):
                depth += 1
            elif t.value in (')', ']'):
                depth -= 1
            parts.append(t.value)
            self.advance()
        return ' '.join(parts)

    def _collect_expr_until(self, *stops) -> str:
        """Collect tokens into expression string until one of stops is seen at depth 0.

        Depth is clamped at ≥0 so a stray closing bracket (from an earlier
        mis-nested parse) doesn't let the loop swallow the rest of the file
        by making every subsequent `;` appear to be at non-zero depth.
        """
        parts = []
        depth = 0
        while self.pos < len(self.tokens):
            t = self.peek()
            if depth == 0 and t.value in stops:
                break
            if t.value in ('(', '['):
                depth += 1
            elif t.value in (')', ']'):
                if depth > 0:
                    depth -= 1
            parts.append(t.value)
            self.advance()
        return ' '.join(parts)

    def _collect_balanced(self, open_ch: str, close_ch: str) -> str:
        """Collect tokens between balanced open/close, not including them."""
        parts = []
        depth = 0
        while self.pos < len(self.tokens):
            t = self.peek()
            if t.value == open_ch:
                depth += 1
                if depth > 0:
                    parts.append(t.value)
                self.advance()
            elif t.value == close_ch:
                if depth <= 0:
                    break
                depth -= 1
                parts.append(t.value)
                self.advance()
            else:
                parts.append(t.value)
                self.advance()
        return ' '.join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_verilog_a(source: str, filename: str = "") -> Module:
    """Parse a Verilog-A source string, return Module AST."""
    tokens = tokenize(source)
    parser = Parser(tokens, source=source, filename=filename)
    mod = parser.parse_module()
    if mod.analog_block:
        mod.analog_block = _lower_ternaries(mod.analog_block)
    return mod


def parse_file(path: str) -> Module:
    """Parse a Verilog-A file, return Module AST."""
    with open(path) as f:
        source = f.read()
    return parse_verilog_a(source, filename=path)


# ---------------------------------------------------------------------------
# Ternary lowering — convert ?: in expressions to if/else + temp vars
# ---------------------------------------------------------------------------

_ternary_counter = [0]


def _find_ternary(s: str):
    """Find the innermost paren-balanced ternary in s.

    Returns (start, end, cond, true_expr, false_expr) or None.
    Finds innermost first so nested ternaries lower from inside out.
    Handles both parenthesized `(cond ? a : b)` and bare `cond ? a : b`.
    """
    best = None
    # First: look for parenthesized ternaries (innermost)
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
            end = i + 1
            cond = s[start + 1:q_pos].strip()
            true_expr = s[q_pos + 1:colon_pos].strip()
            false_expr = s[colon_pos + 1:i].strip()
            if best is None or (end - start) < (best[1] - best[0]):
                best = (start, end, cond, true_expr, false_expr)
        idx = start + 1
    if best:
        return best

    # Second: look for bare top-level ternary (no enclosing parens)
    # Scan for ? at depth 0
    depth = 0
    q_pos = -1
    colon_pos = -1
    for i, ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '?' and depth == 0 and q_pos < 0:
            q_pos = i
        elif ch == ':' and depth == 0 and q_pos >= 0 and colon_pos < 0:
            colon_pos = i
    if q_pos >= 0 and colon_pos > q_pos:
        cond = s[:q_pos].strip()
        true_expr = s[q_pos + 1:colon_pos].strip()
        false_expr = s[colon_pos + 1:].strip()
        return (0, len(s), cond, true_expr, false_expr)

    return None


def _lower_expr_ternaries(expr: str, loc):
    """Lower all ?: in an expression to IF/ASSIGN nodes + temp vars.

    Returns (new_expr, [preceding_IF_nodes]).
    """
    preceding = []
    s = expr
    while True:
        t = _find_ternary(s)
        if t is None:
            break
        start, end, cond, true_e, false_e = t
        # Generate temp var
        _ternary_counter[0] += 1
        tmp = f'_vae_t{_ternary_counter[0]}'
        # Recursively lower ternaries in sub-expressions
        true_e, true_pre = _lower_expr_ternaries(true_e, loc)
        false_e, false_pre = _lower_expr_ternaries(false_e, loc)
        preceding.extend(true_pre)
        preceding.extend(false_pre)
        # Create IF node: if (cond) tmp = true_e; else tmp = false_e;
        true_assign = ASTNode(kind=NodeKind.ASSIGN, lhs=tmp,
                              expr=true_e, loc=loc)
        false_assign = ASTNode(kind=NodeKind.ASSIGN, lhs=tmp,
                               expr=false_e, loc=loc)
        if_node = ASTNode(
            kind=NodeKind.IF, condition=cond,
            children=[true_assign],
            else_body=false_assign, loc=loc,
        )
        preceding.append(if_node)
        # Replace ternary in expression with temp var
        s = s[:start] + tmp + s[end:]
    return s, preceding


def _lower_ternaries(node: ASTNode) -> ASTNode:
    """Walk AST, lowering ?: in ASSIGN/CONTRIB expressions to IF+temp vars."""
    if node is None:
        return None

    if node.kind == NodeKind.BLOCK:
        new_children = []
        for child in node.children:
            lowered = _lower_ternaries(child)
            if lowered is not None:
                new_children.append(lowered)
        node.children = new_children
        return node

    elif node.kind == NodeKind.ASSIGN:
        if node.expr and '?' in node.expr:
            new_expr, preceding = _lower_expr_ternaries(node.expr, node.loc)
            if preceding:
                # Wrap in a block: [IF nodes..., modified ASSIGN]
                node.expr = new_expr
                block = ASTNode(kind=NodeKind.BLOCK, loc=node.loc)
                block.children = preceding + [node]
                return block
        return node

    elif node.kind == NodeKind.CONTRIB:
        if node.expr and '?' in node.expr:
            new_expr, preceding = _lower_expr_ternaries(node.expr, node.loc)
            if preceding:
                node.expr = new_expr
                block = ASTNode(kind=NodeKind.BLOCK, loc=node.loc)
                block.children = preceding + [node]
                return block
        return node

    elif node.kind == NodeKind.IF:
        node.children = [_lower_ternaries(c) for c in node.children
                         if c is not None]
        if node.else_body:
            node.else_body = _lower_ternaries(node.else_body)
        # Also check condition for ternaries — rare but possible
        if node.condition and '?' in node.condition:
            new_cond, preceding = _lower_expr_ternaries(node.condition, node.loc)
            if preceding:
                node.condition = new_cond
                block = ASTNode(kind=NodeKind.BLOCK, loc=node.loc)
                block.children = preceding + [node]
                return block
        return node

    elif node.kind == NodeKind.INITIAL_STEP:
        node.children = [_lower_ternaries(c) for c in node.children
                         if c is not None]
        return node

    return node
