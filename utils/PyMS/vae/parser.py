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
        return str(float(s[:-1]) * _SCALE[s[-1]])
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
        # Skip `include and `define directives
        while self.at('`include', '`define', '`ifdef', '`ifndef', '`else', '`endif'):
            self._skip_directive()

        # Parse optional attributes
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

        self.expect(';')

        # Module body — declarations and analog block
        while not self.at('endmodule'):
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
                mod.analog_block = self._parse_analog()
            elif self.at('(*'):
                # Attribute before parameter
                attrs = self._parse_attributes()
                if self.at('parameter'):
                    p = self._parse_param()
                    # Apply attributes
                    for k, v in attrs.items():
                        if k == 'desc':
                            p.desc = v
                        elif k == 'type' and v == 'instance':
                            p.is_instance = True
                    mod.params.append(p)
            else:
                # Skip unknown
                self.advance()

        self.expect('endmodule')
        return mod

    def _skip_directive(self):
        """Skip preprocessor directive to end of line."""
        start = self.peek().pos
        self.advance()  # `include/`define
        # Skip to next token that starts on a different line
        while self.pos < len(self.tokens):
            t = self.peek()
            if t.type == 'STRING' or t.type == 'IDENT' or t.type == 'NUMBER':
                self.advance()
            elif t.value in ('(', ')'):
                self.advance()
            elif t.type == 'OP' and t.value not in (';',):
                self.advance()
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

        default = "0"
        if self.at('='):
            self.advance()
            default = self._collect_expr_until(';', 'from', 'exclude')

        from_range = None
        if self.at('from'):
            self.advance()
            from_range = self._collect_expr_until(';')
        elif self.at('exclude'):
            self.advance()
            self._collect_expr_until(';')  # skip

        self.expect(';')

        p = Param(name=name, type=typ, default=parse_number(default), from_range=from_range)
        if attrs.get('type') == 'instance':
            p.is_instance = True
        if 'desc' in attrs:
            p.desc = attrs['desc']
        return p

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
        else:
            return self._parse_simple_statement()

    def _parse_begin_end(self) -> ASTNode:
        loc = self._loc()
        self.expect('begin')
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
        # Optional arguments
        while not self.at(')'):
            self.advance()
        self.expect(')')
        body = self._parse_statement()
        if event_name == 'initial_step':
            return ASTNode(kind=NodeKind.INITIAL_STEP, children=[body])
        return body  # Other events: just parse body for now

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

    def _collect_expr_until(self, *stops) -> str:
        """Collect tokens into expression string until one of stops is seen at depth 0."""
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
    return parser.parse_module()


def parse_file(path: str) -> Module:
    """Parse a Verilog-A file, return Module AST."""
    with open(path) as f:
        source = f.read()
    return parse_verilog_a(source, filename=path)
