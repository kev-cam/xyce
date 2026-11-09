"""
VAE Verilog-A parser — parses module declarations and analog blocks.

Parses enough of Verilog-A to extract:
  - Module name, ports, port directions, disciplines
  - Parameter declarations with defaults, ranges, attributes
  - Variable declarations
  - analog begin...end block as an AST

Does NOT parse `include files (the preprocessor does).  nature and
discipline definitions are read only for their access functions and
the potential/flow binding (LRM_DISCIPLINES has the standard ones).
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
    # An event-controlled statement @(...) statement: initial_step,
    # final_step, cross, above (node.event lists the events, see
    # Parser._parse_event; GiNaCEmitter decides per analysis where it runs).
    INITIAL_STEP = auto()
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


# The continuous disciplines of the Verilog-AMS LRM's disciplines.vams with
# the access functions of their potential and flow natures.  To MNA a node
# of any of them is a node like an electrical one: its potential access
# reads the node's unknown (as V does) and its flow access contributes to
# the node's KCL row (as I does), so the parser maps every access function
# to V or I (Parser._expr_token, _parse_contribution) and the rest of PyMS
# sees electrical nodes only.  Natures and disciplines declared in the
# source extend the table (Parser._parse_nature, _parse_discipline).
LRM_DISCIPLINES: dict[str, tuple[Optional[str], Optional[str]]] = {
    'electrical':       ('V', 'I'),
    'voltage':          ('V', None),
    'current':          (None, 'I'),
    'magnetic':         ('MMF', 'Phi'),
    'thermal':          ('Temp', 'Pwr'),
    'kinematic':        ('Pos', 'F'),
    'kinematic_v':      ('Vel', 'F'),
    'rotational':       ('Theta', 'Tau'),
    'rotational_omega': ('Omega', 'Tau'),
}


@dataclass
class Param:
    name: str
    default: str = "0"
    type: str = "real"
    from_range: Optional[str] = None    # e.g. "(0:inf)"
    # Every value range of the declaration, in order: ('from', text) and
    # ('exclude', text), text as collected ("( 0 : inf )", "0", "'{ ... }");
    # va_const.check_param_ranges enforces them per instance.
    value_ranges: list = field(default_factory=list)
    desc: Optional[str] = None
    is_instance: bool = False           # type="instance" attribute
    # xyceAlsoModel="yes" on an instance parameter (the Xyce/ADMS convention,
    # utils/ADMS/xyceImplementationFile_nosac.xml): a .model card may give it
    # too, as the default of the instances that do not.
    also_model: bool = False
    # Array-parameter support: when declared as `parameter real T[0:N-1] = '{...}`
    # array_size is N and elements is the parsed list of initialiser values.
    array_size: Optional[int] = None
    elements: Optional[list[str]] = None
    # Where the declaration starts (the source file and line: the
    # preprocessor's line map).
    loc: Optional[SourceLoc] = None


@dataclass
class Var:
    name: str
    type: str = "real"
    # ``real x = expr;``: the initializer (Parser._parse_variables makes it
    # an assignment at the head of the analog block).
    init: Optional[str] = None
    loc: Optional["SourceLoc"] = None


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
    # A loop (for / while / repeat) is a BLOCK whose children are
    # [init?, body, step?] (one pass of them to a walker that does not know
    # loops) and whose `loop` holds kind ('for' | 'while' | 'repeat'), cond
    # (the condition text; for repeat the count expression), has_init and
    # has_step.  GiNaCEmitter._walk_loop iterates it.
    loop: Optional[dict] = None
    # ASSIGN to an integer variable from a real expression: the value is
    # rounded (LRM 4.2.1.1, halves away from zero) when it is assigned.
    int_lhs: bool = False
    # INITIAL_STEP (an event-controlled statement): the events of its
    # event expression, [{'kind': 'initial_step' | 'final_step' | 'cross' |
    # 'above' | ..., 'args': [argument texts]}] (``@(initial_step("ac",
    # "tran"))`` -> kind initial_step, args ['"ac"', '"tran"']; ``or``
    # separates several).
    event: Optional[list] = None
    # ASSIGN made from a variable declaration assignment (``real x = 1;``):
    # 'module' for a module variable, 'block' for a block-local one.  The
    # emitter gives a held (event) variable its declaration value once,
    # as the initial value of its held state, not at every evaluation.
    decl_init: Optional[str] = None


@dataclass
class Module:
    name: str
    ports: list[Port] = field(default_factory=list)
    params: list[Param] = field(default_factory=list)
    variables: list[Var] = field(default_factory=list)
    internal_nodes: list[str] = field(default_factory=list)
    branch_map: dict[str, str] = field(default_factory=dict)  # branch_name → primary node
    # Secondary endpoint, when the branch is declared with two nodes
    # (e.g. ``branch (i2, i1) b_rb;``). The wrapper needs this to stamp
    # the branch current into both KCL rows with the correct sign.
    branch_neg_map: dict[str, str] = field(default_factory=dict)
    analog_block: Optional[ASTNode] = None
    attributes: dict[str, str] = field(default_factory=dict)
    # The names of ``ground`` declarations (LRM 3.6.4).  Parser._apply_grounds
    # rewrites every reference to one of them to the implicit-ground form
    # (V(a, g) -> V(a), I(g, a) <+ e -> I(a) <+ -(e), ...), so no other part
    # of PyMS sees a declared ground as a node.
    grounds: list[str] = field(default_factory=list)
    # ``genvar`` names (the counters of analog for loops).
    genvars: list[str] = field(default_factory=list)
    # ``analog function`` declarations (AnalogFunction), in source order.
    functions: list = field(default_factory=list)
    # ``aliasparam ALIAS = TARGET;`` (LRM 3.4.7): alias -> target, in source
    # order.  The device shell registers each alias as another name of the
    # target's value on the .model card / instance line (1.1.8, BUGS.md B26).
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class AnalogFunction:
    """``analog function [real|integer] NAME; ... endfunction`` (LRM 4.7).
    args: (name, 'input' | 'output' | 'inout') in declaration order of the
    direction declarations; variables: the declared types of the arguments
    and the locals; body: the statement."""
    name: str
    rtype: str = "real"
    args: list = field(default_factory=list)
    variables: list = field(default_factory=list)
    body: Optional[ASTNode] = None
    loc: Optional[SourceLoc] = None


@dataclass
class Paramset:
    """Verilog-AMS ``paramset NAME UNDERLYING ... endparamset`` declaration.

    Semantically a paramset is a *defaults override* on the underlying
    module: it exposes its own instance parameters (typically a subset
    or with tighter ranges), and via the ``.NAME = expr`` lines it
    rewrites the default values of named parameters of the underlying.
    No new analog behaviour, no macromodule expansion — at parse_file
    time the paramset is resolved into a Module by:

      1. Taking the underlying's Module verbatim (ports, variables,
         analog block).
      2. Prepending the paramset's own instance parameters (caller-
         supplied).
      3. Prepending the paramset's localparams as variable assignments
         at the head of the analog block (so the binding expressions
         can reference them).
      4. For each ``.X = expr`` binding, replacing the default value
         of underlying parameter X with ``expr``.
      5. Renaming the resulting Module to the paramset's name.
    """
    name: str
    underlying_name: str
    params: list[Param] = field(default_factory=list)
    localparams: list[tuple[str, str]] = field(default_factory=list)
    bindings: list[tuple[str, str]] = field(default_factory=list)
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
    # A based literal ('h3e8; a size before it is a NUMBER token of its own).
    ('BASED', r"'[sS]?[hHdDoObB][0-9a-fA-F_xXzZ?]+"),
    # Digits may carry the digit separator _ ; every LRM scale factor (T G M K k m
    # u n p f a) belongs to the number, so ``1a`` and ``1K`` are one token.
    ('NUMBER', r'[+-]?(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?(?:[TGMKkmunpfa])?'),
    ('IDENT', r'[A-Za-z_\$`][A-Za-z0-9_\$]*'),
    # The two-character operators are single tokens (``>=`` was ``>`` ``=``).
    ('OP', r'\*\*|<<|>>|<=|>=|==|!=|&&|\|\||[+\-*/%(){},;:=<>!&|~\[\]@^?.]'),
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
        value = m.group()
        if typ == 'NUMBER':
            value = plain_number(value)
        tokens.append(Token(typ, value, m.start()))
    return tokens


# ---------------------------------------------------------------------------
# Scale factors
# ---------------------------------------------------------------------------

_SCALE = {
    'a': 1e-18, 'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6,
    'm': 1e-3, 'k': 1e3, 'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12,
}


# The exponent of each scale factor, so that ``4.7p`` reads as the double
# nearest 4.7e-12 (the multiplication 4.7 * 1e-12 is one ulp off).
_SCALE_EXP = {
    'a': -18, 'f': -15, 'p': -12, 'n': -9, 'u': -6,
    'm': -3, 'k': 3, 'K': 3, 'M': 6, 'G': 9, 'T': 12,
}


def plain_number(s: str) -> str:
    """A Verilog-A number token as C++ / Python reads it: a scale factor
    (LRM 2.6.2: T G M K k m u n p f a) becomes the real it stands for, and
    the digit separators _ go (an integer stays an integer: ``1_000`` is
    1000, ``1k`` is the real 1000.0).  Every expression the emitters
    translate reaches g++ through the tokens, so ``r / 1k`` there was the
    g++ error "unable to find numeric literal operator 'operator""k'"
    (va_const does the same for the parameter defaults it evaluates)."""
    t = s.replace('_', '')
    if t and t[-1] in _SCALE_EXP:
        mant = t[:-1]
        try:
            if 'e' in mant or 'E' in mant:
                return repr(float(mant) * _SCALE[t[-1]])
            return repr(float(mant + 'e' + str(_SCALE_EXP[t[-1]])))
        except ValueError:
            return s
    return t


def parse_number(s: str) -> str:
    """Convert a Verilog-A number that stands alone (an optional scale suffix,
    digit separators) to a numeric string; any other text is returned as is —
    an expression default is evaluated by va_const, not here."""
    if s and s[-1] in _SCALE:
        try:
            return str(float(s[:-1].replace('_', '')) * _SCALE[s[-1]])
        except ValueError:
            return s
    if '_' in s:
        try:
            return str(float(s.replace('_', '')))
        except ValueError:
            return s
    return s


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


def _strip_strings(text: str) -> str:
    """text with every string literal emptied (its words are no names)."""
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', text or '')


def _also_model(attrs: dict) -> bool:
    """The xyceAlsoModel attribute (any case of the key; "yes"/"YES"/"1")."""
    for k, v in attrs.items():
        if k.lower() == 'xycealsomodel':
            return str(v).strip().lower() not in ('no', '0', 'false')
    return False


class Parser:
    def __init__(self, tokens: list[Token], source: str = "", filename: str = "",
                 line_map: Optional[list[tuple[str, int]]] = None):
        self.tokens = tokens
        self.pos = 0
        self.filename = filename
        # Build line offset table for pos→line mapping
        self._line_offsets = [0]
        for i, ch in enumerate(source):
            if ch == '\n':
                self._line_offsets.append(i + 1)
        # The preprocessor's map from a line of the preprocessed text to its
        # source (file, line) (preprocess.PreprocessedText.line_map): with it
        # every location names the true file and line, also inside an
        # included file; without it (unpreprocessed text) the line of source.
        if line_map is None:
            line_map = getattr(source, 'line_map', None)
        self._line_map = line_map
        # Disciplines a node may be declared with and the access functions
        # they bring into scope, as V-like (potential) or I-like (flow).
        self.disciplines: dict[str, tuple[Optional[str], Optional[str]]] = dict(LRM_DISCIPLINES)
        self.natures: dict[str, str] = {}             # nature name -> its access function
        self.access_kinds: dict[str, ContribKind] = {}
        for potential, flow in self.disciplines.values():
            self._add_access(potential, flow)
        # The access functions in scope in the module being parsed: V and I,
        # plus those of every discipline one of its nodes is declared with.
        self.module_accesses: set[str] = {'V', 'I'}

    def _add_access(self, potential: Optional[str], flow: Optional[str]):
        if potential:
            self.access_kinds[potential] = ContribKind.V
        if flow:
            self.access_kinds[flow] = ContribKind.I

    def _expr_token(self, t: Token) -> str:
        """The text of token t inside a collected expression: the access
        function of a non-electrical discipline of this module followed by
        '(' (Temp(t), Pwr(t), ...) becomes V or I, the electrical spelling
        of the same potential read or flow probe."""
        if t.type == 'IDENT' and t.value in self.module_accesses and t.value not in ('V', 'I'):
            nxt = self.peek(1)
            if nxt is not None and nxt.value == '(':
                return 'V' if self.access_kinds[t.value] == ContribKind.V else 'I'
        if t.type == 'IDENT' and getattr(self, '_scopes', None):
            nxt = self.peek(1)
            prv = self.tokens[self.pos - 1] if self.pos > 0 else None
            if (nxt is None or nxt.value != '(') and (prv is None or prv.value != '.'):
                return self._scoped(t.value)
        return t.value

    def _error(self, msg: str, loc: Optional[SourceLoc] = None) -> ParseError:
        """A ParseError whose message starts with the source file:line."""
        loc = loc if loc is not None else self._loc()
        return ParseError(f"{loc}: {msg}" if loc.line else f"{loc.file or '<source>'}: {msg}")

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
        if self._line_map and lo < len(self._line_map):
            file, line = self._line_map[lo]
            return SourceLoc(file, line)
        return SourceLoc(self.filename, lo + 1)  # 1-based line numbers

    def _eof_loc(self) -> SourceLoc:
        """The location of the last token (for an error at the end of the text)."""
        if self.tokens:
            return self._loc_at(self.tokens[-1].pos)
        return SourceLoc(self.filename, 0)

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
            raise self._error(f"expected '{value}', got '{got}'",
                              self._loc() if t else self._eof_loc())
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
            raise self._error(f"expected an identifier, got '{got}'",
                              self._loc() if t else self._eof_loc())
        self.pos += 1
        return t.value

    # --- Module-level parsing ---

    def parse_module(self) -> Module:
        # Skip preprocessor directives and nature/discipline blocks before module
        self._skip_preamble()

        # No module found (e.g. include-only / macro-definition file)
        if self.pos >= len(self.tokens):
            raise ParseError(f"{self.filename or '<source>'}: no module found in file "
                             f"(macro/include-only file?)")

        # Parse optional attributes (* ... *)
        attrs = {}
        if self.at('(*'):
            attrs = self._parse_attributes()

        self.expect('module')
        name = self.consume_ident()

        mod = Module(name=name, attributes=attrs)
        self._cur_mod = mod
        self._cur_func = None
        # Block-local variable scopes (Parser._parse_begin_end): one dict per
        # open begin...end, name as written -> the module variable it is.
        self._scopes = []
        self._scope_seq = 0
        self.module_accesses = {'V', 'I'}

        # Port list: names (list_of_ports), or the ANSI form (LRM 6.2,
        # list_of_port_declarations) where a direction and/or a discipline
        # precede the names and hold until the next one:
        # ``module a(inout electrical p, n, input thermal t)``.  (Before
        # 1.1.3 the keywords became port names: g++ "redefinition of
        # nodeID_inout".)
        if self.at('('):
            self.advance()  # (
            cur_dir, cur_disc, ansi = None, None, False
            _DIRS = {'inout': PortDir.INOUT, 'input': PortDir.IN, 'output': PortDir.OUT}
            while not self.at(')'):
                if self.peek() is None:
                    raise self._error("module port list without ')'", self._eof_loc())
                if self.at('(*'):
                    self._parse_attributes()
                    continue
                v = self.peek().value
                if v in _DIRS:
                    self.advance()
                    cur_dir, cur_disc, ansi = _DIRS[v], None, True
                    continue
                if v in self.disciplines and self.peek(1) is not None and self.peek(1).type == 'IDENT':
                    self.advance()
                    cur_disc, ansi = v, True
                    self.module_accesses.update(a for a in self.disciplines[v] if a)
                    continue
                if v in ('wire', 'wreal'):
                    self.advance()
                    continue
                pn = self.consume_ident()
                port = Port(name=pn)
                if cur_dir is not None:
                    port.direction = cur_dir
                if cur_disc is not None:
                    port.discipline = cur_disc
                mod.ports.append(port)
                if self.at(','):
                    self.advance()
            self.expect(')')

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
            elif t.value in self.disciplines:
                self._parse_node_decl(mod, t.value)
            elif self.at('parameter'):
                mod.params.extend(self._parse_param())
            elif self.at('real') or self.at('integer'):
                self._parse_variables(mod)
            elif self.at('analog'):
                t2 = self.peek(1)
                if t2 and t2.value == 'function':
                    mod.functions.append(self._parse_analog_function())
                else:
                    mod.analog_block = self._parse_analog()
            elif self.at('(*'):
                # Attribute before parameter or other declaration
                attrs = self._parse_attributes()
                if self.at('parameter'):
                    for p in self._parse_param():
                        for k, v in attrs.items():
                            if k == 'desc':
                                p.desc = v
                            elif k == 'type' and v == 'instance':
                                p.is_instance = True
                        if _also_model(attrs):
                            p.also_model = True
                        mod.params.append(p)
                elif self.at('aliasparam'):
                    self._parse_aliasparam(mod)
            elif self.at('branch'):
                # branch (node[, node2]) name [, name2 ...] ;
                #   Verilog-A allows a comma-separated list of branch
                #   identifiers sharing the same node binding.
                self.advance()  # 'branch'
                if self.at('('):
                    self.advance()
                    node_name = self.consume_ident()
                    # Some declarations list both ends (`branch (p, n) b`).
                    # Keep the second one in branch_neg_map so the
                    # wrapper can stamp the branch current into both
                    # KCL rows; the GiNaC emitter uses the primary
                    # (branch_map[name]) for V(branch_name) lookups.
                    second_name = None
                    while self.at(','):
                        self.advance()
                        second_name = self.consume_ident()
                    self.expect(')')
                    for branch_name, _, _ in self._parse_decl_names('branch'):
                        mod.branch_map[branch_name] = node_name
                        if second_name is not None:
                            mod.branch_neg_map[branch_name] = second_name
                else:
                    self._skip_to_semi()
            elif self.at('aliasparam'):
                self._parse_aliasparam(mod)
            elif self.at('genvar'):
                self.advance()
                mod.genvars.extend(n for n, _, _ in self._parse_decl_names('genvar'))
            elif self.at('ground'):
                self.advance()
                # ``ground electrical g;``: an optional discipline
                if self.peek() is not None and self.peek().value in self.disciplines \
                        and self.peek(1) is not None and self.peek(1).type == 'IDENT':
                    self.advance()
                mod.grounds.extend(n for n, _, _ in self._parse_decl_names('ground'))
            elif t.value.startswith('`'):
                self._skip_directive()
            else:
                # Skip unknown token
                self.advance()

        self.expect('endmodule')
        self._check_port_param_names(mod)
        self._check_aliases(mod)
        self._apply_variable_initializers(mod)
        self._apply_grounds(mod)
        from .int_ops import lower_integer_ops
        lower_integer_ops(mod)
        return mod

    def _parse_aliasparam(self, mod: Module):
        """``aliasparam ALIAS = TARGET ;`` -- recorded in mod.aliases (checked
        against the parameters in _check_aliases, after the body)."""
        loc = self._loc()
        self.advance()  # aliasparam
        alias = self.consume_ident()
        self.expect('=')
        target = self.consume_ident()
        self._skip_inline_attrs()
        self.expect(';')
        if alias in mod.aliases:
            raise self._error(f"aliasparam '{alias}' declared twice", loc)
        mod.aliases[alias] = target
        if not hasattr(self, '_alias_locs'):
            self._alias_locs = {}
        self._alias_locs[(mod.name, alias)] = loc

    def _check_aliases(self, mod: Module):
        """An alias names a parameter of the module (LRM 3.4.7) and is no
        parameter, port or other alias itself, in any case (Xyce folds the
        card's names to capitals)."""
        names = {p.name for p in mod.params}
        upper = {p.name.upper() for p in mod.params}
        for alias, target in mod.aliases.items():
            loc = getattr(self, '_alias_locs', {}).get((mod.name, alias), SourceLoc(self.filename, 0))
            if target not in names:
                raise self._error(f"aliasparam {alias} = {target}: module '{mod.name}' has no parameter "
                                  f"'{target}'", loc)
            if alias.upper() in upper:
                raise self._error(f"aliasparam {alias}: module '{mod.name}' already has a parameter of that "
                                  f"name (Xyce parameter names are case-insensitive)", loc)
        seen = {}
        for alias in mod.aliases:
            if alias.upper() in seen:
                raise self._error(f"aliasparams '{seen[alias.upper()]}' and '{alias}' differ only in case",
                                  getattr(self, '_alias_locs', {}).get((mod.name, alias), SourceLoc(self.filename, 0)))
            seen[alias.upper()] = alias

    # --- ground declarations ---

    _ACCESS_TEXT_RE = re.compile(
        r'("(?:[^"\\]|\\.)*")|\b([VI])\s*\(\s*(<)?\s*(\w+)\s*(>)?\s*(?:,\s*(\w+)\s*)?\)')

    def _apply_grounds(self, mod: Module):
        """A net declared ``ground`` (LRM 3.6.4) is the reference node: every
        reference to it becomes the implicit-ground form the rest of PyMS
        handles -- V(a, g) -> V(a), V(g, a) -> (-V(a)), V(g) -> 0, the same
        for the current probes, a contribution I(g, a) <+ e -> I(a) <+ -(e),
        a branch (g, a) b -> (a) b with its probes and contributions negated
        -- and the net is no internal node.  Before, ``electrical g; ground
        g;`` left g an ordinary floating node (and a short to it collapsed g
        and its partner into one floating node), and ``ground g;`` alone
        reached the C++ as an undeclared symbol."""
        gset = set(mod.grounds)
        if not gset:
            return
        for p in mod.ports:
            if p.name in gset:
                raise self._error(f"port '{p.name}' of module '{mod.name}' is declared ground: "
                                  f"PyMS does not support a ground port", SourceLoc(self.filename, 0))
        mod.internal_nodes = [n for n in mod.internal_nodes if n not in gset]
        # named branches with a ground end: (a, g) -> (a); (g, a) -> (a), negated
        negated = set()
        for br in list(mod.branch_map):
            a, b = mod.branch_map[br], mod.branch_neg_map.get(br)
            ag, bg = a in gset, b is not None and b in gset
            if ag and (b is None or bg):
                raise self._error(f"branch '{br}' has both ends on ground", SourceLoc(self.filename, 0))
            if bg:
                mod.branch_neg_map.pop(br, None)
            elif ag:
                mod.branch_map[br] = b
                mod.branch_neg_map.pop(br, None)
                negated.add(br)

        def ends(n1, n2):
            """(kept names, sign, zero): the reference after grounding."""
            if n2 is None:
                if n1 in gset:
                    return None, 1, True
                return (n1,), (-1 if n1 in negated else 1), False
            g1, g2 = n1 in gset, n2 in gset
            if g1 and g2:
                return None, 1, True
            if g2:
                return (n1,), 1, False
            if g1:
                return (n2,), -1, False
            return (n1, n2), 1, False

        def text(s):
            if not s:
                return s

            def repl(m):
                if m.group(1) is not None or m.group(3) or m.group(5):
                    return m.group(0)
                names, sign, zero = ends(m.group(4), m.group(6))
                if zero:
                    return '0'
                if sign == 1 and len(names) == (2 if m.group(6) else 1):
                    return m.group(0)
                acc = f"{m.group(2)}({', '.join(names)})"
                return acc if sign == 1 else f'(-{acc})'
            return self._ACCESS_TEXT_RE.sub(repl, s)

        def walk(node):
            if node is None:
                return node
            for f in ('expr', 'condition', 'text'):
                v = getattr(node, f, None)
                if isinstance(v, str):
                    setattr(node, f, text(v))
            if node.loop is not None and node.loop.get('cond'):
                node.loop['cond'] = text(node.loop['cond'])
            for ev in node.event or ():
                ev['args'] = [text(a) for a in ev['args']]
            if node.kind == NodeKind.CONTRIB and node.branch:
                names, sign, zero = ends(node.branch[0], node.branch[1] if len(node.branch) > 1 else None)
                if zero:
                    if node.contrib_kind == ContribKind.V and node.expr.strip() not in ('0', '0.0'):
                        raise self._error(f"V({', '.join(node.branch)}) <+ ...: a potential "
                                          f"contribution between ground and ground", node.loc)
                    return ASTNode(kind=NodeKind.BLOCK, loc=node.loc)
                node.branch = names
                if sign < 0 and node.expr.strip() not in ('0', '0.0'):
                    node.expr = f'-( {node.expr} )'
            node.children = [walk(c) for c in node.children]
            node.else_body = walk(node.else_body)
            return node
        mod.analog_block = walk(mod.analog_block)

    def _apply_variable_initializers(self, mod: Module):
        """``real x = expr;`` gives x its value before the analog block runs
        (LRM 4.2.1: the variable declaration assignment): the same as an
        assignment at the head of the block.  IHP sg13g2's resistor.va and
        capacitor.va keep their constants this way (``real CONSTvt0 =
        1.38064852e-23*(27+273.15)/1.6021766208e-19;``); dropped, they read 0."""
        inits = [ASTNode(kind=NodeKind.ASSIGN, lhs=v.name, expr=v.init, loc=v.loc,
                         decl_init='module')
                 for v in mod.variables if v.init is not None]
        if mod.analog_block is None:
            return
        # Block-local declaration assignments (``begin : b real t = 0; ...``)
        # with a constant expression (LRM 2.4 Annex A: real_type ::= identifier
        # [= constant_expression]) are hoisted to the head of the analog block,
        # after the module variables' ones.  An analog variable is static (it
        # keeps its value); the declaration assignment gives it its value once
        # -- for PyMS, whose variables start every evaluation afresh, once per
        # evaluation -- and NOT again when the block is entered again: a loop
        # body that declares ``real t = 0;`` and then ``t = t + 1;`` counts
        # 1, 2, 3 over three passes (VACASK the same), where 1.1.3 re-ran the
        # assignment at the declaration on every pass (1, 1, 1).  A
        # non-constant initializer (not LRM Verilog-A) stays where it is
        # written.  The same inside an analog function: to the head of its body.
        consts = self._param_names_for_inits(mod)
        hoisted = self._take_block_inits(mod.analog_block, consts)
        inits = inits + hoisted
        for f in mod.functions:
            # In a function body a name of the function's own scope (an
            # argument, a local, the function's name) is that variable, not
            # the module parameter it may shadow: not a constant (1.1.5;
            # 1.1.4 hoisted ``real t = r + 1000;`` above the assignment of a
            # local r that shadows a parameter r and read r = 0).
            own = {f.name} | {a for a, _d in f.args} | {v.name for v in f.variables}
            fh = self._take_block_inits(f.body, consts - own)
            if fh:
                fb = f.body
                if fb.kind == NodeKind.BLOCK and fb.loop is None:
                    fb.children[:0] = fh
                else:
                    f.body = ASTNode(kind=NodeKind.BLOCK, children=fh + [fb], loc=fb.loc)
        if not inits:
            return
        block = mod.analog_block
        if block.kind == NodeKind.BLOCK and block.loop is None:
            block.children[:0] = inits
        else:
            mod.analog_block = ASTNode(kind=NodeKind.BLOCK, children=inits + [block],
                                       loc=block.loc)

    _CONST_INIT_FUNCS = frozenset((
        'ln', 'log', 'exp', 'sqrt', 'pow', 'abs', 'min', 'max', 'floor', 'ceil', 'sin', 'cos',
        'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh',
        'atanh', 'hypot', 'limexp', '$simparam'))

    @staticmethod
    def _param_names_for_inits(mod: Module) -> set:
        return {p.name for p in mod.params}

    def _is_constant_init(self, expr: Optional[str], consts: set) -> bool:
        if not expr or not expr.strip():
            return False
        s = _strip_strings(expr)
        for m in re.finditer(r"(?<![\w.'`$])([$]?[A-Za-z_]\w*)", s):
            name = m.group(1)
            if name in consts or name in self._CONST_INIT_FUNCS:
                continue
            return False
        return True

    def _take_block_inits(self, node: Optional[ASTNode], consts: set) -> list:
        """Remove the constant block-local declaration assignments below node
        (in source order) and return them."""
        out = []

        def walk(n):
            if n is None:
                return
            keep = []
            for c in n.children:
                if (c.kind == NodeKind.ASSIGN and c.decl_init == 'block'
                        and self._is_constant_init(c.expr, consts)):
                    out.append(c)
                    continue
                walk(c)
                keep.append(c)
            n.children = keep
            walk(n.else_body)
        walk(node)
        return out

    def _check_port_param_names(self, mod: Module):
        """A parameter may not share its name with a port: both are names
        of the module's scope (LRM 3.4: one name space per module), and
        PyMS would read V(r) and r as one and the same.  openvaf-r refuses
        it too."""
        ports = {p.name for p in mod.ports}
        for p in mod.params:
            if p.name in ports:
                raise self._error(
                    f"parameter '{p.name}' has the name of a port of module "
                    f"'{mod.name}' (a port and a parameter cannot share a name)",
                    p.loc or SourceLoc(self.filename, 0))

    def _skip_preamble(self):
        """Skip preprocessor directives, nature/discipline blocks before module."""
        while self.pos < len(self.tokens):
            t = self.peek()
            if t is None:
                break
            if t.value.startswith('`'):
                self._skip_directive()
            elif t.value == 'nature':
                self._parse_nature()
            elif t.value == 'discipline':
                self._parse_discipline()
            elif t.value in ('module', 'paramset', 'parameter'):
                # ``parameter`` here is a top-level decl between
                # entities (see _parse_all_entities) — let that loop
                # capture it rather than blindly skipping past.
                break
            elif t.value == '(*':
                # Could be attribute on module — check if 'module' follows
                break
            else:
                # Skip unknown preamble token
                self.advance()

    def parse_paramset(self) -> Paramset:
        """Parse ``paramset NAME UNDERLYING <body> endparamset``.

        Each body statement is one of:
          * ``parameter [type] N = V [from ...] ;``  paramset-side instance
            param (the caller-supplied subset of the underlying's params)
          * ``localparam [type] N = E ;``             paramset-local var,
            available to subsequent bindings as a sub-expression
          * ``aliasparam X = Y ;``                    alias (recorded as a
            zero-default parameter named X)
          * ``. NAME = expr ;``                       binding — sets the
            default of underlying parameter NAME to expr
        """
        ps_attrs = {}
        if self.at('(*'):
            ps_attrs = self._parse_attributes()
        self.expect('paramset')
        name = self.consume_ident()
        underlying = self.consume_ident()
        ps = Paramset(name=name, underlying_name=underlying, attributes=ps_attrs)

        while not self.at('endparamset'):
            t = self.peek()
            if t is None:
                break
            # Optional inline attribute on a parameter declaration.
            attrs = {}
            if self.at('(*'):
                attrs = self._parse_attributes()
            if self.at('parameter') or self.at('localparam'):
                is_local = (self.peek().value == 'localparam')
                self.advance()  # parameter / localparam
                # Optional type token (real/integer)
                typ = 'real'
                if self.at('real') or self.at('integer'):
                    typ = self.advance().value
                pname = self.consume_ident()
                # Some gnucap sources put the type AFTER the name:
                # ``parameter rfmode integer = 0 from [0:1];``
                if self.at('real') or self.at('integer'):
                    typ = self.advance().value
                if not self.at('='):
                    # paramset parameters always carry a default — skip
                    self._skip_to_semi()
                    continue
                self.advance()  # =
                default_text = self._collect_param_value()
                # Discard optional ``from [lo:hi]`` / ``exclude``
                while self.at('from') or self.at('exclude'):
                    self.advance()
                    self._collect_param_value()
                self._skip_inline_attrs()
                self.expect(';')
                if is_local:
                    ps.localparams.append((pname, default_text))
                else:
                    p = Param(name=pname, type=typ,
                              default=parse_number(default_text))
                    # Paramset parameters are inherently the
                    # instance-side overrides the caller supplies, so
                    # default to is_instance=True unless the source
                    # explicitly marks them otherwise.
                    p.is_instance = (attrs.get('type') != 'model')
                    if 'desc' in attrs:
                        p.desc = attrs['desc']
                    ps.params.append(p)
                continue
            if self.at('aliasparam'):
                self.advance()
                _alias = self.consume_ident()
                if self.at('='):
                    self.advance()
                    self.consume_ident()  # original
                self._skip_to_semi()
                continue
            if self.at('.'):
                # ``.NAME = expr ;`` binding
                self.advance()  # .
                bname = self.consume_ident()
                self.expect('=')
                bexpr = self._collect_expr_until(';')
                self.expect(';')
                ps.bindings.append((bname, bexpr))
                continue
            if t.value.startswith('`'):
                self._skip_directive()
                continue
            # Anything else inside a paramset body is unusual — skip to
            # the next ``;`` to avoid eating ``endparamset``.
            self._skip_to_semi()
        self.expect('endparamset')
        return ps

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

    def _parse_analog_function(self) -> 'AnalogFunction':
        """``analog function [real|integer] NAME ; {input|output|inout|real|
        integer declarations} statement endfunction`` (LRM 4.7.1).  The body
        is parsed like analog-block statements, in a name space of its own
        (the arguments, the locals and NAME, the return value)."""
        loc = self._loc()
        self.expect('analog')
        self.expect('function')
        rtype = 'real'
        if self.at('real', 'integer'):
            rtype = self.advance().value
        name = self.consume_ident()
        self.expect(';')
        func = AnalogFunction(name=name, rtype=rtype, loc=loc)
        saved = (getattr(self, '_cur_func', None), getattr(self, '_scopes', None),
                 getattr(self, '_scope_labels', None))
        self._cur_func, self._scopes, self._scope_labels = func, [], []
        try:
            while self.at('input', 'output', 'inout', 'real', 'integer', '(*'):
                if self.at('(*'):
                    self._parse_attributes()
                    continue
                kw = self.advance().value
                typ = None
                if kw in ('input', 'output', 'inout') and self.at('real', 'integer'):
                    typ = self.advance().value
                for n, _, nloc in self._parse_decl_names(kw, dims=True):
                    if kw in ('input', 'output', 'inout'):
                        func.args.append((n, kw))
                        if typ:
                            func.variables.append(Var(name=n, type=typ, loc=nloc))
                    else:
                        func.variables.append(Var(name=n, type=kw, loc=nloc))
            body = ASTNode(kind=NodeKind.BLOCK, loc=self._loc())
            while not self.at('endfunction'):
                if self.peek() is None:
                    raise self._error(f"analog function '{name}' without 'endfunction'", self._eof_loc())
                body.children.append(self._parse_statement())
            self.expect('endfunction')
        finally:
            self._cur_func, self._scopes, self._scope_labels = saved
        func.body = body.children[0] if len(body.children) == 1 else body
        # (The body's ternaries are lowered after lower_integer_ops has typed
        # it -- _parse_all_entities -- as the analog block's are: lowered
        # first, ``((a > 0) ? 7 : 9) / 2`` read an untyped temporary and was
        # a real division, x4_func_int_tern_div 0.4286 for 0.5, 1.1.3.)
        return func

    def _skip_analog_function(self):
        """Skip 'analog function ... endfunction'."""
        self.advance()  # analog
        self.advance()  # function
        while self.pos < len(self.tokens):
            t = self.advance()
            if t.value == 'endfunction':
                return

    def _consume_inline_attrs(self) -> dict[str, str]:
        """Parse all immediately-following inline `attr(...) or
        (* ... *) blocks, merging their key/value pairs and returning
        them. Returns {} when nothing follows."""
        merged: dict[str, str] = {}
        while self.pos < len(self.tokens):
            t = self.peek()
            if t.value == '(*':
                merged.update(self._parse_attributes())
            elif t.value.startswith('`'):
                t2 = self.peek(1)
                if t2 and t2.value == '(':
                    # `attr(...) or another macro call. If the
                    # preprocessor already expanded `attr to (* ... *)
                    # we'd never see this path; if a macro slipped
                    # through, skip the contents to keep the parser
                    # advancing.
                    self.advance()  # `name
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
        return merged

    def _skip_inline_attrs(self):
        """Back-compat wrapper for sites that just want to consume the
        attrs and don't need to capture them."""
        self._consume_inline_attrs()

    def _parse_attributes(self) -> dict[str, str]:
        """Parse ``(* key=value, key=value *)`` attributes.

        Real-world .va sources mix two separator conventions in this
        list:
          - comma-separated (Accellera / GNU make convention)
          - space-separated (``\\`attr(type="instance" info="...")``
            after BSIM-CMG/MVS/FBH-HBT macro expansion)
        Both are accepted; trailing commas are tolerated."""
        attrs = {}
        self.expect('(*')
        while not self.at('*)'):
            if self.at(','):
                self.advance()
                continue
            key = self.consume_ident()
            # IEEE 1364-2005 namespaced attribute keys: ``ns:name``
            # (e.g. HICUM L0's ``(* spice:name="is" ... *)``). Treat
            # the namespace and local name as one key for storage.
            if self.at(':'):
                self.advance()
                key = key + ':' + self.consume_ident()
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
        self.expect('*)')
        return attrs

    # Words that end a declaration's list of names: one of them where a name
    # or the ',' / ';' after a name is due means a missing ';' (``real v``
    # before ``analog begin`` read the analog block as variable names).
    _DECL_KEYWORDS = frozenset((
        'analog', 'begin', 'end', 'endmodule', 'module', 'parameter', 'localparam',
        'real', 'integer', 'inout', 'input', 'output', 'branch', 'genvar',
        'if', 'else', 'case', 'for', 'while'))

    def _skip_decl_attrs(self):
        """Attribute instances on a declared name (HICUM L0's ``electrical
        c `P(info="...")``): they name nothing and are dropped."""
        while self.at('(*'):
            self._parse_attributes()

    def _parse_decl_names(self, what: str, dims: bool = False,
                          init: bool = False) -> list[tuple[str, Optional[str], SourceLoc]]:
        """The list of a declaration after its keyword, up to and including
        the ';': ``name {, name} ;`` where a name may carry attribute
        instances, ``[lo:hi]`` dimensions (dims) and ``= expr`` (init).
        Anything else is a ParseError naming the file and line; before, the
        loops took every identifier up to the next ';' as a name and skipped
        the rest, so a missing ';' turned the statements after it into
        names (``real v`` + ``analog begin ... end``: no analog block, a
        device that conducts nothing, exit 0).  Returns (name, initializer
        text or None, location) per name."""
        items = []
        while True:
            self._skip_decl_attrs()
            t = self.peek()
            if t is None or t.type != 'IDENT' or t.value in self._DECL_KEYWORDS:
                got = t.value if t else 'end of file'
                raise self._error(f"in the '{what}' declaration: expected a name, "
                                  f"got '{got}'", self._loc() if t else self._eof_loc())
            loc = self._loc()
            name = self.advance().value
            while dims and self.at('['):
                self.advance()
                self._collect_expr_until(']')
                self.expect(']')
            value = None
            if init and self.at('='):
                self.advance()
                value = self._collect_expr_until(',', ';', *self._DECL_KEYWORDS)
                if not value.strip():
                    raise self._error(f"in the '{what}' declaration: '{name} =' has no value")
            self._skip_decl_attrs()
            items.append((name, value, loc))
            if self.at(','):
                self.advance()
                continue
            if self.at(';'):
                self.advance()
                return items
            t = self.peek()
            got = t.value if t else 'end of file'
            raise self._error(
                f"in the '{what}' declaration: expected ',' or ';' after '{name}', "
                f"got '{got}' (a missing ';'?)", loc)

    def _parse_port_dir(self, mod: Module, direction: PortDir):
        what = self.advance().value  # inout/input/output
        # ``inout electrical p, n;``: the direction and the discipline in
        # one declaration (LRM 6.2 port_declaration).  Before 1.1.3 the
        # discipline was read as the first port name (a parse error).
        disc = None
        if self.peek() is not None and self.peek().value in self.disciplines \
                and self.peek(1) is not None and self.peek(1).type == 'IDENT':
            disc = self.advance().value
            self.module_accesses.update(a for a in self.disciplines[disc] if a)
        for name, _, _ in self._parse_decl_names(what):
            for p in mod.ports:
                if p.name == name:
                    p.direction = direction
                    if disc is not None:
                        p.discipline = disc

    def _parse_node_decl(self, mod: Module, discipline: str):
        """``electrical a, b;``, ``thermal t;``, ...: a port named here
        gets the discipline; any other name is an internal node.  The
        discipline's access functions come into scope for the module."""
        self.advance()  # the discipline
        names = [n for n, _, _ in self._parse_decl_names(discipline)]
        self.module_accesses.update(a for a in self.disciplines[discipline] if a)
        port_names = {p.name for p in mod.ports}
        for name in names:
            if name not in port_names:
                mod.internal_nodes.append(name)
            else:
                for p in mod.ports:
                    if p.name == name:
                        p.discipline = discipline

    def _parse_nature(self):
        """``nature NAME ... access = ACC; ... endnature``: record the
        access function, for a discipline declared in the source that
        binds this nature (the standard ones are in LRM_DISCIPLINES)."""
        self.advance()  # nature
        name = self.peek().value if self.at_type('IDENT') else None
        while self.pos < len(self.tokens) and not self.at('endnature'):
            if name and self.at('access') and self.peek(1) is not None \
                    and self.peek(1).value == '=' and self.peek(2) is not None \
                    and self.peek(2).type == 'IDENT':
                self.natures[name] = self.peek(2).value
            self.advance()
        if self.at('endnature'):
            self.advance()
        if self.at(';'):
            self.advance()

    def _parse_discipline(self):
        """``discipline NAME potential NAT; flow NAT; enddiscipline``: a
        node declared with NAME is an MNA node, and the access functions
        of its natures read it as V (potential) or contribute as I (flow).
        A discipline without a continuous nature (the digital ``logic``)
        is not a node declaration and is skipped."""
        self.advance()  # discipline
        name = self.peek().value if self.at_type('IDENT') else None
        potential = flow = None
        while self.pos < len(self.tokens) and not self.at('enddiscipline'):
            if self.at('potential', 'flow') and self.peek(1) is not None \
                    and self.peek(1).type == 'IDENT':
                access = self.natures.get(self.peek(1).value)
                if self.at('potential'):
                    potential = access
                else:
                    flow = access
            self.advance()
        if self.at('enddiscipline'):
            self.advance()
        if self.at(';'):
            self.advance()
        if name and (potential or flow) and name not in self.disciplines:
            self.disciplines[name] = (potential, flow)
            self._add_access(potential, flow)

    def _untyped_param_type(self, default: str, items: list) -> str:
        """The type of an untyped parameter: the type of its default
        expression (LRM 3.4.1) -- integer for an integer expression (integer
        literals, integer parameters declared before it, the integer
        operators on them), real for a real one, string for a string.
        Before 1.1.5 only a bare integer literal made an integer
        (``parameter m = 2 * 2`` and ``parameter b = a`` with an integer a
        were real: 4 / 3 gave 1.33 instead of 1)."""
        from .int_ops import analyse
        types = {}
        mod = getattr(self, '_cur_mod', None)
        for p in (mod.params if mod is not None else ()):
            types[p.name] = {'integer': 'int', 'string': 'str'}.get((p.type or 'real').lower(), 'real')
        for it in items:
            types[it[0]] = {'integer': 'int', 'string': 'str'}.get(it[1], 'real')
        d = default.strip()
        try:
            _new, t = analyse(d, types)
        except Exception:
            t = None
        if t == 'int':
            return 'integer'
        if t == 'str':
            return 'string'
        if t == 'real':
            return 'real'
        # a text the typer cannot read: as before
        return ('string' if d.startswith('"') else
                'integer' if re.fullmatch(r'[+-]?\d+', d) else 'real')

    def _parse_param(self) -> list:
        """``parameter [real|integer|string] name = default {from|exclude
        range} {, name = default {range}} {attribute} ;`` (LRM 3.4: a
        parameter declaration declares a LIST of parameters).  Returns the
        Params in declaration order; the attributes of the declaration (before
        ``parameter`` or after the last item) belong to every one.  Before
        1.1.4 one name was read and the rest of the statement became its
        default (``parameter real is = 1e-14, nf = 1.5;``: a va_const refusal
        of '=', BUGS.md B20)."""
        # Handle optional attributes before 'parameter'
        attrs = {}
        if self.at('(*'):
            attrs = self._parse_attributes()

        self.expect('parameter')
        typ = None
        if self.at('real', 'integer', 'string'):
            typ = self.advance().value
        items = []
        while True:
            loc = self._loc()
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
                    default = self._collect_param_value(stop_comma=True)

            from_range = None
            value_ranges = []
            while self.at('from') or self.at('exclude'):
                if self.at('from'):
                    self.advance()
                    from_range = self._collect_param_value(stop_comma=True)
                    value_ranges.append(('from', from_range))
                elif self.at('exclude'):
                    self.advance()
                    value_ranges.append(('exclude', self._collect_param_value(stop_comma=True)))
            ptyp = typ
            if ptyp is None:
                ptyp = self._untyped_param_type(default, items)
            items.append((name, ptyp, default, from_range, value_ranges, array_size, elements, loc))
            if self.at(','):
                self.advance()
                continue
            break

        # Capture trailing inline `attr(...) / (* ... *) blocks — this
        # is where BSIM-CMG / MVS / FBH-HBT put type="instance".
        trailing = self._consume_inline_attrs()
        if trailing:
            # Trailing attrs override leading-attr defaults, matching
            # the document order a human would read.
            attrs = {**attrs, **trailing}

        self.expect(';')

        out = []
        for name, ptyp, default, from_range, value_ranges, array_size, elements, loc in items:
            p = Param(name=name, type=ptyp, default=parse_number(default), from_range=from_range,
                      loc=loc, value_ranges=value_ranges)
            if array_size is not None:
                p.array_size = array_size
                p.elements = elements or []
            if attrs.get('type') == 'instance':
                p.is_instance = True
            if _also_model(attrs):
                p.also_model = True
            if 'desc' in attrs:
                p.desc = attrs['desc']
            out.append(p)
        return out

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
        for name, init, loc in self._parse_decl_names(typ, dims=True, init=True):
            mod.variables.append(Var(name=name, type=typ, init=init, loc=loc))

    # --- Analog block parsing ---

    def _parse_analog(self) -> ASTNode:
        self.expect('analog')
        return self._parse_statement()

    def _parse_statement(self) -> ASTNode:
        # Attribute instances may precede any analog statement (LRM
        # analog_statement ::= { attribute_instance } ...).  They carry no
        # behaviour: consume and drop them, then parse the statement.  Left
        # in place, the fallback below read ``(* ... *)`` and the statement
        # after it as one expression statement and dropped both.
        if self.at('(*'):
            while self.at('(*'):
                self._parse_attributes()
            t = self.peek()
            if t is None or t.value in ('end', 'endmodule', 'endcase', 'endfunction', 'else'):
                raise self._error(
                    f"an attribute instance (* ... *) must be followed "
                    f"by a statement, found '{t.value if t else 'end of file'}'",
                    self._loc() if t else self._eof_loc())
        if self.at('begin'):
            return self._parse_begin_end()
        elif self.at('if'):
            return self._parse_if()
        elif self.at('@'):
            return self._parse_event()
        elif self.at('case', 'casex', 'casez'):
            return self._parse_case()
        elif self.at('for'):
            return self._parse_for()
        elif self.at('while'):
            return self._parse_while()
        elif self.at('repeat'):
            return self._parse_repeat()
        elif self.at('real', 'integer') and getattr(self, '_cur_mod', None) is not None:
            return self._parse_block_variables()
        else:
            return self._parse_simple_statement()

    def _parse_block_variables(self) -> ASTNode:
        """``real t, u = e;`` inside a (named) block of the analog block: the
        LRM's block-local variables.  They become variables of the module (a
        name declared twice is one variable: PyMS has one scope per module)
        and an initializer an assignment at the point of the declaration.
        Before, the declaration was an expression statement and was dropped."""
        loc = self._loc()
        typ = self.advance().value
        block = ASTNode(kind=NodeKind.BLOCK, loc=loc)
        items = self._parse_decl_names(typ, dims=True, init=True)
        func = getattr(self, '_cur_func', None)
        scopes = getattr(self, '_scopes', None)
        if func is not None:
            # A local of an analog function (its body has its own name space).
            # LRM 6.8 scoping inside the body too: a block-local whose name is
            # taken in the function (an argument, the function name, a local,
            # an enclosing or an earlier block's local) is a variable of its
            # own under a unique name, with its own type, and the names inside
            # the block are renamed to it.  Before 1.1.4 the function path
            # merged every block-local into one function variable of the first
            # declaration's type (s4_func_two_blocks_types: real t and integer t
            # one real, 0.518867924 for 0.5).
            for name, init, nloc in items:
                internal = name
                if scopes:
                    if name in scopes[-1]:
                        internal = scopes[-1][name]
                    elif self._func_name_taken(func, name):
                        internal = self._unique_func_local(func, name)
                        func.variables.append(Var(name=internal, type=typ, loc=nloc))
                    else:
                        func.variables.append(Var(name=name, type=typ, loc=nloc))
                    scopes[-1][name] = internal
                elif not self._func_name_taken(func, name):
                    func.variables.append(Var(name=name, type=typ, loc=nloc))
                if init is not None:
                    block.children.append(ASTNode(kind=NodeKind.ASSIGN, lhs=internal, expr=init,
                                                  loc=nloc, decl_init='block'))
            return block
        for name, init, nloc in items:
            internal = name
            if scopes:
                # LRM 6.8 scoping: a block-local is a variable of its own
                # block.  A name that is already taken (a module variable,
                # parameter, node, an enclosing block's local or another
                # block's) gets a variable of its own under a unique internal
                # name; references inside the block are renamed to it.
                # (Before, every block-local merged into one module variable
                # of the first declaration's type.)
                if name in scopes[-1]:
                    internal = scopes[-1][name]
                elif self._name_taken(name):
                    internal = self._unique_local(name)
                    self._cur_mod.variables.append(Var(name=internal, type=typ, loc=nloc))
                else:
                    self._cur_mod.variables.append(Var(name=name, type=typ, loc=nloc))
                scopes[-1][name] = internal
            elif name not in {v.name for v in self._cur_mod.variables}:
                self._cur_mod.variables.append(Var(name=name, type=typ, loc=nloc))
            if init is not None:
                block.children.append(ASTNode(kind=NodeKind.ASSIGN, lhs=internal, expr=init,
                                              loc=nloc, decl_init='block'))
        return block

    @staticmethod
    def _func_name_taken(func, name: str) -> bool:
        return (name == func.name or any(a == name for a, _d in func.args)
                or any(v.name == name for v in func.variables))

    def _unique_func_local(self, func, name: str) -> str:
        label = self._scope_labels[-1] if getattr(self, '_scope_labels', None) else ''
        while True:
            self._scope_seq = getattr(self, '_scope_seq', 0) + 1
            cand = f'{name}__{label or "blk"}{self._scope_seq}'
            if not self._func_name_taken(func, cand) and not any(cand in s for s in self._scopes):
                return cand

    def _name_taken(self, name: str) -> bool:
        mod = self._cur_mod
        if any(v.name == name for v in mod.variables):
            return True
        if any(p.name == name for p in mod.params) or any(p.name == name for p in mod.ports):
            return True
        if name in mod.internal_nodes or name in mod.genvars or name in mod.grounds:
            return True
        if any(f.name == name for f in mod.functions):
            return True
        return any(name in s for s in self._scopes)

    def _unique_local(self, name: str) -> str:
        label = self._scope_labels[-1] if getattr(self, '_scope_labels', None) else ''
        while True:
            self._scope_seq += 1
            cand = f'{name}__{label or "blk"}{self._scope_seq}'
            if not self._name_taken(cand):
                return cand

    def _scoped(self, name: str) -> str:
        """The variable a name written inside the analog block refers to:
        the innermost block-local of that name, else the name itself."""
        for s in reversed(getattr(self, '_scopes', None) or ()):
            if name in s:
                return s[name]
        return name

    def _parse_begin_end(self) -> ASTNode:
        loc = self._loc()
        self.expect('begin')
        # Optional named block: begin : label
        label = ''
        if self.at(':'):
            self.advance()
            label = self.consume_ident()  # block label
        if getattr(self, '_scopes', None) is None:
            self._scopes, self._scope_seq = [], 0
        if getattr(self, '_scope_labels', None) is None:
            self._scope_labels = []
        self._scopes.append({})
        self._scope_labels.append(label)
        try:
            return self._parse_begin_end_body(loc)
        finally:
            self._scopes.pop()
            self._scope_labels.pop()

    def _parse_begin_end_body(self, loc) -> ASTNode:
        block = ASTNode(kind=NodeKind.BLOCK, loc=loc)
        # Terminate on ``end`` (normal) OR on a higher-level structural
        # keyword (``endmodule``, ``endcase``, ``endfunction``) — those
        # mean the source has a missing ``end`` somewhere; bail out
        # rather than recursing into a statement that would gobble them
        # and run to EOF.
        STRUCT_END = ('end', 'endmodule', 'endcase', 'endfunction')
        while self.pos < len(self.tokens):
            t = self.peek()
            if t is None or t.value in STRUCT_END:
                break
            block.children.append(self._parse_statement())
        # Allow ``endmodule``/``endcase`` to act as an implicit ``end``
        # when the source is missing one — emit the block we have and
        # leave the keyword for the enclosing parser to consume.
        if self.at('end'):
            self.advance()
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
        """``@ ( event_expression ) statement`` (LRM 5.10): the event
        expression is one or more events separated by ``or``, each a name
        with an optional argument list (``initial_step``, ``initial_step("ac",
        "tran")``, ``final_step``, ``cross(V(p, n) - 0.5, +1)``, ``above(e)``,
        ``timer(t0, period)``).  The result is an INITIAL_STEP node (an
        event-controlled statement) whose ``event`` lists them; the emitter
        decides, per analysis, whether and when the body runs.  Before 1.1.4
        the arguments were dropped and every event but initial_step ran its
        body at every evaluation (an @(final_step) or @(cross(...)) body as
        an ordinary statement)."""
        loc = self._loc()
        self.expect('@')
        self.expect('(')
        events = []
        while True:
            name = self.consume_ident()
            args = []
            if self.at('('):
                self.advance()
                args = [a for a in _split_args(self._collect_balanced('(', ')')) if a != '']
                self.expect(')')
            events.append({'kind': name, 'args': args})
            if self.at('or'):
                self.advance()
                continue
            break
        self.expect(')')
        body = self._parse_statement()
        return ASTNode(kind=NodeKind.INITIAL_STEP, children=[body], event=events, loc=loc)

    def _parse_case(self) -> ASTNode:
        """case (expr) item {item} endcase, as the LRM defines it
        (analog_case_item ::= expr {, expr} : statement_or_null
        | default [:] statement_or_null): the items are compared with the
        case expression in source order, the FIRST arm with a matching
        expression runs and no other (no fall-through), and the default arm
        -- wherever it is written -- runs when no item matches.  Each arm is
        ONE statement (a begin ... end block for more).  The result is the
        equivalent if / else if / else chain, so contributions, shorts and
        branch kinds per arm behave exactly as they do under if/else."""
        loc = self._loc()
        kw = self.advance().value          # case, casex or casez
        self.expect('(')
        raw_expr = self._collect_balanced('(', ')')
        case_expr = self._case_operand(raw_expr)
        self.expect(')')
        if kw != 'case':
            self._check_casexz_operand(kw, raw_expr, loc)
        # The case expression is evaluated ONCE (LRM 5.8: it is compared with
        # each item in turn): an expression with a call that is not a pure
        # builtin is stored in a temporary first, so a side effect (an analog
        # function's output argument, $random) happens once, not once per item.
        hoist = None
        if self._has_impure_call(raw_expr):
            _ternary_counter[0] += 1
            tmp = f'_vae_tcase{_ternary_counter[0]}'
            hoist = ASTNode(kind=NodeKind.ASSIGN, lhs=tmp, expr=raw_expr, loc=loc)
            case_expr = tmp
        arms, default = [], None
        while not self.at('endcase'):
            t = self.peek()
            if t is None:
                raise self._error("'case' without 'endcase'", self._eof_loc())
            aloc = self._loc()
            if self.at('default'):
                if default is not None:
                    raise self._error("a case statement has more than one default arm", aloc)
                self.advance()
                if self.at(':'):
                    self.advance()
                default = self._parse_case_arm()
                continue
            values = []
            while True:
                v = self._collect_case_item()
                if not v:
                    raise self._error(
                        f"expected a case item expression, got "
                        f"'{self.peek().value if self.peek() else 'end of file'}'",
                        self._loc() if self.peek() else self._eof_loc())
                if kw != 'case':
                    self._check_casexz_operand(kw, v, aloc)
                values.append(v)
                if self.at(','):
                    self.advance()
                    continue
                self.expect(':')
                break
            cond = ' || '.join(f'{case_expr} == {self._case_operand(v)}' for v in values)
            arms.append((cond, self._parse_case_arm(), aloc))
        self.expect('endcase')
        node = default
        for cond, body, aloc in reversed(arms):
            node = ASTNode(kind=NodeKind.IF, condition=cond, children=[body],
                           else_body=node, loc=aloc)
        if node is None:
            node = ASTNode(kind=NodeKind.BLOCK, loc=loc)
        if hoist is not None:
            node = ASTNode(kind=NodeKind.BLOCK, children=[hoist, node], loc=loc)
        return node

    # The builtin functions of an expression that have no side effect (a
    # case expression made of them is not hoisted).
    _PURE_CALLS = frozenset((
        'V', 'I', 'ln', 'log', 'exp', 'sqrt', 'pow', 'abs', 'min', 'max', 'floor', 'ceil',
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh',
        'asinh', 'acosh', 'atanh', 'hypot', 'limexp', 'ddt', 'idt', 'ddx', 'white_noise',
        'flicker_noise', '$simparam', '$param_given', '$port_connected', '$vt',
        '$temperature', '$mfactor', '$abstime', '$limit'))

    def _has_impure_call(self, text: str) -> bool:
        for m in re.finditer(r'([$`]?[A-Za-z_]\w*)\s*\(', _strip_strings(text)):
            if m.group(1) not in self._PURE_CALLS and m.group(1) not in self.module_accesses:
                return True
        return bool(re.search(r'\$(random|arandom|rdist_\w+|dist_\w+)\b', text))

    def _check_casexz_operand(self, kw: str, text: str, loc):
        """casex / casez (LRM 5.8.2 / Verilog-AMS analog_case_statement): x
        and z (casez: z and ?) digits of a based literal are don't-care bits.
        An analog value has no x or z bits, so an item without such digits
        compares exactly as in ``case`` -- which is what PyMS builds.  An item
        or expression WITH x / z / ? digits would need a bit-mask compare,
        which PyMS does not build: a file:line error, never a silent 0."""
        for m in re.finditer(r"'[sS]?[hHdDoObB]([0-9a-fA-F_xXzZ?]+)", text):
            if re.search(r'[xXzZ?]', m.group(1)):
                raise self._error(
                    f"{kw}: the item {' '.join(text.split())!r} has x/z/? digits (don't-care bits); "
                    f"PyMS does not support don't-care bits in a {kw} statement "
                    f"(an item without x/z digits works as in case)", loc)

    def _parse_case_arm(self) -> ASTNode:
        """The one statement (or null statement ``;``) of a case arm."""
        if self.at(';'):
            loc = self._loc()
            self.advance()
            return ASTNode(kind=NodeKind.BLOCK, loc=loc)
        return self._parse_statement()

    def _collect_case_item(self) -> str:
        """One case item expression: the tokens up to the ``,`` or ``:``
        that ends it at depth 0 (a ``:`` that closes a ``? :`` inside the
        item is part of it)."""
        parts = []
        depth = quest = 0
        while self.pos < len(self.tokens):
            t = self.peek()
            if depth == 0:
                if t.value == ',' or (t.value == ':' and quest == 0):
                    break
                if t.value in (';', 'endcase', 'begin', 'end'):
                    break
            if t.value in ('(', '['):
                depth += 1
            elif t.value in (')', ']'):
                depth -= 1
            elif t.value == '?' and depth == 0:
                quest += 1
            elif t.value == ':' and depth == 0:
                quest -= 1
            parts.append(self._expr_token(t))
            self.advance()
        return ' '.join(parts)

    @staticmethod
    def _case_operand(text: str) -> str:
        """A case expression or item as an operand of ``==``: parenthesized
        unless it is one name, number or string literal (a string operand
        stays bare: the string comparison of a condition reads it so)."""
        s = text.strip()
        if re.fullmatch(r'[\w.$`]+|"(?:[^"\\]|\\.)*"', s):
            return s
        return f'({s})'

    # Loops (LRM 5.10: repeat, while, for).  Before 1.1.3 the header was
    # skipped and the body kept as one pass: a loop ran once whatever its
    # bounds (a zero-trip loop too).  Now a loop is a BLOCK [init?, body,
    # step?] with node.loop set, and GiNaCEmitter._walk_loop iterates it.

    def _loop_assign(self, *stops) -> ASTNode:
        """The init or step of a for loop: ``name = expr``."""
        loc = self._loc()
        lhs = self._scoped(self.consume_ident())
        self.expect('=')
        expr = self._collect_expr_until(*stops)
        if not expr.strip():
            raise self._error(f"for loop: '{lhs} =' has no value", loc)
        return ASTNode(kind=NodeKind.ASSIGN, lhs=lhs, expr=expr, loc=loc)

    @staticmethod
    def _loop_node(kind, loc, cond, init, body, step) -> ASTNode:
        children = [c for c in (init, body, step) if c is not None]
        return ASTNode(kind=NodeKind.BLOCK, children=children, loc=loc,
                       loop=dict(kind=kind, cond=cond, has_init=init is not None,
                                 has_step=step is not None))

    def _parse_for(self) -> ASTNode:
        """for ( name = init ; cond ; name = step ) statement"""
        loc = self._loc()
        self.expect('for')
        self.expect('(')
        init = self._loop_assign(';')
        self.expect(';')
        cond = self._collect_expr_until(';')
        if not cond.strip():
            raise self._error("for loop without a condition", loc)
        self.expect(';')
        step = self._loop_assign(')')
        self.expect(')')
        body = self._parse_statement()
        return self._loop_node('for', loc, cond, init, body, step)

    def _parse_while(self) -> ASTNode:
        """while ( cond ) statement"""
        loc = self._loc()
        self.expect('while')
        self.expect('(')
        cond = self._collect_balanced('(', ')')
        self.expect(')')
        body = self._parse_statement()
        return self._loop_node('while', loc, cond, None, body, None)

    def _parse_repeat(self) -> ASTNode:
        """repeat ( count ) statement: the count is evaluated once."""
        loc = self._loc()
        self.expect('repeat')
        self.expect('(')
        count = self._collect_balanced('(', ')')
        self.expect(')')
        body = self._parse_statement()
        return self._loop_node('repeat', loc, count, None, body, None)

    def _parse_simple_statement(self) -> ASTNode:
        # Look ahead to determine statement type
        # Could be: contribution (I/V <+), assignment (var = expr), or call

        # Check for contribution: I(a,b) <+ or V(a,b) <+
        t = self.peek()
        if t and t.value in self.module_accesses and self.peek(1) and self.peek(1).value == '(':
            return self._parse_contribution()

        # Assignment: ident = expr ;
        if t and t.type == 'IDENT':
            t2 = self.peek(1)
            if t2 and t2.value == '=':
                loc = self._loc()
                lhs = self._scoped(self.consume_ident())
                self.expect('=')
                expr = self._collect_expr_until(';')
                self.expect(';')
                return ASTNode(kind=NodeKind.ASSIGN, lhs=lhs, expr=expr, loc=loc)

        # Fallback — collect to semicolon
        loc = self._loc()
        text = self._collect_expr_until(';')
        self.expect(';')
        if '<+' in text.split():
            # A contribution through a name that is not an access function of
            # a discipline this module declares nodes with (Pwr(t) with no
            # thermal node): an error, not a dropped statement.
            lhs = text.split('<+', 1)[0].strip()
            raise self._error(
                f"contribution '{lhs} <+ ...': {lhs.split('(')[0].strip()!r} is not "
                f"the access function of a discipline declared in this module", loc)
        return ASTNode(kind=NodeKind.EXPR, text=text)

    def _parse_contribution(self) -> ASTNode:
        loc = self._loc()
        kind_str = self.advance().value  # I, V or another discipline's access function
        ckind = self.access_kinds[kind_str]

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

    def _collect_param_value(self, stop_comma: bool = False) -> str:
        """Collect parameter default or range value, stopping at ;, from, exclude, or `attr(...)
        (and, with stop_comma, at the ',' that ends an item of a declaration list)."""
        parts = []
        depth = 0
        braces = 0   # (a value set '{"fast", "slow"} holds commas)
        while self.pos < len(self.tokens):
            t = self.peek()
            if depth == 0:
                if t.value in (';', 'from', 'exclude'):
                    break
                if stop_comma and t.value == ',' and not braces:
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
            elif t.value == '{':
                braces += 1
            elif t.value == '}' and braces:
                braces -= 1
            parts.append(t.value)
            self.advance()
        return ' '.join(parts)

    def _collect_expr_until(self, *stops) -> str:
        """Collect tokens into expression string until one of stops is seen at depth 0.

        Depth is clamped at ≥0 so a stray closing bracket (from an earlier
        mis-nested parse) doesn't let the loop swallow the rest of the file
        by making every subsequent `;` appear to be at non-zero depth.

        Structural keywords (``end``, ``endmodule``, ``endcase``,
        ``endfunction``) also terminate the collector even when not in
        ``stops`` — they can never appear inside an expression, and
        without this guard a missing ``;`` at the end of an analog block
        statement lets us run to ``EOF`` (BSIM-SOI 4.5/4.6.1/4.7).
        """
        parts = []
        depth = 0
        STRUCT_END = ('end', 'endmodule', 'endcase', 'endfunction')
        while self.pos < len(self.tokens):
            t = self.peek()
            if depth == 0:
                if t.value in stops:
                    break
                if t.value in STRUCT_END:
                    break
            if t.value == '(*':
                # Attribute instances at the end of a statement, before its
                # ``;`` (HICUM L0: ``I(br_sci) <+ ijsc (*spectre:gmin="add"*);``)
                # are dropped; anywhere else in an expression they used to
                # become part of its text (a g++ error of the math library,
                # or a contribution of 0 before that) and are an error.
                loc = self._loc()
                while self.at('(*'):
                    self._parse_attributes()
                if depth == 0 and self.peek() is not None and self.peek().value in stops:
                    continue
                raise self._error(
                    "an attribute instance (* ... *) inside an expression "
                    "is not supported", loc)
            if t.value in ('(', '['):
                depth += 1
            elif t.value in (')', ']'):
                if depth > 0:
                    depth -= 1
            parts.append(self._expr_token(t))
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
                parts.append(self._expr_token(t))
                self.advance()
        return ' '.join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse_all_entities(source: str, filename: str = ""):
    """Parse every top-level ``module`` and ``paramset`` in source.

    Returns a list whose items are Module or Paramset instances, in
    the order they appear. Walks the token stream skipping preamble,
    parsing one entity, advancing, repeating. The Parser instance is
    advanced cooperatively (each parse_module / parse_paramset leaves
    the stream pointed past its ``endmodule`` / ``endparamset``).

    Also collects file-level ``parameter NAME = VALUE;`` declarations
    that sit between entities. gnucap2xyce.pl emits corner-derived
    constants this way at the head of each per-paramset .va file
    (e.g. ``parameter real rsh_rsil = 7.0;``). Without picking these
    up, paramset bindings that reference them by name resolve to 0
    at codegen time and break the simple-R stamp's R value. The
    collected dict is attached to the LAST paramset / module as
    ``_file_level_params`` so downstream code can use it for
    identifier-default substitution.
    """
    # The ternary temporaries (_vae_t<N>) are numbered per parse: a counter
    # carried over from an earlier parse in the same process renamed them, so
    # the emitted program depended on what the process had parsed before
    # (PSP102/103, BSIM6 in a corpus scan; parsing one file twice differed).
    _ternary_counter[0] = 0
    tokens = tokenize(source)
    parser = Parser(tokens, source=source, filename=filename)
    entities = []
    file_level_params = {}
    while parser.pos < len(parser.tokens):
        parser._skip_preamble()
        if parser.pos >= len(parser.tokens):
            break
        t = parser.peek()
        if t is None:
            break
        # Top-level ``parameter [real|integer] NAME = VALUE;``.
        if t.value == 'parameter':
            parser.advance()
            # optional type keyword
            tt = parser.peek()
            if tt and tt.value in ('real', 'integer'):
                parser.advance()
            name_tok = parser.peek()
            if name_tok and name_tok.type == 'IDENT':
                pname = name_tok.value
                parser.advance()
                if parser.at('='):
                    parser.advance()
                    val_tokens = []
                    while parser.peek() and parser.peek().value != ';':
                        val_tokens.append(parser.peek().value)
                        parser.advance()
                    if parser.at(';'):
                        parser.advance()
                    file_level_params[pname] = ' '.join(val_tokens).strip()
                else:
                    # Malformed; skip until semicolon.
                    while parser.peek() and parser.peek().value != ';':
                        parser.advance()
                    if parser.at(';'):
                        parser.advance()
            else:
                parser.advance()
            continue
        if t.value == 'paramset' or (t.value == '(*'):
            # ``(* attrs *) [paramset|module] NAME ...``  Peek past attrs
            # to decide which.
            save = parser.pos
            if t.value == '(*':
                parser._parse_attributes()
                tnxt = parser.peek()
                parser.pos = save
                if tnxt and tnxt.value == 'paramset':
                    entities.append(parser.parse_paramset())
                    continue
                # else fall through to module
            else:
                entities.append(parser.parse_paramset())
                continue
        if parser.peek() and parser.peek().value == 'module':
            mod = parser.parse_module()
            for f in mod.functions:
                f.body = _lower_ternaries(f.body)
            if mod.analog_block:
                mod.analog_block = _lower_ternaries(mod.analog_block)
                _inline_analog_functions(mod)
            entities.append(mod)
            continue
        # Unrecognised top-level token — advance to avoid an infinite
        # loop on garbage between entities.
        parser.advance()
    if file_level_params and entities:
        entities[-1]._file_level_params = file_level_params
    return entities


def parse_verilog_a(source: str, filename: str = "", module: Optional[str] = None) -> Module:
    """Parse a Verilog-A source string, return the primary Module.

    If the source declares any paramsets, each is resolved against its
    underlying module (also expected to live in the same compilation
    unit) by overriding the underlying's parameter defaults — a paramset
    is conceptually just a defaults-rewrite layer, not a new module.

    When the file has a paramset, the resolved Module is returned. The
    plain-module case is unchanged: returns the first ``module``.
    `module`: the name of the module (or paramset) to return instead, the
    one a netlist names (a .va/.hdl file with several modules: each is a
    device of its own, xyce_device_gen.py builds every one).
    """
    entities = _parse_all_entities(source, filename=filename)
    if not entities:
        raise ParseError(f"{filename or '<source>'}: no module found in file "
                         f"(macro/include-only file?)")
    if module is not None:
        return _select_named(entities, module, filename)
    return _select_primary(entities, filename)


def parse_all_modules(source: str, filename: str = "") -> list:
    """(primary, [the other modules of the file]): every ``module`` of a
    file that is a device of its own.  A paramset file keeps its paramset as
    the one device (its modules are its underlying ones)."""
    entities = _parse_all_entities(source, filename=filename)
    if not entities:
        raise ParseError(f"{filename or '<source>'}: no module found in file "
                         f"(macro/include-only file?)")
    primary = _select_primary(entities, filename)
    if any(isinstance(e, Paramset) for e in entities):
        return primary, []
    others = [e for e in entities if isinstance(e, Module) and e.name != primary.name]
    return primary, others


def _select_named(entities, name, filename):
    modules = {e.name: e for e in entities if isinstance(e, Module)}
    for e in entities:
        if isinstance(e, Paramset) and e.name == name:
            underlying = modules.get(e.underlying_name)
            if underlying is None:
                raise ParseError(f"paramset {e.name} references unknown module "
                                 f"{e.underlying_name!r}")
            return _resolve_paramset(e, underlying)
    if name in modules:
        return modules[name]
    raise ParseError(f"{filename or '<source>'}: no module or paramset named {name!r} in the file "
                     f"(it has {', '.join(e.name for e in entities)})")


def _select_primary(entities, filename):
    """Pick the "primary" entity and resolve paramsets into Modules.

    Priority:
      1. The first ``Paramset`` in the file (a paramset_file.va is
         typically a wrapper that ``\\`include``s its underlying — the
         paramset is the entry point).
      2. The first ``Module``.
    """
    # Build a name → Module index for paramset resolution
    modules = {e.name: e for e in entities if isinstance(e, Module)}
    paramsets = [e for e in entities if isinstance(e, Paramset)]
    if paramsets:
        ps = paramsets[0]
        underlying = modules.get(ps.underlying_name)
        if underlying is None:
            raise ParseError(
                f"paramset {ps.name} references unknown module "
                f"{ps.underlying_name!r}; expected in same .va or via "
                f"`include")
        return _resolve_paramset(ps, underlying)
    for e in entities:
        if isinstance(e, Module):
            return e
    raise ParseError(f"{filename or '<source>'}: no module or paramset found")


def _resolve_paramset(ps: Paramset, underlying: Module) -> Module:
    """Apply paramset defaults-overrides to the underlying Module.

    Returns a NEW Module (does not mutate the underlying):
      * name        = paramset's name
      * ports       = underlying's ports
      * variables   = underlying's variables
      * branch_map  = underlying's branch_map
      * internal_nodes = underlying's
      * analog_block = underlying's analog_block, prefixed with
        assignment statements for each paramset localparam so the
        rebound default expressions resolve at evaluation time
      * params      = paramset.params (instance params, declared
        first so binding defaults can reference them) + the
        underlying's params with bindings substituted as defaults
        + paramset.localparams emitted as derived parameters
      * attributes  = underlying ∪ paramset (paramset overrides)
    """
    # New params list:
    #   1. paramset's own instance params (visible to binding defaults)
    #   2. paramset's localparams as derived ``parameter real`` decls
    #      (Verilog-A allows parameter defaults to reference earlier
    #      parameters, so this preserves the localparam semantics
    #      while keeping the AST a plain Module)
    #   3. underlying's params with any binding rewrites applied
    new_params = []
    seen = set()
    for p in ps.params:
        if p.name in seen:
            continue
        new_params.append(p)
        seen.add(p.name)
    for lp_name, lp_expr in ps.localparams:
        if lp_name in seen:
            continue
        new_params.append(Param(name=lp_name, type='real',
                                default=lp_expr))
        seen.add(lp_name)
    binding_map = {b[0]: b[1] for b in ps.bindings}
    # File-level parameter declarations sitting between modules
    # (typical for gnucap2xyce-generated paramset .va files: a list
    # of corner-derived constants). Used to resolve binding RHS
    # identifiers that don't refer to the paramset's own params.
    file_level = getattr(ps, '_file_level_params', None) or {}
    for p in underlying.params:
        if p.name in seen:
            continue
        new_default = binding_map.get(p.name, p.default)
        # If the binding RHS is a bare identifier we know the
        # numeric default for (from file-level decls), substitute
        # it. Without this, the wrapper's addPar default for the
        # underlying param falls back to 0.
        if isinstance(new_default, str):
            stripped = new_default.strip()
            if stripped in file_level:
                new_default = file_level[stripped]
        new_params.append(Param(
            name=p.name, type=p.type, default=new_default,
            from_range=p.from_range,
            value_ranges=list(getattr(p, 'value_ranges', []) or []),
            is_instance=p.is_instance,
            desc=p.desc,
            array_size=p.array_size,
            elements=p.elements,
            loc=p.loc,
            also_model=p.also_model,
        ))
        seen.add(p.name)

    merged_attrs = dict(underlying.attributes)
    merged_attrs.update(ps.attributes)

    return Module(
        name=ps.name,
        ports=list(underlying.ports),
        params=new_params,
        variables=list(underlying.variables),
        internal_nodes=list(underlying.internal_nodes),
        branch_map=dict(underlying.branch_map),
        branch_neg_map=dict(underlying.branch_neg_map),
        analog_block=underlying.analog_block,
        attributes=merged_attrs,
        grounds=list(underlying.grounds),
        genvars=list(underlying.genvars),
        functions=list(underlying.functions),
    )


def parse_file_all(path: str, defines: Optional[list[str]] = None):
    """parse_all_modules of a file (after the preprocessor)."""
    from .preprocess import preprocess_file
    source = preprocess_file(path, defines=defines)
    return parse_all_modules(source, filename=path)


def parse_file(path: str, defines: Optional[list[str]] = None,
               module: Optional[str] = None) -> Module:
    """Parse a Verilog-A file, return Module AST.  `defines`: command-line
    macros (``NAME`` or ``NAME=VALUE``), see preprocess.parse_define_option.

    Runs the textual preprocessor (`\\`include` follow, `\\`define`
    expansion, `\\`ifdef`/`\\`ifndef`) before tokenizing — every real
    compact-model .va in /usr/local/share/xyce/verilog-a defines its
    parameters via macros (`\\`IPRnb`, `\\`MPRcc`, etc.) and brings them
    in via `\\`include`, so without preprocessing the parser sees no
    parameters at all.

    Paramsets in the file are resolved by overriding the underlying
    module's parameter defaults — the returned Module is the paramset
    name with the underlying's analog body and bindings-as-defaults.
    """
    from .preprocess import preprocess_file
    source = preprocess_file(path, defines=defines)
    return parse_verilog_a(source, filename=path, module=module)


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


# ---------------------------------------------------------------------------
# Analog function calls (LRM 4.7) -- inlined into the calling statement
# ---------------------------------------------------------------------------

# Function names PyMS translates itself (ginac_emitter._VA_TO_GINAC and its
# helpers): a module's analog function of one of these names keeps PyMS's
# builtin (lexp / lln / hypsmooth of the CMC models), as before 1.1.3.
_BUILTIN_FUNCS = frozenset({
    'ln', 'limexp', 'lexp', 'lln', 'abs', 'pow', 'exp', 'log', 'sqrt', 'sin', 'cos',
    'tan', 'tanh', 'atan', 'atan2', 'hypot', 'ddt', 'idt', 'min', 'max', 'hypsmooth',
    'hypmax', 'Tempdep', 'floor', 'ceil', 'asin', 'acos', 'sinh', 'cosh', 'asinh',
    'acosh', 'atanh', 'ddx', 'white_noise', 'flicker_noise', 'noise_table', 'V', 'I'})


def _clone(node: Optional[ASTNode], ren: dict) -> Optional[ASTNode]:
    """A deep copy of a statement tree with the identifiers of `ren` renamed."""
    if node is None:
        return None
    pat = ren.get('__re__')

    def t(s):
        if not s or pat is None:
            return s
        parts = re.split(r'("(?:[^"\\]|\\.)*")', s)
        return ''.join(p if i % 2 else pat.sub(lambda m: ren[m.group(0)], p)
                       for i, p in enumerate(parts))
    n = ASTNode(kind=node.kind, text=t(node.text), contrib_kind=node.contrib_kind,
                branch=node.branch, expr=t(node.expr), condition=t(node.condition),
                lhs=ren.get(node.lhs, node.lhs) if node.lhs else node.lhs, loc=node.loc,
                int_lhs=node.int_lhs, decl_init=node.decl_init)
    if node.event is not None:
        n.event = [{'kind': e['kind'], 'args': [t(a) for a in e['args']]} for e in node.event]
    if node.loop is not None:
        n.loop = dict(node.loop)
        n.loop['cond'] = t(n.loop.get('cond'))
    n.children = [_clone(c, ren) for c in node.children]
    n.else_body = _clone(node.else_body, ren)
    return n


def _iter_nodes(node):
    """Every statement node below node (IF arms too), depth first."""
    if node is None:
        return
    yield node
    for c in node.children or ():
        yield from _iter_nodes(c)
    yield from _iter_nodes(node.else_body)


def _split_args(text: str) -> list:
    out, depth, cur, in_str = [], 0, [], False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            cur.append(ch)
            if ch == '\\' and i + 1 < len(text):
                cur.append(text[i + 1])
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            cur.append(ch)
        elif ch in '([{':
            depth += 1
            cur.append(ch)
        elif ch in ')]}':
            depth -= 1
            cur.append(ch)
        elif ch == ',' and depth == 0:
            out.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    last = ''.join(cur).strip()
    if last or out:
        out.append(last)
    return out


class _FunctionInliner:
    """Every call of a module's analog function in a statement of the
    analog block becomes, before that statement: the input (and inout)
    arguments assigned to fresh variables, a copy of the body with the
    arguments, the locals and the function name renamed to fresh variables,
    and the output (and inout) arguments assigned back to the caller's
    variables; the call itself reads the fresh return variable.  So a call
    is evaluated once per execution of its statement, in source order, and
    an output argument is a real assignment.  Before 1.1.3 analog functions
    were skipped by the parser and a call reached the generated C++ as an
    undeclared function (no math library: a netlist error)."""

    MAX_DEPTH = 64

    def __init__(self, mod: Module):
        self.mod = mod
        self.funcs = {f.name: f for f in mod.functions if f.name not in _BUILTIN_FUNCS}
        self.k = 0
        self.depth = 0
        self.call_re = (re.compile(r'(?<![\w$`.])(' + '|'.join(re.escape(n) for n in
                                   sorted(self.funcs, key=len, reverse=True)) + r')\s*\(')
                        if self.funcs else None)
        self._check_recursion()

    def _check_recursion(self):
        """An analog function may not call itself, directly or through other
        functions (LRM 4.7.1): a clear file:line error.  Before 1.1.4 the
        inliner renamed the recursive call with the copy's locals, and the
        call reached g++ as an undeclared function."""
        if self.call_re is None:
            return
        calls = {}
        for name, f in self.funcs.items():
            out = []
            for st in _iter_nodes(f.body):
                for text in (st.expr, st.condition, (st.loop or {}).get('cond')):
                    if text:
                        masked = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)
                        out.extend(m.group(1) for m in self.call_re.finditer(masked))
            calls[name] = list(dict.fromkeys(out))
        state = {}

        def visit(n, path):
            state[n] = 1
            for m in calls.get(n, ()):
                if state.get(m) == 1:
                    cyc = path[path.index(m):] + [m] if m in path else [n, m]
                    f = self.funcs[cyc[0]]
                    via = ' -> '.join(cyc)
                    raise ParseError(f"{f.loc}: analog function {cyc[0]} is recursive ({via}): an "
                                     f"analog function cannot call itself, directly or through "
                                     f"another function (LRM 4.7.1)")
                if state.get(m) is None:
                    visit(m, path + [m])
            state[n] = 2
        for n in self.funcs:
            if state.get(n) is None:
                visit(n, [n])

    def _calls(self, text):
        """(start, name, args-start, end) of the innermost user call in text."""
        if not text or self.call_re is None:
            return None
        masked = re.sub(r'"(?:[^"\\]|\\.)*"', lambda m: '"' + ' ' * (len(m.group(0)) - 2) + '"', text)
        best = None
        for m in self.call_re.finditer(masked):
            depth, j = 1, m.end()
            while j < len(masked) and depth:
                if masked[j] == '(':
                    depth += 1
                elif masked[j] == ')':
                    depth -= 1
                j += 1
            if depth:
                continue
            inner = masked[m.end():j - 1]
            if self.call_re.search(inner):
                continue           # an argument holds a call: take that first
            best = (m.start(), m.group(1), m.end(), j)
            break
        return best

    def _expand(self, name, args, loc):
        from .int_ops import analyse, module_types
        f = self.funcs[name]
        if len(args) != len(f.args):
            raise ParseError(f"{loc}: analog function {name} takes {len(f.args)} argument(s), "
                             f"called with {len(args)}")
        self.k += 1
        pfx = f'_vae_f{self.k}_'
        ftypes = {v.name: v.type for v in f.variables}
        names = [a for a, _ in f.args] + [v.name for v in f.variables] + [name]
        ren = {n: pfx + n for n in names}
        ren['__re__'] = re.compile(r'(?<![\w$`.])(' + '|'.join(re.escape(n) for n in
                                   sorted(set(names), key=len, reverse=True)) + r')(?!\w)')
        have = {v.name for v in self.mod.variables}
        for n in dict.fromkeys(names):
            typ = f.rtype if n == name else ftypes.get(n, 'real')
            if pfx + n not in have:
                self.mod.variables.append(Var(name=pfx + n, type=typ, loc=f.loc))
        types = module_types(self.mod)
        pre = []
        for (a, d), actual in zip(f.args, args):
            if d in ('input', 'inout'):
                _new, typ = analyse(actual, types)
                pre.append(ASTNode(kind=NodeKind.ASSIGN, lhs=pfx + a, expr=actual, loc=loc,
                                   int_lhs=(ftypes.get(a) == 'integer' and typ != 'int')))
        pre.append(_clone(f.body, ren))
        for (a, d), actual in zip(f.args, args):
            if d in ('output', 'inout'):
                tgt = actual.strip()
                if not re.fullmatch(r'[A-Za-z_]\w*', tgt):
                    raise ParseError(f"{loc}: the {d} argument {a} of analog function {name} "
                                     f"must be a variable, not '{tgt}'")
                pre.append(ASTNode(kind=NodeKind.ASSIGN, lhs=tgt, expr=pfx + a, loc=loc,
                                   int_lhs=(types.get(tgt) == 'int' and ftypes.get(a) != 'integer')))
        return pfx + name, pre

    def text(self, s, loc):
        pre = []
        while True:
            c = self._calls(s)
            if c is None:
                return s, pre
            start, name, a0, end = c
            args = _split_args(s[a0:end - 1])
            var, stmts = self._expand(name, args, loc)
            pre.extend(stmts)
            s = s[:start] + var + s[end:]

    def stmt(self, node):
        if node is None:
            return None
        self.depth += 1
        if self.depth > self.MAX_DEPTH:
            raise ParseError(f"{node.loc}: analog functions nested deeper than {self.MAX_DEPTH} "
                             f"calls (a recursive analog function?)")
        try:
            pre = []
            if node.kind == NodeKind.BLOCK:
                if node.loop is not None and self._calls(node.loop.get('cond')):
                    raise ParseError(f"{node.loc}: a call of an analog function in the condition "
                                     f"of a {node.loop['kind']} loop is not supported by PyMS "
                                     f"(assign it to a variable before the loop and at the end of "
                                     f"its body)")
                node.children = [self.stmt(c) for c in node.children]
            elif node.kind in (NodeKind.ASSIGN, NodeKind.CONTRIB):
                node.expr, pre = self.text(node.expr, node.loc)
            elif node.kind == NodeKind.IF:
                node.condition, pre = self.text(node.condition, node.loc)
                node.children = [self.stmt(c) for c in node.children]
                node.else_body = self.stmt(node.else_body)
            elif node.kind == NodeKind.INITIAL_STEP:
                node.children = [self.stmt(c) for c in node.children]
            if pre:
                pre = [self.stmt(p) for p in pre]
                return ASTNode(kind=NodeKind.BLOCK, children=pre + [node], loc=node.loc)
            return node
        finally:
            self.depth -= 1


def _inline_analog_functions(mod: Module):
    if not mod.analog_block:
        return
    inl = _FunctionInliner(mod)
    if inl.funcs:
        mod.analog_block = inl.stmt(mod.analog_block)
