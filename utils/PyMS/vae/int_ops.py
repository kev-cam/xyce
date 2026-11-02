"""Integer arithmetic of analog-block expressions (LRM 4.2: Verilog-A typing).

An operation on two integer operands is an integer operation: ``/``
truncates toward zero and ``%`` takes the sign of its first operand
(``7 / 2`` is 3, ``-7 % 3`` is -1), and a real value assigned to an integer
variable is rounded, halves away from zero (LRM 4.2.1.1).  The parameter
defaults have had these semantics since 1.1.0 (va_const); the statements of
the analog block were translated with C++ / GiNaC real arithmetic, so
``integer k = 3; t = k / 2;`` gave t = 1.5 (BUGS.md A8).

lower_integer_ops(module) types every expression of the analog block (and of
the analog functions) with the declared types of the parameters, variables
and genvars, and rewrites each integer ``a / b`` to ``_vae_idiv ( a , b )``
and each integer ``a % b`` to ``_vae_imod ( a , b )`` (and a based literal,
``'h5``, to its decimal value, which no translator read before); the emitter evaluates
those exactly (Python at elaboration, a GiNaC function in the metaprogram, a
C++ helper in the generated eval code).  An assignment of a real expression
to an integer variable is marked (ASTNode.int_lhs) and rounded there.  A text
the typer cannot read (an array index, a hierarchical name) is left as it
was; no other text changes, so a module without integer division compiles
to the same program as before.
"""

from __future__ import annotations
import re
from typing import Optional

from .va_const import number_value, param_kind, _based_value

_TOKEN_RE = re.compile(r'''
    (?P<ws>\s+)
  | (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<based>'[sS]?[hHdDoObB]\s*[0-9a-fA-F_xXzZ?]+)
  | (?P<number>(?:\d[\d_]*\.\d[\d_]*|\d[\d_]*\.?|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?[TGMKkmunpfa]?)
  | (?P<sysfunc>\$[A-Za-z_]\w*)
  | (?P<macro>`[A-Za-z_]\w*)
  | (?P<ident>[A-Za-z_]\w*)
  | (?P<op>\*\*|<<<|>>>|<<|>>|<=|>=|==|!=|&&|\|\||~\^|\^~|[-+*/%()<>!~&|^?:,\[\]])
''', re.X | re.S)

_BIN_LEVELS = [
    ('||',), ('&&',), ('|',), ('^', '^~', '~^'), ('&',), ('==', '!='),
    ('<', '<=', '>', '>='), ('<<', '>>', '<<<', '>>>'), ('+', '-'),
    ('*', '/', '%'), ('**',),
]

# builtins whose value is an integer when every argument is one
_INT_PRESERVING = {'abs', 'min', 'max'}
_INT_SYSFUNCS = {'$random', '$param_given', '$port_connected'}


class _Unreadable(Exception):
    pass


def _tokenize(text):
    toks, pos = [], 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise _Unreadable(text[pos])
        pos = m.end()
        if m.lastgroup != 'ws':
            toks.append((m.lastgroup, m.group()))
    return toks


class _Node:
    __slots__ = ('kind', 'op', 'kids', 's', 'e', 'type', 'value')

    def __init__(self, kind, s, e, kids=(), op=None, value=None):
        self.kind, self.s, self.e, self.kids, self.op, self.value = kind, s, e, list(kids), op, value
        self.type = 'real'


class _Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else (None, None)

    def at(self, *ops):
        k, v = self.peek()
        return k == 'op' and v in ops

    def expect(self, op):
        if not self.at(op):
            raise _Unreadable(op)
        self.i += 1

    def parse(self):
        e = self.ternary()
        if self.i != len(self.t):
            raise _Unreadable('trailing')
        return e

    def ternary(self):
        s = self.i
        c = self.binary(0)
        if self.at('?'):
            self.i += 1
            a = self.ternary()
            self.expect(':')
            b = self.ternary()
            return _Node('?:', s, self.i, (c, a, b))
        return c

    def binary(self, lvl):
        if lvl >= len(_BIN_LEVELS):
            return self.unary()
        s = self.i
        lhs = self.binary(lvl + 1)
        while self.at(*_BIN_LEVELS[lvl]):
            op = self.t[self.i][1]
            self.i += 1
            rhs = self.binary(lvl + 1)
            lhs = _Node('bin', s, self.i, (lhs, rhs), op=op)
        return lhs

    def unary(self):
        s = self.i
        if self.at('+', '-', '!', '~'):
            op = self.t[self.i][1]
            self.i += 1
            x = self.unary()
            return _Node('un', s, self.i, (x,), op=op)
        return self.primary()

    def args(self):
        out = []
        self.expect('(')
        if not self.at(')'):
            out.append(self.arg())
            while self.at(','):
                self.i += 1
                out.append(self.arg())
        self.expect(')')
        return out

    def arg(self):
        # I(<p>): a port probe -- the bracketed name is no expression
        if self.at('<') and self.peek(1)[0] == 'ident' and self.peek(2) == ('op', '>'):
            s = self.i
            self.i += 3
            return _Node('name', s, self.i, value='<port>')
        return self.ternary()

    def _based(self, s, text, size):
        """A based literal ('h5, 4'b0101): its integer value (va_const's rule);
        rendered as the decimal number (the C++ has no based literals)."""
        try:
            val = _based_value(text.replace(' ', ''), int(size.replace('_', '')) if size else None)
        except Exception:
            raise _Unreadable(text)
        n = _Node('num', s, self.i, value=val)
        n.op = 'based'
        return n

    def primary(self):
        s = self.i
        k, v = self.peek()
        if k is None:
            raise _Unreadable('end')
        if k == 'number':
            self.i += 1
            if self.peek()[0] == 'based':
                b = self.peek()[1]
                self.i += 1
                return self._based(s, b, v)
            try:
                val = number_value(v)
            except Exception:
                raise _Unreadable(v)
            return _Node('num', s, self.i, value=val)
        if k == 'based':
            self.i += 1
            return self._based(s, v, None)
        if k == 'string':
            self.i += 1
            return _Node('str', s, self.i)
        if k == 'op' and v == '(':
            self.i += 1
            e = self.ternary()
            self.expect(')')
            return _Node('paren', s, self.i, (e,))
        if k in ('ident', 'sysfunc', 'macro'):
            self.i += 1
            if self.at('('):
                a = self.args()
                return _Node('call', s, self.i, a, value=v)
            if self.at('['):
                self.i += 1
                idx = self.ternary()
                self.expect(']')
                return _Node('index', s, self.i, (idx,), value=v)
            return _Node('name', s, self.i, value=v)
        raise _Unreadable(v)


def _type(n, types, ftypes):
    for c in n.kids:
        _type(c, types, ftypes)
    k = n.kind
    if k == 'num':
        n.type = 'int' if isinstance(n.value, int) and not isinstance(n.value, bool) else 'real'
    elif k == 'str':
        n.type = 'str'
    elif k == 'name':
        n.type = types.get(n.value, 'real')
    elif k == 'index':
        n.type = types.get(n.value, 'real')
    elif k == 'paren':
        n.type = n.kids[0].type
    elif k == 'call':
        f = n.value
        if f in _INT_PRESERVING and n.kids and all(c.type == 'int' for c in n.kids):
            n.type = 'int'
        elif f in _INT_SYSFUNCS:
            n.type = 'int'
        elif f in ftypes:
            n.type = ftypes[f]
        else:
            n.type = 'real'
    elif k == 'un':
        n.type = 'int' if n.op in ('!', '~') else n.kids[0].type
    elif k == 'bin':
        op = n.op
        if op in ('==', '!=', '<', '<=', '>', '>=', '&&', '||', '&', '|', '^', '^~', '~^',
                  '<<', '>>', '<<<', '>>>'):
            n.type = 'int'
        else:
            a, b = n.kids[0].type, n.kids[1].type
            n.type = 'int' if a == 'int' and b == 'int' else 'real'
    elif k == '?:':
        a, b = n.kids[1].type, n.kids[2].type
        n.type = 'int' if a == 'int' and b == 'int' else 'real'


def _render(n, toks):
    """The text of n with every integer / and % below it rewritten (and a
    based literal as its decimal value)."""
    if n.kind == 'num' and n.op == 'based':
        return str(n.value)
    if n.kind == 'bin' and n.op in ('/', '%') and n.kids[0].type == 'int' and n.kids[1].type == 'int':
        fn = '_vae_idiv' if n.op == '/' else '_vae_imod'
        return f'{fn} ( {_render(n.kids[0], toks)} , {_render(n.kids[1], toks)} )'
    parts, pos = [], n.s
    for c in sorted(n.kids, key=lambda c: c.s):
        parts.extend(v for _, v in toks[pos:c.s])
        parts.append(_render(c, toks))
        pos = c.e
    parts.extend(v for _, v in toks[pos:n.e])
    return ' '.join(parts)


def _has_int_div(n):
    if n.kind == 'bin' and n.op in ('/', '%') and n.kids[0].type == 'int' and n.kids[1].type == 'int':
        return True
    if n.kind == 'num' and n.op == 'based':
        return True
    return any(_has_int_div(c) for c in n.kids)


def analyse(text: str, types: dict, ftypes: Optional[dict] = None):
    """(rewritten text or None when nothing changes, the static type of the
    text or None when it cannot be read)."""
    if not text or not text.strip():
        return None, None
    try:
        toks = _tokenize(text)
        tree = _Parser(toks).parse()
    except (_Unreadable, RecursionError):
        return None, None
    _type(tree, types, ftypes or {})
    if not _has_int_div(tree):
        return None, tree.type
    return _render(tree, toks), tree.type


def rewrite(text: str, types: dict, ftypes: Optional[dict] = None) -> str:
    new, _t = analyse(text, types, ftypes)
    return text if new is None else new


def _walk(node, types, ftypes):
    if node is None:
        return
    if node.expr is not None:
        new, typ = analyse(node.expr, types, ftypes)
        if new is not None:
            node.expr = new
        if node.lhs is not None and types.get(node.lhs) == 'int' and typ != 'int':
            node.int_lhs = True
    if node.condition is not None:
        node.condition = rewrite(node.condition, types, ftypes)
    if node.loop is not None and node.loop.get('cond'):
        node.loop['cond'] = rewrite(node.loop['cond'], types, ftypes)
    for c in node.children:
        _walk(c, types, ftypes)
    _walk(node.else_body, types, ftypes)


def module_types(module) -> dict:
    types = {}
    for p in module.params:
        types[p.name] = param_kind(p)
    for v in module.variables:
        types[v.name] = 'int' if v.type == 'integer' else 'real'
    for g in getattr(module, 'genvars', ()) or ():
        types[g] = 'int'
    return types


def lower_integer_ops(module):
    """Rewrite the integer / and % of every analog-block and analog-function
    expression of `module` (in place) and mark the rounding assignments."""
    types = module_types(module)
    ftypes = {f.name: ('int' if f.rtype == 'integer' else 'real')
              for f in getattr(module, 'functions', ()) or ()}
    _walk(module.analog_block, types, ftypes)
    for f in getattr(module, 'functions', ()) or ():
        ft = {p.name: param_kind(p) for p in module.params}
        for v in f.variables:
            ft[v.name] = 'int' if v.type == 'integer' else 'real'
        ft[f.name] = ftypes[f.name]
        _walk(f.body, ft, ftypes)
