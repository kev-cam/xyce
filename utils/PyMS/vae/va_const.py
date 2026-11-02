"""Verilog-A constant expressions: the evaluator of parameter defaults.

The Verilog-AMS LRM gives a parameter's default as a constant expression over
literals, the other parameters, the standard functions and $simparam.  PyMS
used to read a default as a number, as the name of another parameter, or as a
Python ``eval`` of its tokens (xyce_device_gen._resolve_default and
GiNaCEmitter.__init__), which gave 0 for a function call, the ternary, ``**``,
a scale factor inside an expression, a digit separator, a based literal and a
name in capitals under the device shell's NAME=value overlay, and Python's
arithmetic where the language's differs (1 / 1000 is 0 in Verilog-A, (3 > 2) >
1 is 0, -7 % 3 is -1).  This module evaluates with the language's semantics
(LRM 4.2, 4.3.1, 9.20):

  * integer and real typing: an operation on two integers is an integer, with
    truncating division and a modulus that takes the sign of the first operand;
    the comparisons, ``!``, ``&&``, ``||`` give the integer 0 or 1; the bitwise
    and shift operators need integer operands; ``**`` of two integers follows
    LRM table 4-6; a real assigned to an integer parameter is rounded, halves
    away from zero; integers are 32-bit;
  * SI scale factors on any number (T G M K k m u n p f a), digit separators
    (1_000), based literals ('h3e8, 12'h3E8);
  * the standard functions of LRM 4.3.1 (ln natural, log base 10, exp, sqrt,
    min, max, abs, pow, floor, ceil, the trigonometric and hyperbolic ones,
    atan2, hypot);
  * $simparam("name" [, default]): the simulator's value when one is known
    (the device shell passes gmin and tnom), else the default, else an error;
  * a default that reads other parameters, evaluated in dependency order with
    the instance's given values, so an override of b re-evaluates r = 2000 / b.

Anything the evaluator cannot do is a ConstExprError that names the parameter
and the construct; it never yields 0 for a form it does not understand.

``to_cxx`` translates the same expression tree to C++ with the same typing for
the device shell, which re-evaluates a dependent default per instance from the
netlist's given values (Instance::processParams / Model::processParams).
"""

from __future__ import annotations
import math
import re
from typing import Callable, Optional


class ConstExprError(Exception):
    """A parameter default PyMS cannot evaluate; the message names the construct."""


class ThermalDefault(ConstExprError):
    """A default that reads $temperature or $vt where no temperature is given:
    its value is the instance's at run time, not a constant (1.1.6)."""


# The device temperature in parameter defaults (1.1.6, BUGS.md A17).  The LRM
# lets a default read $temperature / $vt (the instance's ambient
# temperature, kelvin); its value then follows the temperature at run time,
# so it is never baked: the math library computes it in its eval code from
# the state's temperature (T_SYM, VT_SYM there; ginac_emitter), and the
# library's cache key does not see the temperature.  k/q is Xyce's
# (N_DEV_Const.h CONSTboltz / CONSTQ), the correctly rounded double.
VAE_KOQ = '8.617086918058125e-05'
T_SYM = '_vae_T'      # the eval code's $temperature
VT_SYM = '_vae_vt'    # the eval code's $vt (= VAE_KOQ * T_SYM)
THERMAL_SYSFUNCS = ('$temperature', '$vt')
# the key of the temperature in a simparams dict / callable (a value there
# evaluates a thermal default at that temperature: the device shell's
# nominal reading at generation time)
TEMPERATURE_KEY = '$temperature'


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

SCALE_FACTORS = {
    'T': 1e12, 'G': 1e9, 'M': 1e6, 'K': 1e3, 'k': 1e3,
    'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15, 'a': 1e-18,
}

_TOKEN_RE = re.compile(r'''
    (?P<ws>\s+)
  | (?P<comment>/\*.*?\*/)
  | (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<based>'[sS]?[hHdDoObB]\s*[0-9a-fA-F_xXzZ?]+)
  | (?P<number>(?:\d[\d_]*\.\d[\d_]*|\d[\d_]*\.?|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?[TGMKkmunpfa]?)
  | (?P<sysfunc>\$[A-Za-z_]\w*)
  | (?P<macro>`[A-Za-z_]\w*)
  | (?P<ident>[A-Za-z_]\w*)
  | (?P<op>\*\*|<<<|>>>|<<|>>|<=|>=|==|!=|&&|\|\||~\^|\^~|[-+*/%()<>!~&|^?:,])
''', re.X | re.S)

# ('num', int|float) ('str', s) ('name', ident) ('call', fname, [args])
# ('sys', '$name', [args]) ('un', op, x) ('bin', op, a, b) ('?:', c, a, b)


def _tokenize(text: str) -> list[tuple[str, str]]:
    toks = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ConstExprError(f"the character {text[pos]!r} is not part of a constant expression")
        kind = m.lastgroup
        pos = m.end()
        if kind in ('ws', 'comment'):
            continue
        if kind == 'number' and pos < len(text) and (text[pos].isalnum() or text[pos] == '_'):
            raise ConstExprError(f"malformed number {m.group() + text[pos]!r}")
        toks.append((kind, m.group()))
    return toks


def number_value(text: str):
    """The value of a Verilog-A number: int for an integer literal, float for a real
    one (a decimal point, an exponent or a scale factor makes it real)."""
    s = text.replace('_', '')
    scale = 1.0
    if s and s[-1] in SCALE_FACTORS:
        scale = SCALE_FACTORS[s[-1]]
        s = s[:-1]
    if scale != 1.0 or '.' in s or 'e' in s or 'E' in s:
        if s.endswith('.'):
            s = s[:-1]
        return float(s) * scale
    return _i32(int(s))


_BASES = {'h': 16, 'H': 16, 'd': 10, 'D': 10, 'o': 8, 'O': 8, 'b': 2, 'B': 2}


def _based_value(text: str, size: Optional[int]) -> int:
    body = text[1:].replace('_', '')
    signed = body[0] in 'sS'
    if signed:
        body = body[1:]
    base = _BASES[body[0]]
    digits = body[1:].strip()
    if re.search(r'[xXzZ?]', digits):
        raise ConstExprError(f"the based literal {text} has x/z digits, which have no value")
    try:
        v = int(digits, base)
    except ValueError:
        raise ConstExprError(f"the based literal {text} is malformed")
    if size is not None:
        v &= (1 << size) - 1
        if signed and size > 0 and v & (1 << (size - 1)):
            v -= 1 << size
    return _i32(v)


# ---------------------------------------------------------------------------
# Parser (LRM table 4-3 precedence; every binary operator is left-associative,
# ?: right-associative; the unary operators bind tightest)
# ---------------------------------------------------------------------------

_BIN_LEVELS = [
    ('||',), ('&&',), ('|',), ('^', '^~', '~^'), ('&',), ('==', '!='),
    ('<', '<=', '>', '>='), ('<<', '>>', '<<<', '>>>'), ('+', '-'),
    ('*', '/', '%'), ('**',),
]


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.toks = _tokenize(text)
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def at_op(self, *ops) -> bool:
        k, v = self.peek()
        return k == 'op' and v in ops

    def take(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, op: str):
        if not self.at_op(op):
            k, v = self.peek()
            got = 'the end of the expression' if k is None else repr(v)
            raise ConstExprError(f"expected {op!r}, found {got}")
        self.take()

    def parse(self):
        if not self.toks:
            raise ConstExprError("the default is empty")
        e = self.ternary()
        if self.i < len(self.toks):
            raise ConstExprError(f"unexpected {self.peek()[1]!r} after a complete expression")
        return e

    def ternary(self):
        c = self.binary(0)
        if self.at_op('?'):
            self.take()
            a = self.ternary()
            self.expect(':')
            b = self.ternary()
            return ('?:', c, a, b)
        return c

    def binary(self, level: int):
        if level >= len(_BIN_LEVELS):
            return self.unary()
        lhs = self.binary(level + 1)
        while self.at_op(*_BIN_LEVELS[level]):
            op = self.take()[1]
            rhs = self.binary(level + 1)
            lhs = ('bin', op, lhs, rhs)
        return lhs

    def unary(self):
        if self.at_op('+', '-', '!', '~'):
            op = self.take()[1]
            return ('un', op, self.unary())
        return self.primary()

    def primary(self):
        k, v = self.peek()
        if k is None:
            raise ConstExprError("the expression ends where an operand is expected")
        if k == 'number':
            self.take()
            k2, v2 = self.peek()
            if k2 == 'based':              # 12'h3E8: a sized based literal
                self.take()
                size = int(v.replace('_', ''))
                return ('num', _based_value(v2, size))
            return ('num', number_value(v))
        if k == 'based':
            self.take()
            return ('num', _based_value(v, None))
        if k == 'string':
            self.take()
            return ('str', v[1:-1])
        if k == 'op' and v == '(':
            self.take()
            e = self.ternary()
            self.expect(')')
            return e
        if k == 'macro':
            raise ConstExprError(f"the macro {v} is not defined")
        if k in ('ident', 'sysfunc'):
            self.take()
            if self.at_op('('):
                self.take()
                args = []
                if not self.at_op(')'):
                    args.append(self.ternary())
                    while self.at_op(','):
                        self.take()
                        args.append(self.ternary())
                self.expect(')')
                return ('sys' if k == 'sysfunc' else 'call', v, args)
            if k == 'sysfunc':
                if v in THERMAL_SYSFUNCS:
                    return ('sys', v, [])
                raise ConstExprError(f"{v} is not a constant")
            return ('name', v)
        raise ConstExprError(f"unexpected {v!r}")


def parse_const_expr(text: str):
    """Parse a Verilog-A constant expression into a tree (see the node forms above)."""
    return _Parser(text).parse()


def names_in(node) -> set[str]:
    """The identifiers an expression tree reads."""
    kind = node[0]
    if kind == 'name':
        return {node[1]}
    if kind in ('call', 'sys'):
        out = set()
        for a in node[2]:
            out |= names_in(a)
        return out
    if kind in ('un', 'bin', '?:'):
        out = set()
        for a in node[2:]:
            out |= names_in(a)
        return out
    return set()


def uses_temperature(node) -> bool:
    """Whether a tree reads $temperature or $vt (directly)."""
    kind = node[0]
    if kind == 'sys':
        return node[1] in THERMAL_SYSFUNCS or any(uses_temperature(a) for a in node[2])
    if kind == 'call':
        return any(uses_temperature(a) for a in node[2])
    if kind in ('un', 'bin', '?:'):
        return any(uses_temperature(a) for a in node[2:])
    return False


def uses_simparam(node) -> bool:
    kind = node[0]
    if kind == 'sys':
        return True
    if kind in ('call',):
        return any(uses_simparam(a) for a in node[2])
    if kind in ('un', 'bin', '?:'):
        return any(uses_simparam(a) for a in node[2:])
    return False


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _i32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - 0x100000000 if v & 0x80000000 else v


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v, what: str):
    if isinstance(v, str):
        raise ConstExprError(f"{what} is a string where a number is needed")
    return v


def _finite(v, what: str):
    if isinstance(v, float) and not math.isfinite(v):
        raise ConstExprError(f"{what} is not finite ({v})")
    return v


def _int_div(a: int, b: int, op: str) -> int:
    if b == 0:
        raise ConstExprError(f"integer {'division' if op == '/' else 'modulus'} by zero")
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    if op == '/':
        return _i32(q)
    return _i32(a - b * q)           # the sign of the first operand


def _ipow(a: int, b: int) -> int:
    """LRM table 4-6: the power operator on two integers."""
    if b < 0:
        if a == 0:
            raise ConstExprError("0 ** (a negative integer) has no value")
        if a == 1:
            return 1
        if a == -1:
            return -1 if b % 2 else 1
        return 0
    return _i32(a ** b)


def _rpow(a: float, b: float) -> float:
    try:
        return math.pow(a, b)
    except OverflowError:
        return math.inf
    except ValueError:
        if a == 0.0 and b < 0:
            return math.inf
        raise ConstExprError(f"pow({a!r}, {b!r}) is not a real number")


_UNARY_REAL = {
    'ln': math.log, 'log': math.log10, 'exp': math.exp, 'sqrt': math.sqrt,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
    'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
    'asinh': math.asinh, 'acosh': math.acosh, 'atanh': math.atanh,
    'floor': math.floor, 'ceil': math.ceil,
}
_BINARY_REAL = {'pow': _rpow, 'atan2': math.atan2, 'hypot': math.hypot}
# The functions whose result is an integer when every argument is one.
_INT_PRESERVING = {'abs', 'min', 'max'}
FUNCTIONS = set(_UNARY_REAL) | set(_BINARY_REAL) | _INT_PRESERVING


def _call(fname: str, args: list):
    if fname in _UNARY_REAL or fname in _INT_PRESERVING and fname == 'abs':
        if len(args) != 1:
            raise ConstExprError(f"{fname}() takes one argument, not {len(args)}")
    elif fname in _BINARY_REAL or fname in ('min', 'max'):
        if len(args) != 2:
            raise ConstExprError(f"{fname}() takes two arguments, not {len(args)}")
    else:
        raise ConstExprError(f"{fname}() is not a standard function usable in a constant expression")
    args = [_num(a, f"an argument of {fname}()") for a in args]
    if fname == 'abs':
        return abs(args[0])
    if fname in ('min', 'max'):
        v = (min if fname == 'min' else max)(args[0], args[1])
        return v if all(_is_int(a) for a in args) else float(v)
    try:
        if fname in _UNARY_REAL:
            x = float(args[0])
            if fname in ('ln', 'log') and x == 0.0:
                return -math.inf
            v = _UNARY_REAL[fname](x)
        else:
            v = _BINARY_REAL[fname](float(args[0]), float(args[1]))
    except OverflowError:
        return math.inf
    except ValueError:
        raise ConstExprError(f"{fname}({', '.join(repr(a) for a in args)}) is not a real number")
    return float(v)


def static_type(node, types: dict) -> str:
    """'int', 'real' or 'str': the type of an expression without evaluating it
    (types: parameter name -> 'int' | 'real' | 'str')."""
    kind = node[0]
    if kind == 'num':
        return 'int' if _is_int(node[1]) else 'real'
    if kind == 'str':
        return 'str'
    if kind == 'name':
        return types.get(node[1], 'real')
    if kind == 'sys':
        return 'real'
    if kind == 'call':
        if node[1] in _INT_PRESERVING and all(static_type(a, types) == 'int' for a in node[2]):
            return 'int'
        return 'real'
    if kind == 'un':
        if node[1] in ('!', '~'):
            return 'int'
        return static_type(node[2], types)
    if kind == 'bin':
        op = node[1]
        if op in ('==', '!=', '<', '<=', '>', '>=', '&&', '||', '&', '|', '^', '^~', '~^',
                  '<<', '>>', '<<<', '>>>'):
            return 'int'
        a, b = static_type(node[2], types), static_type(node[3], types)
        return 'int' if a == 'int' and b == 'int' else 'real'
    if kind == '?:':
        a, b = static_type(node[2], types), static_type(node[3], types)
        return 'int' if a == 'int' and b == 'int' else 'real'
    raise ConstExprError(f"unknown node {kind}")


def evaluate(node, env: dict, simparam: Optional[Callable[[str], Optional[float]]] = None,
             param_names: Optional[set] = None, types: Optional[dict] = None):
    """Evaluate a tree: env maps parameter names to values (int, float or str);
    simparam(name) gives the simulator's value of a $simparam name or None;
    param_names (when given) tells an unknown name from a not-yet-evaluated parameter."""
    kind = node[0]
    if kind in ('num', 'str'):
        return node[1]
    if kind == 'name':
        nm = node[1]
        if nm in env:
            return env[nm]
        if param_names is not None and nm in param_names:
            raise ConstExprError(f"the parameter {nm} has no value yet")
        raise ConstExprError(f"the name {nm} is not a parameter")
    if kind == 'sys':
        fname, args = node[1], node[2]
        if fname in THERMAL_SYSFUNCS:
            temp = simparam(TEMPERATURE_KEY) if simparam else None
            if fname == '$vt' and args:
                if len(args) > 1:
                    raise ConstExprError('$vt takes one argument, the temperature')
                return float(VAE_KOQ) * float(_num(evaluate(args[0], env, simparam, param_names, types),
                                                   'the $vt argument'))
            if temp is None:
                raise ThermalDefault(f"{fname} is the instance's temperature at run time")
            return float(temp) if fname == '$temperature' else float(VAE_KOQ) * float(temp)
        if fname != '$simparam':
            raise ConstExprError(f"{fname} is not a constant")
        if not args or args[0][0] != 'str' or len(args) > 2:
            raise ConstExprError('$simparam takes a quoted name and an optional default')
        name = args[0][1]
        value = simparam(name) if simparam else None
        if value is not None:
            return float(value)
        if len(args) == 2:
            return float(_num(evaluate(args[1], env, simparam, param_names, types), 'the $simparam default'))
        raise ConstExprError(f'$simparam("{name}") is unknown to Xyce and has no default')
    if kind == 'call':
        return _call(node[1], [evaluate(a, env, simparam, param_names, types) for a in node[2]])
    if kind == 'un':
        op, x = node[1], _num(evaluate(node[2], env, simparam, param_names, types), f"the operand of {node[1]}")
        if op == '+':
            return x
        if op == '-':
            return _i32(-x) if _is_int(x) else -x
        if op == '!':
            return 0 if x else 1
        if not _is_int(x):
            raise ConstExprError(f"the operator ~ needs an integer operand, not the real {x!r}")
        return _i32(~x)
    if kind == '?:':
        c = _num(evaluate(node[1], env, simparam, param_names, types), "the condition of ?:")
        v = evaluate(node[2] if c else node[3], env, simparam, param_names, types)
        other = static_type(node[3] if c else node[2], types or {})
        if _is_int(v) and other == 'real':
            return float(v)
        return v
    # binary
    op = node[1]
    if op in ('==', '!='):
        # equality is the one operator on strings (LRM 4.2: a string parameter
        # compared with a literal or another string parameter)
        a = evaluate(node[2], env, simparam, param_names, types)
        b = evaluate(node[3], env, simparam, param_names, types)
        if isinstance(a, str) or isinstance(b, str):
            if not (isinstance(a, str) and isinstance(b, str)):
                raise ConstExprError(f"{op} compares a string with a number")
            return int((a == b) == (op == '=='))
    a = _num(evaluate(node[2], env, simparam, param_names, types), f"an operand of {op}")
    b = _num(evaluate(node[3], env, simparam, param_names, types), f"an operand of {op}")
    both_int = _is_int(a) and _is_int(b)
    if op in ('==', '!=', '<', '<=', '>', '>='):
        return int({'==': a == b, '!=': a != b, '<': a < b, '<=': a <= b, '>': a > b, '>=': a >= b}[op])
    if op == '&&':
        return int(bool(a) and bool(b))
    if op == '||':
        return int(bool(a) or bool(b))
    if op in ('&', '|', '^', '^~', '~^', '<<', '>>', '<<<', '>>>'):
        if not both_int:
            bad = a if not _is_int(a) else b
            raise ConstExprError(f"the operator {op} needs integer operands, not the real {bad!r}")
        if op == '&':
            return _i32(a & b)
        if op == '|':
            return _i32(a | b)
        if op == '^':
            return _i32(a ^ b)
        if op in ('^~', '~^'):
            return _i32(~(a ^ b))
        if b < 0:
            raise ConstExprError(f"a negative shift count ({b})")
        if op in ('<<', '<<<'):
            return _i32(a << b) if b < 32 else 0
        if op == '>>':
            return _i32((a & 0xFFFFFFFF) >> b) if b < 32 else 0
        return a >> b if b < 32 else (-1 if a < 0 else 0)
    if op == '+':
        return _i32(a + b) if both_int else float(a) + float(b)
    if op == '-':
        return _i32(a - b) if both_int else float(a) - float(b)
    if op == '*':
        return _i32(a * b) if both_int else float(a) * float(b)
    if op in ('/', '%'):
        if both_int:
            return _int_div(a, b, op)
        a, b = float(a), float(b)
        if op == '/':
            if b == 0.0:
                if a == 0.0:
                    raise ConstExprError("0.0 / 0.0 has no value")
                return math.copysign(math.inf, a) * math.copysign(1.0, b)
            return a / b
        if b == 0.0:
            raise ConstExprError("a real modulus by zero has no value")
        return math.fmod(a, b)
    if op == '**':
        return _ipow(a, b) if both_int else _rpow(float(a), float(b))
    raise ConstExprError(f"unknown operator {op}")


# ---------------------------------------------------------------------------
# Parameter defaults in dependency order
# ---------------------------------------------------------------------------

def param_kind(p) -> str:
    """'int', 'real' or 'str' from a Param's declared type."""
    t = (getattr(p, 'type', None) or 'real').lower()
    if t == 'integer':
        return 'int'
    if t == 'string':
        return 'str'
    return 'real'


def convert(value, kind: str, what: str):
    """Assign a value to a parameter of the given kind (LRM 4.2.1.1: a real
    becomes an integer by rounding, halves away from zero)."""
    if kind == 'str':
        if not isinstance(value, str):
            raise ConstExprError(f"{what} is a string parameter but its value {value!r} is a number")
        return value
    value = _finite(_num(value, what), what)
    if kind == 'int':
        if isinstance(value, float):
            r = math.floor(abs(value) + 0.5)
            value = int(r) if value >= 0 else -int(r)
        if not -0x80000000 <= value <= 0x7FFFFFFF:
            raise ConstExprError(f"{what} = {value} does not fit a 32-bit integer")
        return value
    return float(value)


# Xyce's values when the device shell has not passed any (generation time).
XYCE_SIMPARAM_DEFAULTS = {'gmin': 1e-12, 'tnom': 27.0}
# The names the device shell resolves at instance time, with the C++ that reads them.
SIMPARAM_CXX = {'gmin': 'getDeviceOptions().gmin', 'tnom': '(getDeviceOptions().tnom - 273.15)'}


def _simparam_lookup(simparams):
    if simparams is None:
        return None
    if callable(simparams):
        return simparams
    return lambda name: simparams.get(name)


def parse_param_default(p):
    """The expression tree of a Param's default, with the parameter named on error."""
    try:
        return parse_const_expr(p.default if p.default is not None else '0')
    except ConstExprError as e:
        raise ConstExprError(f"parameter {p.name}: its default {p.default!r}: {e}")


def _check_references(module, p, tree, declared: set, names: set):
    """A default may read only parameters declared before it (IEEE 1364 4.10.1,
    which Verilog-AMS inherits: 'constant numbers and previously defined
    parameters'); a later one, itself, or an array parameter is an error."""
    for n in sorted(names_in(tree)):
        if n not in names or n in declared:
            continue
        q = next(q for q in module.params if q.name == n)
        if getattr(q, 'array_size', None):
            why = "an array parameter"
        elif n == p.name:
            why = "the parameter itself"
        else:
            why = "a parameter declared after it"
        raise ConstExprError(f"parameter {p.name} of module {module.name}: its default "
                             f"{p.default} reads {n}, {why}")


def evaluate_param_defaults(module, given: Optional[dict] = None, simparams=None,
                            thermal: Optional[set] = None) -> dict:
    """The value of every scalar parameter of ``module``: the given values where
    supplied (the netlist's, converted to the parameter's type), the Verilog-A
    default of the others, evaluated in declaration order (a default reads only
    the parameters declared before it, so that is dependency order).  given
    maps parameter names (case-sensitive, the module's) to numbers; simparams
    is a dict or a callable giving the simulator's $simparam values.  Raises
    ConstExprError naming the parameter and the construct for anything it
    cannot evaluate.

    A default that reads $temperature or $vt (or a parameter whose default
    does, transitively) is a thermal default: with simparams giving
    TEMPERATURE_KEY it is evaluated at that temperature; else, when the set
    ``thermal`` is passed, its name is added to it and it gets no value (the
    math library computes it at run time); else ThermalDefault."""
    given = given or {}
    simparam = _simparam_lookup(simparams)
    names = {p.name for p in module.params}
    kinds = {p.name: param_kind(p) for p in module.params}
    values: dict = {}
    declared: set = set()
    for p in module.params:
        if getattr(p, 'array_size', None):
            continue
        if p.name in given:
            values[p.name] = convert(given[p.name], kinds[p.name], f"parameter {p.name}")
        else:
            tree = parse_param_default(p)
            _check_references(module, p, tree, declared, names)
            if thermal is not None and (simparam is None or simparam(TEMPERATURE_KEY) is None) \
                    and (uses_temperature(tree) or names_in(tree) & thermal):
                thermal.add(p.name)
                declared.add(p.name)
                continue
            try:
                v = evaluate(tree, values, simparam, names, kinds)
                values[p.name] = convert(v, kinds[p.name], f"parameter {p.name}")
            except ConstExprError as e:
                raise ConstExprError(f"parameter {p.name} of module {module.name}: "
                                     f"its default {p.default}: {e}")
        declared.add(p.name)
    return values


def dependency_order(module) -> list:
    """The scalar parameters whose default reads another parameter or $simparam,
    as (param, tree, dependencies) in declaration order, which evaluates every
    dependency first (evaluate_param_defaults rejects any other reference)."""
    names = {p.name for p in module.params}
    out = []
    for p in module.params:
        if getattr(p, 'array_size', None):
            continue
        tree = parse_param_default(p)
        deps = {n for n in names_in(tree) if n in names}
        if deps or uses_simparam(tree):
            out.append((p, tree, deps))
    return out


# ---------------------------------------------------------------------------
# C++ for the device shell
# ---------------------------------------------------------------------------

def _cxx_real(v: float) -> str:
    s = repr(float(v))
    if s in ('inf', '-inf', 'nan'):
        raise ConstExprError(f"the literal {s} cannot be written in C++")
    return s


def to_cxx(node, types: dict, ref: Callable[[str], str]) -> tuple[str, str]:
    """Translate a tree to a C++ expression with Verilog-A typing.  types maps
    parameter names to 'int' | 'real' | 'str'; ref(name) is the C++ lvalue of a
    parameter (a double).  Returns (code, 'int' | 'real').  The shell defines
    _pyms_ipow for ** on integers."""
    kind = node[0]
    if kind == 'num':
        v = node[1]
        return (str(v), 'int') if _is_int(v) else (_cxx_real(v), 'real')
    if kind == 'str':
        raise ConstExprError("a string in arithmetic")
    if kind == 'name':
        t = types.get(node[1], 'real')
        if t == 'str':
            raise ConstExprError(f"the string parameter {node[1]} in arithmetic")
        if t == 'int':
            return f'((int)({ref(node[1])}))', 'int'
        return f'({ref(node[1])})', 'real'
    if kind == 'sys':
        args = node[2]
        if node[1] == '$temperature':
            return T_SYM, 'real'
        if node[1] == '$vt':
            if args:
                code, _t = to_cxx(args[0], types, ref)
                return f'({VAE_KOQ} * (double){code})', 'real'
            return VT_SYM, 'real'
        name = args[0][1] if args and args[0][0] == 'str' else None
        if name in SIMPARAM_CXX:
            return f'({SIMPARAM_CXX[name]})', 'real'
        if len(args) == 2:
            code, _t = to_cxx(args[1], types, ref)
            return f'((double){code})', 'real'
        raise ConstExprError(f'$simparam("{name}") is unknown to Xyce and has no default')
    if kind == 'call':
        fname = node[1]
        parts = [to_cxx(a, types, ref) for a in node[2]]
        all_int = all(t == 'int' for _c, t in parts)
        if fname == 'abs':
            return (f'(std::abs({parts[0][0]}))', 'int') if all_int else (f'(std::fabs((double){parts[0][0]}))', 'real')
        if fname in ('min', 'max'):
            if all_int:
                return f'(std::{fname}({parts[0][0]}, {parts[1][0]}))', 'int'
            return f'(std::f{fname}((double){parts[0][0]}, (double){parts[1][0]}))', 'real'
        std = {'ln': 'std::log', 'log': 'std::log10'}.get(fname, 'std::' + fname)
        return f'({std}({", ".join("(double)" + c for c, _t in parts)}))', 'real'
    if kind == 'un':
        op = node[1]
        code, t = to_cxx(node[2], types, ref)
        if op == '+':
            return code, t
        if op == '-':
            return f'(-{code})', t
        if op == '!':
            return f'((int)!{code})', 'int'
        if t != 'int':
            raise ConstExprError("the operator ~ needs an integer operand")
        return f'(~{code})', 'int'
    if kind == '?:':
        c, _ct = to_cxx(node[1], types, ref)
        a, at = to_cxx(node[2], types, ref)
        b, bt = to_cxx(node[3], types, ref)
        if at == bt:
            return f'(({c} != 0) ? {a} : {b})', at
        return f'(({c} != 0) ? (double){a} : (double){b})', 'real'
    op = node[1]
    if op in ('==', '!=') and 'str' in (static_type(node[2], types), static_type(node[3], types)):
        return f'((int)({_cxx_string(node[2], types, ref)} {op} {_cxx_string(node[3], types, ref)}))', 'int'
    a, at = to_cxx(node[2], types, ref)
    b, bt = to_cxx(node[3], types, ref)
    both_int = at == 'int' and bt == 'int'
    if op in ('==', '!=', '<', '<=', '>', '>='):
        return f'((int)({a} {op} {b}))', 'int'
    if op in ('&&', '||'):
        return f'((int)({a} {op} {b}))', 'int'
    if op in ('&', '|', '^', '^~', '~^', '<<', '>>', '<<<', '>>>'):
        if not both_int:
            raise ConstExprError(f"the operator {op} needs integer operands")
        if op in ('^~', '~^'):
            return f'(~({a} ^ {b}))', 'int'
        if op == '>>':
            return f'((int)((unsigned)({a}) >> {b}))', 'int'
        cop = {'<<<': '<<', '>>>': '>>'}.get(op, op)
        return f'({a} {cop} {b})', 'int'
    if op == '**':
        if both_int:
            return f'(_pyms_ipow({a}, {b}))', 'int'
        return f'(std::pow((double){a}, (double){b}))', 'real'
    if both_int:
        return f'({a} {op} {b})', 'int'
    if op == '%':
        return f'(std::fmod((double){a}, (double){b}))', 'real'
    return f'((double){a} {op} (double){b})', 'real'


def _cxx_string(node, types: dict, ref: Callable[[str], str]) -> str:
    """A string operand of == / != as a C++ std::string: a literal, or a string
    parameter (the shell holds those as std::string members)."""
    if node[0] == 'str':
        return 'std::string("' + node[1].replace('\\', '\\\\').replace('"', '\\"') + '")'
    if node[0] == 'name' and types.get(node[1]) == 'str':
        return f'({ref(node[1])})'
    raise ConstExprError("== / != compares a string with a number")


IPOW_CXX = (
    '// The shell re-evaluates parameter defaults per instance (va_const.to_cxx):\n'
    '// ** on two Verilog-A integers (LRM table 4-6) and the real-to-integer\n'
    '// conversion of an integer parameter (rounded, halves away from zero).\n'
    'static inline int _pyms_ipow(int a, int b) {\n'
    '  if (b < 0) { if (a == 1) return 1; if (a == -1) return (b % 2) ? -1 : 1; return 0; }\n'
    '  int r = 1; while (b-- > 0) r *= a; return r;\n'
    '}\n'
    'static inline double _pyms_rnd(double x) { return (x >= 0) ? std::floor(x + 0.5) : -std::floor(-x + 0.5); }\n'
)


# ---------------------------------------------------------------------------
# Value ranges: from / exclude (LRM 3.4.2)
# ---------------------------------------------------------------------------

class ParamRangeError(ConstExprError):
    """A parameter value outside its declared from range, or excluded."""


def _split_top(text: str, sep: str) -> list:
    out, depth, cur = [], 0, []
    i = 0
    while i < len(text):
        c = text[i]
        if c == '"':
            j = i + 1
            while j < len(text) and text[j] != '"':
                j += 2 if text[j] == chr(92) else 1
            cur.append(text[i:j + 1])
            i = j + 1
            continue
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        if c == sep and depth == 0:
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append(''.join(cur))
    return out


def parse_value_range(text: str):
    """A value_range_specifier as written after from / exclude:
    ('interval', lo, hi, lo_closed, hi_closed) with lo/hi expression texts
    ('inf' / '-inf' allowed), ('set', [texts]) for '{ a, b } (the value
    set of a string parameter), or ('value', text) for exclude's single
    constant."""
    t = ' '.join(text.split())
    if t.startswith("'"):
        t = t[1:].strip()
    if t.startswith('{') and t.endswith('}'):
        return ('set', [x.strip() for x in _split_top(t[1:-1], ',') if x.strip()])
    if t[:1] in '([' and t[-1:] in ')]':
        parts = _split_top(t[1:-1], ':')
        if len(parts) == 2:
            return ('interval', parts[0].strip(), parts[1].strip(), t[0] == '[', t[-1] == ']')
    return ('value', t)


def _range_end(text: str, env: dict, simparam, names, kinds):
    s = text.replace(' ', '')
    if s in ('inf', '+inf'):
        return math.inf
    if s == '-inf':
        return -math.inf
    return evaluate(parse_const_expr(text), env, simparam, names, kinds)


def _range_text(kind: str, text: str) -> str:
    return f"{kind} {''.join(text.split())}"


def check_param_ranges(module, values: dict, simparams=None):
    """Enforce every parameter's from / exclude ranges on its final value
    (the instance's: the given values over the evaluated defaults), with the
    LRM's semantics: with one or more from ranges the value must lie in one
    of them, and it must lie in no exclude range and equal no excluded
    value; '(' / ')' exclude the end point, '[' / ']' include it, inf is
    unbounded; a string parameter's from '{...} lists its legal values.
    Raises ParamRangeError naming the parameter, its value and the range."""
    simparam = _simparam_lookup(simparams)
    names = {p.name for p in module.params}
    kinds = {p.name: param_kind(p) for p in module.params}
    for p in module.params:
        vr = getattr(p, 'value_ranges', None) or []
        if not vr or getattr(p, 'array_size', None) or p.name not in values:
            continue
        v = values[p.name]
        where = f"{p.loc}: " if getattr(p, 'loc', None) else ""

        def inside(spec, kind_text):
            sp = parse_value_range(spec)
            try:
                if sp[0] == 'interval':
                    lo = _range_end(sp[1], values, simparam, names, kinds)
                    hi = _range_end(sp[2], values, simparam, names, kinds)
                    if isinstance(v, str):
                        raise ConstExprError("a string value has no numeric range")
                    above = v >= lo if sp[3] else v > lo
                    below = v <= hi if sp[4] else v < hi
                    return above and below
                if sp[0] == 'set':
                    return any(v == evaluate(parse_const_expr(x), values, simparam, names, kinds)
                               for x in sp[1])
                return v == evaluate(parse_const_expr(sp[1]), values, simparam, names, kinds)
            except ConstExprError as e:
                raise ConstExprError(f"{where}parameter {p.name} of module {module.name}: "
                                     f"its range {_range_text(kind_text, spec)}: {e}")

        shown = repr(v) if isinstance(v, str) else f"{v:.12g}"
        froms = [t for k, t in vr if k == 'from']
        if froms and not any(inside(t, 'from') for t in froms):
            rng = ' or '.join(_range_text('from', t) for t in froms)
            raise ParamRangeError(f"{where}parameter {p.name} of module {module.name} = {shown} "
                                  f"is outside its range {rng}")
        for k, t in vr:
            if k == 'exclude' and inside(t, 'exclude'):
                raise ParamRangeError(f"{where}parameter {p.name} of module {module.name} = {shown} "
                                      f"is excluded by {_range_text('exclude', t)}")
