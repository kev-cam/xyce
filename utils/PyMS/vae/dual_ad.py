"""The exact Jacobian of a math library (PyMS 1.1.5): forward-mode automatic
differentiation of the eval code itself.

The metaprogram (ginac_emitter) prints the eval code: GiNaC builds, folds
and simplifies every expression symbolically and prints it as C++ on
``double``.  The device shell needs dF/dx and dQ/dx at every evaluation.
From 1.1.0 to 1.1.4 build_vae_so.py took them as forward difference
quotients of the eval (a step of 1e-4*(|V|+1) on the absolute node
potential): a truncation error that grew with the bias (AC gain 9 % wrong
on a diode whose cathode sits at 50 V) and a transient that stalled in
steps of 1e-17 s (a 45 V rectifier).  Before 1.1.0 the metaprogram printed a
GiNaC forward-AD Jacobian: exact where it applied, but it differentiated the
arithmetic select form of a voltage condition while the value came from a
short-circuit ternary (derivatives of wrong sign where the two disagree, a
0 where GiNaC could not build the untaken branch), and it cannot follow the
C++ statements of the eval code 1.1.4 added (run-time loops, held state,
events).

Here the eval code is compiled a second time with every ``double`` a dual
number ``_vd`` carrying the value and its partial derivatives with respect
to all the library's unknowns (node voltages, then branch currents).  One
evaluation gives every column of the Jacobian, exact to rounding, and by
construction the derivative of the very value path the eval computes:
the same ternaries, the same loops, the same held values, the same regime
latches.  A held value or a latched regime (read from the device's held
state) is a constant: its derivative is 0, as for the difference quotient
before; a value an evaluation computes from the voltages keeps its
derivative wherever it goes, an @(initial_step) value in the small-signal
phase included (the user's rule of 2026-10-31).

transform() turns the printed eval.cpp into the dual-number copy (the
exports dropped: they live in the double copy); header() is the dual
number type and the functions of the eval code on it.
"""
import re

# The math functions of the eval code (GiNaC's print_csrc_double and the
# emitter's own C++ text) that the dual copy calls unqualified: their std::
# prefix is dropped so the namespace's dual overloads are found.
_STD_MATH = ('sqrt', 'exp', 'log', 'log10', 'pow', 'fabs', 'abs', 'sin', 'cos', 'tan',
             'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh',
             'atanh', 'floor', 'ceil', 'trunc', 'round', 'rint', 'nearbyint', 'fmin', 'fmax',
             'min', 'max', 'hypot', 'fmod', 'erf', 'erfc', 'cbrt', 'isnan', 'isinf',
             'isfinite', 'copysign', 'exp2', 'log2', 'expm1', 'log1p')
_STD_RE = re.compile(r'\bstd::(%s)\s*\(' % '|'.join(_STD_MATH))
_DOUBLE_RE = re.compile(r'\bdouble\b')
_META_RE = re.compile(r'^// (n_nodes|n_branches|n_currents) = (\d+)\s*$', re.M)


def meta(eval_text: str) -> dict:
    """The library's shape from the metadata lines of the printed eval code."""
    return {k: int(v) for k, v in _META_RE.findall(eval_text)}


def _strip_exports(text: str) -> str:
    """The eval code without its extern "C" definitions (one line or a
    braced body; braces inside string and character literals do not
    count) and without its #include lines."""
    out = []
    i, n = 0, len(text)
    key = 'extern "C"'
    while i < n:
        j = text.find(key, i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        # the definition: up to its first '{' (or ';' for a declaration),
        # then to the matching '}'
        k = j + len(key)
        depth = 0
        started = False
        while k < n:
            c = text[k]
            if c in '"\'':
                q = c
                k += 1
                while k < n and text[k] != q:
                    k += 2 if text[k] == '\\' else 1
                k += 1
                continue
            if c == '{':
                depth += 1
                started = True
            elif c == '}':
                depth -= 1
                if started and depth == 0:
                    k += 1
                    break
            elif c == ';' and not started:
                k += 1
                break
            k += 1
        i = k
    s = ''.join(out)
    return '\n'.join(l for l in s.split('\n') if not l.lstrip().startswith('#include'))


_FN_RE = re.compile(r'^(?:static )?void (vae_eval|_vae_eval_s\d+)\(VaeState\* _vae_s, double\* _vae_F, double\* _vae_Q\) \{\s*$')
_STR_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_ID_RE = re.compile(r'(?<!->)(?<![\w.])([A-Za-z_]\w*)')
_SEED_RE = re.compile(r'_vae_s->(V|I)\[(\d+)\]')
_ASSIGN_RE = re.compile(r'^\s*(?:(?:const\s+)?(?:double|int|bool|long|unsigned)\s+)?([A-Za-z_]\w*)\s*([-+*/]?=)(?!=)(.*)$', re.S)
_OUT_RE = re.compile(r'^\s*_vae_(F|Q)\[(\d+)\]\s*([-+]?=)(?!=)(.*)$', re.S)
_INCDEC_RE = re.compile(r'^\s*(?:\+\+|--)?\s*[A-Za-z_]\w*\s*(?:\+\+|--)?\s*$')
# an '=' that assigns (not ==, !=, <=, >=)
_EQ_RE = re.compile(r'(?<![=!<>])=(?!=)')
# helpers of the eval code whose value has no slope (a 0/1 regime or loop
# flag): kept on double in the dual copy
_FLAT_HELPERS = ('_vae_latch', '_vae_loop_over')


class Plan:
    """How the dual copy is built.  ndir dual directions; color[j] the
    direction of unknown j (seeded with slope 1 there); pat[i] the unknowns
    row i (of F and Q) can depend on; scalar: the eval-code variables that
    depend on no unknown, kept double.  Full mode (the fallback): one
    direction per unknown, every row dense, nothing kept double."""

    def __init__(self, nn, nc, nb):
        self.nn, self.nc, self.nb = nn, nc, nb
        self.nt = nn + nc
        self.full()

    def full(self, why=''):
        self.mode, self.why = 'full', why
        self.ndir = max(self.nt, 1)
        self.color = list(range(self.nt))
        self.pat = [set(range(self.nt)) for _ in range(self.nb)]
        self.scalar = set()
        return self


def _pieces(body: str) -> list:
    """The statements of a function body: split at ; { } outside
    parentheses, string literals and // comments."""
    out, cur, depth, i, n = [], [], 0, 0, len(body)
    while i < n:
        c = body[i]
        if c in '"\'':
            m = _STR_RE.match(body, i)
            j = m.end() if m else n
            cur.append(body[i:j])
            i = j
            continue
        if c == '/' and body.startswith('//', i):
            j = body.find('\n', i)
            i = n if j < 0 else j
            continue
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if depth == 0 and c in ';{}':
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append(''.join(cur))
    return [p for p in (x.strip() for x in out) if p]


def _deps(expr: str, deps: dict, nn: int) -> set:
    e = _STR_RE.sub('""', expr)
    d = set()
    for k, idx in _SEED_RE.findall(e):
        d.add(int(idx) + (nn if k == 'I' else 0))
    for name in _ID_RE.findall(_SEED_RE.sub(' ', e)):
        d |= deps.get(name, set())
    return d


def _bodies(eval_text: str) -> list:
    lines = eval_text.split('\n')
    bodies, i = [], 0
    while i < len(lines):
        if _FN_RE.match(lines[i]):
            j = i + 1
            while j < len(lines) and lines[j] != '}':
                j += 1
            bodies.append('\n'.join(lines[i + 1:j]))
            i = j
        i += 1
    return bodies


def _statements(body: str):
    """('set', name, expr) / ('out', 'F'|'Q', row, expr), or a str: why the
    body cannot be read."""
    stmts = []
    for p in _pieces(body):
        if p.startswith(('switch', 'case', 'default', 'return', 'break', 'continue')):
            continue
        o = _OUT_RE.match(p)
        if o:
            if _EQ_RE.search(_STR_RE.sub('""', o.group(4))):
                return 'an assignment inside an output: %s' % p[:80]
            stmts.append(('out', o.group(1), int(o.group(2)), o.group(4)))
            continue
        head = re.match(r'^(for|while|if)\s*\((.*)\)\s*(.*)$', p, re.S)
        if head:
            # a loop header assigns its int counter only (no slope); an if
            # carries no assignment (the eval code puts none there)
            if head.group(1) == 'for':
                for part in head.group(2).split(';'):
                    part = part.strip()
                    if part and _EQ_RE.search(part) and not re.match(r'^(?:int\s+)?\w+\s*=[^=]*$', part) \
                            and not _INCDEC_RE.match(part):
                        return 'a loop header: %s' % p[:80]
            elif _EQ_RE.search(_STR_RE.sub('""', head.group(2))):
                return 'an assignment in a condition: %s' % p[:80]
            rest = head.group(3).strip()
            if rest and _EQ_RE.search(_STR_RE.sub('""', rest)):
                return 'an assignment after a condition: %s' % p[:80]
            continue
        a = _ASSIGN_RE.match(p)
        if a:
            if _EQ_RE.search(_STR_RE.sub('""', a.group(3))):
                return 'a chained assignment: %s' % p[:80]
            stmts.append(('set', a.group(1), a.group(3)))
            continue
        if _EQ_RE.search(_STR_RE.sub('""', p)):
            return 'a statement not read: %s' % p[:80]
    return stmts


def plan(eval_text: str, full: bool = False) -> Plan:
    """Static dependency analysis of the eval code (every shadow): which
    unknowns each variable and each output row can depend on (a union over
    every path and every assignment, loops to the fixed point: an
    over-approximation), then a greedy colouring of the unknowns: two
    unknowns that share a row never share a direction (Curtis-Powell-Reid
    compression), so the derivative of row i in the direction of the colour
    of x_j IS dRow_i/dx_j for every x_j the row can depend on.  A variable
    no unknown reaches stays a double (the compiler refuses a dual assigned
    to it, so a dependency the analysis missed cannot pass silently there).
    Anything the analysis cannot read gives the full mode (exact as well,
    only slower)."""
    m = meta(eval_text)
    P = Plan(m.get('n_nodes', 0), m.get('n_currents', 0), m.get('n_branches', 0))
    if full:
        return P.full('requested')
    bodies = _bodies(eval_text)
    if not bodies:
        return P.full('no eval function found')
    pat = [set() for _ in range(P.nb)]
    zero, nonzero = set(), set()
    for body in bodies:
        stmts = _statements(body)
        if isinstance(stmts, str):
            return P.full(stmts)
        deps = {}
        for _ in range(10000):
            changed = False
            for st in stmts:
                if st[0] == 'set':
                    d = _deps(st[2], deps, P.nn)
                    old = deps.get(st[1])
                    if old is None or not d <= old:
                        deps[st[1]] = (old or set()) | d
                        changed = True
            if not changed:
                break
        else:
            return P.full('no fixed point')
        for st in stmts:
            if st[0] == 'out' and st[2] < P.nb:
                pat[st[2]] |= _deps(st[3], deps, P.nn)
        zero |= {k for k, v in deps.items() if not v}
        nonzero |= {k for k, v in deps.items() if v}
    # colouring: the column-intersection graph, largest degree first
    nbr = [set() for _ in range(P.nt)]
    for r in pat:
        for a in r:
            nbr[a].update(x for x in r if x != a)
    color = [-1] * P.nt
    for j in sorted(range(P.nt), key=lambda j: (-len(nbr[j]), j)):
        used = {color[x] for x in nbr[j] if color[x] >= 0}
        c = 0
        while c in used:
            c += 1
        color[j] = c
    P.mode, P.why = 'sparse', ''
    P.ndir = max(max(color) + 1, 1) if color else 1
    P.color, P.pat, P.scalar = color, pat, zero - nonzero
    return P


def _helper_params(line: str):
    m = re.match(r'^static inline [\w\s*&]+?\b(\w+)\s*\(([^)]*)\)\s*\{', line)
    return (m.group(1), m.group(2)) if m else (None, None)


def transform(eval_text: str, P: Plan) -> str:
    """The dual-number copy of the printed eval code: every double a dual,
    except the variables of P.scalar (no unknown reaches them); a helper on
    doubles is kept beside its dual form (an overload), a slope-free helper
    (_FLAT_HELPERS) stays on doubles."""
    s = _strip_exports(eval_text)
    out, in_fn = [], False
    decl = re.compile(r'^(\s*)double (\w+)( = |;)')
    for line in s.split('\n'):
        if _FN_RE.match(line):
            in_fn = True
            out.append(_DOUBLE_RE.sub('_vd', line))
            continue
        if in_fn:
            if line == '}':
                in_fn = False
                out.append(line)
                continue
            m = decl.match(line)
            t = _STD_RE.sub(lambda mm: mm.group(1) + '(', line)
            if m is not None and m.group(2) in P.scalar:
                t = m.group(1) + '__VAE_DOUBLE__' + t[len(m.group(1)) + len('double'):]
            out.append(_DOUBLE_RE.sub('_vd', t).replace('__VAE_DOUBLE__', 'double'))
            continue
        name, params = _helper_params(line)
        if name in _FLAT_HELPERS:
            out.append(line)
            continue
        t = _DOUBLE_RE.sub('_vd', _STD_RE.sub(lambda mm: mm.group(1) + '(', line))
        if name is not None and params is not None and _DOUBLE_RE.search(params):
            out.append(line)      # the double overload
        out.append(t)
    return '\n'.join(out)


def header(P: Plan) -> str:
    """The dual number type (P.ndir directions), its arithmetic and its
    functions, and the dual VaeState the dual eval reads (V, I as duals;
    the held state as the double copy has it, Hw a dual sink the Jacobian
    never writes)."""
    return _HEADER % dict(nt=P.ndir, nn=max(P.nn, 1), nc=max(P.nc, 1))


def jacobian_cpp(P: Plan) -> str:
    """vae_jacobian: one dual evaluation, unknown j seeded with slope 1 in
    the direction of its colour; dRow_i/dx_j read from that direction where
    the row can depend on x_j, 0 elsewhere."""
    pairs = [(i, j, P.color[j]) for i, r in enumerate(P.pat) for j in sorted(r)]
    tab = ', '.join('{%d, %d, %d}' % p for p in pairs) or '{-1, -1, -1}'
    col = ', '.join(str(c) for c in P.color) or '0'
    return _JAC % dict(mode=P.mode, ndir=P.ndir, nt=P.nt, why=(': ' + P.why) if P.why else '',
                       nn=P.nn, nc=P.nc, nb=P.nb, col=col, tab=tab, npairs=len(pairs), nbx=max(P.nb, 1))


_JAC = r"""// The Jacobian (%(mode)s%(why)s: %(ndir)d dual directions for %(nt)d unknowns).
// It reads the held state and writes none (Hw = 0): the held values are
// those of the evaluation the device shell makes itself (vae_eval).  The dual
// arithmetic gives 0, not a NaN, where a value that does not move meets an
// infinite slope (sqrt at 0); an overflow stays infinite, as the difference
// quotient's was, so the Newton step fails and Xyce goes on to its
// continuation methods.
extern "C" void vae_jacobian(VaeState* s, double* dFdV, double* dQdV){
  using namespace _vae_ad;
  const int N = %(nn)d, NC = %(nc)d, NB = %(nb)d, NT = %(nt)d;
  static const int color[] = {%(col)s};
  static const int nz[][3] = {%(tab)s};
  const int NZ = %(npairs)d;
  static _vae_ad::VaeState a;
  static _vd F[%(nbx)d], Q[%(nbx)d];
  for (int j = 0; j < N; ++j) { a.V[j] = _vd(s->V[j]); a.V[j].d[color[j]] = 1.0; }
  for (int k = 0; k < NC; ++k) { a.I[k] = _vd(s->I[k]); a.I[k].d[color[N + k]] = 1.0; }
  a.Vt = s->Vt; a.T = s->T; a.t = s->t; a.an = s->an; a.phase = s->phase;
  a.Hr = s->Hr; a.Hw = 0; a.Hl = s->Hl;
  _vae_eval_ad(&a, F, Q);
  for (int i = 0; i < NB * NT; ++i) { dFdV[i] = 0.0; dQdV[i] = 0.0; }
  for (int p = 0; p < NZ; ++p) {
    const int i = nz[p][0], j = nz[p][1], c = nz[p][2];
    dFdV[i*NT + j] = F[i].d[c];
    dQdV[i*NT + j] = Q[i].d[c];
  }
  // A row whose VALUE is not a number (sqrt(-1)) or infinite has no slope:
  // its row of the Jacobian is NaN, as the difference quotient's was, so the
  // Newton step fails and the analysis fails loudly (the device shell sets
  // the value itself to 0; a finite slope there would let a NaN model
  // converge silently).
  for (int i = 0; i < NB; ++i) {
    const double fv = F[i].v - F[i].v, qv = Q[i].v - Q[i].v;
    if (fv != 0.0) for (int j = 0; j < NT; ++j) dFdV[i*NT + j] = fv;
    if (qv != 0.0) for (int j = 0; j < NT; ++j) dQdV[i*NT + j] = qv;
  }
  // A branch equation whose slopes are all exactly 0 at this point (the
  // contribution cancels its own branch voltage: V(p, n) <+ tanh(V(p, n))
  // at 0, where 1 - tanh'(0) = 0) makes the matrix singular however exact it
  // is (VACASK fails there too).  That row alone takes a central difference
  // quotient, a relative step on each unknown it can depend on: a slope of
  // the size of the curvature, so Newton moves along the branch (1.1.0-1.1.4
  // took a forward difference of every row).  Nowhere else.
  for (int k = 0; k < NC; ++k) {
    const int i = vae_current_row(k);
    if (i < 0 || i >= NB) continue;
    bool flat = true;
    for (int j = 0; j < NT && flat; ++j) if (dFdV[i*NT + j] != 0.0 || dQdV[i*NT + j] != 0.0) flat = false;
    if (!flat) continue;
    static double Fp[%(nbx)d], Qp[%(nbx)d], Fm[%(nbx)d], Qm[%(nbx)d];
    for (int p = 0; p < NZ; ++p) {
      if (nz[p][0] != i) continue;
      const int j = nz[p][1];
      ::VaeState sp = *s, sm = *s; sp.Hw = 0; sm.Hw = 0;
      double h;
      if (j < N) { h = 1e-6 * (std::fabs(s->V[j]) + 1.0); sp.V[j] += h; sm.V[j] -= h; }
      else { h = 1e-6 * (std::fabs(s->I[j - N]) + 1e-3); sp.I[j - N] += h; sm.I[j - N] -= h; }
      _vae_eval_impl(&sp, Fp, Qp); _vae_eval_impl(&sm, Fm, Qm);
      dFdV[i*NT + j] = (Fp[i] - Fm[i]) / (2.0 * h);
      dQdV[i*NT + j] = (Qp[i] - Qm[i]) / (2.0 * h);
    }
  }
}
"""


_HEADER = r'''
namespace _vae_ad {
// the functions on doubles (a variable no unknown reaches stays a double)
using std::sqrt; using std::exp; using std::log; using std::log10; using std::pow; using std::fabs;
using std::abs; using std::sin; using std::cos; using std::tan; using std::asin; using std::acos;
using std::atan; using std::atan2; using std::sinh; using std::cosh; using std::tanh; using std::asinh;
using std::acosh; using std::atanh; using std::floor; using std::ceil; using std::trunc; using std::round;
using std::rint; using std::nearbyint; using std::fmin; using std::fmax; using std::hypot; using std::fmod;
using std::erf; using std::erfc; using std::cbrt; using std::isnan; using std::isinf; using std::isfinite;
using std::copysign; using std::exp2; using std::log2; using std::expm1; using std::log1p;
using std::min; using std::max;
enum { NT = %(nt)d };
struct _noinit {};
struct _vd {
    double v; double d[NT];
    inline _vd() : v(0.0) { for (int i = 0; i < NT; ++i) d[i] = 0.0; }
    inline _vd(double x) : v(x) { for (int i = 0; i < NT; ++i) d[i] = 0.0; }
    inline explicit _vd(_noinit) {}
    explicit operator double() const { return v; }
    explicit operator int() const { return (int)v; }
    explicit operator bool() const { return v != 0.0; }
    inline _vd& operator+=(const _vd& b) { v += b.v; for (int i = 0; i < NT; ++i) d[i] += b.d[i]; return *this; }
    inline _vd& operator-=(const _vd& b) { v -= b.v; for (int i = 0; i < NT; ++i) d[i] -= b.d[i]; return *this; }
    inline _vd& operator*=(const _vd& b);
    inline _vd& operator/=(const _vd& b);
    inline _vd& operator+=(double b) { v += b; return *this; }
    inline _vd& operator-=(double b) { v -= b; return *this; }
    inline _vd& operator*=(double b) { v *= b; if (b == 0.0) { for (int i = 0; i < NT; ++i) d[i] = 0.0; } else { for (int i = 0; i < NT; ++i) d[i] *= b; } return *this; }
    inline _vd& operator/=(double b) { v /= b; for (int i = 0; i < NT; ++i) d[i] /= b; return *this; }
};
// g * a.d: a derivative of 0 stays 0 where the local derivative g is not
// finite (sqrt at 0, log at 0, a power of 0), as a value that does not move
// has no slope; elsewhere the product.
static inline void _vad_chain(_vd& r, const _vd& a, double g) {
    if (g == 0.0) { for (int i = 0; i < NT; ++i) r.d[i] = 0.0; }
    else if (g - g == 0.0) { for (int i = 0; i < NT; ++i) r.d[i] = g * a.d[i]; }
    else { for (int i = 0; i < NT; ++i) r.d[i] = (a.d[i] != 0.0) ? g * a.d[i] : 0.0; }
}
static inline void _vad_chain2(_vd& r, const _vd& a, double ga, const _vd& b, double gb) {
    for (int i = 0; i < NT; ++i) {
        double x = (a.d[i] != 0.0 && ga != 0.0) ? ga * a.d[i] : 0.0;
        double y = (b.d[i] != 0.0 && gb != 0.0) ? gb * b.d[i] : 0.0;
        r.d[i] = x + y;
    }
}
static inline _vd operator+(const _vd& a) { return a; }
static inline _vd operator-(const _vd& a) { _vd r{_noinit()}; r.v = -a.v; for (int i = 0; i < NT; ++i) r.d[i] = -a.d[i]; return r; }
static inline _vd operator+(const _vd& a, const _vd& b) { _vd r{_noinit()}; r.v = a.v + b.v; for (int i = 0; i < NT; ++i) r.d[i] = a.d[i] + b.d[i]; return r; }
static inline _vd operator-(const _vd& a, const _vd& b) { _vd r{_noinit()}; r.v = a.v - b.v; for (int i = 0; i < NT; ++i) r.d[i] = a.d[i] - b.d[i]; return r; }
// A factor whose value is exactly 0 contributes no slope through the other
// factor's slope (0 * an infinite slope is 0, not a NaN): x * sqrt(x) at 0
// has the slope 0, as its derivative 1.5 sqrt(x) has.  The same for a
// quotient whose value is 0.
static inline _vd operator*(const _vd& a, const _vd& b) {
    _vd r{_noinit()}; r.v = a.v * b.v;
    if (a.v != 0.0 && b.v != 0.0) { for (int i = 0; i < NT; ++i) r.d[i] = a.d[i] * b.v + a.v * b.d[i]; }
    else if (a.v != 0.0) { for (int i = 0; i < NT; ++i) r.d[i] = a.v * b.d[i]; }
    else if (b.v != 0.0) { for (int i = 0; i < NT; ++i) r.d[i] = a.d[i] * b.v; }
    else { for (int i = 0; i < NT; ++i) r.d[i] = 0.0; }
    return r;
}
static inline _vd operator/(const _vd& a, const _vd& b) {
    _vd r{_noinit()}; r.v = a.v / b.v; double ib = 1.0 / b.v;
    if (r.v != 0.0) { for (int i = 0; i < NT; ++i) r.d[i] = (a.d[i] - r.v * b.d[i]) * ib; }
    else { for (int i = 0; i < NT; ++i) r.d[i] = a.d[i] * ib; }
    return r;
}
inline _vd& _vd::operator*=(const _vd& b) { *this = *this * b; return *this; }
inline _vd& _vd::operator/=(const _vd& b) { *this = *this / b; return *this; }
static inline _vd operator+(const _vd& a, double b) { _vd r = a; r.v += b; return r; }
static inline _vd operator+(double a, const _vd& b) { _vd r = b; r.v += a; return r; }
static inline _vd operator-(const _vd& a, double b) { _vd r = a; r.v -= b; return r; }
static inline _vd operator-(double a, const _vd& b) { _vd r{_noinit()}; r.v = a - b.v; for (int i = 0; i < NT; ++i) r.d[i] = -b.d[i]; return r; }
static inline _vd operator*(const _vd& a, double b) { _vd r{_noinit()}; r.v = a.v * b; if (b == 0.0) { for (int i = 0; i < NT; ++i) r.d[i] = 0.0; } else { for (int i = 0; i < NT; ++i) r.d[i] = a.d[i] * b; } return r; }
static inline _vd operator*(double a, const _vd& b) { _vd r{_noinit()}; r.v = a * b.v; if (a == 0.0) { for (int i = 0; i < NT; ++i) r.d[i] = 0.0; } else { for (int i = 0; i < NT; ++i) r.d[i] = a * b.d[i]; } return r; }
static inline _vd operator/(const _vd& a, double b) { _vd r{_noinit()}; r.v = a.v / b; for (int i = 0; i < NT; ++i) r.d[i] = a.d[i] / b; return r; }
static inline _vd operator/(double a, const _vd& b) { _vd r{_noinit()}; r.v = a / b.v; double g = -r.v / b.v; if (g == 0.0) { for (int i = 0; i < NT; ++i) r.d[i] = 0.0; } else { for (int i = 0; i < NT; ++i) r.d[i] = g * b.d[i]; } return r; }
#define _VAD_CMP(op) \
    static inline bool operator op(const _vd& a, const _vd& b) { return a.v op b.v; } \
    static inline bool operator op(const _vd& a, double b) { return a.v op b; } \
    static inline bool operator op(double a, const _vd& b) { return a op b.v; }
_VAD_CMP(<) _VAD_CMP(>) _VAD_CMP(<=) _VAD_CMP(>=) _VAD_CMP(==) _VAD_CMP(!=)
#undef _VAD_CMP
// (!, && and || take a dual through its explicit operator bool: the built-in
// operators, so && and || still short-circuit)
#define _VAD_U(name, val, grad) \
    static inline _vd name(const _vd& a) { _vd r{_noinit()}; const double x = a.v; (void)x; r.v = (val); \
        const double y = r.v; (void)y; _vad_chain(r, a, (grad)); return r; }
_VAD_U(exp, std::exp(x), y)
_VAD_U(log, std::log(x), 1.0 / x)
_VAD_U(log10, std::log10(x), 1.0 / (x * 2.302585092994045684))
_VAD_U(log2, std::log2(x), 1.0 / (x * 0.693147180559945309))
_VAD_U(exp2, std::exp2(x), y * 0.693147180559945309)
_VAD_U(expm1, std::expm1(x), y + 1.0)
_VAD_U(log1p, std::log1p(x), 1.0 / (1.0 + x))
_VAD_U(sqrt, std::sqrt(x), 0.5 / y)
_VAD_U(cbrt, std::cbrt(x), 1.0 / (3.0 * y * y))
_VAD_U(sin, std::sin(x), std::cos(x))
_VAD_U(cos, std::cos(x), -std::sin(x))
_VAD_U(tan, std::tan(x), 1.0 + y * y)
_VAD_U(asin, std::asin(x), 1.0 / std::sqrt(1.0 - x * x))
_VAD_U(acos, std::acos(x), -1.0 / std::sqrt(1.0 - x * x))
_VAD_U(atan, std::atan(x), 1.0 / (1.0 + x * x))
_VAD_U(sinh, std::sinh(x), std::cosh(x))
_VAD_U(cosh, std::cosh(x), std::sinh(x))
_VAD_U(tanh, std::tanh(x), 1.0 - y * y)
_VAD_U(asinh, std::asinh(x), 1.0 / std::sqrt(x * x + 1.0))
_VAD_U(acosh, std::acosh(x), 1.0 / std::sqrt(x * x - 1.0))
_VAD_U(atanh, std::atanh(x), 1.0 / (1.0 - x * x))
_VAD_U(erf, std::erf(x), 1.12837916709551257390 * std::exp(-x * x))
_VAD_U(erfc, std::erfc(x), -1.12837916709551257390 * std::exp(-x * x))
// |x|: the slope of the side x >= 0 at 0 (the forward difference before)
_VAD_U(fabs, std::fabs(x), x < 0.0 ? -1.0 : 1.0)
_VAD_U(abs, std::fabs(x), x < 0.0 ? -1.0 : 1.0)
// steps: slope 0
_VAD_U(floor, std::floor(x), 0.0)
_VAD_U(ceil, std::ceil(x), 0.0)
_VAD_U(trunc, std::trunc(x), 0.0)
_VAD_U(round, std::round(x), 0.0)
_VAD_U(rint, std::rint(x), 0.0)
_VAD_U(nearbyint, std::nearbyint(x), 0.0)
#undef _VAD_U
static inline bool isnan(const _vd& a) { return std::isnan(a.v); }
static inline bool isinf(const _vd& a) { return std::isinf(a.v); }
static inline bool isfinite(const _vd& a) { return std::isfinite(a.v); }
static inline _vd conjugate(const _vd& a) { return a; }
// a ** b
static inline _vd pow(const _vd& a, double b) {
    _vd r{_noinit()}; r.v = std::pow(a.v, b);
    double g = (b == 0.0) ? 0.0 : ((a.v != 0.0) ? b * r.v / a.v : b * std::pow(a.v, b - 1.0));
    _vad_chain(r, a, g); return r;
}
static inline _vd pow(double a, const _vd& b) {
    _vd r{_noinit()}; r.v = std::pow(a, b.v);
    _vad_chain(r, b, r.v * std::log(a)); return r;
}
static inline _vd pow(const _vd& a, const _vd& b) {
    _vd r{_noinit()}; r.v = std::pow(a.v, b.v);
    double ga = (b.v == 0.0) ? 0.0 : ((a.v != 0.0) ? b.v * r.v / a.v : b.v * std::pow(a.v, b.v - 1.0));
    _vad_chain2(r, a, ga, b, r.v * std::log(a.v)); return r;
}
static inline _vd atan2(const _vd& a, const _vd& b) {
    _vd r{_noinit()}; r.v = std::atan2(a.v, b.v); double q = a.v * a.v + b.v * b.v;
    _vad_chain2(r, a, b.v / q, b, -a.v / q); return r;
}
static inline _vd atan2(const _vd& a, double b) { return atan2(a, _vd(b)); }
static inline _vd atan2(double a, const _vd& b) { return atan2(_vd(a), b); }
static inline _vd hypot(const _vd& a, const _vd& b) {
    _vd r{_noinit()}; r.v = std::hypot(a.v, b.v);
    _vad_chain2(r, a, a.v / r.v, b, b.v / r.v); return r;
}
static inline _vd hypot(const _vd& a, double b) { return hypot(a, _vd(b)); }
static inline _vd hypot(double a, const _vd& b) { return hypot(_vd(a), b); }
static inline _vd fmod(const _vd& a, const _vd& b) {
    _vd r{_noinit()}; r.v = std::fmod(a.v, b.v); double k = std::trunc(a.v / b.v);
    for (int i = 0; i < NT; ++i) r.d[i] = a.d[i] - k * b.d[i]; return r;
}
static inline _vd fmod(const _vd& a, double b) { return fmod(a, _vd(b)); }
static inline _vd fmod(double a, const _vd& b) { return fmod(_vd(a), b); }
static inline _vd copysign(const _vd& a, const _vd& b) { return (std::signbit(a.v) == std::signbit(b.v)) ? a : -a; }
// min / max: the operand chosen (a on a tie), with its slope
static inline _vd fmin(const _vd& a, const _vd& b) { return (b.v < a.v) ? b : a; }
static inline _vd fmax(const _vd& a, const _vd& b) { return (b.v > a.v) ? b : a; }
static inline _vd fmin(const _vd& a, double b) { return (b < a.v) ? _vd(b) : a; }
static inline _vd fmax(const _vd& a, double b) { return (b > a.v) ? _vd(b) : a; }
static inline _vd fmin(double a, const _vd& b) { return (b.v < a) ? b : _vd(a); }
static inline _vd fmax(double a, const _vd& b) { return (b.v > a) ? b : _vd(a); }
static inline _vd min(const _vd& a, const _vd& b) { return fmin(a, b); }
static inline _vd max(const _vd& a, const _vd& b) { return fmax(a, b); }
static inline _vd min(const _vd& a, double b) { return fmin(a, b); }
static inline _vd max(const _vd& a, double b) { return fmax(a, b); }
static inline _vd min(double a, const _vd& b) { return fmin(a, b); }
static inline _vd max(double a, const _vd& b) { return fmax(a, b); }
// the device state of the dual evaluation.  T, the device temperature, is
// a double: no direction is seeded on it (a thermal unknown would make it a
// _vd with its own colour, see ginac_emitter's eval prologue).
struct VaeState { _vd V[%(nn)d]; double Vt; _vd I[%(nc)d]; double t;
                  int an; int phase; const double* Hr; _vd* Hw; const double* Hl; double T; };
}  // namespace _vae_ad
'''
