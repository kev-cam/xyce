#!/usr/bin/env python3
"""cir2vacask.py -- translate a (Xyce-dialect) SPICE cosim deck to a VACASK netlist.

The mixed-signal front ends (simetrix_cosim.pl, xyce_from_vhdl.py) emit a Xyce
deck whose analog/digital boundaries are PWL sources with a code: URI

    V_x n 0 PWL FILE "code:libcosim_bridge.so:nvc_bridge_init:d2a:<sig>"
    I_x n 0 PWL FILE "code:libcosim_bridge.so:nvc_bridge_init:a2d:<sig>"

VACASK takes the same URI in the file parameter of a pwl vsource/isource

    v_x (n 0) vsrc type="pwl" file="code:libcosim_bridge.so:vacask_bridge_init:d2a:<sig>"

so the same .boundary file and VHDL drive either analog engine
(nvc --xyce-netlist=deck.cir  or  nvc --vacask-netlist=deck.sim).

Supported: R C L K, V I (dc, PULSE, PWL, SIN, EXP, code:), E G (linear),
B (V=/I= expressions -> VACASK behavioral sources), S (VSWITCH, Xyce's smooth
switch law as a behavioral source), D Q M J (VACASK's SPICE-distilled models,
spice/*.osdi), X, .subckt (nested), .param, .model (scoped like SPICE),
.tran, .end; .ic/.options/.print are dropped.  Anything else is reported and
the translation fails, rather than silently dropping analog.

usage: cir2vacask.py deck.cir [-o deck.sim]
"""
import argparse
import re
import sys

# model type -> (osdi file, VACASK module, extra model parameters)
MODEL_TYPES = {
    'd':    ('spice/diode.osdi', 'sp_diode', {}),
    'npn':  ('spice/bjt.osdi', 'sp_bjt', {'type': '1'}),
    'pnp':  ('spice/bjt.osdi', 'sp_bjt', {'type': '-1'}),
    'njf':  ('spice/jfet1.osdi', 'sp_jfet1', {'type': '1'}),
    'pjf':  ('spice/jfet1.osdi', 'sp_jfet1', {'type': '-1'}),
}
# MOSFET level -> osdi/module (nmos/pmos set type=+-1)
MOS_LEVELS = {
    1: ('spice/mos1.osdi', 'sp_mos1'),
    2: ('spice/mos2.osdi', 'sp_mos2'),
    3: ('spice/mos3.osdi', 'sp_mos3'),
    6: ('spice/mos6.osdi', 'sp_mos6'),
    9: ('spice/mos9.osdi', 'sp_mos9'),
    8: ('spice/bsim3v3.osdi', 'sp_bsim3v3'),
    49: ('spice/bsim3v3.osdi', 'sp_bsim3v3'),
    14: ('spice/bsim4v8.osdi', 'sp_bsim4v8'),
    54: ('spice/bsim4v8.osdi', 'sp_bsim4v8'),
}
BUILTIN_MODELS = {
    'vsrc': 'vsource', 'isrc': 'isource',
    'vcvs': 'vcvs', 'vccs': 'vccs', 'mutual': 'mutual',
}
PASSIVE = {
    'r': ('resistor.osdi', 'resistor', 'r'),
    'c': ('capacitor.osdi', 'capacitor', 'c'),
    'l': ('inductor.osdi', 'inductor', 'l'),
}
# Xyce VSWITCH defaults
VSWITCH_DEFAULTS = {'ron': '1', 'roff': '1e6', 'von': '1', 'voff': '0'}


class TranslateError(Exception):
    pass


def num(tok):
    """SPICE number -> VACASK literal.  Both take SI suffixes (meg, mil, k,
    m, u, n, p, f, g, t) followed by an ignored unit; anything that is not a
    number is an expression."""
    t = tok.strip()
    m = re.fullmatch(r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([a-zA-Z_]*)', t)
    if not m:
        return expr(t)
    mant, suf = m.group(1), m.group(2).lower()
    if not suf:
        return mant
    for p in ('meg', 'mil'):
        if suf.startswith(p):
            return mant + p
    if suf[0] in 'afpnumkgt':
        return mant + {'g': 'G', 't': 'T'}.get(suf[0], suf[0])
    return mant     # trailing unit only (5V, 10s)


def ident(n):
    """SPICE name -> VACASK identifier: lowercase, characters VACASK does not
    allow in identifiers (e.g. - in bzx79-18) become _."""
    n = re.sub(r'[^a-z0-9_]', '_', n.lower())
    return '_' + n if n[0].isdigit() else n


def node(n):
    """SPICE node name -> VACASK (numeric names get an n prefix, 0 stays
    ground)."""
    n = n.lower()
    if n == '0':
        return n
    return 'n' + n if re.fullmatch(r'\d+', n) else ident(n)


def split_top(s, sep=','):
    """split s at sep outside parentheses"""
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    out.append(cur)
    return out


def expr(t):
    """SPICE/Xyce expression -> VACASK expression: braces and quotes dropped,
    lowercase, IF(c,a,b) -> (c ? a : b), V(a,b) node names mapped, ** -> pow."""
    t = t.strip()
    if (t.startswith('{') and t.endswith('}')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1]
    t = t.lower()

    def rewrite(s):
        out, i = '', 0
        while i < len(s):
            m = re.match(r'(if|v)\s*\(', s[i:])
            if m and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] == '_')):
                # find matching paren
                j, depth = i + m.end(), 1
                while j < len(s) and depth:
                    depth += {'(': 1, ')': -1}.get(s[j], 0)
                    j += 1
                inner = s[i + m.end():j - 1]
                args = split_top(inner)
                if m.group(1) == 'if':
                    if len(args) != 3:
                        raise TranslateError('IF() needs 3 arguments: ' + inner)
                    c, a, b = (rewrite(x.strip()) for x in args)
                    out += '((%s) ? (%s) : (%s))' % (c, a, b)
                else:
                    out += 'v(%s)' % ', '.join(node(x.strip()) for x in args)
                i = j
            else:
                out += s[i]
                i += 1
        return out

    t = re.sub(r'([a-z_]\w*)\s+\(', r'\1(', t)   # no space before a call's (
    t = rewrite(t)
    if '**' in t:
        raise TranslateError('** operator in expression (use pow): ' + t)
    # SPICE numbers with suffixes inside expressions (e.g. 20u)
    t = re.sub(r'(?<![\w.])(\d+\.?\d*(?:e[+-]?\d+)?)(meg|mil|[afpnumkgt])(?![\w(])',
               lambda m: num(m.group(1) + m.group(2)), t)
    return '(' + t + ')' if re.search(r'[-+*/ ?]', t.lstrip('-+')) else t


def split_params(toks):
    """['a=1', 'b', '=', '2', 'x'] -> positional ['x'], [('a','1'), ('b','2')]"""
    joined = re.sub(r'\s*=\s*', '=', ' '.join(toks))
    pos, kv = [], []
    for tok in re.findall(r'\S+=\{[^}]*\}|\S+=\'[^\']*\'|\{[^}]*\}|\S+', joined):
        if '=' in tok and not tok.startswith('{'):
            k, v = tok.split('=', 1)
            kv.append((k.lower(), v))
        else:
            pos.append(tok)
    return pos, kv


def fmt_params(kv):
    return ' '.join('%s=%s' % (k, num(v)) for k, v in kv)


def logical_lines(text):
    """Join + continuations, strip comments; line 1 is the title."""
    raw = text.splitlines()
    title = raw[0] if raw else ''
    out = []
    for line in raw[1:]:
        s = line.rstrip()
        st = s.lstrip()
        if not st:
            continue
        if st.startswith('*'):
            out.append(('comment', st[1:].strip()))
            continue
        s = re.sub(r'\s;.*$', '', st)
        if s.startswith('+'):
            if out and out[-1][0] == 'line':
                out[-1] = ('line', out[-1][1] + ' ' + s[1:].strip())
            continue
        out.append(('line', s))
    return title, out


def tokenize(line):
    # quoted strings and {expressions} stay whole; ( ) become separate tokens
    toks = []
    for t in re.findall(r'"[^"]*"|\{[^}]*\}|[^\s()]+|[()]', line):
        toks.append(t)
    return toks


class Translator:
    def __init__(self):
        self.loads = []
        self.need_builtin = set()
        self.passive_models = set()
        self.scoped = {}        # subckt path tuple -> {model: (mtype, kv)}
        self.passed = {}        # subckt name -> {param: value} passed by X instances
        self.tstep = self.tstop = None   # from .tran, for SPICE waveform defaults
        self.body = []
        self.control = []

    def load(self, f):
        if f not in self.loads:
            self.loads.append(f)

    # ---- models -------------------------------------------------------------
    @staticmethod
    def parse_model(toks):
        name = toks[1].lower()
        rest = [t for t in toks[2:]]
        mtype = rest[0].lower()
        params = [t for t in rest[1:] if t not in ('(', ')')]
        _, kv = split_params(params)
        return name, mtype, kv

    def lookup(self, path, name):
        for i in range(len(path), -1, -1):
            m = self.scoped.get(tuple(path[:i]), {}).get(name)
            if m:
                return m
        return None

    def model_module(self, name, mtype, kv):
        """-> (osdi, module, VACASK params) or None for models expanded inline"""
        if mtype in ('nmos', 'pmos'):
            level = 1
            rest = []
            for k, v in kv:
                if k == 'level':
                    level = int(float(v))
                elif k != 'version':
                    rest.append((k, v))
            if level not in MOS_LEVELS:
                raise TranslateError('.model %s: MOSFET level %d not supported' % (name, level))
            osdi, module = MOS_LEVELS[level]
            return osdi, module, [('type', '1' if mtype == 'nmos' else '-1')] + rest
        if mtype == 'vdmos':
            return 'spice/vdmos.osdi', 'sp_vdmos', \
                [('type', '-1' if any(k == 'pchan' for k, _ in kv) else '1')] + \
                [(k, v) for k, v in kv if k not in ('pchan', 'nchan')]
        if mtype in MODEL_TYPES:
            osdi, module, extra = MODEL_TYPES[mtype]
            return osdi, module, list(extra.items()) + [(k, v) for k, v in kv if k != 'level']
        if mtype in ('r', 'res', 'c', 'cap', 'l', 'ind'):
            osdi, module, _ = PASSIVE[mtype[0]]
            return osdi, module, kv
        if mtype in ('vswitch', 'sw'):
            return None
        raise TranslateError('.model %s: model type %s not supported' % (name, mtype))

    def dot_model(self, toks, ind):
        name, mtype, kv = self.parse_model(toks)
        mm = self.model_module(name, mtype, kv)
        if mm is None:
            self.body.append('%s// .model %s %s: expanded at its instances' % (ind, name, mtype))
            return
        osdi, module, params = mm
        self.load(osdi)
        self.body.append('%smodel %s %s %s' % (ind, ident(name), module, fmt_params(params)))

    # ---- sources ------------------------------------------------------------
    def source(self, toks, ind, kind):
        name = ident(toks[0])
        n1, n2 = node(toks[1]), node(toks[2])
        rest = toks[3:]
        model = 'vsrc' if kind == 'v' else 'isrc'
        self.need_builtin.add(model)
        params = []
        i = 0
        while i < len(rest):
            t = rest[i].lower()
            if t == 'dc':
                params.append(('dc', num(rest[i + 1])))
                i += 2
            elif t == 'ac':
                params.append(('mag', num(rest[i + 1])))
                i += 2
                if i < len(rest) and re.match(r'^[-+\d.]', rest[i]):
                    params.append(('phase', num(rest[i])))
                    i += 1
            elif t in ('pulse', 'pwl', 'sin', 'exp'):
                j = i + 1
                args = []
                if j < len(rest) and rest[j] == '(':
                    j += 1
                    while j < len(rest) and rest[j] != ')':
                        args.append(rest[j])
                        j += 1
                    j += 1
                else:
                    while j < len(rest) and not re.match(r'^(dc|ac)$', rest[j].lower()):
                        args.append(rest[j])
                        j += 1
                i = j
                params += self.waveform(t, args, name)
            elif re.match(r'^[-+\d.{]', t):
                params.append(('dc', num(rest[i])))
                i += 1
            else:
                raise TranslateError('%s: cannot translate source argument %r' % (name, rest[i]))
        self.body.append('%s%s (%s %s) %s %s' % (
            ind, name, n1, n2, model, ' '.join('%s=%s' % kv for kv in params)))

    def waveform(self, fn, args, name):
        if fn == 'pwl':
            if len(args) >= 2 and args[0].lower() == 'file':
                uri = args[1].strip('"')
                if not uri.startswith('code:'):
                    raise TranslateError('%s: PWL FILE %s (only code: URIs)' % (name, uri))
                uri = uri.replace(':nvc_bridge_init:', ':vacask_bridge_init:')
                return [('type', '"pwl"'), ('file', '"%s"' % uri)]
            vals = [num(a) for a in args]
            return [('type', '"pwl"'), ('wave', '[' + ', '.join(vals) + ']')]
        vals = [num(a) for a in args]
        if fn == 'pulse':
            # SPICE defaults: TD=0, TR=TF=TSTEP (also when given as 0),
            # PW=PER=TSTOP; VACASK needs rise>0 and takes width=0 as a
            # triangle, so fill them in (no period = a single pulse).
            keys = ['val0', 'val1', 'delay', 'rise', 'fall', 'width', 'period']
            p = dict(zip(keys, vals))
            p.setdefault('delay', '0')
            for k in ('rise', 'fall'):
                if k not in p or re.fullmatch(r'0+\.?0*', p[k]):
                    p[k] = self.tstep
            if 'width' not in p:
                p['width'] = self.tstop
            return [('type', '"pulse"')] + [(k, p[k]) for k in keys if k in p]
        if fn == 'sin':
            keys = ['sinedc', 'ampl', 'freq', 'delay', 'theta', 'tdphase']
            return [('type', '"sine"')] + list(zip(keys, vals))
        if fn == 'exp':
            keys = ['val0', 'val1', 'delay', 'tau1', 'td2', 'tau2']
            return [('type', '"exp"')] + list(zip(keys, vals))
        raise TranslateError('%s: waveform %s' % (name, fn))

    def vswitch(self, name, nodes, kv, ind):
        """Xyce/PSpice smooth switch: Ron above VON, Roff below VOFF, and in
        between R = exp(Lm + 3 Lr (vc-Vm)/(2 Vd) - 2 Lr (vc-Vm)^3/Vd^3) with
        Lm = ln(sqrt(Ron Roff)), Lr = ln(Ron/Roff), Vm = (VON+VOFF)/2,
        Vd = VON-VOFF.  Written for VON > VOFF (the usual case)."""
        p = dict(VSWITCH_DEFAULTS)
        p.update({k: num(v) for k, v in kv})
        np_, nn, cp, cn = nodes
        vc = 'v(%s, %s)' % (cp, cn)
        vm = '((%s)+(%s))/2' % (p['von'], p['voff'])
        vd = '((%s)-(%s))' % (p['von'], p['voff'])
        lm = 'ln(sqrt((%s)*(%s)))' % (p['ron'], p['roff'])
        lr = 'ln((%s)/(%s))' % (p['ron'], p['roff'])
        mid = 'exp(%s + 3*%s*(%s-%s)/(2*%s) - 2*%s*pow(%s-%s, 3)/pow(%s, 3))' % (
            lm, lr, vc, vm, vd, lr, vc, vm, vd)
        r = '(%s >= %s ? %s : (%s <= %s ? %s : %s))' % (
            vc, p['von'], p['ron'], vc, p['voff'], p['roff'], mid)
        self.body.append('%s%s (%s %s) i=v(%s, %s)/%s' % (ind, name, np_, nn, np_, nn, r))

    # ---- instances ----------------------------------------------------------
    def instance(self, toks, ind, path):
        name = ident(toks[0])
        c = name[0]
        if c in PASSIVE:
            osdi, module, pname = PASSIVE[c]
            pos, kv = split_params(toks[3:])
            n = [node(t) for t in toks[1:3]]
            model = None
            if pos and self.lookup(path, pos[0].lower()):
                model = ident(pos.pop(0))
            if model is None:
                self.load(osdi)
                model = 'vc_' + module
                self.passive_models.add((model, module))
            if pos:
                kv = [(pname, pos[0])] + kv
            if c == 'r':
                kv = [(k, v) for k, v in kv if k not in ('tc1', 'tc2')]
            self.body.append('%s%s (%s %s) %s %s' % (ind, name, n[0], n[1], model, fmt_params(kv)))
        elif c in 'vi':
            self.source(toks, ind, c)
        elif c == 'b':
            # Bname n+ n- V={expr} | I={expr}
            m = re.match(r'(\S+)\s+(\S+)\s+(\S+)\s+([vi])\s*=\s*(.*)$', ' '.join(toks), re.I)
            if not m:
                raise TranslateError('%s: B source needs V= or I=' % name)
            kind = m.group(4).lower()
            e = expr(m.group(5).replace('( ', '(').replace(' )', ')'))
            self.body.append('%s%s (%s %s) %s=%s' % (ind, name, node(m.group(2)), node(m.group(3)), kind, e))
        elif c == 's':
            nodes = [node(t) for t in toks[1:5]]
            mname = toks[5].lower()
            m = self.lookup(path, mname)
            if not m or m[0] not in ('vswitch', 'sw'):
                raise TranslateError('%s: switch model %s not found' % (name, mname))
            self.vswitch(name, nodes, m[1], ind)
        elif c in 'eg':
            if len(toks) != 6:
                raise TranslateError('%s: only linear 4-node %s sources are supported' % (name, c.upper()))
            model = 'vcvs' if c == 'e' else 'vccs'
            self.need_builtin.add(model)
            n = ' '.join(node(t) for t in toks[1:5])
            self.body.append('%s%s (%s) %s gain=%s' % (ind, name, n, model, num(toks[5])))
        elif c == 'k':
            self.need_builtin.add('mutual')
            self.body.append('%s%s () mutual ind1="%s" ind2="%s" k=%s' % (
                ind, name, ident(toks[1]), ident(toks[2]), num(toks[3])))
        elif c in 'dqmj':
            nnodes = {'d': 2, 'q': 3, 'm': 4, 'j': 3}[c]
            nodes = [node(t) for t in toks[1:1 + nnodes]]
            rest = toks[1 + nnodes:]
            if c == 'q' and rest and not self.lookup(path, rest[0].lower()):
                nodes.append(node(rest.pop(0)))    # substrate
            if not rest or not self.lookup(path, rest[0].lower()):
                raise TranslateError('%s: model %s not defined' % (name, rest[0] if rest else '?'))
            mtype = self.lookup(path, rest[0].lower())[0]
            model = ident(rest[0])
            pos, kv = split_params(rest[1:])
            if pos:
                kv = [('area', pos[0])] + kv
            if c == 'q' and len(nodes) == 3:
                nodes.append('0')
            if mtype == 'vdmos':
                nodes = nodes[:3] + ['vdmos_t_' + name, 'vdmos_tc_' + name]
            self.body.append('%s%s (%s) %s %s' % (ind, name, ' '.join(nodes), model, fmt_params(kv)))
        elif c == 'x':
            pos, kv = split_params([t for t in toks[1:] if t.lower() != 'params:'])
            sub = ident(pos[-1])
            nodes = ' '.join(node(t) for t in pos[:-1])
            self.body.append('%s%s (%s) %s %s' % (ind, name, nodes, sub, fmt_params(kv)))
        else:
            raise TranslateError('%s: device type %s not supported' % (name, c.upper()))

    # ---- deck ---------------------------------------------------------------
    def translate(self, text):
        title, lines = logical_lines(text)
        # Pre-pass: model cards per subcircuit scope (SPICE allows a .model
        # after its first use; lookups walk outward like SPICE scoping)
        path = []
        for kind, line in lines:
            if kind != 'line':
                continue
            toks = tokenize(line)
            h = toks[0].lower()
            if h == '.subckt':
                path.append(toks[1].lower())
            elif h == '.ends':
                path.pop()
            elif h == '.model':
                name, mtype, kv = self.parse_model(toks)
                self.scoped.setdefault(tuple(path), {})[name] = (mtype, kv)
            elif h == '.tran':
                pos, _ = split_params([t for t in toks[1:] if t.lower() != 'uic'])
                self.tstep, self.tstop = num(pos[0]), num(pos[1])
            elif h.startswith('x'):
                pos, kv = split_params([t for t in toks[1:] if t.lower() != 'params:'])
                d = self.passed.setdefault(pos[-1].lower(), {})
                for k, v in kv:
                    d.setdefault(k, v)
        path = []
        for kind, line in lines:
            ind = '  ' * len(path)
            if kind == 'comment':
                self.body.append('%s// %s' % (ind, line))
                continue
            toks = tokenize(line)
            head = toks[0].lower()
            if head == '.subckt':
                pos, kv = split_params([t for t in toks[2:] if t.lower() != 'params:'])
                self.body.append('%ssubckt %s (%s)' % (ind, ident(toks[1]), ' '.join(node(p) for p in pos)))
                # Xyce accepts instance parameters a subcircuit never declares
                # (SIMetrix emits them), VACASK needs them declared
                declared = {k for k, _ in kv}
                for k in self.passed.get(toks[1].lower(), {}):
                    if k not in declared:
                        kv.append((k, VSWITCH_DEFAULTS.get(k, '0')))
                if kv:
                    self.body.append('%s  parameters %s' % (ind, fmt_params(kv)))
                path.append(toks[1].lower())
            elif head == '.ends':
                path.pop()
                self.body.append('%sends' % ('  ' * len(path)))
            elif head == '.model':
                self.dot_model(toks, ind)
            elif head == '.param':
                _, kv = split_params(toks[1:])
                self.body.append('%sparameters %s' % (ind, fmt_params(kv)))
            elif head == '.tran':
                pos, _ = split_params([t for t in toks[1:] if t.lower() != 'uic'])
                args = ['step=%s' % num(pos[0]), 'stop=%s' % num(pos[1])]
                if len(pos) > 2 and float(re.sub(r'[a-zA-Z]+$', '', pos[2]) or 0) > 0:
                    args.append('start=%s' % num(pos[2]))
                if len(pos) > 3:
                    args.append('maxstep=%s' % num(pos[3]))
                if any(t.lower() == 'uic' for t in toks):
                    args.append('icmode="uic"')
                self.control.append('  analysis tran1 tran ' + ' '.join(args))
            elif head in ('.print', '.options', '.option', '.probe', '.graph', '.ic', '.nodeset'):
                self.body.append('%s// dropped: %s' % (ind, line))
            elif head == '.end':
                break
            elif head.startswith('.'):
                raise TranslateError('unsupported control line: %s' % line)
            else:
                self.instance(toks, ind, path)
        if not self.control:
            raise TranslateError('no .tran analysis in deck')
        out = [title.lstrip('*').strip() or 'cir2vacask', '']
        out += ['load "%s"' % f for f in self.loads]
        out.append('')
        for b in sorted(self.need_builtin):
            out.append('model %s %s' % (b, BUILTIN_MODELS[b]))
        for m, module in sorted(self.passive_models):
            out.append('model %s %s' % (m, module))
        out.append('')
        out += self.body
        # SPICE controls the LTE of charges, not of the ddt() currents OpenVAF
        # turns into implicit equations in the SPICE-distilled models (spice/*.osdi);
        # checking those collapses the step when a device is driven hard
        options = ['  options tran_lteimplicit=0']
        out += ['', 'control'] + options + self.control + ['endc', '']
        return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('deck')
    ap.add_argument('-o', '--output')
    a = ap.parse_args()
    text = open(a.deck).read()
    try:
        sim = Translator().translate(text)
    except TranslateError as e:
        sys.exit('cir2vacask: %s: %s' % (a.deck, e))
    out = a.output or re.sub(r'\.(cir|sp|net)$', '', a.deck) + '.sim'
    open(out, 'w').write(sim)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
