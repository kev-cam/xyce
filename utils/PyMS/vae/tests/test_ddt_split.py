"""A contribution that adds ddt() to a static term must conduct the static
term (PYMS_BUGS.md P5, 2026-10-24).

THE BUG: ``analog I(p, n) <+ V(p, n) / r + ddt(c * V(p, n));`` built a device
that conducts nothing: Xyce's out node of the reference divider (1 V source,
the module, 1 kOhm to ground) read 0.0 with exit status 0 where VACASK and the
analytic value give 0.5. GiNaCEmitter._emit_contribution unwrapped the first
ddt(...) it found and sent the WHOLE right-hand side to the charge vector, so
the conductance became a charge (eval.cpp: ``F[0] = 0.0; Q[0] = 1e-3*V_p ...``).
A second ddt() in the same right-hand side was left as a call the C++ compiler
did not know, and a ddt() reaching the contribution through a variable
(``q = ddt(x); I(p) <+ g*V(p,n) + q``, the sg13g2 capacitor's form) failed the
same way. Real models write these forms: FBH HBT's ``I(bi,ci) <+ Ibx + ddt(qb1
+ qbtra)``, L_UTSOI's ``M*TYPE*(ddt(Qg) + ddt(Qgse) + ...)``, MVSG's
``ddt(qgsfp4) + ddt(minc * V(gi,fp3))``, EKV's ``ddt_QD = ddt(QD)``.

THE FIX: ddt() is kept as an opaque GiNaC function (vae_ddt) while the
right-hand side is built; _vae_split_ddt in the generated program then
expands the expression and moves every linear term c*ddt(q) (c constant) to
the charge, the rest to the current. A variable whose value carries ddt() is
inlined into its readers. What the F(V) + d/dt Q(V) form of a Xyce device
cannot hold is a build error naming file, line and construct, never a silent
0: ddt() under a function or a power, ddt() times ddt(), a voltage-dependent
factor on ddt(), idt(). The device shell turns the missing math library into
a netlist error (test_hdl_model_card_params.py's fix, 2026-10-24).

Each test runs Xyce end-to-end on the reference divider with private
PYMS_CACHE / PYMS_VAE_CACHE directories and PYMS_DIR set to the PyMS copy this
file sits in, so the test exercises whichever copy it is run from. Skips when
there is no Xyce binary.

Run:
    cd utils/PyMS && python3 vae/tests/test_ddt_split.py
$XYCE names the binary (default /usr/local/src/xyce-build/src/Xyce);
HDLTEST_KEEP=1 keeps the scratch directories.
"""
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
# The PyMS copy this test belongs to: <PyMS>/vae/tests/test_ddt_split.py
PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TIMEOUT = 600          # seconds per Xyce run (the first builds two libraries)
RTOL = 1e-6            # dc agreement with the analytic value
OUT_DC = 0.5           # the divider's out node, 1 V through r = 1k into 1k
R, R2 = 1000.0, 1000.0
FREQ = 1000.0          # the transient's sine source

HEAD = '''`include "disciplines.vams"
`include "constants.vams"
module a1(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  parameter real c = {c};
'''


def module(body, c='1e-12', decl=''):
    """The reference module around one analog body; c is the capacitance."""
    return HEAD.format(c=c) + decl + '  ' + body + '\nendmodule\n'


DC_DECK = '''* ddt split: the reference divider, dc
.hdl "a1.va"
V1 in 0 DC 1.0
.model a1__t a1
ya1 X1 in out a1__t
R2 out 0 1000.0
.options device temp=27.0 tnom=27.0 gmin=1e-12
.OP
.print dc format=noindex v(out)
.end
'''

TRAN_DECK = '''* ddt split: the reference divider, 1 kHz sine, 2 ms
.hdl "a1.va"
V1 in 0 SIN(0 1 1k)
.model a1__t a1
ya1 X1 in out a1__t
R2 out 0 1000.0
.options device temp=27.0 tnom=27.0 gmin=1e-12
.tran 1u 2m 0 2u
.print tran format=noindex v(out)
.end
'''

# The forms of P5 and its neighbours: every one is the resistor r in parallel
# with the capacitor c, so out = 0.5 at dc. (decl, body)
MIXED_FORMS = {
    'static_plus_ddt':      ('', 'analog I(p, n) <+ V(p, n) / r + ddt(c * V(p, n));'),
    'ddt_first':            ('', 'analog I(p, n) <+ ddt(c * V(p, n)) + V(p, n) / r;'),
    'ddt_subtracted':       ('', 'analog I(p, n) <+ V(p, n) / r - ddt(c * V(p, n));'),
    'constant_times_ddt':   ('', 'analog I(p, n) <+ V(p, n) / r + 2 * ddt(c * V(p, n));'),
    'two_ddt_terms':        ('', 'analog I(p, n) <+ V(p, n) / r + ddt(c * V(p, n)) + ddt(c * V(p, n));'),
    'parenthesised_sum':    ('', 'analog I(p, n) <+ (V(p, n) / r + ddt(c * V(p, n)));'),
    'scaled_parenthesised': ('', 'analog I(p, n) <+ 0.5 * (2 * V(p, n) / r + ddt(2 * c * V(p, n)));'),
    'ddt_of_zero':          ('', 'analog I(p, n) <+ V(p, n) / r + ddt(0.0);'),
    'ddt_with_tolerance':   ('', 'analog I(p, n) <+ V(p, n) / r + ddt(c * V(p, n), 1e-15);'),
    'ddt_of_variable':      ('  real q;\n',
                             'analog begin\n    q = c * V(p, n);\n'
                             '    I(p, n) <+ V(p, n) / r + ddt(q);\n  end'),
    'sum_through_variable': ('  real i;\n',
                             'analog begin\n    i = V(p, n) / r + ddt(c * V(p, n));\n'
                             '    I(p, n) <+ i;\n  end'),
    'negated_ddt_group':    ('', 'analog I(p, n) <+ V(p, n) / r - (ddt(-c * V(p, n)));'),
    'divided_ddt':          ('', 'analog I(p, n) <+ V(p, n) / r + ddt(c * V(p, n)) / 0.5;'),
    'parameter_factor':     ('', 'analog I(p, n) <+ V(p, n) / r + (r / 1000.0) * ddt(c * V(p, n));'),
    'gated_mixed':          ('', 'analog begin\n    if (V(p, n) > -10.0)\n'
                                 '      I(p, n) <+ V(p, n) / r + ddt(c * V(p, n));\n'
                                 '    else\n      I(p, n) <+ 0.0;\n  end'),
    # the sg13g2 capacitor's shape: ddt() through two variables, negated once
    'ddt_variable_chain':   ('  real q, cc, ceq;\n',
                             'analog begin\n    q = c * V(p, n);\n    cc = ddt(q);\n'
                             '    ceq = cc;\n    I(p) <+ V(p, n) / r + ceq;\n'
                             '    I(n) <+ V(n, p) / r - ceq;\n  end'),
}

# What the F + d/dt Q form cannot hold: a build error naming the construct.
# (body, a phrase of the message)
ERROR_FORMS = {
    'ddt_inside_function':      ('analog I(p, n) <+ V(p, n) / r + exp(ddt(c * V(p, n))) - 1.0;',
                                 'ddt() inside exp(...)'),
    'idt':                      ('analog I(p, n) <+ V(p, n) / r + idt(0.0 * V(p, n));',
                                 'idt()'),
    'voltage_dependent_factor': ('analog I(p, n) <+ V(p, n) / r + V(p, n) * ddt(c * V(p, n));',
                                 'depends on a node voltage'),
    'ddt_times_ddt':            ('analog I(p, n) <+ V(p, n) / r + ddt(c * V(p, n)) * ddt(c * V(p, n));',
                                 'non-linearly'),
}

# Forms that ran right before the fix and must still.
PURE_CAPACITOR = 'analog I(p, n) <+ ddt(c * V(p, n));'
TWO_STATEMENTS = ('analog begin\n    I(p, n) <+ V(p, n) / r;\n'
                  '    I(p, n) <+ ddt(c * V(p, n));\n  end')
# A voltage contribution with ddt(): the same dc value as without it (the
# value itself is the voltage contribution's business, PYMS_BUGS.md P7).
V_PLAIN = 'analog V(p, n) <+ 0.5;'
V_WITH_DDT = 'analog V(p, n) <+ 0.5 + ddt(c * V(n));'


def run_xyce(va_text, deck, keep_name):
    """Run one deck with the module in a private scratch directory and
    private caches. Returns (exit status, stdout+stderr, prn text or '')."""
    work = tempfile.mkdtemp(prefix=f'pyms_ddt_{keep_name}_')
    try:
        with open(os.path.join(work, 'a1.va'), 'w') as f:
            f.write(va_text)
        with open(os.path.join(work, 'deck.cir'), 'w') as f:
            f.write(deck)
        env = dict(os.environ)
        env['PYMS_DIR'] = PYMS_DIR
        env['PYMS_CACHE'] = os.path.join(work, 'cache_hdl')
        env['PYMS_VAE_CACHE'] = os.path.join(work, 'cache_vae')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        r = subprocess.run([XYCE, 'deck.cir'], cwd=work, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=TIMEOUT)
        prn = ''
        p = os.path.join(work, 'deck.cir.prn')
        if os.path.exists(p):
            with open(p) as f:
                prn = f.read()
        return r.returncode, r.stdout, prn
    finally:
        if not os.environ.get('HDLTEST_KEEP'):
            shutil.rmtree(work, ignore_errors=True)


def dc_out(prn):
    """The one V(OUT) value of a .print dc format=noindex file."""
    for line in prn.splitlines():
        line = line.strip()
        if not line or line.startswith('V(') or line.startswith('End'):
            continue
        return float(line.split()[0])
    raise AssertionError(f'no dc value in prn:\n{prn}')


def tran_points(prn):
    """(time, v(out)) pairs of a .print tran format=noindex file."""
    pts = []
    for line in prn.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return pts


def sine_fit(points, t_from):
    """Least-squares fit of out(t) = A sin(wt) + B cos(wt) + C over t >= t_from;
    returns (amplitude, phase in degrees, offset)."""
    w = 2 * math.pi * FREQ
    rows = [(math.sin(w * t), math.cos(w * t), 1.0, v) for t, v in points if t >= t_from]
    # normal equations of the 3-parameter fit
    n = len(rows)
    s = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    for r_ in rows:
        for i in range(3):
            b[i] += r_[i] * r_[3]
            for j in range(3):
                s[i][j] += r_[i] * r_[j]
    # solve 3x3 by Gaussian elimination
    m = [s[i] + [b[i]] for i in range(3)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda k: abs(m[k][i]))
        m[i], m[piv] = m[piv], m[i]
        for k in range(3):
            if k != i:
                f = m[k][i] / m[i][i]
                m[k] = [a - f * c for a, c in zip(m[k], m[i])]
    A, B, C = (m[i][3] / m[i][i] for i in range(3))
    assert n > 50, f'too few transient points after {t_from}: {n}'
    return math.hypot(A, B), math.degrees(math.atan2(B, A)), C


def analytic_divider(c):
    """|H| and phase (degrees) of out/in for r || c into R2 at FREQ."""
    w = 2 * math.pi * FREQ
    g1, g2 = 1.0 / R, 1.0 / R2
    h = complex(g1, w * c) / complex(g1 + g2, w * c)
    return abs(h), math.degrees(math.atan2(h.imag, h.real))


@unittest.skipUnless(os.path.exists(XYCE), f'no Xyce binary at {XYCE} (set $XYCE)')
class TestDdtSplit(unittest.TestCase):

    def _dc(self, name, va_text):
        rc, log, prn = run_xyce(va_text, DC_DECK, name)
        self.assertEqual(rc, 0, f'{name}: Xyce exit {rc}\n{log[-3000:]}')
        return dc_out(prn), log

    def test_static_term_beside_ddt_conducts(self):
        """Every mixed form conducts the static term: out = 0.5 at dc."""
        for name, (decl, body) in MIXED_FORMS.items():
            with self.subTest(form=name):
                out, _ = self._dc(name, module(body, decl=decl))
                self.assertAlmostEqual(out, OUT_DC, delta=RTOL * OUT_DC,
                                       msg=f'{name}: out = {out!r}, the static term '
                                           f'V(p, n) / r was lost (PYMS_BUGS.md P5)')

    def test_pure_forms_unchanged(self):
        """The pure capacitor and the two-statement form ran right before and still do."""
        out, _ = self._dc('pure_capacitor', module(PURE_CAPACITOR))
        self.assertLess(abs(out), 1e-9, f'pure capacitor: out = {out!r}, expected 0 at dc')
        out, _ = self._dc('two_statements', module(TWO_STATEMENTS))
        self.assertAlmostEqual(out, OUT_DC, delta=RTOL * OUT_DC,
                               msg=f'two statements: out = {out!r}')

    def test_voltage_contribution_with_ddt(self):
        """V(p, n) <+ 0.5 + ddt(...) has the dc value of V(p, n) <+ 0.5."""
        plain, _ = self._dc('v_plain', module(V_PLAIN))
        with_ddt, _ = self._dc('v_with_ddt', module(V_WITH_DDT))
        self.assertAlmostEqual(with_ddt, plain, delta=RTOL * abs(plain),
                               msg=f'V() with ddt: {with_ddt!r} vs plain {plain!r}')

    def test_charge_part_in_transient(self):
        """In a transient the one-statement form equals the two-statement form
        and both match the analytic divider: the ddt() term IS the charge."""
        c = 1e-7
        fits = {}
        for name, body in (('static_plus_ddt', MIXED_FORMS['static_plus_ddt'][1]),
                           ('two_statements', TWO_STATEMENTS)):
            rc, log, prn = run_xyce(module(body, c=repr(c)), TRAN_DECK, name)
            self.assertEqual(rc, 0, f'{name}: Xyce exit {rc}\n{log[-3000:]}')
            fits[name] = sine_fit(tran_points(prn), 1e-3)
        (a1, p1, _), (a2, p2, _) = fits['static_plus_ddt'], fits['two_statements']
        self.assertAlmostEqual(a1, a2, delta=1e-4 * a2,
                               msg=f'amplitudes differ: {a1!r} vs {a2!r}')
        self.assertAlmostEqual(p1, p2, delta=0.01, msg=f'phases differ: {p1!r} vs {p2!r}')
        amp, phase = analytic_divider(c)
        self.assertAlmostEqual(a1, amp, delta=1e-3 * amp,
                               msg=f'amplitude {a1!r} vs analytic {amp!r}')
        self.assertAlmostEqual(p1, phase, delta=0.1,
                               msg=f'phase {p1!r} deg vs analytic {phase!r}')

    def test_unbuildable_forms_are_errors(self):
        """What the F + d/dt Q form cannot hold stops Xyce with a message that
        names the file, the line and the construct; never exit 0 with out = 0."""
        for name, (body, phrase) in ERROR_FORMS.items():
            with self.subTest(form=name):
                rc, log, prn = run_xyce(module(body), DC_DECK, name)
                self.assertNotEqual(rc, 0, f'{name}: Xyce exit 0 with prn:\n{prn}')
                self.assertIn(phrase, log, f'{name}: message lacks {phrase!r}:\n{log[-3000:]}')
                # the source line of the contribution (line 8 of a1.va), not
                # the line of the preprocessed text with the headers inlined
                self.assertRegex(log, r'a1\.va:8:', f'{name}: no file:line 8 in\n{log[-3000:]}')
                # Xyce wraps its error lines: compare with blanks folded
                self.assertIn('math library for module a1 did not build', ' '.join(log.split()),
                              f'{name}: the shell did not report the missing library')


if __name__ == '__main__':
    unittest.main(verbosity=2)
