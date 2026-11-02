#!/usr/bin/env python3
"""Command-line macros for PyMS: ``xyce_device_gen.py --define NAME[=VALUE]``,
the ``PYMS_DEFINES`` environment variable, ``build_vae_so.py --define``.

WHY (2026-10-24, PYMS1-I; PYMS_BUGS.md P11): openvaf-r takes ``-D NAME[=VALUE]``
and vamos's spectre personality passes ``-va,define`` macros through it on
VACASK; PyMS had no such channel, so the same Verilog-A could not be built the
same way on Xyce.  Xyce runs PyMS as ``python3 xyce_device_gen.py <va> --output
<cache>`` (N_DEV_PyMS.C) with no further arguments, so the only route through
an ``.hdl`` line is the environment: PYMS_DEFINES, shell words ``NAME`` or
``NAME=VALUE``.  The macros must reach every pass that preprocesses the .va:
the attribute scan, the shell generation, and the math-library build that
the generated shell runs later (build_vae_so.py), whose .so cache must key
on them; and the device-shell cache of N_DEV_PyMS.C, which keys on the .va's
path + content only, must not hand a shell built with other macros to the
next run (xyce_device_gen._record_defines drops its identity record).

The preprocess/CLI tests need no engine.  The Xyce class runs the module
below four times in ONE cache: without macros (r = 1k, out = 0.5), with
PYMS_DEFINES=RVAL=3000 (r = 3k, out = 0.25), without again (0.5), with again
(0.25): the second run proves the macro reaches the math library, the third
that a shell built with macros is not reused without them, the fourth that
the device-shell cache of N_DEV_PyMS.C keys on PYMS_DEFINES (its identity
hash covers the variable since 2026-10-25; a Xyce whose hash covers only the
.va's path + content hands run b and run d the shell of run a, 0.5).  Set
XYCE to the Xyce under test.  Skips without a Xyce binary.

Run:  python3 vae/tests/test_define_option.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

import vae  # noqa: E402
import vae.preprocess as _pp  # noqa: E402
from vae.preprocess import preprocess_source, PreprocessError  # noqa: E402
from vae.parser import parse_file  # noqa: E402

# Absent before the fix: the tests that need them fail on it, cleanly.
parse_define_option = getattr(_pp, 'parse_define_option', None)
__version__ = getattr(vae, '__version__', None)

GEN = os.path.join(_PYMS_DIR, 'vae', 'xyce_device_gen.py')
XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
RTOL = 1e-6

VA_D1 = """`include "disciplines.vams"
module d1(p, n);
  inout p, n;
  electrical p, n;
`ifdef RVAL
  parameter real r = `RVAL;
`else
  parameter real r = 1k;
`endif
  analog I(p, n) <+ V(p, n) / r;
endmodule
"""

DECK = """* d1 in a 1k divider: out = 1k / (r + 1k)
.hdl "d1.va"
V1 in 0 1
.model m1 d1
YD1 X1 in out m1
R2 out 0 1k
.op
.print dc v(out)
.end
"""


def _run(cmd, env=None, cwd=None):
    e = dict(os.environ)
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e.pop("PYMS_DEFINES", None)
    if env:
        e.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                          env=e, cwd=cwd)


class PreprocessTests(unittest.TestCase):

    def test_parse_define_option(self):
        self.assertIsNotNone(parse_define_option, "preprocess.parse_define_option is missing")
        self.assertEqual(parse_define_option('RVAL=3000'), ('RVAL', '3000'))
        self.assertEqual(parse_define_option('X = 2 * 3'), ('X', '2 * 3'))
        self.assertEqual(parse_define_option('FOO'), ('FOO', ''))
        self.assertEqual(parse_define_option('_x1=a=b'), ('_x1', 'a=b'))
        for bad in ('1X=2', '=3', '', 'a b=1'):
            with self.assertRaises(PreprocessError, msg=bad):
                parse_define_option(bad)

    def test_define_takes_the_ifdef_branch_and_expands(self):
        out = preprocess_source(VA_D1, filename='d1.va', defines=['RVAL=3000'])
        self.assertIn('parameter real r = 3000;', out)
        self.assertNotIn('1k', out)
        out = preprocess_source(VA_D1, filename='d1.va')
        self.assertIn('parameter real r = 1k;', out)
        self.assertNotIn('3000', out)

    def test_bare_define_is_empty_like_a_define_line(self):
        src = "`ifdef FOO\nreal a; `FOO\n`else\nreal b;\n`endif\n"
        self.assertEqual(preprocess_source(src, defines=['FOO']).strip(), 'real a;')
        self.assertEqual(preprocess_source(src).strip(), 'real b;')

    def test_source_define_wins_over_the_option(self):
        src = "`define RVAL 7\n`ifdef RVAL\nx = `RVAL;\n`endif\n"
        self.assertEqual(preprocess_source(src, defines=['RVAL=3000']).strip(), 'x = 7;')

    def test_source_define_replaces_the_option_from_its_line_on(self):
        src = "x = `R;\n`define R 7\ny = `R;\n"
        out = preprocess_source(src, defines=['R=3'])
        self.assertEqual([ln for ln in out.split('\n') if ln.strip()], ['x = 3;', 'y = 7;'])

    def test_command_line_macro_is_named_in_errors(self):
        with self.assertRaises(PreprocessError) as cm:
            preprocess_source("x = `A;\n", filename='m.va', defines=['A=`B'])
        msg = str(cm.exception)
        self.assertIn('m.va:1: `B is not defined', msg)
        self.assertIn('the body of `A (defined at <command line>:0)', msg)

    def test_parse_file_defines(self):
        work = tempfile.mkdtemp(prefix="pyms_define_")
        try:
            va = os.path.join(work, 'd1.va')
            with open(va, 'w') as f:
                f.write(VA_D1)
            self.assertEqual(parse_file(va, defines=['RVAL=3000']).params[0].default.strip(), '3000')
            self.assertEqual(parse_file(va).params[0].default.strip(), '1000.0')
        finally:
            shutil.rmtree(work, ignore_errors=True)


class CliTests(unittest.TestCase):
    """xyce_device_gen.py's command line, run as Xyce runs it."""

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="pyms_define_cli_")
        self.va = os.path.join(self.work, 'd1.va')
        with open(self.va, 'w') as f:
            f.write(VA_D1)

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def _gen(self, *args, env=None):
        out = os.path.join(self.work, 'out_%d' % len(os.listdir(self.work)))
        os.makedirs(out)
        r = _run([sys.executable, GEN, self.va, '--output', out] + list(args), env=env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('Module: d1, Ports:', r.stdout)
        with open(os.path.join(out, 'N_DEV_PYMS_D1.C')) as f:
            return out, f.read()

    def test_version_option(self):
        self.assertIsNotNone(__version__, "vae.__version__ is missing")
        r = _run([sys.executable, GEN, '--version'])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), 'PyMS %s' % __version__)

    def test_no_defines_generates_the_same_shell_as_before(self):
        _, text = self._gen()
        self.assertNotIn('--define', text)
        self.assertNotIn('__DEFINES__', text)

    def test_define_option_reaches_the_shell(self):
        for args in (['--define', 'RVAL=3000'], ['--define=RVAL=3000'],
                     ['-D', 'RVAL=3000'], ['-DRVAL=3000']):
            _, text = self._gen(*args)
            self.assertIn("--define 'RVAL=3000'", text, args)
            self.assertIn('__DEFINES__=', text, args)
            self.assertIn('build_vae_so.py', text)

    def test_pyms_defines_environment(self):
        _, text = self._gen(env={'PYMS_DEFINES': "RVAL=3000 'X=2 * 3' FOO"})
        self.assertIn("--define 'RVAL=3000' --define 'X=2 * 3' --define 'FOO'", text)

    def test_record_and_identity_drop(self):
        out, _ = self._gen()
        rec = os.path.join(out, 'pyms_d1.so.defines')
        src = os.path.join(out, 'pyms_d1.so.src')
        self.assertFalse(os.path.exists(rec))        # no macros, no record
        with open(src, 'w') as f:                     # N_DEV_PyMS.C's identity record
            f.write("%s\nabc\n" % self.va)
        r = _run([sys.executable, GEN, self.va, '--output', out, '--define', 'RVAL=3000'])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(os.path.exists(src))         # other macros: shell must be rebuilt
        with open(rec) as f:
            self.assertEqual(f.read(), 'RVAL=3000\n')
        with open(src, 'w') as f:
            f.write("%s\nabc\n" % self.va)
        r = _run([sys.executable, GEN, self.va, '--output', out, '--define', 'RVAL=3000'])
        self.assertTrue(os.path.exists(src))          # same macros: record kept
        r = _run([sys.executable, GEN, self.va, '--output', out])
        self.assertFalse(os.path.exists(src))         # back to none: rebuilt again
        with open(rec) as f:
            self.assertEqual(f.read(), '')

    def test_bad_define_is_a_clear_error(self):
        r = _run([sys.executable, GEN, self.va, '--output', self.work, '--define', '1X=2'])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("'1X=2'", r.stdout + r.stderr)
        self.assertIn('not a macro name', r.stdout + r.stderr)

    def test_unknown_argument_is_an_error(self):
        r = _run([sys.executable, GEN, self.va, '--output', self.work, '--frobnicate'])
        self.assertEqual(r.returncode, 1)
        self.assertIn("unknown argument '--frobnicate'", r.stderr)


def _xyce_env(work, defines=None):
    env = dict(os.environ)
    env["PYMS_DIR"] = _PYMS_DIR
    env["PYMS_CACHE"] = os.path.join(work, "hdl_cache")
    env["PYMS_VAE_CACHE"] = os.path.join(work, "vae_cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.pop("VAE_SO_PATH", None)
    env.pop("VAE_SO_DIR", None)
    env.pop("PYMS_DEFINES", None)
    if defines:
        env["PYMS_DEFINES"] = defines
    return env


def _v_out(prn_text):
    lines = [ln for ln in prn_text.splitlines() if ln.strip()]
    header = [h.upper() for h in lines[0].split()]
    return float(lines[1].split()[header.index('V(OUT)')])


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce at %s" % XYCE)
class XyceTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix="pyms_define_xyce_")
        with open(os.path.join(cls.work, "d1.va"), "w") as f:
            f.write(VA_D1)
        cls.outs = []
        for tag, defines in (("a", None), ("b", "RVAL=3000"), ("c", None),
                             ("d", "RVAL=3000")):
            name = "d1_%s.cir" % tag
            with open(os.path.join(cls.work, name), "w") as f:
                f.write(DECK)
            r = subprocess.run([XYCE, name], capture_output=True, text=True, timeout=600,
                               cwd=cls.work, env=_xyce_env(cls.work, defines))
            prn = os.path.join(cls.work, name + ".prn")
            if r.returncode != 0 or not os.path.exists(prn):
                raise AssertionError("Xyce failed on %s (rc %d):\n%s"
                                     % (name, r.returncode, (r.stdout + r.stderr)[-3000:]))
            with open(prn) as f:
                cls.outs.append(_v_out(f.read()))

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("HDLTEST_KEEP"):
            shutil.rmtree(cls.work, ignore_errors=True)

    def _close(self, got, want, what):
        self.assertTrue(abs(got - want) <= RTOL * abs(want),
                        "%s: v(out) = %.12g, expected %.12g" % (what, got, want))

    def test_without_macros(self):
        self._close(self.outs[0], 0.5, "run a (no PYMS_DEFINES)")

    def test_macro_reaches_the_math_library(self):
        self._close(self.outs[1], 0.25, "run b (PYMS_DEFINES=RVAL=3000)")

    def test_shell_built_with_macros_is_not_reused_without(self):
        self._close(self.outs[2], 0.5, "run c (no PYMS_DEFINES, warm cache)")

    def test_macros_again_on_a_warm_cache(self):
        # The device-shell cache of N_DEV_PyMS.C must key on PYMS_DEFINES
        # (its identity hash covers getenv("PYMS_DEFINES")): with the slot
        # keyed on the .va's path + content alone, runs b and d reuse the
        # macro-less shell of run a and give 0.5.
        self._close(self.outs[3], 0.25, "run d (PYMS_DEFINES=RVAL=3000, warm cache)")


if __name__ == "__main__":
    unittest.main()
