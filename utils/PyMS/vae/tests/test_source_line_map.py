"""Every PyMS error names the true source file and line (2026-10-25).

THE BUG: the parser numbered the lines of the PREPROCESSED text: the standard
headers PyMS inlines (`include "disciplines.vams" and "constants.vams" bring
~245 lines), the dropped directive lines and the included files all shifted
the count, so a statement on line 8 of a1.va was reported as a1.va:253, and a
statement in an included file was reported under the entry file's name with
a line of neither file. Parser, emitter and device-generator errors all
inherited it (clusters A 3, B 2, D 1 and E 3 of PYMS round 1).

THE FIX: preprocess.py returns a PreprocessedText (a str) with a line map, the
(file, line) of the source of every line of the text; a macro invocation over
several lines keeps its expansion on its first line and blank lines after it,
so the lines after it keep their place. Parser._loc_at reads the map, every
ParseError starts with ``file:line:``, Param.loc names the declaration's true
file and line (xyce_device_gen's default and case errors use it), and an
emitter error raised without a location gets its statement's.

No Xyce needed: the preprocessor, the parser, the emitter and the device
generator's command line.

Run:
    cd utils/PyMS && python3 vae/tests/test_source_line_map.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

_VAE_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _VAE_PKG_DIR not in sys.path:
    sys.path.insert(0, _VAE_PKG_DIR)

from vae.parser import parse_verilog_a, parse_file, ParseError, NodeKind  # noqa: E402
from vae.preprocess import preprocess_source, preprocess_file, PreprocessError  # noqa: E402
from vae.ginac_emitter import emit_ginac_program, EmitError  # noqa: E402

DEVICE_GEN = os.path.join(_VAE_PKG_DIR, 'vae', 'xyce_device_gen.py')

HEAD = '`include "disciplines.vams"\n`include "constants.vams"\n'
MODULE = (HEAD +                                   # lines 1-2
          'module a1(p, n);\n'                     # 3
          '  inout p, n;\n'                        # 4
          '  electrical p, n;\n'                   # 5
          '  parameter real r = 1k;\n'             # 6
          '  real v;\n'                            # 7
          '  analog begin\n'                       # 8
          '    v = V(p, n);\n'                     # 9
          '    I(p, n) <+ v / r;\n'                # 10
          '  end\n'                                # 11
          'endmodule\n')                           # 12


def line_of(text, needle, nth=1):
    """The 1-based line of the nth occurrence of needle in text."""
    i = -1
    for _ in range(nth):
        i = text.index(needle, i + 1)
    return text[:i].count('\n') + 1


def nodes(node, kind):
    if node is None:
        return []
    found = [node] if node.kind == kind else []
    for child in list(node.children or []) + ([node.else_body] if node.else_body else []):
        found += nodes(child, kind)
    return found


class Scratch(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp(prefix='pyms_linemap_')

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def write(self, name, text):
        path = os.path.join(self.work, name)
        with open(path, 'w') as f:
            f.write(text)
        return path

    def gen(self, va_path):
        """xyce_device_gen.py on va_path: (exit status, output)."""
        out = os.path.join(self.work, 'out')
        os.makedirs(out, exist_ok=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        p = subprocess.run([sys.executable, DEVICE_GEN, va_path, '--output', out],
                           capture_output=True, text=True, timeout=300, env=env)
        return p.returncode, p.stdout + p.stderr


class TestLineMap(Scratch):
    """The preprocessor's map and the parser's locations."""

    def test_the_map_names_each_line_s_source(self):
        text = preprocess_source(MODULE, filename='a1.va')
        lmap = text.line_map
        self.assertEqual(len(lmap), text.count('\n') + 1)
        lines = text.split('\n')
        for needle in ('module a1', 'parameter real r', 'v = V(p, n);', 'I(p, n) <+ v / r;', 'endmodule'):
            i = next(k for k, ln in enumerate(lines) if needle in ln)
            self.assertEqual(lmap[i], ('a1.va', line_of(MODULE, needle)), needle)
        # the lines of the headers carry the headers' names
        i = next(k for k, ln in enumerate(lines) if 'discipline electrical' in ln)
        self.assertTrue(lmap[i][0].endswith('disciplines.vams'), lmap[i])

    def test_statement_and_parameter_locations_are_source_lines(self):
        mod = parse_verilog_a(preprocess_source(MODULE, filename='a1.va'), filename='a1.va')
        (contrib,) = nodes(mod.analog_block, NodeKind.CONTRIB)
        (assign,) = nodes(mod.analog_block, NodeKind.ASSIGN)
        self.assertEqual((contrib.loc.file, contrib.loc.line), ('a1.va', 10))
        self.assertEqual((assign.loc.file, assign.loc.line), ('a1.va', 9))
        (r,) = [p for p in mod.params if p.name == 'r']
        self.assertEqual((r.loc.file, r.loc.line), ('a1.va', 6))

    def test_a_parse_error_names_the_source_line(self):
        va = MODULE.replace('    I(p, n) <+ v / r;\n', '    I(p, n) <+ v / r;\n    (*info="x"*)\n')
        with self.assertRaises(ParseError) as cm:
            parse_verilog_a(preprocess_source(va, filename='a1.va'), filename='a1.va')
        self.assertRegex(str(cm.exception), r'^a1\.va:%d: an attribute instance \(\* \.\.\. \*\) must be '
                         r"followed by a statement, found 'end'" % line_of(va, '  end'))

    def test_an_expected_token_error_names_the_source_line(self):
        va = MODULE.replace('module a1(p, n);\n', 'module a1(p, n)\n')
        with self.assertRaises(ParseError) as cm:
            parse_verilog_a(preprocess_source(va, filename='a1.va'), filename='a1.va')
        # the parser sees the missing ';' at the next token, 'inout' on line 4
        self.assertRegex(str(cm.exception), r"^a1\.va:4: expected ';', got 'inout'")

    def test_after_a_macro_invocation_over_several_lines(self):
        va = (HEAD +
              '`define P(nm, dflt) parameter real nm = dflt;\n'   # 3
              'module a1(p, n);\n'                                  # 4
              '  inout p, n;\n'                                     # 5
              '  electrical p, n;\n'                                # 6
              '  `P(\n'                                             # 7
              '     r,\n'                                           # 8
              '     1k)\n'                                          # 9
              '  analog begin\n'                                    # 10
              '    I(p, n) <+ V(p, n) / r;\n'                       # 11
              '    (*info="x"*)\n'                                  # 12
              '  end\n'                                             # 13
              'endmodule\n')
        text = preprocess_source(va, filename='a1.va')
        lines = text.split('\n')
        i = next(k for k, ln in enumerate(lines) if 'parameter real r' in ln)
        self.assertEqual(text.line_map[i], ('a1.va', 7))
        with self.assertRaises(ParseError) as cm:
            parse_verilog_a(text, filename='a1.va')
        self.assertRegex(str(cm.exception), r'^a1\.va:13: ')
        good = va.replace('    (*info="x"*)\n', '')
        mod = parse_verilog_a(preprocess_source(good, filename='a1.va'), filename='a1.va')
        (contrib,) = nodes(mod.analog_block, NodeKind.CONTRIB)
        self.assertEqual(contrib.loc.line, 11)
        self.assertEqual(mod.params[0].loc.line, 7)

    def test_after_continuation_lines_and_conditionals(self):
        va = (HEAD +
              '`define TWO_R \\\n'                                  # 3
              '   (2.0 * \\\n'                                      # 4
              '    1k)\n'                                            # 5
              '`ifdef NOT_DEFINED\n'                                 # 6
              '`define X 1\n'                                        # 7
              '`else\n'                                              # 8
              '`define X 2\n'                                        # 9
              '`endif\n'                                             # 10
              'module a1(p, n);\n'                                   # 11
              '  inout p, n;\n'                                      # 12
              '  electrical p, n;\n'                                 # 13
              '  parameter real r = `TWO_R;\n'                       # 14
              '  analog I(p, n) <+ V(p, n) / r * `X;\n'              # 15
              'endmodule\n')
        mod = parse_verilog_a(preprocess_source(va, filename='a1.va'), filename='a1.va')
        (contrib,) = nodes(mod.analog_block, NodeKind.CONTRIB)
        self.assertEqual((contrib.loc.file, contrib.loc.line), ('a1.va', 15))
        self.assertEqual(mod.params[0].loc.line, 14)

    def test_an_error_inside_an_included_file(self):
        body = ('module a1(p, n);\n'          # 1
                '  inout p, n;\n'             # 2
                '  electrical p, n;\n'        # 3
                '  parameter real r = 1k;\n'  # 4
                '  analog begin\n'            # 5
                '    I(p, n) <+ V(p, n) / r;\n'  # 6
                '    (*info="x"*)\n'          # 7
                '  end\n'                     # 8
                'endmodule\n')
        inc = self.write('a1_body.vams', body)
        top = self.write('a1.va', HEAD + '`include "a1_body.vams"\n')
        with self.assertRaises(ParseError) as cm:
            parse_file(top)
        self.assertRegex(str(cm.exception), r'^%s:8: an attribute instance' % re.escape(inc))
        # and the locations of a good module in an included file
        self.write('a1_body.vams', body.replace('    (*info="x"*)\n', ''))
        mod = parse_file(top)
        (contrib,) = nodes(mod.analog_block, NodeKind.CONTRIB)
        self.assertEqual((contrib.loc.file, contrib.loc.line), (inc, 6))
        self.assertEqual((mod.params[0].loc.file, mod.params[0].loc.line), (inc, 4))

    def test_a_preprocessor_directive_error_names_its_line(self):
        va = HEAD + 'module a1(p, n);\n`endif\nendmodule\n'
        with self.assertRaises(PreprocessError) as cm:
            preprocess_source(va, filename='a1.va')
        self.assertRegex(str(cm.exception), r'^a1\.va:4: `endif without `ifdef')


class TestEmitterAndDeviceGenerator(Scratch):
    """The emitter's and xyce_device_gen's errors name the source line."""

    def test_an_emitter_error_names_the_statement_s_line(self):
        va = MODULE.replace('    I(p, n) <+ v / r;\n',
                            '    I(p, n) <+ v / r + exp(ddt(1e-12 * V(p, n))) - 1.0;\n')
        mod = parse_verilog_a(preprocess_source(va, filename='a1.va'), filename='a1.va')
        with self.assertRaises(EmitError) as cm:
            emit_ginac_program(mod)
        self.assertRegex(str(cm.exception), r'^a1\.va:10: ')

    def test_an_emitter_error_without_its_own_location_gets_the_statement_s(self):
        # GiNaCEmitter._cond... raises 'module a1: the variable q carries ddt()'
        va = (HEAD + 'module a1(p, n);\n  inout p, n;\n  electrical p, n;\n'
              '  parameter real r = 1k;\n  real q;\n'
              '  analog begin\n'                                  # 8
              '    q = ddt(1e-12 * V(p, n));\n'                   # 9
              '    if (q > 0.0)\n'                                # 10
              '      I(p, n) <+ V(p, n) / r;\n'                   # 11
              '    else\n'
              '      I(p, n) <+ V(p, n) / (2 * r);\n'
              '  end\nendmodule\n')
        mod = parse_verilog_a(preprocess_source(va, filename='a1.va'), filename='a1.va')
        with self.assertRaises(EmitError) as cm:
            emit_ginac_program(mod)
        self.assertRegex(str(cm.exception), r'^a1\.va:10: ')

    def test_a_default_error_in_an_included_file(self):
        inc = self.write('a1_params.vams', '// the parameters\n'
                                           '  parameter real r = rx * 1000.0;\n')
        top = self.write('a1.va', HEAD + 'module a1(p, n);\n  inout p, n;\n  electrical p, n;\n'
                                         '`include "a1_params.vams"\n'
                                         '  analog I(p, n) <+ V(p, n) / r;\nendmodule\n')
        rc, out = self.gen(top)
        self.assertNotEqual(rc, 0, out)
        self.assertIn('%s:2: parameter r of module a1' % inc, out)

    def test_a_case_collision_in_an_included_file(self):
        inc = self.write('a1_params.vams', '  parameter real temp = 1.0;\n'
                                           '\n'
                                           '  parameter real TEMP = 2.0;\n')
        top = self.write('a1.va', HEAD + 'module a1(p, n);\n  inout p, n;\n  electrical p, n;\n'
                                         '`include "a1_params.vams"\n'
                                         '  analog I(p, n) <+ V(p, n) / 1k;\nendmodule\n')
        rc, out = self.gen(top)
        self.assertNotEqual(rc, 0, out)
        self.assertIn('%s:1: module a1: parameters' % inc, out)

    def test_a_parse_error_on_the_command_line(self):
        top = self.write('a1.va', MODULE.replace('module a1(p, n);\n', 'module a1(p, n)\n'))
        rc, out = self.gen(top)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("xyce_device_gen: error: %s:4: expected ';'" % top, out)
        self.assertNotIn('Traceback', out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
