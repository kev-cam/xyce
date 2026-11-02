"""Undeclared nodes, branches and assignment targets are file:line errors.

THE DEFECT (PYMS2 skeptic, 2026-10-26; BUGS.md CHANGES 1.1.2): ``I(p, q) <+ ...``
with q undeclared, ``I(bq) <+ ...`` with no branch bq, and ``zz = 1;`` with
no variable zz all ran with exit status 0: the contribution went to no node
(V(out) = 0 on the 1 V / 1k divider), zz became a local of the generated
code.

THE FIX: ginac_emitter.check_declarations, called by xyce_device_gen before
any C++ is written, refuses a contribution target or a V()/I() argument
that is not a port, node, ground or named branch, and an assignment to a
name that is neither a declared variable nor a parameter, with the
statement's file and line (an assignment to a parameter is the Xyce/ADMS
dialect VBIC 1.3 uses for its type parameter: BUGS.md C11).  Legal forms it must keep: variables
declared in a (named) block -- the parser now makes them variables of the
module instead of dropping the declaration --, ground nets, genvar loops,
the parser's ternary temporaries, named branches, $strobe text.

Run:  cd utils/PyMS && python3 vae/tests/test_undeclared_names.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PYMS_DIR not in sys.path:
    sys.path.insert(0, _PYMS_DIR)

from vae.parser import parse_verilog_a  # noqa: E402
from vae.ginac_emitter import check_declarations, EmitError  # noqa: E402

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'


def mod(body, name="u1"):
    return (HDR + "module %s(p, n);\n  inout p, n;\n  electrical p, n;\n%s\nendmodule\n" % (name, body))


BAD = {
    "node_target": ("  analog I(p, q) <+ V(p, n) / 1000;", r"u\.va:6: 'q' in the contribution target \(p, q\)"),
    "branch_target": ("  analog I(bq) <+ V(p, n) / 1000;", r"u\.va:6: 'bq' in the contribution target \(bq\)"),
    "node_probe": ("  analog I(p, n) <+ V(p, q) / 1000;", r"u\.va:6: 'q' in V\(p, q\)"),
    "branch_probe": ("  analog I(p, n) <+ V(p, n) / 1000 + 0 * I(bq);", r"u\.va:6: 'bq' in I\(bq\)"),
    "cond_node": ("  analog begin if (V(q) > 0) I(p, n) <+ V(p, n) / 1000; end", r"u\.va:6: 'q' in V\(q\)"),
    "assign": ("  analog begin\n    zz = 1;\n    I(p, n) <+ V(p, n) / 1000;\n  end",
               r"u\.va:7: assignment to 'zz': not declared in module u1"),
}
GOOD = {
    "named_block_local": "  analog begin : top\n    begin : blk\n      real t;\n      t = 1000;\n"
                         "      I(p, n) <+ V(p, n) / t;\n    end\n  end",
    "block_local_init": "  analog begin\n    begin : blk\n      integer k = 3;\n      I(p, n) <+ V(p, n) / (500 * (k / 2));\n"
                        "    end\n  end",
    "ground": "  ground gnd;\n  analog begin I(p, n) <+ V(p, n) / 1000; I(n, gnd) <+ 0; end",
    "genvar": "  genvar i;\n  analog begin\n    for (i = 0; i < 1; i = i + 1) I(p, n) <+ V(p, n) / 1000;\n  end",
    "named_branch": "  branch (p, n) b;\n  analog I(b) <+ V(b) / 1000;",
    "ternary": "  real r;\n  analog begin r = (V(p, n) > 0) ? 1000 : 2000; I(p, n) <+ V(p, n) / r; end",
    "strobe": "  analog begin I(p, n) <+ V(p, n) / 1000; $strobe(\"V(q)=%g\", V(p, n)); end",
    "internal_node": "  electrical x;\n  analog begin I(p, x) <+ V(p, x) / 500; I(x, n) <+ V(x, n) / 500; end",
    # the Xyce/ADMS dialect of a hidden type parameter (VBIC 1.3's VBICtype)
    "assign_param": "  parameter integer t = 1;\n  analog begin t = -1; I(p, n) <+ V(p, n) / 1000; end",
}


class TestCheck(unittest.TestCase):

    def test_refused(self):
        for name, (body, msg) in BAD.items():
            with self.subTest(name):
                with self.assertRaisesRegex(EmitError, msg):
                    check_declarations(parse_verilog_a(mod(body), filename="u.va"))

    def test_legal_forms_pass(self):
        for name, body in GOOD.items():
            with self.subTest(name):
                m = parse_verilog_a(mod(body), filename="u.va")
                check_declarations(m)
        m = parse_verilog_a(mod(GOOD["block_local_init"]), filename="u.va")
        self.assertEqual([(v.name, v.type) for v in m.variables], [("k", "integer")])


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestXyce(unittest.TestCase):

    def _run(self, va):
        work = tempfile.mkdtemp(prefix="pyms_undecl_")
        try:
            env = dict(os.environ)
            env.update(PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(work, "hc"),
                       PYMS_VAE_CACHE=os.path.join(work, "vc"), PYTHONDONTWRITEBYTECODE="1")
            env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            open(os.path.join(work, "u.va"), "w").write(va)
            open(os.path.join(work, "d.cir"), "w").write(
                '* u\n.hdl "u.va"\nV1 in 0 DC 1\n.model m1 u1\nYu1 X1 in out m1\nR2 out 0 1k\n'
                '.op\n.print dc v(out)\n.end\n')
            r = subprocess.run([XYCE, "d.cir"], cwd=work, env=env, capture_output=True, text=True, timeout=900)
            got = None
            try:
                got = float(open(os.path.join(work, "d.cir.prn")).read().split("\n")[1].split()[1])
            except (OSError, IndexError, ValueError):
                pass
            return r.returncode, r.stdout + r.stderr, got
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_undeclared_node_is_a_netlist_error(self):
        rc, log, _got = self._run(mod(BAD["node_target"][0]))
        self.assertNotEqual(rc, 0, log[-1500:])
        self.assertRegex(log, r"xyce_device_gen: error: \S*u\.va:6: 'q' in the contribution target")

    def test_undeclared_target_is_a_netlist_error(self):
        rc, log, _got = self._run(mod(BAD["assign"][0]))
        self.assertNotEqual(rc, 0, log[-1500:])
        self.assertRegex(log, r"xyce_device_gen: error: \S*u\.va:7: assignment to 'zz'")

    def test_named_block_local_runs(self):
        rc, log, got = self._run(mod(GOOD["named_block_local"]))
        self.assertEqual(rc, 0, log[-1500:])
        self.assertAlmostEqual(got, 0.5, delta=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
