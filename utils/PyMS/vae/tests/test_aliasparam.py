"""aliasparam names as device parameters (1.1.8, BUGS.md B26).

THE DEFECT (vamos round 8, sv2ghdl FIX2E8, on 1.1.7): with ``aliasparam gal = g``
the instance line ``ya2 X1 in out c gal=1m`` was "Unrecognized parameter GAL",
and on the card ``ral=2k`` gave only the warning "No model parameter RAL found
..., parameter ignored" and exit 0 (the default used, silently in a bare Xyce
run).  openvaf-r and VACASK accept both.  The parser skipped the declaration.

Now the alias is registered as another netlist name of the target (on the
card when the target is a model parameter, on the line when it is an instance
parameter, both for xyceAlsoModel), and the constructors hand a given alias to
the target, so the value, its given flag ($param_given) and a .global_param /
.step expression all reach the target.  Both names given is an error, as is
an alias of a name that is no parameter.

V1 = 1 V, the module between in and out, R2 = 1k to ground:
V(out) = 1k / (1k + Req).

Run:  cd utils/PyMS && python3 vae/tests/test_aliasparam.py
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

from vae.tests._decks import HDR, XYCE, XYCE_LIBDIR  # noqa: E402

A2 = HDR + """module a2(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  parameter real g = 0 (* type="instance" *);
  parameter real ga = 0 (* type="instance" xyceAlsoModel="yes" *);
  aliasparam ral = r;
  aliasparam gal = g;
  (* desc="alias of ga" *) aliasparam gaa = ga;
  analog I(p, n) <+ V(p, n) * (1 / r + g + ga);
endmodule
"""
# $param_given through the alias
B3 = HDR + """module a2(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  aliasparam ral = r;
  analog I(p, n) <+ V(p, n) * ($param_given(r) ? 1 / r : 2m);
endmodule
"""
BAD = HDR + """module a2(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1k;
  aliasparam ral = rr;
  analog I(p, n) <+ V(p, n) / r;
endmodule
"""


def deck(card, line, extra="", analysis=".dc V1 1 1 1"):
    return ("* t\n.hdl \"a2.va\"\nV1 in 0 1\n.model c1 a2 %s\nya2 y1 in out c1 %s\nR2 out 0 1k\n%s%s\n"
            ".print dc v(out)\n.end\n" % (card, line, extra, analysis))


T3, T5, T6, T2 = 1 / 3, 0.2, 2 / 3, 0.5
# name -> (module, deck, V(out) of every row | "ERR")
CASES = {
    "card_model_alias": (A2, deck("ral=2k", ""), [T3]),
    "line_instance_alias": (A2, deck("", "gal=1m"), [T6]),
    "line_also_model_alias": (A2, deck("", "gaa=1m"), [T6]),
    "card_also_model_alias": (A2, deck("gaa=1m", ""), [T6]),
    "card_alias_global": (A2, deck("ral={rr}", "", ".global_param rr=2k\n"), [T3]),
    "card_alias_step": (A2, deck("ral={rr}", "", ".param rr=1k\n.step rr list 2k 4k\n"), [T3, T5]),
    "line_alias_dc_param": (A2, deck("", "gal={gg}", ".param gg=1m\n", ".dc gg list 1m 3m"), [T6, 0.8]),
    "param_given_via_alias": (B3, deck("ral=4k", ""), [T5]),
    "param_given_none": (B3, deck("", ""), [T6]),
    "target_still_works": (A2, deck("r=2k", "g=1m"), [0.6]),
    "defaults": (A2, deck("", ""), [T2]),
    "both_names_given": (A2, deck("r=2k ral=4k", ""), "ERR"),
    "both_names_given_line": (A2, deck("", "g=1m gal=3m"), "ERR"),
    "alias_of_no_parameter": (BAD, deck("", ""), "ERR"),
}


def rows(path):
    out = []
    with open(path) as fh:
        for ln in fh:
            c = ln.split()
            if not c or c[0] == "Index" or c[0].lower().startswith("end"):
                continue
            out.append(float(c[-1]))
    return out


class TestParser(unittest.TestCase):

    def test_parser_records_aliases(self):
        from vae.parser import parse_verilog_a
        m = parse_verilog_a(A2.replace(HDR, ''))
        self.assertEqual(m.aliases, {'ral': 'r', 'gal': 'g', 'gaa': 'ga'})

    def test_alias_of_no_parameter_is_an_error(self):
        from vae.parser import parse_verilog_a, ParseError
        with self.assertRaisesRegex(ParseError, "no parameter 'rr'"):
            parse_verilog_a(BAD.replace(HDR, ''))


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce binary")
class TestAliasparam(unittest.TestCase):

    def test_decks(self):
        work = tempfile.mkdtemp(prefix='pyms_alias_')
        env = dict(os.environ, PYMS_DIR=_PYMS_DIR, PYTHONDONTWRITEBYTECODE='1')
        env['LD_LIBRARY_PATH'] = XYCE_LIBDIR + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        env.pop('PYMS_DEFINES', None)
        bad = []
        try:
            for name, (va, text, want) in CASES.items():
                d = os.path.join(work, name)
                os.makedirs(d)
                with open(os.path.join(d, 'a2.va'), 'w') as fh:
                    fh.write(va)
                with open(os.path.join(d, 'c.cir'), 'w') as fh:
                    fh.write(text)
                env.update(PYMS_CACHE=os.path.join(d, 'hc'), PYMS_VAE_CACHE=os.path.join(d, 'vc'))
                r = subprocess.run([XYCE, 'c.cir'], cwd=d, env=env, capture_output=True, text=True, timeout=1800)
                log = r.stdout + r.stderr
                try:
                    got = rows(os.path.join(d, 'c.cir.prn'))
                except OSError:
                    got = None
                if want == "ERR":
                    ok = r.returncode != 0 and not got
                else:
                    ok = (r.returncode == 0 and got is not None and len(got) == len(want)
                          and all(abs(a - b) <= 1e-6 for a, b in zip(got, want))
                          and 'parameter ignored' not in log)
                print('%-24s want %s got %s rc %d' % (name, want if want == "ERR" else ['%.6g' % w for w in want],
                      got and ['%.6g' % g for g in got], r.returncode), flush=True)
                if not ok:
                    bad.append((name, got, log[-500:]))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
