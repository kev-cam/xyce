"""Shared deck runner of the 1.1.3 tests: a module X1 between in (1 V dc, or a
1 V 1 kHz sine for tran) and out, R2 = 1k from out to ground; the analytic
V(out) of a device of resistance R is 1000 / (R + 1000).

run_decks(cases, prefix) runs every case in a fresh Xyce process with private
PyMS caches and returns the list of mismatches (empty = all as expected).
cases: name -> (module text, .model card, want[, 'tran' | 'ac']) where want is a
number (V(out) at the operating point), a list (V(out) of the transient at
TIMES; for 'ac' [|V(out)| at 1 kHz] with V1 = DC 0 AC 1) or "ERR" (Xyce must
stop with a PyMS message: generation refused it).
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile

_PYMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)
HDR = '`include "disciplines.vams"\n`include "constants.vams"\n'
TIMES = [0.1e-3, 0.25e-3, 0.6e-3, 1.25e-3, 1.9e-3]


def mod(name, body, ports="p, n", decl="  inout p, n;\n  electrical p, n;\n"):
    return HDR + "module %s(%s);\n%s%s\nendmodule\n" % (name, ports, decl, body)


def rv(r):
    return 1000.0 / (r + 1000.0)


def tran(gain):
    return [gain * math.sin(2 * math.pi * 1e3 * t) for t in TIMES]


def _prn(path, sig="V(OUT)"):
    with open(path) as fh:
        rows = [l.split() for l in fh.read().splitlines() if l.strip()]
    hdr = [c.upper() for c in rows[0]]
    k = hdr.index(sig)
    tk = hdr.index("TIME") if "TIME" in hdr else (hdr.index("FREQ") if "FREQ" in hdr else None)
    out = []
    for c in rows[1:]:
        if c[0].lower().startswith("end"):
            break
        out.append((float(c[tk]) if tk is not None else 0.0, float(c[k])))
    return out


def _interp(rows, t):
    for (t0, v0), (t1, v1) in zip(rows, rows[1:]):
        if t0 <= t <= t1:
            return v0 if t1 == t0 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return None


def run_decks(cases, prefix="pyms_t_", verbose=True, logs=None):
    work = tempfile.mkdtemp(prefix=prefix)
    env = dict(os.environ)
    env.update(PYMS_DIR=_PYMS_DIR, PYMS_CACHE=os.path.join(work, "hc"),
               PYMS_VAE_CACHE=os.path.join(work, "vc"), PYTHONDONTWRITEBYTECODE="1")
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    bad = []
    try:
        for name, spec in cases.items():
            va, card, want = spec[0], spec[1], spec[2]
            an = spec[3] if len(spec) > 3 else ("tran" if isinstance(want, list) else "op")
            d = os.path.join(work, name)
            os.makedirs(d)
            with open(os.path.join(d, "c.va"), "w") as fh:
                fh.write(va)
            if an == "tran":
                src, ana = "SIN(0 1 1k)", ".options timeint reltol=1e-6 abstol=1e-12\n.tran 1u 2m\n.print tran v(out)"
            elif an == "tranj":
                # a transient whose solution jumps (an event body that changes
                # the circuit): Xyce's step-error control rejects a jump from
                # any device, so only the Newton test steps it
                src, ana = "SIN(0 1 1k)", ".options timeint reltol=1e-6 abstol=1e-12 erroption=1\n.tran 1u 2m 0 1u\n.print tran v(out)"
            elif an == "ac":
                src, ana = "DC 0 AC 1", ".ac dec 1 1k 1k\n.print ac vm(out)"
            else:
                src, ana = "DC 1", ".op\n.print dc v(out)"
            with open(os.path.join(d, "c.cir"), "w") as fh:
                fh.write("* %s\n.hdl \"c.va\"\nV1 in 0 %s\n.model m1 %s %s\nY%s X1 in out m1\n"
                         "R2 out 0 1k\n%s\n.end\n" % (name, src, name, card, name, ana))
            r = subprocess.run([XYCE, "c.cir"], cwd=d, env=env, capture_output=True,
                               text=True, timeout=1800)
            log = r.stdout + r.stderr
            if logs is not None:
                logs[name] = log
            got = None
            try:
                if an == "ac":
                    got = [_prn(os.path.join(d, "c.cir.FD.prn"), "VM(OUT)")[0][1]]
                else:
                    rows = _prn(os.path.join(d, "c.cir.prn"))
                    got = [_interp(rows, t) for t in TIMES] if an in ("tran", "tranj") else rows[0][1]
            except (OSError, IndexError, ValueError):
                pass
            if want == "ERR":
                empty = got is None or (isinstance(got, list) and all(g is None for g in got))
                ok = r.returncode != 0 and empty and ("PyMS" in log or "xyce_device_gen" in log)
            elif isinstance(want, list):
                ok = (r.returncode == 0 and got is not None and None not in got
                      and all(abs(g - w) <= 1e-4 for g, w in zip(got, want)))
            else:
                ok = (r.returncode == 0 and got is not None
                      and abs(got - want) <= 1e-6 * max(1.0, abs(want)))
            if verbose:
                print("%-28s want %s got %s rc %d" % (name, want, got, r.returncode), flush=True)
            if not ok:
                tail = "" if (got is not None and want != "ERR") else log[-700:]
                bad.append("%s: want %s got %s (rc %d) %s" % (name, want, got, r.returncode, tail))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return bad
