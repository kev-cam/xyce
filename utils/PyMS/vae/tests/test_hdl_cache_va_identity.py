#!/usr/bin/env python3
"""Regression: .hdl device-shell cache must key on the .va IDENTITY, not just
the module name + mtime ordering.

THE BUG (found 2026-09-26, stat-sim qal_gate "v3 toolchain regression"): the
shell cache slot is pyms_<module>.so — module NAME only. A same-named module
compiled from a DIFFERENT .va file captured the slot; later runs of the real
.va (whose mtime predated the poisoned .so) reused the stale shell, whose
baked _va path made even the freshly JIT-built vae eval .so compile the WRONG
.va. Result: the model silently lost its subthreshold-tail reassignments
(`ov = ov + K*ln(1+limexp(-t/K))`) — zero log/exp in eval.cpp, hard-clamp
answers, no error anywhere.

THE FIX: N_DEV_PyMS.C records a <so>.src sidecar (va path + content hash) and
reuses the shell only on an exact match; xyce_device_gen.py additionally keys
the vae .so cache on the .va CONTENT (path+params alone went stale on any
equation-only edit).

This test replays the collision end-to-end through Xyce (private PYMS_CACHE /
PYMS_VAE_CACHE): v1 (hard clamp) from dir_a first, then the softplus variant
from dir_b with an OLDER mtime. Asserts the second run measures the softplus
analytic value (K*ln2/(1.2*R)*0.5), not the poisoned 0, and that the built
eval.cpp carries the reassignment chain. Exit 0 = PASS; skips (exit 77) if no
Xyce binary. Canonical twin (fixture files on disk):
stat-sim/qal/va/probe_runs/test_shell_cache_poison.py.
"""
import os, re, shutil, subprocess, sys, tempfile, time

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(XYCE)

K, R = 0.0478, 6229.0
I_SOFTPLUS = (K * 0.6931471805599453) / (1.2 * R) * 0.5
RTOL = 1e-3

VA_V1 = """`include "disciplines.vams"
module softclamp(p, n);
  inout p, n; electrical p, n;
  parameter real K = 0.0478;  parameter real R = 6229;
  real ov, g;
  analog begin
    ov = V(p,n) - 0.5;
    if (ov < 0.0) ov = 0.0;
    g = ov/(1.2*R);
    I(p,n) <+ g*V(p,n);
  end
endmodule
"""

VA_SOFTPLUS = """`include "disciplines.vams"
module softclamp(p, n);
  inout p, n; electrical p, n;
  parameter real K = 0.0478;  parameter real R = 6229;
  real ov, t, g;
  analog begin
    ov = V(p,n) - 0.5;
    t = ov;  if (t < 0.0) t = -t;
    if (ov < 0.0) ov = 0.0;
    ov = ov + K*ln(1.0 + limexp(-t/K));
    g = ov/(1.2*R);
    I(p,n) <+ g*V(p,n);
  end
endmodule
"""

DECK = """* softclamp cache-identity deck (V=0.5 = the clamp knee)
.hdl "%(va)s"
.model m1 softclamp
YSOFTCLAMP X1 p 0 m1
VS p 0 0.5
.tran 1p 10p
.measure tran IDEV FIND I(VS) AT=9p
.end
"""


def run_deck(va_path, work, env, tag):
    deck = os.path.join(work, "run_%s.cir" % tag)
    with open(deck, "w") as f:
        f.write(DECK % {"va": va_path})
    subprocess.run([XYCE, deck], capture_output=True, text=True,
                   timeout=230, cwd=work, env=env)
    mt0 = deck + ".mt0"
    if not os.path.exists(mt0):
        print("FAIL: no .mt0 for %s" % deck)
        sys.exit(1)
    for ln in open(mt0):
        m = re.match(r"\s*IDEV\s*=\s*(\S+)", ln)
        if m:
            return float(m.group(1))
    print("FAIL: IDEV not in %s" % mt0)
    sys.exit(1)


def main():
    if not os.path.exists(XYCE):
        print("SKIP: no Xyce at %s" % XYCE)
        return 77
    work = tempfile.mkdtemp(prefix="hdlcacheid_")
    dir_a = os.path.join(work, "dir_a"); os.makedirs(dir_a)
    dir_b = os.path.join(work, "dir_b"); os.makedirs(dir_b)
    va_a = os.path.join(dir_a, "softclamp.va")
    va_b = os.path.join(dir_b, "softclamp.va")
    open(va_a, "w").write(VA_V1)
    open(va_b, "w").write(VA_SOFTPLUS)

    env = dict(os.environ)
    env["PYMS_CACHE"] = os.path.join(work, "hdl_cache")
    env["PYMS_VAE_CACHE"] = os.path.join(work, "vae_cache")
    env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.pop("VAE_SO_PATH", None); env.pop("VAE_SO_DIR", None)

    i_a = run_deck(va_a, work, env, "a")          # poison the slot
    if abs(i_a) > 1e-12:
        print("FAIL (control): v1 measured %.4g, expected 0" % i_a)
        return 1
    old = time.time() - 3600                       # the v3 mtime situation
    os.utime(va_b, (old, old))
    i_b = run_deck(va_b, work, env, "b")           # same module, other file

    if not (abs(-i_b - I_SOFTPLUS) <= RTOL * I_SOFTPLUS):
        print("FAIL: softplus run measured I=%.6g (expected |I|=%.6g);"
              " 0 = cache poisoning is back" % (i_b, I_SOFTPLUS))
        return 1

    chain = False
    for root, _d, files in os.walk(env["PYMS_VAE_CACHE"]):
        for fn in files:
            if fn == "eval.cpp" and "softclamp" in root:
                txt = open(os.path.join(root, fn)).read()
                if "log(" in txt and "exp(" in txt and re.search(r"\bov__\d+\s*=", txt):
                    chain = True
    if not chain:
        print("FAIL: no eval.cpp carries the softplus reassignment chain")
        return 1

    print("PASS: hdl cache .va identity (control %.3g A; softplus %.6g A)"
          % (i_a, i_b))
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
