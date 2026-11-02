#!/usr/bin/env python3
"""Regression: the .hdl device-shell cache must survive CONCURRENT Xyce runs
that share one PYMS_CACHE / PYMS_VAE_CACHE (two users, two parallel sweeps,
a test harness).

THE BUG (P8, measured 2026-10-24): the shell cache slot was pyms_<module>.so,
named by MODULE NAME alone; the generator wrote N_DEV_PYMS_<MODULE>.C into the
cache directory itself; every file was written in place. Two concurrent runs
of DIFFERENT modules both named a1 (r=1k and r=3k) through one shared cache:
5 of 8 pairs wrong. The loser compiled the other module's shell source or
dlopen'ed the .so the other was still writing: a silently wrong value (exit 0,
the OTHER module's answer -- the shell's baked _va path builds the other
module's math library) or ".HDL: failed to load compiled plugin ...: file too
short".

THE FIX: N_DEV_PyMS.C runs the generator into a per-process directory
(gen.<identity>.<pid>), names the slot pyms_<module>_<identity>.so where the
identity hashes the .va path and content, and compiles the .so (and writes
its .src sidecar) under a private name that is rename()d into place.
build_vae_so.py builds the math .so in a per-process directory and installs
it by os.replace(); the shell writes its .params file the same way.

Tests, all end-to-end through Xyce with caches private to the test but SHARED
by the two concurrent runs (that is the point):
  1. the deterministic interleaving: a g++ wrapper first on PATH pauses each
     run's shell compile; B's generator runs while A's compile is paused, then
     A compiles. Before the fix A compiles B's source and reads B's value.
     The wrapper also logs every g++ call, from which the atomic-write
     property is checked: no shell .so and no math .so is ever linked under
     its final cache name, and no private build file is left behind.
  2. two runs of the SAME .va (a parallel sweep): both right, one shell slot,
     the same atomic-write property.
Xyce: $XYCE or /usr/local/src/xyce-build/src/Xyce; PyMS: the copy this file
sits in (PYMS_DIR is set from the test's own location). Skipped when there is
no Xyce binary. PYMS_TEST_KEEP=1 keeps the work directory.
"""
import os, re, shutil, subprocess, sys, tempfile, time, unittest

XYCE = os.environ.get("XYCE", "/usr/local/src/xyce-build/src/Xyce")
XYCE_LIBDIR = os.path.dirname(os.path.abspath(XYCE))
HERE = os.path.dirname(os.path.abspath(__file__))
PYMS_DIR = os.path.dirname(os.path.dirname(HERE))      # .../PyMS: this copy

RTOL = 1e-6
RUN_TIMEOUT = 300    # s per Xyce run: a cold shell compile is ~12 s, a cold math library ~15 s
WAIT_TIMEOUT = 120   # s for a sync file to appear

VA = """`include "disciplines.vams"
`include "constants.vams"
module a1(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = %(r)s;
  analog I(p, n) <+ V(p, n) / r;
endmodule
"""

# the native form of a module instance, as vamos writes it (vamos_a1.cir)
DECK = """* concurrent-cache deck %(tag)s
.hdl "%(va)s"
.model a1__t a1
ya1 X1 in out a1__t
V1 in 0 DC 1.0
R2 out 0 1000.0
.dc V1 1 1 1
.print dc format=noindex V(out)
.end
"""

# Fault injection: N_DEV_PyMS.C compiles the shell with popen("g++ ..."), so a
# g++ first on PATH can hold that one compile until the test says go, and
# log every call (the math-library builds of build_vae_so.py included).
GXX_WRAPPER = """#!/bin/bash
case " $* " in
  *N_DEV_PYMS_*)
    touch "$PYMS_TEST_SYNC/$PYMS_TEST_TAG.compiling"
    n=0
    while [ ! -e "$PYMS_TEST_SYNC/$PYMS_TEST_TAG.go" ] && [ $n -lt %(ticks)d ]; do sleep 0.05; n=$((n+1)); done
    ;;
esac
printf '%%s\\n' "$*" >> "$PYMS_TEST_SYNC/$PYMS_TEST_TAG.gxx.log"
exec %(gxx)s "$@"
"""

BAD_MESSAGES = ("file too short", "invalid ELF header", "failed to load compiled plugin",
                "g++ compilation failed", "PyMS compilation failed", "Netlist error")


def _wait_for(path, timeout=WAIT_TIMEOUT):
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            return False
        time.sleep(0.05)
    return True


def _read_out(prn):
    """V(out) from a .print dc format=noindex file: a header line, then the value."""
    with open(prn) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    for ln in lines[1:]:
        try:
            return float(ln.split()[0])
        except ValueError:
            continue
    return None


class Run(object):
    """One Xyce process on one deck, tagged for the g++ wrapper."""

    def __init__(self, work, tag, env, va_text=None, va_path=None):
        self.tag = tag
        self.dir = os.path.join(work, tag)
        os.makedirs(self.dir)
        if va_path is None:
            va_path = os.path.join(self.dir, "a1.va")
            with open(va_path, "w") as f:
                f.write(va_text)
        self.va = va_path
        self.deck = os.path.join(self.dir, "a1.cir")
        with open(self.deck, "w") as f:
            f.write(DECK % {"tag": tag, "va": self.va})
        self.env = dict(env)
        self.env["PYMS_TEST_TAG"] = tag
        self.log_path = os.path.join(self.dir, "xyce.log")
        self.proc = None
        self.rc = None
        self.out = None
        self.text = ""

    def start(self):
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen([XYCE, "a1.cir"], cwd=self.dir, env=self.env,
                                     stdout=self._log, stderr=subprocess.STDOUT)
        return self

    def wait(self):
        try:
            self.rc = self.proc.wait(timeout=RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)
            self.rc = -9
        self._log.close()
        with open(self.log_path) as f:
            self.text = f.read()
        prn = self.deck + ".prn"
        self.out = _read_out(prn) if os.path.exists(prn) else None
        return self


@unittest.skipUnless(os.path.exists(XYCE), "no Xyce at %s (set XYCE)" % XYCE)
class HdlCacheConcurrent(unittest.TestCase):

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="hdlcacheconc_")
        self.sync = os.path.join(self.work, "sync")
        os.makedirs(self.sync)
        bin_dir = os.path.join(self.work, "bin")
        os.makedirs(bin_dir)
        real_gxx = shutil.which("g++")
        self.assertTrue(real_gxx, "no g++ on PATH")
        wrapper = os.path.join(bin_dir, "g++")
        with open(wrapper, "w") as f:
            f.write(GXX_WRAPPER % {"gxx": real_gxx, "ticks": WAIT_TIMEOUT * 20})
        os.chmod(wrapper, 0o755)
        env = dict(os.environ)
        env["PYMS_DIR"] = PYMS_DIR
        env["PYMS_CACHE"] = os.path.join(self.work, "hdl_cache")      # shared by both runs
        env["PYMS_VAE_CACHE"] = os.path.join(self.work, "vae_cache")  # shared by both runs
        env["PYMS_TEST_SYNC"] = self.sync
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = XYCE_LIBDIR + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        env.pop("VAE_SO_PATH", None)
        env.pop("VAE_SO_DIR", None)
        self.env = env
        self.runs = []

    def tearDown(self):
        for r in self.runs:
            if r.proc is not None and r.proc.poll() is None:
                r.proc.kill()
                r.proc.wait(timeout=30)
        if not os.environ.get("PYMS_TEST_KEEP"):
            shutil.rmtree(self.work, ignore_errors=True)

    # -- helpers ----------------------------------------------------------
    def _go(self, tag):
        open(os.path.join(self.sync, tag + ".go"), "w").close()

    def _start_until_compiling(self, run):
        run.start()
        self.assertTrue(_wait_for(os.path.join(self.sync, run.tag + ".compiling")),
                        "%s never reached its shell compile (log: %s)" % (run.tag, run.log_path))

    def _check(self, run, expect):
        self.assertEqual(run.rc, 0, "%s: Xyce exit %s\n%s" % (run.tag, run.rc, run.text[-3000:]))
        for bad in BAD_MESSAGES:
            self.assertNotIn(bad, run.text, "%s: %r in the Xyce log" % (run.tag, bad))
        self.assertIsNotNone(run.out, "%s: no V(out) in %s.prn" % (run.tag, run.deck))
        self.assertLessEqual(
            abs(run.out - expect), RTOL * expect,
            "%s: V(out) = %r, expected %r (the other module's value = a silently wrong device)"
            % (run.tag, run.out, expect))

    def _gxx_calls(self, tag):
        p = os.path.join(self.sync, tag + ".gxx.log")
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return f.read().splitlines()

    def _check_atomic_writes(self, runs):
        """No .so was linked IN PLACE under its final cache name; no private build file is left."""
        hdl, vae = self.env["PYMS_CACHE"], self.env["PYMS_VAE_CACHE"]
        final = set()
        for d in (hdl, vae):
            for fn in os.listdir(d):
                if fn.endswith(".so"):
                    final.add(os.path.realpath(os.path.join(d, fn)))
        self.assertTrue(final, "no .so in the caches")
        linked = []
        for r in runs:
            for call in self._gxx_calls(r.tag):
                args = call.split()
                if "-shared" in args and "-o" in args:
                    linked.append(os.path.realpath(args[args.index("-o") + 1]))
        self.assertTrue(linked, "the g++ wrapper logged no shared-library link")
        for target in linked:
            self.assertNotIn(target, final,
                             "%s was linked in place under its final cache name (not an atomic write)" % target)
        leftovers = [fn for d in (hdl, vae) for fn in os.listdir(d)
                     if fn.endswith(".tmp") or fn.startswith("gen.") or re.search(r"\.build\.\d+$", fn)]
        self.assertEqual(leftovers, [], "private build files left in the caches: %r" % leftovers)

    # -- tests ------------------------------------------------------------
    def test_different_modules_of_one_name_interleaved(self):
        """A (a1, r=1k) is held in its shell compile while B (a1, r=3k) generates; A must read 0.5, B 0.25."""
        A = Run(self.work, "A", self.env, va_text=VA % {"r": "1k"})
        B = Run(self.work, "B", self.env, va_text=VA % {"r": "3k"})
        self.runs += [A, B]
        self._start_until_compiling(A)      # A's generator has written its shell source
        self._start_until_compiling(B)      # B's generator has written ITS shell source
        self._go("A")
        A.wait()                            # A compiles now: whose source?
        self._go("B")
        B.wait()
        self._check(A, 0.5)
        self._check(B, 0.25)
        self._check_atomic_writes([A, B])
        slots = sorted(fn for fn in os.listdir(self.env["PYMS_CACHE"]) if fn.endswith(".so"))
        self.assertEqual(len(slots), 2, "two identities of module a1 need two cache slots: %r" % slots)

    def test_same_module_twice(self):
        """Two concurrent runs of the SAME a1.va through one cache (a parallel sweep): both right, one slot."""
        shared_va = os.path.join(self.work, "shared_a1.va")
        with open(shared_va, "w") as f:
            f.write(VA % {"r": "1k"})
        A = Run(self.work, "A", self.env, va_path=shared_va)
        B = Run(self.work, "B", self.env, va_path=shared_va)
        self.runs += [A, B]
        self._start_until_compiling(A)
        self._start_until_compiling(B)
        self._go("A")
        self._go("B")                       # both compile the same identity at once
        A.wait()
        B.wait()
        self._check(A, 0.5)
        self._check(B, 0.5)
        self._check_atomic_writes([A, B])
        slots = sorted(fn for fn in os.listdir(self.env["PYMS_CACHE"]) if fn.endswith(".so"))
        self.assertEqual(len(slots), 1, "one identity of module a1 needs one cache slot: %r" % slots)


if __name__ == "__main__":
    unittest.main()
