# SMP device-load threading — research prototypes

Investigation into on-node shared-memory parallelism for Xyce, targeting the
regime where MPI loses (small/medium circuits, single box). MPI is the wrong
primitive there — it's message-multiplexed and sync/load-imbalance-bound (np=8
is ~2x *slower* than serial on small circuits). The shared-memory approach uses
core-pinned threads + sense-reversing spinlock barriers (no futex/kernel).

## Where the time actually goes (and what's NOT the lever)

After fixing the build (everything was `-O0`/debug; `-O3 -march=native` gave a
3.3x on Xyce + a further 1.6x from Trilinos → 5.1x total, gap to QSPICE on a
small bench cut from 21x to ~4.2x), profiling a *large* circuit (chain10000,
KLU) shows:

| phase | % wall |
|---|--:|
| **device load** (Residual F 58% + Jacobian dF 18%) | **76%** |
| linear solve (KLU) | **4.5%** |
| Newton + time-integrator | ~19% |

So **the solve is NOT the bottleneck** — KLU (BTF+AMD, purpose-built for SPICE
matrices) is best-in-class. We tried Amesos2 **Basker** (the threaded sparse LU):
serial Basker is **38x slower** than KLU on chain10000, and the solve is only
4.5% anyway — a dead end. **The device LOAD (76%, embarrassingly parallel) is
the lever.**

## Prototypes

### `modeleval.c` — affinity + parallel model-eval microbench
N independent "devices" eval'd per Newton iter, partitioned across P core-pinned
threads, spinlock-barriered. Findings (5955WX, 16C/32T, under WSL2):
- **~9x on 16 cores while the working set fits L3 (≤32 MB)**; collapses to ~3.5x
  past L3 → memory-bandwidth-bound. The L3 is the cliff ("32 caches vs 1" holds
  up to the aggregate-cache working set).
- **Pinning gains +12 / +19 / +28 % at P=8 / 16 / 32** — affinity matters once
  the machine is fully subscribed (the OS load-balancer thrashes L2 otherwise).

### `threadload.c` — threaded device LOAD, faithful to Xyce's pattern
Models `i0 = eval(sol[a],sol[b]); fVec[a]+=i0; fVec[b]-=i0` over a chain's
connectivity (the real stamp into a shared raw F-vector, which RACES on shared
nodes). Compares serial vs two threading strategies, core-pinned:
- **`tlbuf` (thread-local stamp buffers + locality-aware O(M) merge): 9.5x @16
  cores, 15.3x @32 threads.** Beats atomic-stamp (per-op CAS overhead).
- **Affinity is decisive at full subscription:** at P=32, pinned **15.3x** vs
  unpinned **2.1x** — a 7x swing. Pinning is load-bearing, not an optimization.

## Projection & integration plan

Device load = 76% ⇒ Amdahl with a 9.5x load speedup → **~3.1x total** on a large
circuit (chain10000 6.2s → ~2s). The remaining serial floor (solve + Newton +
integrator, ~24%) caps it. Won't help small circuits (load ~30% → ~1.4x), so the
QSPICE small-circuit gap stays ~3x — the win is the medium/large + scale-out
regime, plus breadth (open, digital cosim, Verilog-A via PyMS).

**Real Xyce integration** (the next step): the per-instance load reads `fVec`
from the shared `extData` raw pointer, so threading needs splitting each device's
`loadDAEFVector` into a parallel `computeContributions()` (the expensive math →
instance members, *zero* shared-write contention) + a serial `stamp()` (cheap).
Drive it with a core-pinned thread pool reusing the ShmComm spinlock barrier.
Start with one device (Resistor or MOSFET1) in `DeviceModelPKG/OpenModels`.

## Build / run
```
gcc -O3 -march=native modeleval.c  -o modeleval  -lpthread -lm
gcc -O3 -march=native threadload.c -o threadload -lpthread -lm
./threadload <nodes> <newton-iters> <pin 0|1>     # e.g. ./threadload 200000 500 1
```
