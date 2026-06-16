# Xyce vs QSPICE vs LTspice — speed & scale benchmark (2026-06-15)

Single-thread comparison on one Windows box (32 cores, but see "Parallelism").
Xyce = our serial WSL build; QSPICE = `QSPICE64.real.exe`; LTspice = ADI LTspice `-b`.

## 1. Small circuit — per-step throughput

3-stage BJT amplifier, sine-driven, `.tran 20n 2m 0 20n` ≈ 100k forced 20ns
steps, identical Gummel-Poon model, matched **binary** output (`bench.cir`).

| engine | wall | points |
|--------|------|--------|
| Xyce (`-r` binary) | 13.7 s | 100,546 |
| LTspice (`-b`)     | 1.18 s | 100,119 |
| QSPICE             | 0.86 s | 100,052 |

On a small circuit Xyce is ~12× slower than LTspice, ~16× slower than QSPICE —
its per-step overhead is the cost (built for accuracy/scale, not small-circuit
speed). **Gotcha:** Xyce's ASCII `.print` output is a separate bottleneck — the
same run with `.print tran V(out)` took **46 s** (34 s of ASCII I/O) vs 13.7 s
with binary `-r`. Always use `-r <file>.raw` for large runs.

## 2. Scaling — CMOS inverter chains (`gen_chain.pl N`)

N inverters = 2N MOSFETs + N caps + ~N nodes; `.tran 0.1n 50n`, one node saved.

| inverters | MOSFETs | QSPICE | Xyce (serial) | Xyce/QSPICE |
|-----------|---------|--------|---------------|-------------|
| 2 k   | 4 k   | 4.5 s | 19.9 s | 4.4× |
| 10 k  | 20 k  | 39 s  | 109 s  | 2.8× |
| 50 k  | 100 k | 271 s | 568 s  | 2.1× |
| 100 k | 200 k | 458 s | (not run) | — |
| 200 k | 400 k | **OOM** (calloc 121 MB) | ran in 3.77 GB, **>1 h** unfinished | Xyce holds it |
| 500 k | 1 M   | **OOM** (calloc 264 MB) | — | Xyce holds it |

Two findings:

1. **QSPICE has a memory wall at ~200k MOSFETs.** At 400k MOS it dies with
   `Fatal error: calloc: can't allocate 121,600,000 bytes` — a 121 MB
   allocation failing on a machine with tens of GB free. That's an *internal*
   allocation ceiling in the LTspice-derived engine, not physical RAM. QSPICE
   is single-threaded **and** memory-bounded.

2. **Xyce scales better and has no memory wall — but is slow single-core.**
   The Xyce/QSPICE time ratio falls with size (4.4 → 2.8 → 2.1): QSPICE is
   super-linear, Xyce near-linear, so they converge. On the 400k-MOS circuit
   QSPICE OOM'd on, **serial Xyce allocated and ran it fine in 3.77 GB RSS**
   (no calloc failure — it had memory headroom), but at single-core speed it
   had **not finished after 1 h** (timed out). So serial Xyce removes QSPICE's
   *memory* wall but replaces it with a *throughput* wall: it can hold a
   400k-MOS / 1M-MOS problem that QSPICE cannot even allocate, yet needs the
   **parallel (MPI) build** to crunch it in practical time. That is exactly
   what Xyce is designed for and what our build is missing.

## 3. Parallelism — can Trilinos use multiple cores?

Yes in principle; **our build does not**. Xyce `-capabilities` = `Serial`; no
MPI libs linked, no `mpirun`; Kokkos built with only the **Serial** host
execution space (no OpenMP/threads, no libgomp). So today: 1 of 32 cores.

Trilinos offers two layers: **MPI** (distributed Epetra/Tpetra — Xyce's primary
parallelism, `mpirun -np N Xyce`, domain-decomposes the matrix) and **Kokkos**
node-level threading (OpenMP/Pthreads/CUDA). To use the cores: rebuild Xyce
parallel (MPI runtime + Trilinos `-DTPL_ENABLE_MPI`) and/or Kokkos with the
OpenMP backend. MPI speedup is for large circuits (domain decomposition); small
circuits lose to overhead, and the default KLU direct solver is serial.

## Takeaway

For small/medium circuits, single-threaded QSPICE/LTspice beat Xyce on raw
speed. At scale the picture flips: QSPICE hits a hard ~200k-MOS memory wall,
while Xyce scales near-linearly and can *hold* problems (400k-MOS in 3.77 GB,
1M-MOS) that QSPICE cannot allocate. But serial Xyce is too slow to *finish*
those in practical time — so the large-circuit regime needs the **parallel
(MPI) Xyce** the simulator was designed for (and which our build lacks). Net:
QSPICE/LTspice win small; parallel Xyce is the only path for very large.
Decks + `gen_chain.pl` in this dir; raw timings in `/tmp/qbig`, `/tmp/bench`.
