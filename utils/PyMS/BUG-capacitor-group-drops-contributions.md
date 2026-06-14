# BUG: PyMS Verilog-A device in `xyceModelGroup="Capacitor"` silently ignores all non-capacitive contributions (RSER / RPAR)

**Component:** PyMS / VAE (Verilog-A → Xyce device codegen), `xyceModelGroup`/`xyceLevelNumber` model-group binding
**Severity:** High — *silent* wrong answers (no error, no warning). A lossy capacitor model simulates as an ideal capacitor.
**Status:** open — found while validating the QSPICE→Xyce `qspice_cap` loss model before retiring `qspice2xyce.pl`'s companion-resistor expansion.
**Reference engine:** ngspice 45.2 (and closed-form analytic); cross-checked vs QSPICE.

## Summary

A Verilog-A capacitor that carries series/parallel loss (`RSER`, `RPAR`) and is bound to
`(* xyceModelGroup="Capacitor" xyceLevelNumber="91" *)` simulates as if `RSER=0` and
`RPAR=∞`. The capacitance is honored; **every other branch contribution in the module is
dropped.** Both the internal-node series branch (`RSER`) and a *direct* p–n branch
(`RPAR`, no internal node) disappear, in both `.TRAN` and `.AC`.

## Module (qspice_va/qspice_cap/qspice_cap.va)

```verilog
(* xyceModelGroup="Capacitor" xyceLevelNumber="91" *)
module qspice_cap(p, n);
    inout p, n;  electrical p, n;  electrical mid;
    parameter real C = 1e-12, RSER = 0.0, RPAR = 1e30;
    analog begin
        V(p, mid) <+ RSER * I(p, mid);     // series ESR  -> DROPPED
        I(mid, n) <+ C * ddt(V(mid, n));   // capacitance -> kept
        I(p, n)   <+ V(p, n) / RPAR;        // leakage     -> DROPPED
    end
endmodule
```

## Minimal reproducer

`C=1u, RSER=10, RPAR=100k`. Drive 1 A AC into the node so `V(out) = |Z(f)|`.

```spice
* VA path (Xyce, level=91)
I1 0 out AC 1
.MODEL qc c level=91 C=1u RSER=10 RPAR=100k
C1 out 0 qc
.ac dec 20 1 10meg
.print ac vm(out) vp(out)
.end
```

Reference (ngspice / any engine) — the equivalent explicit network:

```spice
* ideal R+C reference
I1 0 out AC 1
Rser ltz 0 10
C1  out ltz 1u
Rpar out 0 100k
.ac dec 20 1 10meg
.end
```

## Results — |Z| (Ω)

| f       | analytic `Rpar∥(Rser+1/jωC)` | ngspice 45.2 | QSPICE | **Xyce VA level=91** |
|---------|------------------------------|--------------|--------|----------------------|
| 1 Hz    | 84671                        | 84671        | 84682  | **159150** (= 1/ωC, Rpar gone) |
| 1 kHz   | 159.45                       | 159.45       | 159.48 | 159.15               |
| 10 MHz  | 9.999                        | 9.999        | 10.00  | **0.0159** (= 1/ωC, Rser floor gone) |

Transient (`.tran`, same network, pulse drive): ngspice vs QSPICE agree to 0.011%; the
Xyce VA is 1.98% off gold — and that 1.98% is simply the magnitude of the loss effect the
VA is dropping: **VA-with-loss vs VA-ideal-cap = 0.6%**, whereas the correct loss changes
the waveform by 1.97%. I.e. the VA tracks the *lossless* cap regardless of RSER/RPAR.

## Diagnosis / lead

`RPAR` is a direct `I(p,n) <+ V(p,n)/RPAR` contribution with **no internal node**, yet it
is dropped along with the internal-node `RSER` branch. That rules out an internal-node
collapse bug and points at the **model-group binding**: when the VAE device is registered
into the built-in `Capacitor` group, only the capacitive (charge/`ddt`) contribution is
extracted into the group's load/stamp path; the module's additional resistive
contributions (and their AC small-signal conductances) are never stamped.

Suggested next steps for whoever owns the VAE↔model-group glue:
1. Confirm the `Capacitor` group's VAE adapter only forwards the `ddt`/charge term.
2. Either (a) make the group's adapter stamp *all* of the module's branch contributions
   (DC/transient load + AC small-signal Jacobian), or (b) bind loss-bearing passives to a
   generic device group rather than `Capacitor`/`Inductor`.
3. Add the AC reproducer above to the PyMS regression set (assert |Z|→RSER at HF and
   →`Rpar∥…` at LF), since the failure is silent.

## Inductor (`xyceModelGroup="Inductor"`) — verified, and WORSE

`qspice_va/qspice_ind` (level=91, same RSER/RPAR shorthands) was tested the same way
(`L=1m, RSER=10, RPAR=100k`, 1 A AC drive). It is not merely loss-dropped — **the device
does not conduct at all.** It compiles and binds (`compiled and registered qspice_ind`),
but presents `|Z| ≈ 1e12 Ω` (the gmin floor) flat across frequency, and the transient is
100% off gold. The external node floats: the only thing that linked `p` to the internal
`mid` node was the `RSER` series branch, and with that branch dropped — and `RPAR`
(the one direct p–n path) dropped too — `p` has no connection to the `L` branch at all.

| f       | analytic `Rpar∥(Rser+jωL)` | ngspice | QSPICE | **Xyce VA level=91** |
|---------|----------------------------|---------|--------|----------------------|
| 1 Hz    | 9.999                      | 9.999   | 9.999  | **1e12** (open)      |
| 1 kHz   | 11.809                     | 11.81   | 11.809 | **1e12** (open)      |
| 10 MHz  | 53198                      | 53198   | 53198  | **1e12** (open)      |

The difference vs the capacitor is instructive: the cap branch is a **current**
contribution `I(mid,n) <+ C*ddt(...)` and survived as an ideal cap p–n (mid effectively
collapsed); the inductor branch is a **potential** contribution `V(mid,n) <+ L*ddt(I(...))`
that needs a branch-current unknown, and with the series branch gone the node is left
open rather than collapsed. So the model-group VAE adapter both (a) forwards only the
group's primary branch and (b) handles the potential-contribution + internal-node case by
leaving it disconnected. Both need fixing.

## Impact on the QSPICE→Xyce port

`qspice2xyce.pl` is **kept on the companion-resistor expansion** (explicit Rser/Rpar
elements) for now — that path matches analytic + ngspice + QSPICE to <0.02% in both
domains. The VA loss path cannot replace it until the above is fixed, or it would silently
turn every lossy C/L ideal under `.AC`.

Repro decks: `/mnt/c/cygwin64/tmp/vacmp/` (`gold_ac.cir`, `va_ac.cir`, `ng_ac.cir`, plus the
`.tran` variants); impedance extractor `~/zext2.pl`.
