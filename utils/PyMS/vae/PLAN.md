# VAE Regime Compilation Plan

## Architecture

Per-instance, JIT-compiled, branch-free eval/jacobian code for compact models.

### Key principles

1. **All parameters are constants** — model card + instance params (L, W, NF, etc.)
   are resolved at compile time. Only node voltages remain as runtime variables.

2. **Temperature is a node** — it can change fast in power scenarios, not fixed at
   compile time.

3. **Regime = unique branch path** — at each timestep all node voltages are known,
   so every condition resolves deterministically. Each unique set of condition
   outcomes defines a "regime" (off, subthreshold, linear, saturation, etc.).

4. **Branch-free compiled code per regime** — no if/else in eval or jacobian,
   exact Jacobians within a regime, no convergence issues from branch switching
   during Newton iterations.

5. **JIT compile on demand, cache by hash** — first time a regime is hit, compile
   it. Cache to disk keyed by `(model_hash, instance_hash, regime_key)`.
   Transistors with same L/W/NF share cached .so files.

6. **Limit timestep on regime change** — when condition outcomes change between
   timesteps, signal the simulator to back up and binary-search for the exact
   crossing point, then continue just past the transition.

### Cache hierarchy

```
~/.vae/cache/
  <model_hash>/              # hash of .va source
    <instance_hash>/          # hash of param_values (L/W/NF combo)
      regime_<key>.so         # compiled regime
      regime_<key>_ginac.cpp  # GiNaC source (optional, for debug)
      regime_<key>_eval.cpp   # generated eval/jacobian code
```

L/W combos are limited in practice (5-20 in a digital design), and each typically
hits 3-5 regimes, so total .so count is small (~50 for an entire design).

### Compilation pipeline (per regime)

```
.va source
  → iverilog -E (preprocess)
  → Python parser → AST
  → GiNaCEmitter(forced_conditions={idx: True/False, ...})
  → GiNaC C++ program (branch-free, all conditions resolved)
  → g++ -lginac -lcln → run → eval.cpp (branch-free eval + jacobian via CSE + forward-mode AD)
  → g++ -shared -fPIC → regime_<key>.so
  → dlopen → function pointers
```

## Condition analysis (BSIM-CMG 110 example)

- 381 total IF conditions in the model
- 306 resolved from model card parameters alone
- 12 more resolved from computed intermediate variables
- ~50-63 remain voltage-dependent (define regimes)

Voltage-dependent condition groups:
- Temperature update block (`DevTemp != TempLast`) — always true on eval
- Source/drain swap (`vds_noswap < 0`)
- Core MOSFET: `qis`, `qid`, `Vdseff` comparisons (saturation/linear/subthreshold)
- Junction diodes: `Isbs > 0`, `Isbd > 0`, bias vs VjsmFwd/VjsmRev
- Junction tunneling: `JTSS_t > 0`, etc.
- Junction capacitance: `Czbs > 0`, `T1 < 0.9`, etc.

Many conditions are nested — child conditions only matter when parent is True/False.
This dramatically reduces the feasible regime count.

## Implementation status

### Done
- [x] `vae/parser.py` — Verilog-A parser producing AST
- [x] `vae/ginac_emitter.py` — GiNaC C++ emitter with CSE (symbol-per-intermediate,
      forward-mode AD for Jacobian)
- [x] `forced_conditions` parameter in GiNaCEmitter — allows regime compilation
      to force all voltage-dependent conditions to specific True/False outcomes
- [x] Condition registry in GiNaCEmitter — tracks voltage-dependent conditions
      by index, synchronized with RegimeAnalyzer
- [x] `vae/regime.py` — RegimeAnalyzer (condition extraction + nesting analysis),
      RegimeCache (JIT compile + disk cache), CompiledRegime (ctypes function ptrs)

### Done (continued)
- [x] `assume_true` parameter — conditions matching patterns (e.g. 'DevTemp',
      'TempLast') are always resolved True (for varying temperature)
- [x] AST node identity mapping — conditions identified by id(node) not
      sequential index, avoiding sync issues between analyzer and emitter
- [x] Compiled C++ condition evaluator — `vae_eval_regime()` computes all
      intermediates as plain C++ doubles and returns regime key bitmask
- [x] End-to-end pipeline working: elaborate → condition eval → JIT compile
      regime → load .so via ctypes
- [x] extern "C" wrapper for .so exports

### TODO
- [ ] Regime change detection and timestep limiting interface
- [ ] Wrapper bash script for CLI usage
- [ ] Handle `#define temperature V_t` mapping in wrapper code
- [ ] Test with multiple regimes and verify different compiled outputs
- [ ] Validate results against ADMS reference
- [ ] Clean up GiNaC intermediary files after compilation
- [ ] Parallel compilation (multiple regimes simultaneously)

## Files

| File | Purpose |
|------|---------|
| `vae/parser.py` | Verilog-A → AST |
| `vae/ginac_emitter.py` | AST + params + forced_conditions → GiNaC C++ program |
| `vae/regime.py` | Regime analysis, JIT compilation, caching |
| `vae/codegen.py` | (existing) Direct C++ codegen without GiNaC |

## Runtime API (for simulator integration)

```python
from vae.regime import RegimeCache, parse_modelcard

# Once per instance type
params = parse_modelcard("modelcard.nmos")
params.update({"L": 14e-9, "W": 100e-9, "NF": 2})
cache = RegimeCache("bsimcmg.va", params)
cache.elaborate()

# Per timestep (inside Newton loop)
regime, changed = cache.get_regime(voltages=[Vd, Vg, Vs, Ve, Vt, Vsi, Vdi])
if changed:
    signal_regime_change()  # limit timestep

# Eval
state = VaeState(V=voltages, Vt=kT_q)
regime.eval_fn(state, F, Q)
regime.jacobian_fn(state, dFdV, dQdV)
```
