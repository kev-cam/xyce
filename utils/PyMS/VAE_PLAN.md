# Verilog-A Extension for Xyce (VAE)
## Replacing ADMS with a Python/GiNaC/JIT Pipeline

---

## Overview

VAE is a standalone Verilog-A compiler and runtime extension for Xyce that replaces the
ADMS/XSLT toolchain. It operates as a netlist-level extension — the compiler emits standard
extended SPICE `.subckt` descriptions containing embedded C++ fragments, which Xyce loads
via a single generic device wrapper at simulation time. Xyce itself requires no internal
modifications beyond the generic wrapper device and the Python CLI hook already in place.

### Design Principles

- **Netlist as integration point.** The compiler output is SPICE text. Xyce has no knowledge
  of what produced it.
- **Analog blocks are opaque strings.** The sv2ghdl/NVC elaboration pipeline accumulates
  analog block strings and connectivity during VHDL-AMS elaboration and emits subcircuit
  descriptions. The compiler never parses VHDL.
- **Blended expressions over branching.** Verilog-A `if/else` conditions are translated to
  smooth sigmoid multipliers by default, producing single differentiable expressions that
  GiNaC differentiates in one pass with no path tracking or bisection required.
- **Tiered models for power electronics.** Background threads pre-evaluate device state and
  select the simplest valid model for the next timestep, reserving full nonlinear evaluation
  for switching transients only.
- **Thevenin decoupling at net boundaries.** Net interconnect is modeled as reduced-order
  Thevenin equivalents derived from SPEF back-annotation, decoupling analog simulation
  domains into independently solvable clusters.

---

## Architecture

```
Verilog-A source (.vams)
        │
        ▼
┌───────────────────┐
│  Python VAE       │  parser, path extractor, blend substitutor,
│  (xyce/utils/vae) │  sympy diff for simple cases
└────────┬──────────┘
         │ hard expressions
         ▼
┌───────────────────┐
│  GiNaC diff       │  transcendentals, implicit equations,
│  service (C++)    │  composite expressions
└────────┬──────────┘
         │ C++ expression strings
         ▼
┌───────────────────┐
│  JIT-1            │  clang++/LLVM → per-model .so
│  (compile + cache)│  shared with NVC LLVM infrastructure
└────────┬──────────┘
         │ .so path
         ▼
┌───────────────────┐
│  SPICE emitter    │  .subckt + .model_code cpp + .param
└────────┬──────────┘
         │ extended SPICE netlist
         ▼
┌───────────────────┐
│  Xyce             │  unchanged netlist parser
│                   │
│  GenericAnalog    │  one device class handles all .model_code cpp
│  Device           │  dlopen → function pointers → eval/jacobian
│                   │
│  JIT-2            │  compiles minimal binding fragment only
│                   │  heavy code lives in JIT-1 .so
└───────────────────┘
```

### NVC/sv2ghdl Integration

```
Verilog-AMS / SV source
        │  sv2ghdl
        ▼
VHDL-AMS (MMXL)
        │  NVC elaboration
        ▼
elaborated design
  + analog block strings       ← accumulated per instance
  + port maps                  ← connectivity resolved
  + parameter bindings         ← elaboration-time constants
  + SPEF Thevenin models       ← per net boundary
        │  SPICE emitter (new, trivial)
        ▼
extended SPICE netlist → Xyce
```

The SPICE emitter is a walk of the elaborated design structure. It has no analog semantics.
Each analog block instance becomes one `.subckt`. Each net boundary becomes one Thevenin
`.subckt`. Both are emitted in the same pass.

---

## Component Specifications

### 1. Python VAE Package — `xyce/utils/vae/`

#### `vae/parser.py`

Parses the `analog begin...end` block only. Not a full Verilog-A compiler.

Responsibilities:
- Tokenize analog block content
- Identify contribution operators `<+`
- Identify branch conditions `if/else`, `case`
- Identify `ddt()`, `idt()`, `ddx()` operators
- Identify `$limit`, `$bound_step`, `$discontinuity` calls
- Identify internal nodes (branch declarations not on port list)
- Emit AST as Python dataclasses

Does not parse: module headers, port declarations, discipline/nature definitions.
These are provided by the NVC elaboration context.

#### `vae/pathextract.py`

Static path analysis over the AST.

```python
@dataclass
class AnalogPath:
    conditions: list[Expr]       # conjunction reaching this path
    contributions: list[Contrib] # I<+, V<+, Q<+ on this path
    path_id: int                 # stable integer for runtime use
    blend_eligible: bool         # all conditions safe to smooth
```

Produces a path tree via DFS over `if/else`/`case` nodes. Leaf nodes are paths.
Detects boundary singularities (expression diverges where condition is zero) and
marks those paths `blend_eligible=False`.

#### `vae/blend.py`

Sigmoid substitution for eligible conditions.

```python
def blend_if_else(condition: Expr, expr_A: Expr, expr_B: Expr,
                  epsilon: float) -> Expr:
    boundary = condition.lhs - condition.rhs
    H = sigmoid(boundary / epsilon)
    return H * expr_A + (1.0 - H) * expr_B
```

Default epsilon selection (in priority order):
1. Per-condition annotation `// VAE_EPSILON=value`
2. Module-level `\`pragma vae_epsilon value`
3. Natural scale: `kT/q ≈ 0.02585V` for voltage conditions
4. Global CLI `--epsilon` flag
5. Fallback: `1e-3`

Blending is skipped when:
- `// VAE_NO_BLEND` annotation on the condition
- `` `pragma vae_blend off `` at module level
- `--no-blend` CLI flag
- `blend_eligible=False` (singularity detected)
- Topology change between branches (different internal nodes active)

When blending is skipped the condition falls back to hard path tracking
(Phase 3b bisection, see GenericAnalogDevice below).

#### `vae/diffgen.py`

Symbolic differentiation with tiered fallback.

```python
def differentiate(expr: Expr, wrt: Symbol) -> str:
    # Tier 1: direct C++ — linear, constant coefficients
    if expr.is_linear(wrt):
        return emit_linear_deriv(expr, wrt)

    # Tier 2: sympy — polynomial, rational
    result = sympy.diff(expr.to_sympy(), wrt)
    if is_simple(result):           # no transcendentals, no implicit deps
        return emit_cpp_expr(result)

    # Tier 3: GiNaC diff service
    return ginac_diff_service(expr, wrt)
```

`is_simple()` passes: polynomial, rational, piecewise-linear.
`is_simple()` fails: `exp`, `log`, `sin`, `tanh`, implicit dependencies, nested compositions.

Finite difference fallback is available as last resort when GiNaC service is unavailable,
controlled by `--fd-jacobian` CLI flag.

#### `vae/ginac_service.py`

Interface to GiNaC differentiation service.

GiNaC runs as a small C++ helper binary (`vae-ginac-svc`) that accepts an expression
string on stdin and returns a C++ expression string on stdout. This keeps GiNaC in its
native C++ environment with no Python binding requirements. The helper is compiled as
part of the VAE build and lives in the Xyce binary directory.

The helper uses GiNaC `print_csrc_double` for output, producing expressions ready for
direct inclusion in generated C++.

Expression caching: results are memoized by expression hash to avoid repeated GiNaC
invocations for the same expression across multiple instantiations.

#### `vae/tiering.py`

Model tier extraction for power electronics.

```python
@dataclass
class ModelTier:
    name: str           # "off", "linear", "saturation", "switching"
    condition: Expr     # when this tier is valid
    cpp_so: Path        # compiled .so for this tier
    stamp_cost: int     # estimated flops (scheduler hint)
```

Tiers are derived directly from the path conditions already extracted by `pathextract.py`.
No additional analysis required. Each path → one tier. Tier conditions are the path
condition conjunctions.

Per-tier `.so` compilation reuses JIT-1 infrastructure with the tier-specific expression
subset. All tiers are compiled at model load time and cached.

The sigmoid value `H` for each blended condition is a free byproduct of eval and is
exposed via `vae_sigmoid_values()` in the `.so` ABI for use by the tier predictor.

#### `vae/jit1.py`

First-stage JIT: Verilog-A expressions → compiled `.so`.

```
hash(cpp_source) → cache key
~/.cache/vae/{model_name}/{hash}.so
```

Compilation:
```
clang++ -O2 -shared -fPIC -std=c++17
        -I{xyce_include}/vae
        -o {cache_path}.so
        {generated}.cpp
```

Uses NVC's LLVM infrastructure when available as an in-process alternative to
subprocess clang++. Falls back to system clang++ unconditionally if LLVM path fails.

The same `.so` is shared across all instances of the same subcircuit. Instance-specific
parameter values are passed at call time, not compiled in.

#### `vae/emit.py`

Extended SPICE subcircuit emitter.

```python
def emit_analog_instance(block: AnalogBlock) -> str:
    # emits .subckt with:
    #   port list from Verilog-AMS port declarations
    #   .param for each elaboration-time bound parameter
    #   .model_code cpp: minimal JIT-2 fragment that dlopen()s JIT-1 .so
    #   internal node declarations if present

def emit_net_boundary(net: Net, thevenin: TheveninModel) -> str:
    # emits .subckt with:
    #   driver + receiver ports
    #   .param poles=... residues=... (AWE/PRIMA reduction results)
    #   .model_code cpp: rational function evaluator
```

Thevenin reduction order is selected per net based on the connected analog block's
declared accuracy tier:

| Mode | Reduction | Use case |
|---|---|---|
| Ideal | Wire | Functional verification |
| Elmore | Single RC pole | Fast timing signoff |
| AWE order-N | Rational approx | SI / eye diagram |
| PRIMA | Passive ROM | PI / PDN analysis |
| Full SPEF | Xyce subcircuit | Golden signoff |

#### `vae/inspect.py`

Analysis and debugging output. Used by `xyce vae inspect`.

Reports per model:
- Extracted path tree with conditions
- Blend decision per condition (blended / fallback / reason)
- Epsilon value per condition
- Tier assignments and cost estimates
- Internal nodes detected
- Constructs requiring special handling (`$limit`, `@cross`, etc.)
- Per-condition derivative expressions (sympy or GiNaC source)

Emits sidecar JSON alongside `.so`:
```json
{
  "model": "bsim4",
  "conditions": [
    { "id": 0, "expr": "V(ds) - (V(gs) - Vth)",
      "blended": true, "epsilon": 0.02585,
      "tier": "saturation" },
    { "id": 1, "expr": "V(gs) - Vth",
      "blended": false, "reason": "singularity",
      "tier_boundary": "off/linear" }
  ],
  "internal_nodes": ["tnode_b", "tnode_d"],
  "special": ["$limit:pnjlim", "@cross:V(osc)"]
}
```

---

### 2. GiNaC Differentiation Service — `xyce/utils/vae/ginac/`

Small standalone C++ binary. Single responsibility: differentiate one expression.

ABI:
```
stdin:   expression string + "\nwrt\n" + symbol name
stdout:  C++ expression string (print_csrc_double format)
stderr:  error messages
exit 0:  success
exit 1:  parse error
exit 2:  differentiation failed
```

Built as part of the Xyce build system. No runtime dependencies beyond GiNaC.
Located at `{xyce_bin}/vae-ginac-svc`.

Handles:
- Transcendental functions: `exp`, `log`, `sin`, `cos`, `tan`, `tanh`, `sqrt`
- Implicit differentiation (expression contains both lhs symbol and derivative target)
- Composite and nested expressions
- Temperature-dependent expressions with `$temperature` as a GiNaC symbol

---

### 3. Xyce Generic Analog Device — `src/DeviceModelPKG/`

One device class handles all `.model_code cpp` instances. Written once.

#### `.so` ABI

All VAE-compiled `.so` files export this interface:

```cpp
extern "C" {
    // Core evaluation
    void vae_eval(VaeState* s, double* contributions);
    void vae_jacobian(VaeState* s, double* dF_dV);

    // Path / tier tracking
    int    vae_path_id(VaeState* s);
    double vae_path_boundary(VaeState* s);   // signed: zero at changeover
    void   vae_sigmoid_values(VaeState* s, double* H); // per blended condition

    // Tier interface (power electronics)
    int    vae_tier_id(VaeState* s);
    double vae_tier_lookahead(VaeState* s, double dt, int steps);

    // Metadata
    const char* vae_model_name();
    int         vae_port_count();
    int         vae_internal_node_count();
    int         vae_path_count();
    int         vae_tier_count();
}
```

`VaeState` contains node voltages, parameter values, temperature, and timestep context.
It is populated by `GenericAnalogDevice` before each call.

#### `GenericAnalogDevice`

```cpp
class GenericAnalogDevice : public DeviceInstance {
    // JIT-1 .so handle and function pointers
    void*        so_handle_;
    VaeEvalFn    eval_;
    VaeJacobFn   jacob_;           // null → finite difference fallback
    VaePathIdFn  path_id_;
    VaePathBdyFn path_boundary_;
    VaeSigmoidFn sigmoid_values_;
    VaeTierIdFn  tier_id_;

    // State
    int          current_path_;    // path at last accepted timestep
    int          current_tier_;    // tier at last accepted timestep
    VaeState     state_;

    // Xyce DeviceInstance interface
    bool loadDAEVectors()   override;
    bool loadDAEMatrices()  override;
    bool updateIntermediateVars() override;
    bool updatePrimaryState() override;
    bool getBreakPoints(BreakPointVector&) override;

    // Internal node allocation
    void registerInternalNodes();  // called at instantiation
};
```

`loadDAEMatrices` uses analytic Jacobian from `jacob_` if available, otherwise
falls back to finite difference (N+1 evaluations of `eval_`).

#### JIT-2 Fragment

The `.model_code cpp` body emitted by `vae/emit.py` is minimal:

```cpp
// AUTO-GENERATED — do not edit
#include "vae_runtime.h"
#include <dlfcn.h>
static VaeSo* _vae = nullptr;
void vae_fragment_init(const char* so_path) {
    _vae = vae_load(so_path);  // dlopen + symbol binding
}
// vae_runtime.h provides the shim that GenericAnalogDevice calls into
```

JIT-2 compiles this fragment at Xyce netlist load time. It is trivially small.
The heavy model code lives entirely in the JIT-1 `.so`. The JIT-2 compile cost
is negligible even for large netlists.

#### Hard Path Fallback (Phase 3b)

When a condition is not blend-eligible, `getBreakPoints` performs bisection:

```cpp
bool GenericAnalogDevice::getBreakPoints(BreakPointVector& bps) {
    if (path_id_(&state_) != current_path_) {
        double t_cross = brent_bisect(
            t_start_, t_now_,
            [this](double t) { return path_boundary_(&state_at(t)); },
            1e-12   // tolerance
        );
        bps.insert(BreakPoint(t_cross));
    }
    return true;
}
```

Brent's method converges in ~15 evaluations worst case. A timestep-halving fallback
is used when `path_boundary_` has poor numeric conditioning near the crossing.

#### Special Construct Handling

| Construct | Handler |
|---|---|
| `$limit(v, "pnjlim", ...)` | Newton step clamp in `updateIntermediateVars` |
| `$bound_step(dt)` | `setMaxTimeStep()` call in `updatePrimaryState` |
| `$discontinuity(n)` | Direct `BreakPoint` insertion, no bisection |
| `@cross(expr, dir)` | `getBreakPoints` using `path_boundary_` for `expr` |
| `@above(expr)` | Same as `@cross` |
| `analog initial` | `initializeFromFile()` / IC injection at `t=0` |
| `white_noise()` | `loadNoiseSources()` additive contribution |
| `flicker_noise()` | `loadNoiseSources()` additive contribution |

---

### 4. Background Tier Predictor — `src/DeviceModelPKG/VaeTierPredictor`

Runs in a dedicated thread per simulation partition. Reads device state (no writes).
Updates a tier assignment table consumed by `GenericAnalogDevice` at timestep start.

```cpp
class VaeTierPredictor {
    // Reads: current device states from all GenericAnalogDevice instances
    // Writes: tier_table_[] — read-mostly, updated between timesteps only

    void run() {
        while (simulation_active_) {
            wait_for_timestep_complete();

            for (auto& dev : monitored_devices_) {
                int predicted = predict_next_tier(dev, dt_, lookahead_steps_);

                if (predicted != dev.current_tier_) {
                    // pre-warm .so into cache
                    dev.tiers_[predicted].prefetch();
                    // estimate crossing time
                    double t_switch = estimate_crossing(dev, predicted, dt_);
                    // inject breakpoint
                    breakpoint_queue_.push(BreakPoint(t_switch));
                    // update assignment
                    tier_table_[dev.id_] = predicted;
                }

                // near-switching early upgrade
                if (near_switching(dev, dt_, lookahead_steps_)) {
                    tier_table_[dev.id_] = FULL_MODEL;
                }
            }
        }
    }
};
```

Near-switching detection uses sigmoid values as a free indicator:

```cpp
bool near_switching(GenericAnalogDevice& dev, double dt, int lookahead) {
    // H values are already computed during eval — no extra cost
    for (auto H : dev.sigmoid_values_) {
        if (H > 0.05 && H < 0.95)   // inside transition region
            return true;
    }
    // extrapolate N steps
    return predict_next_tier(dev, dt, lookahead).cost >
           dev.current_tier_.cost * ESCALATION_THRESHOLD;
}
```

The predictor thread is only activated when tiered models are present in the netlist
(i.e., when any loaded `.so` exports `vae_tier_count() > 1`). For standard analog
models it does not run.

---

### 5. Thin Shim Header — `include/vae/ltz_analog.h`

All generated C++ fragments include only this header. It is the complete
public API surface visible to model authors.

```cpp
// ltz_analog.h — VAE model author API
#pragma once

// Node voltage access
double V(port_t p);
double V(port_t p, port_t q);

// Current / charge contributions
void   I_contrib(port_t p, double val);
void   Q_contrib(port_t p, double val);   // ddt() maps here

// Simulator environment
extern double $temperature;
extern double $timescale;
double        $vt(double T);
double        param(const char* name);

// Convergence / timestep hints (structural — handled by wrapper)
double $limit(double v, const char* func, double arg1, double arg2);
void   $bound_step(double dt);
void   $discontinuity(int order);

// Port type (opaque)
typedef int port_t;
```

Model authors never see Xyce internals, MNA matrices, sparse indices, or
Newton iteration state.

---

### 6. CLI — `xyce vae`

Integrated with the existing Python CLI in Xyce.

```
xyce vae compile  <file.vams>              Compile model, emit .subckt
xyce vae compile  --inline <file.vams>     Embed C++ directly (no .so ref)
xyce vae compile  --no-blend <file.vams>   Force hard paths everywhere
xyce vae compile  --epsilon=<v> <f.vams>   Global epsilon override
xyce vae compile  --fd-jacobian <f.vams>   Force finite difference Jacobian
xyce vae compile  --tiers <file.vams>      Enable power electronics tiering
xyce vae inspect  <file.vams>              Show paths, blend decisions, tiers
xyce vae cache    --list                   Show cached .so entries
xyce vae cache    --clear                  Purge JIT-1 .so cache
xyce vae cache    --clear <model>          Purge specific model
xyce vae validate <file.vams> <ref.raw>    Compare against reference waveform
```

---

## Deliverable Sequence

| Phase | Deliverable | Unlocks |
|---|---|---|
| 1a | `parser.py` + `pathextract.py` | `xyce vae inspect` |
| 1b | `blend.py` + sympy diff tier | blended single-expression models |
| 1c | `ginac_service` + `diffgen.py` | transcendental / implicit models |
| 1d | `jit1.py` + `.so` cache | compiled model artifacts |
| 2a | `GenericAnalogDevice` (FD jacobian, no tiering) | first Xyce simulations |
| 2b | `emit.py` + JIT-2 fragment | end-to-end pipeline |
| 2c | Analytic jacobian via `jacob_` fn pointer | Newton convergence at PDK level |
| 3a | Hard path fallback + `getBreakPoints` | blend-ineligible conditions |
| 3b | Brent bisection on `path_boundary_` | accurate crossing time |
| 4a | `tiering.py` + per-tier `.so` compilation | power electronics model tiers |
| 4b | `VaeTierPredictor` background thread | runtime tier switching |
| 4c | Near-switching early upgrade via sigmoid | GaN/SiC fast switching support |
| 5a | Thevenin SPEF reduction + `emit_net_boundary` | back-annotated parasitics |
| 5b | NVC elaboration SPICE emitter | full Verilog-AMS → Xyce pipeline |
| 6  | `xyce vae validate` + ADMS comparison suite | sign-off confidence |

Phase 3b is the primary technical risk item. The Brent bisection is robust in principle
but `path_boundary_` conditioning near region boundaries varies by model. The timestep-
halving fallback must be proven stable before phase 3b is considered complete.

Phase 2a and 2b are the minimum viable pipeline. Everything after is additive
without architectural change.

---

## File Layout

```
xyce/
  utils/
    vae/
      __init__.py
      parser.py
      pathextract.py
      blend.py
      diffgen.py
      ginac_service.py
      tiering.py
      jit1.py
      emit.py
      inspect.py
      cli.py                   ← registered with existing xyce Python CLI
      ginac/
        CMakeLists.txt
        vae_ginac_svc.cpp      ← GiNaC differentiation service binary
        vae_ginac_svc.h
  include/
    vae/
      ltz_analog.h             ← model author API
      vae_runtime.h            ← JIT-2 fragment support
      vae_so_abi.h             ← .so export interface
  src/
    DeviceModelPKG/
      VAE/
        N_DEV_GenericAnalog.h
        N_DEV_GenericAnalog.cpp
        N_DEV_VaeTierPredictor.h
        N_DEV_VaeTierPredictor.cpp
        N_DEV_VaeRuntime.h
        N_DEV_VaeRuntime.cpp   ← dlopen/dlsym management, VaeState population
  test/
    vae/
      models/                  ← reference .vams files (diode, MOSFET, BJT, IGBT)
      reference/               ← reference .raw waveforms from ADMS baseline
      test_pipeline.py
      test_blend.py
      test_tiering.py
      test_thevenin.py
```

---

## Notes

**ADMS compatibility.** Models currently supported by the Xyce ADMS flow are the
validation baseline. Phase 6 (`xyce vae validate`) runs the VAE pipeline against
ADMS-generated results for all models in the Xyce model library and reports
deviation. VAE replaces ADMS only when validation passes for a given model.

**PDK model support.** BSIM4, BSIM-CMG, PSP, and HiSIM all use internal nodes
extensively. Internal node support in `GenericAnalogDevice` (Phase 2a) must be
validated against a BSIM4 instance before phase 2 is considered complete.

**Thread safety.** `VaeTierPredictor` reads `GenericAnalogDevice` state between
timesteps only. The tier assignment table is written by the predictor and read
by compute threads at timestep start — a single atomic integer per device is
sufficient for the assignment; no locking on the compute path.

**GiNaC availability.** The build system detects GiNaC at configure time. If
absent, `diffgen.py` falls back to sympy for all cases and emits a warning for
expressions that exceed sympy's simplification capability. Finite difference
Jacobians are used as final fallback. Simulation correctness is not affected;
Newton convergence may degrade for stiff models.
