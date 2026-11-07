# PyMS: open defects and design notes

This file is the work list for PyMS (`utils/PyMS`, the Verilog-A front end of
Xyce's `.hdl` route). Each section stands on its own: the deck and module, what
Xyce and VACASK (openvaf-r) measure, the mechanism (file and function), the fix
or design, and the test that should pass once it is fixed. Values marked
"(PYMS1-x)" were measured by the 2026-10-24/25 fix round's cluster x; values
marked "(2026-10-25)" were measured when this file was written. Work one
defect at a time: one test that fails first, one commit (`git log --
utils/PyMS`).

How to run a deck through PyMS without touching the shared caches:

    export PYMS_DIR=/path/to/your/PyMS PYMS_CACHE=$PWD/hdl PYMS_VAE_CACHE=$PWD/vae
    /usr/local/src/xyce-build/src/Xyce deck.cir

VACASK: `/opt/build.VACASK/Release/simulator/vacask` (openvaf-r compiles the
.va; through vamos: `spectre deck.scs` with `/usr/local/src/sv2ghdl/shims`
first on PATH). Every test in `vae/tests/` runs as a plain script
(`python3 vae/tests/test_x.py`) and makes its own caches.

Contents: CHANGES (1.1.7, 1.1.6, 1.1.5, 1.1.4, 1.1.3, 1.1.2, 1.1.1, 1.1.0) · A. silent wrong answers · B. loud failures of
legal Verilog-A · C. decisions and strictness · D. error messages ·
E. caches and installation · F. upstream (openvaf-r/VACASK) · G. vamos
follow-up.

---------------------------------------------------------------------------

## CHANGES

### 1.1.7 (2026-11-07)

The findings of the 1.1.6 code review (PYMS7-REVIEW, decks in its REV7/t1..t4) and the skeptic's A18:
- B22 (the review's F1, BLOCKING): the generated `Instance::acceptStep` cached `dlsym(vae_dl_, "vae_needs_accept")`
  in a function-static pointer keyed on the dlopen handle. A `.step` of a baked parameter makes processParams
  dlclose the math library and dlopen the new one; glibc returns the same handle while the code lies elsewhere, so
  the pointer was stale: SIGSEGV in 1.1.6, "Illegal instruction" in 1.1.5 (REV7/t1 b.cir: a held peak, `.step
  m1:mode list 0 1 0`, `.tran`). xyce_device_gen.py: the pointer is an instance member (`pyms_needs_accept_`), read
  with the held state when a library is loaded and cleared when it is dropped; it was the shell's only
  function-static dlsym cache. test_library_swap_accept.
- B23 (F2): the dual-number `pow(double, dual)` / `pow(dual, dual)` (dual_ad.py) took the exponent slope
  `r.v*log(a)` = 0*(-inf) = NaN at a zero base, and every Jacobian column the exponent depends on was NaN
  (`pow(max(V,0), 1+V)`: every DC point failed). The slope is 0 where the value is 0. test_pow_zero_base.
- A22 (F4, silent, since before 1.1.0): preprocess.py `_substitute` replaced the macro parameters one after
  another, so an argument naming a later parameter was substituted again (`define ADD2(a, b) ((a) + 2*(b))`,
  `ADD2(b, 250)` gave ((250) + 2*(250)): V(2) 0.428571428 for 0.6), and re.sub read backslashes in an argument as
  escapes. One combined pattern, one pass, the argument inserted literally. test_macro_args_one_pass.
- A18 (the 1.1.6 skeptic): `@(initial_step) th = $temperature;` was folded into `th = _vae_T` (an "invariant"
  initial_step body runs at every point) and followed a `.dc temp` sweep. ginac_emitter.py: a body that reads
  `$temperature`, `$vt` without an argument or a thermal parameter is no longer invariant, so it is held like a
  bias-dependent capture: the value of the initial step stays for the whole analysis (the user's thermal design:
  `pyms_tsrc_` may move within an analysis later), and every `.step` point captures again. `$vt(300.15)` and other
  temperature-free bodies stay invariant. test_temperature (test_initial_step_capture_held_under_dc_temp).

Not done: the review's F3 (B24, a V/I switch row that is a NaN when the untaken arm is non-finite) -- the blend is
the general IF-arm indicator mechanism, a select there is not a contained change. Found on the way: B25 (a held
peak in a transient with tight tolerances stalls; 1.1.6 too).

Gate (measured, PYMS8-FIX): vae/tests 59/59 modules (409 tests) on the chain; the review decks right (b.cir runs
its 3 steps, z2 converges, ms.cir 0.6); the 21 deck families (1255 decks) against 1.1.6: right 1137 / loud 43 / n/a
60 / wrong 11 / ok? 4, the same as 1.1.6, no deck worse, no value moved by more than 1e-9; corpus at 27 C 101 / 86 /
82 with every printed value equal to 1.1.6's; the SK7 temperature toys: a3_dc_temp and a5_multi_dc right, the only
failures left are A19 (noise) and A20 (`.step m1:TEMP`), with 1.1.6's values.

### 1.1.6 (2026-11-02)

Temperature (A17). Up to 1.1.5 `$temperature` and `$vt` were fixed at 300.15 K in every version (the shell set
`state.Vt = 8.617087e-5 * 300.15`, the wrapper had `static const double temperature = 300.15`): at
`.options device temp=127` or `.step temp` a PyMS device kept its 27 C behaviour while built-ins followed the
temperature. Every CMC model in the corpus reads `$temperature`.

Design (the user's direction: temperature may become local to the instance once thermal modelling exists, so it is a
per-instance run-time value reached through a pointer, not a constant):
- `VaeState` carries `double T`. The eval prologue computes `_vae_T = state->T` and `_vae_vt = VAE_KOQ * _vae_T`;
  `$temperature`, `$vt` and `$vt(x)` read those. Nothing temperature-dependent is in the math library or its cache
  key: one library serves every temperature (corpus at 27 C and 127 C: 1 library per file).
- Each shell instance has a temperature source pointer `pyms_tsrc_`. Today it points at the instance's device
  temperature `pyms_tdev_` (the circuit TEMP, or the instance's own TEMP / DTEMP when the module has no such
  parameter); `state.T = *pyms_tsrc_` at every evaluation. A thermal network node or a self-heating state can be
  plugged in later by re-pointing `pyms_tsrc_`, without regenerating code.
- The dual-number AD copy carries `T` too; it is a plain input today, and the place where a future thermal unknown
  would be seeded as an AD direction is marked in the emitter.
- Parameter defaults that read `$temperature` / `$vt` are thermal parameters: symbols computed in the eval prologue,
  so they follow a temperature change between analyses and `.step` points.
- `VAE_KOQ` = Xyce's k/q (8.617086918058125e-05, `CONSTKoverQ`), so PyMS and built-in devices agree. VACASK's
  constants differ slightly (diode at 127 C: 5e-4 relative).

Also: B21, a later module in a file registers its own xyceModelGroup / level (m6b_dd_second).

Gate (measured, FIX7): 1255 decks in 21 families: right 1137 / wrong 11 (1.1.5: 1083 / 45; 1.1.4: 1032 / 95), no deck
worse except the expected f_th_vt (now runs, 2.96e-5 from P_K/P_Q constants) and the 1.1.5 AC rule deck vs 1.1.4; at
27 C no non-temperature value moved by more than 1e-9. Temperature tracking at 127 C vs VACASK (diode, bjtd504,
HiCUM L2, PSP103, BSIM-CMG 107): ~1e-8 except the diode's k/q 5e-4. vae/tests 56/56 on the chain. Corpus 101/86/82 at
27 C and 127 C; 27 C values equal 1.1.5's (BSIM6 V(out) 0 -> 1.4e-48). vamos: the same 20 failures as 1.1.5.
Not done this round: an independent skeptic of 1.1.5/1.1.6 (paused, SK6) and the code review.

Filed, not PyMS: two `.step` lines plus `.op` segfault Xyce with built-in devices only; Xyce ignores `.temp` for
built-ins too (use `.options device temp=` or `.step temp`).

### 1.1.5 (2026-11-01)

`__version__` and `VERSION` say 1.1.5. Fixed in this round (PYMS6-FIX, from
the PYMS5 skeptic's findings), each with a test in `vae/tests/` that fails on
1.1.4 and passes on 1.1.5 (test_exact_jacobian, test_held_step_reset,
test_scope_and_param_type, test_step_params_and_modules); values on the 1 V /
1k divider unless said otherwise, 1.1.4 -> 1.1.5, decks of the skeptic
(`sp2e/pyms/SK5`) and of this round (`sp2e/pyms/FIX6`). The device shell
(xyce_device_gen.py) and N_DEV_PyMS.C change: libxyce.so is relinked.

- The exact Jacobian (A12): forward-mode automatic differentiation of the
  eval code on dual numbers (new vae/dual_ad.py; build_vae_so.py), with a
  static dependency analysis, a Curtis-Powell-Reid colouring of the unknowns
  and the variables no unknown reaches kept double; it replaces the forward
  difference quotient of 1.1.0-1.1.4. ac gains: a diode on 1 kOhm 0.0650491
  -> 0.0652353; cathode at 10 V 1.0392 -> 0.997241; at 50 V 1.21578 ->
  0.999476 (the analytic gain of the operating point to 1e-9); bjtd504 Ib
  7.41928e-1 -> 7.43543e-1 (VACASK 7.43543e-1), with the emitter at 20 V
  7.23115e-1 -> 7.43544e-1; MVSG 3.5e-5 -> 1.5e-7 relative to VACASK; the
  45 V rectifier's transient 600 s timeout -> 9 s (324 steps; the built-in
  diode 347). A branch equation that is exactly flat at a point (c_own
  v_tanh) takes a central difference in that row only (A12). Faster:
  transient solver time PSP103 0.968 -> 0.649 s, BSIM-CMG 0.917 -> 0.658
  s, HiCUM L2 0.631 -> 0.492 s (1.1.3: 0.675, 0.788, 0.517); the library
  build 3-25 s longer (the dual copy's g++).
- Every analysis starts its held state from the declaration values, every
  .step point too (A13; a 1.1.4 regression): the accumulating
  initial_step over two steps 0.667 sin -> 0.5 sin; the cross hysteresis
  restarts from r = 4k.
- A .step of a baked parameter rebuilds the math library (A14): `.step
  m1:rp list 1000 4000` 0.5, 0.5 -> 0.5, 0.2 (a .global_param the card
  reads the same).
- A function local or argument that shadows a parameter is not a constant
  for a block-local declaration value (A15; a 1.1.4 regression):
  b5b_func_local_shadow_silent 0.5 -> 0.2, b5_hoist_func_local_param
  singular -> 0.2. Non-constant declaration values stay where they are
  written (the choice; not LRM-strict).
- Untyped parameters take the type of their default expression (B20
  corrected): `parameter m = 2 * 2`, `parameter b = a` (a integer)
  0.571428571 -> 0.5.
- A module's xyceModelGroup / level are its own (B21):
  m5_second_has_group_attr netlist error -> 0.5.
- One module name in two .hdl files: the first stays and the run warns
  (A16): m5_same_helper_two_files_use 0.473684 (silently the later file's)
  -> 0.466667 with the warning.
- In ac, a voltage condition inside a run-time loop at exactly 0 V takes
  the operating point's branch slope: s5_ac_regime_in_rtloop 0.8 -> 0.2
  (the exact Jacobian; C16).
- Filed: C18 (.ic with a V-reading initial_step: the model's own pole at
  Xyce's first .ic iterate, not PyMS), C19 (limexp is exp).
- BUGS.md corrected: C16 (the held state's start), B20 (untyped
  parameters), the 1.1.4 entries on untyped parameters and non-constant
  declaration values.

Gate (this round's measurements, `sp2e/pyms/FIX6`): the 1119 decks of every
family (the 977 of the earlier rounds and the PYMS5 skeptic's 142 attack
decks) on 1.1.5: right 1024, loud 37, n/a 47, wrong 8, ok? 3 (1.1.4 as the
skeptic measured it: 1005 / 39 / 47 / 25 / 3); against every earlier version
(1.1.0-1.1.3 as FIX5 and SK5 measured them, on their libraries) no deck
worse except three that are the same as 1.1.4 and older than this round
(smp3_cond_gmin_card and b5_hoist_param_assigned loud on 1.1.0/1.1.1, the
1.1.2-1.1.5 value; s4_loop_body_local_init, the 1.1.4 static rule = VACASK)
and s5_ac_is_vread_held (0.199948 -> 0.2: the user's rule, exact now); of
the decks with an expected value whose value moved, 21 closer to it, 1
farther by 1.8e-10 (h5_held_peak). vae/tests: 54 modules OK (50 + the 4
new). Corpus (the 111 files, one fresh Xyce each): 101 generate / 86 build /
82 run, as 1.1.4; the 82 give 1.1.4's values to 1e-9 but PSP103 JUNCAP200
(4e-9 / 6e-9 relative); the scan 3252 s (1.1.4 2766 s: the dual copy's
g++). vamos (run_xyce, trial2eg/eh/ei, engine_facts, cmc, e2e; sv2ghdl
5d45dbe): the same 20 failures as 1.1.4 on that tree.

Open after 1.1.5: as after 1.1.4, plus C18 (.ic at a model pole), C19
(limexp); b5_hoist_param_assigned (an assignment to a parameter, C11
dialect, is not seen by a block-local declaration value that reads it:
0.5 for 0.2, 1.1.2-1.1.5).

### 1.1.4 (2026-10-31)

`__version__` and `VERSION` say 1.1.4. Fixed in this round (PYMS5-FIX, from
the PYMS4 skeptic's findings), each with a test in `vae/tests/` that fails on
1.1.3 and passes on 1.1.4; values on the 1 V / 1k divider (X1 between in and
out, 1k from out to ground; analytic V(out) = 1000 / (R + 1000)), 1.1.3 ->
1.1.4, measured with the skeptic's decks (`sp2e/pyms/SK4/c_sk4*.py`) and this
round's (`sp2e/pyms/FIX5/c_fix5.py`, 69 decks). The device shell
(xyce_device_gen.py) and N_DEV_PyMS.C change: libxyce.so is relinked.

- Analysis-specific code (the user's rule; design in C16). The parser keeps
  an event's analysis list (parser.py `_parse_event`: `node.event`); the
  emitter builds one shadow of the eval code per analysis context whose
  event behaviour differs (ginac_emitter.py `GiNaCEmitter.emit`,
  `_analyse_events`, `_walk_event`) and the device shell passes the analysis
  and its phase (VaeState an, phase; held state Hr / Hw / Hl). A bare .op
  runs neither initial_step("ac") nor ("tran"): is4_ac_only_op /
  is4_tran_only_op 0.2 -> 0.5; a plain initial_step runs once at the start
  of each analysis and holds its values: is4_vread_tran 0.5 sin -> 1/3 sin;
  `$abstime` (B14) works there and anywhere: is4_abstime_tran no math
  library -> 0.5 sin; @(final_step) and @(cross) bodies no longer run at
  every evaluation: is4_final_step_op, is4_cross_op 0.2 -> 0.5;
  initial_step("static" | "dc" | "noise" | lists), analysis(), above() and
  cross() events (fx5_is_*, fx5_analysis_*, fx5_cross_*, fx5_above_*), ac
  with initial_step("ac") (0.2) and the operating point's regime held for
  ac (`if (V(p, n) > 0) 1k else 4k` at 0 V: 0.5 -> 0.2) (test_analysis_shadows).
  `@(initial_model)`, `@(initial_instance)` and `@(noise)` blocks of the CMC
  models run where they stand, as before; timer and other events are a
  file:line error.
- A11 fixed (the stale gated value, all versions): x5_stale_if_cond,
  x5_real_stale_if 0.2 -> 0.5, x5_two_vdep_loops_seq, l4_nested_vdep_inner
  0.333333333 -> 0.5 (test_stale_values).
- C14 fixed: a loop whose condition only the run time knows is a C++ loop
  of the eval code, exact: l4_vdep_counter_20 0.444444444 -> 0.5,
  l4_vdep_unbounded_17 0.666666666 -> 0.5, 100 passes, contributions in
  the body (accumulators), nested run-time loops; past 100000 passes a hard
  error naming the loop; a body with ddt(), a voltage contribution or an
  event is unrolled 16 times and an evaluation needing more is a hard error
  naming the loop (test_loops).
- Analog functions: block-locals scoped in the function path
  (s4_func_two_blocks_types 0.518867924 -> 0.5), function bodies typed before
  their ternaries are lowered (x4_func_int_tern_div 0.428632662 -> 0.5,
  x4_func_real_tern_div 0.461538461 -> 0.5), a recursion (direct or mutual) a
  ParseError naming the function's file:line (test_function_scope_type).
- Block-local declaration assignments are given once (static; the LRM
  makes them constant expressions): s4_loop_body_local_init 0.5 -> 2/3, =
  VACASK (test_decl_init_static). A non-constant one stays where it is
  written. [Corrected in 1.1.5: inside an analog function a local or an
  argument that shadows a parameter was taken for the constant parameter
  and the declaration was hoisted above its assignment: A15.]
- An integer / or % by a zero known only at run time warns, once per site,
  with the statement's file:line ("an integer division by zero at run time
  (Verilog-A leaves it undefined); PyMS gives 0"); the value stays 0
  (i4_div_runtime_zero 0.5, now with the warning; test_runtime_div_zero).
- The determinism guard: the re-execution's marker is the process id, so
  a PYMS_GINAC_NOASLR the environment brings is not taken for one (preset:
  the eval code the same in 6 of 6 runs); when personality() fails or the
  system ignores the flag the metaprogram prints "PyMS warning: ... term
  order ..." and build_vae_so.py passes it on (test_ginac_determinism).
- A file with several modules is several devices (N_DEV_PyMS.C registers
  every "Module:" line; xyce_device_gen.py writes every module's shell, a
  further module as a Y device of its own name, build_vae_so.py --module):
  g4_two_modules_ground_leak loud -> 0.5 (test_multi_module).
- B20 fixed: parameter lists (`parameter real ra = 500 from (0:inf), rb =
  250 from [0:inf);`), value sets, untyped parameters (test_param_list).
  [Corrected in 1.1.5: an untyped parameter was typed by an integer-literal
  regex, not by its default expression: B20.]
- Performance: the metaprogram no longer prints the analytic vae_jacobian,
  which build_vae_so.py's difference-quotient wrapper never called (eval.cpp
  8-15 times smaller: L_UTSOI 102 1150 -> 136 KB, its g++ 28 -> 2 s; the
  metaprogram's run 5 s -> 0.02 s), and it is compiled at -O0 (it runs once,
  for milliseconds: L_UTSOI 102 158 s at -O2 -> 22 s, the same eval code
  byte for byte); a run-time loop is one body, not 16 copies. Corpus
  build-and-run times (one fresh Xyce per file): HiCUM L2 1.1.2 34.9 s,
  1.1.3 98.8 s, 1.1.4 19.2 s; MVSG 1.1.3 65.5 s -> 33.2 s; L_UTSOI 102
  133.8 s -> 40.4 s; BSIM6.1.1 77.4 -> 43.2 s; the 111 files 4386 s -> 2713 s.

Corpus (the 111 files of the PYMS3 scan, one fresh Xyce each): 1.1.3 101 generate
/ 86 build / 76 run -> 1.1.4 101 / 86 / 82: Mextram bjt504.va, bjt504t.va and
bjtd504t.va (utils/ADMS and share copies) now converge at the corpus deck's
bias (B18; with the regime held and the exact loops; not traced further). No
file changed class for the worse; the values of the 76 that ran on 1.1.3 are
unchanged but at noise level (BSIM6.1.1 V(out) 1.37e-48 -> 0, L_UTSOI
-3.3e-55 -> 0, BSIM-CMG 107/108 I(V1) to 4.3e-9 relative, bjtd504 1.2e-8).
On bias, against VACASK (openvaf-r, the `$limit` calls replaced): bjtd504
Vb = 0.95 V I(V1) -5.92562707e-3 A vs -5.92562724e-3 (1.1.3: -5.92273247e-3,
0.05 % off), Vb = 0.85 V 6.6e-6 relative; fbh_hbt 2.1 0.13 % (as 1.1.3);
MVSG exact to the printed digits.

Open after 1.1.4 (the sections below): silent -- A1, A2 (a decision), A3, A4,
A5, A6, A7, A10; loud -- B1-B5, B6 (fbhhbt-2.1.va with noise), B7, B8,
B10-B13, B15, B17-B19 (B18 deck not re-run); C2-C7, C9, C12, C13, C17 (Xyce and a jump of the
solution); D1, D2, D6; E1-E4; F1, F6, F8 (upstream); G (vamos).

### 1.1.3 (2026-10-26)

`__version__` and `VERSION` say 1.1.3. Fixed in this round (PYMS4-FIX, from
the PYMS3 skeptic's findings and the A8/A9 entries), each with a test in
`vae/tests/` that fails on 1.1.2 and passes on 1.1.3; values on the 1 V / 1k
divider (X1 between in and out, 1k from out to ground; analytic V(out) =
1000 / (R + 1000)), 1.1.2 -> 1.1.3, measured with the skeptic's decks
(`sp2e/pyms/SK3/c_sk3*.py`) and this round's (`sp2e/pyms/FIX4/c_fix4.py`,
109 decks: 18 right on 1.1.2, 109 on 1.1.3):

- `ground` declarations. The 1.1.2 entry below said `ground` declarations
  "are recorded (`Module.grounds`)": true, and misleading -- they were never
  applied, the net stayed an ordinary floating node. `electrical g; ground
  g;` then `I(n, g) <+ V(n, g)/1k` (g_elec_contrib): 0.5 -> 0.333333333;
  `V(n, g) <+ 0.25`: 0.5 -> 0.25; a probe `I(a, g)` of a short to g
  (g_elec_probe_pair): 5e-10 -> 0.5; a short `V(a, g) <+ 0`
  (gnd_decl_short_elec, g_elec_short(_rev); REGRESSION of 1.1.1, 1.1.0 had
  1/3): 0.5 -> 0.333333333; `ground g;` without a discipline declaration
  (gnd_decl_contrib, pr_gnd_decl_*) had no math library (`V_g` undeclared)
  -> 1/3 and 0.5. Now parser.py `Parser._apply_grounds` rewrites every
  reference to a declared ground into the implicit-ground form the rest of
  PyMS handles (`V(a, g)` -> `V(a)`, `V(g, a)` -> `(-V(a))`, `V(g)` -> 0, the
  probes the same, `I(g, a) <+ e` -> `I(a) <+ -(e)`, a branch `(g, a) b` ->
  `(a) b` with its probes and contributions negated; `V(a, g) <+ 0` stays a
  short, now to ground) and drops the net from the internal nodes, so the
  emitter, `_short_classes`, the probe matching, check_declarations and the
  device shell see one consistent form. `ground electrical g;` is accepted;
  a ground port is refused (test_ground_decl).
- A net merely NAMED gnd: the emitter's ground was the string 'gnd', so an
  internal node or a port called gnd was the reference node:
  g_named_gnd_internal_short (`electrical gnd, x; V(x, gnd) <+ 0; I(gnd, n)
  <+ ...`) 0 -> 0.333333333; a port gnd shorted to an internal node
  (gn_port_short, 0.5 on 1.1.0) 0 -> 0.5. The emitter's ground is now `GND =
  '<ground>'`, no Verilog-A name (ginac_emitter.py `_branch_endpoints`,
  `_finalize_current_rows`, `_short_classes`, `_resolve_shorted_node`). The
  only other code with a literal 'gnd' is vae/codegen.py, which no Xyce path
  imports (C12).
- casex / casez were not statements: their arms were dropped by the
  expression-statement fallback (cs3_casex: V(out) = 0, rc 0; openvaf
  refuses). They parse as case now: an x / z (casez: z / ?) digit of a
  based literal is a don't-care bit and an analog value has no x or z bits,
  so an item without such digits compares as in case -- what PyMS builds
  (cs3_casex 0 -> 0.5) -- and an item with them is a file:line ParseError
  (parser.py `_parse_case`, `_check_casexz_operand`; test_casexz). A based
  literal in any analog expression (`'h5`, `4'b0101`) reached the C++
  verbatim (no math library); it is its decimal value now (int_ops.py).
- Block-local variables merged into ONE module variable of the first
  declaration's type: ud3_shadow_block_local 0.25 -> 0.5, ud3_shadow_type 0.8
  -> 0.5, ud3_two_blocks_types_rev 0.535714285 -> 0.5 (VACASK 0.5 for all
  three). The parser keeps a scope per begin...end; a block-local whose name
  is taken (a module variable, parameter, node, an enclosing or an earlier
  block's local) is a variable of its own under a unique internal name
  (`x__blk1`) with its own type, and the names inside the block are renamed
  to it; a fresh name keeps its name (parser.py `_parse_block_variables`,
  `_scoped`; test_block_scope). 19 of the ADMS example files have such names (PSP103 160,
  VBIC 1.3 65, BSIM-SOI 59-63, BSIM-CMG 48-49, ...); none changed class.
- Analog functions were SKIPPED by the parser: a call reached the C++ as an
  undeclared function (no math library: ud3_afunc_local,
  ud3_afunc_output_arg, cs3_in_afunc; fbh_hbt 2.1/2.3, MVSG: B6), and
  check_declarations never saw a body (ud3_bad_in_afunc, `zz = a;` in a
  body, passed generation). Now `_parse_analog_function` parses the body
  (arguments with directions, locals, the statement), check_declarations
  walks it (an assignment target must be an output / inout argument, a
  local or the function name; an input is read-only; no node access), and
  `_FunctionInliner` turns every call into, before its statement, the
  inputs assigned to fresh variables, a renamed copy of the body and the
  outputs assigned back (nested calls, integer functions, calls in
  conditions; a wrong argument count is an error). A function named like a
  PyMS builtin (lexp, lln, hypsmooth, hypmax, Tempdep, ...) keeps the builtin
  as before (C13). af_* decks: build failure -> 0.5 (test_analog_functions).
  [CORRECTED in 1.1.4: this overclaimed. The function path of
  `_parse_block_variables` did not scope block-locals (two blocks with
  `real t` and `integer t` were one real: s4_func_two_blocks_types
  0.518867924 for 0.5), a body's ternaries were lowered before int_ops typed
  it (`((a > 0) ? 7 : 9) / 2` a real division: x4_func_int_tern_div
  0.428632662, x4_func_real_tern_div 0.461538461), and a recursive function
  reached g++ as an undeclared function. Fixed in 1.1.4 (test_function_scope_type).]
- The case expression was copied into each item compare, so a side effect
  in it ran once per item. A case expression with a call that is not a pure
  builtin is now stored in a temporary first, so it is evaluated once
  (parser.py `_parse_case`, `_has_impure_call`): `case (bump(k))`, bump an
  integer function incrementing its inout argument, three items and a
  default: k = 1, 0.5 (on 1.1.2 no math library: functions; per item it would
  be k = 3, 0.25) (test_analog_functions case_once, case_once_match).
- `$simparam("name", default)` with nested parentheses in the default
  (smp4_nested_default, `max(1.0, min(2.0, 3.0))`): no math library -> 0.5.
  A balanced scan replaces the one-level regular expression
  (ginac_emitter.py `_sub_simparam`); `_try_eval_expr` now evaluates the
  substituted text, so a `$simparam` expression folds to a constant
  (test_simparam_balanced).
- A8 fixed: integer `/` and `%`, and the rounding of a real assigned to an
  integer variable. iv_module_int_div / iv_param_int_div / iv_block_int_div
  0.4 -> 0.5; `-7 % 3` = -1 (ia_neg_mod -0.999999998 -> 0.5); `k = 2.5` is 3
  (ia_round_assign 0.545454545 -> 0.5), also for a voltage-dependent value
  (ia_round_vdep 0.415654642 -> 0.5). int_ops.py types every analog-block
  and function-body expression with the declared types and rewrites an
  integer `a / b` to `_vae_idiv(a, b)`, `a % b` to `_vae_imod(a, b)`, and
  marks a rounding assignment (`ASTNode.int_lhs`, `_vae_rint`); the emitter
  evaluates them in Python at elaboration, as GiNaC functions in the
  metaprogram (piecewise constant, derivative 0) and as C++ helpers in the
  eval code; an integer division by a constant 0 at elaboration is an
  EmitError (test_integer_arith). Corpus: BSIM6.1.1's `NF % 2` (l.2596-2609)
  is now an integer modulus.
- A9 fixed: loops iterate. lp_for3 0.25 -> 0.5, lp_for0 0.666666666 -> 0.5,
  lp_while 0.333333333 -> 0.5; `repeat` was no statement (lp_repeat 0 ->
  0.5). A loop is a BLOCK [init, body, step] with `node.loop` (parser.py
  `_parse_for`, `_parse_while`, `_parse_repeat`); GiNaCEmitter `_walk_loop`
  unrolls it exactly when the condition is known at elaboration (a
  zero-trip loop runs not at all; the NF loops of BSIM6 / BSIM-SOI / PSP /
  L-UTSOI), and under a chain of runtime indicators when the condition reads
  the node voltages (the first indicator includes the enclosing gates): up
  to the bound of a counter conjunct (`k < N`; the counter's possible start
  values are tracked through gated assignments), at most 16 iterations. A
  loop that may need more (no bound, or a larger one) gets a guard that
  prints `PyMS warning: <file>:<line>: this while loop needs more than the
  16 iterations PyMS unrolls ...` the first time an evaluation would go on;
  that evaluation's values are those after 16 (C14). Unrolling all 101
  iterations of HiCUM L2's `l_it <= l_itmax` Q_pT loop made a 2 MB
  metaprogram that took 1607 s and 6.7 GB to compile at -O2. A `repeat`
  count unknown at elaboration is an error (test_loops).
- Loud on legal, accepted now: ANSI port declarations `module a1(inout
  electrical p, inout electrical n)` (ud3_ansi_ports: g++ "redefinition of
  nodeID_inout" -> 0.5), `inout electrical p;` (ud3_ansi_ports_inout: a
  parse error -> 0.5), analog functions with begin/end bodies / locals /
  case (above), and named-block locals in `@(initial_step)`
  (ud3_init_step_local -> 0.5): the emitter SKIPPED initial_step blocks, so
  their variables read 0; they run where they stand now, at every evaluation
  (C15) (test_port_and_initial_step).
- D3 was not harmless (the PYMS3 skeptic: Mextram bjtd504 converged in 2
  runs of 4 on 1.1.2; here measured again here: 3 of 4 fresh runs). GiNaC orders the terms of a sum or
  product by hash values seeded from the address of each class's type name
  (GiNaC hash_seed.h), so with address-space randomization one compiled
  metaprogram printed its eval code in another term order in every run:
  two runs of one bjtd504 metaprogram printed different eval.cpp (md5
  2e655281 / a411383b), and under `setarch -R` identical ones across runs and
  across two separately compiled binaries (5553f516). The metaprogram now
  re-executes itself once with ADDR_NO_RANDOMIZE (personality), so the
  eval code is the same in every run (test_ginac_determinism); bjtd504:
  4 of 4 fresh runs on 1.1.3, the same V(out) 1.00029009e-10.

Corpus (the 111 files of the PYMS3 scan, one fresh Xyce each): 1.1.2 101 generate / 72 build / 62 run (bjtd504 ran in that scan) -> 1.1.3 101 / 86 / 76.
No file changed class for the worse; fbh_hbt 2.1 (4 of 5 variants) and 2.3,
MVSG and (no longer flaky) bjtd504 now run; HiCUM L2's corpus deck takes 105 s
(36 s on 1.1.2: 16 unrolled Q_pT iterations).

New in this file: A10 (a probe in an if condition, silent, older than 1.1.0), C12 (codegen.py), C13 (builtin-named functions), C14 (the
loop guard), C15 (initial_step at every evaluation), B20 (parameter lists).

Open after 1.1.3 (the sections below): silent -- A1, A2 (a decision), A3,
A4, A5, A6, A7, A10; loud -- B1-B5, B6 (fbhhbt-2.1.va with noise still does not
build), B7, B8, B10-B15, B17-B20; C2-C7, C9, C12-C15; D1, D2, D6; E1-E4;
F1, F6, F8 (upstream); G (vamos).

### 1.1.2 (2026-10-26)

`__version__` and `VERSION` say 1.1.2. Fixed in this round (PYMS3-FIX, from
the PYMS2 skeptic's findings), each with a test in `vae/tests/` that fails
on 1.1.1; values on the 1 V / 1k divider unless said otherwise:

- REGRESSION of 1.1.1, the ammeter: `_short_classes` collapsed an
  unconditional short `V(br) <+ 0` even when its current was read, so the
  probe sat on a degenerate branch and read 0. A short whose current is
  probed anywhere (`I(br)`, or `I(a, b)` on its node pair in either order,
  in a contribution, an assignment or a condition) never collapses; it stays
  a 0 V source on its branch-current unknown (ginac_emitter.py
  `_short_classes`; test_node_collapse). c_ammeter_double / am_double 0.5 ->
  0.666666667 (analytic 2/3; 1.1.0 0.666666667), am_int 0.5 -> 0.666666666,
  am_gnd 0.333333333 -> 0.5, am_assign 0.5 -> 0.666666667, am_scaled 0.5 ->
  0.6; am_unnamed (`V(p, x) <+ 0; ... I(p, n) <+ I(p, x)`, wrong on 1.1.0
  too) 0.5 -> 0.666666667. An unprobed short still collapses.
- case statements (pristine to 1.1.1): `_parse_case` made each item an
  independent `if` and put the default arm's statements in the enclosing
  block unconditionally, so the default ran beside the arm that matched, and
  an item of more than one token (`-1`, `1 + 1`) became one item per token.
  Now the case is the equivalent if / else-if / else chain, as the LRM
  defines it: items compared in source order, the first match runs and no
  other, the default (anywhere, `:` optional) when none matches, several
  items per arm, an item any expression, one statement or the null
  statement per arm; two defaults are an error (parser.py `_parse_case`;
  test_case_statement). `case (mode) 0: I <+ V/1k; default: I <+ V/2k;`
  mode 0: 0.6 -> 0.5; nc_case0 and cs_short_block (a short in arm 0) 0.333 ->
  0.5; cs_vsrc (`0: V(p, n) <+ 0.2;`) 0.5 -> 0.8; cs_short_ground (`0: V(x)
  <+ 0;`) 0.4 -> 0.333333333. BSIM6.1.1's `BSIM6RdsEndIso`/`Sha` macros (a
  `default: Rend = 0.0;` arm) set Rend = 0 whatever rgeo before.
- Undeclared names were accepted, rc 0: `I(p, q) <+ ...` with q undeclared,
  `I(bq) <+ ...`, `zz = 1;`. Now a file:line error before any C++: every
  contribution target and V()/I() argument must be a port, node, ground or
  named branch, every assignment target a declared variable or a parameter
  (C11) (ginac_emitter.py `check_declarations`, called by xyce_device_gen.py;
  test_undeclared_names).
  So that no legal file is refused, the parser keeps what it dropped before:
  `real`/`integer` declarations inside a (named) block become variables of
  the module (parser.py `_parse_block_variables`) and `ground` declarations
  are recorded (`Module.grounds`). The corpus generation scan refuses no file
  that 1.1.1 generated. (Corrected in 1.1.3: "recorded" meant only that --
  the ground net was never grounded and the block-locals all merged into one
  variable; see CHANGES 1.1.3.)
- The device-shell cache reused a shell generated by another PyMS (1.1.0 then
  1.1.1 through one `PYMS_CACHE`: 0 for 0.5, rc 0). N_DEV_PyMS.C (the Xyce
  library, relinked) now keys the slot `pyms_<module>_<hash>.so` on the .va
  identity AND the generated shell source (.C and .h, which carry the PyMS
  version in the math-library key); the generator's output is deterministic,
  so an unchanged setup still reuses its slot (test_hdl_cache_pyms_version).
- `$simparam` in a condition reached g++ ("'$simparam' was not declared"):
  conditions now take the simulator's value / the default like expressions
  (ginac_emitter.py `_preprocess_sys_funcs`; test_simparam_expr). smp_cond
  rc 1 -> 0.5.
- A string compare in a parameter default (`parameter real r = (mode ==
  "slow") ? 2000 : 1000;`) was refused: `==`/`!=` on two strings in
  va_const.py `evaluate`, and in `to_cxx` over the shell's std::string
  members for the per-instance re-evaluation (a card `mode="slow"`). Two
  string parameters compared in a condition (`mode == kind`, right on 1.1.0,
  a library build failure on 1.1.1) fold like a literal compare
  (`_fold_string_compares`; test_string_params).

Investigated, not a defect: BSIM-SOI 4.7's V(out) = gmin x V(in) (section
C10). Corrected in this file: F6 and `upstream/vacask/c_ammeter_double/README`
(they said 1.1.1 was right on the ammeter), C8 (the singular matrix was the
pristine tree, not 1.1.0), B10 (its mechanism is unknown). New: F8 (Mextram
against VACASK, 0.03-0.125 %). Found in this round and filed, not fixed:
A8 (integer division in analog code is real division) and A9 (loop bodies
run once).

Open after 1.1.2 (the sections below): silent -- A1, A2 (a decision), A3,
A4, A5, A6, A7, A8, A9; loud -- B1-B8, B10-B15, B17-B19; C2-C7, C9; D1-D3,
D6; E1-E4; F1, F6, F8 (upstream); G (vamos). Nothing of the PYMS2
skeptic's list is left open except what it called notes (B10's
mechanism, F8).

### 1.1.1 (2026-10-25)

`__version__` and `VERSION` say 1.1.1, and the version is now part of the
math-library cache key (`vae_<module>_<key>.so`), so no library built by
another PyMS is reused. Fixed in this round (PYMS2-FIX), each with a test in
`vae/tests/` that failed on 1.1.0:

- Node collapse kept no node: a short `V(x, y) <+ 0` between two internal
  nodes removed both (ginac_emitter.py `_find_shorted_nodes` added every
  internal end) and every contribution at them went to ground; a short in
  one arm of a parameter condition collapsed in the shell (decided from the
  defaults) and per instance in the math library, which then disagreed on
  the node list (the old D4). Now only an unconditional short collapses,
  each class of shorts keeps one survivor (a port, else ground, else the
  first internal node declared: `_short_classes`), and a conditional short
  is a 0 V source on its branch-current unknown. col_flag0 0 -> 0.571428571
  (analytic 0.5714285714), col_chain 0 -> 0.5, the Mextram c3/c4/c1 shape
  0 -> 0.498753117 (analytic 0.4987531172), a card that flips the condition
  right both ways (test_node_collapse). Mextram 504 (bjt504, bjtd504,
  bjtd504t) went from Ic = Vce/12.16
  ohm whatever Vb (-0.1233 A at Vce = 1.5 V) to the transistor: Ic at Vce
  = 1.5 V, Vb = 0.6/0.7/0.8/0.9 V = 2.8295e-7 / 1.27864e-5 / 5.15403e-4 /
  6.13165e-3 A against VACASK (the `$limit` calls replaced by their
  argument) 2.8330e-7 / 1.28011e-5 / 5.15879e-4 / 6.13341e-3 A (0.03-0.12
  %); Ib 0.003-0.09 %. bjt504t needs its dt port connected (B17).
- An unrelated sibling lent its model group: `scan_va_attributes` gave a
  module without xyce* attributes those of ANY `.va` in its directory that
  has them. toys/rlc.va (and rlc2, rlc3, isrc, vsrc) became Capacitor level
  6 and ran as the simple R/L/C stamp of one capacitor (10 MHz: |out| 0.0627
  at +86.4 deg; the series RLC is 0.0213 at -87.6 deg); psp103_nqs.va became
  a JUNCAP diode, bjtd504/bjtd504t 4-terminal BJT level 504. Now only a
  variant of the same model inherits (one file name extends the other:
  psp102b.va from psp102.va, the fbhhbt-2.1 variants). The rlc of the toys
  is a Y device and = the analytic RLC (test_model_group_route).
- A math library that does not build is a device error on every route; the
  simple R/L/C stamp (Resistor/Capacitor/Inductor groups) ran rc 0 without
  it (toys/capacitor x2, sg13g2 r3_cmc). toys/capacitor.va's own failure
  (old B9, `double Q` shadowed the eval's charge vector) is gone: the eval's
  arguments are `_vae_s`, `_vae_F`, `_vae_Q` (test_model_group_route).
- Value ranges: `from` (open, closed, half-open, inf, a range that reads
  another parameter, a string parameter's `'{...}` set) and `exclude`
  (ranges and single values) are checked per instance on the final values,
  defaults included; a violation is one line naming the parameter, the
  value and the range (va_const.py `check_param_ranges`, build_vae_so.py;
  test_param_ranges).
- String parameters: defaults, cards (`.model m1 s1 mode=slow`, a
  std::string member in the shell) and the compares `mode == "fast"`,
  `"slow" != mode` in conditions and ternaries, folded at elaboration
  (ginac_emitter.py `_fold_string_compares`, `_cond_ns`); `if (mode == ...)`
  was silently false. Any other use of one in a condition is an error
  (test_string_params).
- `$simparam` in analog code takes the simulator's value (gmin, tnom from
  the shell), then the default, else an error; `$simparam("gmin")` was a
  silent 0 (old B10; test_simparam_expr).
- Mid-line conditional directives (`` r = `ifdef BIG 2k `else 1k `endif; ``):
  split at the directives, the kept arms one output line (the
  line map unchanged); an undefined macro in an inactive arm is no error
  (preprocess.py `_split_conditionals`; test_preprocess_midline).
- One-line file:line errors for a port current probe `I(<p>)` (was a
  Python traceback), `ddt()` in a condition (was g++ "'ddt' was not
  declared"), an undeclared identifier (was g++'s page): xyce_device_gen.py
  main, ginac_emitter.py `check_conditions`, build_vae_so.py
  (test_clear_errors).

Not fixed, filed below: B10 (a 1e12 S conductance, the DC op), B17
(bjt504t with dt left open: SIGSEGV), B18 (Mextram 504 switched off: the DC
op), B19 (sg13g2 r3_cmc: a GiNaC pole), C8 (the E64 toy diode: not a PyMS
defect), C9 (gmin on the nodes a conditional short keeps), D6, F6
(`upstream/vacask/c_ammeter_double/`).

Corpus (the same 111 files and divider decks as 1.1.0's, 2026-10-25):
generates 101 -> 101, builds 70 -> 72, runs 69 -> 61, ran without a math
library 3 -> 0. Changed class: toys/capacitor x2 ran-dead -> ran (B9);
sg13g2 r3_cmc ran-dead -> a loud build error (B19); Mextram x7 ran (Ic =
0.0822 A, wrong) -> a loud DC non-convergence of the off transistor of that
deck (B18; ADMS bjtd504 runs, 5.1e-12 A). Values: toys/rlc3 (a parallel
RLC: a short at dc) 0 -> 1.0, right; the other changes are of the gmin
order, i(V1) 1e-12 -> 2e-12 ... 4e-12 A (BSIM-CMG 111, BSIM-SOI x3, L-UTSOI,
PSP103 x2, toys rlc/rlc2: C9).

### 1.1.0 (2026-10-25)

`vae/__init__.py` `__version__` and `VERSION` say 1.1.0;
`xyce_device_gen.py --version` prints it. Fixed in this round (each with a
test in `vae/tests/` that failed before):

- Parameter values from a `.model` card or an instance line reach parameters
  whose names are not all capitals (44 % of real-world names were affected);
  only given values are baked; a math library that does not build is a device
  error, no longer a device that silently conducts nothing (on the Y route
  and the compact-model groups only: the simple R/L/C stamp of the
  Resistor/Capacitor/Inductor groups stayed silent until 1.1.1); two parameters
  that differ only in case are an error (test_hdl_param_name_case,
  test_hdl_model_card_params).
- Parameter defaults: an LRM constant-expression evaluator (`vae/va_const.py`:
  int/real typing, truncating integer division, `%`, `**`, `?:`, SI scales,
  based literals, `$simparam`, dependent defaults per instance); unevaluable
  defaults are a file:line error (test_param_defaults).
- Attributes `(* *)` at statement starts and before terminators; an undefined
  macro is a file:line error (test_attributes_and_undefined_macros).
- Preprocessor: object-like `define` with a blank before `(`, string-aware
  comment stripping that keeps comment state across lines (BSIM-SOI 4.x had
  silently lost 15 charge statements to `*//*`), macros expanded at their
  point of use, shipped `disciplines.vams`/`constants.vams` (Accellera 2.4.0
  text, `P_K` 1.3806503e-23) (test_preprocess_define, test_std_headers,
  test_preprocess_point_of_use).
- `ddt`: charge terms split from the static terms (Q and F) exactly;
  unsupported forms refused with a clear error; `pow(2, -1.0/0.5)` and other
  integer-literal calls emitted as reals; GiNaC program emission deterministic
  (test_ddt_split, test_emit_numeric_and_determinism).
- Branch currents as MNA unknowns: `I(br)` / `I(p, n)` reads, `V(p, n) <+ r *
  I(p, n)`, voltage contributions that read their own branch (the 0.2 % P7
  offset) exact (test_current_probes).
- A source line map: every parser and emitter error names the file and line
  of the original source, includes included; SI scale factors inside analog
  expressions (`r / 1k`); internal-node gmin from the device options instead
  of a hard-coded 1e-9; `xyceAlsoModel` instance parameters are also model
  parameters (test_source_line_map, test_expr_scale_factors,
  test_device_gmin, test_also_model_params).
- Cache race: the math library builds in a private directory and is renamed
  into place; with the matching Xyce library (N_DEV_PyMS.C) the device shell
  is generated in a private directory, its slot is
  `pyms_<module>_<identity>.so`, written atomically, and the identity covers
  `PYMS_DEFINES` (test_hdl_cache_concurrent, test_define_option).
- Non-electrical disciplines (`thermal t;`, `Temp(t)`, `Pwr(t) <+`) as MNA
  nodes; command-line macros (`--define`, `-D`, `PYMS_DEFINES`); a missing
  `;` in a declaration and a port and parameter of the same name are errors;
  variable initializers (`real g = 1m;`) no longer dropped
  (test_discipline_ports, test_define_option, test_version,
  test_declaration_syntax).

- A variable reassigned from its own constant value with a non-constant
  right-hand side (`x = a; x = x + tc * ($temperature - 300.15);`, or any
  voltage) read the old value as 0 (or as an uninitialized C++ variable,
  `double y = y;`): vae/ginac_emitter.py `_walk_statement` dropped the known
  value before translating the right-hand side. Silent in PSP102 (`beta =
  0 * tf_bet`, the temperature-scaled mobility and threshold terms), BSIM6,
  DIODE_CMC (10 -> 0) and BSIM-SOI 4.x (13 terms; the junction potential
  PhiBSWG = 0 gave a NaN charge and no DC convergence: the BSIM-SOI
  "regression" of the corpus). With the fix BSIM-SOI 4.5 matches ngspice's
  B4SOI to the gmin current. The emitted GiNaC programs of the 98 corpus
  entry points: 60 identical, 12 refused by both trees (unchanged), 10
  value-equal (`0` -> `0.0`: EKV, MVSG, PSP103, PSP103T, VBIC 1.3), 16 with
  value changes (BSIM6, BSIM-SOI x3, DIODE_CMC, PSP102 x3, each in both
  trees), every one of the form `0 * f` / `0 + f` -> `old * f` / `old + f`
  (test_self_reference_assign).

Corpus (the 54 ADMS examples + 57 installed `verilog-a` files, a 1 V / 1k
divider .op per module, PYMS1-I): generates 103 -> 101, builds 53 -> 70, runs
83 -> 53. "Runs" fell because the pristine tree "ran" devices whose math
library had not built (they conducted nothing, exit 0); those are now loud.
Newly right: EKV, L_UTSOI, HICUM L2/L0, VBIC 1.3, sg13g2 cap. (Corrected by
the PYMS1 skeptic: Mextram x4 ran silently wrong, Ic = Vce/12.16 ohm whatever
Vb, and toys/rlc on the model-group route had lost its inductor; rlc2/vsrc
ran the same sibling-group capacitor stamp. All fixed in 1.1.1.)

---------------------------------------------------------------------------

## A. Silent wrong answers

### A1. The caches do not see a change to an `include file

Module `d2.va`, its value in `d2_value.vams`:

    `include "disciplines.vams"
    module d2(p, n);
      inout p, n;
      electrical p, n;
    `include "d2_value.vams"
      analog I(p, n) <+ V(p, n) / r;
    endmodule

    * d2 in a 1k divider: out = 1k / (r + 1k)
    .hdl "d2.va"
    V1 in 0 1
    .model m1 d2
    YD2 X1 in out m1
    R2 out 0 1k
    .op
    .print dc v(out)
    .end

Run with `parameter real r = 1k;` in `d2_value.vams`, then edit it to
`r = 3k` and run again in the same caches. MEASURED (2026-10-25, the
relinked Xyce of this round): run a (1k) 0.5; run b (include edited to 3k,
warm caches) 0.5, SILENTLY STALE; run c (same, caches cleared) 0.25 (right).
VACASK recompiles from source on every run (0.25).

Mechanism: both caches key on the ENTRY file only. N_DEV_PyMS.C
`pyms_register_hdl` hashes `va_path + content (+ PYMS_DEFINES)` for the shell
slot; the shell's `processParams` (emitted by xyce_device_gen.py
`generate_device_cpp`, the `_key = std::hash(_va | _p | _vc | ABI)` line)
hashes the entry file's bytes for the math library `vae_<module>_<key>.so`.
An `include changes neither. CMC models keep their equations in includes
(BSIM-CMG `bsimcmg_body.include`, PSP `*.include`), so an edited model
silently runs the old code until the caches are cleared.

Fix: let the preprocessor report the files it read (preprocess.py already
builds `PreprocessedText.line_map`, which names every source file), have
xyce_device_gen.py write them, with their content hashes, into a sidecar
`<slot>.deps` beside the shell, and key both caches on the hash of all of
them. In C++ (`pyms_register_hdl`): reuse a slot only when every file in its
`.deps` still has the recorded hash; the shell's math-library key hashes the
same list (the generated shell can embed the list and re-hash at run time).

Test: `vae/tests/test_hdl_cache_include_identity.py`: the deck above, three
runs in one cache (1k: 0.5; include edited to 3k: 0.25; back to 1k: 0.5).

### A2. A flow probe on a branch with no contribution is a short on Xyce, open on VACASK

    module fp(p, n, o);
      inout p, n, o; electrical p, n, o;
      branch (p, n) b;
      analog I(o) <+ -I(b);      // b has no contribution of its own
    endmodule

(PYMS1-E) Xyce 0.50025 (the branch acts as a 0 V source: a short), VACASK 0
(open). Mechanism: vae/ginac_emitter.py `collect_current_unknowns` gives every
probed branch an MNA current unknown and `_finalize_current_rows` writes the
row V(p) - V(n) = 0 when the branch has no contribution. Decide by the LRM
(Verilog-AMS 2.4, the section on branch contributions and probes): the
common reading is that a branch whose flow is probed and which has no
contribution is a flow probe of zero potential (an ammeter), which is what
Xyce does; if that reading holds, VACASK is the outlier (section F) and PyMS
needs only a test that pins the short; if not, the row must be I = 0. Test: the module
above between a 1 V source and 1k: `v(o)` as the LRM says.

### A3. A nonzero voltage contribution on a same-node branch is accepted silently

    analog V(p, p) <+ 1.0;

(PYMS1-I) Since I's fix 1b a same-node branch's current is pinned to 0
(`li_gnd` from the output map, xyce_device_gen.py), which is right for `V(p,
p) <+ 0` and for the probes the BSIM-CMG 111 / BSIM6 sources write, but a
nonzero potential across one node is a contradiction and is dropped without a
word. Fix: in `_finalize_current_rows`, when both ends of a V-contribution
branch are the same node and the contribution is not identically 0, raise
EmitError with the statement's file:line ("V(p, p) <+ 1.0: a potential
across a single node must be 0"). Test: that module is a clear error; `V(p,
p) <+ 0` still builds.

### A4. `ddt` under a voltage-dependent condition is accepted

    analog if (V(p, n) > 0.5) I(p, n) <+ ddt(1p * V(p, n));

(PYMS1-D) Accepted and stamped as if the condition were evaluated once;
openvaf-r refuses it, as the LRM's analog-operator rules say (an analog
operator shall not be used in a conditionally executed statement whose
condition depends on the simulation state). Mechanism: vae/ginac_emitter.py
`_vae_split_ddt` checks the factor of `ddt` for voltage dependence but not the
enclosing conditions. Fix: carry the stack of enclosing conditions into the
split and refuse when any depends on a probe (`EmitError` with file:line).
Test: in vae/tests/test_ddt_split.py, the module above is a clear error.

### A5. A model parameter whose default reads an instance parameter keeps the all-defaults value in the shell

    parameter real w = 1u from (0:inf) (* type="instance" *);
    parameter real cap = 1n * w / 1u;          // model parameter, default reads w

(PYMS1-A) The math library evaluates `cap` per instance with the given `w`
(right), but the shell's Model member `cap` (what `.op` lists and what any
shell-side use reads) keeps the value computed from all defaults. Mechanism:
xyce_device_gen.py `generate_device_cpp` emits `Model::processParams`
re-evaluation of dependent defaults with model parameters only. Fix: either
evaluate such defaults in `Instance::processParams` into an instance-side
copy, or list the parameter as "per instance" in the shell and stop exposing
the model member. Test: instance `w=2u`: the device current uses cap = 2n
(already right) and the shell's listing shows 2n.

### A6. The nmos/pmos type variable is always baked

(PYMS1-H) The shell writes `__SET__=TYPE` whenever the module has a type
variable, so a `.model mm m level=N` card (neither nmos nor pmos) bakes the
shell's own reading of the default (0 for an expression default). Real models
default it to a plain 1 or -1, so no measured case is wrong. Fix:
`Model::Model` records whether it set the variable (a bool member) and
`processParams` writes `__SET__` only then (xyce_device_gen.py). Test: a
MOSFET-group module whose type variable defaults to an expression, on a
`level=N` card.

### A7. `.hdl "f.va" NAME=VALUE` silently ignores everything after the file name

    .hdl "d1.va" RVAL=3000

Measured (2026-10-25 reading of the code): N_IO_CircuitBlock.C, the `.HDL`
branch, reads `line[1]` only and drops the rest; it also sets `result = true`
after a failed `pyms_register_hdl` (the UserError still stops the run). The
macros reach PyMS today only through the `PYMS_DEFINES` environment variable.
Design: collect `line[2..]` (NAME or NAME=VALUE; the tokenizer splits `=`
into its own token) into a vector, pass it to `pyms_register_hdl(va_path,
defines)`, add each as a shell-quoted `--define` to the generator command
(xyce_device_gen.py already accepts `--define`), and hash them into the
identity beside `PYMS_DEFINES` (they are baked into the shell exactly like the
environment macros). Unknown tokens (not a macro name) are a netlist error.
Test: vae/tests/test_define_option.py, a Xyce run with `.hdl "d1.va"
RVAL=3000` gives 0.25, and the same deck without the token in the same cache
gives 0.5.

### A8. Integer division in analog code is real division

FIXED in 1.1.3 (CHANGES 1.1.3; test_integer_arith). The original entry:

    parameter integer k = 3;
    real t;
    analog begin
      t = k / 2;                         // LRM: 3 / 2 = 1 (integer operands)
      I(p, n) <+ V(p, n) / (1000 * t);
    end

MEASURED (2026-10-26, PyMS 1.1.1 and 1.1.2): V(out) = 0.4 on the 1 V / 1k
divider (t = 1.5); the LRM's value is 0.5 (t = 1). The same with `integer
k; k = 3;` at module level or in a named block. Only the parameter-default
evaluator (va_const.py, CHANGES 1.1.0) types integers; the analog-block
translators of ginac_emitter.py (`_to_ginac_expr`, `_to_cpp_expr`) print
every operand as a double (GiNaC numerics, C++ doubles), so `/` and `%` on
two integer operands are real operations. Fix: carry the static type of
every variable (Module.variables[].type), parameter and literal into the
translators -- va_const.static_type is the rule -- and emit integer `/`
and `%` as truncating operations (`trunc(a / b)` on the values, with the
sign of the first operand for `%`); the GiNaC side needs it only for
operands that are not voltage-dependent (an integer cannot hold a voltage
without an explicit conversion). Check the corpus: a model that relies on
real division of integer variables is itself wrong by the LRM. Test: the
module above gives 0.5; `real` t = 3.0 / 2 still 1.5.

### A9. Loop bodies run once

FIXED in 1.1.3 (CHANGES 1.1.3; test_loops; the runtime-condition limit is C14). The original entry:

    genvar i;
    analog begin
      for (i = 0; i < 3; i = i + 1) I(p, n) <+ V(p, n) / 3000;   // 3 x 3k = 1k
    end

MEASURED (2026-10-26, PyMS 1.1.2, the 1 V / 1k divider): 0.25 (one 3k),
analytic 0.5. `for (i = 0; i < 0; ...)` adding a second 1k: 0.666666666
(the body ran), analytic 0.5. `while (k < 2) begin g = g + 0.5m; k = k +
1; end` from g = 0: 0.333333333 (one pass), analytic 0.5. Mechanism:
parser.py `_parse_for` and `_parse_while` skip the header and return the
body as a plain statement ("loop unrolling not needed for symbolic"), so
it runs exactly once whatever the bounds. In the corpus BSIM-SOI 4.5/4.6.1
use `for (i = 0; i < B4SOInf; i = i+1)` (right only for nf = 1) and a
`while ((niter<=4) && ...)` iteration (one pass). Fix: a FOR/WHILE node
kind; unroll a genvar `for` whose bounds are constant at elaboration
(va_const evaluates them), and refuse with file:line a loop whose trip
count depends on the simulation state (the analog operators inside one
are illegal there anyway); a `while` over variables needs the bounds to
fold the same way or a refusal. Test: the three modules above.

### A10. A branch current probed in an if condition reads the wrong current

    electrical x;
    branch (p, x) am;
    analog begin
      V(am) <+ 0;                                    // an ammeter p -> x
      I(x, n) <+ V(x, n) / 1000;
      if (I(am) > 0.5m) I(p, n) <+ V(p, n) / 1000;   // I(am) = 0.5 mA ... 1 mA
      else I(p, n) <+ V(p, n) / 4000;
    end

MEASURED (PYMS3 skeptic deck SK3 pr_probe_in_if_cond; here 2026-10-26 on
1.1.0, 1.1.1, 1.1.2 and 1.1.3): 0.555555555; analytic 2/3 (the two 1k in
parallel: I(am) = V(p, x)/1k > 0.5 mA at the solution). Silent, rc 0, not
touched by 1.1.3. Suspect: the runtime test of the condition
(`_cond_final_cpp`, the branch-current unknown as a double of the eval)
against the select of the contribution; trace with the eval code. Test: the
module above = 2/3.

### A11. A run-time condition reads a stale gated value

FIXED in 1.1.4 (CHANGES 1.1.4; test_stale_values). Silent, in every version
from 1.1.0 to 1.1.3 (the PYMS4 skeptic, round b):

    if (V(p, n) > -1) k = 5;
    k = 0;
    if (k < 1 && V(p, n) > -1) g = 1m; else g = 0.25m;   // k is 0: 1m

gave 0.2 (g = 0.25m) for 0.5 on the 1 V / 1k divider (x5_stale_if_cond; a
real variable the same, x5_real_stale_if), and every second pass of a loop
that resets its counter read the counter's last gated symbol
(x5_two_vdep_loops_seq, l4_nested_vdep_inner 0.333 for 0.5). The CLASS: a
variable whose last assignment was a constant, after an assignment under a
voltage-dependent condition. A constant assignment outside a run-time
condition leaves the variable in `var_values` and does not drop its older
symbol from `_var_sym_name`; the condition translator `_cond_final_cpp`
(`_idsub`) looked up the symbol FIRST, the expression translators
(`_subst_known_cpp`, `_var_current_ex_str`, `_to_ginac_expr`) the constant
first. Now `_idsub` takes the constant first too: `var_values` holds a
variable only while its last reaching assignment is that constant (every
non-constant assignment removes it), so the last reaching assignment
decides in every translator. Checked with a ternary, a case, min/max and
two loops in sequence (fx5_stale_*).

### A12. The Jacobian was a difference quotient

FIXED in 1.1.5 (CHANGES 1.1.5; test_exact_jacobian). Silent, in every
version from 1.1.0 to 1.1.4 (commit 739eb5ca, 2026-09-19, before 1.1.0, the
PYMS5 skeptic found what it cost). HISTORY: the metaprogram printed a
forward-AD Jacobian with GiNaC (each intermediate differentiated
symbolically, the chain rule through the CSE symbols). It differentiated the
ARITHMETIC select form `base + (alt - base) * S` of a voltage condition while
the value came from a short-circuit ternary, and `_safe_ex` turned a branch
GiNaC could not build into 0 -- so where the untaken branch was not finite,
or could not be built, the slope was NaN or 0 and the charge derivatives of
PSP103 at narrow W / long L came out of wrong magnitude and sign (transient:
"time step too small"). 739eb5ca replaced it with a FORWARD DIFFERENCE
QUOTIENT of the eval in build_vae_so.py's wrapper, step 1e-4 * (|V| + 1) on
the ABSOLUTE node potential: consistent with the value path, but with a
truncation error that grows with the bias, and 1.1.4 stopped printing the
analytic one. Measured (PYMS5 skeptic; this round, `sp2e/pyms/FIX6`): the ac
gain of a diode on 1 kOhm 0.0650491 (exact 0.0652353, -0.29 %); a diode
whose cathode sits at 10 V 1.0392 (exact 0.997241); at 50 V 1.21578 (exact
0.999476); Mextram bjtd504 Ib ac 7.41928e-1 (VACASK 7.43543e-1), with the
emitter at 20 V 7.23115e-1 (-2.7 %), IR(V1) -3.7091e-2 (VACASK -3.45008e-2,
+7.5 %); MVSG 3.5e-5 relative; Newton stopped early (operating points a few
uV off); a 45 V rectifier's transient stalled in steps of 3e-17 s (600 s
timeout; the built-in diode 8 s).

THE FIX (vae/dual_ad.py, build_vae_so.py): forward-mode automatic
differentiation of the eval code itself. The eval code GiNaC prints is
compiled a second time, in namespace `_vae_ad`, with every `double` a dual
number (`_vd`: the value and its slopes) and the math functions overloaded
on it; one dual evaluation gives every column of dF/dx and dQ/dx, exact to
rounding, and it differentiates the very value path the eval computes -- the
same ternaries, the run-time loops, the held values (a held value or a
latched regime is a constant: slope 0), the event bodies -- which the GiNaC
symbolic Jacobian could not. Kept from GiNaC: the whole value path (folding,
simplification, the ddt split); the derivative is taken of what GiNaC
printed. A static dependency analysis of the eval code (every shadow; a
union over every path and assignment, loops to their fixed point) gives
the unknowns each row can depend on; the unknowns are coloured so that two
of one row never share a direction (Curtis-Powell-Reid compression: the
slope of row i in the direction of x_j's colour IS dRow_i/dx_j), and a
variable no unknown reaches stays a double (g++ refuses a dual assigned to
it, so a dependency the analysis missed cannot pass silently there). PSP103:
20 unknowns -> 5 directions, 393 of 1014 intermediates double; BSIM-CMG 22
-> 4; HiCUM L2 20 -> 5. What the analysis cannot read, or a plan g++
refuses, builds the full plan (one direction per unknown; exact as well);
PYMS_AD_FULL=1 asks for it (a check: the same Jacobian, 3.1e-14 relative at most on
PSP103 / BSIM-CMG / HiCUM at 60 random states each). A slope that is not a
number where the value does not move (0 times the infinite slope of sqrt at
0) is 0, and so is the slope a factor of value exactly 0 passes on from the
other factor's (x * sqrt(x) at 0: its derivative 1.5 sqrt(x) is 0, where
plain dual arithmetic gives 0 * inf = NaN; BSIM-CMG 107/108/110 at Vds = 0,
the corpus deck: a NaN Jacobian and a singular matrix without it); a row
whose VALUE is not a number or infinite (sqrt(-1), an exp overflow) has a
NaN row in the Jacobian, as the difference quotient had, because the device
shell sets that value to 0: a finite slope there let a NaN model converge
silently (vamos test_spectre_run_xyce's NaN source; test_exact_jacobian);
an overflow stays infinite (Newton fails and Xyce steps, as with
the difference quotient). The one numeric corner left: a branch equation
whose slopes are ALL exactly 0 at the point (the contribution cancels its
own branch voltage, `V(p, n) <+ tanh(V(p, n))` at its solution 0, where 1 -
tanh'(0) = 0) makes the matrix singular however exact the Jacobian is
(VACASK fails on that deck; 1.1.0-1.1.4 passed it only because their
difference quotient was 3.3e-9 there): that row alone, and only when it is
flat, takes a central difference quotient with a relative step (1e-6 (|V|
+ 1), 1e-6 (|I| + 1 mA)) on each unknown it can depend on, so Newton moves
along the branch (c_own v_tanh 1.0; test_exact_jacobian). Results: the diode decks above 0.0652353 at 0, 1,
10 and 50 V (1.2e-7 apart), the ac tests 1e-9 of the analytic gain of the
operating point; bjtd504 IR(V1) -3.45006823e-2 (e0) / -3.45006606e-2 (e20)
vs VACASK -3.45007895e-2 (3e-6; the operating points differ by Xyce's Newton
tolerance and the two simulators' constants), MVSG -2.48797815e-3 vs
-2.48797777e-3 (1.5e-7); the 45 V rectifier 324 transient steps, 763 Jacobians,
9 s with the build (the built-in diode 347 / 682). Cost: the dual eval with its colours is
cheaper than the 1 + n difference quotients: the eval + Jacobian call 5.5 us
for PSP103 (25.6 us with the difference quotient), 4.0 (20.1) BSIM-CMG, 2.6
(10.7) HiCUM L2; the transient's solver time 0.649 s (1.1.4 0.968) PSP103,
0.658 (0.917) BSIM-CMG, 0.492 (0.631) HiCUM L2 (1.1.3: 0.675, 0.788, 0.517);
the second compile of the eval (the dual copy) 3-9 s, so a library builds a
few seconds longer (corpus build-and-run: HiCUM L2 19.6 -> 23.4 s, PSP103
58.8 -> 67.7, BSIM-CMG 81.8 -> 85.7, L_UTSOI 102 40.6 -> 65.4, BSIM6.1.1
43.8 -> 60.2; 1.1.3 104 / 105 / - / 139 / 77).

### A13. The held state carried across .step points

FIXED in 1.1.5 (CHANGES 1.1.5; test_held_step_reset). A regression of 1.1.4
against 1.1.3 (the PYMS5 skeptic): the device shell's first evaluation of an
analysis (phase 0) read the committed held values, which with .step were the
values the previous step point ended with: `integer cnt = 0;
@(initial_step) cnt = cnt + 1;` counted 1, 2, 3 over three .step points
(s5b_mstep_tran_is_accum, s5b_gstep_tran_is_accum: step 2 0.667 sin for 0.5
sin), and a cross hysteresis started step 2 in step 1's state
(s5_step_tran_cross_reset 0.0314 at 10 us for 0.0126). The shell now keeps
the declaration values (vae_held_init) and restores them, with the crossing
prediction, on the first evaluation of every analysis (phase 0 after any
other phase): 0.5 sin at every step, the hysteresis from r = 4k.

### A14. A .step of a baked parameter changed nothing

FIXED in 1.1.5 (CHANGES 1.1.5; test_step_params_and_modules). Silent, in
every version: `.step m1:rp list 1000 4000` (or a .global_param the card
reads) with `I(p, n) <+ V(p, n) / rp` gave 0.5, 0.5 for 0.5, 0.2
(s5b_mstep_op_ctrl, s5b_gstep_op_ctrl, s5_step_*). The shell's
Model::processInstanceParams did nothing, and Instance::processParams loaded
its math library once. Now the model passes a parameter change on to its
instances, and processParams computes the key of the current values again:
a library of other values is replaced by the one of these (built once per
value set, cached; the row remap and the held state are read again). A
.step over N values of a baked parameter builds N libraries.

### A15. A function local that shadows a parameter read 0 in a declaration value

FIXED in 1.1.5 (CHANGES 1.1.5; test_scope_and_param_type). A regression of
1.1.4 against 1.1.3: parser.py `_is_constant_init` took every module
parameter name for a constant, also inside an analog function whose local
or argument of that name shadows it, so `real r; r = 3 * x; begin : b real t
= r + 1000; ... end` hoisted `t = r + 1000` to the head of the function
body, before r was assigned (b5b_func_local_shadow_silent 0.5 for 0.2;
b5_hoist_func_local_param a singular operating point). The names of the
function's own scope (arguments, locals, its name) are now not constants:
the declaration value stays where it is written and reads the local. The
choice: PyMS keeps a non-constant declaration value where it is written (not
LRM Verilog-A, which wants a constant expression there; VACASK refuses it),
as 1.1.4 meant to; it does not refuse it.

### A16. One module name in two .hdl files: the later file's, silently

FIXED in 1.1.5 (CHANGES 1.1.5; test_step_params_and_modules). 1.1.4
registered every module of every .hdl file; a module name the netlist had
from an earlier file was registered again from the later one and the
netlist's X was the later file's (m5_same_helper_two_files_use: helper of
b.va, 9k, 0.473684 for the first file's 7k, 0.466667). N_DEV_PyMS.C now
registers the first and warns: ".HDL: module helper of b.va has the name of
the module helper of a1.va, registered before: not registered; the
netlist's helper is the one of a1.va".

### A18. A `$temperature` captured in @(initial_step) is not held under `.dc temp`

FIXED in 1.1.7 (CHANGES 1.1.7; test_temperature
test_initial_step_capture_held_under_dc_temp; the SK7 toys a3_dc_temp and
a5_multi_dc now right to 1e-9). ginac_emitter.py `_invariant_body`: a body
that reads `$temperature`, `$vt` without an argument or a thermal parameter
is not invariant, so its variables are held (captured at the initial step,
kept through the analysis, captured again at every `.step` point).
`$vt(<constant>)` stays invariant. The original report:

OPEN (the 1.1.6 skeptic, PYMS7-SKEPTIC step A; t7_live2.txt a3_dc_temp,
a5_multi_dc). `@(initial_step) th = $temperature;` follows every point of a
`.dc temp` sweep (283.15 at the second point for the captured 233.15). The
emitter folds a capture that does not depend on the bias into `double th =
_vae_T;` and gives it no held slot (the deck's eval.cpp line 50). A held
V(p) capture does hold under `.dc temp`, and `.step temp` captures again at
each step as intended. It matters for the thermal design: once `pyms_tsrc_`
moves within an analysis (a thermal node), a captured `$temperature` must
stay what it was at the initial step. hicumL0's INITIAL_MODEL block is an
@(initial_step). Fix: a capture that reads `_vae_T` gets a held slot like a
bias-dependent one.

### A19. `white_noise` / `flicker_noise` contribute nothing in `.noise`

OPEN, never supported (the 1.1.6 skeptic). ginac_emitter
`_strip_noise_call` (codegen.py:308) replaces the noise sources with 0
without a word: a deck whose module has `I(p, n) <+ white_noise(...)`
gives onoise 1.6576e-26 (only the 1e12 ohm resistor) for 1.6576e-17. Until
noise is implemented it should at least warn.

### A20. `.step m1:TEMP` on a module's own TEMP parameter keeps a point's value

OPEN (the 1.1.6 skeptic, t7_a12.txt). `.step m1:TEMP list 50 27 80` with the
circuit at 27 C gives 323.15 / 323.15 / 353.15: the second point keeps 50
for 27. The `_tp_restore` ambiguity (a stepped value that equals the circuit
temperature is taken for "not given") that the source comment admits.

### A21. PSP102: a 4 % Jacobian entry at an unphysical state and T >= 350 K

OPEN, minor (the 1.1.6 skeptic; probe7*.txt). PSP102/102B/102E at a random
state with F = -7e3 A and T >= 350 K: J of F I(D,S) column 0 is -91.39, the
central difference -87.83; the full dual plan gives the same J, and the
central difference is erratic at the neighbouring state (F not smooth
there). 40-state scans at +-0.6 V and +-1.2 V at 300 K and 400 K show
nothing above the noise floor. Probably the difference, not AD; kept to be
checked against a smooth reference.

### A22. A macro argument naming a later macro parameter was substituted twice

FIXED in 1.1.7 (CHANGES 1.1.7; test_macro_args_one_pass). The 1.1.6 code
review (F4, REV7/t4/ms.va); silent, since before 1.1.0. preprocess.py
`_Preprocessor._substitute` replaced the parameters one after another in
the body: with `define ADD2(a, b) ((a) + 2*(b))` the use `ADD2(b, 250)`
first became ((b) + 2*(b)), then ((250) + 2*(250)); the divider gave V(2) =
0.428571428 for 0.6. `SUB(b, a)` gave ((a) - (a)). And re.sub took each
argument as a replacement template (a backslash -- an escaped identifier --
was an escape: `\n1` became a newline). Now one combined pattern of every
parameter (longest first), one pass, a function replacement. The CMC corpus
had no hit (REV7/t4/scan.py: HiCUM QJMOD/QJMODF pass identity-position
arguments only). Left as is: arguments are stripped, so an escaped
identifier as the last argument loses its terminating blank.

---------------------------------------------------------------------------

## B. Loud failures of legal Verilog-A

### B1. BSIM-CMG 111.2.1: strong-inversion drain current 3-8 % above VACASK

Deck (Xyce; all model defaults, e and t at 0):

    .hdl "utils/ADMS/examples/BSIM-CMG_111.2.1_06062022/code/bsimcmg.va"
    .model nm nmos level=111
    VD d 0 0.5
    VG g 0 0
    M1 d g 0 0 0 nm
    .dc VG 0 1.0 0.25
    .print dc v(g) i(vd)

VACASK: the same .va with its module-header attribute removed (openvaf-r
refuses it there, F5), compiled by openvaf-r, `model nm bsimcmg_111`, `m1 (d
g 0 0 0) nm`, one `op` per gate voltage. MEASURED (2026-10-25), I(VD):

| Vg   | Xyce (PyMS 1.1.0) | VACASK        | Xyce - VACASK        |
|------|-------------------|---------------|----------------------|
| 0    | -6.46490646e-13   | -1.464906e-13 | -5.0e-13 (gmin x Vd) |
| 0.25 | -1.00627389e-09   | -1.005774e-09 | -5.0e-13 (gmin)      |
| 0.5  | -1.88654361e-06   | -1.884306e-06 | 0.12 %               |
| 0.75 | -2.19759299e-05   | -2.122496e-05 | 3.5 %                |
| 1.0  | -5.65635491e-05   | -5.245811e-05 | 7.8 %                |

Identical on PyMS before cluster F (the gmin change) and after this
round's fixes, so neither moved it. Subthreshold agrees exactly (the
difference is Xyce's gmin), the error grows with the current: a series
resistance or a mobility/velocity term that one engine drops. Not traced.
Xyce's ADMS-compiled BSIM-CMG (level 111) would decide which engine is
right, but this build has `Xyce_ADMS_MODELS=FALSE`. Next steps: rebuild with
the ADMS models or compare against ngspice's OSDI load of the same .osdi;
then print the internal drain/source node voltages (PyMS: the shell's
`Internal:` list) and the `RDSMOD`-dependent resistances. Test: the table
above to 1e-6 against the reference engine.

### B2. MVS 2.0.0 etsoi/hemt: the DC operating point does not converge

Corpus deck (1 V / 1k divider, port 1 = in, port 2 = out, the rest 0):
`ADMS/mvs_2.0.0/mvs_2_0_0_etsoi.va`, `mvs_2_0_0_hemt.va` (newly built by this
round, PYMS1-C; before they did not build and "ran" dead) end with "the DC
operating point did not converge" (PYMS1-I corpus, measured before the
self-reference fix of CHANGES; not re-run since). PSP102 (`psp102.va`, `b`,
`e`), which failed the same way before and after the round, converges since
that fix (2026-10-25: rc 0, v(out) = 0, the gate node). Method that found the
BSIM-SOI cause: a probe that dlopens the math library `vae_<module>_<key>.so`
and calls `vae_eval` / `vae_jacobian` at the bias point (non-finite F, Q or
Jacobian entries mean a failed linear solve), then an instrumented copy of
eval.cpp that prints the first non-finite intermediate. Test: both decks
converge and match Xyce's ADMS build or VACASK.

### B3. `idt` / `idtmod` are refused

    analog I(p, n) <+ idt(V(p, n)) / 1u;

(PYMS1-D) Refused with a clear error (before: a g++ failure). Affects the PSP
NQS variants (`psp102*_nqs.va`, `psp103_nqs.va`: 8 corpus files). Design: an
internal state node per `idt` instance x with the equation `ddt(x) = arg` (a
charge row: Q = x, F = -arg) and the `idt` expression replaced by V(x);
`idt(arg, ic)` adds the initial condition at the DC op (x = ic, the `ddt`
row replaced by `x - ic = 0` at DC), `idtmod` wraps x by its modulus in the
output. vae/ginac_emitter.py `_vae_split_ddt` already does the Q/F split the
new row needs. Test: `I(p, n) <+ idt(V(p, n))/1u` against VACASK in a 1 kHz
transient, and the NQS PSP103 decks build.

### B4. A voltage-dependent factor on `ddt` is refused

    analog I(p, n) <+ V(p, n) * ddt(1p * V(p, n));

(PYMS1-D) Refused; VACASK runs it (0.5 on the reference deck). The Q/F split
(`_vae_split_ddt`) needs `c * ddt(q)` with c voltage-free. Design: the product
rule is not available for a charge term, so model `c * ddt(q)` as a current
`c * i_q` with `i_q` a branch-current unknown of an internal charge node
(`ddt(q) = i_q`), the same machinery as B3. Test: that module against VACASK
in dc and a 1 kHz transient.

### B5. Voltage and current contributions to overlapping, non-exclusive paths are refused

(PYMS1-E) `if (sw) V(p, n) <+ 0; else I(p, n) <+ g * V(p, n);` works (exclusive
arms); `V(p, n) <+ ...; I(p, n) <+ ...;` in a sequence where the kinds overlap
is refused ("last-kind-wins" is not decidable statically). Design: a runtime
kind indicator per branch (a 0/1 value the eval writes) and a row that blends
`kind * (V - e) + (1 - kind) * (I - f)`. Test: a switch whose condition depends
on a parameter only (decidable) and one on a voltage (runtime).

### B6. Analog functions not expanded: fbh_hbt `diode`, MVSG `calc_iq`

Mostly FIXED in 1.1.3: analog functions are parsed and inlined (CHANGES
1.1.3). In the corpus scan fbhhbt-2.1_nonoise*.va (4 files), fbhhbt-2.3*.va
and mvsg_cmc_1.1.0.va now build and run (ADMS/ and share/ copies);
`fbh_hbt-2.1/fbhhbt-2.1.va` (with noise) still has no math library
(not traced in 1.1.3). The original entry:

`ADMS/fbh_hbt-2.1/fbhhbt-2.1.va` (5 files, 2.3 too), `mvsg_cmc_1.1.0.va`:
"math library did not build: ginac.cpp:238: error: 'diode' was not declared"
(PYMS1-I corpus). Before this round they "ran" dead. The call reaches the
GiNaC program verbatim, so the function was not inlined. Not yet traced; what
the sources show (2026-10-25): fbh's `analog function real diode;` (header
form, `input U, Is, ...;` and `real ...;` after it) calls another analog
function (`Vth(20.0)`) and contains `if` / `begin` blocks, and is called with
expression arguments (`diode((-BVebo - vbei), Jsf, ...)`, fbhhbt-2.1.va
l.195-445); MVSG's `calc_iq` has eight `output` arguments
(mvsg_cmc_1.1.0.va l.314-323) and is called as `idsfp4 = calc_iq(idsfp4,
...)`. Start in ginac_emitter.py's function inlining: find which of nested
calls, block bodies or output arguments it does not handle, then support
every LRM analog-function form (output/inout arguments by reference,
`integer` functions). Test: a module with a header-form function that calls
another function, and one with an `output` argument, = VACASK.

### B7. DIODE_CMC: `D_0_conjugate` in eval.cpp

`ADMS/DIODE_CMC_2/diode_cmc_2.0.0/diode_cmc.va`: "eval.cpp:854: error:
'D_0_conjugate' was not declared" (PYMS1-D/I). PYMS1-D traced it to `abs`
on complex symbols: GiNaC's derivative of `abs(x)` for a symbol that is not
known to be real carries `conjugate(x)` terms, which the C printer emits as
an unknown function. Suggested fix: declare the emitted symbols real
(`realsymbol` in the GiNaC program of ginac_emitter.py
`emit_ginac_program`), so `abs` differentiates to `csgn`; check that the
other models' eval.cpp do not change. Test: DIODE_CMC builds and its forward
bias point = VACASK or Xyce's built-in ADMS DIODE_CMC.

### B8. toys/diode2.va: the GiNaC metaprogram dies with SIGFPE

(PYMS1-H) `.MODEL D1N3940 D(level=1002002 IS=4E-10 N=1.48 CJO=1.95E-11 M=.38
PHI=.4 FC=.9 TT=8E-7 BV=600 RS=.105)`, the clipper deck
`sp2e/pyms/H/decks/clipper/clip_pyms.cir`: "build_vae_so: ... ginac > eval.cpp
/ Floating point exception" (pristine: exit 0, the diodes conducted nothing:
V(2) 6.577/-2.296 V against the built-in diode's 5.586/-0.606 V). SIGFPE is an
integer division by zero or overflow in the metaprogram: suspects are integer
arithmetic in the `from (0:1]` / `exclude` ranges and an integer-typed
intermediate of the `$limit` / `$vt` expansion. Fix: run the metaprogram under
gdb with that params file, make the construct right or a clear error. Test:
the clipper deck = the built-in level-1 diode to 1e-3.

### B10. A 1e12 S conductance: the DC operating point does not converge

    parameter real r = $simparam("gmin", 1000.0);     // or r = 1e-12
    analog I(p, n) <+ V(p, n) / r;

In the 1 V / 1k divider (2026-10-25, PyMS 1.1.1; 1.1.2 the same): rc 1,
"Step size reached minimum step size bound" after 4020 Jacobians; r = 1e-10
and larger run (1.0). Xyce's own `R1 in out 1e-12` gives 1.0 (also with a
diode on out, a nonlinear circuit); with r a callback parameter
(`PYMS_CALLBACK_PARAMS=R`, kept symbolic) the module runs (1.0). Not
`$simparam`: the literal fails too. (The 1.1.0 section here said
`$simparam("gmin")` in an analog expression did not converge; it was a
silent 0, V(out) = 0 rc 0 -- the expression became 0 S -- fixed in 1.1.1.)

Mechanism: UNKNOWN. The 1.1.1 text blamed PyMS's eval -- GiNaC distributing
1e12 over `V_p - V_n` so the products round before they cancel, and a
difference-quotient Jacobian inheriting the noise. The PYMS2 skeptic
(2026-10-26) showed that this does not explain it: Xyce's own B source fails
the same way written either way, `B1 in out I={V(in,out)*1e12}` and
`I={V(in)*1e12-V(out)*1e12}` (rc 1, the same message), and Xyce derives a
B source's Jacobian analytically, so neither the distributed form nor the
Jacobian is the cause. What the evidence supports: a 1e12 S element loaded
as a nonlinear device (a PyMS device or a B source) does not converge in
Xyce's DC Newton, where the linear resistor of the same value does. A
plausible cause, not verified: the residual of such an element cannot be
smaller than ulp(1 V) x 1e12 S = 2.2e-4 A at a node near 1 V, far above the
residual tolerance, while the linear resistor is handled differently. Next
step: Xyce's nonlinear-solver trace (`.options nonlin debuglevel`) on the
B-source deck, then decide whether PyMS can do anything about it (it cannot
if a B source has the same limit). Test: the module above gives 1/(1 +
r/1k) like Xyce's resistor, or the section is closed as Xyce's limit.

### B11. Vector nodes are a ParseError

    electrical [0:3] bus;

(PYMS1-I) Now a clear ParseError (before: silently mis-parsed). Design: expand
`bus[i]` into scalar nodes `bus_0`..`bus_3` in the parser (ports too), with
`V(bus[i])` indices constant-evaluated by va_const. Test: a 4-bit resistive
ladder = VACASK.

### B12. Access functions of undeclared disciplines

(PYMS1-I) A module that uses a discipline neither declared in the source nor
in the shipped `disciplines.vams` falls back to the old Temp -> V_ mapping.
Fix: an undeclared discipline or access function is a clear error naming
file:line. Test: `foo t; analog Bar(t) <+ 1;` is an error.

### B13. Two parameters that differ only in case

(PYMS1-H) Legal Verilog-A; Xyce's parameter names are case-insensitive, so
PyMS refuses the module (file:line, both names). The claim that EKV 2.6 hits
this is wrong: its `Temp` line (l.217) is commented out and only `TEMP` is
declared (2026-10-25: `xyce_device_gen.py` generates ekv_va, 79 parameters,
on the pristine tree, H, B, C and the final tree; the corpus runs it). No corpus file needs it. Design if one
does: register the second name mangled (`TEMP__2` or the escaped-names rule)
and map it back in build_vae_so.py, with a warning. Test:
test_hdl_param_name_case, a module with `r` and `R`.

### B15. The CMC benchmark decks: segfaults in the matrix setup, few pass

`utils/PyMS/run_adms_tests.sh --pyms` (2026-10-25, PyMS 1.1.0, the relinked
library, 3062 s): 3 of 48 decks OK (BSIMCMG108 `cfrgeo.sp` and
`ringosc_17stg.sp`, bsimcmg_107 `cfrgeo.sp`), 45 FAIL; many by SIGSEGV. Deck BSIM6.1.1/Benchmark_tests/inv_dc.sp
(an nmos + pmos inverter of `BSIM6.1.1_inlined.va`, two model cards, `.dc
vin 0 1 0.01`, through `cadence2xyce.pl`): backtrace

    #0 Xyce::Topo::Indexor::matrixGlobalToLocal(...)
    #1 Xyce::Topo::CktGraphBasic::registerJacLIDswithDevs(...)
    #2 Xyce::Topo::Topology::registerLIDswithDevs()
    #3 Xyce::Circuit::Simulator::setUpMatrixStructure_()

i.e. a device's Jacobian stamp pattern that does not match its node list.
Not caused by this round's library: it crashes the same way with the
pristine PyMS on the stock (pre-1.1.0) library and with G's relink of the
UNCHANGED N_DEV_PyMS.C, which is byte-identical to the stock library (md5
c19177af...); one run (1.1.0 Python, stock library, cold caches) instead
ended with 101 nonlinear convergence failures, so the crash is not
deterministic per library. Suspects: the shell's per-instance node
collapse vs the math library's (D4), and the branch-current unknowns of
PYMS1-E for instances of one module on two cards. Others fail with netlist
errors from the cadence2xyce translation (BSIMCMG108 ac.sp line 12) or no
convergence (BSIMCMG108 gummel_n.sp: 301 failures). Logs:
`$WORK/pyms/<test>/run.log` of the script. Test: inv_dc.sp runs and matches
Xyce's ADMS BSIM6.

### B14. `$abstime` reaches g++ verbatim

FIXED in 1.1.4 (CHANGES 1.1.4): the device shell passes Xyce's time
(VaeState t, 0 at a dc operating point) and `$abstime` (and `$realtime`) is
the eval code's `_vae_abstime`; the module below gives V(A) = 1 at the op
(the vamos fixture's V(A) = 0 pins the old failure: G). The original entry:

    `include "disciplines.vams"
    module nanv(a);
      inout a; electrical a;
      analog V(a) <+ ($abstime > 1u) ? sqrt(-1.0 - V(a)*0) : 1.0;
    endmodule

    .hdl "nan.va"
    .model nanv__vamos nanv
    ynanv X1 a nanv__vamos
    R1 a 0 1000.0
    .OP
    .PRINT DC V(A)

(vamos fixtures engine_facts/E125/abstime.cir, E126/nan.cir; 2026-10-25)
Xyce rc 1: "build_vae_so: error: ... eval.cpp:11:24: error: '$abstime' was
not declared in this scope", then the device error "the Verilog-A math
library for module nanv did not build or load". Before 1.1.0 the same build
failure ran a dead device (V(A) = 0, rc 0). Analytically V(A) = 1 at the
op (time 0; VACASK not run on it here). Mechanism: ginac_emitter.py maps
`$vt` and `$temperature` (`_to_ginac_expr`, `_to_cpp_expr`, the
`$temperature` regexes) but has no case for `$abstime`, so it is printed as
an identifier. Fix: pass the simulation time into the math library (a
`VaeState` field the shell fills from Xyce's current time, 0 at the DC op)
and map `$abstime` to it; until then a clear EmitError ("$abstime is not supported", file:line)
before any g++ run. Test: the module above gives V(A) = 1 at the op and,
in a transient past 1 us, the NaN branch is a clear error or VACASK's value.

### B17. Mextram 504 with self-heating: an open optional port is a SIGSEGV

    .hdl "utils/ADMS/examples/mextram_504.12.1/bjt504t.va"
    .model q1 npn level=505
    Q1 c b 0 0 q1            * dt (port 5) not given

(2026-10-25, PyMS 1.0-1.1.1) Xyce exit 139 in "Setting up matrix
structure". With dt given (`Q1 c b 0 0 dt q1`, `Rdt dt 0 1e6`) it runs and
equals bjtd504t.va on the Y route (Ic 2.82954e-7 ... 6.46088e-3 A for Vb
0.6 ... 0.9 V at Vce = 1.5 V). Mechanism: the BJT group needs 4 terminals
and the shell reports the fifth as optional (xyce_device_gen.py
`num_nodes_optional`), but `Instance::Instance` sets `numExtVars` to all 5
and `jacStamp_` covers every port: an absent port's LID is -1 and the
stamp indexes it. Fix: `numExtVars` from the instance block's node count,
the stamp and the KCL rows of an absent port skipped (state.V already reads
it as 0), or pin it like a dead node. Test: the deck above runs and gives
the same Ic as with dt tied to a 0 V node through 1e6.

### B18. Mextram 504 switched off: the DC operating point does not converge

    .hdl "utils/ADMS/examples/mextram_504.12.1/bjt504.va"
    V1 in 0 1
    VB b 0 0
    .model m1 npn level=504
    Q1 in b 0 0 m1

(2026-10-25, PyMS 1.1.1) rc 1, "Step size reached minimum step size bound"
(the corpus deck, base to ground through 1k, the same; `.nodeset` does not
help). A `.dc VB 0.9 0 -0.3` sweep runs down to Vb = 0.3 V (Ic 6.0315e-3,
2.8091e-7, 8.03e-12 A) and fails at 0. VACASK on the same file (the
`$limit` calls replaced by their argument) converges at Vb = 0 (Ic
1.573e-13 A). Before 1.1.1 every Mextram ran at any bias with Ic = Vce/12.16
ohm (the node collapse, now fixed); the forward-active points now agree
with VACASK to 0.03-0.12 % (CHANGES). The four files behave alike; the same
bjtd504.va converged from one path (utils/ADMS: 5.1e-12 A) and not from the
other (share/verilog-a). Suspects: the eval's rounding at currents of 1e-13
A beside 1/RCC-sized conductances (the mechanism of B10), and `$limit`
(PyMS passes its argument: no limiting). Fix: B10's branch-voltage
intermediates first, then compare the Newton path with VACASK's. Test: the
deck above gives VACASK's Ic to 1 %.

### B19. sg13g2 r3_cmc: the GiNaC metaprogram divides by zero

`share/sg13g2/r3_cmc.va` on the corpus deck (a `.model m1 r level=...` card
with no parameters): build_vae_so "terminate called after throwing an
instance of 'GiNaC::pole_error' / what(): power::eval(): division by zero"
(2026-10-25, PyMS 1.1.1). A loud device error since 1.1.1 (CHANGES 1.1.1); before, the
Resistor group's simple stamp ran rc 0 without a math library. A default of
0 in a denominator that elaboration folds (a geometry parameter w or l, or a
sheet resistance) is the likely cause. Fix: report a GiNaC exception of the
metaprogram with the statement's file:line (D1) and decide whether the
model is meant to run at those defaults (VACASK on the same card is the
reference). Test: the corpus deck builds, or fails with one file:line line.

### B20. Several parameters in one declaration

FIXED in 1.1.4 (CHANGES 1.1.4; test_param_list). CORRECTED in 1.1.5: the
1.1.4 entry said untyped parameters were fixed too, but an untyped parameter
was an integer only when its default was a bare integer literal (an
integer-literal regex): `parameter m = 2 * 2` and `parameter b = a` with an
integer a were real, m / 3 = 1.33 (p5_untyped_int_expr, p5_untyped_ref_int
0.571428571 for 0.5). Since 1.1.5 the default expression is typed by the
integer-arithmetic typer (parser.py `_untyped_param_type`, int_ops.analyse:
integer literals, integer parameters declared before, the integer operators
on them; real for a real literal, a scale factor, a real parameter or a real
function): 0.5 (test_scope_and_param_type). The original entry:

    parameter real is = 1e-14, nf = 1.5;

Refused with a misleading message: `_parse_param` reads one name and takes the rest
of the statement (`1e-14 , nf = 1.5`) as the default text, which va_const
then refuses ("the character '=' is not part of a constant expression",
found by this round's test_ginac_determinism, 2026-10-26). Loud, not
silent. Fix: parse a list of `name = default [range]` items. Test: the
line above gives is = 1e-14 and nf = 1.5.

### B21. A later module's xyceModelGroup landed on the first module of a file

FIXED in 1.1.5 (CHANGES 1.1.5; test_step_params_and_modules). Loud, 1.1.0 to
1.1.4: xyce_device_gen.py `scan_va_attributes` took the first
`(* xyceModelGroup=... *)` block of the whole file for the first module, so
`module a1 ...` followed by `(* xyceModelGroup="Diode" ... *) module dd`
registered a1 as a Diode level 2001 and `.model m1 a1` was a netlist error
(m5_second_has_group_attr). The scan now reads the declaration of the module
the shell is for (`module_name`).

### B22. A .step of a baked parameter crashed a module with held state in a transient

FIXED in 1.1.7 (CHANGES 1.1.7; test_library_swap_accept). The 1.1.6 code
review (F1, BLOCKING; REV7/t1/stp2.va + b.cir); present since the 1.1.5
library swap (A14). The generated `Instance::acceptStep` cached
`dlsym(vae_dl_, "vae_needs_accept")` in function statics keyed on the dlopen
handle. processParams dlclose's the old math library and dlopen's the new
one at a `.step` of a baked parameter; glibc reuses the link_map, so the
handle compares equal while the code moved: a stale function pointer,
SIGSEGV (1.1.6, rc 139 at the second point of `.step m1:mode list 0 1 0`
with `.tran`) or "Illegal instruction" (1.1.5), i.e. whatever code sits at
the old address runs. Needs held state (pyms_nheld_ > 0), a transient and a
rebuild of the library. Now the pointer is an instance member
(`pyms_needs_accept_`), read with the held state when a library is loaded
and cleared with everything else read from it when the library is dropped.
It was the shell's only function-static dlsym cache (the param callback is
reinstalled at every processParams; the other dlsym results are members
cleared at the swap).

### B23. pow with a zero base and a bias-dependent exponent: a NaN Jacobian

FIXED in 1.1.7 (CHANGES 1.1.7; test_pow_zero_base). The 1.1.6 code review
(F2, REV7/t2/z2.va): `I(p,n) <+ V(p,n)/1k + 1e-6*pow(max(V(p,n),0.0),
1.0+V(p,n))` on a 1k divider failed every point of `.dc V1 0 1 0.5` (Failed
Linear Solves); with a constant exponent (z3) it converged. dual_ad.py
`pow(double, _vd)` and `pow(_vd, _vd)` took the slope along the exponent as
`r.v*log(a.v)` = 0*(-inf) = NaN at a zero base. pow(0, y) is 0 for y > 0
with slope 0 along y: the slope is now 0 wherever the value is 0. z2 now
gives V(2) = 0, 0.24991164, 0.499823295.

### B24. A V/I switch row is NaN when the untaken arm is not finite

OPEN (the 1.1.6 code review, F3; REV7/t3/tc.va, tv.va). A branch with a
voltage contribution in one arm and a current contribution in the other of a
run-time condition is one blended row: `parameter real rs = rs0*($temperature
/300.15); if (rs > 0) I(p,m) <+ V(p,m)/rs; else V(p,m) <+ 0;` with rs0 = 0
gives `(V_p-V_m)*(_sel-1) + 1.0/rs*(V_p-V_m)*_sel - I*_sel` and 1/0*0 = NaN:
every point fails (tc0.cir, tc2.cir rc 1; tc1 with rs0 = 1k is right at 27 C
and 327.15 C). 1.1.5 refused tc.va (a parameter default reading
$temperature), so tc0 is not a regression; the same with a variable computed
from $temperature (tv0.cir) fails in 1.1.5 and 1.1.6, and so would any
voltage-dependent or callback-parameter condition whose untaken arm is not
finite. Mechanism: ginac_emitter.py builds the contributions of the arms of
a run-time condition as indicator products (`_sel`, `1 - _sel`) at the GiNaC
level and `_finalize_current_rows` adds them in one row (B5 describes the
blend). Fix: emit the switch row (and the gated contributions) as a select
(`_sel ? a : b`) whose AD takes the slope of the taken arm only -- a GiNaC
conditional function through the printer and dual_ad; not a contained change,
left for a round of its own. Until then: guard the division in the model
(`rs > 0 ? V/rs : 0`).

### B25. A held peak in a transient with tight tolerances stalls

OPEN, found by the 1.1.7 round (FIX8/decks/h1.cir; 1.1.6 the same). The
REV7 stp2.va module (`@(initial_step) pk = 0; if (V(p,n) > pk) pk = V(p,n);
I(p,n) <+ V(p,n)/r + 1e-4*pk;`, mode=0) placed directly on `V1 in 0 SIN(0 1
1k)` with R2 = 1k to ground, `.options timeint reltol=1e-6 abstol=1e-12`,
`.tran 10u 2m`: the time stops at 1.48771230e-03 and Xyce prints millions of
rows there with V(out) alternating 6.237404e-2 / 6.237420e-2 (2.6e6 rows in
60 s, a 1 GB .prn). Without the .options line (h2.cir), or with R1 between
the source and the device (h0.cir), it finishes. Not yet analysed: probably
the held value and the accepted-step re-evaluation (acceptStep's
vae_needs_accept evaluation) disagreeing at a point where V(p,n) is
nowhere near the peak; to be checked with a trace of pyms_Hc_ / pyms_Hp_
per accepted step.

---------------------------------------------------------------------------

## C. Decisions and strictness

- C1. `1000.0/0.0` in a parameter default is refused (PYMS1-A); VACASK gives
  0. The LRM leaves it undefined; PyMS's error is the safe side. Keep.
- C2. Attributes at positions openvaf-r refuses (inside expressions, before
  `else`) are accepted (PYMS1-B). Decide whether to match openvaf-r; a
  warning is the low-risk choice.
- C3. A function-like macro whose arguments start on the next line, and a
  `define inside a multi-line macro invocation, are errors (PYMS1-C; 0 uses
  in the corpus). The LRM allows neither explicitly; keep unless a model
  needs them.
- C4. `bsimcmg_nqsmod3.va` (BSIM-CMG 107/108) uses `` `attr `` without
  defining it (its user is expected to); refused as an undefined macro
  (PYMS1-B). Works with `PYMS_DEFINES=attr=` or `--define attr=`.
- C5. An instance-only parameter on a `.model` card: VACASK takes it (1/3 on
  the vsrc deck), Xyce warns and ignores it (0.5) unless the parameter is
  `xyceAlsoModel` (PYMS1-E/F). Xyce's rule, not a PyMS defect; vamos moves
  such a value to the instance line.
- C6. The legacy nonzero-only retry in build_vae_so.py (drop zero-valued
  baked parameters and build again) runs only for shells without a
  `__GIVEN__` marker, which no current generator writes; delete it
  (PYMS1-H).
- C7. vae/codegen.py (l.523-525) keeps its own `CONSTboltz` / `CONSTq`
  instead of the shipped constants.vams values (PYMS1-C); unify.
- C8. Not a PyMS defect: the E64 toy diode (vamos
  `fixtures/spectre/engine_facts/E64/toy_diode.va`, the pattern of
  utils/ADMS/examples/toys/diode.va) with `.model dm d level=1002001
  Is=1e-14 Rs=10` and `I1 0 a 1m`, `D1 a 0 dm` gives V(a) = 1e9 V (PyMS
  1.1.0 and 1.1.1, rc 0; 1.01e9 with "Numerically singular matrix" was the
  pristine tree before 1.1.0, HEAD 919b89d2 -- corrected by the PYMS2 skeptic). The
  module contributes `I(a, ci) <+ Is*(limexp(V(ci, a)/26m) - 1)`: the
  exponent reads V(ci, a), the reverse of a diode, so 1 mA into a has two
  roots of the gmin-augmented KCL: V(a) = 0.01 - 26m*ln(1e11 + 1) =
  -0.648539 V (VACASK with `is=1e-14 rs=10`: -0.6485393699, V(ci) = 0.01)
  and V(a) = 1 mA / gmin = 1e9 V, where the reversed exponential is flat.
  Newton from 0 sees a negative conductance (-Is/26m) and takes the second.
  The same equations as Xyce built-ins (`B1 a ci I={1e-14*(exp(V(ci,a)/26e-3)-1)}`,
  `R1 ci 0 10`) fail outright (singular, rc 1), and ngspice's B source gives
  V(a) = 1e9 too; with `.nodeset v(a)=-0.6` PyMS gives -0.648539206 (=
  VACASK to 2.5e-7). The analytic 0.669 V is that of a diode written the
  right way round (`V(a, ci)`). vamos's E64 check pins 1.01e9 and the
  singular-matrix message; it should pin the root it means with a nodeset.
- C9. A short inside a conditional (`if (RGATEMOD == 0) V(gi, g) <+ 0;
  else ...`) keeps its internal node since 1.1.1 (a 0 V source on a
  branch-current unknown, the first item of the 1.1.1 CHANGES), and the shell's diagonal gmin
  (`.options device gmin`, 1e-12) now sits on that node too: one gmin more
  from the port to ground per kept node (the corpus: i(V1) 1e-12 -> 2e-12
  ... 4e-12 A on switched-off MOSFETs). Xyce's own devices put gmin on their
  junctions; if the leak matters, skip the diagonal gmin on a node that is
  an end of a voltage-contribution branch alive in the elaboration (the
  library's `vae_current_row` >= 0 says so at run time).
- C10. Not a defect: BSIM-SOI 4.7 (`BSIM-SOI_4/bsimsoi4.7.0/bsimsoi.va`,
  `.model m1 nmos level=70470`, `M1 in out 0 0 0 0 0 m1`, the gate on out
  with 1k to ground, 1 V on the drain) gives V(out) = gmin x V(in) x 1k:
  9.99999995e-10 at gmin = 1e-12, 9.99995e-7 at 1e-9 (PYMS2 skeptic; the
  same on 1.1.2). The current into the gate is the model's own term,
  l.7395 `I(gi,di) <+ ... + MULT_I * ($simparam("gmin", 0) * V(gi, di))`,
  a gmin conductance from the gate to the drain: with that one term
  replaced by `0.0 * V(gi, di)` V(out) = -2.7e-48 and i(V1) falls by one
  gmin (-4.0000e-12 -> -3.0000e-12 A; -4.0e-9 -> -3.0e-9 A at gmin = 1e-9),
  measured 2026-10-26 on PyMS 1.1.2. No gmin sits on a wrong row.
- C12. vae/codegen.py (the sympy-era emitter of the vae_*.cpp toys; no Xyce
  path imports it) still takes a branch end named 'gnd' for ground. Delete
  it or give it the emitter's GND.
- C13. A module's analog function named like a PyMS builtin (lexp, lln,
  hypsmooth, hypmax, Tempdep, limexp, ...) is not inlined: PyMS's builtin is
  used, as before 1.1.3 (BSIM-CMG / BSIM-SOI / BSIM6 define lexp, lln,
  hypsmooth). The builtins differ from the CMC sources: `lexp` and `lln`
  are plain exp / log in the eval code (the CMC lexp is linear above
  EXPL_THRESHOLD and floors below -EXPL_THRESHOLD; lln is ln(max(x,
  N_MINLOG))). Decide: inline the source's definition (the corpus would
  change) or document the builtins as the dialect.
- C14. FIXED in 1.1.4 (CHANGES 1.1.4). It said: a loop whose condition
  reads the node voltages is unrolled 16 times under runtime indicators and
  an evaluation that would need more prints one warning per library and
  keeps the values after 16 -- silently wrong past the warning
  (l4_vdep_counter_20 0.444, l4_vdep_unbounded_17 0.667 for 0.5). Now such
  a loop is a C++ loop of the eval code (ginac_emitter.py
  `_walk_runtime_loop`; the Jacobian is a difference quotient of the eval,
  so no derivative of the loop is built), exact, with no cap but an
  endless-loop guard: past 100000 passes a hard error of the simulation
  naming the loop's file:line. Only a body with ddt(), a voltage
  contribution or an event statement is still unrolled (16 passes); an
  evaluation that would need more is a hard error naming the loop, never a
  warning and a value.
- C15. CORRECTED and FIXED in 1.1.4 (CHANGES 1.1.4). The 1.1.3 entry was
  incomplete: it said `@(initial_step)` runs at every evaluation, the same
  as once for a parameter-only block; it did not say that an
  analysis-qualified block -- `@(initial_step("ac"))`, `("tran")` -- ran in
  EVERY analysis, a bare .op included (the parser dropped the analysis list:
  is4_ac_only_op / is4_tran_only_op 0.2 for 0.5), that `@(final_step)` and
  `@(cross(...))` bodies ran at every evaluation too, and that `$abstime` in
  an initial_step was refused (g++ "'$abstime' was not declared", where
  1.1.2 had skipped the block). Now: analysis shadows, held variables,
  final_step / cross / above events (C16). An integer `/` or `%` by a zero
  known only at run time gives 0 AND a warning naming the statement's
  file:line, once per site (a constant zero at elaboration is an error).
- C16. Analysis-specific code (1.1.4). The user's rule (2026-10-30):
  `@(initial_step("ac"))` and `@(initial_step("tran"))` imply different
  analyses and may be entirely different code, so the math library holds a
  SHADOW of the eval code per analysis context that needs one, chosen by
  the analysis the device shell passes, not one body with run-time tests.
  The contexts (VaeState an, phase; ginac_emitter.py VAE_CONTEXTS): an = dc
  (.op, .dc), tran, ac, noise; phase 0 = the first point an analysis starts
  from (its operating point; a transient's t = 0 point, its "ic"), 1 = the
  later points (dc sweep points, transient steps), 2 = the small-signal
  sweep of ac / noise, 3 = the evaluation of an accepted transient point
  (the shell's acceptStep). The names analysis() is true for and an
  initial_step list matches (LRM 4.6.1): phase 0 of dc {dc, static}, of tran
  {tran, ic, static}, of ac {ac, static}, of noise {noise, static}; phase 1
  {dc, static} / {tran}. Rules: an initial_step whose list is empty or names
  the analysis runs in phase 0 only -- a bare .op runs neither "ac" nor
  "tran"; a plain @(initial_step) runs once at the start of each analysis;
  a variable an event statement assigns is HELD (Verilog-A variables are
  static): every evaluation starts it from its value at the last accepted
  point (the declaration value at the start of every analysis -- every
  .step point's; 1.1.4 went on from the values the previous .step point
  ended with, corrected in 1.1.5, A13) and its value at the end
  of the evaluation is the new held one (is4_vread_tran: v0 = V(p, n) of the
  operating point, then 1/3 sin). The small-signal phase linearises the
  operating point: it runs the operating point's shadow again (what
  initial_step("ac") set there holds: AC gain 0.2 with `r = 1000;
  @(initial_step("ac")) r = 4000;`) and reads the regime -- every indicator
  of a voltage-dependent condition outside a run-time loop -- of the
  operating point's last evaluation instead of deciding it again
  (`_vae_latch`): `if (V(p, n) > 0) 1k else 4k` at a 0 V operating point is
  4k in ac (0.2), where the difference quotient across the condition gave
  the other branch's slope (0.5). Inside a run-time loop a condition is not
  latched: the ac evaluation decides it again at the operating point's
  voltages, which gives the operating point's branch, and the exact
  Jacobian (1.1.5, A12) takes that branch's slope (s5_ac_regime_in_rtloop
  0.8 -> 0.2). A value an @(initial_step) computes from the voltages keeps
  its slope in ac (the user's decision of 2026-10-31: `@(initial_step) r =
  1000 + 3000 * V(p, n)` linearises through r; 0.199948 with the
  difference quotient, 0.2 exact). @(final_step) runs after the last point,
  where no evaluation follows: it changes no value PyMS computes (system
  tasks are not executed by PyMS) and is not built. cross / above: the
  expression is held from accepted point to accepted point; an accepted
  point that finds it crossed (above: upwards; above also at the first point
  when >= 0) marks the event pending, the shell puts a breakpoint just after
  it, and the accepted point of that breakpoint runs the body and holds its
  values -- the change comes at a breakpoint; the shell also puts a
  breakpoint where the last two accepted values predict the crossing (one
  per approach). timer and the other events are a file:line error. An
  invariant initial_step (it reads parameters only and its variables are
  assigned nowhere else, toys/diode2.va) computes the same at every point
  and runs where it stands in one body: no shadow, no held state. A module
  without analysis-specific code has one body, as before; the held regime
  costs one store per indicator.
- C17. Xyce cannot step past a jump of the solution that a device makes:
  an @(cross) / @(above) body that changes the circuit (a resistance from 4k
  to 1k) makes V(out) jump at the event, and Xyce's step-error control
  rejects every step after it ("Time step too small", rc 1; it is loud, the
  same for any device, and the event itself is right). With `.options
  timeint erroption=1` (the Newton test alone steps the transient)
  fx5_cross_hyst_tran gives [0.293891, 0.499998, -0.117556, 0.499998,
  -0.117556] for the analytic [0.293893, 0.5, -0.117557, 0.5, -0.117557].
  A smooth change (transition(), not supported by PyMS) would need no option.
- C18. OPEN (not PyMS): `.ic V(out)=0.5` with `@(initial_step) r = 1000 +
  2000 * V(p, n); I(p, n) <+ V(p, n) / r;` (s5_tran_ic_is_vread,
  s5_tran_ic_is_tran_vread): the operating point fails, all versions,
  1.1.5 too. Traced (1.1.5, a phase trace of the shell): every evaluation of
  the .ic operating point is phase 0 (dcop, tranop) and the library is
  right (r, F and the exact slopes at (1, 0.5)); Xyce's first Newton iterate
  of the .ic operating point is V(in) = 0, V(out) = 0.5, where V(p, n) =
  -0.5 and r = 0: the model's own pole, I = V / 0 infinite, three failed
  linear solves, every continuation restarting there. A deck whose device
  is finite at that point runs (s5b_ic_ctrl_pyms_r 0.5, s5b_ic_other_node
  0.333389). Possible: a Xyce-side initial guess from the .ic values of the
  sources; not attempted.
- C19. limexp() is exp() (ginac_emitter `_VA_TO_GINAC`): no linear
  continuation above a threshold as openvaf-r has; a forward-biased diode
  that starts from 0 V overflows in its first Newton iterate and the
  operating point goes through Xyce's continuation (acc_g10_op: 471
  Jacobians; the built-in diode with its junction limiting 6). Values at a
  solution are the same; only the iteration counts. Not changed in 1.1.5.
- C11. An assignment to a PARAMETER is accepted (check_declarations,
  1.1.2), although the LRM makes parameters constants: it is the Xyce/ADMS
  dialect of a hidden type parameter, and VBIC 1.3 (`vbic_1p3.va` l.266
  `parameter integer VBICtype ... (* hidden="yes" *)` under
  `__XYCE_COMPACT_MODELING__`, assigned at l.523-537) runs right with it.
  Refusing it refused that corpus file (found by the 1.1.2 generation
  scan). Keep, or refuse it outside the type variable named by
  `xyceTypeVariable`.

---------------------------------------------------------------------------

## D. Error messages

- D1. Some walkers still raise without a location: ginac_emitter.py
  `_walk_condition_eval` and the walker near the `case` handling, and GiNaC
  runtime errors (the metaprogram's own exceptions) (PYMS1-F). `` `line ``
  directives are not honoured by the line map. Fix: wrap with the statement's
  `loc` like the other walkers; teach preprocess.py's line map `` `line ``.
  Test: test_source_line_map, one case per walker.
- D2. `xyce_device_gen.py <va> --output <dir>` with a missing `<dir>` ends in
  a Python traceback (`FileNotFoundError`, 2026-10-25). Create the directory
  or print one line. Test: the CLI with a missing directory exits 1 with one
  line.
- D3. FIXED in 1.1.3 (CHANGES 1.1.3; test_ginac_determinism). It was
  filed as "eval.cpp is not byte-reproducible across processes (GiNaC's
  term order) ... Harmless for results": FALSE -- the term order is the
  summation order of the eval code, and Mextram bjtd504's DC point
  converged in 2 runs of 4 on 1.1.2 (PYMS3 skeptic). The cause is GiNaC's
  hash seeds (the address of each class's type name) under address-space
  randomization; the metaprogram now runs with randomization off.
- D6. The refusal of a string parameter used other than in `==` / `!=`
  against a string literal (ginac_emitter.py `_fold_string_compares`) and of
  a `` `define `` after a mid-line conditional (preprocess.py) name the
  construct; the first has no file:line (the condition text only, like the
  walkers of D1). Thread the statement's `loc` into `_preprocess_sys_funcs`.

---------------------------------------------------------------------------

## E. Caches and installation

- E1. The root-owned `/usr/local/bin/Xyce` and `/usr/local/lib/libxyce.so`
  (installed 2026-09-29) still have the device-shell cache race of P8
  (slot named by module only, written in place; measured 7 of 12 wrong
  pairs, silent) and ignore `PYMS_DEFINES` in the slot identity. The build
  tree's `/usr/local/src/xyce-build/src/libxyce.so` is fixed. Needs the user:
  reinstall (`make install` from the build tree, as root). Test:
  `XYCE=/usr/local/bin/Xyce python3 vae/tests/test_hdl_cache_concurrent.py`
  and `test_define_option.py`.
- E2. Old `pyms_<module>.so` slots (before the identity name) stay in every
  `PYMS_CACHE`; delete them once. Crash leftovers (`gen.<hash>.<pid>/`,
  `*.<pid>.tmp`, `*.build.<pid>/`) are never collected: sweep entries whose
  pid is not alive at cache open (PYMS1-G).
- E3. Two processes that miss the same slot at once both build it (one
  redundant build; the rename makes it harmless). A lock file per slot would
  save the time.
- E4. Two `.hdl` lines in one deck with two different files that declare
  the same module name register two devices of one name; the second shadows
  the first (PYMS1-G). Refuse with both file names.

---------------------------------------------------------------------------

## F. upstream (openvaf-r/VACASK)

Not PyMS defects; recorded so that a comparison is read right. VACASK and
openvaf-r are not patched from here: each upstream defect has a
self-contained failing case under `utils/PyMS/upstream/vacask/<name>/` (the
.va, a .sim in the form of VACASK's own `test/` directory with an embedded
`runme.py` that exits non-zero under `SIM_TEST=yes`, the command, the
observed output, the analytic expected value, a README). When VACASK fails on
a model or deck, that is not evidence against PyMS: judge PyMS by the
analytic value, by Xyce's ADMS build of the model, or by another engine.

- F1. DEFECT, `upstream/vacask/two_ddt_terms/`: a second, textually identical
  `ddt()` term in one contribution is dropped. `I(p,n) <+ V(p,n)/r +
  ddt(c*V(p,n)) + ddt(c*V(p,n))`, r = 1k, c = 50n, 1k load, ac at 1 kHz:
  VACASK 0.517745 at 8.51 deg (exactly the C = c answer); analytic (C = 2c)
  0.563358 at 14.70 deg; the control `ddt(2*c*V(p,n))` gives 0.563358 at
  14.70 deg on VACASK; Xyce through PyMS = analytic (PYMS1-D F05; re-measured
  2026-10-25).
- F2. Informational, `upstream/vacask/thermal_own_node_op/`: the self-heating
  resistor with its thermal port on its own node (PYMS1-I) is 1.69e-5 off the
  analytic op (V(out) 0.546593090 vs 0.546602348) with VACASK's default
  tolerances and exact with reltol=1e-12 vntol=1e-15 abstol=1e-18. Compare
  VACASK op results with a relative tolerance of 1e-4 or tighten its
  options.
- F3. Not an upstream defect: openvaf-r refuses BSIM-SOI 4.5 as shipped in
  `utils/ADMS/examples` (1280 errors; command `openvaf-r -o bsimsoi450.osdi
  utils/ADMS/examples/BSIM-SOI_4/bsimsoi4.5.0/bsimsoi450.va`, 2026-10-25),
  but the first errors are the Xyce/ADMS dialect of that copy: the
  module-header attribute (F5, l.103) and `@(initial_instance)` (l.1893,
  "expected 'initial_step' or 'final_step'"), which cascade into "'db' was
  not found in the current scope" for nodes that are declared (l.113). PyMS
  runs it: 1.62086677e-06 A at Vg = 1.2 V, Vd = 0.5 V, against ngspice's B4SOI
  1.62087e-06. References for these models: ngspice B4SOI (level 10), or
  Xyce's ADMS build (BSIM-SOI 4.5 level 70450, 4.6.1 level 70, 4.7 level
  70470, BSIM-CMG 107-111, PSP103 level 103) once Xyce is configured with
  `Xyce_ADMS_MODELS=ON` (OFF in /usr/local/src/xyce-build).
- F5. Not an upstream defect (the LRM puts attribute instances BEFORE
  `module`): openvaf-r refuses an attribute instance between a module's
  port list and its `;`, the place every Xyce ADMS example puts its Xyce
  attributes (`module PSP103VA(D, G, S, B) (* xyceModelGroup="MOSFET" ...
  *);`). Command:
  `openvaf-r -o psp103.osdi utils/ADMS/examples/psp103/psp103.va`; output:

      error: unexpected token '(*'; expected ';'
         --> .../psp103/psp103.va:48:1
      47 | module PSP103VA(D, G, S, B)
         |                            - expected ';'
      48 | (*
         | ^^ unexpected token
      error: could not compile `psp103.va` due to 1 previous errors

  and the same at `bsimcmg.va:25:35` for BSIM-CMG 111.2.1. With the header
  attribute removed both compile, and PSP103 then agrees with PyMS to the
  gmin current (I(VD) at Vd = 0.6 V, Vg = 0.6 V: VACASK -4.031075855e-04,
  Xyce -4.03107586e-04; the VACASK ring-benchmark model card, w = 10u, l =
  1u). PyMS reads these attributes (vamos spice.va_modules too).
- F4. Where PyMS follows the LRM and VACASK differs: bitwise operators on
  reals, `2 ** -1` on integers (0 in the LRM), `1000.0/0.0` (C1) (PYMS1-A).
- F7. A flow probe on a branch without contributions (A2): VACASK treats it
  as open.
- F6. DEFECT, `upstream/vacask/c_ammeter_double/`: the flow of a
  voltage-contribution branch read in another contribution is 0.
  `branch (p, x) am; V(am) <+ 0; I(x, n) <+ V(x, n)/1k; I(p, n) <+ I(am);`
  is two 1k paths (the ammeter's current added again from p to n), 500 ohm:
  out = 0.666667 in the 1 V / 1k divider. VACASK 0.500000000 (= I(am) read
  as 0); the control without the last line 0.5 on both. Xyce through PyMS
  1.1.0 0.666666667 (right); 1.1.1 0.5, WRONG -- it collapsed the probed
  short (CHANGES 1.1.2; the 1.1.1 text here said 1.1.1 was right, which the
  PYMS2 skeptic showed false); 1.1.2 0.666666667 (2026-10-26).
- F8. Open note, not investigated: Mextram 504 forward active (bjt504.va,
  Vce = 1.5 V, Vb = 0.6 ... 0.9 V) agrees with VACASK (the `$limit` calls
  replaced by their argument) to 0.03-0.125 % in Ic, not to the 1e-6 of
  the simple models (PYMS2 skeptic, 2026-10-26: 0.029-0.125 %). Cause not
  investigated (`$limit` is not it: PyMS passes its argument and the VACASK
  copy had the calls replaced by it). Candidates: the constants (`P_K`,
  `P_Q`: the shipped constants.vams against openvaf-r's), the default
  temperature behind `$vt` / `$temperature`, VACASK's default tolerances
  (F2: 1.7e-5 on a thermal resistor). First step: rerun VACASK with
  reltol=1e-12 vntol=1e-15 abstol=1e-18; if the gap stays, print both
  engines' Vt and internal node voltages at one bias.

---------------------------------------------------------------------------

## G. vamos follow-up (sv2ghdl, not this repository)

vamos pins the old PyMS behaviour in places; a vamos round should require
PyMS >= 1.1.0 (`vae.__version__`, `VERSION`) and lift the guards it carries
for the defects fixed here: card parameters on Xyce (E63:
`card_param_ignored_on_y`, `module_in_included_file`,
`level_card_binds_module` now give -0.0005, PyMS right), the include refusal
on a cold/warm cache, E125/E126/E64/E96, the trial2ei annotation case, and
the single-terminal source (E126, 1.0 not 1.001). The list of what fails
today is in the 1.1.0 integration report. 1.1.1 (PYMS2): vamos should
require >= 1.1.1 where it relies on the node collapse, model-group
attributes, value ranges or string parameters; E64's
`letter_device_diode_singular` pins 1.01e9 and the singular-matrix message
of a node-list mismatch that 1.1.1 removed (C8). 1.1.2 (PYMS3): require >=
1.1.2 where vamos relies on ammeters (a probed `V(br) <+ 0`), case
statements, `$simparam` in a condition or string compares in parameter
defaults; a deck that assigns an undeclared variable or contributes to an
undeclared node is now refused. 1.1.4 (PYMS5): E126's
`single_terminal_nan_va_leaves_its_node_at_0` pins V(A) = 0 for nan.va, the
1.1.3 failure of `$abstime` (B14); 1.1.4 gives V(A) = 1.0, the analytic
value at the operating point (t = 0), and E125's `no_abstime` expects the
refusal; both were in the 20 failing vamos tests of 1.1.3 already and stay
(the same 20 names on 1.1.4). vamos should require >= 1.1.4 where it relies
on $abstime, initial_step lists, analysis(), final_step / cross events,
parameter lists or a file with several modules.
