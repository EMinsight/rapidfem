# TD Backend Session Audit

What was built this session, what is verified, and what is honestly a
work-in-progress.

## Scope

The session set out to (1) extend the TD port catalogue, (2) build a
compact macromodel pipeline (`docs/td-macromodel-plan.md`), and (3) a
Python API for both. 18 commits land on `master`, ~7300 lines of code +
docs net add across 24 files, all 91 Rust unit tests in
`rapidfem-td --lib` green.

## What ships and is gated by a green test

### Ports (extension to `docs/td-ports-plan.md`)

| WP | Deliverable | Gate test | Status |
|----|-------------|-----------|--------|
| C1 | `CoaxPort` + `PortMode` enum, coax-from-mesh-tag | `coax_port_carries_a_matched_tem_wave` | ✓ green |
| C2 | `PeriodicSpec` + matcher, periodic DGTD BCs | periodic energy / coherence tests | ✓ green |
| C3 | `FloquetPort` (TE/TM plane wave on unit cell) | `floquet_*` tests | ✓ green |
| ABC| `PortSpec::absorbing_from_mesh_tag` (Silver-Müller) | covered via WP-A regression | ✓ green |
| PEC| `PecSpec` for internal-PEC plates (trace, ground sheet) | not yet gated, covered indirectly | ⚠ works, no dedicated gate test yet |
| Z0 | `RectPort.z0` reference impedance for lumped (0,0) | covered via existing tests, default z0=1 | ⚠ works, no dedicated gate test yet |

### Macromodel pipeline (`docs/td-macromodel-plan.md`)

| WP | Deliverable | Gate test | Status |
|----|-------------|-----------|--------|
| M1 | Block-Krylov MIMO build (`build`) | `macromodel_matched_two_port_reproduces_s_parameters` (r=350: \|S11\|≤0.15, \|S21\|≥0.96, reciprocity exact) | ✓ green |
| M2 | Sweep + Touchstone export | `macromodel_sweep_matches_pointwise_evaluate`, `macromodel_touchstone_writer_roundtrip` | ✓ green |
| M3 | SPRIM passivity + SVD clip | `macromodel_sprim_passivity_matched_two_port` (sigma_raw 1.06 vs plain 1.10, sigma_passive=1.0 hard) | ✓ green |
| M4 WP 4.1 | Push-r covers half-octave | `macromodel_wp41_push_r_covers_wider_band` (r=500) | ✓ green |
| M4 WP 4.2 | Chebyshev band-pass filter | `macromodel_polyfilter_runs_and_returns_finite_smatrix` (sanity only, accuracy claim deferred) | ⚠ scaffold |
| M4 WP 4.3 | Shift-invert via matrix-free GMRES | `macromodel_shift_invert_concentrates_near_target`, `macromodel_multi_shift_unions_per_shift_bases` | ✓ green (sanity gates only) |
| M5 | Python `ProblemTD.macromodel()` verb + RFIC example | smoke test (`td_macromodel_smoke.py`) | ✓ green on validated geometry |
| M6 | GPU macromodel build (apply-closure pattern) | `macromodel_gpu_build_matches_cpu_within_mixed_precision_tol` (CPU/GPU diff 2.6e-6 vs budget 5e-3) | ✓ green |
| M7 | Parametric MOR | deferred per user instruction | – |

### Python API surface

- `ProblemTD.macromodel(r, sprim=False, shift_freq_hz=None, shift_freqs_hz=None, n_shift_steps=2)`
  → `TdMacroModel` with `evaluate(f_Hz, passive=)`, `sweep(freqs_Hz)`, `to_touchstone(...)`
- `rf.ABC(...)` now wired through to TD (was FD-only)
- `rf.PEC(...)` on internal plates now wired through to TD (was ignored)
- `rf.LumpedPort(..., z0=50.0)` now honoured by TD (was hardcoded to 377Ω)
- `ProblemTD.sparams` ABC-contamination panic fixed
- Diagnostic getters exposed: `port_has_mode`, `port_n_faces`, `port_n_interior_faces`

### Bugs found and fixed this session

| Bug | Symptom | Fix |
|-----|---------|-----|
| `ProblemTD.sparams` panicked on ABC faces | `port has no mode for extraction` panic | filter to modal ports in `sparams` |
| Lumped port wire `z0` was dropped | physical 50Ω was always treated as 377Ω | thread `z0` through Python → pyo3 → PortSpec → RectPort |
| Internal PEC plates (trace, ground sheet) silently ignored | microstrip transient \|S21\| = 0 | new `PecSpec`, retag both element-sides as boundary so PEC ghost-state applies |

## What does not yet work end-to-end

| Use case | What happens | Why |
|----------|--------------|-----|
| Microstrip 2-port transient via lumped ports | \|S11\| matches FD within 1.4%, \|S21\| = 0.25 vs FD 0.88 | The uniform `(0,0)` lumped-port mode profile is structurally a bad match for a microstrip's concentrated quasi-TEM mode — most injected energy ends up in evanescent / higher-order modes the ABC absorbs. Same on both flux types (upwind dissipates, central excites null-modes). The macromodel inherits this. **The fix is a real wave port: a 2D Maxwell eigensolve on the port face for the actual TEM profile.** That is an architectural addition, not a tuning issue. |
| RFIC spiral inductor via lumped ports | same as above | same root cause |
| RFIC macromodel μs-per-frequency claim | timing-wise correct on tiny resonant cases (`td_macromodel_bench.py`: r=60, 58 μs / eval, 58 ms / 1000-pt sweep, 17000× vs PARDISO-FD) | but the S-parameters from those macromodels reflect the broken lumped-port physics, so the speed gain has no business value yet |

## What does work end-to-end

| Use case | Evidence |
|----------|----------|
| Rectangular waveguide modal ports (TE_mn) | original P1-P5 gates from the ports plan, plus M1-M4 macromodel gates on a matched 2-port |
| Coax TEM ports | C1 gate (`coax_port_carries_a_matched_tem_wave`) |
| Periodic DGTD boundaries | C2 gate (energy drift 3e-14, forward-wave coherence 0.985) |
| Floquet plane-wave ports | C3 gate (transmission 0.996, reflection ~machine eps) |
| Single-port + absorbing-end-port lumped TEM box | `lumped_port_carries_a_dispersionless_tem_wave` gate |
| Two-active-lumped-ports on a vacuum box (port-port coupling) | `two_lumped_ports_carry_tem_between_them` (10% modal projection at far port) |
| Macromodel build → sweep → Touchstone | smoke test on a 5mm WR-90 setup green, S-matrix consistent with sweep, σ_max passive clipped to 1.0 exactly |
| GPU macromodel build | M6 gate, agrees with CPU within mixed-precision budget |

## What is scaffolding (compiles + runs, accuracy claim deferred)

| Item | What is honest |
|------|----------------|
| `MacroModel::build_polyfilter` (M4 WP 4.2) | Two-stage Chebyshev high-pass + low-pass composition works arithmetically; sanity test confirms it produces finite reciprocal S. **The "matches M1 accuracy at smaller r" claim was withdrawn after diagnostics showed it is a tool for *resonant* macromodels (well-separated in-band eigenmodes), not propagating guides.** No real-world geometry yet exercises this path. |
| `MacroModel::build_shift_invert` (M4 WP 4.3 single-shift) | Matrix-free complex GMRES + Krylov chain works arithmetically. On the matched 2-port gives \|S11\|≈0.18, \|S21\|≈0.83 at r=24 (vs M1's r=350 for the same accuracy). On a non-resonant microstrip line the single-shift iteration collapses to the dominant in-band mode after a few steps — a known weakness. |
| `MacroModel::build_multi_shift` (M4 WP 4.3 broadband) | Multi-shift distributes basis across band; gate test passes (4 shifts produce r=32 vs single-shift's r=14). On the microstrip improves \|S11\| error from 0.65 → 0.56 but \|S21\| stays at 0.000 — the bottleneck is the underlying TD lumped-port physics, not the basis. |

## Python examples (status by file)

| File | Purpose | Status |
|------|---------|--------|
| `td_macromodel_smoke.py` | end-to-end Python pipeline sanity on small WR-90 | ✓ runs green, validates evaluate/sweep/touchstone/passive |
| `td_macromodel_bench.py` | timing comparison on RFIC symmetric inductor | ✓ runs, performance numbers honest, S-parameters reflect known lumped-port limitation |
| `td_microstrip_macromodel.py` | macromodel on microstrip vs FD | ✓ runs, S-parameters off due to lumped-port limit |
| `td_microstrip_transient_sparams.py` | transient `sparams` on microstrip vs FD | ✓ runs, isolates that the issue is TD operator level not macromodel |
| `td_lumped_port_diag.py` | inspects port_source / port_modal_projections / spectral propagation | ✓ runs, exposes the diagnosed root cause |
| `td_two_lumped_box.py` | minimal two-lumped-port box for python reproduction | ✓ runs |
| `td_two_lumped_box_trace.py` | step-by-step modal projection trace | ✓ runs |
| `td_gmres_micro.py` | scaling test for shift-invert GMRES | ✓ runs |
| `td_rfic_spiral_from_json.py` (in `rapidfem.examples`) | end-to-end RFIC pipeline | ✓ runs, S-parameters reflect lumped-port limit |

## Honest next-step priorities

In rough order of value × effort ratio for unlocking the RFIC market case:

1. **Wave port: 2D Maxwell eigensolve on port face** for the actual TEM mode profile. Solves the lumped-(0,0)-uniform-profile mismatch at its root. Once the port carries the true microstrip TEM mode, both transient and macromodel paths give physically meaningful S-parameters. **One focused day of work.** This is the missing piece.
2. Dedicated gate tests for the new internal-PEC and Z0 paths (currently covered indirectly). A few hours.
3. M4 WP 4.2 (polyfilter) accuracy on a resonant geometry — a small cavity-coupled-port macromodel that has well-separated in-band modes, where the Chebyshev band-pass should pay off.
4. M7 parametric MOR (deferred per user instruction this session).
