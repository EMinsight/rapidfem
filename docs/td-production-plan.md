# RapidFEM Time-Domain Backend — Production Roadmap

Follow-up to `td-backend-plan.md`. That plan delivered a *first validated
implementation* of the DGTD backend (vacuum Maxwell, structured box
cavities, validated to π√2 at 1e-4). This roadmap takes it to **production
level**: arbitrary geometry, real materials, absorbing boundaries, ports,
and systematic cross-validation against the frequency-domain solver.

## Where we are

**Phases 1–6 complete** (see commit log, `prod P1.x`–`P6.x`):

- **P1** — runs on arbitrary unstructured gmsh meshes; convergence and
  conditioning verified.
- **P2** — heterogeneous, lossy and diagonal-anisotropic materials, wired
  from the geometry API.
- **P3** — soft sources, field probes, `driven_transient`, second-order
  ETD source step, `GaussianPulse` excitation.
- **P4** — physical-units (`c`) mapping; `ProblemTD.resonances()`; FD↔TD
  cross-validation (0.04 % agreement on a shared cavity).
- **P5** — graded matched absorbing layer; magnetic conductivity; Debye
  dispersive material via ADE, validated against the analytic `ε(ω)`.
- **P6** — parallel matrix-free `apply` (rayon, all cores); element-wise
  sparse `A` assembly with no densifying (`O(nnz)`, scales to 10⁵ DOF);
  zero-copy numpy in/out for `apply`/`step`/`state_space`; benchmark.

Remaining: P7 (API completion / MOR), P8 (hardening, examples, docs).

Note: a full modal-port `sparams` verb (P7.1) needs waveguide-mode
injection/extraction — soft sources + probe RFT give the scalar transfer
function today; modal ports are the larger remaining item.

## What "production level" means here

1. Runs on arbitrary unstructured tetrahedral meshes from the geometry API.
2. Heterogeneous, lossy, anisotropic and dispersive materials.
3. Open-domain problems via PML absorbing boundaries.
4. Ports and excitation — can compute S-parameters and drive real problems.
5. Cross-validated against the frequency-domain backend on real structures.
6. Scales to 10⁵–10⁶ degrees of freedom.

## Guiding principles

- **Cross-validation is the production gate.** Once ports exist, every
  feature is checked by running the same structure in FD and TD.
- **The flux is the danger zone — again.** Heterogeneous-media and PML
  change the numerical flux; each gets an analytic gate before being built on.
- **Dense-assembly validation does not scale.** Beyond Phase 1, correctness
  rests on FD cross-checks, convergence studies and structural (energy)
  checks, not dense eigensolves.
- **Linearity is preserved.** Materials, losses, dispersion and PML all keep
  the system linear with constant `A` (dispersion/PML via auxiliary state) —
  the Krylov/ETD propagator and MOR keep working unchanged.

---

## Phase 1 — General geometry & mesh robustness

Run on arbitrary unstructured tet meshes, not just structured boxes.

| WP | Deliverable | Gate |
|----|-------------|------|
| 1.1 | TD native constructor from gmsh `.msh` bytes (reuse `load_mesh`); `ProblemTD` accepts a `Geometry`. | A meshed `Geometry` builds a `MaxwellOperator`. |
| 1.2 | DG operator validated on irregular/skewed tets — geometric factors, flux on non-uniform faces. | Cavity eigenmode on an **unstructured** mesh matches analytic. |
| 1.3 | h- and p-refinement convergence study. | Error decreases at the theoretical `O(h^{p+1})` rate. |
| 1.4 | Warp&Blend interpolation nodes replacing equispaced. | Operators re-validated; conditioning healthy through `p = 6`. |

## Phase 2 — Materials (non-dispersive)

Heterogeneous `ε`, `μ`, `σ`, anisotropic tensors.

| WP | Deliverable | Gate |
|----|-------------|------|
| 2.1 | Per-element `ε`, `μ` in the volume term; **impedance-aware upwind flux** (material-interface Riemann solver, `Z⁻`/`Z⁺`). | Dielectric-loaded cavity — resonance shift matches analytic. |
| 2.2 | Conductivity `σ` — the `-σ/ε·E` loss term; `A` stays constant. | Lossy cavity — `Q`-factor matches analytic. |
| 2.3 | Anisotropic `ε`/`μ` tensors in the volume term. | Anisotropic slab vs analytic. |
| 2.4 | Material data flows from the existing physics API into the TD TOML. | Partially-filled waveguide — cutoff matches FD. |

## Phase 3 — Ports, excitation & observables

Excite and measure — the prerequisite for real examples.

| WP | Deliverable | Gate |
|----|-------------|------|
| 3.1 | Source term `b(t)` wired from an excitation; `Excitation` object (Gaussian / modulated pulse); ETD `φ₂` term for 2nd-order time-varying sources. | ETD with a time-varying source vs analytic. |
| 3.2 | Waveguide ports (modal injection + extraction) and lumped ports. | Single-port reflection of a matched load ≈ 0. |
| 3.3 | Field probes; S-parameter extraction by on-the-fly RFT (+ FFT); `sparams` / `transient` turnkey verbs on `ProblemTD`. | WR-90 — TD S-parameters match analytic; 2-port reciprocity. |

## Phase 4 — FD ↔ TD cross-validation

The production-confidence gate.

| WP | Deliverable | Gate |
|----|-------------|------|
| 4.1 | Shared harness — one geometry, both backends, S-parameter comparison. | Harness runs FD + TD on a common structure. |
| 4.2 | Cross-validate on existing structures: coax step, iris filter, stepped waveguide. | TD vs FD S-parameters agree within discretisation error. |
| 4.3 | Convergence + regression — h/p studies; tolerances locked as regression tests. | TD↔FD agreement `<~1 %` at adequate resolution on ≥3 structures. |

## Phase 5 — ADE infrastructure & PML

Open-domain problems and dispersive materials — both via auxiliary
differential equations.

| WP | Deliverable | Gate |
|----|-------------|------|
| 5.1 | ADE framework — auxiliary per-node state; the augmented constant `A`. | Augmented operator still energy-consistent where expected. |
| 5.2 | CFS-PML absorbing boundary layer. | Pulse into PML reflects `< −40 dB`, normal + oblique incidence. |
| 5.3 | Dispersive materials (Debye, Drude) via ADE — reuses 5.1. | Debye slab matches analytic dispersion / FD. |
| 5.4 | A radiating example (dipole / patch) with PML termination. | Far-field / input impedance vs FD. |

## Phase 6 — Performance & scale

Production-size meshes. Can start once Phase 1 lands; runs parallel to 2–5.

| WP | Deliverable | Gate |
|----|-------------|------|
| 6.1 | Parallel matrix-free `apply` (rayon), per element; dense assembly off the hot path. | Correctness unchanged; near-linear thread scaling. |
| 6.2 | Element-wise sparse `A` assembly; memory-layout pass. | `A` for a 10⁵-DOF mesh assembles without densifying. |
| 6.3 | Benchmarks — apply throughput, propagation cost, scaling study. | A 10⁵–10⁶-DOF problem runs within a sane time/memory budget. |

## Phase 7 — `ProblemTD` API completion & MOR maturity

The full progressive-disclosure API in Python.

| WP | Deliverable | Gate |
|----|-------------|------|
| 7.1 | `ProblemTD` from a `Geometry`; `sparams`, `transient(excitation)`, `stepper`, `ode`, `state_space`, `reduce` wired through pyo3. | Each verb works end-to-end from Python. |
| 7.2 | `ReducedModel` exposed to Python; modal MOR (reuse the FD eigensolver). | Reduced model reproduces the transfer function. |
| 7.3 | TD result types — field/S-parameter objects; VTK field-animation export. | Result objects integrate with the notebook UI. |

## Phase 8 — Robustness, regression suite & docs

| WP | Deliverable | Gate |
|----|-------------|------|
| 8.1 | Validation + cross-validation suite wired into CI as regression tests. | CI green including TD↔FD regression. |
| 8.2 | TD examples: waveguide, iris filter, dielectric resonator, a PML radiation problem. | Examples run and produce expected results. |
| 8.3 | Docs — TD section, method notes, API reference (`rapidfem.problem`). | Docs published; TD documented alongside FD. |

---

## Ordering & critical path

```
1 ──▶ 2 ──▶ 3 ──▶ 4            (critical path: geometry → materials → ports → cross-validation)
          └▶ 5                (ADE/PML — needs excitation from Phase 3)
   └▶ 6                        (performance — parallel, after Phase 1)
            2,3,5 ──▶ 7 ──▶ 8  (API completion, then hardening)
```

- **Phase 1 unblocks everything** — no complex examples without general meshes.
- **Phase 3 (ports) gates Phase 4** — cross-validation needs S-parameters.
- **Phase 4 is the central confidence milestone** — until TD matches FD on
  real structures, the backend is not production-trusted.
- **Phase 5 (ADE/PML)** needs Phase 3's excitation to be validated.
- **Phase 6** is independent — start after Phase 1, run alongside 2–5.

## Risks

- **Heterogeneous-media flux** — the material-interface Riemann solver is the
  error-prone part, exactly like the original vacuum flux. Phase-2 analytic
  gates (dielectric cavity) catch it before anything builds on it.
- **PML stability** — CFS-PML can develop late-time instabilities; needs
  careful formulation and a long-time stability check.
- **Validation that scales** — dense eigensolves end at Phase 1; production
  correctness rests on FD cross-checks + convergence + energy checks.
- **Performance vs correctness** — the parallel/sparse rewrite (Phase 6) must
  not perturb results; gated by re-running the full validation suite.

## Explicitly out of scope (for now)

- **Curvilinear / isoparametric elements** — affine tets with adequate mesh
  refinement is the pragmatic production approach; revisit only if curved
  geometry accuracy proves insufficient.
- **Nonlinear materials** — the backend stays linear (constant `A`).
- **PathSim / fastsim co-simulation block** — deferred by request.
