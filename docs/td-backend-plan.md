# RapidFEM — Time-Domain (DGTD) Backend — Implementation Plan

## Goal

Turn RapidFEM into a **two-backend FEM solver** in a single repo, single name:

- **Frequency domain** — the existing Nédélec-FEM solver. Unchanged.
- **Time domain** — a new DGTD solver. Discontinuous Galerkin is an FEM
  family member, so "RapidFEM" stays accurate.

The Geometry / Materials / Physics API is shared and solver-agnostic — the
`(mesh, TOML)` hand-off is the seam. The split happens at `Problem`:

- `ProblemFD` — analysis tool: `sweep`, `eigenmode`, `farfield`.
- `ProblemTD` — model-export tool with progressive disclosure: `sparams`,
  `transient`, `stepper`, `ode`, `state_space`, `operators`, `reduce`.
- `Problem` stays as an alias for `ProblemFD` (backward compatibility).

The TD backend rests on one fact: linear Maxwell, DG-semidiscretised
(method of lines), is a linear time-invariant system `ẏ = A·y + B·u(t)`
with **constant sparse `A`**. The Jacobian *is* `A`; an exponential
integrator (ETD, Krylov) handles the stiff homogeneous part exactly.

## Guiding principles

1. **Refactor first, build second.** Phases 1–2 restructure the repo with
   *zero behaviour change* — FD stays bit-identical, all tests green.
   Only then is the TD slot filled. Never debug a refactor and a new
   solver at the same time.
2. **FD is never touched after Phase 1.** Every later phase is additive.
3. **Validate before building on top.** Each numerical phase has a gate
   (analytic comparison / FD cross-check) that must pass before the next.
4. **One repo, one name, two crates behind one Python API.**

## Repo target structure

```
rapidfem/
  Cargo.toml                  [workspace]
  crates/
    rapidfem-core/   mesh, topology, quadrature, config (TOML schema),
                     constants, materials (data), interp, touchstone, vtk
    rapidfem-fd/     Nédélec basis, assembly, sparse solvers, sweep,
                     eigenmode, farfield, coefficients   (+ build.rs)
    rapidfem-td/     DG operators, flux, RHS, state-space, ETD/Krylov, MOR
  python/            pyo3 cdylib — depends on -fd and -td, exposes _native
    python_src/rapidfem/
      geometry.py  materials.py  physics.py  excitation.py  io.py  ...
      problem/  __init__.py  _base.py  fd.py  td.py  _td_models.py
```

---

## Phase 1 — Workspace refactor (behaviour-neutral)

| WP | Goal | Gate |
|----|------|------|
| 1.1 | Convert root to a Cargo workspace; create `crates/`. | — |
| 1.2 | Extract `rapidfem-core` (`git mv` mesh, mesh_io, quadrature, config, constants, materials, interp, touchstone, vtk_export). New `Cargo.toml` + `lib.rs`. | — |
| 1.3 | Extract `rapidfem-fd` (remaining `src/` + `build.rs`); features `pardiso`/`parallel`; fix `use` paths to `rapidfem_core::`. | — |
| 1.4 | Stub `rapidfem-td` — empty `lib.rs`, `Cargo.toml` depends on `-core`. | — |
| 1.5 | Re-point `python/Cargo.toml` to `rapidfem-fd` (+ `-core`); `_native` builds unchanged. | — |
| 1.6 | `cargo build --workspace`, `cargo test`, `maturin develop`, run examples, `cargo test --release` validation suite. | **All FD tests green; results bit-identical.** |

Deliverable: restructured repo, FD untouched. Commit.

## Phase 2 — Python `problem/` package (behaviour-neutral)

| WP | Goal | Gate |
|----|------|------|
| 2.1 | `problem.py` → `problem/` package: `_base.py` (geometry + `(mesh,TOML)`), `fd.py` (`ProblemFD`), `__init__.py` (`Problem = ProblemFD`). | — |
| 2.2 | `rapidfem/__init__.py` re-exports `Problem`, `ProblemFD`. | — |
| 2.3 | Existing examples (`rf.Problem(g).sweep(...)`) run unchanged. | **API backward-compatible.** |

Phases 1–2 = **Stage 1**: seams created, no new solver code. Commit & checkpoint.

## Phase 3 — TD spatial operator (`rapidfem-td`)

The heavy new work — the DG discretisation.

| WP | Goal | Gate |
|----|------|------|
| 3.1 | Mesh topology for DG — face/cell adjacency, normals, orientations (in `-core` `topology.rs`). | — |
| 3.2 | DG reference element — nodal basis (Warp&Blend nodes on the reference tet), Vandermonde, mass `M`, derivative `D`, lift `L`; order-`p` parametrised. Reuses `quadrature.rs`. | — |
| 3.3 | Geometric factors — per-element affine Jacobian, metric terms, face scaling. | — |
| 3.4 | The RHS operator — volume curl + upwind numerical flux + material scaling + BC-as-flux (PEC/PMC/ABC) + source. Matrix-free `A·v`. | — |
| 3.5 | Validate the operator — eigenvalues of `A` vs analytic cavity modes; p-refinement convergence study. | **Cavity resonances match analytic.** |

## Phase 4 — TD time integration (exponential propagator)

| WP | Goal | Gate |
|----|------|------|
| 4.1 | Krylov machinery — Arnoldi/Lanczos for `e^{Ah}·v` (matrix-free, reuses 3.4), φ-functions via the augmented-matrix trick, adaptive Krylov dimension. | — |
| 4.2 | ETD stepper — per-step `y ← e^{Ah}y + φ-input-terms`; piecewise-poly input quadrature; embedded-error adaptive step size. | — |
| 4.3 | Excitation / source — `b(t)` from port pulses; the `Excitation` concept. | — |
| 4.4 | Validate — energy conservation (lossless); transient vs analytic; S-params of WR-90 vs the FD backend. | **TD S-params match FD within tolerance.** |

## Phase 5 — TD state-space + native interface

| WP | Goal |
|----|------|
| 5.1 | Assemble `A` explicitly as a sparse matrix from the RHS operator; `B`, `C`, `D`. |
| 5.2 | Native API — `_native` TD: construct from `(mesh,TOML)`; expose `rhs`, `jacobian`, `state_space`, `stepper`, `step`, dense output. pyo3 binding (`python/src/td.rs`). |
| 5.3 | TD result types in `io.py` (transient fields, S-params). |

## Phase 6 — `ProblemTD` Python API

| WP | Goal |
|----|------|
| 6.1 | `problem/td.py` — `ProblemTD`: `sparams` (RFT on-the-fly), `transient`, `stepper`, `ode`, `state_space`, `operators`. |
| 6.2 | `_td_models.py` — `Stepper`, `ODEModel`, `StateSpace` objects (progressive disclosure). |
| 6.3 | `excitation.py` — `GaussianPulse` etc. |
| 6.4 | Wire `_native` TD ↔ `ProblemTD`; observer hook so `sparams` is built on `stepper`. |

## Phase 7 — Model order reduction

| WP | Goal | Gate |
|----|------|------|
| 7.1 | Krylov moment-matching MOR (reuses Phase-4 Arnoldi) → `(Â,B̂,Ĉ)`. | — |
| 7.2 | Modal MOR — project via the existing FD `eigenmode` solver. | — |
| 7.3 | `ReducedModel` — same interface as `ProblemTD`, order `r`; `ptd.reduce(...)`. | — |
| 7.4 | Validate — reduced vs full transfer function. | **ROM matches full model to tolerance.** |

## Phase 8 — Adaptivity

| WP | Goal |
|----|------|
| 8.1 | Formalise adaptive time-stepping (embedded ETD error). |
| 8.2 | Residual-based a-posteriori error estimator for the reduced model (O(r) cost). |
| 8.3 | Greedy / online-adaptive MOR — basis enrichment driven by the estimator; hot-swap fidelity. |

## Phase 9 — Ecosystem, docs, examples

| WP | Goal |
|----|------|
| 9.1 | PathSim/fastsim integration — `ODEModel` as a co-simulation block; a coupled EM↔system example. |
| 9.2 | TD examples — `examples/td_*.py` (transient, S-params, ODE export, MOR). |
| 9.3 | Docs — README "frequency- and time-domain FEM"; add `rapidfem.problem` to the curated API-reference module list; TD quickstart snippet. |
| 9.4 | `bridge.py` / TOML schema — additive TD fields. |

## Phase 10 — Packaging & CI

| WP | Goal |
|----|------|
| 10.1 | maturin / `pyproject.toml` for the workspace + two crates; wheel-size check. |
| 10.2 | CI — `cargo test --workspace`; TD validation in the matrix. |

---

## Ordering & critical path

```
1 → 2  ──(Stage 1: refactor, gated, behaviour-neutral)
     → 3 → 4 → 5 → 6  ──(Stage 2: build TD, critical path)
                  ├→ 7 → 8   (MOR + adaptivity, after 4+5)
                  └→ 9 → 10  (integration + packaging, last)
```

- **Phases 1–2 block everything** — do them first, fully gated.
- **Phase 3 is the foundation**; its gate (3.5) must pass before Phase 4 —
  never build the time integrator on an unvalidated spatial operator.
- **Phase 4 gate (4.4)** — TD↔FD S-param agreement — is the project's
  central correctness milestone.
- Phases 7–8 depend on the Krylov machinery (4.1) and the state-space
  assembly (5.1). Phases 9–10 are last.

## Key risks

- **DG operator correctness** — the bulk of the difficulty. Mitigated by
  the 3.5 gate (analytic cavity modes) before any time integration.
- **State-vector size** — `~6·Np·K` can reach 10⁷. The exposed RHS must be
  fast (matrix-free matvec); the ETD/Krylov path keeps it non-stiff.
- **Dispersive media / PML** — auxiliary-differential-equation
  formulations; they enlarge the state but keep `A` constant. Plan as a
  follow-up within Phase 3/4, not a blocker for the first lossless gate.
- **Refactor scope creep** — Phases 1–2 must stay strictly behaviour-
  neutral; resist "while we're here" changes.
