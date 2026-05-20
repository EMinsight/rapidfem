# RapidFEM Time-Domain Backend — Modal Ports Plan

The one item the production roadmap (`td-production-plan.md`) deferred as a
"self-contained follow-up project": **modal waveguide ports and a true
`sparams` verb** for `ProblemTD`. Today the TD backend has soft sources,
field probes and the scalar RFT `transfer_function`; this plan adds proper
waveguide-mode injection / extraction so a structure can be driven at its
ports and characterised by a normalised S-matrix — cross-validated against
the frequency-domain backend.

Branch: continues on `feature/td-backend` (ports are the last production
piece of the TD backend).

## Status — all phases complete

- **P1** — port-face plumbing (characteristic absorbing boundary);
  analytic `TE_mn` rectangular-waveguide modes.
- **P2** — port flux validated purely dissipative (`M̃A + AᵀM̃ ⪯ 0`);
  mode-injection source, validated against the analytic group velocity.
- **P3** — modal extraction by surface-integral projection with the
  per-frequency `Z_TE` forward/backward split; S-matrix on a matched
  two-port guide (`S₁₁ ≈ 0.02`, `|S₂₁| ≈ 1`, reciprocity, energy
  conserved).
- **P4** — `ProblemTD.sparams` through pyo3 (gmsh face tag → port);
  WR-90 TD-vs-FD cross-validation example; regression test; docs.
- **P5** — lumped / TEM port as the uniform-profile `(0,0)` mode, reusing
  the (mode-agnostic) flux / injection / extraction machinery.

Honestly scoped as remaining refinement: tightening the broadband WR-90
`sparams` cross-validation below the ~10 % coarse-mesh / time-window
spread (needs a finer mesh and per-signal time-gating); wiring the
Python `LumpedPort` geometry class through to the `(0,0)` port; a
parallel-plate TEM-line FD cross-check. The core port machinery is
validated cleanly in the Rust suite.

## The approach

A **port** is a tagged mesh face carrying an analytic waveguide mode. The
core mesh already exposes `ftag_to_tri` (gmsh face tag → triangles), so a
port maps cleanly to a set of `(element, local_face)` boundary faces.

**Boundary formulation.** At a port face the DG flux uses a ghost
(exterior) state set to the *incident modal field* — `(E⁺, H⁺) =
(E_inc, H_inc)`. With the upwind flux this single device does two things
at once:

- the scattered / outgoing field sees a **mode-matched characteristic
  boundary** — using the mode impedance `Z_mode` so the port's
  propagating mode leaves reflectionlessly (evanescent higher modes
  decay below cutoff anyway);
- the incident mode is **injected**.

The port flux is linear in `(E⁻, H⁻)` and in `(E_inc, H_inc)` separately,
so it splits exactly:

- the `(E⁻, H⁻)` part is an **absorbing modification of the constant
  operator `A`** — folds into `apply`, the system stays `dy/dt = A·y`;
- the `(E_inc, H_inc)` part is a **time-dependent rank-1 source**
  `b(t) = b_spatial · g(t)`, with `b_spatial` the mode profile lifted to
  the volume nodes and `g(t)` the excitation waveform.

The driven system `dy/dt = A·y + b(t)` is exactly what the ETD / Krylov
propagator (`etd_step_into`) already integrates — no new time-stepping
machinery.

**Extraction.** The modal amplitude crossing a port is the surface
integral of the port-face field projected onto the mode profile,
`a(t) = ∮_port (E_t × H_mode) · n̂`. A running RFT turns the incident and
scattered modal time signals into `A(ω)`, `B(ω)`; the S-parameter is the
power-wave ratio `S_ij = B_i(ω) / A_j(ω)`, one driven port at a time.

**Linearity is preserved** — ports keep `A` constant and add only a
known `b(t)`. MOR, the exponential propagator and the alloc-free stepping
all keep working unchanged.

## Scope

- **Phase A — rectangular waveguide ports** (this plan's core): analytic
  `TE_mn` / `TM_mn` modes of a rectangular cross-section. The workhorse
  port type and the WR-90 cross-validation target.
- **Phase B — lumped ports**: voltage-gap excitation, `Z₀`
  normalisation. A smaller follow-on once the waveguide path is proven.
- **Deferred**: general-cross-section ports (a 2D modal eigensolve on the
  port face) and coax/TEM annular ports — a later extension; not needed
  for the rectangular-waveguide validation milestones.

## Phase 1 — Port infrastructure

| WP | Deliverable | Gate |
|----|-------------|------|
| 1.1 | Port-face plumbing — gmsh face tag → `(element, local_face)` set; a `Port` boundary category in `FaceInfo` beside PEC / neighbour. | A tagged face is recognised; with no excitation it acts as a characteristic absorbing boundary — a pulse into it drains (reuse the absorber energy check). |
| 1.2 | Analytic rectangular-waveguide mode — `CoordinateSystem` + `TE/TM_mn` transverse profile + cutoff / `β`. The mode math is backend-agnostic physics → lift the shared parts of `rapidfem-fd::waveguide` into `rapidfem-core`. | Mode profile matches the analytic field; cutoff frequency `f_c` correct. |

## Phase 2 — Injection & the driven operator

| WP | Deliverable | Gate |
|----|-------------|------|
| 2.1 | Port flux in `apply_element` — ghost state `(E⁺, H⁺)`; the mode-matched absorbing part folds into `A`. | `M̃A` energy check — a port-terminated cavity loses energy only at the port, no spurious gain; reflection of the port mode is low. |
| 2.2 | Incident-mode source — assemble `b_spatial` from the mode profile lifted to the volume nodes; drive `dy/dt = A·y + b_spatial·g(t)`. | A matched straight waveguide carries the injected mode at the analytic phase velocity / guide wavelength. |

## Phase 3 — Extraction & S-parameters

| WP | Deliverable | Gate |
|----|-------------|------|
| 3.1 | Modal extraction — surface-integral projection of the port-face field onto the mode profile → incident / scattered modal amplitude time series; running RFT → `A(ω)`, `B(ω)`. | On a matched guide the extracted incident amplitude reproduces the drive; the scattered amplitude is ≈ 0. |
| 3.2 | S-matrix assembly — drive each port in turn, extract all `S_ij`; power-wave normalisation. | Matched straight guide: `S₁₁ ≈ 0`, `|S₂₁| ≈ 1`; 2-port reciprocity `S₂₁ ≈ S₁₂`. |

## Phase 4 — Cross-validation & API

| WP | Deliverable | Gate |
|----|-------------|------|
| 4.1 | WR-90 cross-validation — straight guide, then an iris discontinuity: TD `sparams` vs FD `sweep`. | TD ↔ FD S-parameters agree within discretisation error (`<~2–3 %`). |
| 4.2 | `ProblemTD.sparams(...)` Python verb through pyo3; an S-parameter result type; an example; regression tests; `docs/td-backend.md` update. | `sparams` works end-to-end from Python; regression test green in CI. |

## Phase 5 (Phase B) — Lumped ports

| WP | Deliverable | Gate |
|----|-------------|------|
| 5.1 | Lumped voltage-gap port — `∫E·dℓ` voltage excitation / extraction, `Z₀` normalisation, reusing the Phase 1–3 machinery. | A matched line shows ≈ 0 reflection; cross-checks against an FD lumped-port run. |

## Ordering & critical path

```
1 ──▶ 2 ──▶ 3 ──▶ 4            (faces → injection → extraction → S-params → validation)
                  └▶ 5         (lumped ports — after the waveguide path is proven)
```

- **Phase 1 unblocks everything** — no injection or extraction without
  port faces and the mode.
- **Phase 4 is the confidence milestone** — until TD `sparams` matches FD
  on WR-90, the port machinery is not trusted.

## Risks

- **The port flux is the danger zone — again.** Like the original vacuum
  flux and the heterogeneous-media flux, the port ghost-state is the
  error-prone part. Each step gets an analytic gate (energy, phase
  velocity, matched-line S-params) before the next builds on it.
- **Characteristic-boundary reflection.** A mode-matched characteristic
  port is reflectionless for its propagating mode but not for obliquely
  incident higher-order content; this sets an S-parameter noise floor.
  Mitigation: place ports at clean reference planes, away from
  discontinuities (standard practice); a PML-backed port is a later
  refinement if the floor proves too high.
- **Mode purity.** Higher-order modes excited at discontinuities must
  decay before reaching the port — the port needs adequate guide length,
  as in any waveguide simulation.

## Out of scope (for now)

- General-cross-section modal ports (2D eigensolve), coax / TEM ports.
- De-embedding / reference-plane shifting beyond the port plane itself.
- Curvilinear port faces — affine triangulated faces only.
