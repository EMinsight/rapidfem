# RapidFEM — Time-Domain DGTD Backend

RapidFEM has two FEM backends behind one geometry / material / physics API:

- **`ProblemFD`** — the frequency-domain solver (second-kind Nédélec edge
  elements, complex-symmetric sparse linear algebra). Geometry in,
  S-parameters out.
- **`ProblemTD`** — the time-domain solver documented here.

`Problem` is a backward-compatible alias of `ProblemFD`.

## What the time-domain backend is

`ProblemTD` discretises Maxwell's curl equations in space with a **nodal
discontinuous Galerkin** method on tetrahedra (DG *is* a finite-element
method — a discontinuous one). The result is an explicit linear ODE

```
dy/dt = A·y               (+ b(t) for a driven source)
```

with a constant, sparse operator `A`. `ProblemTD` is a **model-export
tool**: where `ProblemFD` answers "what are the S-parameters", `ProblemTD`
hands you that ODE at every level of abstraction — the right-hand side,
the verbatim sparse operator, an exponential stepper, a turnkey transient,
or a model-order-reduced surrogate.

Use the time-domain backend when you want broadband behaviour from a
single run, the explicit state-space model for control / system
identification, or a reduced-order model of a structure.

## The method

**Spatial discretisation.** Nodal DG on tetrahedra. The reference element
carries a Lagrange basis of order `p` on equispaced nodes, built through
the monomial Vandermonde so the mass, differentiation and lift operators
are assembled in closed form. The numerical flux blends central
(`flux="central"`, exactly energy-conserving) and upwind (`flux="upwind"`,
additionally dissipates the discontinuous spurious modes). PEC walls enter
as a ghost-state flux.

**State layout.** `y[(e·Np + node)·6 + comp]`, with `comp` 0..3 the
electric field and 3..6 the magnetic field.

**Time integration.** The semi-discrete system is linear with a constant
`A`, so the exact propagator is the matrix exponential. RapidFEM advances
it with a **matrix-free Krylov / ETD exponential integrator** — `exp(hA)·y`
via Arnoldi, never forming `A`. Because the step is *exact* for the linear
homogeneous system at any `h`, the time step is set by output cadence, not
by a CFL stability limit.

**Linearity is preserved everywhere.** Materials, Ohmic and magnetic
losses, matched absorbers and dispersive (Debye) media all keep the system
linear with a constant `A` — dispersion via an auxiliary differential
equation (ADE) that augments the state. The exponential propagator and the
model-order reduction therefore work unchanged across all of them.

## The `ProblemTD` API

### Construction

```python
import rapidfem as rf

# From a meshed geometry (arbitrary unstructured tetrahedral meshes):
g = rf.Geometry(maxh=...)
...                                 # build geometry, attach materials
g.mesh()
ptd = rf.ProblemTD(g, order=2, flux="upwind")

# Or directly on a structured box cavity — handy for validation:
ptd = rf.ProblemTD.box(size=(1, 1, 1), cells=(2, 2, 2), order=2)
```

`order` is the DG polynomial order; `flux` is `"upwind"` or `"central"`;
`c` is the speed of light in the mesh's length units (it maps operator
time/frequency to physical SI units — default SI metres for a geometry,
normalised `c = 1` for `box`).

### Progressive-disclosure verbs

The API exposes the model at every level — pick the abstraction you need.

| Verb | Returns | Use |
|------|---------|-----|
| `rhs(y)` | `dy/dt = A·y` | the ODE right-hand side, matrix-free |
| `jacobian()` / `state_space()` | sparse `A` (`scipy.sparse.csr_matrix`) | the verbatim state-space operator |
| `ode()` | `TdODE` | handoff for an external integrator (`scipy.integrate.solve_ivp`) |
| `step(y, h)` | advanced state | one exact exponential step |
| `stepper(dt)` | `TdStepper` | a reusable `dt`-bound propagator |
| `transient(y0, dt, steps)` | trajectory `[steps+1, n_dof]` | turnkey free propagation |
| `driven_transient(source, waveform, probes, ...)` | `(times, responses)` | soft source + field probes |
| `transfer_function(source, probe, pulse, ...)` | `(freqs, H)` | scalar frequency response by RFT |
| `sparams(freqs, dt, steps)` | `(freqs, S)` | modal-port scattering matrix |
| `probe_dof(point, field, comp)` | DOF index | place a source or probe |
| `reduce(start, dim)` | `TdReducedModel` | Krylov model-order reduction |
| `resonances(n)` | frequencies (Hz) | cavity modes from the spectrum |
| `export_vtk(states, path)` | `.pvd` path | VTK field-animation export |

### Examples

```python
# Low level — the ODE and the verbatim operator
dy = ptd.rhs(y)
A  = ptd.state_space()                       # sparse, no densifying

# Export to an external integrator
ode = ptd.ode()
from scipy.integrate import solve_ivp
sol = solve_ivp(ode.rhs, (0, T), y0)

# Mid level — exponential stepping
advance = ptd.stepper(dt=0.02)
y = advance(y)                               # call repeatedly

# Turnkey — a transient run
traj = ptd.transient(y0, dt=0.02, steps=200)

# Ports — drive a soft source, record probes
times, resp = ptd.driven_transient(
    source=([0.5, 0.5, 0.5], "E", "z"),
    waveform=rf.GaussianPulse(t0=0.4, tau=0.1, f0=0.0),
    probes=[([0.25, 0.25, 0.5], "E", "z")],
    dt=0.01, steps=200)

# Model-order reduction — a few dozen DOFs reproduce the propagation
rom = ptd.reduce(y0, dim=60)
y_t = rom.propagate(y0, t)                   # cheap, exact in the subspace

# VTK animation — open the .pvd in ParaView
ptd.export_vtk(traj, "out/cavity")
```

Runnable scripts in `rapidfem/examples/`: `td_cavity.py`,
`td_model_reduction.py`, `td_field_export.py`, `td_transfer_function.py`,
`td_dielectric_cavity.py` (geometry-based), `td_waveguide_sparams.py`
(WR-90 S-parameters vs FD), `td_fd_crossvalidation.py`.

## Waveguide ports

A `RectWaveguidePort` or `LumpedPort` attached to a geometry face becomes
a time-domain **modal port**. The port boundary uses a characteristic
flux with the ghost state set to the incident modal field: it absorbs
outgoing waves *and* injects the mode. The absorbing part folds into the
constant operator `A`; the injection is a rank-1 time-dependent source
`b(t)`. A `RectWaveguidePort` carries the analytic `TE_mn`
rectangular-waveguide mode; a `LumpedPort` carries the `(0, 0)` sentinel
mode — a uniform transverse profile with zero cutoff and a flat
impedance, i.e. a TEM port.

`ProblemTD.sparams(freqs, dt, steps)` drives each port in turn with a
broadband pulse, projects the port-face field onto the mode profile to
extract the incident / scattered modal amplitudes (the forward/backward
split uses the dispersive modal impedance `Z(ω)` per frequency), and
assembles `S_ij(f) = B_i(f)/A_j(f)`. The matched-guide S-matrix is
validated to ~2 % in the Rust suite, and that the lumped port carries a
dispersionless `c`-velocity TEM wave is a further Rust gate.
`td_waveguide_sparams.py` cross-checks a WR-90 guide against the
frequency-domain backend — TD↔FD `|S₂₁|` agrees to ≤ 4 % over the
9–12 GHz core band (≤ 9 % including the dispersive near-cutoff edge),
`|S₁₁|` to ≤ 0.04.

The guide must be long enough that, within the transient window, the
incident / reflected / transmitted pulses are time-separated — the
characteristic port has a residual modal-mismatch reflection, so its
multiply-reflected energy must not return before the run ends.

## Materials and boundaries

- Heterogeneous, lossy and **diagonal-anisotropic** `ε`, `μ`; Ohmic
  conductivity `σ` and magnetic conductivity `σ*`. Material data flows
  from the geometry's physics API.
- **Matched absorbing layers** — a graded lossy layer with `σ/ε = σ*/μ`
  absorbs outgoing waves with no reflection at the layer interface.
- **Dispersive media** — Debye relaxation via the ADE machinery,
  validated against the analytic `ε(ω)`.
- PEC outer walls (ghost-state flux).

## Validation

- **FD ↔ TD cross-validation** is the production-confidence gate: the same
  cavity run through both backends agrees to **0.04 %**.
- The Rust validation suite (`cargo test -p rapidfem-td`) covers the
  cavity-eigenmode gate (`π√2` to `1e-4`), exact energy conservation of
  the central-flux operator (`M̃A` skew-symmetric), the dielectric
  resonance shift, conductivity / matched-absorber decay rates, the ETD
  integrator against analytic ODEs, Krylov MOR, and sparse-assembly
  scaling.
- The Python-API regression suite (`python/tests/test_td.py`) runs in CI
  alongside `cargo test`.

## Performance

- The matrix-free `apply` is **parallel across cores** (rayon, per
  element) and **allocation-free** — each worker reuses a pooled scratch
  buffer instead of allocating per element.
- The exponential propagator reuses a **`KrylovWorkspace`**: a transient
  step allocates nothing beyond its result array, so a long step loop
  does no per-step heap work. This alone made stepping ~2.6× faster.
- `state_space()` assembles `A` **element-wise and sparse** — `O(nnz)`
  memory, never densifying — so it scales to 10⁵-DOF meshes where a dense
  matrix would need gigabytes.
- numpy data crosses the Python ↔ Rust boundary **zero-copy**.
- Benchmarks: `cargo run --release -p rapidfem-td --example bench`;
  `--example allocaudit` counts allocations on the hot paths.

## Not yet / out of scope

- **General-cross-section ports** — `sparams` handles rectangular
  waveguide ports (analytic `TE_mn` modes) and lumped / TEM ports (the
  uniform `(0, 0)` profile); arbitrary cross-sections would need a 2D
  modal eigensolve on the port face, and coax ports an annular TEM
  profile — both a further extension.
- **Curvilinear / isoparametric elements** — affine tets with adequate
  refinement is the pragmatic choice.
- **Nonlinear materials** — the backend stays linear (constant `A`).

See `docs/td-backend-plan.md` and `docs/td-production-plan.md` for the
development roadmap and work-package history.
