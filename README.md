

[![PySimHub](https://pysimhub.io/badge.svg)](https://pysimhub.io/projects/rapidfem)

# RapidFEM

Electromagnetic FEM solver written in Rust, distributed as a Python package
on PyPI. Two backends behind one geometry/material/physics API: a
**frequency-domain** solver (Nédélec first-kind order-2 edge elements,
complex-symmetric sparse linear algebra) and a **time-domain** DGTD solver
(discontinuous Galerkin, Krylov/ETD exponential time integration,
model-order reduction). The solver is scale-invariant, so sub-micron RFIC
passives (with GDS / PDK-stack import) solve as reliably as metre-scale
structures. Optional Flask-based local UI with code editor and live viewer.

## Install

```bash
pip install rapidfem            # solver only
pip install rapidfem[ui]        # solver + local UI
```

Wheels for Windows, Linux, and macOS are built via CI. The Rust core is
compiled ahead of time — no Rust toolchain required on the user's machine.
Gmsh (Python wheel `gmsh`) is pulled in automatically and provides the
OpenCASCADE-based geometry + mesher used by `rapidfem.Geometry`.

## Quick start (Python API)

```python
import numpy as np
import rapidfem as rf

# Build geometry; attach materials + physics directly to entities
g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
air = g.box(22.86e-3, 10.16e-3, 30e-3, position=(-11.43e-3, -5.08e-3, 0),
            material=rf.Air())

rf.RectWaveguidePort(air.faces.min(axis="z"))
rf.RectWaveguidePort(air.faces.max(axis="z"))
rf.PEC(*air.faces.unassigned)

g.mesh()

# Define the problem once, run any number of analyses on it
prob = rf.Problem(g)                      # Problem is the frequency-domain ProblemFD
result = prob.sweep(np.linspace(8e9, 12e9, 21))
print(result.frequencies.shape, result.sparams.shape)

# Same Problem can also drive an eigenmode solve or a far-field pattern:
# modes   = prob.eigenmode(target_frequency=10e9, n_modes=6)
# pattern = prob.farfield(result, freq_idx=10, port_idx=0)
```

See `python_src/rapidfem/examples/` for end-to-end runs: microstrip and
coupled lines, iris / stepped-impedance filters, patch / Vivaldi / inverted-F
antennas (PML + far-field), pyramidal horns, dielectric resonators, and the
`fd_rfic_*` on-chip passives. RFIC geometry comes from a process stack and
layout via `rapidfem.rfic` (`rfic.Stack`, `Geometry.from_gds`).

## Local UI

```bash
rapidfem serve ./my_project/
```

Opens a browser window with a CodeMirror Python editor, a 3D geometry / mesh /
field viewer (raw WebGL2), and S-parameter plots. The geometry view updates
on save (`Ctrl+S`); mesh and solver runs are explicit. Results stream in as
the solve runs; fields are fetched on demand as you scrub frequency and port.
Use `rapidfem.show(g)` to send a geometry to the viewer.

## Features

- **Geometry builder** — OpenCASCADE primitives with boolean ops, transforms
  and fillet/chamfer; ready-made RF structures in `rf.structures` (coax,
  microstrip, CPW, stripline, waveguides, helix) build geometry + ports in one
  call
- **RFIC / GDS** — `rapidfem.rfic` process stacks and `Geometry.from_gds` build
  on-chip passives, solved scale-invariantly down to sub-micron features
- **Canonical Nédélec R2 elements** — first-kind order-2 curl–curl vector
  element, 20 DOFs per tetrahedron
- **Excitations** — rectangular waveguide ports (arbitrary TE modes), lumped
  ports (TEM, multi-line voltage integral), coax and wave ports, Floquet
  plane-wave port (normal incidence), first-order absorbing boundary
- **PML** — anisotropic stretched-coordinate perfectly matched layer
- **Lossy materials** — complex permittivity with loss tangent + conductivity,
  surface impedance for metals, Debye dispersion; cached across sweeps
- **Sparse solvers** — pure-Rust [`faer`](https://github.com/sarah-quinones/faer-rs)
  LU baseline; optional MKL PARDISO (complex-symmetric LDLᵀ) on Windows / Linux;
  Apple Accelerate Bunch-Kaufman on macOS (~3× faster than faer)
- **Frequency sweep** — assembles E/B once, refactors only the frequency-
  dependent K per point, reuses the symbolic LU pattern
- **Eigenmode solver** — shift-invert Lanczos on the complex-symmetric system
- **Adaptive refinement** — residual error estimator with Dörfler marking,
  exports a size field for gmsh re-meshing
- **Output** — Touchstone (.s1p/.s2p/.snp), VTK field export, far-field NFFT
- **Parallel assembly** — rayon-based element matrix evaluation

## Time-domain backend (DGTD)

`ProblemTD`, behind the same API, compiles a structure into an explicit linear
ODE `dy/dt = A·y` and exposes it as a model at every level:

- **DGTD** — nodal discontinuous Galerkin on tetrahedra, upwind or
  energy-conserving central flux
- **Exponential time integration** — matrix-free Krylov/ETD propagator, exact
  for the linear system at any step size (no CFL limit)
- **Model export / reduction** — the RHS, the verbatim sparse operator `A`, an
  exponential stepper, or Krylov-projected reduced models
- **Materials** — heterogeneous, lossy, anisotropic and Debye dispersive media;
  matched absorbing layers; periodic boundaries
- **Output** — field probes, RFT transfer function, VTK field-animation export

```python
import rapidfem as rf

ptd  = rf.ProblemTD.box(size=(1, 1, 1), cells=(2, 2, 2), order=2)
traj = ptd.transient(y0, dt=0.02, steps=200)   # turnkey transient
rom  = ptd.reduce(y0, dim=60)                   # model-order reduction
A    = ptd.state_space()                        # the verbatim operator
```

Method notes and the `ProblemTD` API are in [`docs/td-backend.md`](docs/td-backend.md).

## Solver backends

| Solver | Type | Notes |
|--------|------|-------|
| faer | General sparse LU | Pure Rust, no native dependencies — always available |
| MKL PARDISO | Complex-symmetric LDLᵀ | Fastest on Windows / Linux; opt-in, needs `mkl_rt` on PATH |
| Apple Accelerate | Sparse Bunch-Kaufman LDLᵀ | macOS only; ~3× faster than faer, ships with macOS |

Select with `RAPIDFEM_SOLVER` (`"auto"`, `"pardiso"`, `"accelerate"`, `"faer"`),
set before `import rapidfem`. Default `"auto"` tries PARDISO → Accelerate →
faer. Optional MKL: `conda install mkl` / `pip install mkl` (ensure `mkl_rt`
is on PATH).

## Performance

WR-90 iris waveguide driven sweep, 10 GHz, 2-port:

| Mesh | DOFs | PARDISO | faer |
|------|------|---------|------|
| 693 tets | 5 512 | 0.14 s | 0.22 s |
| 1 096 tets | 8 382 | 0.06 s | 0.45 s |
| 2 595 tets | 19 196 | 0.17 s | 1.39 s |
| 3 284 tets | 23 968 | 0.21 s | 1.98 s |

Larger: 327 k DOFs driven sweep (PARDISO) ~5 s/freq; 905 k DOFs eigenmode
(3-turn spiral, shift-invert Lanczos) ~54 s.

## Verification

`cargo test --release` checks element-level functions to machine precision
(1e-12 – 1e-16). End-to-end S-parameter accuracy is tracked in
`tests/validation/` against analytical solutions and reference solvers.

## License

GPL-3.0-or-later with the Gmsh additional permission — see [LICENSE](LICENSE).
Copyright (C) Milan Rother and rapidfem contributors; commercial terms
available.
