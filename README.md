# RapidFEM

Frequency-domain electromagnetic FEM solver written in Rust. Second-kind Nedelec
edge elements, complex-symmetric sparse linear algebra, native + WebAssembly
targets.

## Features

- **Nedelec-2 elements** — 20 DOFs per tetrahedron, vector edge basis for the
  curl–curl form of Maxwell's equations
- **Excitations** — rectangular waveguide ports (arbitrary TE modes), lumped
  ports (TEM, multi-line voltage integral), and absorbing boundary conditions
  of order 1 and 2 (selectable coefficient types A–E)
- **PML** — anisotropic stretched-coordinate perfectly matched layer
  (configurable thickness, polynomial grading, peak attenuation)
- **Lossy materials** — complex permittivity with loss tangent + conductivity;
  frequency-independent caching speeds up sweeps
- **Sparse solvers** — MKL PARDISO (complex-symmetric LDLᵀ, fastest path on
  Windows/Linux) with a pure-Rust [`faer`](https://github.com/sarah-quinones/faer-rs)
  fallback for WASM and other targets
- **Frequency sweep** — assembles E/B once, refactors only the frequency-
  dependent K per point, reuses the symbolic LU pattern
- **Eigenmode solver** — shift-invert Lanczos on the complex-symmetric system
  for cavity / waveguide modes
- **Adaptive refinement** — residual error estimator (volume residual + face
  jumps) with Dörfler marking, exports a size field for Gmsh re-meshing
- **In-browser meshing** — companion `rapidfem-mesher` crate produces tagged
  tetrahedral meshes from a declarative `MeshSpec` (layered RFIC-style stacks
  with conductors, ports, and PML). No external meshing tool required for the
  WASM workflow
- **Output** — Touchstone (.s1p/.s2p/.snp), VTK field export, far-field NFFT
- **Parallel assembly** — rayon-based element matrix evaluation on native

## Quick start

```bash
cargo build --release
cargo run --release -- config.toml
```

## Configuration

Simulations are described by a TOML file. Required sections: `[mesh]`,
`[pec]`, and at least one `[[ports]]` entry.

### Minimal example

```toml
[mesh]
file = "waveguide.msh"

[frequency]
values = [10.0e9]

[[ports]]
type = "rectangular"
tag = 3

[[ports]]
type = "rectangular"
tag = 4

[pec]
tags = [1]

[output]
touchstone = "result.s2p"
```

### Full reference

```toml
[mesh]
file = "model.msh"               # Gmsh .msh, v4 format

[frequency]
values = [10.0e9]                # explicit list (Hz)
# OR
range = [9.0e9, 11.0e9, 21]      # [start, stop, n_points]

# Materials — repeatable
[[materials]]
volume_tag = 2
er = 4.0                         # default 1.0
ur = 1.0                         # default 1.0
tand = 0.01                      # default 0.0
conductivity = 0.0               # S/m, default 0.0

# Ports — repeatable; mix types freely
[[ports]]
type = "rectangular"
tag = 3
mode = [1, 0]                    # TE mode [m, n], default [1, 0]
er = 1.0
power = 1.0                      # W, default 1.0
width = 22.86e-3                 # m; auto-detected if omitted
height = 10.16e-3

[[ports]]
type = "lumped"
tag = 5
z0 = 50.0
direction = [0, 0, 1]            # E-field direction, required
width = 1.0e-3                   # auto-detected if 0
height = 1.0e-3
power = 1.0

[[ports]]
type = "abc"
tag = 6
order = 2                        # 1 or 2
abctype = "B"                    # A, B, C, D, E

# Perfectly Matched Layer — repeatable, one per absorbing direction
[[pml]]
volume_tag = 11
direction = [1, 0, 0]            # absorption axis (unit vector along ±x/y/z)
inner_face = 0.0010              # coordinate of inner boundary along the axis
thickness = 0.0003               # outward thickness
er_base = 1.0
ur_base = 1.0
exponent = 1.5                   # σ ~ uⁿ grading, default 1.5
delta_max = 8.0                  # peak stretch, default 8.0

[pec]
tags = [1]

[solver]
prefer = "auto"                  # "pardiso", "faer", or "auto" (default)

[output]
touchstone = "result.s2p"
z0 = 50.0
```

### Mesh requirements

Native runs consume Gmsh `.msh` files (v4 format) with physical groups for
every volume (materials) and surface (PEC, ports, ABC). For the in-browser
WASM path the bundled `rapidfem-mesher` crate generates the mesh directly
from a `MeshSpec` — no external meshing step.

## Solver backends

| Solver | Type | Notes |
|--------|------|-------|
| MKL PARDISO | Complex-symmetric LDLᵀ | Fastest; requires `mkl_rt` on PATH |
| faer | General sparse LU | Pure Rust; the only option in WASM builds |

Pick via `solver.prefer = "auto" | "pardiso" | "faer"`.

### Installing MKL (optional)

- **conda**: `conda install mkl`
- **pip**: `pip install mkl`
- **NuGet**: `nuget install intelmkl.redist.win-x64`
- **Intel oneAPI**: [download](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)

Ensure `mkl_rt.dll` (or `mkl_rt.2.dll`) is on the system PATH.

## Performance

WR-90 iris waveguide driven sweep, 10 GHz, 2-port:

| Mesh | DOFs | PARDISO | faer |
|------|------|---------|------|
| 693 tets | 5 512 | 0.14 s | 0.22 s |
| 1 096 tets | 8 382 | 0.06 s | 0.45 s |
| 2 595 tets | 19 196 | 0.17 s | 1.39 s |
| 3 284 tets | 23 968 | 0.21 s | 1.98 s |

Larger problems:

- 327 k DOFs driven sweep (PARDISO): ~5 s per frequency
- 905 k DOFs eigenmode (3-turn spiral, shift-invert Lanczos): ~54 s

11-point sweep at 10 k DOFs: **2.5 s** with PARDISO.

## Verification

Element-level functions (curl–curl integrals, Robin BC, second-order ABC,
mode-power normalization, surface integrals) are checked to machine precision
(1e-12 – 1e-16) by `cargo test --release`.

End-to-end S-parameter accuracy is tracked in `tests/validation/` against
analytical solutions and external reference solvers — see that directory's
README for per-case status and tolerances.

```bash
cargo test --release
```

## In-browser demo

A SvelteKit + WebAssembly demo lives under `wasm-demo/`. Solves a Sky130
microstrip end-to-end client-side — meshing, FEM, S-params, and a 3D field
viewer with live wave animation. See `wasm-demo/README.md` for details.

## License

GPL-2.0.
