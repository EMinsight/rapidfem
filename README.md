# RapidFEM

Production frequency-domain electromagnetic FEM solver in Rust. Exact port of [EMerge](https://github.com/rfen/emerge)'s computational path using Nedelec second-kind elements.

## Features

- **Nedelec-2 elements** — 20 DOFs/tet, second-kind edge elements
- **Rectangular waveguide ports** — Robin BC with arbitrary TE modes
- **Lumped ports** — TEM excitation with voltage-based S-parameters
- **Absorbing boundary conditions** — Order 1 and 2 with selectable type (A-E)
- **Lossy materials** — Complex permittivity with loss tangent and conductivity
- **MKL PARDISO solver** — Complex-symmetric LDLt (mtype=6), 7-10x faster than pure Rust. Optional: falls back to faer if MKL not installed
- **Frequency sweep** — Cached E/B matrices, symbolic LU reuse across frequencies
- **Touchstone export** — S1P/S2P/SNP output
- **Parallel assembly** — rayon-parallelized element matrix computation

## Quick Start

```bash
cargo build --release
cargo run --release -- config.toml
```

## Configuration

Simulations are defined via TOML config files. All fields have sensible defaults — only `[mesh]`, `[pec]`, and at least one `[[ports]]` entry are required.

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

### Full configuration reference

```toml
[mesh]
file = "model.msh"              # Gmsh .msh file (v4 format)

[frequency]
values = [10.0e9]               # Explicit frequency list (Hz)
# OR
range = [9.0e9, 11.0e9, 21]     # [start, stop, n_points]

# Materials (optional, repeatable)
[[materials]]
volume_tag = 2                   # Gmsh physical volume tag
er = 4.0                        # Relative permittivity (default: 1.0)
ur = 1.0                        # Relative permeability (default: 1.0)
tand = 0.01                     # Loss tangent (default: 0.0)
conductivity = 0.0              # Conductivity in S/m (default: 0.0)

# Ports (repeatable, multiple types)
[[ports]]
type = "rectangular"             # Rectangular waveguide port
tag = 3                          # Gmsh physical surface tag
mode = [1, 0]                    # TE mode [m, n] (default: [1, 0])
er = 1.0                        # Port fill dielectric (default: 1.0)
power = 1.0                     # Port power in W (default: 1.0)
width = 22.86e-3                 # Broad wall in m (default: auto-detected)
height = 10.16e-3                # Narrow wall in m (default: auto-detected)

[[ports]]
type = "lumped"                  # Lumped element port
tag = 5                          # Gmsh physical surface tag
z0 = 50.0                       # Characteristic impedance in Ohm (default: 50)
direction = [0, 0, 1]           # E-field direction (required)
width = 1.0e-3                  # Gap width in m (default: auto-detected)
height = 1.0e-3                 # Gap height in m (default: auto-detected)
power = 1.0                     # Port power in W (default: 1.0)

[[ports]]
type = "abc"                     # Absorbing boundary condition
tag = 6                          # Gmsh physical surface tag
order = 2                        # ABC order: 1 or 2 (default: 1)
abctype = "B"                   # Coefficient type: A, B, C, D, E (default: B)

[pec]
tags = [1]                       # Gmsh physical surface tags for PEC walls

[solver]
prefer = "auto"                  # "pardiso", "faer", or "auto" (default: auto)

[output]
touchstone = "result.s2p"        # Touchstone output file (optional)
z0 = 50.0                       # Reference impedance in Ohm (default: 50)
```

### Mesh requirements

Meshes are created externally with [Gmsh](https://gmsh.info/) and exported as `.msh` (v4 format). The mesh must have:

- **Volume physical groups** — for material assignment
- **Surface physical groups** — for PEC walls, ports, and ABCs

Each surface/volume tag referenced in the config must exist as a physical group in the mesh.

## Solvers

RapidFEM supports two sparse direct solvers:

| Solver | Type | Speed | Dependency |
|--------|------|-------|------------|
| **MKL PARDISO** | Complex-symmetric LDLt | Fast (7-10x) | `mkl_rt.dll` on PATH |
| **faer** | General sparse LU | Baseline | None (pure Rust) |

The solver is selected at runtime:
- `prefer = "auto"` (default): tries PARDISO, falls back to faer
- `prefer = "pardiso"`: PARDISO only (fails if MKL not installed)
- `prefer = "faer"`: faer only (no MKL needed, works on all platforms including WASM)

### Installing MKL (optional)

PARDISO requires Intel MKL. Install via any of:
- **conda**: `conda install mkl`
- **pip**: `pip install mkl`
- **NuGet**: `nuget install intelmkl.redist.win-x64`
- **Intel oneAPI**: [download](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)

Ensure `mkl_rt.2.dll` (or `mkl_rt.dll`) is on your system PATH.

## Performance

Benchmarks on WR-90 iris waveguide (10 GHz, 2-port):

| Mesh | DOFs | PARDISO | faer |
|------|------|---------|------|
| 693 tets | 5,512 | 0.14s | 0.22s |
| 1,096 tets | 8,382 | 0.06s | 0.45s |
| 2,595 tets | 19,196 | 0.17s | 1.39s |
| 3,284 tets | 23,968 | 0.21s | 1.98s |

11-point frequency sweep (10K DOFs): **2.5s** with PARDISO.

## Verification

All element-level functions verified against EMerge to machine precision (1e-12 to 1e-16). S-parameters match EMerge to 6 significant digits on identical meshes.

```bash
cargo test --release
```

## License

Based on computational methods from [EMerge](https://github.com/rfen/emerge) (GPL-2.0).
