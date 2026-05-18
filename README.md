# RapidFEM

Frequency-domain electromagnetic FEM solver written in Rust, distributed as a
Python package on PyPI. Second-kind Nedelec edge elements, complex-symmetric
sparse linear algebra, optional Flask-based local UI with a code editor and
live geometry viewer.

## Install

```bash
pip install rapidfem            # solver only
pip install rapidfem[ui]        # solver + local UI
```

Wheels for Windows, Linux, and macOS are built via CI. The Rust core is
compiled ahead of time — no Rust toolchain required on the user's machine.

Gmsh (Python wheel `gmsh`) is pulled in automatically as a dependency and
provides the OpenCASCADE-based geometry + mesher used by `rapidfem.Geometry`.

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
prob = rf.Problem(g)
result = prob.sweep(np.linspace(8e9, 12e9, 21))
print(result.frequencies.shape, result.sparams.shape)

# Same Problem can also drive an eigenmode solve or a far-field pattern:
# modes  = prob.eigenmode(target_frequency=10e9, n_modes=6)
# pattern = prob.farfield(result, freq_idx=10, port_idx=0)
```

See `python_src/rapidfem/examples/` for end-to-end runs of microstrip lines,
patch antennas (with PML enclosure + far-field), pyramidal horns, iris
filters, dielectric resonators, and more.

## Local UI

```bash
rapidfem serve ./my_project/
```

Opens a browser window with:

- a CodeMirror Python editor on the left,
- a 3D geometry / mesh / field viewer on the right (raw WebGL2, viridis
  colormap for scalar fields),
- S-parameter plots in a separate tab,
- a `Generate Mesh` button (gmsh) and a `Run Simulation` button (FEM sweep).

The geometry view updates automatically every time you save the file
(`Ctrl+S`). Mesh and solver runs are explicit.

Use `rapidfem.show(g)` at the bottom of your script to send a geometry to
the viewer.

## Features

- **Nedelec-2 elements** — 20 DOFs per tetrahedron, vector edge basis for the
  curl–curl form of Maxwell's equations
- **Excitations** — rectangular waveguide ports (arbitrary TE modes), lumped
  ports (TEM, multi-line voltage integral), and absorbing boundary conditions
  of order 1 and 2 (selectable coefficient types A–E)
- **PML** — anisotropic stretched-coordinate perfectly matched layer
- **Lossy materials** — complex permittivity with loss tangent + conductivity;
  frequency-independent caching speeds up sweeps
- **Sparse solvers** — pure-Rust [`faer`](https://github.com/sarah-quinones/faer-rs)
  LU as the default in the PyPI wheel; optional MKL PARDISO
  (complex-symmetric LDLᵀ) for the fastest path
- **Frequency sweep** — assembles E/B once, refactors only the frequency-
  dependent K per point, reuses the symbolic LU pattern
- **Eigenmode solver** — shift-invert Lanczos on the complex-symmetric system
- **Adaptive refinement** — residual error estimator (volume residual + face
  jumps) with Dörfler marking, exports a size field for gmsh re-meshing
- **Output** — Touchstone (.s1p/.s2p/.snp), VTK field export, far-field NFFT
- **Parallel assembly** — rayon-based element matrix evaluation

## Solver backends

| Solver | Type | Notes |
|--------|------|-------|
| faer | General sparse LU | Pure Rust, no native dependencies — **the default** in the PyPI wheel |
| MKL PARDISO | Complex-symmetric LDLᵀ | Fastest path; opt-in, requires `mkl_rt` on PATH |

Choose at simulation time with the `RAPIDFEM_SOLVER` environment variable
(`"faer"`, `"pardiso"`, or `"auto"`) — set before `import rapidfem`.

### Installing MKL (optional)

- **conda**: `conda install mkl`
- **pip**: `pip install mkl`
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

## Verification

Element-level functions (curl–curl integrals, Robin BC, second-order ABC,
mode-power normalization, surface integrals) are checked to machine precision
(1e-12 – 1e-16) by `cargo test --release`.

End-to-end S-parameter accuracy is tracked in `tests/validation/` against
analytical solutions and external reference solvers.

```bash
cargo test --release
```

## License

[GNU Affero General Public License v3.0 or later](LICENSE). If you run a
modified version of rapidfem as a network service (e.g. SaaS), the AGPL
requires that you make the modified source available to the users of that
service. For commercial use under different terms, get in touch.
