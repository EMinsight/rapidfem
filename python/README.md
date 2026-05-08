# rapidfem (Python)

Python bindings for [rapidfem](https://github.com/milanofthe/rapidfem) — a
frequency-domain electromagnetic FEM solver in Rust. Designed to feel like
NGSolve: build geometry by named primitives + selectors, no integer physical
group tags or TOML config strings in user code.

## Install

Requires Rust + maturin. From the repo root:

```bash
cd python
maturin develop --release
```

This builds the Rust extension and installs `rapidfem` as an editable wheel
into the active Python environment.

## Quickstart — WR-90 waveguide

```python
import numpy as np
import rapidfem

a, b, L = 22.86e-3, 10.16e-3, 30e-3   # WR-90 broad/narrow walls + length

g = rapidfem.Geometry()
box = g.box(a, b, L)

# Tag walls and ports by selector — no bounding-box matching, no integer tags
box.faces.where(lambda c, _: abs(c[0] - 0) < 1e-9).name = "pec_wall"
box.faces.where(lambda c, _: abs(c[0] - a) < 1e-9).name = "pec_wall"
box.faces.where(lambda c, _: abs(c[1] - 0) < 1e-9).name = "pec_wall"
box.faces.where(lambda c, _: abs(c[1] - b) < 1e-9).name = "pec_wall"
box.faces.min(axis="z").name = "port1"
box.faces.max(axis="z").name = "port2"

sim = (
    rapidfem.SimulationBuilder()
    .from_geometry(g, maxh=3e-3)
    .frequencies(np.linspace(9e9, 11e9, 11))
    .pec("pec_wall")
    .rect_waveguide("port1", mode=(1, 0), width=a, height=b)
    .rect_waveguide("port2", mode=(1, 0), width=a, height=b)
    .build()
)
g.close()

result = sim.run_sweep()
print(result.frequencies, result.sparams.shape)   # numpy arrays
```

## Two-layer architecture

- **`rapidfem.Geometry`** — gmsh OCC backend with NGSolve-style attribute writes.
  Primitives return `GeoObject` wrappers exposing `.faces` and `.edges` collections
  with `.min(axis=…)`, `.max(…)`, `.where(predicate)` selectors.
- **`rapidfem.SimulationBuilder`** — fluent API mapping geometry names to ports
  and materials. Resolves names to integer physical-group tags internally,
  produces a `Simulation` instance ready for `.run_sweep()` / `.run_eigenmode()` /
  `.compute_farfield()`.

Both layers compose with the lower-level `Simulation.from_files(mesh_path,
config_path)` if you prefer the TOML-driven flow for legacy meshes.

## Geometry primitives

| Method | Result | Notes |
| --- | --- | --- |
| `g.box(w, d, h, position=(0,0,0))` | 3D | axis-aligned box, `position` = lower corner |
| `g.cylinder(r, h, position, axis=(0,0,1))` | 3D | along arbitrary axis |
| `g.cone(r1, r2, h, position, axis)` | 3D | truncated cone or cylinder |
| `g.sphere(r, center)` | 3D | |
| `g.wedge(dx, dy, dz, top_x=0, position)` | 3D | base prism, top edge offset |
| `g.torus(R, r, center, angle=2π)` | 3D | major × minor radii |
| `g.xy_plate(w, h, position)` | 2D | rectangle in xy-plane |
| `g.yz_plate(w, h, position)` | 2D | rectangle in yz-plane |
| `g.xz_plate(w, h, position)` | 2D | rectangle in xz-plane |
| `g.plate(p0, width=, height=)` | 2D | arbitrary orientation, edge vectors |

## Boolean ops

```python
g.fragment(a, b, c)   # makes geometry conformal at interfaces; names survive
g.cut(target, tool)   # boolean subtract; outer faces survive
g.fuse(target, tool)  # boolean union; WARNING: face names are lost (faces merge)
```

After any boolean op, named entities are re-resolved by matching their
stored bounding box and center-of-mass against the new gmsh topology.
`fuse` is supported but warns — face merging shifts COGs unrecoverably.

## Selectors

```python
box.faces.min(axis="z").name = "ground"
box.faces.max(axis="z").name = "top"
box.faces.where(lambda cog, bbox: cog[0] > 5).name = "x_high"
patch.edges.min(axis="z").maxh = 0.1e-3   # refine the patch's bottom edge
```

Selectors return new collections, so you can chain:
`box.faces.where(predicate).max(axis="x")`.

## Mesh refinement

Set `obj.maxh = h` on volumes, faces, or edges. The mesh emit step builds a
`Distance` + `Threshold` background field for each refined entity, so the size
grows smoothly from `h` near the entity to the global `maxh` at distance ~5h.
Overlapping refinement zones use the local minimum.

```python
patch.maxh = 2e-3       # global refinement near the patch
feed.maxh = 0.5e-3      # finer at the feed
patch.edges.maxh = 0.3e-3   # extra refinement at antenna edges
```

Pass an explicit `transition_distance` to `g.mesh()` for finer control over
the gradation distance.

## SimulationBuilder fluent API

| Method | Action |
| --- | --- |
| `.from_geometry(g, maxh=…)` | Mesh the Geometry and load it |
| `.mesh(bytes, name_to_tag)` | Use a pre-built mesh + name map |
| `.frequencies([…])` | Frequency points (Hz) |
| `.frequency_range(start, stop, n)` | Linear sweep |
| `.pec("name", …)` | Mark named surfaces as PEC |
| `.pmc("name", …)` | Mark as PMC (natural BC) |
| `.rect_waveguide("name", mode=(1,0), …)` | Rectangular waveguide port |
| `.lumped_port("name", direction=, z0=)` | Lumped port |
| `.coax_port("name", ri=, ro=)` | Coaxial port |
| `.user_defined_port("name", e_field=)` | User-defined mode (constant E) |
| `.floquet_port("name", scan_theta_deg=, …)` | Floquet plane-wave port |
| `.abc("name", order=1)` | Absorbing boundary |
| `.surface_impedance("name", conductivity=)` | Lossy conductor surface |
| `.lumped_element("name", r=, l=, c=)` | R/L/C surface element |
| `.material("name", er=, ur=, tand=, conductivity=)` | Material on a volume |
| `.material("name", er=, debye={…})` | Debye dispersion |
| `.material("name", drude={…})` | Drude dispersion |

## Examples

In `python/examples/`:

- `wr90.py` — WR-90 via TOML files (legacy `Simulation.from_files`)
- `geometry_wr90.py` — WR-90 via Geometry only, manual TOML
- `builder_wr90.py` — WR-90 via Geometry + SimulationBuilder (recommended)
- `builder_patch_antenna.py` — Edge-fed patch antenna, full pipeline incl. far-field
- `geometry_fragment.py` — Smoke test for name preservation through `fragment()`
- `patch_antenna.py` — Sweep + far-field on a TOML-mesh patch antenna

## Limitations

- `g.fuse()` does not preserve face names (face merging shifts COGs); set names
  *after* the fuse, or use `g.fragment()` if names matter.
- Per-entity refinement assumes 3D entities have boundary surfaces (most do).
  If a primitive returns no boundary, that refinement field is silently
  skipped.
- Simulation is run on a single thread for the Rust core (rayon-parallel
  assembly is enabled by default; PARDISO uses MKL OpenMP).
- Python binding is `unsendable` (the underlying Rust types hold trait
  objects). A Simulation must stay on the thread that created it.

## Lower-level entry points

If you prefer not to use the Geometry/Builder layer:

```python
sim = rapidfem.Simulation.from_files("mesh.msh", "config.toml")
sim = rapidfem.Simulation.from_bytes(mesh_bytes, config_toml_str)
```

Available simulation methods: `run_sweep()`, `run_eigenmode()`,
`compute_farfield(result, freq_idx, port_idx, n_theta, n_phi)`. See the
top-level Rust crate documentation for TOML config field reference.
