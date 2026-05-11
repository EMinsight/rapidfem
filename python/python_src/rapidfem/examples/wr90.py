"""
WR-90 rectangular waveguide — 21-point sweep across the single-mode band.

Notebook-style: each `# %%` block is a cell. Run cells one at a time with
Shift+Enter, or hit Run-All. The script also runs end-to-end as plain
`python wr90.py` — `# %%` is just a comment.
"""

# %% [markdown]
# # WR-90 waveguide
#
# Build a short WR-90 section (22.86 × 10.16 × 30 mm), drive it with two
# rectangular-waveguide ports, and sweep across 8–12 GHz.

# %% Setup
import numpy as np
import rapidfem

A, B, L = 22.86e-3, 10.16e-3, 30.0e-3

# %% Geometry
g = rapidfem.Geometry()
air = g.box(A, B, L, position=(-A / 2, -B / 2, 0))
air.material = "air"
air.maxh = 5e-3

air.faces.min(axis="z").name = "port_in"
air.faces.max(axis="z").name = "port_out"
for face in air.faces:
    if face.name is None:
        face.name = "pec"

rapidfem.show(g)

# %% Mesh
# Generate the tet mesh on the OCC geometry. show(g) now picks up the
# meshed state and renders the FEM discretization instead of the coarse
# OCC preview tessellation.
g.mesh(maxh=5e-3)
rapidfem.show(g)

# %% Builder
builder = (
    rapidfem.SimulationBuilder()
    .mesh_from(g)
    .frequencies(np.linspace(8.0e9, 12.0e9, 21))
    .rect_waveguide("port_in")
    .rect_waveguide("port_out")
    .pec("pec")
    .material("air", er=1.0)
)

# %% Solve
sim = builder.build()
result = sim.run_sweep()
rapidfem.show(sim)
rapidfem.show(result)

# %% Inspect
print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")
print(f"|S11| at first freq: {abs(result.sparams[0, 0, 0]):.4g}")
print(f"|S21| at first freq: {abs(result.sparams[0, 1, 0]):.4g}")
