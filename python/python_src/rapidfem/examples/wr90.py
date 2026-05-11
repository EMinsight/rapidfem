"""
WR-90 rectangular waveguide — 21-point sweep across the single-mode band.

Notebook-style flow with explicit stages:

    Parameters  →  Geometry + Materials  →  Mesh  →  Simulation

Each `# %%` block is a cell. Run them top-to-bottom with Shift+Enter, or
hit Run-All. The script also runs end-to-end as plain `python wr90.py` —
`# %%` is just a comment.
"""

# %% [markdown]
# # 1. Parameters
# All knobs in one place — geometry dimensions, frequency sweep, mesh density.

# %% Parameters
import numpy as np
import rapidfem

# WR-90 waveguide dimensions
A, B, L = 22.86e-3, 10.16e-3, 30.0e-3        # width, height, length [m]

# Frequency sweep (single-mode band: 8.2 – 12.4 GHz nominal)
FREQUENCIES = np.linspace(8.0e9, 12.0e9, 21)

# Mesh size
MAXH = 5.0e-3                                 # max edge length [m]


# %% [markdown]
# # 2. Geometry + Materials
# OCC primitives, named surfaces (ports + PEC walls), material assignments.

# %% Geometry + Materials
g = rapidfem.Geometry()

# Air-filled rectangular waveguide section, centered on the x/y axes.
air = g.box(A, B, L, position=(-A / 2, -B / 2, 0))
air.material = "air"

# Name the two faces at z = 0 and z = L as ports; everything else is PEC.
air.faces.min(axis="z").name = "port_in"
air.faces.max(axis="z").name = "port_out"
for face in air.faces:
    if face.name is None:
        face.name = "pec"

rapidfem.show(g)


# %% [markdown]
# # 3. Mesh
# Generate the tetrahedral discretization with gmsh. The viewer picks up
# the meshed state and renders the FEM cells.

# %% Mesh
g.mesh(maxh=MAXH)
rapidfem.show(g)


# %% [markdown]
# # 4. Simulation
# Wire the ports, materials, and frequency list to the meshed geometry,
# build the FEM operator, and run the sweep.

# %% Simulation
sim = (
    rapidfem.SimulationBuilder()
    .mesh_from(g)
    .frequencies(FREQUENCIES)
    .rect_waveguide("port_in")
    .rect_waveguide("port_out")
    .pec("pec")
    .material("air", er=1.0)
    .build()
)
result = sim.run_sweep()
rapidfem.show(sim)
rapidfem.show(result)

print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")
print(f"|S11| at f₀: {abs(result.sparams[0, 0, 0]):.4g}")
print(f"|S21| at f₀: {abs(result.sparams[0, 1, 0]):.4g}")
