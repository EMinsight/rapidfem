"""WR-90 rectangular waveguide — 21-point sweep across the single-mode band.

Notebook-style flow:

    Parameters  ->  Geometry + Materials  ->  Mesh  ->  Simulation

Each `# %%` block is a cell. Run them top-to-bottom with Shift+Enter, or hit
Run-All. The script also runs end-to-end as plain `python wr90.py` — `# %%`
is just a comment.
"""

# %% Parameters
# All knobs in one place: waveguide dims, frequency sweep, mesh density.
import numpy as np
import rapidfem

A, B, L = 22.86e-3, 10.16e-3, 30.0e-3        # WR-90 width, height, length [m]
FREQUENCIES = np.linspace(8.0e9, 12.0e9, 21) # 21-point sweep, single-mode band
MAXH = 3.0e-3                                # max tet edge length [m]


# %% Geometry + Materials
# OCC box, named ports + PEC walls, material assignment for the volume.
g = rapidfem.Geometry()
air = g.box(A, B, L, position=(-A / 2, -B / 2, 0))
air.material = "air"

air.faces.min(axis="z").name = "port_in"
air.faces.max(axis="z").name = "port_out"
for face in air.faces:
    if face.name is None:
        face.name = "pec"

rapidfem.show(g)


# %% Mesh
# Generate the tetrahedral discretization. show(g) now picks up the meshed
# state and renders the FEM cells instead of the OCC wireframe.
g.mesh(maxh=MAXH)
rapidfem.show(g)


# %% Simulation
# Wire ports + materials + frequencies onto the mesh, build the FEM operator,
# run the sweep, and hand both the simulation + result to the viewer.
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
print(f"|S11| at f0: {abs(result.sparams[0, 0, 0]):.4g}")
print(f"|S21| at f0: {abs(result.sparams[0, 1, 0]):.4g}")
