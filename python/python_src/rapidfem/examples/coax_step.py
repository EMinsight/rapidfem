"""Coaxial impedance step (50Ω → 75Ω) — quarter-wave matching section.

Two coax sections of different radii (same outer conductor radius, smaller
inner radius for higher Z0) joined back-to-back. With nominal lengths this
is just a step discontinuity — the |S11| dip vs frequency shows how the
mismatch evolves with electrical length.

Notebook-style:  Parameters -> Geometry -> Mesh -> Simulation
"""

# %% Parameters
import numpy as np
import rapidfem

# Section 1: 50 Ω air-filled coax  (Z = 60 ln(b/a) → b/a ≈ 2.30)
RI_1, RO_1 = 1.50e-3, 3.45e-3

# Section 2: 75 Ω air-filled coax  (b/a ≈ 3.49) — same outer radius, smaller inner
RI_2, RO_2 = 0.99e-3, 3.45e-3

# Section lengths
L1, L2 = 15.0e-3, 15.0e-3

FREQUENCIES = np.linspace(1.0e9, 10.0e9, 31)
# Air-filled coax — wavelength bound + curvature-based facets on the cylinder
# walls (enabled by default) give a geometry-accurate mesh without manual tuning.
MAXH = rapidfem.lambda_maxh(f_max=10.0e9)   # ~2.5 mm = λ_air/12 at f_max


# %% Geometry + Materials
g = rapidfem.Geometry()

# Outer dielectric: a single cylindrical air volume spanning both sections.
air = g.cylinder(radius=RO_1, height=L1 + L2, position=(0, 0, 0))
air.material = "air"

# Inner conductor (subtracted from air via fragment / cut). We model it as
# two stacked cylinders of different radii so the step is sharp.
inner_a = g.cylinder(radius=RI_1, height=L1, position=(0, 0, 0))
inner_b = g.cylinder(radius=RI_2, height=L2, position=(0, 0, L1))

g.fragment(air, inner_a, inner_b)

# The two end faces of the outer air volume are the coax ports.
air.faces.min(axis="z").name = "port_in"
air.faces.max(axis="z").name = "port_out"

# Everything else (inner-conductor surface + outer wall) is PEC.
for face in air.faces:
    if face.name is None:
        face.name = "pec"

rapidfem.show(g)


# %% Mesh
g.mesh(maxh=MAXH)
rapidfem.show(g)


# %% Simulation
# Coax ports are described by (ri, ro, origin); rapidfem analytically
# constructs the TEM mode field on the port face.
sim = (
    rapidfem.SimulationBuilder()
    .mesh_from(g)
    .frequencies(FREQUENCIES)
    .coax_port("port_in",  ri=RI_1, ro=RO_1, origin=(0, 0, 0))
    .coax_port("port_out", ri=RI_2, ro=RO_2, origin=(0, 0, L1 + L2))
    .pec("pec")
    .material("air", er=1.0)
    .build()
)
result = sim.run_sweep()
rapidfem.show(sim)
rapidfem.show(result)

print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")
print(f"|S11| at {FREQUENCIES[0]/1e9:.1f} GHz: {abs(result.sparams[0, 0, 0]):.4f}")
print(f"|S11| at {FREQUENCIES[-1]/1e9:.1f} GHz: {abs(result.sparams[-1, 0, 0]):.4f}")
