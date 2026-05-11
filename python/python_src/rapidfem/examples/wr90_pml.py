"""WR-90 waveguide terminated in a PML — single-port matched load.

Demonstrates how to use a Perfectly Matched Layer instead of (or alongside)
ABC. A PML volume absorbs outgoing waves with vastly less reflection than
a 1st/2nd order ABC, especially at oblique incidence — the trade is that
PML is a *volumetric* region (a mesh-resolved shell) rather than a single
boundary surface.

Setup: WR-90 with the driven port at z=0 and a PML cap at z=L_inner. With
the PML tuned correctly, |S11| should be at the numerical floor across
the whole single-mode band — that's the matched-load signature.

Notebook-style:  Parameters -> Geometry -> Mesh -> Simulation
"""

# %% Parameters
import numpy as np
import rapidfem

A, B = 22.86e-3, 10.16e-3       # WR-90 cross-section
L_INNER = 40.0e-3                # length of the regular (air) section
PML_T  = 15.0e-3                 # PML thickness along +z

FREQUENCIES = np.linspace(8.0e9, 12.0e9, 21)
MAXH = 5.0e-3


# %% Geometry + Materials
g = rapidfem.Geometry()

# Regular air section — the driven port lives at its z=0 face.
inner = g.box(A, B, L_INNER, position=(-A / 2, -B / 2, 0))
inner.material = "air"

# PML cap sitting on top of the air section. Material is NOT set — the
# PML region's permittivity is configured via the builder's .pml(...) call
# (er_base / ur_base / stretching profile). Just give it a name so the
# builder can resolve it to a physical-group tag.
pml = g.box(A, B, PML_T, position=(-A / 2, -B / 2, L_INNER))
pml.name = "pml_back"

# Make the two volumes share their interface face cleanly.
g.fragment(inner, pml)

# Port at z=0 of the inner volume; all other faces (incl. the PML's outer
# walls and the +z back cap) are PEC. The PML alone handles the absorption.
inner.faces.min(axis="z").name = "port_in"
for face in inner.faces:
    if face.name is None:
        face.name = "pec"
for face in pml.faces:
    if face.name is None:
        face.name = "pec"

rapidfem.show(g)


# %% Mesh
g.mesh(maxh=MAXH)
rapidfem.show(g)


# %% Simulation
sim = (
    rapidfem.SimulationBuilder()
    .mesh_from(g)
    .frequencies(FREQUENCIES)
    .rect_waveguide("port_in")
    .pec("pec")
    .pml("pml_back",
         direction=(0, 0, 1),       # absorbs waves traveling in +z
         inner_face=L_INNER,        # the PML's inner boundary along z
         thickness=PML_T,
         exponent=1.5,
         delta_max=8.0)
    .material("air", er=1.0)
    .build()
)
result = sim.run_sweep()
rapidfem.show(sim)
rapidfem.show(result)

import math
s11_dB = [20 * math.log10(max(abs(result.sparams[i, 0, 0]), 1e-12))
          for i in range(len(FREQUENCIES))]
print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")
print(f"|S11| range across band: {min(s11_dB):.1f} dB to {max(s11_dB):.1f} dB")
print("(a well-tuned PML termination shows |S11| well below -30 dB everywhere)")
