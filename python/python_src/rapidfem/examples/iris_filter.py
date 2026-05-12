"""3-iris X-band bandpass filter in WR-90.

Three inductive irises (thin PEC strips spanning the y-direction) inside
the waveguide form a 3rd-order Chebyshev-style bandpass response around
10 GHz. S21 should drop sharply outside the passband.

Notebook-style:  Parameters -> Geometry -> Mesh -> Simulation
"""

# %% Parameters
import numpy as np
import rapidfem

# WR-90 cross-section
A, B = 22.86e-3, 10.16e-3

# Three irises with apertures (width of the slot) and pitch along z
APERTURES = [10.0e-3, 8.0e-3, 10.0e-3]    # iris slot widths [m]
SPACING   = 15.0e-3                         # distance between iris centers [m]
IRIS_T    = 1.0e-3                          # iris thickness [m]

# Tube length: enough room for input, three irises, output
INPUT_LEN  = 12.0e-3
OUTPUT_LEN = 12.0e-3
L = INPUT_LEN + (len(APERTURES) - 1) * SPACING + 2 * IRIS_T + OUTPUT_LEN

# Sweep across the X-band single-mode region
FREQUENCIES = np.linspace(8.2e9, 12.4e9, 41)

MAXH = rapidfem.lambda_maxh(f_max=12.4e9)   # air-wavelength bound, ~2 mm


# %% Geometry + Materials
g = rapidfem.Geometry()
air = g.box(A, B, L, position=(-A / 2, -B / 2, 0))
air.material = "air"

# Iris z-centers, from input toward output
z_centers = [INPUT_LEN + IRIS_T / 2 + k * SPACING for k in range(len(APERTURES))]

# Each iris = two PEC strips along ±x that occupy the full B height, leaving
# a vertical slot of width APERTURES[k] centered on x=0.
for k, (zc, w) in enumerate(zip(z_centers, APERTURES)):
    slot = w
    strip_w = (A - slot) / 2
    for side in (-1, +1):
        x0 = side * (slot / 2) if side > 0 else -A / 2
        x0 = -A / 2 if side < 0 else slot / 2
        iris = g.box(strip_w, B, IRIS_T, position=(x0, -B / 2, zc - IRIS_T / 2))
        iris.material = "_iris_solid"  # not used by solver — just to track

# Cut the iris solids out of air so the air remains a connected volume
# around them; the iris surfaces will be auto-classified as PEC walls below.
g.fragment(air, *[o for o in g._objects if o is not air])

# Ports: z = 0 and z = L
air.faces.min(axis="z").name = "port_in"
air.faces.max(axis="z").name = "port_out"

# Everything else (waveguide walls + iris faces) is PEC
for face in air.faces:
    if face.name is None:
        face.name = "pec"

rapidfem.show(g)


# %% Mesh
# (No auto_refine_features here — the iris solids are only PEC-wall scaffolding;
# their volumes are cut out by `fragment` and don't see the FEM solve.)
g.mesh(maxh=MAXH)
rapidfem.show(g)


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

# Quick scan for the passband — where |S21| > -3 dB
import math
mags = [abs(result.sparams[i, 1, 0]) for i in range(len(FREQUENCIES))]
db = [20 * math.log10(max(m, 1e-12)) for m in mags]
passband = [FREQUENCIES[i] / 1e9 for i, d in enumerate(db) if d > -3]
print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")
if passband:
    print(f"3-dB band: {passband[0]:.2f}–{passband[-1]:.2f} GHz ({len(passband)} pts)")
else:
    print("No 3-dB passband found in the sweep — try widening the frequency range.")
