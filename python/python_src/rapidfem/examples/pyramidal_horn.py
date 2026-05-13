"""Pyramidal horn antenna — WR-90 feed + PML in the main-beam direction.

A standard rectangular horn for the X-band: a short WR-90 feed waveguide
followed by an 80 mm taper to a 30 × 22 mm aperture. The horn's four
metal flares are modelled as infinitely thin PEC sheets embedded in a
single air domain; the throat and aperture are interior interfaces with
that domain. A modal port at the back of the feed launches the TE₁₀
mode.

The +x face — directly in front of the aperture, where the main lobe
exits — is terminated by a PML slab so the strong forward radiation is
absorbed cleanly. The other five outer faces of the air box use a
first-order ABC (cheap, and the off-axis radiation is much weaker).

The horn cavity is built with the new geometry API: two parallel
rectangular profiles (``polygon`` in the yz-plane) lofted into the
frustum via :meth:`loft`.

Adapted from EMerge ``demo10_sgh``, scaled from W-band to X-band so the
full (non-symmetry-exploited) model fits in a comfortable mesh budget.

Notebook-style:  Parameters -> Geometry -> Mesh -> Simulation
"""

# %% Parameters
import numpy as np
import rapidfem

mm = 1e-3

# WR-90 feed waveguide
wga, wgb = 22.86 * mm, 10.16 * mm
Lfeed = 15.0 * mm

# Horn taper — modest gain, kept compact so the full (non-symmetry) model
# stays in a sensible mesh budget.
Lhorn = 50.0 * mm
WH, HH = 30.0 * mm, 22.0 * mm

# Radiation padding outside the horn (air buffer between the aperture
# and the PML slab — the wave needs a few λ/8 elements to look planar
# before the PML sees it).
LPAD = 15.0 * mm

# PML slab thickness for the +x absorption. ~λ/2 at f_max is plenty for
# 1st-order radiation; the slab gets ~2 tets across so DOF cost is low.
PML_T = 15.0 * mm

# Sweep — 15 pts across X-band, finely sampling the |S11| dip near 10 GHz
FREQUENCIES = np.linspace(8.0e9, 12.0e9, 15)
F0 = 10.0e9

# Global mesh cap — λ_air / 8 at f_max. Coarser than the usual λ/12 to
# keep DOF count manageable for a tutorial; far-field gain is still
# within ~0.5 dB of a converged solution.
MAXH = rapidfem.lambda_maxh(f_max=11.0e9, per_lambda=8)


# %% Geometry + Materials
# Air box wraps the horn on five sides; the +x face is occupied by a PML
# slab in front of the aperture to absorb the main lobe cleanly.
AIR_X0, AIR_X1 = -Lfeed,        Lhorn + LPAD       # inner air ends at AIR_X1
PML_X1          = AIR_X1 + PML_T                    # PML extends to here
AIR_Y0, AIR_Y1 = -WH / 2 - LPAD, WH / 2 + LPAD
AIR_Z0, AIR_Z1 = -HH / 2 - LPAD, HH / 2 + LPAD

g = rapidfem.Geometry()

air = g.box(AIR_X1 - AIR_X0, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
            position=(AIR_X0, AIR_Y0, AIR_Z0))
air.material = "air"

# PML slab in front of the aperture (+x absorption only). Spans the same
# yz extent as the inner air box so the air→PML interface is one clean
# planar rectangle at x = AIR_X1.
pml_xp = g.box(PML_T, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
               position=(AIR_X1, AIR_Y0, AIR_Z0))
pml_xp.material = "air"

# Feed waveguide cavity — sits inside the air box from x=-Lfeed to x=0.
feed = g.box(Lfeed, wga, wgb, position=(-Lfeed, -wga / 2, -wgb / 2))
feed.material = "air"

# Horn cavity — frustum from the throat at x=0 to the aperture at x=Lhorn,
# built by lofting between two parallel rectangular profiles in yz.
throat = g.polygon([
    (0, -wga / 2, -wgb / 2), (0,  wga / 2, -wgb / 2),
    (0,  wga / 2,  wgb / 2), (0, -wga / 2,  wgb / 2),
])
aperture = g.polygon([
    (Lhorn, -WH / 2, -HH / 2), (Lhorn,  WH / 2, -HH / 2),
    (Lhorn,  WH / 2,  HH / 2), (Lhorn, -WH / 2,  HH / 2),
])
horn = g.loft(throat, aperture)
horn.material = "air"

# Conformal cuts — feed, horn, and the PML slab all become conformal
# sub-regions of the surrounding air box.
g.fragment(air, feed, horn, pml_xp)

# Modal port at the back of the feed waveguide
feed.faces.min(axis="x").name = "feed_port"

# Feed waveguide PEC walls — the four yz-side faces (the throat at x=0 and
# the port at x=-Lfeed are excluded by selecting on y/z extremes).
feed.faces.min(axis="y").name = "pec"
feed.faces.max(axis="y").name = "pec"
feed.faces.min(axis="z").name = "pec"
feed.faces.max(axis="z").name = "pec"

# The four trapezoidal flares of the horn — picked by x-centroid strictly
# inside (0, Lhorn). The throat (x=0) and aperture (x=Lhorn) caps are
# interior interfaces with the feed and outer air, so we skip them.
horn.faces.where(lambda c, b: 1e-6 < c[0] < Lhorn - 1e-6).name = "pec"

# PML face naming.
#   - The shared air↔PML interface at x = AIR_X1 must stay unnamed so it
#     does NOT get tagged as PEC (which would wall the inner air off
#     from the absorber and the wave would bounce instead of decay).
#   - The five outer faces of the PML slab terminate the absorber as PEC.
pml_xp.name = "pml_xp"
pml_xp.faces.min(axis="x").name = "_iface"   # x = AIR_X1, shared with inner air
for face in pml_xp.faces:
    if face.name is None:
        face.name = "pec"

# Five outer faces of the air box → ABC. The +x face is shared with the
# PML and is already named "_iface" or absent (depending on how fragment
# split it), so it won't be re-tagged.
for face in air.faces:
    if face.name is None:
        face.name = "abc"

# Mesh sizing — the feed has the smallest features (wgb = 10 mm ≈ λ/3),
# so cap it tighter than the outer radiation region.
feed.maxh = wgb / 3
horn.maxh = MAXH

# PML field decays exponentially — 2 tets across the slab thickness is
# enough. Without this it inherits MAXH and bloats DoF count for free.
pml_xp.maxh = 2 * MAXH

rapidfem.show(g)


# %% Mesh
g.mesh(maxh=MAXH)
rapidfem.show(g)


# %% Simulation
sim = (
    rapidfem.SimulationBuilder()
    .mesh_from(g)
    .frequencies(FREQUENCIES)
    .pec("pec")
    .rect_waveguide("feed_port", mode=(1, 0), power=1.0)
    .abc("abc", order=1)
    .pml("pml_xp", direction=(1, 0, 0), inner_face=Lhorn + LPAD, thickness=PML_T)
    .material("air", er=1.0)
    .build()
)
result = sim.run_sweep()
rapidfem.show(sim)
rapidfem.show(result)

print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")
print(f"|S11| range: {min(abs(result.sparams[i, 0, 0]) for i in range(len(FREQUENCIES))):.3f} – "
      f"{max(abs(result.sparams[i, 0, 0]) for i in range(len(FREQUENCIES))):.3f}")

# Far-field directivity at the centre of the band
fi0 = int(min(range(len(FREQUENCIES)), key=lambda i: abs(FREQUENCIES[i] - F0)))
pattern = sim.compute_farfield(result, freq_idx=fi0, port_idx=0, n_theta=91, n_phi=72)
if pattern is not None:
    print(f"Far-field @ {FREQUENCIES[fi0] / 1e9:.1f} GHz: "
          f"D = {pattern.peak_directivity_dbi:.2f} dBi, "
          f"G = {pattern.peak_gain_dbi:.2f} dBi")
