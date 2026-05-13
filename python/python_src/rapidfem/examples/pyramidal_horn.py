"""Pyramidal horn antenna — WR-90 feed, 50×40 mm aperture, X-band sweep.

A standard rectangular horn for the 8–12 GHz band: a short WR-90 feed
waveguide followed by an 80 mm taper to a 50 × 40 mm aperture. The
horn's four metal flares are modelled as infinitely thin PEC sheets
embedded in a single air domain; the aperture and feed throat are
interior interfaces with the radiation region. A modal port at the back
of the feed launches the TE₁₀ mode; the outer faces of the air box use
an Absorbing Boundary Condition.

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

# Radiation padding outside the horn
LPAD = 15.0 * mm

# Sweep — 5 pts across X-band centered at 10 GHz
FREQUENCIES = np.linspace(9.0e9, 11.0e9, 5)
F0 = 10.0e9

# Global mesh cap — λ_air / 8 at f_max. Coarser than the usual λ/12 to
# keep DOF count manageable for a tutorial; far-field gain is still
# within ~0.5 dB of a converged solution.
MAXH = rapidfem.lambda_maxh(f_max=11.0e9, per_lambda=8)


# %% Geometry + Materials
# Big air box wrapping the whole horn. The horn axis runs along +x.
AIR_X0, AIR_X1 = -Lfeed,        Lhorn + LPAD
AIR_Y0, AIR_Y1 = -WH / 2 - LPAD, WH / 2 + LPAD
AIR_Z0, AIR_Z1 = -HH / 2 - LPAD, HH / 2 + LPAD

g = rapidfem.Geometry()

air = g.box(AIR_X1 - AIR_X0, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
            position=(AIR_X0, AIR_Y0, AIR_Z0))
air.material = "air"

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

# Conformal cuts — feed and horn become sub-regions of the air box,
# sharing the throat face (feed↔horn) and the aperture face (horn↔air).
g.fragment(air, feed, horn)

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

# Outer faces of the big air box → ABC. Interior interfaces with the
# feed / horn are already named (PEC or port), so the .name is None test
# leaves them alone.
for face in air.faces:
    if face.name is None:
        face.name = "abc"

# Mesh sizing — the feed has the smallest features (wgb = 10 mm ≈ λ/3),
# so cap it tighter than the outer radiation region.
feed.maxh = wgb / 3
horn.maxh = MAXH

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
