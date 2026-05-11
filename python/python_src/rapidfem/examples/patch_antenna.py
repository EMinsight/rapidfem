"""Edge-fed microstrip patch antenna on FR-4 — single frequency, with far-field.

A ~2.4 GHz patch with a lumped feed at the radiating edge, ABC walls all
around, ground plane on the bottom. After the solve, ``sim.compute_farfield``
gives a directivity pattern.

Notebook-style:  Parameters -> Geometry -> Mesh -> Simulation
"""

# %% Parameters
import numpy as np
import rapidfem

mm = 1e-3

# Substrate (FR-4)
SUB_W, SUB_L, SUB_H = 60 * mm, 60 * mm, 1.6 * mm
ER_SUB = 4.4

# Patch
PATCH_W, PATCH_L = 38 * mm, 29 * mm

# Lumped feed at the radiating edge
FEED_X, FEED_Y = 0.0, -PATCH_L / 2
FEED_WIDTH = 1.5 * mm

# Air padding around the antenna (ABC truncates this volume)
PAD_XY = 25 * mm
PAD_Z = 25 * mm

# Frequency
F0 = 2.4e9

# Mesh density
MAXH = 8 * mm


# %% Geometry + Materials
total_w = SUB_W + 2 * PAD_XY
total_l = SUB_L + 2 * PAD_XY
total_h = SUB_H + PAD_Z

g = rapidfem.Geometry()

# Air-box (encloses everything)
air = g.box(total_w, total_l, total_h, position=(-total_w / 2, -total_l / 2, 0))

# Substrate slab on the bottom
sub = g.box(SUB_W, SUB_L, SUB_H, position=(-SUB_W / 2, -SUB_L / 2, 0))

# Conducting patch on top of substrate
patch = g.xy_plate(PATCH_W, PATCH_L,
                   position=(-PATCH_W / 2, -PATCH_L / 2, SUB_H))

# Vertical lumped-port rectangle, on the substrate edge under the patch
feed = g.plate(
    p0=(FEED_X - FEED_WIDTH / 2, FEED_Y, 0),
    width=(FEED_WIDTH, 0, 0),
    height=(0, 0, SUB_H),
)

# Conformal cuts (so all faces share edges)
g.fragment(air, sub, patch, feed)

# Name the conductive + absorbing surfaces
sub.faces.min(axis="z").name = "ground_pec"
patch.name = "patch_pec"
feed.name = "feed"
air.faces.where(lambda c, _: abs(c[2] - total_h) < 1e-9).name = "abc"
air.faces.where(lambda c, _: abs(c[0] + total_w / 2) < 1e-9).name = "abc"
air.faces.where(lambda c, _: abs(c[0] - total_w / 2) < 1e-9).name = "abc"
air.faces.where(lambda c, _: abs(c[1] + total_l / 2) < 1e-9).name = "abc"
air.faces.where(lambda c, _: abs(c[1] - total_l / 2) < 1e-9).name = "abc"

# Materials
sub.material = "fr4"
air.material = "air"

rapidfem.show(g)


# %% Mesh
g.mesh(maxh=MAXH)
rapidfem.show(g)


# %% Simulation
sim = (
    rapidfem.SimulationBuilder()
    .mesh_from(g)
    .frequencies([F0])
    .pec("ground_pec", "patch_pec")
    .lumped_port("feed", direction=(0, 0, 1), z0=50.0)
    .abc("abc", order=1)
    .material("fr4", er=ER_SUB)
    .material("air", er=1.0)
    .build()
)
result = sim.run_sweep()
rapidfem.show(sim)
rapidfem.show(result)

s11 = abs(result.sparams[0, 0, 0])
print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")
print(f"|S11| @ {F0/1e9:.2f} GHz: {s11:.4f}")

pattern = sim.compute_farfield(result, freq_idx=0, port_idx=0, n_theta=91, n_phi=72)
if pattern is not None:
    print(f"Peak directivity: {pattern.peak_directivity_dbi:.2f} dBi")
    print(f"Peak gain:        {pattern.peak_gain_dbi:.2f} dBi")
