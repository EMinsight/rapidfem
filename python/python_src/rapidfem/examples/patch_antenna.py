"""Edge-fed microstrip patch antenna on FR-4 — single frequency, with far-field.

A ~2.4 GHz patch with a lumped feed at the radiating edge, a full PML
enclosure on the four side walls AND the +z cap, and a ground plane at
the bottom. After the solve, ``sim.compute_farfield`` gives a directivity
pattern.

The PML is built from five non-overlapping cuboid slabs (one per outer
face of the air box). To avoid the PML-corner ambiguity (each volume can
only carry a single absorption direction), the ±x slabs are extended in
y to cover the y-corners, and the +z slab is extended in both x and y to
cover the top corners. Every outgoing ray hits exactly one PML on its
way out.

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

# Air padding around the antenna (PML sits beyond this).
# A half-wavelength of headroom in z keeps the PML cap far enough from
# the patch for clean far-field calculation (λ/2 ≈ 62 mm at 2.4 GHz).
PAD_XY = 25 * mm
PAD_Z = 60 * mm

# PML thickness on every outer face
PML_T = 20 * mm

# Frequency sweep around the design point — 21 pts across 2.0–2.8 GHz
# shows the |S11| dip at resonance.
FREQUENCIES = np.linspace(2.0e9, 2.8e9, 21)
F0 = 2.4e9

# Mesh density: the global cap is the air-wavelength bound at f_max.
# Higher-εᵣ volumes (the substrate) get their own tighter size below.
MAXH = rapidfem.lambda_maxh(f_max=2.8e9)


# %% Geometry + Materials
total_w = SUB_W + 2 * PAD_XY
total_l = SUB_L + 2 * PAD_XY
AIR_TOP = SUB_H + PAD_Z          # top of the regular (non-PML) air region

# Inner-box outer extents (these define where PML slabs begin).
X_OUT = total_w / 2
Y_OUT = total_l / 2

g = rapidfem.Geometry()

# Inner air column up to AIR_TOP — patch radiates into this volume.
air = g.box(total_w, total_l, AIR_TOP, position=(-X_OUT, -Y_OUT, 0))

# Five PML slabs surrounding the air box (no -z slab — ground plane there).
# ±x slabs span the full y range incl. corners, ±y slabs the inner x only,
# and the +z slab extends in x+y to seal the upper corners.
pml_xp = g.box(PML_T, total_l + 2 * PML_T, AIR_TOP,
               position=(X_OUT, -Y_OUT - PML_T, 0))
pml_xm = g.box(PML_T, total_l + 2 * PML_T, AIR_TOP,
               position=(-X_OUT - PML_T, -Y_OUT - PML_T, 0))
pml_yp = g.box(total_w, PML_T, AIR_TOP,
               position=(-X_OUT, Y_OUT, 0))
pml_ym = g.box(total_w, PML_T, AIR_TOP,
               position=(-X_OUT, -Y_OUT - PML_T, 0))
pml_zp = g.box(total_w + 2 * PML_T, total_l + 2 * PML_T, PML_T,
               position=(-X_OUT - PML_T, -Y_OUT - PML_T, AIR_TOP))

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

# Conformal cuts (all faces share edges)
g.fragment(air, pml_xp, pml_xm, pml_yp, pml_ym, pml_zp, sub, patch, feed)

# Conductive surfaces
sub.faces.min(axis="z").name = "ground_pec"
patch.name = "patch_pec"
feed.name = "feed"

# Name every PML volume so the builder can resolve it.
pml_xp.name = "pml_xp"
pml_xm.name = "pml_xm"
pml_yp.name = "pml_yp"
pml_ym.name = "pml_ym"
pml_zp.name = "pml_zp"

# Outer faces of the PML enclosure → PEC.
for vol in (pml_xp, pml_xm, pml_yp, pml_ym, pml_zp):
    for face in vol.faces:
        if face.name is None:
            face.name = "pec"

# Materials
sub.material = "fr4"
air.material = "air"

# Substrate wavelength is √εᵣ shorter than air → resolve it there explicitly.
sub.maxh = rapidfem.lambda_maxh(f_max=2.8e9, er_max=ER_SUB)

# PML fields decay exponentially — ~2 tets across the thickness is enough.
# Without this the PML inherits MAXH and bloats the DoF count for nothing.
for pml in (pml_xp, pml_xm, pml_yp, pml_ym, pml_zp):
    pml.maxh = 2 * MAXH

rapidfem.show(g)


# %% Mesh
# Auto-refine thin features (the 1.6 mm substrate, 1.5 mm feed) so they get
# at least 3 tets across their smallest dimension. Volumes that are large
# in every direction are untouched; explicit `obj.maxh = ...` overrides win.
g.auto_refine_features(base_maxh=MAXH)
g.mesh(maxh=MAXH)
rapidfem.show(g)


# %% Simulation
sim = (
    rapidfem.SimulationBuilder()
    .mesh_from(g)
    .frequencies(FREQUENCIES)
    .pec("ground_pec", "patch_pec", "pec")
    .lumped_port("feed", direction=(0, 0, 1), z0=50.0)
    .pml("pml_xp", direction=( 1, 0, 0), inner_face= X_OUT,    thickness=PML_T)
    .pml("pml_xm", direction=(-1, 0, 0), inner_face=-X_OUT,    thickness=PML_T)
    .pml("pml_yp", direction=( 0, 1, 0), inner_face= Y_OUT,    thickness=PML_T)
    .pml("pml_ym", direction=( 0,-1, 0), inner_face=-Y_OUT,    thickness=PML_T)
    .pml("pml_zp", direction=( 0, 0, 1), inner_face= AIR_TOP,  thickness=PML_T)
    .material("fr4", er=ER_SUB)
    .material("air", er=1.0)
    .build()
)
result = sim.run_sweep()
rapidfem.show(sim)
rapidfem.show(result)

print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")

# Locate the resonance — frequency with the smallest |S11|.
mags = [abs(result.sparams[i, 0, 0]) for i in range(len(FREQUENCIES))]
fi_min = int(min(range(len(mags)), key=lambda i: mags[i]))
print(f"|S11| min: {mags[fi_min]:.4f} @ {FREQUENCIES[fi_min]/1e9:.3f} GHz")

# Far-field pattern at the design frequency F0 (closest sample point).
fi0 = int(min(range(len(FREQUENCIES)), key=lambda i: abs(FREQUENCIES[i] - F0)))
pattern = sim.compute_farfield(result, freq_idx=fi0, port_idx=0, n_theta=91, n_phi=72)
if pattern is not None:
    print(f"Far-field @ {FREQUENCIES[fi0]/1e9:.2f} GHz: "
          f"D = {pattern.peak_directivity_dbi:.2f} dBi, "
          f"G = {pattern.peak_gain_dbi:.2f} dBi")
