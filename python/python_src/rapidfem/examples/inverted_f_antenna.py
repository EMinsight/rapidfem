"""Printed inverted-F antenna (IFA) on FR-4 — single-layer 2.45 GHz design.

Antenna trace and ground plane are coplanar on the same copper layer of
a 1 mm FR-4 board. The trace is a meandered F-shape: a feed strip
shorted to ground at its base, a parallel feed strip excited via a
lumped port, a horizontal top arm tying them together, and a zig-zag
section that tunes the resonance toward 2.45 GHz.

The TI reference design is a 3D pour of finite-thickness copper; we
approximate it with an infinitesimal-thickness PEC layer, which shifts
the resonance somewhat higher than the published 2.45 GHz. The sweep is
extended accordingly so the |S₁₁| dip lands in band.

Adapted from EMerge ``demo21``, modelled after TI App Note SWRA117D:
https://www.ti.com/lit/an/swra117d/swra117d.pdf

Notebook-style:  Parameters -> Geometry -> Mesh -> Simulation
"""

# %% Parameters
import numpy as np
import rapidfem

mm = 1e-3

# Trace section lengths / widths (TI reference)
L1 = 3.94 * mm
L2 = 2.47 * mm
L3 = 4.76 * mm
L4 = 2.64 * mm
L5 = 1.77 * mm
L6 = 4.90 * mm
W1 = 0.90 * mm
W2 = 0.50 * mm
D1 = 0.50 * mm
D4 = 0.50 * mm
D5 = 0.65 * mm

# Overall trace-block width (x extent of all 12 segments combined)
TRACE_W = D1 + L3 + L5 + L2 + L5 + L2 + W2

# Substrate / ground plate
SUB_H = 1.0 * mm
ER_SUB = 4.4
GROUND_L = 30.0 * mm   # ground extends 30 mm to the -y side

# Air-box padding around the PCB. ABC reflectivity falls off with
# distance; a quarter wavelength at the lower band edge (~30 mm at
# 2.5 GHz) keeps return loss reasonably clean.
PAD = 35.0 * mm

# Sweep — wide enough that the resonance dip is well inside the band.
# An ideal-thin-conductor model resonates a bit higher than the original
# 3D-trace TI design (the bare ABC contributes some mistuning too); the
# upper end of the band catches it.
FREQUENCIES = np.linspace(2.0e9, 4.0e9, 21)
F0 = 2.45e9

# Global mesh cap: λ_air / 12 at f_max
MAXH = rapidfem.lambda_maxh(f_max=3.0e9)


# %% Geometry + Materials
# Substrate is centred so antenna is at y > 0 and ground at y < 0.
SUB_X0, SUB_X1 = -D1, TRACE_W - D1
SUB_Y0, SUB_Y1 = -GROUND_L, L6 - D4 + W2
SUB_DX = SUB_X1 - SUB_X0
SUB_DY = SUB_Y1 - SUB_Y0

g = rapidfem.Geometry()

# Substrate slab — top face at z=0, ground/antenna sit on top
sub = g.box(SUB_DX, SUB_DY, SUB_H, position=(SUB_X0, SUB_Y0, -SUB_H))
sub.material = "fr4"

# Air enclosure — ABC on the outer faces handles radiation
AIR_X0, AIR_X1 = SUB_X0 - PAD, SUB_X1 + PAD
AIR_Y0, AIR_Y1 = SUB_Y0 - PAD, SUB_Y1 + PAD
AIR_Z0, AIR_Z1 = -SUB_H - PAD, PAD
air = g.box(AIR_X1 - AIR_X0, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
            position=(AIR_X0, AIR_Y0, AIR_Z0))
air.material = "air"

# 12 segments of the meandered F. Each entry is (x, y, w, h) — lower-left
# corner and extents in the xy-plane at z=0.
ant_segs_xy = [
    (0,                          0,                W1, L6 - D4),               # 1: feed strip
    (0,                          L6 - D4,          L3, W2),                    # 2: top horizontal
    (W1 + D5,                    0.5 * mm,         W2, L6 - D4 - 0.5 * mm),    # 3: shorting strip
    (L3 - W2,                    L6 - D4 - L4,     W2, L4),                    # 4: meander down
    (L3,                         L6 - D4 - L4,     L5, W2),                    # 5: meander right
    (L3 + L5,                    L6 - D4 - L4,     W2, L4),                    # 6: meander up
    (L3 + L5,                    L6 - D4,          L2, W2),                    # 7: top connector
    (L3 + L5 + L2 - W2,          L6 - D4 - L4,     W2, L4),                    # 8: meander down
    (L3 + L5 + L2,               L6 - D4 - L4,     L5, W2),                    # 9
    (L3 + L5 + L2 + L5,          L6 - D4 - L4,     W2, L4),                    # 10
    (L3 + L5 + L2 + L5,          L6 - D4,          L2, W2),                    # 11
    (L3 + L5 + L2 + L5 + L2 - W2, L6 - D4 - L1,    W2, L1),                    # 12: end stub
]
ant_plates = [g.xy_plate(w, h, position=(x, y, 0)) for x, y, w, h in ant_segs_xy]

# Lumped port — small plate in the gap between ground (y < 0) and the
# shorting strip ant_3 (which starts at y = 0.5 mm). Bridges the gap
# along +y, so the port direction is +y.
port = g.xy_plate(W2, 0.5 * mm, position=(W1 + D5, 0, 0))

# Ground plane on the same copper layer, to the -y side of the antenna
ground = g.xy_plate(SUB_DX, GROUND_L, position=(SUB_X0, SUB_Y0, 0))

# Conformal cuts — substrate, air, and all top-layer copper pieces
g.fragment(air, sub, ground, port, *ant_plates)

# Tag PEC and port surfaces. All 12 antenna pieces share the name; the
# builder collects faces by name so they appear as a single conductor.
for plate in ant_plates:
    plate.name = "antenna_pec"
ground.name = "ground_pec"
port.name = "port"

# All outer faces of the air box → ABC. After fragment, interior faces
# (substrate-air interfaces, copper trace faces) are already named, so we
# only re-tag the ones still unnamed.
for face in air.faces:
    if face.name is None:
        face.name = "abc_outer"

# Substrate wavelength is √εᵣ shorter than air → resolve it there
sub.maxh = rapidfem.lambda_maxh(f_max=3.0e9, er_max=ER_SUB)

# Smallest trace dimension is W2 = 0.5 mm; resolve the trace ribbon with
# ~2 tets across the narrowest strip, otherwise the meander geometry
# washes out at the default MAXH (~12 mm).
TRACE_MAXH = 0.4 * mm
for plate in ant_plates:
    plate.maxh = TRACE_MAXH
port.maxh = TRACE_MAXH

rapidfem.show(g)


# %% Mesh
g.auto_refine_features(base_maxh=MAXH)
g.mesh(maxh=MAXH)
rapidfem.show(g)


# %% Simulation
sim = (
    rapidfem.SimulationBuilder()
    .mesh_from(g)
    .frequencies(FREQUENCIES)
    .pec("antenna_pec", "ground_pec")
    .lumped_port("port", direction=(0, 1, 0), z0=50.0)
    .abc("abc_outer", order=1)
    .material("fr4", er=ER_SUB)
    .material("air", er=1.0)
    .build()
)
result = sim.run_sweep()
rapidfem.show(sim)
rapidfem.show(result)

print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")

# Find the resonance — frequency with the smallest |S11|.
mags = [abs(result.sparams[i, 0, 0]) for i in range(len(FREQUENCIES))]
fi_min = int(min(range(len(mags)), key=lambda i: mags[i]))
print(f"|S11| min: {mags[fi_min]:.4f} @ {FREQUENCIES[fi_min] / 1e9:.3f} GHz")

# Far-field pattern at the design frequency F0 (closest sample point).
fi0 = int(min(range(len(FREQUENCIES)), key=lambda i: abs(FREQUENCIES[i] - F0)))
pattern = sim.compute_farfield(result, freq_idx=fi0, port_idx=0, n_theta=91, n_phi=72)
if pattern is not None:
    print(f"Far-field @ {FREQUENCIES[fi0] / 1e9:.2f} GHz: "
          f"D = {pattern.peak_directivity_dbi:.2f} dBi, "
          f"G = {pattern.peak_gain_dbi:.2f} dBi")
