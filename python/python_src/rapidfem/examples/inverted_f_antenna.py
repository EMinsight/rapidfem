"""Printed inverted-F antenna (IFA) on FR-4 — single-layer 2.45 GHz design.

Antenna trace and ground plane are coplanar on the same copper layer of a
1 mm FR-4 board. Adapted from EMerge ``demo21`` and TI App Note SWRA117D.
"""

# %% Parameters
import numpy as np
import rapidfem as rf

mm = 1e-3

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

TRACE_W = D1 + L3 + L5 + L2 + L5 + L2 + W2

SUB_H = 1.0 * mm
ER_SUB = 4.4
GROUND_L = 30.0 * mm

PAD = 35.0 * mm

FREQUENCIES = np.linspace(2.0e9, 4.0e9, 21)
F0 = 2.45e9

MAXH = rf.lambda_maxh(f_max=3.0e9)
TRACE_MAXH = 0.4 * mm


# %% Geometry + Materials
SUB_X0, SUB_X1 = -D1, TRACE_W - D1
SUB_Y0, SUB_Y1 = -GROUND_L, L6 - D4 + W2
SUB_DX = SUB_X1 - SUB_X0
SUB_DY = SUB_Y1 - SUB_Y0

g = rf.Geometry(maxh=MAXH)

sub = g.box(SUB_DX, SUB_DY, SUB_H, position=(SUB_X0, SUB_Y0, -SUB_H),
            material=rf.Dielectric(er=ER_SUB),
            maxh=rf.lambda_maxh(f_max=3.0e9, er_max=ER_SUB))

AIR_X0, AIR_X1 = SUB_X0 - PAD, SUB_X1 + PAD
AIR_Y0, AIR_Y1 = SUB_Y0 - PAD, SUB_Y1 + PAD
AIR_Z0, AIR_Z1 = -SUB_H - PAD, PAD
air = g.box(AIR_X1 - AIR_X0, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
            position=(AIR_X0, AIR_Y0, AIR_Z0),
            material=rf.Air())

# 12 trace segments of the meandered F.
ant_segs_xy = [
    (0,                          0,                W1, L6 - D4),
    (0,                          L6 - D4,          L3, W2),
    (W1 + D5,                    0.5 * mm,         W2, L6 - D4 - 0.5 * mm),
    (L3 - W2,                    L6 - D4 - L4,     W2, L4),
    (L3,                         L6 - D4 - L4,     L5, W2),
    (L3 + L5,                    L6 - D4 - L4,     W2, L4),
    (L3 + L5,                    L6 - D4,          L2, W2),
    (L3 + L5 + L2 - W2,          L6 - D4 - L4,     W2, L4),
    (L3 + L5 + L2,               L6 - D4 - L4,     L5, W2),
    (L3 + L5 + L2 + L5,          L6 - D4 - L4,     W2, L4),
    (L3 + L5 + L2 + L5,          L6 - D4,          L2, W2),
    (L3 + L5 + L2 + L5 + L2 - W2, L6 - D4 - L1,    W2, L1),
]
ant_plates = [g.xy_plate(w, h, position=(x, y, 0), maxh=TRACE_MAXH)
              for x, y, w, h in ant_segs_xy]

port = g.xy_plate(W2, 0.5 * mm, position=(W1 + D5, 0, 0), maxh=TRACE_MAXH)
ground = g.xy_plate(SUB_DX, GROUND_L, position=(SUB_X0, SUB_Y0, 0))

g.fragment(air, sub, ground, port, *ant_plates)

# Physics
rf.LumpedPort(port, direction=(0, 1, 0), z0=50.0)
rf.PEC(*ant_plates, ground)
rf.ABC(*air.faces.outer, order=1)

rf.show(g)


# %% Mesh
g.auto_refine_features(base_maxh=MAXH)
g.mesh()
rf.show(g)


# %% Problem + Sweep
prob = rf.Problem(g)
result = prob.sweep(FREQUENCIES)
rf.show(prob)
rf.show(result)

print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")

mags = [abs(result.sparams[i, 0, 0]) for i in range(len(FREQUENCIES))]
fi_min = int(min(range(len(mags)), key=lambda i: mags[i]))
print(f"|S11| min: {mags[fi_min]:.4f} @ {FREQUENCIES[fi_min] / 1e9:.3f} GHz")

fi0 = int(min(range(len(FREQUENCIES)), key=lambda i: abs(FREQUENCIES[i] - F0)))
pattern = prob.farfield(result, freq_idx=fi0, port_idx=0, n_theta=91, n_phi=72)
if pattern is not None:
    print(f"Far-field @ {FREQUENCIES[fi0] / 1e9:.2f} GHz: "
          f"D = {pattern.peak_directivity_dbi:.2f} dBi, "
          f"G = {pattern.peak_gain_dbi:.2f} dBi")
