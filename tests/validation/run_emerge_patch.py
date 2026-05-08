"""
Run a simple edge-fed rectangular patch antenna in EMerge.
Geometry mirrors scripts/patch_antenna.py (rapidfem) for direct comparison.

Output: tests/validation/emerge_patch.csv (S11 vs frequency).
"""
from __future__ import annotations
import os
import sys
import numpy as np

# Headless: prevent any plot/view from blocking
os.environ["EMERGE_NO_GUI"] = "1"

import emerge as em
from compare import save_csv

mm = 0.001

# Geometry (matches scripts/patch_antenna.py)
sub_w = 60 * mm
sub_l = 60 * mm
sub_h = 1.6 * mm
er_sub = 4.4

patch_w = 38 * mm
patch_l = 29 * mm

feed_x = 0.0
feed_y = -patch_l / 2
feed_width = 1.5 * mm

air_pad = 25 * mm
air_pad_top = 25 * mm

f1, f2, nf = 1.5e9, 3.5e9, 21

model = em.Simulation("PatchValidation")
model.check_version("2.4.3")

# Substrate at z=0 to z=sub_h, sphere airbox covers everything.
# Ground = substrate.bottom face selection (no separate plate, avoids coplanar PLC issues).
substrate = em.geo.Box(sub_w, sub_l, sub_h, position=(-sub_w / 2, -sub_l / 2, 0))
substrate.set_material(em.Material(er_sub))

R_air = max(sub_w, sub_l) / 2 + air_pad_top
air = em.geo.Sphere(R_air).background()

# Patch at z=sub_h
patch = em.geo.XYPlate(patch_w, patch_l, position=(-patch_w / 2, -patch_l / 2, sub_h))
patch.set_material(em.lib.PEC)

# Lumped port: vertical plate at y=feed_y from z=0 to z=sub_h
port_plate = em.geo.Plate(
    np.array([feed_x - feed_width / 2, feed_y, 0]),
    np.array([feed_width, 0, 0]),
    np.array([0, 0, sub_h]),
)

model.mw.set_resolution(0.2)
model.mw.set_frequency_range(f1, f2, nf)
model.commit_geometry()

# Mesh refinement near port
model.mesher.set_face_size(port_plate, 0.4 * mm)
model.mesher.set_boundary_size(patch, 2 * mm)

model.generate_mesh()

# Boundary conditions
model.mw.bc.PEC(substrate.bottom)  # Ground plane
port_bc = model.mw.bc.LumpedPort(port_plate, 1, width=feed_width, height=sub_h,
                                 direction=em.ZAX, Z0=50)
abc = model.mw.bc.AbsorbingBoundary(air.boundary())

# Run sweep
data = model.mw.run_sweep()

# Extract S11
freqs = np.array(data.scalar.grid.freq, dtype=float)
n = len(freqs)
s = np.zeros((n, 1, 1), dtype=complex)
for k, f in enumerate(freqs):
    s[k, 0, 0] = data.scalar.grid.S(1, 1, f)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emerge_patch.csv")
save_csv(out, freqs, s)
print(f"Wrote {out}")
print("EMerge S11:")
for k, f in enumerate(freqs):
    print(f"  f={f/1e9:.4f}GHz  |S11|={abs(s[k,0,0]):.4f}")
