"""
EMerge demo4-style patch antenna with inset feed (headless).
Geometry per demo4_patch_antenna.py with tweaks for non-interactive run.

Output: tests/validation/emerge_patch_inset.csv
"""
from __future__ import annotations
import os
import numpy as np

import emerge as em
from compare import save_csv

mm = 0.001

Wpatch = 53 * mm
Lpatch = 52 * mm
wline = 3.2 * mm
wstub = 7 * mm
lstub = 15.5 * mm
wsub = 100 * mm
hsub = 100 * mm
th = 1.524 * mm
Rair = 100 * mm

f1, f2, nf = 1.55e9, 1.60e9, 7

model = em.Simulation('PatchInset')
model.check_version("2.4.3")
model.set_solver(em.SolverSuperLU('none'))

dielectric = em.geo.Box(wsub, hsub, th, position=(-wsub/2, -hsub/2, -th))
air = em.geo.Sphere(Rair).background()

rpatch = em.geo.XYPlate(Wpatch, Lpatch, position=(-Wpatch/2, -Lpatch/2, 0))
ground = em.geo.XYPlate(wsub, hsub, position=(-wsub/2, -hsub/2, -th)).set_material(em.lib.PEC)

cutout1 = em.geo.XYPlate(wstub, lstub, position=(-wline/2 - wstub, -Lpatch/2, 0))
cutout2 = em.geo.XYPlate(wstub, lstub, position=(wline/2, -Lpatch/2, 0))
line = em.geo.XYPlate(wline, lstub, position=(-wline/2, -Lpatch/2, 0))

port = em.geo.Plate(
    np.array([-wline/2, -Lpatch/2, -th]),
    np.array([wline, 0, 0]),
    np.array([0, 0, th]),
)

rpatch = em.geo.remove(rpatch, cutout1)
rpatch = em.geo.remove(rpatch, cutout2)
rpatch = em.geo.add(rpatch, line)
rpatch.set_material(em.lib.PEC)

dielectric.set_material(em.Material(3.38, color="#207020", opacity=0.9))

model.mw.set_resolution(0.2)
model.mw.set_frequency_range(f1, f2, nf)
model.commit_geometry()
model.mesher.set_boundary_size(rpatch, 2 * mm)
model.mesher.set_face_size(port, 1 * mm)
model.generate_mesh()

port_bc = model.mw.bc.LumpedPort(port, 1, width=wline, height=th, direction=em.ZAX, Z0=50)
boundary_selection = air.boundary()
abc = model.mw.bc.AbsorbingBoundary(boundary_selection)

data = model.mw.run_sweep()

freqs = np.array(data.scalar.grid.freq, dtype=float)
n = len(freqs)
s = np.zeros((n, 1, 1), dtype=complex)
s[:, 0, 0] = np.asarray(data.scalar.grid.S(1, 1))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emerge_patch_inset.csv")
save_csv(out, freqs, s)
print(f"Wrote {out}")
print("EMerge demo4 patch antenna:")
for k, f in enumerate(freqs):
    print(f"  f={f/1e9:.4f}GHz  |S11|={abs(s[k,0,0]):.4f}")
