"""
EMerge demo0-style: parallel-plate TL with UserDefinedPort at each end.
Geometry: 20mm × 20mm × 50mm box. PEC top+bottom, PMC sides, ports at z=0/L.
Mode: Ey constant (TEM parallel plate).

Output: tests/validation/emerge_parallel_plate.csv
"""
from __future__ import annotations
import os
import numpy as np

import emerge as em
from compare import save_csv

mm = 0.001
W = 20 * mm
H = 20 * mm
L = 50 * mm

f1, f2, nf = 8.0e9, 12.0e9, 9

model = em.Simulation("ParallelPlateValidation")
model.check_version("2.4.3")
model.set_solver(em.SolverSuperLU('none'))

box = em.geo.Box(W, H, L, position=(-W/2, -H/2, 0))

model.mw.set_resolution(0.15)
model.mw.set_frequency_range(f1, f2, nf)
model.commit_geometry()
model.generate_mesh()


def Ey_field(k0, x, y, z):
    return np.ones_like(x)


p1 = model.mw.bc.UserDefinedPort(box.bottom, 1, Ey=Ey_field)
p2 = model.mw.bc.UserDefinedPort(box.top, 2, Ey=Ey_field)
model.mw.bc.PMC(box.left + box.right)

data = model.mw.run_sweep()

freqs = np.array(data.scalar.grid.freq, dtype=float)
n = len(freqs)
s = np.zeros((n, 2, 2), dtype=complex)
for i in range(2):
    for j in range(2):
        s[:, i, j] = np.asarray(data.scalar.grid.S(i + 1, j + 1))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emerge_parallel_plate.csv")
save_csv(out, freqs, s)
print(f"Wrote {out}")
print("EMerge parallel plate results:")
for k, f in enumerate(freqs):
    print(f"  f={f/1e9:5.2f}GHz  |S11|={abs(s[k,0,0]):.4f}  |S21|={abs(s[k,1,0]):.4f}")
