"""
WR-90 straight waveguide section in EMerge.
Two rectangular waveguide ports (TE10), no obstacles.
Expected physics: |S11| ~ 0, |S21| ~ 1, phase = exp(-j*beta*L).

Output: tests/validation/emerge_wr90.csv
"""
from __future__ import annotations
import os
import numpy as np

import emerge as em
from compare import save_csv

mm = 0.001

# WR-90 dimensions
a = 22.86 * mm   # broad wall
b = 10.16 * mm   # narrow wall
L = 30.0 * mm    # length

f1, f2, nf = 9.0e9, 11.0e9, 11

model = em.Simulation("WR90Validation")
model.check_version("2.4.3")

# Box: x ∈ [0,a], y ∈ [0,b], z ∈ [0,L]
wg = em.geo.Box(a, b, L, position=(0, 0, 0))

model.mw.set_resolution(0.15)
model.mw.set_frequency_range(f1, f2, nf)
model.commit_geometry()
model.generate_mesh()

# Two rect waveguide ports at z=0 and z=L (front/back faces)
# In emerge box face naming: bottom = -z, top = +z, front = -y, back = +y, left = -x, right = +x
# We want the z faces.
p1 = model.mw.bc.RectangularWaveguide(wg.bottom, 1)
p2 = model.mw.bc.RectangularWaveguide(wg.top, 2)

data = model.mw.run_sweep()

freqs = np.array(data.scalar.grid.freq, dtype=float)
n = len(freqs)
s = np.zeros((n, 2, 2), dtype=complex)
for i in range(2):
    for j in range(2):
        s[:, i, j] = np.asarray(data.scalar.grid.S(i + 1, j + 1))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emerge_wr90.csv")
save_csv(out, freqs, s)
print(f"\nWrote {out}")
print(f"\nEMerge WR-90 results:")
for k, f in enumerate(freqs):
    print(f"  f={f/1e9:5.2f}GHz  |S11|={abs(s[k,0,0]):.4f}  |S21|={abs(s[k,1,0]):.4f}")
