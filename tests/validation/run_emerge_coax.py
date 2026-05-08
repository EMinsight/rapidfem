"""
EMerge: simple coaxial line section, two CoaxPorts.
Expected physics: |S11| ≈ 0, |S21| ≈ 1, phase = -k*sqrt(er)*L.

Output: tests/validation/emerge_coax.csv
"""
from __future__ import annotations
import os
import numpy as np

import emerge as em
from compare import save_csv

mm = 0.001
ri = 0.5 * mm   # inner conductor radius
ro = 1.7 * mm   # outer conductor radius (Z0 ≈ 73 Ω in air)
L = 30.0 * mm

f1, f2, nf = 1.0e9, 5.0e9, 9

model = em.Simulation("CoaxValidation")
model.check_version("2.4.3")
model.set_solver(em.SolverSuperLU('none'))

# Outer cylinder = dielectric region (air); inner cylinder = PEC center conductor
outer = em.geo.Cylinder(ro, L, em.cs(origin=(0, 0, 0)), Nsections=24)
inner = em.geo.Cylinder(ri, L, em.cs(origin=(0, 0, 0)), Nsections=16).set_material(em.lib.PEC)

model.mw.set_resolution(0.15)
model.mw.set_frequency_range(f1, f2, nf)
model.commit_geometry()
model.generate_mesh()

# Coax ports at the two end faces
p1 = model.mw.bc.CoaxPort(outer.face('-z'), 1, rad_in_out=(ri, ro),
                          cs=em.cs('xyz', origin=(0, 0, 0)), er=1.0)
p2 = model.mw.bc.CoaxPort(outer.face('+z'), 2, rad_in_out=(ri, ro),
                          cs=em.cs('xyZ', origin=(0, 0, L)), er=1.0)

data = model.mw.run_sweep()

freqs = np.array(data.scalar.grid.freq, dtype=float)
n = len(freqs)
s = np.zeros((n, 2, 2), dtype=complex)
for i in range(2):
    for j in range(2):
        s[:, i, j] = np.asarray(data.scalar.grid.S(i + 1, j + 1))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emerge_coax.csv")
save_csv(out, freqs, s)
print(f"Wrote {out}")
print("EMerge Coax results:")
for k, f in enumerate(freqs):
    print(f"  f={f/1e9:5.2f}GHz  |S11|={abs(s[k,0,0]):.4f}  |S21|={abs(s[k,1,0]):.4f}")
