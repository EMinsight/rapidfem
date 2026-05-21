"""Time-domain DGTD — a dielectric ring resonator.

A high-permittivity ceramic ring sits in an air-filled PEC cavity. An
impulse lit up at one point on the ring couples into it: the high-εᵣ ring
traps the field, which then runs around the ring as a travelling
whispering-gallery-style field while it slowly leaks back into the
cavity. The 3-D animation shows the energy orbiting the ring.

Unlike the box-cavity TD examples, this exercises the DGTD operator on a
curved, unstructured tetrahedral mesh — a torus embedded in air — built
through the geometry API.
"""

# %% Parameters
import numpy as np

import rapidfem as rf

mm = 1e-3
R_MAJ = 11.0 * mm        # ring radius — tube-centre to torus axis
R_MIN = 2.6 * mm         # ring tube (cross-section) radius
ER = 10.0                # high-εᵣ ceramic ring
BOX = 38.0 * mm          # cubic air-cavity edge

# %% Geometry — a dielectric ring embedded in an air-filled PEC cavity
g = rf.Geometry(maxh=5.0 * mm)
air = g.box(BOX, BOX, BOX, position=(-BOX / 2, -BOX / 2, -BOX / 2),
            material=rf.Air())
ring = g.torus(R_MAJ, R_MIN, material=rf.Dielectric(er=ER), maxh=2.4 * mm)
g.fragment(air, ring)              # embed the ring in the air volume

# Only the 6 axis-aligned cavity walls are PEC — selecting min/max per axis
# leaves the air↔ring interface faces (exposed by fragment) un-walled.
rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"),
       air.faces.min(axis="y"), air.faces.max(axis="y"),
       air.faces.min(axis="z"), air.faces.max(axis="z"))
g.mesh()
rf.show(g)

# %% Build the time-domain problem on the curved unstructured mesh
ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"DGTD ring resonator - {ptd.n_dof // 60} tets, "
      f"{ptd.n_dof} state DOFs, ring er = {ER}")

# %% Light up one point on the ring and let the field orbit it
y0 = np.zeros(ptd.n_dof)
y0[ptd.probe_dof((R_MAJ, 0.0, 0.0), field="E", component="z")] = 1.0
traj = ptd.transient(y0, dt=4e-12, steps=180)

amp = np.linalg.norm(traj, axis=1)
assert np.all(np.isfinite(traj)), "transient must stay finite"
print(f"transient - {traj.shape[0]} snapshots, "
      f"amplitude {amp[0]:.3f} -> {amp[-1]:.3f}")

# %% Visualise — the field circulating in the dielectric ring
rf.show(traj)
