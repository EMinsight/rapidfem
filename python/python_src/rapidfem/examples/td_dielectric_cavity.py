"""Time-domain DGTD on the geometry API — a dielectric-filled cavity.

Unlike ``ProblemTD.box`` (a bare validation cavity), this builds the
time-domain problem from a meshed :class:`~rapidfem.Geometry`: an
arbitrary unstructured tetrahedral mesh with a material attached through
the same physics API the frequency-domain solver uses. The DG operator
runs end-to-end on that mesh — heterogeneous materials and all.
"""

# %% Parameters
import numpy as np

import rapidfem as rf

mm = 1e-3
L = 50.0 * mm          # cubic cavity edge
ER = 4.0               # dielectric fill

# %% Geometry + material: a PEC box filled with a dielectric.
g = rf.Geometry(maxh=L / 11)        # finer mesh, affordable on the GPU
cavity = g.box(L, L, L, material=rf.Dielectric(er=ER))
rf.PEC(*cavity.faces.unassigned)        # all six walls are PEC
g.mesh()
rf.show(g)

# %% Build the time-domain problem from the meshed geometry
ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"ProblemTD from a meshed geometry - {ptd.n_dof // 60} tets, "
      f"{ptd.n_dof} state DOFs, dielectric er = {ER}")

# %% Propagate an impulse — the DG operator stays stable on the
#    unstructured mesh, and the upwind flux dissipates the field energy.
y0 = np.zeros(ptd.n_dof)
y0[ptd.probe_dof([0.5 * L, 0.5 * L, 0.5 * L], field="E", component="z")] = 1.0
traj = ptd.transient(y0, dt=5e-12, steps=140, device="gpu")

amp = np.linalg.norm(traj, axis=1)
assert np.all(np.isfinite(traj)), "transient must stay finite"
print(f"transient - {traj.shape[0]} snapshots, "
      f"amplitude {amp[0]:.3f} -> {amp[-1]:.3f} (upwind dissipation)")
print("time-domain DGTD runs end-to-end on the geometry / material API")

# %% Visualise — the impulse spreading and dissipating in the cavity
rf.show(traj)
