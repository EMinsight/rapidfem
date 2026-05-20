"""Time-domain pulse propagation, field probes and VTK animation export.

Two workflows on one DGTD cavity:

* ``driven_transient`` — inject a soft Gaussian-pulse source and record
  the field at a probe point (the time-domain "measurement");
* ``transient`` + ``export_vtk`` — propagate a field state and write the
  whole trajectory as a ParaView-openable VTK animation.
"""

# %% Build the cavity
import os
import tempfile

import numpy as np

import rapidfem as rf

ptd = rf.ProblemTD.box(
    size=(1.0, 1.0, 1.0),
    cells=(3, 3, 3),
    order=2,
    flux="upwind",
)
print(f"DGTD cavity — {ptd.n_dof} state DOFs")

# %% Driven run — a soft Gaussian pulse injected at the cavity centre
pulse = rf.GaussianPulse(t0=0.4, tau=0.1, f0=0.0)
times, response = ptd.driven_transient(
    source=([0.5, 0.5, 0.5], "E", "z"),         # where / what to inject
    waveform=pulse,                              # the excitation g(t)
    probes=[([0.25, 0.25, 0.5], "E", "z")],      # where to measure
    dt=0.01,
    steps=200,
)
print(f"probe — peak |E_z| = {np.abs(response[0]).max():.4f} "
      f"over {len(times)} samples")

# %% Free transient — propagate an initial field and capture every snapshot
y0 = np.zeros(ptd.n_dof)
y0[ptd.probe_dof([0.5, 0.5, 0.5], field="E", component="z")] = 1.0
traj = ptd.transient(y0, dt=0.02, steps=120)
print(f"transient — {traj.shape[0]} full-field snapshots")

# %% Export the trajectory as a VTK animation — open the .pvd in ParaView
out_dir = os.path.join(tempfile.gettempdir(), "rapidfem_td_field")
pvd = ptd.export_vtk(
    traj,
    os.path.join(out_dir, "cavity"),
    times=np.arange(traj.shape[0]) * 0.02,
)
print(f"VTK animation written — open in ParaView:\n  {pvd}")
