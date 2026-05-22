"""Time-domain DGTD cavity — the ProblemTD model-export API.

Where ProblemFD is an analysis tool (geometry in, S-parameters out),
ProblemTD compiles a cavity into a linear ODE ``dy/dt = A·y`` and exposes it
at every level: the right-hand side, the verbatim sparse operator, a
matrix-free exponential stepper, and a turnkey transient run.
"""

# %% Parameters
import numpy as np
import rapidfem as rf

mm = 1e-3
L = 30.0 * mm                        # cubic PEC cavity edge

# %% Build a PEC air cavity through the geometry API
g = rf.Geometry(maxh=L / 8)
air = g.box(L, L, L, material=rf.Air())
rf.PEC(*air.faces.unassigned)        # closed cavity, six PEC walls
g.mesh()
rf.show(g)

ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"DGTD cavity - {ptd.n_dof // 60} tets, {ptd.n_dof} state DOFs")

# %% Low level: the ODE right-hand side and the verbatim operator
rng = np.random.default_rng(0)
y0 = rng.standard_normal(ptd.n_dof)

dy = ptd.rhs(y0)                  # dy/dt = A·y
A = ptd.state_space()             # the explicit sparse operator A
print(f"state-space A: {A.shape[0]}x{A.shape[1]}, {A.nnz} nonzeros "
      f"({100 * A.nnz / A.shape[0] ** 2:.1f}% dense)")

# %% Turnkey: seed a field pulse and propagate it in time
# A localised pulse spreading through the cavity reads far better than
# random noise; the model-export verbs above accept any state vector.
y_pulse = np.zeros(ptd.n_dof)
y_pulse[ptd.probe_dof((0.5 * L, 0.5 * L, 0.5 * L), field="E", component="z")] = 1.0
traj = ptd.transient(y_pulse, dt=4e-12, steps=160, device="gpu")
print(f"transient run - {traj.shape[0]} snapshots of {traj.shape[1]} DOFs")

# The exponential propagator is exact for the linear homogeneous system at
# any step size: the step is set by the output cadence, not a CFL limit.

# %% Visualise: play the trajectory back as a 3D field animation
rf.show(traj)
