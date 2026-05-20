"""Time-domain model-order reduction — the ProblemTD `reduce` verb.

A DGTD cavity compiles to a linear ODE ``dy/dt = A·y`` with thousands of
state DOFs. For a *given* initial state, a Krylov model-order-reduced
model captures the propagation in a subspace of a few dozen dimensions:
``reduce()`` builds it, and the reduced model then propagates that state
(and its Krylov orbit) at a fraction of the cost — with no loss of
accuracy inside the subspace.
"""

# %% Build the time-domain problem — a unit cubic PEC cavity
import time

import numpy as np

import rapidfem as rf

ptd = rf.ProblemTD.box(
    size=(1.0, 1.0, 1.0),   # cavity dimensions
    cells=(3, 3, 3),        # structured-mesh resolution
    order=2,                # DG polynomial order
    flux="upwind",
)
print(f"DGTD cavity — {ptd.n_dof} state DOFs")

# %% An initial field state to propagate
rng = np.random.default_rng(0)
y0 = rng.standard_normal(ptd.n_dof)

# %% Reduce — project the operator onto a Krylov subspace seeded by y0
rom = ptd.reduce(y0, dim=60)
print(f"reduced model — order r={rom.r}, from n={rom.n} "
      f"({rom.n / rom.r:.0f}x smaller)")

# %% The reduced model reproduces the full exponential propagation
for t in (0.05, 0.2, 0.5):
    y_full = ptd.step(y0, t)
    y_rom = rom.propagate(y0, t)
    rel = np.linalg.norm(y_rom - y_full) / np.linalg.norm(y_full)
    print(f"  t={t:.2f}:  ROM vs full step  rel.err = {rel:.2e}")

# %% Cost — the reduced propagation is just a small dense exponential
n_rep = 200
t0 = time.perf_counter()
for _ in range(n_rep):
    rom.propagate(y0, 0.3)
t_rom = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(n_rep):
    ptd.step(y0, 0.3)
t_full = time.perf_counter() - t0

print(f"{n_rep} propagations — ROM {t_rom:.3f}s vs full {t_full:.3f}s "
      f"({t_full / max(t_rom, 1e-9):.0f}x speedup)")
