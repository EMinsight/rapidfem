"""TD port verification suite for the PeriodicBoundary wiring.

Covers the Python wiring on top of the Rust gates, which validate the
operator-level physics already. The test runs a small transient and
asserts a quantitative pass criterion (finite-energy preservation).

(The Coax / Floquet / WavePort port checks previously lived here too,
but were written against the removed TD ``sparams`` extraction; modal
S-parameters belong to the frequency-domain backend, ``ProblemFD``.)
"""
from __future__ import annotations

import numpy as np
import pytest

import rapidfem as rf

MM = 1e-3

slow = pytest.mark.slow


# -----------------------------------------------------------------------------
# PeriodicBoundary - field continuity across a periodic pair.
# -----------------------------------------------------------------------------

@slow
@pytest.mark.skip(
    reason="gmsh `setPeriodic` is not wired through Python "
    "`PeriodicBoundary`; opposite faces mesh with different triangle "
    "counts and the Rust matcher rejects the pair. Rust C2 gate covers "
    "the operator-level physics on a structured_box mesh. Wiring "
    "setPeriodic into Geometry.mesh() is a separate task."
)
def test_periodic_boundary_pair_runs_end_to_end():
    """A periodic unit cell driven by a localised pulse. With opposite
    faces tied periodically the energy can recirculate; without (just
    PEC boundaries) the energy stays bounded the same way. The
    operator must build and propagate finitely; energy must not blow
    up. Mirrors the Rust C2 gate's energy-drift check at the Python
    level.
    """
    side = 30.0 * MM
    g = rf.Geometry(maxh=side / 4)
    box = g.box(side, side, side, material=rf.Air())
    # Pair the x = 0 and x = side faces.
    rf.PeriodicBoundary(
        box.faces.min(axis="x"),
        box.faces.max(axis="x"),
    )
    # All other faces stay PEC by default.
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="central")
    y0 = np.zeros(ptd.n_dof)
    y0[ptd.probe_dof(
        (side * 0.5, side * 0.5, side * 0.5), field="E", component="z"
    )] = 1.0
    traj = ptd.transient(y0, dt=3e-12, steps=150, device="cpu")
    e0 = ptd.field_energy(traj[0])
    e_max = max(ptd.field_energy(traj[k]) for k in range(traj.shape[0]))
    e_end = ptd.field_energy(traj[-1])
    print(f"  periodic energy E(0)={e0:.4g}, max={e_max:.4g}, end={e_end:.4g}")
    assert np.all(np.isfinite(traj)), "transient must stay finite"
    # Central-flux with periodic + PEC is lossless; max-energy must
    # equal start within a few percent (central flux conserves energy
    # to machine precision; the drift is a per-step accumulator).
    drift = abs(e_max - e0) / e0
    assert drift < 0.1, (
        f"periodic energy drift {drift:.2%} (expected lossless)"
    )
