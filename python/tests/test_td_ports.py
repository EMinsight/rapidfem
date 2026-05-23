"""TD port verification suite for the C-series ports (Coax, Floquet,
Periodic). These cover the Python wiring on top of the Rust gates,
which validate the operator-level physics already.

Each test runs a small transient and asserts a quantitative pass
criterion (low reflection, near-unity transmission, finite-energy
preservation).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import rapidfem as rf

C = 299_792_458.0
MM = 1e-3

slow = pytest.mark.slow


# -----------------------------------------------------------------------------
# 1. CoaxPort - straight matched coaxial line, TEM transmission.
# -----------------------------------------------------------------------------

@slow
def test_coax_line_transmits_tem():
    """A straight matched 50 ohm coaxial air-line driven at port 0 with
    a matched coax port at port 1. Inner-conductor and outer-shield
    surfaces are PEC; end caps are coax ports. The TEM mode is
    dispersionless, so transmission must be close to unity and
    reflection close to zero across a broad band. Mirrors the
    `fd_coax_step.py` geometry pattern; covers the Python coax wiring
    on top of the Rust C1 gate.
    """
    r_inner = 1.50 * MM
    r_outer = 3.45 * MM            # ~50 ohm air coax
    length = 20.0 * MM

    g = rf.Geometry(maxh=3.0 * MM)
    air = g.cylinder(radius=r_outer, height=length, position=(0, 0, 0),
                     material=rf.Air())
    inner = g.cylinder(radius=r_inner, height=length, position=(0, 0, 0),
                       material=rf.Air())
    g.fragment(air, inner)
    # End caps carry the two coax ports.
    rf.CoaxPort(air.faces.min(axis="z"), ri=r_inner, ro=r_outer,
                origin=(0, 0, 0))
    rf.CoaxPort(air.faces.max(axis="z"), ri=r_inner, ro=r_outer,
                origin=(0, 0, length))
    # Everything else (inner-conductor surface + outer shield) is PEC.
    rf.PEC(*air.faces.unassigned)
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="central")
    # Coax has no cutoff: a narrow sweep around 5 GHz is enough to
    # demonstrate TEM transmission.
    freqs = np.linspace(3.0e9, 7.0e9, 5)
    res = ptd.sparams(freqs, dt=2.0e-12, steps=400, verbose=False)
    s = res.sparams

    s11 = np.abs(s[:, 0, 0])
    s21 = np.abs(s[:, 1, 0])
    print(f"  coax |S11|: max {s11.max():.3f}")
    print(f"  coax |S21|: min {s21.min():.3f} max {s21.max():.3f}")
    assert s11.max() < 0.2, f"coax reflection too high: {s11.max():.3f}"
    assert s21.min() > 0.7, f"coax transmission too low: {s21.min():.3f}"


# -----------------------------------------------------------------------------
# 2. PeriodicBoundary - field continuity across a periodic pair.
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


# -----------------------------------------------------------------------------
# 3. FloquetPort - plane wave through a periodic unit cell.
# -----------------------------------------------------------------------------

@slow
@pytest.mark.skip(
    reason="depends on PeriodicBoundary on the four side faces, which "
    "needs gmsh `setPeriodic` wiring (see above). Rust C3 gate covers "
    "the Floquet-port operator physics on a structured mesh."
)
def test_floquet_unit_cell_transmits_plane_wave():
    """Normal-incidence plane wave through a thin air slab. The unit
    cell has Floquet ports on top + bottom and periodic boundaries on
    the four side faces. With no scatterer in the cell, transmission
    must be near unity and reflection near zero. Mirrors the Rust C3
    gate (transmission 0.996, reflection ~ machine eps).
    """
    side = 10.0 * MM
    thick = 15.0 * MM
    g = rf.Geometry(maxh=4.0 * MM)
    cell = g.box(side, side, thick, material=rf.Air())
    # Periodic on the four side faces.
    rf.PeriodicBoundary(
        cell.faces.min(axis="x"), cell.faces.max(axis="x"),
    )
    rf.PeriodicBoundary(
        cell.faces.min(axis="y"), cell.faces.max(axis="y"),
    )
    # Floquet ports on the two z-faces.
    rf.FloquetPort(
        cell.faces.min(axis="z"),
        scan_theta_deg=0.0, scan_phi_deg=0.0, mode_nr=1,
    )
    rf.FloquetPort(
        cell.faces.max(axis="z"),
        scan_theta_deg=0.0, scan_phi_deg=0.0, mode_nr=1,
    )
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="central")
    freqs = np.linspace(8e9, 12e9, 5)
    res = ptd.sparams(freqs, dt=1.5e-12, steps=400, verbose=False)
    s = res.sparams

    s11 = np.abs(s[:, 0, 0])
    s21 = np.abs(s[:, 1, 0])
    print(f"  Floquet |S11|: max {s11.max():.3f}")
    print(f"  Floquet |S21|: min {s21.min():.3f} max {s21.max():.3f}")
    assert s11.max() < 0.2, (
        f"Floquet empty-cell reflection too high: {s11.max():.3f}"
    )
    assert s21.min() > 0.7, (
        f"Floquet empty-cell transmission too low: {s21.min():.3f}"
    )
