"""Regression tests for the time-domain DGTD backend (``ProblemTD``).

Mirrors the Rust-side validation suite at the Python-API level: the
model-export verbs (``rhs``, ``state_space``, ``ode``, ``step``,
``stepper``, ``reduce``), the free and driven transient runs, and the VTK
export. Kept fast — a small structured-box cavity — so it runs as a CI
regression gate alongside ``cargo test``.
"""
import numpy as np
import pytest

from rapidfem import GaussianPulse, ProblemTD


@pytest.fixture(scope="module")
def cavity():
    """A small unit-cube PEC cavity in normalised units (c = 1)."""
    return ProblemTD.box(
        size=(1, 1, 1), cells=(2, 2, 2), order=2, flux="upwind", c=1.0
    )


@pytest.fixture
def spike(cavity):
    """A single-DOF initial state — an E_z impulse at the cavity centre."""
    y = np.zeros(cavity.n_dof)
    y[cavity.probe_dof([0.5, 0.5, 0.5], field="E", component="z")] = 1.0
    return y


def test_construction(cavity):
    # order 2 ⇒ Np = 10; 2³ cells × 6 tets = 48 elements; 6·Np·n_elem DOFs.
    assert cavity.n_dof == 2880
    assert cavity.order == 2


def test_rhs_is_finite(cavity, spike):
    dy = cavity.rhs(spike)
    assert dy.shape == (cavity.n_dof,)
    assert np.all(np.isfinite(dy))


def test_state_space_matches_rhs(cavity, spike):
    # The explicit sparse operator A reproduces the matrix-free rhs.
    pytest.importorskip("scipy")
    a = cavity.state_space()
    assert a.shape == (cavity.n_dof, cavity.n_dof)
    ref = cavity.rhs(spike)
    rel = np.linalg.norm(a @ spike - ref) / np.linalg.norm(ref)
    assert rel < 1e-12


def test_step_and_stepper_agree(cavity, spike):
    # The dt-bound stepper reproduces repeated `step` calls exactly.
    adv = cavity.stepper(0.02)
    y_step, y_adv = spike.copy(), spike.copy()
    for _ in range(5):
        y_step = cavity.step(y_step, 0.02)
        y_adv = adv(y_adv)
    assert np.allclose(y_step, y_adv)


def test_transient_decays_under_upwind(cavity, spike):
    # Upwind flux dissipates — the field amplitude must not grow.
    traj = cavity.transient(spike, dt=0.02, steps=20, verbose=False)
    assert traj.shape == (21, cavity.n_dof)
    assert np.all(np.isfinite(traj))
    assert np.linalg.norm(traj[-1]) < np.linalg.norm(traj[0])


def test_ode_export_integrates(cavity, spike):
    # `ode()` hands the system to scipy's solve_ivp; the integrated result
    # must match the exact exponential step.
    integrate = pytest.importorskip("scipy.integrate")
    ode = cavity.ode()
    assert ode.n_dof == cavity.n_dof
    sol = integrate.solve_ivp(
        ode.rhs, (0.0, 0.05), spike, method="RK45", rtol=1e-8, atol=1e-10
    )
    ref = cavity.step(spike, 0.05)
    rel = np.linalg.norm(sol.y[:, -1] - ref) / np.linalg.norm(ref)
    assert rel < 1e-6


def test_reduce_reproduces_full_propagation(cavity, spike):
    # A Krylov ROM built around `spike` propagates it exactly.
    rom = cavity.reduce(spike, dim=80)
    assert rom.r <= 80 and rom.n == cavity.n_dof
    for t in (0.01, 0.05, 0.1):
        y_rom = rom.propagate(spike, t)
        y_full = cavity.step(spike, t)
        rel = np.linalg.norm(y_rom - y_full) / np.linalg.norm(y_full)
        assert rel < 1e-8, f"t={t}: rel.err {rel}"


def test_reduce_rejects_zero_start(cavity):
    with pytest.raises(ValueError):
        cavity.reduce(np.zeros(cavity.n_dof))


def test_driven_transient_injects_energy(cavity):
    # A soft source driven from rest puts energy into the cavity.
    wf = GaussianPulse(t0=0.3, tau=0.08, f0=0.0)
    times, resp = cavity.driven_transient(
        source=([0.5, 0.5, 0.5], "E", "z"),
        waveform=wf,
        probes=[([0.5, 0.5, 0.5], "E", "z")],
        dt=0.01,
        steps=60,
        verbose=False,
    )
    assert times.shape == (61,)
    assert resp.shape == (1, 61)
    assert np.all(np.isfinite(resp))
    assert np.abs(resp).max() > 0.0


def test_transfer_function_peaks_at_a_resonance(cavity):
    # The RFT transfer function of the linear cavity peaks at its
    # resonances — within coarse-mesh discretisation error of the
    # analytic unit-cube set f = √(m²+n²+p²)/2.
    freqs, h = cavity.transfer_function(
        source=([0.5, 0.5, 0.5], "E", "z"),
        probe=([0.3, 0.7, 0.5], "E", "z"),
        pulse=GaussianPulse(t0=2.0, tau=0.6, f0=0.0),
        dt=0.05,
        steps=300,
        verbose=False,
    )
    assert freqs.shape == h.shape
    assert np.all(np.isfinite(h))

    mag = np.abs(h)
    band = (freqs > 0.1) & (freqs < 3.0)
    # a genuine resonance peak stands well clear of the band floor
    assert mag[band].max() > 5.0 * np.median(mag[band])
    f_peak = freqs[band][np.argmax(mag[band])]

    analytic = {
        np.sqrt(m * m + n * n + q * q) / 2
        for m in range(4)
        for n in range(4)
        for q in range(4)
        if 0 < m * m + n * n + q * q <= 9
    }
    assert min(abs(a - f_peak) for a in analytic) < 0.15


def test_export_vtk_is_well_formed(cavity, spike, tmp_path):
    import xml.etree.ElementTree as ET

    traj = cavity.transient(spike, dt=0.02, steps=4, verbose=False)
    pvd = cavity.export_vtk(traj, str(tmp_path / "cav"))

    collection = ET.parse(pvd).getroot()
    datasets = collection.findall(".//DataSet")
    assert len(datasets) == 5  # steps + 1 snapshots
    for ds in datasets:
        root = ET.parse(tmp_path / ds.get("file")).getroot()
        piece = root.find(".//Piece")
        # 48 elements ⇒ 48 discontinuous linear tets, 4 corner points each.
        assert int(piece.get("NumberOfCells")) == 48
        assert int(piece.get("NumberOfPoints")) == 192
        names = {
            da.get("Name")
            for da in root.findall(".//PointData/DataArray")
        }
        assert names == {"E", "H"}
