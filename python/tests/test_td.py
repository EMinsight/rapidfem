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


def test_sparams_verb_runs():
    # WP4.2: ProblemTD.sparams runs end-to-end on a geometry with waveguide
    # ports and returns a plausible 2-port S-matrix. The tight accuracy
    # gate is the Rust matched-guide S-parameter test.
    import rapidfem as rf

    mm = 1e-3
    g = rf.Geometry(maxh=12 * mm)
    air = g.box(22.86 * mm, 10.16 * mm, 80 * mm, material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(
        air.faces.min(axis="x"), air.faces.max(axis="x"),
        air.faces.min(axis="y"), air.faces.max(axis="y"),
    )
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="central")
    assert ptd._op.n_ports() == 2

    freqs = np.array([9e9, 11e9])
    f, s = ptd.sparams(freqs, dt=5e-12, steps=200, verbose=False)
    assert s.shape == (2, 2, 2)
    assert np.all(np.isfinite(s))
    for k in range(len(f)):
        # Transmission present, guide roughly matched.
        assert abs(s[k, 1, 0]) > 0.5, f"no transmission at f[{k}]"
        assert abs(s[k, 0, 0]) < 0.5, f"reflection too high at f[{k}]"


def test_lumped_port_wires_through():
    # WP-B: a LumpedPort is collected as a TD port and maps to the (0,0)
    # sentinel mode — the operator's uniform-profile / TEM port. The native
    # operator must see it as a driven port with zero cutoff.
    import rapidfem as rf

    mm = 1e-3
    g = rf.Geometry(maxh=12 * mm)
    air = g.box(20 * mm, 10 * mm, 60 * mm, material=rf.Air())
    rf.LumpedPort(air.faces.min(axis="z"), direction=(0, 1, 0))
    rf.LumpedPort(air.faces.max(axis="z"), direction=(0, 1, 0))
    rf.PEC(
        air.faces.min(axis="x"), air.faces.max(axis="x"),
        air.faces.min(axis="y"), air.faces.max(axis="y"),
    )
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="central")
    assert ptd._op.n_ports() == 2
    # The (0,0) lumped/TEM port is non-dispersive — zero cutoff frequency.
    assert abs(ptd._op.port_cutoff(0)) < 1e-9
    assert abs(ptd._op.port_cutoff(1)) < 1e-9


def test_mixed_ports_keep_declaration_order():
    # WP-B: RectWaveguidePort and LumpedPort coexist; the TD port index
    # follows geometry declaration order, so sparams rows/cols stay aligned.
    import rapidfem as rf
    from rapidfem.problem.td import _collect_ports

    mm = 1e-3
    g = rf.Geometry(maxh=12 * mm)
    air = g.box(22.86 * mm, 10.16 * mm, 60 * mm, material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))   # port 0 — TE10
    rf.LumpedPort(air.faces.max(axis="z"), direction=(0, 1, 0))  # port 1
    rf.PEC(
        air.faces.min(axis="x"), air.faces.max(axis="x"),
        air.faces.min(axis="y"), air.faces.max(axis="y"),
    )
    g.mesh()

    ports = _collect_ports(g)
    assert [(m, n) for _, m, n, _ in ports] == [(1, 0), (0, 0)]


def test_lumped_port_direction_sets_the_field_axis():
    # WP-B.2: a LumpedPort's `direction` becomes the (0,0) port's transverse
    # field axis, overriding the geometric auto-fit. On a 20x8 mm port face
    # the auto-fit would pick the narrow (y) axis; an explicit x direction
    # must override it. A uniform E_x probe state confirms it: the port
    # projects the field onto its axis, so an x-axis port registers it in
    # full and an orthogonal y-axis port does not.
    import rapidfem as rf

    mm = 1e-3
    g = rf.Geometry(maxh=10 * mm)
    air = g.box(20 * mm, 8 * mm, 60 * mm, material=rf.Air())
    rf.LumpedPort(air.faces.min(axis="z"), direction=(1, 0, 0))  # axis x
    rf.LumpedPort(air.faces.max(axis="z"), direction=(0, 1, 0))  # axis y
    rf.PEC(
        air.faces.min(axis="x"), air.faces.max(axis="x"),
        air.faces.min(axis="y"), air.faces.max(axis="y"),
    )
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="central")
    y = np.zeros(ptd.n_dof)
    y[0::6] = 1.0  # uniform E_x at every node
    pe_x, _ = ptd._op.port_projections(y, 0)  # field axis x — sees E_x
    pe_y, _ = ptd._op.port_projections(y, 1)  # field axis y — orthogonal
    assert pe_x == pytest.approx(1.0, abs=1e-9)
    assert abs(pe_y) < 1e-9


def test_lumped_port_rejects_out_of_plane_direction():
    # WP-B.2: a direction parallel to the port face normal has no in-plane
    # part to use as the field axis — it is rejected at construction.
    import rapidfem as rf

    mm = 1e-3
    g = rf.Geometry(maxh=12 * mm)
    air = g.box(20 * mm, 10 * mm, 60 * mm, material=rf.Air())
    rf.LumpedPort(air.faces.min(axis="z"), direction=(0, 0, 1))  # ∥ normal
    rf.PEC(
        air.faces.min(axis="x"), air.faces.max(axis="x"),
        air.faces.min(axis="y"), air.faces.max(axis="y"),
    )
    g.mesh()
    with pytest.raises(RuntimeError):
        rf.ProblemTD(g, order=2, flux="central")


def test_sparams_runs_on_lumped_ports():
    # WP-C: the sparams verb runs end-to-end with LumpedPort excitation —
    # the (0,0) lumped port flows through drive, modal extraction and
    # S-matrix assembly. The clean TEM-physics gate (a dispersionless
    # c-velocity mode) is the Rust test
    # `lumped_port_carries_a_dispersionless_tem_wave`.
    import rapidfem as rf

    mm = 1e-3
    g = rf.Geometry(maxh=12 * mm)
    air = g.box(22.86 * mm, 10.16 * mm, 80 * mm, material=rf.Air())
    rf.LumpedPort(air.faces.min(axis="z"), direction=(0, 1, 0))
    rf.LumpedPort(air.faces.max(axis="z"), direction=(0, 1, 0))
    rf.PEC(
        air.faces.min(axis="x"), air.faces.max(axis="x"),
        air.faces.min(axis="y"), air.faces.max(axis="y"),
    )
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="central")
    assert ptd._op.n_ports() == 2
    # The (0,0) lumped port is non-dispersive — flat Z, zero cutoff.
    assert ptd._port_impedance(0, 9e9) == pytest.approx(1.0)

    freqs = np.array([9e9, 11e9])
    f, s = ptd.sparams(freqs, dt=5e-12, steps=200, verbose=False)
    assert s.shape == (2, 2, 2)
    assert np.all(np.isfinite(s))


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
