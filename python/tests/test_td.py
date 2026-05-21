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

# PML drain test tolerances. A closed PEC cavity must retain at least this
# fraction of the seeded impulse energy; the PML-terminated run must keep
# below this fraction of whatever the closed cavity keeps - a decisive
# absorb-vs-conserve contrast.
_PML_CLOSED_MIN_KEEP = 1.0
_PML_DRAIN_FRACTION = 0.5

# Field-energy diagnostic tolerance. The central flux is exactly energy-
# conserving in the continuous form; on a discrete transient the matrix-free
# field_energy must stay within this relative band over a free run. The
# upwind flux is dissipative, so its energy is non-increasing up to a tiny
# slack for floating-point round-off.
_ENERGY_CONSERVE_TOL = 0.03
_ENERGY_MONOTONE_SLACK = 1e-9


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


def test_field_energy_is_finite_and_positive(cavity, spike):
    # The matrix-free field-energy diagnostic of a seeded state is a finite,
    # strictly positive scalar -- the material-weighted EM field energy.
    e = cavity.field_energy(spike)
    assert np.isscalar(e) or np.ndim(e) == 0
    assert np.isfinite(e)
    assert e > 0.0
    # The rest state carries no energy.
    assert cavity.field_energy(np.zeros(cavity.n_dof)) == pytest.approx(0.0)


def test_field_energy_conserved_under_central_flux(spike):
    # The central flux is energy-conserving: a free transient keeps the
    # field energy roughly constant across the run.
    cav = ProblemTD.box(
        size=(1, 1, 1), cells=(2, 2, 2), order=2, flux="central", c=1.0
    )
    sp = np.zeros(cav.n_dof)
    sp[cav.probe_dof([0.5, 0.5, 0.5], field="E", component="z")] = 1.0
    traj = cav.transient(sp, dt=0.02, steps=40, verbose=False)
    energies = np.array([cav.field_energy(traj[k]) for k in range(len(traj))])
    assert np.all(np.isfinite(energies))
    e0 = energies[0]
    assert e0 > 0.0
    rel = np.abs(energies - e0) / e0
    assert rel.max() < _ENERGY_CONSERVE_TOL, (
        f"central flux drifted energy by {rel.max():.3%}"
    )


def test_field_energy_non_increasing_under_upwind(cavity, spike):
    # The upwind flux is dissipative: a free transient's field energy is
    # monotonically non-increasing (up to floating-point round-off).
    traj = cavity.transient(spike, dt=0.02, steps=40, verbose=False)
    energies = np.array(
        [cavity.field_energy(traj[k]) for k in range(len(traj))]
    )
    assert np.all(np.isfinite(energies))
    assert energies[0] > 0.0
    diffs = np.diff(energies)
    assert np.all(diffs <= _ENERGY_MONOTONE_SLACK * energies[0]), (
        f"upwind flux raised the field energy: max step {diffs.max():.3e}"
    )
    # And it actually decays over the run.
    assert energies[-1] < energies[0]


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

    from rapidfem.problem.td import _SPARAM_PASSIVITY_TOL

    ptd = rf.ProblemTD(g, order=2, flux="central")
    assert ptd._op.n_ports() == 2

    freqs = np.array([9e9, 11e9])
    f, s = ptd.sparams(freqs, dt=5e-12, steps=400, verbose=False)
    assert s.shape == (2, 2, 2)
    assert np.all(np.isfinite(s))
    # The split-window extraction must stay passive: a lossless matched
    # guide cannot have |S| above unity. Guards against the truncated-DFT
    # artefact that previously inflated |S21| past 0 dB.
    assert np.abs(s).max() <= 1.0 + _SPARAM_PASSIVITY_TOL, (
        f"non-physical |S| = {np.abs(s).max():.3f}"
    )
    for k in range(len(f)):
        # Near-unity transmission, guide roughly matched.
        assert abs(s[k, 1, 0]) > 0.9, f"transmission low at f[{k}]"
        assert abs(s[k, 0, 0]) < 0.2, f"reflection too high at f[{k}]"


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


def test_pml_collects_as_matched_absorber():
    # WP: an rf.PML region is collected for the TD backend as a graded
    # matched-absorber spec (volume_tag, axis, inner_face, thickness,
    # nu_max, is_low). The axis/is_low derive from the outward direction,
    # nu_max from the loss budget over the slab thickness.
    import rapidfem as rf
    from rapidfem.problem.td import _ABSORBER_LOSS_BUDGET, _collect_absorbers

    mm = 1e-3
    g = rf.Geometry(maxh=14 * mm)
    air = g.box(24 * mm, 12 * mm, 50 * mm, material=rf.Air())
    slab = g.box(
        24 * mm, 12 * mm, 80 * mm, position=(0, 0, 50 * mm),
        material=rf.Air(),
    )
    rf.PML(slab, direction=(0, 0, 1), inner_face=50 * mm, thickness=80 * mm)
    rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"))
    g.mesh()

    specs = _collect_absorbers(g)
    assert len(specs) == 1
    tag, axis, inner_face, thickness, nu_max, is_low = specs[0]
    assert axis == 2                       # +z direction
    assert is_low is False                 # outward toward increasing z
    assert inner_face == pytest.approx(50 * mm)
    assert thickness == pytest.approx(80 * mm)
    assert nu_max == pytest.approx(_ABSORBER_LOSS_BUDGET / (80 * mm))


def test_pml_drains_td_field_energy():
    # WP: an rf.PML slab wires through to the TD operator as a graded
    # matched absorber. An impulse in the slab region drains away under the
    # PML, but the SAME geometry with the slab left as plain air (a closed
    # PEC cavity) conserves the bulk of it. The decisive contrast confirms
    # the PML is no longer silently ignored in time domain.
    import rapidfem as rf

    mm = 1e-3
    air_l, slab_t = 50 * mm, 80 * mm
    w, h = 24 * mm, 12 * mm

    def build(with_pml):
        g = rf.Geometry(maxh=14 * mm)
        air = g.box(w, h, air_l, material=rf.Air())
        slab = g.box(
            w, h, slab_t, position=(0, 0, air_l), material=rf.Air()
        )
        if with_pml:
            rf.PML(
                slab, direction=(0, 0, 1), inner_face=air_l,
                thickness=slab_t,
            )
        rf.PEC(
            air.faces.min(axis="x"), air.faces.max(axis="x"),
            air.faces.min(axis="y"), air.faces.max(axis="y"),
            air.faces.min(axis="z"),
            slab.faces.min(axis="x"), slab.faces.max(axis="x"),
            slab.faces.min(axis="y"), slab.faces.max(axis="y"),
            slab.faces.max(axis="z"),
        )
        g.mesh()
        return rf.ProblemTD(g, order=2, flux="central")

    def retained(with_pml):
        ptd = build(with_pml)
        y = np.zeros(ptd.n_dof)
        seed_z = air_l + 0.5 * slab_t
        y[ptd.probe_dof([w / 2, h / 2, seed_z], field="E", component="y")] = 1.0
        e0 = float(np.dot(y, y))
        traj = ptd.transient(y, dt=20e-12, steps=300, verbose=False)
        return float(np.dot(traj[-1], traj[-1])) / e0

    frac_pml = retained(True)
    frac_closed = retained(False)
    # The closed PEC cavity keeps a substantial fraction; the PML drains it.
    assert frac_closed > _PML_CLOSED_MIN_KEEP, (
        f"closed cavity should retain energy - kept {frac_closed:.4f}"
    )
    assert frac_pml < _PML_DRAIN_FRACTION * frac_closed, (
        f"PML must drain decisively - kept {frac_pml:.5f} vs closed "
        f"{frac_closed:.4f}"
    )


def _debye_geometry(debye_mat, *, maxh_mm=14.0, size_mm=(24, 12, 40)):
    """A small PEC box filled with one volume material — the shared fixture
    geometry for the dispersive-material tests."""
    import rapidfem as rf

    mm = 1e-3
    w, h, lz = (s * mm for s in size_mm)
    g = rf.Geometry(maxh=maxh_mm * mm)
    air = g.box(w, h, lz, material=debye_mat)
    rf.PEC(
        air.faces.min(axis="x"), air.faces.max(axis="x"),
        air.faces.min(axis="y"), air.faces.max(axis="y"),
        air.faces.min(axis="z"), air.faces.max(axis="z"),
    )
    g.mesh()
    return g


def test_non_dispersive_problem_keeps_the_plain_dof_count():
    # Safety property: a problem with NO Debye material has the operator
    # byte-identical to before — n_dof is exactly 6*Np*n_elem and the
    # native operator reports zero dispersive regions.
    import rapidfem as rf

    g = _debye_geometry(rf.Air())
    ptd = rf.ProblemTD(g, order=2, flux="upwind")
    assert ptd._op.n_dispersive() == 0
    np_ = (2 + 1) * (2 + 2) * (2 + 3) // 6  # order-2 Np = 10
    n_elem = ptd.n_dof // (6 * np_)
    assert ptd.n_dof == 6 * np_ * n_elem


def test_debye_material_appends_a_polarisation_block():
    # A Debye-material problem carries the augmented state: n_dof exceeds
    # the plain 6*Np*n_elem by exactly 3*Np per dispersive element, and a
    # free transient stays finite and bounded under the upwind flux.
    import rapidfem as rf

    debye = rf.Material(debye=rf.Debye(
        er_inf=2.0, er_static=6.0, tau_s=20e-12,
    ))
    g = _debye_geometry(debye)
    ptd = rf.ProblemTD(g, order=2, flux="upwind")

    n_disp = ptd._op.n_dispersive()
    assert n_disp > 0, "the Debye volume produced no dispersive elements"
    np_ = (2 + 1) * (2 + 2) * (2 + 3) // 6  # order-2 Np = 10
    n_elem = (ptd.n_dof - 3 * np_ * n_disp) // (6 * np_)
    # n_dof = 6*Np*n_elem + 3*Np*n_disp_elem.
    assert ptd.n_dof == 6 * np_ * n_elem + 3 * np_ * n_disp

    # A free transient from a seeded impulse stays finite and does not grow.
    y = np.zeros(ptd.n_dof)
    y[ptd.probe_dof([12e-3, 6e-3, 20e-3], field="E", component="z")] = 1.0
    traj = ptd.transient(y, dt=5e-12, steps=40, verbose=False)
    assert traj.shape == (41, ptd.n_dof)
    assert np.all(np.isfinite(traj))
    assert np.linalg.norm(traj[-1]) <= np.linalg.norm(traj[0]) + 1e-9


def test_debye_state_space_matches_rhs():
    # The augmented sparse operator A (with the P rows/cols and the E<->P
    # coupling) reproduces the matrix-free rhs on a dispersive problem.
    import rapidfem as rf

    pytest.importorskip("scipy")
    debye = rf.Material(debye=rf.Debye(
        er_inf=2.5, er_static=7.0, tau_s=15e-12,
    ))
    g = _debye_geometry(debye)
    ptd = rf.ProblemTD(g, order=2, flux="upwind")
    a = ptd.state_space()
    assert a.shape == (ptd.n_dof, ptd.n_dof)

    rng = np.random.default_rng(0)
    v = rng.standard_normal(ptd.n_dof)
    ref = ptd.rhs(v)
    rel = np.linalg.norm(a @ v - ref) / np.linalg.norm(ref)
    assert rel < 1e-10, f"sparse vs matrix-free rel.err {rel:.2e}"


def test_debye_operator_reproduces_the_analytic_permittivity():
    # Physics gate: the assembled time-domain operator implements the Debye
    # auxiliary-differential-equation, so its polarisation block reproduces
    # the analytic complex permittivity eps(omega) = eps_inf +
    # (eps_s - eps_inf)/(1 + j*omega*tau).
    #
    # The augmented operator's polarisation rows carry the ADE Pdot = a*P +
    # g*E with a = -1/tau, g = (eps_s - eps_inf)/tau. Sinusoidal steady
    # state of that linear ODE gives the polarisation phasor
    # P = g/(j*omega - a) * E, and D = eps_inf*E + P, so the medium's
    # complex permittivity is eps(omega) = eps_inf + g/(j*omega - a).
    # Extracting (a, g) straight from the verbatim sparse state-space matrix
    # and reconstructing eps(omega) is an exact, operator-level check that
    # the TD backend is consistent with rapidfem.Debye over a full sweep.
    import rapidfem as rf
    from rapidfem.materials import Debye

    pytest.importorskip("scipy")
    er_inf, er_static, tau_s = 4.5, 80.1, 8.27e-12  # water-like Debye
    mm = 1e-3
    side = 20 * mm

    g = rf.Geometry(maxh=14 * mm)
    air = g.box(side, side, side, material=rf.Material(debye=Debye(
        er_inf=er_inf, er_static=er_static, tau_s=tau_s,
    )))
    rf.PEC(
        air.faces.min(axis="x"), air.faces.max(axis="x"),
        air.faces.min(axis="y"), air.faces.max(axis="y"),
        air.faces.min(axis="z"), air.faces.max(axis="z"),
    )
    g.mesh()
    ptd = rf.ProblemTD(g, order=2, flux="upwind")
    n_disp = ptd._op.n_dispersive()
    assert n_disp > 0, "the Debye volume produced no dispersive elements"

    # The augmented state is [E,H] (6*Np*n_elem) then the appended P block
    # (3*Np per dispersive element). Read the operator units back from the
    # mesh: c maps operator time to seconds, so a = -1/tau and g have the
    # 1/length scaling 1/c relative to the SI tau.
    a_csr = ptd.state_space()
    np_ = (ptd.order + 1) * (ptd.order + 2) * (ptd.order + 3) // 6
    n_dof = ptd.n_dof
    eh_len = n_dof - 3 * np_ * n_disp
    assert eh_len == n_dof - 3 * np_ * n_disp and eh_len > 0

    # Every polarisation row carries exactly two entries: g (the E coupling,
    # in the [E,H] block) and a (the P self-relaxation, on the diagonal).
    a_lil = a_csr.tolil()
    a_vals, g_vals = [], []
    for row in range(eh_len, n_dof):
        cols = a_lil.rows[row]
        data = a_lil.data[row]
        assert len(cols) == 2, (
            f"polarisation row {row} has {len(cols)} entries, expected 2 "
            f"(the g*E coupling and the a*P diagonal)"
        )
        # The diagonal entry is `a`; the off-diagonal (into the [E,H]
        # block) is `g`.
        for c, v in zip(cols, data):
            if c == row:
                a_vals.append(v)
            else:
                assert c < eh_len, "P row couples outside the [E,H] block"
                g_vals.append(v)
    a_op = float(np.mean(a_vals))
    g_op = float(np.mean(g_vals))
    # All dispersive elements share one material, so the coefficients are
    # uniform across the whole P block.
    assert np.allclose(a_vals, a_op, rtol=1e-9)
    assert np.allclose(g_vals, g_op, rtol=1e-9)

    # The operator runs in normalised units (c = 1, time in metres): the
    # ADE coefficients carry a 1/c factor relative to the SI tau, but their
    # ratio reconstructs eps(omega) at SI omega all the same. Convert to SI.
    c = ptd.c
    a_si = a_op * c          # a = -1/tau   (rad/s)
    g_si = g_op * c          # g = (eps_s - eps_inf)/tau   (rad/s)
    tau_rec = -1.0 / a_si
    es_rec = er_inf + g_si * tau_rec
    assert abs(tau_rec - tau_s) < 1e-3 * tau_s, (
        f"recovered tau {tau_rec:.4e} != {tau_s:.4e}"
    )
    assert abs(es_rec - er_static) < 1e-6 * er_static, (
        f"recovered er_static {es_rec:.4f} != {er_static}"
    )

    # Reconstruct eps(omega) from the operator's ADE coefficients and
    # compare to the analytic Debye permittivity across a microwave sweep.
    debye = Debye(er_inf=er_inf, er_static=er_static, tau_s=tau_s)
    for f_hz in (1e9, 5e9, 19.2e9, 60e9):  # spans omega*tau from 0.05 .. 3
        omega = 2.0 * np.pi * f_hz
        eps_op = er_inf + g_si / (1j * omega - a_si)
        # Analytic Debye permittivity (the rapidfem.Debye model).
        denom = 1.0 + 1j * omega * debye.tau_s
        eps_ana = debye.er_inf + (debye.er_static - debye.er_inf) / denom
        assert abs(eps_op - eps_ana) < 1e-9 * abs(eps_ana), (
            f"f={f_hz:.2e}: operator eps {eps_op:.6f} != analytic "
            f"{eps_ana:.6f}"
        )

    # Sanity: the static and high-frequency limits bracket the sweep.
    assert abs((er_inf + g_si / (1j * 1e14 - a_si)).real - er_inf) < 1e-3
    assert abs((er_inf + g_si / (1j * 1e6 - a_si)).real - er_static) < 1e-2


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
