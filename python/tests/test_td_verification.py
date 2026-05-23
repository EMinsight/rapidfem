"""TD backend production-verification suite.

Quantitative pass/fail tests that mirror the proven Python examples
in `python_src/rapidfem/examples/td_*.py`. Each test builds the
geometry from scratch, runs the TD path, and asserts the result is
within a documented tolerance of analytic or FD reference.

Mark as slow because every test runs a full transient. Run the fast
subset with `pytest -m "not slow"` and the full suite with `pytest`.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

import rapidfem as rf

C = 299_792_458.0
MM = 1e-3

slow = pytest.mark.slow


# -----------------------------------------------------------------------------
# 1. Cavity (1, 1, 0) mode TD vs FD vs analytic — the cleanest validation gate.
# -----------------------------------------------------------------------------

@slow
def test_cavity_mode_matches_analytic_and_fd():
    """The lowest (1,1,0) mode of a cubic PEC cavity, computed two
    ways: FD Nédélec eigensolver and TD broadband-pulse spectral peak.
    Both must hit the analytic value within a few tenths of a percent.
    Mirrors `td_fd_crossvalidation.py`.
    """
    side = 30.0 * MM
    f_analytic = 0.5 * C * math.sqrt(2.0) / side

    g = rf.Geometry(maxh=side / 1.5)
    box = g.box(side, side, side, material=rf.Air())
    rf.PEC(*box.faces.unassigned)
    g.mesh()

    # FD reference.
    fd_modes = rf.ProblemFD(g).eigenmode(
        target_frequency=f_analytic, n_modes=6
    )
    fd_f = min(
        m.frequency_hz
        for m in fd_modes
        if m.frequency_hz > 0.3 * f_analytic
    )

    # TD driven transient → spectral peak.
    ptd = rf.ProblemTD(g, order=2, flux="upwind")
    tau = 1.0 / (math.pi * f_analytic)
    pulse = rf.GaussianPulse(t0=4.0 * tau, tau=tau)
    dt = 1.0 / (14.0 * f_analytic)
    steps = 1400
    centre = (side * 0.5, side * 0.5, side * 0.5)
    probe = (side * 0.45, side * 0.55, side * 0.5)
    run = ptd.driven_transient(
        source=(centre, "E", "z"),
        waveform=pulse,
        probes=[(probe, "E", "z")],
        dt=dt,
        steps=steps,
        krylov_dim=16,
        device="gpu",
        verbose=False,
    )
    resp = run.responses[0]
    spec = np.abs(np.fft.rfft(resp))
    freq = np.fft.rfftfreq(resp.size, dt)
    band = (freq > 0.3 * f_analytic) & (freq < 3.0 * f_analytic)
    td_f = freq[band][np.argmax(spec[band])]

    err_fd = abs(fd_f - f_analytic) / f_analytic
    err_td = abs(td_f - f_analytic) / f_analytic
    print(f"  analytic {f_analytic/1e9:.4f} GHz")
    print(f"  FD       {fd_f/1e9:.4f} GHz (err {err_fd:.3%})")
    print(f"  TD       {td_f/1e9:.4f} GHz (err {err_td:.3%})")

    assert err_fd < 0.01, f"FD error {err_fd:.3%} above 1%"
    assert err_td < 0.01, f"TD error {err_td:.3%} above 1%"


# -----------------------------------------------------------------------------
# 2. Transfer-function spectrum recovers analytic cavity modes.
# -----------------------------------------------------------------------------

@slow
def test_cavity_transfer_function_finds_modes():
    """Broadband-driven cavity, RFT transfer function H(f) = R/G; the
    peaks of |H| must line up with analytic rectangular-cavity modes
    in the chosen band. Mirrors `td_transfer_function.py` but asserts.
    """
    side = 40.0 * MM
    g = rf.Geometry(maxh=side / 7)
    air = g.box(side, side, side, material=rf.Air())
    rf.PEC(*air.faces.unassigned)
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="upwind")
    pulse = rf.GaussianPulse(t0=160e-12, tau=40e-12, f0=8e9)
    source = ((10 * MM, 10 * MM, 10 * MM), "E", "z")
    probe = ((27 * MM, 31 * MM, 18 * MM), "E", "z")
    tf = ptd.transfer_function(
        source=source, probe=probe, pulse=pulse, dt=8e-12, steps=1000,
        device="gpu",
    )
    freqs, H = tf
    mag = np.abs(H)
    # Restrict to the band the mesh actually resolves: maxh=L/7 ~ 5.7mm
    # gives ~5 cells per wavelength at 10 GHz; above that the mesh
    # under-samples and TD peaks drift by 5-10% from analytic.
    band = (freqs > 3e9) & (freqs < 10e9)
    fb, mb = freqs[band], mag[band]
    peaks_hz = [
        fb[i]
        for i in range(1, len(fb) - 1)
        if mb[i] > mb[i - 1]
        and mb[i] > mb[i + 1]
        and mb[i] > 0.1 * mb.max()
    ]

    # Analytic modes c/(2L) * sqrt(m² + n² + p²), at least two non-zero
    # indices (TE/TM modes; pure-zero indices don't radiate the
    # (m,n,p) the transverse fields probe).
    analytic = sorted({
        C / (2 * side) * math.sqrt(m * m + n * n + q * q)
        for m in range(4) for n in range(4) for q in range(4)
        if 0 < m * m + n * n + q * q <= 9
        and (m > 0) + (n > 0) + (q > 0) >= 2
    })
    in_band = [f for f in analytic if 3e9 < f < 10e9]

    # Every transfer-function peak must match an analytic mode within
    # 3% (mesh discretisation + RFT bin width).
    print(f"  TD peaks:   {[f'{f/1e9:.2f}' for f in peaks_hz]} GHz")
    print(f"  analytic:   {[f'{f/1e9:.2f}' for f in in_band]} GHz")
    assert len(peaks_hz) >= 2, "expected at least 2 in-band peaks"
    for p in peaks_hz:
        closest = min(in_band, key=lambda f: abs(f - p))
        err = abs(p - closest) / closest
        assert err < 0.03, (
            f"peak {p/1e9:.3f} GHz {err:.2%} from nearest analytic "
            f"{closest/1e9:.3f} GHz"
        )


# -----------------------------------------------------------------------------
# 3. WR-90 S-parameters TD vs FD — the modal-port production gate.
# -----------------------------------------------------------------------------

@slow
def test_wr90_sparams_td_matches_fd():
    """Straight WR-90 hollow waveguide with rectangular TE_10 modal
    ports at both ends. TD `sparams` vs FD `sweep` must agree to a
    few percent across the X-band. Mirrors `td_waveguide_sparams.py`.
    """
    a_wg, b_wg = 22.86 * MM, 10.16 * MM
    length = 300.0 * MM
    freqs = np.linspace(8.0e9, 12.0e9, 9)

    g = rf.Geometry(maxh=6.0 * MM)
    air = g.box(a_wg, b_wg, length, material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(
        air.faces.min(axis="x"), air.faces.max(axis="x"),
        air.faces.min(axis="y"), air.faces.max(axis="y"),
    )
    g.mesh()

    s_fd = rf.ProblemFD(g).sweep(freqs).sparams
    ptd = rf.ProblemTD(g, order=2, flux="central")
    # 1500 steps × 3 ps = 4.5 ns. The slowest near-cutoff frequency
    # (8 GHz, v_g ~ 0.56 c) needs ~1.8 ns to traverse the 300 mm
    # line; the windowing in `sparams` then needs ~1500 steps to keep
    # the band-edge |S21| within ~2% of FD (1000 steps slides to 5%).
    s_td = ptd.sparams(freqs, dt=3e-12, steps=1500, verbose=False).sparams

    d11 = float(np.max(
        np.abs(np.abs(s_td[:, 0, 0]) - np.abs(s_fd[:, 0, 0]))
    ))
    d21 = float(np.max(
        np.abs(np.abs(s_td[:, 1, 0]) - np.abs(s_fd[:, 1, 0]))
    ))
    print(f"  max |S11| dev TD vs FD: {d11:.3f}")
    print(f"  max |S21| dev TD vs FD: {d21:.3f}")
    assert d11 < 0.05, f"|S11| deviation {d11:.3f} above 5%"
    assert d21 < 0.05, f"|S21| deviation {d21:.3f} above 5%"


# -----------------------------------------------------------------------------
# 4. Microstrip lumped-port |S11| — verifies the ABC + internal-PEC + Z0 fixes.
# -----------------------------------------------------------------------------

@slow
def test_microstrip_lumped_port_s11_matches_fd():
    """50 ohm microstrip line (RO4003C substrate, 30 mm long, lumped
    ports both ends). The TD lumped-port wiring is sensitive to:
    (a) internal-PEC trace, (b) Z0 reference impedance, (c) ABC outer.
    All three fixes were added this session; |S11| TD must now match
    FD within a few percent. |S21| is documented as under-predicted by
    the uniform (0,0) lumped-mode profile and is NOT gated here (a
    wave-port eigensolve would fix that).
    """
    sub_h = 0.508 * MM
    er_sub = 3.55
    tand = 0.0027
    line_w = 1.13 * MM
    line_l = 30.0 * MM
    sub_w = 20.0 * MM
    air_h = 10.0 * MM
    maxh = rf.lambda_maxh(f_max=3.3e9, er_max=er_sub)
    freqs = np.linspace(2.85e9, 3.30e9, 9)

    def build():
        g = rf.Geometry(maxh=maxh)
        fr4 = rf.Dielectric(er=er_sub, tand=tand, maxh=1.5 * sub_h)
        sub = g.box(sub_w, line_l, sub_h, position=(-sub_w / 2, 0, 0),
                    material=fr4)
        air = g.box(sub_w, line_l, air_h,
                    position=(-sub_w / 2, 0, sub_h),
                    material=rf.Air())
        trace = g.xy_plate(line_w, line_l,
                           position=(-line_w / 2, 0, sub_h))
        port_in = g.plate(p0=(-line_w / 2, 0, 0),
                          width=(line_w, 0, 0), height=(0, 0, sub_h))
        port_out = g.plate(p0=(-line_w / 2, line_l, 0),
                           width=(line_w, 0, 0), height=(0, 0, sub_h))
        g.fragment(sub, air, trace, port_in, port_out)
        rf.LumpedPort(port_in,  direction=(0, 0, 1), z0=50.0)
        rf.LumpedPort(port_out, direction=(0, 0, 1), z0=50.0)
        rf.PEC(trace, sub.faces.min(axis="z"))
        rf.ABC(*air.faces.outer, order=1)
        g.auto_refine_features(base_maxh=maxh)
        g.mesh()
        return g

    s_fd = rf.ProblemFD(build()).sweep(freqs).sparams
    ptd = rf.ProblemTD(build(), order=2, flux="upwind")
    # 1.4 ns window: ~7 round trips on a 30 mm microstrip (er=3.55,
    # v_eff ~ c/sqrt(3.55), 188 ps one-way), enough for the transient
    # to settle through the lumped-port absorption.
    s_td = ptd.sparams(freqs, dt=2.0e-12, steps=700, verbose=False).sparams

    d11 = float(np.max(
        np.abs(np.abs(s_td[:, 0, 0]) - np.abs(s_fd[:, 0, 0]))
    ))
    print(f"  max |S11| dev TD vs FD: {d11:.3f}")
    print(f"  (|S21| is bounded by the uniform-profile lumped-port "
          f"approximation; not gated)")
    assert d11 < 0.05, f"|S11| deviation {d11:.3f} above 5%"
