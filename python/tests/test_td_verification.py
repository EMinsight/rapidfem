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
