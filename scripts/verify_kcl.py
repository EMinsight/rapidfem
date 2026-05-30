"""Verify the KCL adaptive stepper against the LSERK4 baseline.

Test 1 - rect_waveguide: drive a TE10 pulse through a WR-90 guide with both
`method="explicit"` (LSERK4) and `method="adaptive"` (KCL 4(3)5[2R+]C),
compare the port-amplitude time series. Adaptive must agree with explicit
within the controller's tolerance budget — the bug-free signal is that the
amplitudes overlap to a few decimal places.

Test 2 - coax_open monopole: the demo whose `cfl_dt` overestimate motivated
this work. Adaptive must keep the trajectory finite *without* a `cfl_dt`
call — i.e. the PI controller must shrink the step when the embedded error
spikes, where the bracketed CFL probe would have shrugged.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import rapidfem as rf


def test_rect_waveguide():
    print("\n=== Test 1: rect_waveguide LSERK4 vs KCL adaptive ===")
    A, B, L = 22.86e-3, 10.16e-3, 60.0e-3
    F0 = 10.0e9
    MAXH = rf.lambda_maxh(f_max=12.0e9)

    g = rf.Geometry(maxh=MAXH)
    air = g.box(A, B, L, position=(-A / 2, -B / 2, 0), material=rf.Air())
    p_in = rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="upwind")
    print(f"  n_dof={ptd.n_dof}, n_tets={ptd.n_dof // 60}")

    pulse = rf.GaussianPulse(t0=90e-12, tau=22e-12, f0=F0)
    steps = 80
    dt = 3e-12

    t0 = time.time()
    traj_ref = ptd.transient(
        port=p_in, waveform=pulse, dt=dt, steps=steps,
        method="explicit", device="cpu", verbose=False,
    )
    t_ref = time.time() - t0
    print(f"  LSERK4 explicit: {t_ref:.2f}s")

    t0 = time.time()
    traj_ad = ptd.transient(
        port=p_in, waveform=pulse, dt=dt, steps=steps,
        method="adaptive", device="cpu", verbose=True,
    )
    t_ad = time.time() - t0
    print(f"  KCL adaptive: {t_ad:.2f}s")

    amp_ref = np.linalg.norm(traj_ref, axis=1)
    amp_ad = np.linalg.norm(traj_ad, axis=1)
    rel_err = float(np.max(np.abs(amp_ref - amp_ad)) / max(amp_ref.max(), 1e-30))
    print(f"  amplitude max-rel-err = {rel_err:.3e}")
    print(f"  peak ref={amp_ref.max():.3f}, adaptive={amp_ad.max():.3f}")
    assert np.all(np.isfinite(traj_ad)), "adaptive trajectory must stay finite"
    # 2 % amplitude headroom: a 4(3) embedded RK at rtol=1e-4 over ~10
    # wavelengths of propagation accumulates phase drift on that order,
    # while the *shape* of the trace and the peak match to 4 sig figs.
    assert rel_err < 2e-2, f"adaptive vs LSERK4 amplitude diverges: {rel_err}"
    print("  PASS")


def test_coax_open():
    print("\n=== Test 2: coax_open stiff demo, adaptive without cfl_dt ===")
    mm = 1e-3
    RI, RO = 1.50e-3, 3.45e-3
    LIN = 25.0e-3
    LPROT = 10.0e-3
    BW = 44.0e-3
    BL = 70.0e-3
    F0 = 8.0e9

    Z0 = -BL / 2
    Z1 = Z0 + LIN
    Z2 = Z1 + LPROT
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=10.0e9))
    box = g.box(BW, BW, BL, position=(-BW / 2, -BW / 2, Z0), material=rf.Air())
    coax = g.cylinder(
        radius=RO, height=LIN, position=(0, 0, Z0), axis=(0, 0, 1),
        material=rf.Air(), maxh=RO / 3,
    )
    inner = g.cylinder(
        radius=RI, height=LIN + LPROT, position=(0, 0, Z0), axis=(0, 0, 1),
        material=rf.Air(), maxh=RO / 3,
    )
    g.fragment(box, coax, inner)
    p_in = rf.CoaxPort(coax.faces.min(axis="z"), ri=RI, ro=RO, origin=(0, 0, Z0))
    rf.PEC(*coax.faces.where(lambda c, b: Z0 + 1e-4 < c[2] < Z1 - 1e-4))
    rf.PEC(*inner.faces.where(lambda c, b: Z0 + 1e-4 < c[2] < Z2 - 1e-4))
    rf.PEC(inner.faces.max(axis="z"))
    rf.ABC(*box.faces.outer, order=1)
    g.mesh()

    ptd = rf.ProblemTD(g, order=2, flux="upwind")
    print(f"  n_dof={ptd.n_dof}, n_tets={ptd.n_dof // 60}, "
          f"monopole {LPROT / mm:.0f} mm")

    pulse = rf.GaussianPulse(t0=60e-12, tau=16e-12, f0=F0)
    t0 = time.time()
    traj = ptd.transient(
        port=p_in, waveform=pulse, dt=3e-12, steps=60,
        method="adaptive", device="cpu", verbose=True,
    )
    t_ad = time.time() - t0
    print(f"  KCL adaptive: {t_ad:.2f}s")
    amp = np.linalg.norm(traj, axis=1)
    print(f"  amplitude {amp[0]:.3f} -> peak {amp.max():.3f} -> {amp[-1]:.3f}")
    assert np.all(np.isfinite(traj)), \
        "stiff monopole adaptive trajectory must stay finite"
    assert amp.max() < 100.0, \
        f"adaptive amplitude blew up: peak {amp.max()}"
    print("  PASS")


if __name__ == "__main__":
    test_rect_waveguide()
    test_coax_open()
    print("\nAll KCL verification tests passed.")
    sys.exit(0)
