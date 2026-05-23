"""TD macromodel on a 50 ohm microstrip — the simplest 2-port lumped
benchmark.

Same geometry as the FD `fd_microstrip_line.py` example (straight
30 mm trace on 0.508 mm RO4003C), driven through the TD block-Krylov
macromodel pipeline and cross-validated against the FD direct sweep
at the same points.

A microstrip is a single-mode quasi-TEM transmission line, so it sits
between a propagating waveguide (no compression possible) and a
resonant inductor (rich compression): the lumped ports excite the
TEM mode, the line phase rotates, and a handful of through-trip
moments span the in-band behaviour. The minimal `r` for clean
agreement is the diagnostic we want.
"""
import math
import time
from pathlib import Path

import numpy as np

import rapidfem as rf

mm = 1e-3

# Geometry parameters - copied verbatim from fd_microstrip_line.py.
SUB_H  = 0.508 * mm
ER_SUB = 3.55
TAND   = 0.0027

LINE_W = 1.13 * mm
LINE_L = 30.0 * mm

SUB_W  = 20.0 * mm
AIR_H  = 10.0 * mm

# Sweep the same band as the FD example.
FREQS = np.linspace(2.85e9, 3.30e9, 21)
MAXH = rf.lambda_maxh(f_max=3.3e9, er_max=ER_SUB)


def build_geometry():
    """Return a meshed geometry with the microstrip + lumped ports + ABC."""
    g = rf.Geometry(maxh=MAXH)
    fr4 = rf.Dielectric(er=ER_SUB, tand=TAND, maxh=1.5 * SUB_H)
    sub = g.box(SUB_W, LINE_L, SUB_H, position=(-SUB_W / 2, 0, 0), material=fr4)
    air = g.box(SUB_W, LINE_L, AIR_H, position=(-SUB_W / 2, 0, SUB_H),
                material=rf.Air())
    trace = g.xy_plate(LINE_W, LINE_L, position=(-LINE_W / 2, 0, SUB_H))
    port_in = g.plate(
        p0=(-LINE_W / 2, 0, 0),
        width=(LINE_W, 0, 0),
        height=(0, 0, SUB_H),
    )
    port_out = g.plate(
        p0=(-LINE_W / 2, LINE_L, 0),
        width=(LINE_W, 0, 0),
        height=(0, 0, SUB_H),
    )
    g.fragment(sub, air, trace, port_in, port_out)
    rf.LumpedPort(port_in,  direction=(0, 0, 1), z0=50.0)
    rf.LumpedPort(port_out, direction=(0, 0, 1), z0=50.0)
    rf.PEC(trace, sub.faces.min(axis="z"))
    rf.ABC(*air.faces.outer, order=1)
    g.auto_refine_features(base_maxh=MAXH)
    g.mesh()
    return g


def main():
    print("=== TD macromodel: 50 ohm microstrip ===\n")

    print("[1] Building FD reference ...")
    g_fd = build_geometry()
    prob_fd = rf.Problem(g_fd)
    t0 = time.perf_counter()
    res_fd = prob_fd.sweep(FREQS)
    t_fd = time.perf_counter() - t0
    print(f"    FD sweep: {t_fd:.1f} s ({len(FREQS)} pts, {prob_fd.n_dofs} DOFs)")

    print("\n[2] Building TD operator ...")
    g_td = build_geometry()
    ptd = rf.ProblemTD(g_td, order=2, flux="upwind")
    print(f"    TD operator: {ptd.n_dof} DOFs, {ptd._op.n_ports()} ports")

    # Diagnostic at several r values, comparing plain impulse-Krylov
    # vs shift-invert (M4 WP 4.3). The microstrip is in the regime
    # where the design band sits far below the operator's spectral
    # radius - plain impulse-Krylov cannot reach the in-band physics,
    # shift-invert biases the basis around the centre frequency.
    s_fd = res_fd.sparams       # [n_freq, 2, 2]
    f_centre = float(np.mean(FREQS))

    print("\n[3] Macromodel accuracy vs method ...")
    print(
        f"\n{'method':>20} {'r':>5} {'build [s]':>10} {'sweep [ms]':>11} "
        f"{'max|S11| err':>13} {'max|S21| err':>13}"
    )

    def bench(label, r, **kwargs):
        t0 = time.perf_counter()
        mac = ptd.macromodel(r=r, **kwargs)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        s_td = mac.sweep(FREQS)
        t_sweep = time.perf_counter() - t0
        d11 = float(np.max(np.abs(np.abs(s_td[:, 0, 0]) - np.abs(s_fd[:, 0, 0]))))
        d21 = float(np.max(np.abs(np.abs(s_td[:, 1, 0]) - np.abs(s_fd[:, 1, 0]))))
        print(
            f"{label:>20} {r:>5} {t_build:>10.2f} {t_sweep*1e3:>11.1f} "
            f"{d11:>13.3f} {d21:>13.3f}"
        )
        return mac, s_td

    bench("plain Krylov", 150)
    bench("plain Krylov", 300)
    bench("single-shift", 40, shift_freq_hz=f_centre)
    # Multi-shift: 8 shift points evenly across the band, 2 GMRES
    # applications per port per shift.
    shifts = list(np.linspace(FREQS[0], FREQS[-1], 8))
    mac_ms, s_td = bench(
        "multi-shift x8", 40, shift_freqs_hz=shifts, n_shift_steps=2,
    )

    print(f"\n[4] Detailed |S| with multi-shift across {FREQS[0]/1e9:.2f}-"
          f"{FREQS[-1]/1e9:.2f} GHz ({len(shifts)} shifts) ...")
    print(f"\n{'f [GHz]':>8} {'|S11| TD':>10} {'|S11| FD':>10} "
          f"{'|S21| TD':>10} {'|S21| FD':>10}")
    for k, f in enumerate(FREQS):
        print(
            f"{f/1e9:>8.3f} {abs(s_td[k,0,0]):>10.3f} "
            f"{abs(s_fd[k,0,0]):>10.3f} {abs(s_td[k,1,0]):>10.3f} "
            f"{abs(s_fd[k,1,0]):>10.3f}"
        )


if __name__ == "__main__":
    main()
