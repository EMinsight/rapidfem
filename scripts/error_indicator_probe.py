"""Probe the residual error indicator on a handful of geometries.

For each test case: build, sweep at a single representative frequency,
compute the Monk error indicator, then report the top-N hot-spot tet
centroids so we can visually sanity-check whether they sit on the
expected features (sharp edges, port plane, patch corners, conductor
discontinuities).

This is WP-A of the AMR effort: indicator only, no re-mesh.

Run with:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/error_indicator_probe.py
"""
from __future__ import annotations

import sys

import numpy as np

import rapidfem as rf


# ── Geometries ──────────────────────────────────────────────────────────────

def case_coax_open():
    """Coax monopole — sharp-edged conductors, classic sliver source."""
    mm = 1e-3
    RI, RO = 1.50e-3, 3.45e-3
    LIN, LPROT = 25.0e-3, 10.0e-3
    BW, BL = 44.0e-3, 70.0e-3
    Z0 = -BL / 2
    Z1 = Z0 + LIN
    Z2 = Z1 + LPROT
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=10.0e9))
    box = g.box(BW, BW, BL, position=(-BW / 2, -BW / 2, Z0), material=rf.Air())
    coax = g.cylinder(radius=RO, height=LIN, position=(0, 0, Z0),
                      axis=(0, 0, 1), material=rf.Air(), maxh=RO / 3)
    inner = g.cylinder(radius=RI, height=LIN + LPROT, position=(0, 0, Z0),
                       axis=(0, 0, 1), material=rf.Air(), maxh=RO / 3)
    g.fragment(box, coax, inner)
    rf.CoaxPort(coax.faces.min(axis="z"), ri=RI, ro=RO, origin=(0, 0, Z0))
    rf.PEC(*coax.faces.where(lambda c, b: Z0 + 1e-4 < c[2] < Z1 - 1e-4))
    rf.PEC(*inner.faces.where(lambda c, b: Z0 + 1e-4 < c[2] < Z2 - 1e-4))
    rf.PEC(inner.faces.max(axis="z"))
    rf.ABC(*box.faces.outer, order=1)
    return g, 5.0e9, "coax_open @ 5 GHz"


def case_patch_clean():
    """Patch antenna near resonance — fields concentrate at radiating edges."""
    mm = 1e-3
    SUB_W, SUB_L, SUB_H = 60 * mm, 60 * mm, 1.6 * mm
    PATCH_W, PATCH_L = 38 * mm, 29 * mm
    PAD_XY, PAD_Z = 25 * mm, 60 * mm
    total_w = SUB_W + 2 * PAD_XY
    total_l = SUB_L + 2 * PAD_XY
    AIR_TOP = SUB_H + PAD_Z
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=2.8e9))
    fr4 = rf.Dielectric(er=4.4, maxh=1.5 * SUB_H)
    air = g.box(total_w, total_l, AIR_TOP,
                position=(-total_w / 2, -total_l / 2, 0), material=rf.Air())
    sub = g.box(SUB_W, SUB_L, SUB_H,
                position=(-SUB_W / 2, -SUB_L / 2, 0), material=fr4)
    patch = g.xy_plate(PATCH_W, PATCH_L,
                       position=(-PATCH_W / 2, -PATCH_L / 2, SUB_H))
    feed = g.plate(p0=(-0.75e-3, -PATCH_L / 2, 0),
                   width=(1.5e-3, 0, 0), height=(0, 0, SUB_H))
    g.fragment(air, sub, patch, feed)
    rf.LumpedPort(feed, direction=(0, 0, 1), z0=50.0)
    rf.PEC(patch, sub.faces.min(axis="z"))
    rf.ABC(*air.faces.outer.unassigned, order=1)
    return g, 2.4e9, "patch @ 2.4 GHz (resonance)"


# ── Probe ───────────────────────────────────────────────────────────────────

def report(name, g, freq):
    print(f"\n=== {name} ===")
    g.mesh()
    prob = rf.Problem(g)
    result = prob.sweep([freq])
    print(f"  DOFs={prob.n_dofs}, tets={prob.n_tets}")

    for theta in (0.3, 0.5):
        errs = prob.element_errors(result, freq_idx=0, theta=theta)
        print(f"  theta={theta}: {errs!r}")
        # Top-5 hot-spot tet centroids.
        order = np.argsort(errs.eta)[::-1][:5]
        for i, idx in enumerate(order):
            c = errs.tet_centroids[idx]
            eta = errs.eta[idx]
            print(f"    #{i+1}: tet {idx:>6} eta={eta:.3e} "
                  f"at ({c[0]*1e3:+.1f}, {c[1]*1e3:+.1f}, {c[2]*1e3:+.1f}) mm")


def main():
    for builder in (case_coax_open, case_patch_clean):
        g, freq, name = builder()
        report(name, g, freq)
    sys.exit(0)


if __name__ == "__main__":
    main()
