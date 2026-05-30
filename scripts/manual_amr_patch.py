"""User-driven AMR loop on a patch antenna — manual iteration, no wrapper.

Shows the WP-A + WP-B pieces working together:

    1. Build the patch geometry + initial mesh.
    2. Sweep, find the resonance, compute element_errors at it.
    3. Mark high-η tets, refine_near_points at half the local h.
    4. Re-mesh, build a fresh Problem, sweep again.
    5. Report n_tets, |S11|_min, resonance frequency per iteration.

If the indicator is doing its job, |S11|_min and the resonance
frequency should stabilise across iterations even though n_tets grows.

Run with:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/manual_amr_patch.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

import rapidfem as rf


def build_patch():
    """Patch + ground + lumped feed in a coarse ABC-terminated air box."""
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
    return g


FREQUENCIES = np.linspace(2.0e9, 2.8e9, 21)
N_ITERATIONS = 3
THETA = 0.3            # Dörfler fraction
REFINE_RATIO = 0.5     # new tet size = current h_k / REFINE_RATIO^{-1}


def main():
    g = build_patch()
    print(f"\nManual AMR loop on patch antenna, {N_ITERATIONS} iterations")
    print(f"  Doerfler theta = {THETA}, refine_ratio = {REFINE_RATIO}")
    print(f"  sweep: {len(FREQUENCIES)} pts {FREQUENCIES[0]/1e9:.1f}..."
          f"{FREQUENCIES[-1]/1e9:.1f} GHz")

    prev_s11min = None
    prev_f_res = None

    for it in range(N_ITERATIONS):
        t0 = time.time()
        g.mesh()
        prob = rf.Problem(g)
        result = prob.sweep(FREQUENCIES)
        sweep_t = time.time() - t0

        mags = np.array([abs(result.sparams[i, 0, 0])
                         for i in range(len(FREQUENCIES))])
        i_res = int(mags.argmin())
        f_res = FREQUENCIES[i_res]
        s11_min = mags[i_res]

        # Element-error indicator at the resonance frequency.
        errs = prob.element_errors(result, freq_idx=i_res, theta=THETA)
        marked = errs.marked
        n_marked = len(marked)

        # Delta vs previous iteration.
        if prev_s11min is not None:
            ds = s11_min - prev_s11min
            df = (f_res - prev_f_res) / 1e6
            delta_str = f"d|S11|={ds:+.4f}  df_res={df:+.1f} MHz"
        else:
            delta_str = "(initial)"

        print(f"\niter {it}: tets={prob.n_tets:>6}  DOFs={prob.n_dofs:>6}  "
              f"|S11|_min={s11_min:.4f} @ {f_res/1e9:.3f} GHz  "
              f"marked={n_marked} ({100*n_marked/prob.n_tets:.1f}%)  "
              f"[{sweep_t:.1f}s]  {delta_str}")

        if it == N_ITERATIONS - 1:
            break  # last iteration just reports, no further refinement

        # Refinement step: shrink h at marked centroids by REFINE_RATIO.
        hot_centroids = errs.tet_centroids[marked]
        h_marked = errs.h_k[marked]
        target_h = float(h_marked.mean() * REFINE_RATIO)
        print(f"  refining {n_marked} tets, target h = "
              f"{target_h*1e3:.3f} mm "
              f"(current marked h: mean {h_marked.mean()*1e3:.3f}, "
              f"min {h_marked.min()*1e3:.3f} mm)")
        g.refine_near_points(hot_centroids, h=target_h,
                             distance=5.0 * target_h)

        prev_s11min = s11_min
        prev_f_res = f_res

    sys.exit(0)


if __name__ == "__main__":
    main()
