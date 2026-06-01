"""Wave-port passivity diagnostic.

Builds a perfectly-closed, lossless stripline-style line:
  - thick substrate slab + thin air slab, all 6 faces PEC except the two ports
  - centred trace as internal PEC (treats as ideal stripline)
  - no ABC, no tand, no conductor loss

Expected: |S11|^2 + |S21|^2 == 1 to machine precision (modulo
mesh-discretisation error). Anything below 0.95 points at a wave-port
normalisation / projection bug; anything above 1.0 is a clear non-passivity
violation. Run with both scalar TE and vector hybrid modes to cross-check.
"""
from __future__ import annotations
import math
import sys
import rapidfem as rf


def run(label: str, mode_kind: str, er_sub: float, f0_hz: float,
        refine: float = 1.0) -> tuple[float, float, float]:
    mm = 1e-3
    W, L = 5 * mm, 10 * mm
    H_SUB, H_AIR = 1.0 * mm, 1.0 * mm
    TRACE_W = 0.5 * mm

    g = rf.Geometry(maxh=0.4 * mm / refine)
    sub_mat = rf.Dielectric(er=er_sub, tand=0.0, maxh=0.15 * mm / refine)
    sub = g.box(W, L, H_SUB, position=(-W / 2, 0, 0), material=sub_mat)
    air = g.box(W, L, H_AIR, position=(-W / 2, 0, H_SUB), material=rf.Air())
    trace = g.xy_plate(TRACE_W, L,
                       position=(-TRACE_W / 2, 0, H_SUB),
                       maxh=0.1 * mm / refine)
    g.fragment(sub, air, trace)

    # Trace + substrate-bottom ground are the only PEC surfaces inside the
    # cross-section / line. The 4 lateral walls and the air top are closed
    # with PEC so the box is a perfect stripline (no loss path at all).
    pec_trace = rf.PEC(trace, sub.faces.min(axis="z"))
    rf.PEC(air.faces.max(axis="z"),
           sub.faces.min(axis="x"), sub.faces.max(axis="x"),
           air.faces.min(axis="x"), air.faces.max(axis="x"))

    rf.WavePort(sub.faces.min(axis="y"), air.faces.min(axis="y"),
                f0=f0_hz, mode_kind=mode_kind, pec=[pec_trace])
    rf.WavePort(sub.faces.max(axis="y"), air.faces.max(axis="y"),
                f0=f0_hz, mode_kind=mode_kind, pec=[pec_trace])

    g.mesh()
    res = rf.Problem(g).sweep([f0_hz])
    S = res.sparams[0]
    s11 = abs(S[0, 0])
    s21 = abs(S[1, 0])
    psum = s11 * s11 + s21 * s21
    print(f"  {label}: |S11|={s11:.4f}  |S21|={s21:.4f}  sum={psum:.4f}  "
          f"(deviation from passive: {abs(1.0 - psum) * 100:+.2f}%)")
    return s11, s21, psum


def main() -> int:
    print("Closed stripline-style box, perfectly lossless. Expected sum ~ 1.")
    print()
    print("--- coarse mesh (refine=1) ---")
    run("homogeneous vector",  "auto", 1.0, 10.0e9, refine=1.0)
    run("FR4 vector",          "auto", 3.55, 10.0e9, refine=1.0)
    print("--- finer mesh (refine=2) ---")
    run("homogeneous vector",  "auto", 1.0, 10.0e9, refine=2.0)
    run("FR4 vector",          "auto", 3.55, 10.0e9, refine=2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
