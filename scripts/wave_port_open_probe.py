"""Wave-port passivity on an OPEN microstrip through-line.

The closed-box probe (wave_port_passivity_probe.py) is passive to <0.2%. This
one isolates the next variable: an open structure (ABC on the air top + side
walls instead of PEC) — the setup the coupled-line filter uses, where the
wave port reads |S| > 0 dB. A short, straight, NON-resonant line removes the
high-Q variable, so any passivity violation here is the wave-port-vs-ABC
interaction, not resonant amplification.

Expect |S11|^2 + |S21|^2 ~ 1 (a touch of radiation loss is physical and makes
the sum slightly < 1; a sum > 1 is the bug).
"""
from __future__ import annotations
import sys
import numpy as np
import rapidfem as rf

mm = 1e-3


def run(label: str, length: float, walls: str, freqs, maxh=2.0 * mm,
        trace_maxh=0.3 * mm, er=3.55, box_w=6 * mm) -> None:
    W, H_SUB, H_AIR = box_w, 0.5 * mm, 3 * mm
    TRACE_W = 1.1 * mm   # ~50 ohm on 0.5 mm FR4

    g = rf.Geometry(maxh=maxh)
    fr4 = rf.Dielectric(er=er, tand=0.0, maxh=min(H_SUB / 3, maxh / 3))
    sub = g.box(W, length, H_SUB, position=(-W / 2, 0, 0), material=fr4)
    air = g.box(W, length, H_AIR, position=(-W / 2, 0, H_SUB), material=rf.Air())
    trace = g.xy_plate(TRACE_W, length, position=(-TRACE_W / 2, 0, H_SUB),
                       maxh=trace_maxh)
    g.fragment(sub, air, trace)

    pec = rf.PEC(trace, sub.faces.min(axis="z"))   # trace + ground
    f0 = float(freqs[len(freqs) // 2])
    rf.WavePort(sub.faces.min(axis="y"), air.faces.min(axis="y"),
                f0=f0, mode_kind="auto", pec=[pec])
    rf.WavePort(sub.faces.max(axis="y"), air.faces.max(axis="y"),
                f0=f0, mode_kind="auto", pec=[pec])

    # Side/top closure: PEC (closed) or ABC (open).
    side_top = (air.faces.max(axis="z"),
                sub.faces.min(axis="x"), sub.faces.max(axis="x"),
                air.faces.min(axis="x"), air.faces.max(axis="x"))
    if walls == "pec":
        rf.PEC(*side_top)
    else:
        rf.ABC(*side_top, order=2)

    g.mesh()
    res = rf.Problem(g).sweep(freqs)
    print(f"\n{label} ({walls} side/top walls, L={length/mm:.0f} mm):")
    for i, f in enumerate(freqs):
        S = res.sparams[i]
        s11, s21 = abs(S[0, 0]), abs(S[1, 0])
        psum = s11 * s11 + s21 * s21
        flag = "  <-- |S|^2 > 1 !!" if psum > 1.02 else ""
        print(f"  f={f/1e9:4.1f} GHz  |S11|={s11:.3f}  |S21|={s21:.3f}  "
              f"sum={psum:.3f}{flag}")


def main() -> int:
    f1 = np.array([6.0e9])
    # (A) Single-quasi-TEM-mode passivity vs inhomogeneity (the power-weighted
    #     extraction fix targets this; deficit ~1% on a 6x-air stress geometry).
    print("=== closed lossless line: inhomogeneity sweep (narrow box, 1 mode) ===")
    for er in [1.0, 3.55]:
        run(f"er={er}", 12 * mm, "pec", f1, er=er, box_w=6 * mm)
    # (B) Box-width sweep: a wide PEC box supports a 2nd transverse mode in-band
    #     (cutoff f_c = c/(2·W·√ε_eff)). A single-mode wave port can't represent
    #     it, so |S11|²+|S21|² should run OVER unity once W is wide enough — the
    #     suspected mechanism behind the coupled-line filter's |S| > 0 dB.
    print("=== box-width sweep (er=3.55, watch for sum > 1 as W grows) ===")
    for w in [6 * mm, 12 * mm, 20 * mm, 30 * mm]:
        run(f"W={w/mm:.0f}mm", 12 * mm, "pec", f1, er=3.55, box_w=w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
