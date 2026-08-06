"""
RFIC microstrip line on the SKY130 stack, fed by a coplanar GSG probe pad.

Demonstrates the hand-coded rfic primitives end to end, no GDS involved:
  - substrate + bulk oxide from the `Stack` preset via `create_substrate`
  - 2D metal trace via `rfic.microstrip` (thin-conductor approximation)
  - coplanar GSG pad with its lumped-port plate via `rfic.gsg_port`

The trace is a 200 μm long, 5 μm wide signal line on met5 (top metal).
"""
import sys

import numpy as np

import rapidfem as rf
import rapidfem.rfic as rfic


def main() -> int:
    um = 1e-6

    stack = rfic.Stack.sky130()
    foot = (500 * um, 300 * um)

    g = rf.Geometry(maxh=40 * um, scale=1e-6)

    # ── metal: trace + GSG pad on met5 ─────────────────────────────────────
    trace = rfic.microstrip(
        g, stack, layer="met5", width=5 * um, length=200 * um,
        position=(-100 * um, -2.5 * um),
    )
    trace.maxh = 10 * um

    pads = rfic.gsg_port(
        g, stack, layer="met5",
        center=(-150 * um, 0.0),
        pad_size=40 * um, pitch=80 * um,
    )
    pads.port_plate.maxh = 8 * um

    # ── substrate + oxide + air enclosure ──────────────────────────────────
    slabs = stack.create_substrate(g, footprint=foot, center=True,
                                   fragment_existing=False)
    air = g.box(foot[0], foot[1], 100 * um,
                position=(-foot[0] / 2, -foot[1] / 2, stack.top_z),
                material=rf.Air())

    g.fragment(slabs["oxide"], slabs["substrate"], air, trace,
               pads.signal_pad, *pads.ground_pads, pads.port_plate)

    # ── physics ────────────────────────────────────────────────────────────
    rf.PEC(trace, pads.signal_pad, *pads.ground_pads)
    rf.LumpedPort(pads.port_plate, direction=(1, 0, 0), z0=50.0)
    rf.ABC(*air.faces.outer,
           slabs["oxide"].faces.min(axis="x"), slabs["oxide"].faces.max(axis="x"),
           slabs["oxide"].faces.min(axis="y"), slabs["oxide"].faces.max(axis="y"),
           slabs["substrate"].faces.min(axis="x"), slabs["substrate"].faces.max(axis="x"),
           slabs["substrate"].faces.min(axis="y"), slabs["substrate"].faces.max(axis="y"),
           slabs["substrate"].faces.min(axis="z"))

    g.mesh()
    print(g.mesh_stats)

    prob = rf.Problem(g)
    result = prob.sweep(np.array([1e9, 5e9, 10e9]))
    g.close()

    print()
    print("RFIC microstrip + GSG: |S11| sweep")
    for k, f in enumerate(result.frequencies):
        print(f"  f = {f/1e9:5.2f} GHz   |S11| = {abs(result.sparams[k, 0, 0]):.4f}")

    # Sanity bound: a small mismatched stub on lossy silicon reflects, but
    # passively (|S11| <= 1 up to numerical tolerance).
    s11 = np.abs(result.sparams[:, 0, 0])
    if not (s11 < 1.05).all():
        print(f"FAIL: |S11| > 1 ({s11.max():.4f})")
        return 1
    print()
    print("OK - RFIC stack + microstrip + GSG port pipeline working end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
