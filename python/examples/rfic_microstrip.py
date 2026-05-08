"""
RFIC microstrip line on a stacked silicon/oxide substrate, fed by a GSG probe.

Demonstrates the rapidfem.rfic builder helpers end to end:
  - Layered substrate via `Stack`
  - 2D metal trace via `microstrip`
  - Co-planar GSG pad with auto-generated lumped-port plate

The trace is a 200 μm long, 5 μm wide signal line on M2 (top metal).
"""
import sys

import numpy as np

import rapidfem
import rapidfem.rfic as rfic


def main() -> int:
    um = 1e-6

    # ── 1. Define the technology stack ─────────────────────────────────────
    stack = (
        rfic.Stack()
        .add_substrate("si",  thickness=200 * um, er=11.9, sigma=10.0)
        .add_oxide(    "ox1", thickness=4 * um,   er=4.2)
        .add_oxide(    "ox2", thickness=2 * um,   er=4.2)
        .add_metal(    "M1",  on_top_of="ox1")
        .add_metal(    "M2",  on_top_of="ox2")
    )

    # ── 2. Build geometry ───────────────────────────────────────────────────
    g = rapidfem.Geometry()
    foot = (300 * um, 300 * um)
    stack.create_substrate(g, footprint=foot, center=True)

    # Air box on top of the stack
    air = g.box(foot[0], foot[1], 100 * um,
                position=(-foot[0] / 2, -foot[1] / 2, stack.top_z))
    air.material = "air"

    # Microstrip signal trace on M2 — runs along x, 200 μm long, 5 μm wide
    trace = rfic.microstrip(
        g, stack, layer="M2", width=5 * um, length=200 * um,
        position=(-100 * um, -2.5 * um),
    )
    trace.name = "signal_pec"

    # GSG pad at the trace start (-x end)
    port = rfic.gsg_port(
        g, stack, layer="M2",
        center=(-130 * um, 0.0),
        pad_size=40 * um, gap=20 * um, pitch=80 * um,
    )
    port.signal_pad.name = "signal_pec"          # signal pad merges with the trace
    port.ground_pads[0].name = "ground_pec"
    port.ground_pads[1].name = "ground_pec"
    port.port_plate.name = "feed"

    # Tag ABC on the air-box outer walls (top + 4 sides)
    air.faces.where(lambda c, _: abs(c[2] - (stack.top_z + 100 * um)) < 1e-12).name = "abc"
    for sign in (-1, 1):
        air.faces.where(lambda c, _, s=sign: abs(c[0] - s * foot[0] / 2) < 1e-12).name = "abc"
        air.faces.where(lambda c, _, s=sign: abs(c[1] - s * foot[1] / 2) < 1e-12).name = "abc"

    g.fragment(air, *(stack.dielectrics[i].geo for i in range(len(stack.dielectrics))),
               trace, port.signal_pad, port.ground_pads[0], port.ground_pads[1], port.port_plate)

    # ── 3. Wire the simulation ─────────────────────────────────────────────
    builder = (
        rapidfem.SimulationBuilder()
        .from_geometry(g, maxh=15 * um)
        .frequencies([1e9, 5e9, 10e9])
        .pec("signal_pec", "ground_pec")
        .lumped_port("feed", direction=(1, 0, 0), z0=50.0)
        .abc("abc", order=1)
        .material("air", er=1.0)
    )
    for spec in stack.material_specs():
        builder = builder.material(**spec)

    sim = builder.build()
    g.close()

    print(f"Simulation: {sim.n_tets} tets, {sim.n_dofs} DOFs, {sim.n_driven_ports} ports")

    result = sim.run_sweep()
    print()
    print("RFIC microstrip + GSG: |S11| sweep")
    for k, f in enumerate(result.frequencies):
        print(f"  f = {f/1e9:5.2f} GHz   |S11| = {abs(result.sparams[k, 0, 0]):.4f}")

    # Sanity bounds (this is a small mismatched stub on lossy silicon → significant return loss)
    s11 = np.abs(result.sparams[:, 0, 0])
    if not (s11 < 1.05).all():
        print(f"FAIL: |S11| > 1 ({s11.max():.4f})")
        return 1
    print()
    print("OK — RFIC stack + microstrip + GSG port pipeline working end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
