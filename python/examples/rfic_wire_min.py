"""
Minimal RFIC L/C-extraction smoke test: a PEC wire above a PEC ground plane in air.

Strips out PDK stack + GDS to isolate the lumped-port topology. Once this works
the multi-layer Sky130 case can build on top.

Geometry: 200um x 5um wire at z=h above a ground plane at z=0.
Two lumped ports at the wire ends, each a vertical plate down to the ground.

Reference: microstrip-line theory in air. With w=h=5um, Hammerstad gives
Z0 ~ 126 Ohm and v=c, so L_pm = Z0/c, C_pm = 1/(Z0*c). Rosa's partial-L
formula is the WRONG reference here — it's for an isolated wire and ignores
image-current cancellation from the ground plane right below.
"""
import math
import sys

import numpy as np

import rapidfem

MU0 = 4 * math.pi * 1e-7
C0 = 2.99792458e8


def microstrip_z0_air(w, h):
    """Hammerstad-Jensen Z0 for a thin microstrip in air (er=1).

    Valid for w/h up to a few. For w/h <= 1 use the narrow-line form."""
    u = w / h
    if u <= 1:
        return 60.0 * math.log(8.0 / u + u / 4.0)
    return 120.0 * math.pi / (u + 1.393 + 0.667 * math.log(u + 1.444))


def microstrip_LC(length, w, h):
    z0 = microstrip_z0_air(w, h)
    L = (z0 / C0) * length
    C = (1.0 / (z0 * C0)) * length
    return z0, L, C


def main() -> int:
    um = 1e-6
    L_wire = 200 * um
    w = 5 * um
    t = 1.26 * um
    h = 5 * um   # wire height above ground

    # ── Geometry ───────────────────────────────────────────────────────────
    g = rapidfem.Geometry()

    # Air box (large enough to contain everything)
    box_xy = 400 * um
    box_z = 50 * um
    air = g.box(box_xy, box_xy, box_z, position=(-box_xy / 2, -box_xy / 2, 0))
    air.material = "air"

    # Wire as a 2D PEC plate at z=h (matches the patch-antenna idiom that works)
    wire = g.xy_plate(L_wire, w, position=(-L_wire / 2, -w / 2, h))
    wire.name = "wire_pec"

    # Two port plates near the wire ends. Place them 1μm INSIDE the wire's
    # endpoints so the top edge lies WITHIN the wire's bottom face (not on
    # its boundary edge), forcing a clean fragment-induced sub-face.
    inset = 1 * um
    p1 = g.plate(p0=(-L_wire / 2 + inset, -w / 2, 0),
                 width=(0, w, 0),
                 height=(0, 0, h))
    p1.name = "p1"
    p2 = g.plate(p0=(+L_wire / 2 - inset, -w / 2, 0),
                 width=(0, w, 0),
                 height=(0, 0, h))
    p2.name = "p2"

    # Fragment everything BEFORE running face selectors — that way selectors
    # find ALL sub-pieces of split faces (e.g., the air box's z=0 face gets
    # carved by the port-plate bottom edges; we want every piece to be ground_pec).
    g.fragment(air, wire, p1, p2)

    # Now name face groups by selector — runs on the post-fragment topology.
    air.faces.where(lambda c, _: abs(c[2] - box_z) < 1e-12).name = "abc"
    for s in (-1, 1):
        air.faces.where(lambda c, _, s=s: abs(c[0] - s * box_xy / 2) < 1e-12).name = "abc"
        air.faces.where(lambda c, _, s=s: abs(c[1] - s * box_xy / 2) < 1e-12).name = "abc"
    air.faces.where(lambda c, _: abs(c[2]) < 1e-12).name = "ground_pec"

    # ── Build sim ──────────────────────────────────────────────────────────
    sim = (
        rapidfem.SimulationBuilder()
        .from_geometry(g, maxh=15 * um)
        .frequencies([1e9])
        .pec("ground_pec", "wire_pec")
        .lumped_port("p1", direction=(0, 0, 1), z0=50.0)
        .lumped_port("p2", direction=(0, 0, 1), z0=50.0)
        .abc("abc", order=1)
        .material("air", er=1.0)
        .build()
    )
    g.close()

    print(f"Simulation: {sim.n_tets} tets, {sim.n_dofs} DOFs, {sim.n_driven_ports} driven")

    z0_ms, L_ana, C_ana = microstrip_LC(L_wire, w, h)
    print(f"Microstrip (Hammerstad, air): Z0={z0_ms:.1f} Ohm")
    print(f"  L_analytical (total) = {L_ana * 1e12:.1f} pH")
    print(f"  C_analytical (total) = {C_ana * 1e15:.2f} fF")

    result = sim.run_sweep()
    s = result.sparams[0]
    print(f"\nS-matrix at 1 GHz:")
    for i in range(2):
        for j in range(2):
            print(f"  S{i+1}{j+1} = {s[i,j].real:+.4f} {s[i,j].imag:+.4f}j   |S|={abs(s[i,j]):.4f}")

    omega = 2 * math.pi * 1e9
    I = np.eye(2)
    Z = np.sqrt(50.0) * (I + s) @ np.linalg.inv(I - s) * np.sqrt(50.0)
    Y = np.linalg.inv(Z)
    print(f"\nZ-matrix at 1 GHz:")
    for i in range(2):
        for j in range(2):
            print(f"  Z{i+1}{j+1} = {Z[i,j].real:+.2f} {Z[i,j].imag:+.2f}j Ohm")
    print(f"\nY-matrix at 1 GHz:")
    for i in range(2):
        for j in range(2):
            print(f"  Y{i+1}{j+1} = {Y[i,j].real:+.4e} {Y[i,j].imag:+.4e}j S")

    # π-equivalent: Y21 = +j/(wL_series), so L = 1/(w·Im(Y21))
    L_from_Y21 = 1.0 / (omega * Y[1, 0].imag) if Y[1, 0].imag != 0 else float("nan")
    # Each shunt is jwC/2; total C = 2·(Y11.imag + Y21.imag)/w  (Y21 < 0 for shunt-only)
    C_shunt_each = (Y[0, 0].imag + Y[1, 0].imag) / omega
    C_total = 2 * C_shunt_each
    print(f"\nL_fem (from 1/(w·Im(Y21)))  = {L_from_Y21 * 1e12:.1f} pH")
    print(f"C_shunt total                = {C_total * 1e15:.2f} fF")
    print(f"L_analytical (microstrip)    = {L_ana * 1e12:.1f} pH")
    print(f"C_analytical (microstrip)    = {C_ana * 1e15:.2f} fF")
    rel_err_L = abs(L_from_Y21 - L_ana) / L_ana
    rel_err_C = abs(C_total - C_ana) / C_ana
    print(f"L relative error: {rel_err_L * 100:.1f}%")
    print(f"C relative error: {rel_err_C * 100:.1f}%")
    if rel_err_L < 0.30 and rel_err_C < 0.30:
        print("OK")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
