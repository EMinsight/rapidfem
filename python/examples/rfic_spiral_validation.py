"""
RFIC validation: spiral inductor on Sky130 met5, FEM-extracted L vs Mohan analytical.

The Mohan modified-Wheeler formula gives an octagonal spiral's self-inductance
within ~5% of measurement (Mohan et al., JSSC 1999):

    L = K1 · μ0 · n² · davg / (1 + K2 · ρ)
    davg = (Dout + Din) / 2
    ρ    = (Dout - Din) / (Dout + Din)
    K1=2.25, K2=3.55  (octagonal)

The FEM run uses Sky130 top metal (met5), with two lumped ports at each
spiral end referenced to the substrate. From 2-port Z-params:

    L_fem = Im(Z21) / ω   (low frequency, where the inductor dominates)

Pass criterion: FEM matches Mohan within ±20%.  Mohan itself has ~5% error
vs measurement at typical RFIC dimensions, so anything tighter is suspect.
"""
import math
import os
import sys
import tempfile

import gdstk
import numpy as np

import rapidfem
import rapidfem.rfic as rfic

MU0 = 4 * math.pi * 1e-7


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: octagonal spiral + ground pads
# ─────────────────────────────────────────────────────────────────────────────

def octagonal_polygon(cx, cy, radius, sides=8, rotation=math.pi / 8):
    """Regular polygon vertices (counter-clockwise)."""
    return [(cx + radius * math.cos(rotation + 2 * math.pi * i / sides),
             cy + radius * math.sin(rotation + 2 * math.pi * i / sides))
            for i in range(sides)]


def make_spiral_gds(path, *, dout_um=130, n_turns=2, width_um=10, spacing_um=4):
    """Generate an octagonal spiral inductor on Sky130 met5 (gds=72, dt=20).
    Returns a dict with the analytical Mohan L and geometric end positions for ports.
    """
    lib = gdstk.Library(name="spiral_test", unit=1e-6, precision=1e-9)
    cell = lib.new_cell("spiral")

    pitch = width_um + spacing_um
    r_outer = dout_um / 2
    r_inner = r_outer - n_turns * pitch
    if r_inner <= 0:
        raise ValueError(f"too many turns: r_inner={r_inner}, increase Dout or reduce N")

    # Build the spiral as concatenated octagonal annular ring segments.
    # gdstk has FlexPath for spirals — easier than managing polygons by hand.
    pts = []
    sides = 8
    total_steps = n_turns * sides
    for k in range(total_steps + 1):
        # Linearly interpolate radius from r_outer (outer end) inward.
        r = r_outer - (k / sides) * pitch
        ang = math.pi / 8 + 2 * math.pi * k / sides
        pts.append((r * math.cos(ang), r * math.sin(ang)))

    path_geom = gdstk.FlexPath(pts, width_um, layer=72, datatype=20, ends="flush")
    cell.add(path_geom)

    # Two square pads (one at outer start, one at inner end) for port placement
    pad_size = 2 * width_um
    p_outer = pts[0]
    p_inner = pts[-1]
    cell.add(gdstk.rectangle(
        (p_outer[0] - pad_size / 2, p_outer[1] - pad_size / 2),
        (p_outer[0] + pad_size / 2, p_outer[1] + pad_size / 2),
        layer=72, datatype=20,
    ))
    cell.add(gdstk.rectangle(
        (p_inner[0] - pad_size / 2, p_inner[1] - pad_size / 2),
        (p_inner[0] + pad_size / 2, p_inner[1] + pad_size / 2),
        layer=72, datatype=20,
    ))
    lib.write_gds(path)

    # Analytical Mohan
    din_um = dout_um - 2 * (n_turns * pitch)
    davg = (dout_um + din_um) / 2 * 1e-6  # in meters
    rho = (dout_um - din_um) / (dout_um + din_um)
    K1, K2 = 2.25, 3.55
    L_analytical = K1 * MU0 * n_turns ** 2 * davg / (1 + K2 * rho)

    return {
        "L_analytical_H": L_analytical,
        "p_outer": p_outer,
        "p_inner": p_inner,
        "din_um": din_um,
        "dout_um": dout_um,
        "n_turns": n_turns,
        "pad_size_um": pad_size,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEM: load + simulate + extract L
# ─────────────────────────────────────────────────────────────────────────────

def run_fem(gds_path, geom, *, freq_hz=1e9):
    um = 1e-6
    stack = rfic.Stack.sky130()

    g = rapidfem.Geometry.from_gds(gds_path, stack=stack, merge=True)

    # Footprint a bit larger than the spiral
    foot_um = 1.4 * geom["dout_um"]
    foot = (foot_um * um, foot_um * um)
    stack.create_substrate(g, footprint=foot, center=True)

    # Air box above the stack
    air_h = 100 * um
    air = g.box(foot[0], foot[1], air_h,
                position=(-foot[0] / 2, -foot[1] / 2, stack.top_z))
    air.material = "air"
    air.faces.where(lambda c, _, h=stack.top_z + air_h: abs(c[2] - h) < 1e-12).name = "abc"
    for s in (-1, 1):
        air.faces.where(lambda c, _, s=s, f=foot[0]: abs(c[0] - s * f / 2) < 1e-12).name = "abc"
        air.faces.where(lambda c, _, s=s, f=foot[1]: abs(c[1] - s * f / 2) < 1e-12).name = "abc"

    # Two lumped-port plates: vertical from each spiral pad down to the substrate top.
    # The spiral is on met5 at z = 4.365 um; substrate top is at z = -0.18 um (poly z).
    # The lumped port spans the full oxide stack at the pad's xy location.
    z_metal = stack.by_name("met5").z
    z_ref = stack.bottom_z   # substrate top (~poly's z, -0.18 um)
    pad_um = geom["pad_size_um"]
    for label, (px, py) in [("p1", geom["p_outer"]), ("p2", geom["p_inner"])]:
        port_plate = g.plate(
            p0=(px * um - pad_um * um / 2, py * um, z_ref),
            width=(pad_um * um, 0, 0),
            height=(0, 0, z_metal - z_ref),
        )
        port_plate.name = label

    builder = (
        rapidfem.SimulationBuilder()
        .from_geometry(g, maxh=20 * um)
        .frequencies([freq_hz])
        .pec("met5")
        .lumped_port("p1", direction=(0, 0, 1), z0=50.0)
        .lumped_port("p2", direction=(0, 0, 1), z0=50.0)
        .abc("abc", order=1)
        .material("air", er=1.0)
    )
    for spec in stack.material_specs():
        builder = builder.material(**spec)
    sim = builder.build()
    g.close()

    print(f"  Simulation: {sim.n_tets} tets, {sim.n_dofs} DOFs, {sim.n_driven_ports} driven ports")
    result = sim.run_sweep()
    return result


def extract_L_from_S(s2x2, freq_hz, z0=50.0):
    """Convert S to Z, extract L from Im(Z21)/w. s2x2 is (2,2) complex."""
    omega = 2 * math.pi * freq_hz
    s = s2x2
    I = np.eye(2)
    Z0 = z0 * I
    # Z = sqrt(Z0) · (I + S) · (I - S)^-1 · sqrt(Z0)
    Z = np.sqrt(z0) * (I + s) @ np.linalg.inv(I - s) * np.sqrt(z0)
    L_from_Z21 = Z[1, 0].imag / omega
    return Z, L_from_Z21


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
        gds_path = f.name
    try:
        geom = make_spiral_gds(gds_path, dout_um=130, n_turns=2,
                                width_um=10, spacing_um=4)
        print(f"Spiral: Dout={geom['dout_um']}um Din={geom['din_um']}um "
              f"N={geom['n_turns']} turns")
        print(f"  Mohan L_analytical = {geom['L_analytical_H'] * 1e9:.3f} nH")

        result = run_fem(gds_path, geom, freq_hz=1e9)
        s = result.sparams[0]   # (2, 2)
        print(f"\n  S-matrix at 1 GHz:")
        for i in range(2):
            for j in range(2):
                print(f"    S{i+1}{j+1} = {s[i,j].real:+.4f} {s[i,j].imag:+.4f}j   "
                      f"|S| = {abs(s[i,j]):.4f}")

        Z, L_fem = extract_L_from_S(s, freq_hz=1e9, z0=50.0)
        print(f"\n  Z-matrix at 1 GHz:")
        for i in range(2):
            for j in range(2):
                print(f"    Z{i+1}{j+1} = {Z[i,j].real:+.2f} {Z[i,j].imag:+.2f}j Ohm")
        print(f"\n  L_fem (from Im(Z21)/w) = {L_fem * 1e9:.3f} nH")
        print(f"  L_analytical (Mohan)    = {geom['L_analytical_H'] * 1e9:.3f} nH")

        rel_err = abs(L_fem - geom["L_analytical_H"]) / geom["L_analytical_H"]
        print(f"\n  Relative error: {rel_err * 100:.1f}%")
        if rel_err < 0.20:
            print("OK — within 20% of Mohan analytical")
            return 0
        print(f"FAIL — {rel_err*100:.1f}% > 20%")
        return 1
    finally:
        os.unlink(gds_path)


if __name__ == "__main__":
    sys.exit(main())
