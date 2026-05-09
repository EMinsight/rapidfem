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

Pass criterion: FEM matches Mohan within a factor of 2. The FEM result will
typically OVERESTIMATE Mohan because of port-plate parasitic loop inductance
plus the small spiral-end extension pads we add for clean port topology.
Mohan itself has ~5-10% error vs measurement and assumes idealized geometry,
so a factor-of-2 corridor is realistic for a smoke test (not a calibration).
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
    # Port plates in the FEM are placed AT the spiral endpoints with a width
    # smaller than the spiral so they land on the trace without needing extra
    # pad geometry on the GDS (which would create pinched-polygon unions).
    p_outer = pts[0]
    p_inner = pts[-1]
    port_w = 0.5 * width_um
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
        "port_w_um": port_w,
        "trace_w_um": width_um,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEM: load + simulate + extract L
# ─────────────────────────────────────────────────────────────────────────────

def run_fem(gds_path, geom, *, freq_hz=1e9):
    um = 1e-6
    stack = rfic.Stack.sky130()

    # Load spiral as a 2D PEC plate (thin-conductor approximation). The PEC BC
    # in the simulator matches a SURFACE physical group; a 3D extruded spiral
    # would put "met5" on a volume group → invisible to PEC.
    g = rapidfem.Geometry.from_gds(gds_path, stack=stack, merge=False,
                                    thin_conductors=True)
    spiral = next((o for o in g._objects if o.dim == 2 and o._entity.name == "met5"), None)
    if spiral is None:
        raise RuntimeError("spiral 2D plate not found in extruded GDS")

    # Footprint a bit larger than the spiral
    foot_um = 1.4 * geom["dout_um"]
    foot = (foot_um * um, foot_um * um)

    # Substrate + oxide — DO NOT fragment yet; we batch everything below.
    sub_objs = stack.create_substrate(g, footprint=foot, center=True,
                                       fragment_existing=False)
    oxide = sub_objs["oxide"]
    substrate = sub_objs["substrate"]

    # Air box above the stack
    air_h = 100 * um
    air = g.box(foot[0], foot[1], air_h,
                position=(-foot[0] / 2, -foot[1] / 2, stack.top_z))
    air.material = "air"

    # Local ground patches on li1 ONLY under the port footprints — gives each
    # lumped port a return-current reference without forming a continuous shield
    # that would short the spiral's mutual inductance via image currents.
    pdk_met5 = stack.by_name("met5")
    pdk_li1 = stack.by_name("li1")
    z_metal = pdk_met5.z
    z_gnd = pdk_li1.z + pdk_li1.thickness

    # Port + 2D extension pads at each spiral endpoint. Extension pad is also
    # "met5" so it merges into the spiral PEC after fragment.
    ext_size = max(6e-6, geom["trace_w_um"] * um)
    gnd_pad_size = 4 * ext_size
    port_w = ext_size
    port_objs = []
    for label, (px, py) in [("p1", geom["p_outer"]), ("p2", geom["p_inner"])]:
        cx, cy = px * um, py * um
        ext = g.xy_plate(ext_size, ext_size,
                         position=(cx - ext_size / 2, cy - ext_size / 2, z_metal))
        ext.name = "met5"
        gnd = g.xy_plate(gnd_pad_size, gnd_pad_size,
                         position=(cx - gnd_pad_size / 2, cy - gnd_pad_size / 2, z_gnd))
        gnd.name = f"{label}_gnd"
        port_plate = g.plate(
            p0=(cx - port_w / 2, cy - port_w / 2, z_gnd),
            width=(port_w, 0, 0),
            height=(0, 0, z_metal - z_gnd),
        )
        port_plate.name = label
        port_objs += [ext, gnd, port_plate]

    # ── ONE conformal fragment over everything ────────────────────────────
    g.fragment(substrate, oxide, spiral, *port_objs)
    # Air is separate (sits above oxide) — no need to fragment with it,
    # but tag its outer faces for ABC.
    air.faces.where(lambda c, _, h=stack.top_z + air_h: abs(c[2] - h) < 1e-12).name = "abc"
    for s in (-1, 1):
        air.faces.where(lambda c, _, s=s, f=foot[0]: abs(c[0] - s * f / 2) < 1e-12).name = "abc"
        air.faces.where(lambda c, _, s=s, f=foot[1]: abs(c[1] - s * f / 2) < 1e-12).name = "abc"


    builder = (
        rapidfem.SimulationBuilder()
        .from_geometry(g, maxh=15 * um)
        .frequencies([freq_hz])
        .pec("met5", "p1_gnd", "p2_gnd")
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
    """Convert S to Z and Y, extract series L from π-equivalent.

    For a short 2-port containing a series-L between two shunt capacitances
    (typical of any RFIC trace/inductor), Z21 is dominated by the shunt arms
    at low frequency. The series inductance is recovered from
        Y21 = +j / (w · L_series)   →   L = 1 / (w · Im(Y21))
    """
    omega = 2 * math.pi * freq_hz
    s = s2x2
    I = np.eye(2)
    Z = np.sqrt(z0) * (I + s) @ np.linalg.inv(I - s) * np.sqrt(z0)
    Y = np.linalg.inv(Z)
    L_from_Y21 = 1.0 / (omega * Y[1, 0].imag) if Y[1, 0].imag != 0 else float("nan")
    return Z, Y, L_from_Y21


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

        Z, Y, L_fem = extract_L_from_S(s, freq_hz=1e9, z0=50.0)
        print(f"\n  Z-matrix at 1 GHz:")
        for i in range(2):
            for j in range(2):
                print(f"    Z{i+1}{j+1} = {Z[i,j].real:+.2f} {Z[i,j].imag:+.2f}j Ohm")
        print(f"\n  Y-matrix at 1 GHz:")
        for i in range(2):
            for j in range(2):
                print(f"    Y{i+1}{j+1} = {Y[i,j].real:+.4e} {Y[i,j].imag:+.4e}j S")
        print(f"\n  L_fem (from 1/(w·Im(Y21))) = {L_fem * 1e9:.3f} nH")
        print(f"  L_analytical (Mohan)        = {geom['L_analytical_H'] * 1e9:.3f} nH")

        rel_err = abs(L_fem - geom["L_analytical_H"]) / geom["L_analytical_H"]
        print(f"\n  Relative error: {rel_err * 100:.1f}%")
        # Smoke test: factor-of-2 corridor, see module docstring for rationale.
        ratio = L_fem / geom["L_analytical_H"]
        if 0.5 < ratio < 2.0 and L_fem > 0:
            print(f"OK — L_fem/L_mohan = {ratio:.2f} (within factor-of-2 corridor)")
            return 0
        print(f"FAIL — L_fem/L_mohan = {ratio:.2f} outside [0.5, 2.0]")
        return 1
    finally:
        os.unlink(gds_path)


if __name__ == "__main__":
    sys.exit(main())
