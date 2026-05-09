"""
RFIC FEM L-extraction validation: straight microstrip on Sky130 met5 vs
Hammerstad-Jensen analytical inductance.

The wire on met5 is referenced by a continuous li1 ground STRIP underneath
(true microstrip topology). For a microstrip with effective εr ≈ 3 (oxide-air
mix, εr_oxide=4.2) and substrate well below:

    Z0 ≈ Z0_air / sqrt(εeff),    v = c / sqrt(εeff)
    L_total = (Z0 / v) · length

Pass criterion: factor-of-2 corridor on L (analytical assumes lossless,
infinite ground; FEM has finite ground strip width and substrate eddy losses).
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
C0 = 2.99792458e8


def microstrip_L(length, w, h, er):
    """Hammerstad-Jensen microstrip total inductance.

    Args:
        length: trace length [m]
        w: trace width [m]
        h: substrate height [m]
        er: substrate relative permittivity (use εr_eff if mixed dielectric)
    """
    u = w / h
    if u <= 1:
        z0_air = 60.0 * math.log(8.0 / u + u / 4.0)
    else:
        z0_air = 120.0 * math.pi / (u + 1.393 + 0.667 * math.log(u + 1.444))
    er_eff = (er + 1) / 2 + (er - 1) / 2 / math.sqrt(1 + 12 * h / w)
    z0 = z0_air / math.sqrt(er_eff)
    v = C0 / math.sqrt(er_eff)
    return (z0 / v) * length, z0, er_eff


def make_wire_gds(path, length_um=200.0, width_um=5.0):
    """Straight wire on met5 (gds=72, dt=20). Returns geometric handles."""
    lib = gdstk.Library(name="wire_test", unit=1e-6, precision=1e-9)
    cell = lib.new_cell("wire")

    cell.add(gdstk.rectangle(
        (-length_um / 2, -width_um / 2),
        (+length_um / 2, +width_um / 2),
        layer=72, datatype=20,
    ))
    lib.write_gds(path)
    return {"length_um": length_um, "width_um": width_um}


def run_fem(gds_path, geom, *, freq_hz=1e9):
    um = 1e-6
    stack = rfic.Stack.sky130()

    # Thin-conductor (2D PEC plate) for the wire — required so the simulator's
    # PEC BC sees a SURFACE physical group (volumes wouldn't match tris_for_tag).
    g = rapidfem.Geometry.from_gds(gds_path, stack=stack, merge=False,
                                    thin_conductors=True)
    wire = next((o for o in g._objects if o.dim == 2 and o._entity.name == "met5"), None)
    if wire is None:
        raise RuntimeError("wire 2D plate not found in GDS")

    # Footprint: a bit larger than the wire
    foot = (geom["length_um"] * 1.5 * um, 80 * um)
    sub_objs = stack.create_substrate(g, footprint=foot, center=True,
                                       fragment_existing=False)
    oxide = sub_objs["oxide"]
    substrate = sub_objs["substrate"]

    # Air box on top — kept outside the big fragment.
    air_h = 100 * um
    air = g.box(foot[0], foot[1], air_h,
                position=(-foot[0] / 2, -foot[1] / 2, stack.top_z))
    air.material = "air"

    # Continuous li1 ground STRIP running below the entire wire length (a true
    # RFIC microstrip return). Wider than the wire (3×) for low parasitic L.
    # Without this the wire is essentially radiating into the lossy substrate
    # and Z0 is undefined.
    pdk_met5 = stack.by_name("met5")
    pdk_li1 = stack.by_name("li1")
    z_metal = pdk_met5.z
    z_gnd = pdk_li1.z + pdk_li1.thickness
    half_L = geom["length_um"] * um / 2
    w_wire = geom["width_um"] * um
    gnd_strip_w = 5 * w_wire
    gnd_strip_L = geom["length_um"] * 1.2 * um
    gnd_strip = g.xy_plate(gnd_strip_L, gnd_strip_w,
                            position=(-gnd_strip_L / 2, -gnd_strip_w / 2, z_gnd))
    gnd_strip.name = "gnd"

    # Vertical port plates straddling the oxide between gnd_strip and wire,
    # plus extension pads on met5 for clean PEC contact with the port top edge.
    ext_size = max(6e-6, w_wire)
    port_w = ext_size
    port_objs = [gnd_strip]
    for label, cx in [("p1", -half_L), ("p2", +half_L)]:
        cy = 0.0
        ext = g.xy_plate(ext_size, ext_size,
                         position=(cx - ext_size / 2, cy - ext_size / 2, z_metal))
        ext.name = "met5"
        port_plate = g.plate(
            p0=(cx - port_w / 2, cy - port_w / 2, z_gnd),
            width=(port_w, 0, 0),
            height=(0, 0, z_metal - z_gnd),
        )
        port_plate.name = label
        port_objs += [ext, port_plate]

    # Single batched fragment so substrate↔oxide↔wire↔gnd↔ports are all conformal.
    g.fragment(substrate, oxide, wire, *port_objs)

    air.faces.where(lambda c, _, h=stack.top_z + air_h: abs(c[2] - h) < 1e-12).name = "abc"
    for s in (-1, 1):
        air.faces.where(lambda c, _, s=s, f=foot[0]: abs(c[0] - s * f / 2) < 1e-12).name = "abc"
        air.faces.where(lambda c, _, s=s, f=foot[1]: abs(c[1] - s * f / 2) < 1e-12).name = "abc"

    builder = (
        rapidfem.SimulationBuilder()
        .from_geometry(g, maxh=15 * um)
        .frequencies([freq_hz])
        .pec("met5", "gnd")
        .lumped_port("p1", direction=(0, 0, 1), z0=50.0)
        .lumped_port("p2", direction=(0, 0, 1), z0=50.0)
        .abc("abc", order=1)
        .material("air", er=1.0)
    )
    for spec in stack.material_specs():
        builder = builder.material(**spec)
    sim = builder.build()
    g.close()

    print(f"  Simulation: {sim.n_tets} tets, {sim.n_dofs} DOFs, {sim.n_driven_ports} driven")
    return sim.run_sweep()


def extract_L_from_S(s, freq_hz, z0=50.0):
    """Extract series-L from π-equivalent: L = 1/(w·Im(Y21)).
    Im(Z21) is dominated by shunt-C at low frequency for short traces."""
    omega = 2 * math.pi * freq_hz
    I = np.eye(2)
    Z = np.sqrt(z0) * (I + s) @ np.linalg.inv(I - s) * np.sqrt(z0)
    Y = np.linalg.inv(Z)
    L_from_Y21 = 1.0 / (omega * Y[1, 0].imag) if Y[1, 0].imag != 0 else float("nan")
    return Z, Y, L_from_Y21


def main() -> int:
    um = 1e-6
    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
        gds_path = f.name
    try:
        geom = make_wire_gds(gds_path, length_um=200, width_um=5)

        # h_eff = met5 z - li1 z_top = 4.365 - 0.10 = 4.265 um. εr_oxide = 4.2.
        h_eff = 4.265 * um
        er_oxide = 4.2
        L_ana, Z0_eff, er_eff = microstrip_L(
            geom["length_um"] * um, geom["width_um"] * um, h_eff, er_oxide,
        )
        print(f"Wire: {geom['length_um']}um x {geom['width_um']}um x 1.26um (met5)")
        print(f"  Microstrip Z0={Z0_eff:.1f} Ohm, er_eff={er_eff:.2f}")
        print(f"  L_analytical = {L_ana * 1e9:.4f} nH = {L_ana * 1e12:.1f} pH")

        result = run_fem(gds_path, geom, freq_hz=1e9)
        s = result.sparams[0]
        print(f"\n  S-matrix at 1 GHz:")
        for i in range(2):
            for j in range(2):
                print(f"    S{i+1}{j+1} = {s[i,j].real:+.4f} {s[i,j].imag:+.4f}j   |S|={abs(s[i,j]):.4f}")

        Z, Y, L_fem = extract_L_from_S(s, freq_hz=1e9, z0=50.0)
        print(f"\n  Z-matrix at 1 GHz:")
        for i in range(2):
            for j in range(2):
                print(f"    Z{i+1}{j+1} = {Z[i,j].real:+.2f} {Z[i,j].imag:+.2f}j Ohm")
        print(f"\n  Y-matrix at 1 GHz:")
        for i in range(2):
            for j in range(2):
                print(f"    Y{i+1}{j+1} = {Y[i,j].real:+.4e} {Y[i,j].imag:+.4e}j S")

        print(f"\n  L_fem (1/(w·Im(Y21))) = {L_fem * 1e12:.1f} pH = {L_fem * 1e9:.4f} nH")
        print(f"  L_analytical (Hammerstad) = {L_ana * 1e12:.1f} pH")
        ratio = L_fem / L_ana if L_ana > 0 else float("inf")
        print(f"  Ratio L_fem / L_ana = {ratio:.2f}")
        if L_fem > 0 and 0.5 < ratio < 2.0:
            print("OK — within factor-of-2 corridor")
            return 0
        print("FAIL — outside corridor")
        return 1
    finally:
        os.unlink(gds_path)


if __name__ == "__main__":
    sys.exit(main())
