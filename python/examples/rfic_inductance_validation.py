"""
RFIC FEM L-extraction validation: straight wire on Sky130 met5 vs analytical
partial inductance (Rosa formula).

For a straight rectangular conductor of length L, width w, thickness t (w,t << L):

    L_partial ≈ (μ0 / 2π) · L · [ ln(2L / (w + t)) + 0.5 + (w + t) / (3L) ]

Test:
1. Build a 200 μm × 5 μm wire on met5 over a Sky130 substrate
2. Lumped port at each end, referenced to a PEC ground patch at li1
3. Sweep at 1 GHz, convert S → Z, extract L = Im(Z21) / ω

Pass criterion: FEM L within ±25% of Rosa (Rosa is itself ~10% off vs full FEM
because it ignores substrate-image effects, so anything tighter is suspect).
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


def rosa_partial_L(length, width, thickness):
    """Rosa's partial inductance for a flat rectangular conductor."""
    return (MU0 / (2 * math.pi)) * length * (
        math.log(2 * length / (width + thickness))
        + 0.5
        + (width + thickness) / (3 * length)
    )


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

    g = rapidfem.Geometry.from_gds(gds_path, stack=stack, merge=False)
    # The wire from the GDS is the only met5 volume — grab it for fragmentation
    wire_obj = next((o for o in g._objects if o.dim == 3 and o._entity.name == "met5"), None)

    # Footprint: a bit larger than the wire
    foot = (geom["length_um"] * 1.5 * um, 80 * um)
    sub_objs = stack.create_substrate(g, footprint=foot, center=True)
    oxide_obj = sub_objs.get("oxide")

    # Air box on top
    air_h = 100 * um
    air = g.box(foot[0], foot[1], air_h,
                position=(-foot[0] / 2, -foot[1] / 2, stack.top_z))
    air.material = "air"
    air.faces.where(lambda c, _, h=stack.top_z + air_h: abs(c[2] - h) < 1e-12).name = "abc"
    for s in (-1, 1):
        air.faces.where(lambda c, _, s=s, f=foot[0]: abs(c[0] - s * f / 2) < 1e-12).name = "abc"
        air.faces.where(lambda c, _, s=s, f=foot[1]: abs(c[1] - s * f / 2) < 1e-12).name = "abc"

    # Use the rfic.trace_port helper at each wire end — handles fragment internally
    half_L = geom["length_um"] * um / 2
    fragment_targets = [oxide_obj] if oxide_obj is not None else []
    if wire_obj is not None:
        fragment_targets.append(wire_obj)

    rfic.trace_port(g, stack, layer="met5", gnd_layer="li1",
                    position=(-half_L, 0), name="p1",
                    fragment_with=fragment_targets)
    rfic.trace_port(g, stack, layer="met5", gnd_layer="li1",
                    position=(+half_L, 0), name="p2",
                    fragment_with=fragment_targets)

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

        L_ana = rosa_partial_L(geom["length_um"] * um,
                                geom["width_um"] * um,
                                1.26 * um)  # met5 thickness
        print(f"Wire: {geom['length_um']}um x {geom['width_um']}um x 1.26um (met5)")
        print(f"  Rosa L_analytical = {L_ana * 1e9:.4f} nH = {L_ana * 1e12:.1f} pH")

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
        print(f"  L_analytical (Rosa) = {L_ana * 1e12:.1f} pH")
        rel_err = abs(L_fem - L_ana) / L_ana
        print(f"  Relative error: {rel_err * 100:.1f}%")

        if abs(L_fem) > 0 and rel_err < 0.25:
            print("OK")
            return 0
        print("FAIL")
        return 1
    finally:
        os.unlink(gds_path)


if __name__ == "__main__":
    sys.exit(main())
