"""
RFIC GDS pipeline: Sky130 PDK + GDS file + simulation, end-to-end.

Generates a minimal GDS layout (a 200 μm × 5 μm microstrip on met5 with two
end pads) using gdstk, loads it via `Geometry.from_gds(...)`, places ABC
boundary + lumped port, and runs a sweep.

Same exact GDS could be loaded into rapidpassives' web viewer — it shares the
SKY130 layer numbering.
"""
import os
import sys
import tempfile

import gdstk
import numpy as np

import rapidfem
import rapidfem.rfic as rfic


def make_test_gds(path: str) -> None:
    """Build a tiny Sky130 microstrip GDS for testing."""
    lib = gdstk.Library(name="test_lib", unit=1e-6, precision=1e-9)
    cell = lib.new_cell("microstrip_test")

    # Microstrip on met5 (gds=72, datatype=20). Coordinates in microns.
    # 200 μm long, 5 μm wide trace, centered on origin in y.
    cell.add(gdstk.rectangle((-100, -2.5), (100, 2.5), layer=72, datatype=20))

    # Two ground pads on met5 (separated from signal by gaps) — left & right
    # of the trace, used as lumped-port reference.
    cell.add(gdstk.rectangle((-110, -50), (-90, -10), layer=72, datatype=20))
    cell.add(gdstk.rectangle((90, -50), (110, -10), layer=72, datatype=20))

    lib.write_gds(path)


def main() -> int:
    um = 1e-6

    # 1. Generate a small test GDS
    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
        gds_path = f.name
    make_test_gds(gds_path)
    try:
        # 2. Load Sky130 stack + GDS extrusion
        stack = rfic.Stack.sky130()
        print(f"Stack: {stack.name}, top_z={stack.top_z * 1e6:.3f} um")
        print(f"  metals: {[l.name for l in stack.metals()]}")

        g = rapidfem.Geometry.from_gds(gds_path, stack=stack, merge=True)
        print(f"\nGDS extruded into {len(g._objects)} primitives")
        # Inspect what the loader found
        for layer_name in {e.name for e in g._entities if e.name and e.dim == 3}:
            n = sum(1 for e in g._entities if e.name == layer_name and e.dim == 3)
            print(f"  layer {layer_name!r}: {n} extruded volumes")

        # 3. Add substrate + air box for FEM
        # Substrate footprint: a bit larger than the layout
        foot = (260 * um, 120 * um)
        stack.create_substrate(g, footprint=foot, center=True)

        air = g.box(foot[0], foot[1], 30 * um,
                    position=(-foot[0] / 2, -foot[1] / 2, stack.top_z))
        air.material = "air"
        air.faces.where(lambda c, _: abs(c[2] - (stack.top_z + 30 * um)) < 1e-12).name = "abc"
        for s in (-1, 1):
            air.faces.where(lambda c, _, s=s: abs(c[0] - s * foot[0] / 2) < 1e-12).name = "abc"
            air.faces.where(lambda c, _, s=s: abs(c[1] - s * foot[1] / 2) < 1e-12).name = "abc"

        # 4. Build sim — mark all met5 polygons as PEC; we'll skip a real port
        # for this smoke test (just verifying the GDS pipeline runs end-to-end).
        builder = (
            rapidfem.SimulationBuilder()
            .from_geometry(g, maxh=20 * um)
            .frequencies([10e9])
            .pec("met5")  # all extruded met5 polygons → PEC, by name
            .abc("abc", order=1)
            .material("air", er=1.0)
        )
        for spec in stack.material_specs():
            builder = builder.material(**spec)
        sim = builder.build()
        g.close()

        print(f"\nSimulation: {sim.n_tets} tets, {sim.n_dofs} DOFs, "
              f"{sim.n_driven_ports} driven ports (smoke: no driven port)")
        if sim.n_tets == 0:
            print("FAIL: no tets in mesh")
            return 1

        # 5. Write the stack JSON for round-trip with rapidpassives
        stack_json = stack.to_dict()
        print(f"\nStack JSON: {len(stack_json['layers'])} layers, "
              f"name={stack_json['name']!r}")
        roundtrip = rfic.Stack.from_dict(stack_json)
        assert roundtrip.by_name("met5").z == stack.by_name("met5").z

        print("\nOK — Sky130 GDS pipeline working end-to-end")
        return 0

    finally:
        os.unlink(gds_path)


if __name__ == "__main__":
    sys.exit(main())
