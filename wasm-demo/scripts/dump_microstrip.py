"""Pre-build a microstrip example for the WASM demo.

Replicates the geometry from `python/examples/rfic_inductance_validation.py`
but stops short of solving — instead dumps mesh + TOML config to
`wasm-demo/web/<name>.msh` + `<name>.toml` so the in-browser FEM can pick
them up and run the solve client-side.
"""
import math
import os
import sys
import tempfile
from pathlib import Path

import gdstk

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "python_src"))

import rapidfem
import rapidfem.rfic as rfic


def make_wire_gds(path, length_um=200.0, width_um=5.0):
    lib = gdstk.Library(name="microstrip", unit=1e-6, precision=1e-9)
    cell = lib.new_cell("wire")
    cell.add(gdstk.rectangle(
        (-length_um / 2, -width_um / 2),
        (+length_um / 2, +width_um / 2),
        layer=72, datatype=20,
    ))
    lib.write_gds(path)


def build_microstrip_demo(out_msh: Path, out_toml: Path,
                          length_um=200.0, width_um=5.0,
                          freqs_hz=None):
    if freqs_hz is None:
        freqs_hz = [1e9, 2e9, 3e9, 4e9, 5e9]   # 5 pts to keep WASM memory usage modest
    um = 1e-6

    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
        gds_path = f.name
    try:
        make_wire_gds(gds_path, length_um=length_um, width_um=width_um)

        stack = rfic.Stack.sky130()
        # Thin substrate for WASM-friendly mesh (real Sky130 is 280um thick;
        # 30um still gives the right lossy-ground reference behavior).
        stack.substrate_thickness = 30e-6
        g = rapidfem.Geometry.from_gds(gds_path, stack=stack, merge=False,
                                        thin_conductors=True)
        wire = next(o for o in g._objects if o.dim == 2 and o._entity.name == "met5")

        foot = (length_um * 1.5 * um, 80 * um)
        sub_objs = stack.create_substrate(g, footprint=foot, center=True,
                                            fragment_existing=False)
        oxide, substrate = sub_objs["oxide"], sub_objs["substrate"]

        air_h = 30 * um   # thin air for WASM mesh budget
        air = g.box(foot[0], foot[1], air_h,
                    position=(-foot[0] / 2, -foot[1] / 2, stack.top_z))
        air.material = "air"

        pdk_met5 = stack.by_name("met5")
        pdk_li1 = stack.by_name("li1")
        z_metal = pdk_met5.z
        z_gnd = pdk_li1.z + pdk_li1.thickness
        half_L = length_um * um / 2
        w_wire = width_um * um
        gnd_strip_w = 5 * w_wire
        gnd_strip_L = length_um * 1.2 * um
        gnd_strip = g.xy_plate(gnd_strip_L, gnd_strip_w,
                                position=(-gnd_strip_L / 2, -gnd_strip_w / 2, z_gnd))
        gnd_strip.name = "gnd"

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
        g.fragment(substrate, oxide, wire, *port_objs)

        air.faces.where(lambda c, _, h=stack.top_z + air_h: abs(c[2] - h) < 1e-12).name = "abc"
        for s in (-1, 1):
            air.faces.where(lambda c, _, s=s, f=foot[0]: abs(c[0] - s * f / 2) < 1e-12).name = "abc"
            air.faces.where(lambda c, _, s=s, f=foot[1]: abs(c[1] - s * f / 2) < 1e-12).name = "abc"

        # 15um maxh gives proper trace resolution; thin air/substrate keep
        # total tet count tractable (~5-8k tets) for browser.
        builder = (
            rapidfem.SimulationBuilder()
            .from_geometry(g, maxh=15 * um)
            .frequencies(freqs_hz)
            .pec("met5", "gnd")
            .lumped_port("p1", direction=(0, 0, 1), z0=50.0)
            .lumped_port("p2", direction=(0, 0, 1), z0=50.0)
            .abc("abc", order=1)
            .material("air", er=1.0)
        )
        for spec in stack.material_specs():
            builder = builder.material(**spec)

        out_msh.parent.mkdir(parents=True, exist_ok=True)
        builder.dump(str(out_msh), str(out_toml))
        g.close()
        sz = out_msh.stat().st_size / 1024
        print(f"wrote {out_msh} ({sz:.1f} KB)")
        print(f"wrote {out_toml}")
    finally:
        os.unlink(gds_path)


if __name__ == "__main__":
    web_dir = REPO / "wasm-demo" / "web"
    build_microstrip_demo(
        web_dir / "microstrip.msh",
        web_dir / "microstrip.toml",
        length_um=200.0, width_um=5.0,
    )
