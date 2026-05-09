"""Pre-build a spiral-inductor example for the WASM demo (mirrors
`python/examples/rfic_spiral_validation.py`)."""
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


def make_spiral_gds(path, *, dout_um=130, n_turns=2, width_um=10, spacing_um=4):
    lib = gdstk.Library(name="spiral", unit=1e-6, precision=1e-9)
    cell = lib.new_cell("spiral")
    pitch = width_um + spacing_um
    r_outer = dout_um / 2
    r_inner = r_outer - n_turns * pitch
    if r_inner <= 0:
        raise ValueError("too many turns")
    pts = []
    sides = 8
    for k in range(n_turns * sides + 1):
        r = r_outer - (k / sides) * pitch
        ang = math.pi / 8 + 2 * math.pi * k / sides
        pts.append((r * math.cos(ang), r * math.sin(ang)))
    cell.add(gdstk.FlexPath(pts, width_um, layer=72, datatype=20, ends="flush"))
    lib.write_gds(path)
    return {"p_outer": pts[0], "p_inner": pts[-1],
            "trace_w_um": width_um, "dout_um": dout_um}


def build_spiral_demo(out_msh: Path, out_toml: Path,
                       freqs_hz=None, dout_um=80, n_turns=1):
    if freqs_hz is None:
        # Sweep through the self-resonance. With L≈190pH, C≈28fF the SRF is
        # around 70 GHz, so 1..100 GHz captures the inductive region, the
        # resonance peak, and the capacitive tail beyond.
        freqs_hz = [1e9, 5e9, 10e9, 20e9, 30e9, 40e9, 50e9, 60e9, 70e9, 80e9, 100e9]
    um = 1e-6

    with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
        gds_path = f.name
    try:
        geom = make_spiral_gds(gds_path, dout_um=dout_um, n_turns=n_turns,
                                width_um=8, spacing_um=4)
        stack = rfic.Stack.sky130()
        # Aggressively thin the substrate for the WASM demo. Real Sky130 wafer
        # is 280um thick — fine native, but a 280um × 1.4·dout box meshed even
        # at maxh=20um is hundreds of thousands of tets. A 30um substrate
        # gives the right qualitative behavior (lossy ground reference) while
        # keeping the FEM tractable in the browser.
        stack.substrate_thickness = 30e-6
        g = rapidfem.Geometry.from_gds(gds_path, stack=stack, merge=False,
                                        thin_conductors=True)
        spiral = next(o for o in g._objects
                      if o.dim == 2 and o._entity.name == "met5")

        foot = (1.4 * dout_um * um, 1.4 * dout_um * um)
        sub_objs = stack.create_substrate(g, footprint=foot, center=True,
                                            fragment_existing=False)
        oxide, substrate = sub_objs["oxide"], sub_objs["substrate"]

        air_h = 30 * um   # shrunk from 100um to keep tet count low
        air = g.box(foot[0], foot[1], air_h,
                    position=(-foot[0]/2, -foot[1]/2, stack.top_z))
        air.material = "air"

        pdk_met5 = stack.by_name("met5")
        pdk_li1 = stack.by_name("li1")
        z_metal = pdk_met5.z
        z_gnd = pdk_li1.z + pdk_li1.thickness
        ext_size = max(6e-6, geom["trace_w_um"] * um)
        gnd_pad_size = 4 * ext_size
        port_w = ext_size
        port_objs = []
        for label, (px, py) in [("p1", geom["p_outer"]), ("p2", geom["p_inner"])]:
            cx, cy = px*um, py*um
            ext = g.xy_plate(ext_size, ext_size,
                             position=(cx-ext_size/2, cy-ext_size/2, z_metal))
            ext.name = "met5"
            gnd = g.xy_plate(gnd_pad_size, gnd_pad_size,
                             position=(cx-gnd_pad_size/2, cy-gnd_pad_size/2, z_gnd))
            gnd.name = f"{label}_gnd"
            port_plate = g.plate(p0=(cx-port_w/2, cy-port_w/2, z_gnd),
                                  width=(port_w, 0, 0),
                                  height=(0, 0, z_metal-z_gnd))
            port_plate.name = label
            port_objs += [ext, gnd, port_plate]
        g.fragment(substrate, oxide, spiral, *port_objs)

        air.faces.where(lambda c, _, h=stack.top_z + air_h: abs(c[2] - h) < 1e-12).name = "abc"
        for s in (-1, 1):
            air.faces.where(lambda c, _, s=s, f=foot[0]: abs(c[0] - s*f/2) < 1e-12).name = "abc"
            air.faces.where(lambda c, _, s=s, f=foot[1]: abs(c[1] - s*f/2) < 1e-12).name = "abc"

        builder = (
            rapidfem.SimulationBuilder()
            .from_geometry(g, maxh=15 * um)   # finer now that air+substrate are thinner
            .frequencies(freqs_hz)
            .pec("met5", "p1_gnd", "p2_gnd")
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
    build_spiral_demo(web_dir / "spiral.msh", web_dir / "spiral.toml")
