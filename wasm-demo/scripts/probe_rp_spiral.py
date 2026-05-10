"""Step-by-step probe of build_from_spec on the rp_spiral spec.
Prints a marker line before each major operation so we can see where it hangs."""
import sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "python_src"))
sys.path.insert(0, str(REPO / "wasm-demo" / "scripts"))
sys.path.insert(0, "C:/Repositories/rapidpassives")

import numpy as np
import rapidfem
import gmsh
from dump_rp_designs import build_spiral_spec

t0 = time.perf_counter()
def stamp(msg):
    print(f"[{time.perf_counter() - t0:6.2f}s] {msg}", flush=True)

stamp("building spec")
spec = build_spiral_spec()

stamp("creating Geometry")
g = rapidfem.Geometry()
gmsh.option.setNumber('General.Terminal', 1)

# --- Substrate / oxide / air boxes ---
stack = spec["stack"]
def by_name(n): return next(L for L in stack if L["name"] == n)

xmin = ymin = float("inf"); xmax = ymax = float("-inf")
for poly in spec["polygons"]:
    for i in range(0, len(poly["xy"]), 2):
        x, y = poly["xy"][i], poly["xy"][i+1]
        xmin = min(xmin, x); xmax = max(xmax, x)
        ymin = min(ymin, y); ymax = max(ymax, y)
pad = spec["boundary"]["air_padding_xy"]
fxy = (max(xmax-xmin, 1e-12)+2*pad, max(ymax-ymin, 1e-12)+2*pad)
cx, cy = (xmin+xmax)/2, (ymin+ymax)/2

sub = by_name("substrate"); ox = by_name("oxide")
stamp("box: substrate")
sub_box = g.box(fxy[0], fxy[1], sub["thickness"],
                position=(cx-fxy[0]/2, cy-fxy[1]/2, sub["z"]))
sub_box.name = "substrate"; sub_box.material = "substrate"

stamp("box: oxide")
ox_box = g.box(fxy[0], fxy[1], ox["thickness"],
               position=(cx-fxy[0]/2, cy-fxy[1]/2, ox["z"]))
ox_box.name = "oxide"; ox_box.material = "oxide"

air_h = spec["boundary"]["air_padding_z"]
air_top_z = ox["z"] + ox["thickness"]
stamp("box: air")
air = g.box(fxy[0], fxy[1], air_h,
            position=(cx-fxy[0]/2, cy-fxy[1]/2, air_top_z))
air.material = "air"

# --- Conductor extrusions ---
conductors = []
metal_maxh = spec["maxh"] / 3.0
for i, poly in enumerate(spec["polygons"]):
    layer = by_name(poly["layer"])
    if layer["type"] not in ("metal", "via"):
        continue
    pts = [(poly["xy"][k], poly["xy"][k+1]) for k in range(0, len(poly["xy"]), 2)]
    arr = np.asarray(pts, dtype=np.float64)
    stamp(f"extrude poly {i}: layer={layer['name']} ({len(arr)} verts) z={layer['z']:.3e} t={layer['thickness']:.3e}")
    vol = g._extrude_polygon(arr, z=layer["z"], thickness=layer["thickness"])
    vol.name = layer["name"]
    if layer["type"] == "metal":
        vol.maxh = metal_maxh
    conductors.append((vol, layer["name"]))

# --- Ports ---
port_objs = []
for port in spec["ports"]:
    layer = by_name(port["layer"]); gnd = by_name(port["gnd_layer"])
    z_metal_bot = layer["z"]; z_gnd_top = gnd["z"] + gnd["thickness"]
    size = port.get("size", 6e-6)
    gnd_pad_size = 4 * size
    cxp, cyp = port["x"], port["y"]
    stamp(f"port {port['name']}: ext box on {layer['name']}")
    ext = g.box(size, size, layer["thickness"],
                position=(cxp-size/2, cyp-size/2, z_metal_bot))
    ext.name = layer["name"]; ext.maxh = metal_maxh
    stamp(f"port {port['name']}: gnd plate on {gnd['name']}")
    gnd_pad = g.xy_plate(gnd_pad_size, gnd_pad_size,
                         position=(cxp-gnd_pad_size/2, cyp-gnd_pad_size/2, z_gnd_top))
    gnd_pad.name = f"{port['name']}_gnd"
    stamp(f"port {port['name']}: vertical plate")
    port_plate = g.plate(p0=(cxp-size/2, cyp-size/2, z_gnd_top),
                          width=(size,0,0), height=(0,0,z_metal_bot-z_gnd_top))
    port_plate.name = port["name"]
    port_objs += [ext, gnd_pad, port_plate]

stamp(f"FRAGMENT: substrate + oxide + {len(conductors)} conductors + {len(port_objs)} port objs")
g.fragment(sub_box, ox_box, *(c[0] for c in conductors), *port_objs)
stamp("fragment OK")

stamp("tagging boundary faces")
for vol, name in conductors:
    try:
        vol.faces.name = name
    except Exception as e:
        stamp(f"  warn: {name}: {e}")

stamp("ABC face naming")
air.faces.where(lambda c, _, h=air_top_z + air_h: abs(c[2] - h) < 1e-12).name = "abc"
for s in (-1, 1):
    air.faces.where(lambda c, _, s=s, f=fxy[0]: abs(c[0] - cx - s * f / 2) < 1e-12).name = "abc"
    air.faces.where(lambda c, _, s=s, f=fxy[1]: abs(c[1] - cy - s * f / 2) < 1e-12).name = "abc"
stamp("ABC OK")

stamp(f"calling g.mesh(maxh={spec['maxh']:.3e})")
mesh_bytes, name_to_tag = g.mesh(maxh=spec["maxh"])
stamp(f"mesh OK: {len(mesh_bytes)/1024:.1f} KB, {len(name_to_tag)} groups")
g.close()
stamp("DONE")
