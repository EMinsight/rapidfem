"""End-to-end demo: rapidpassives Python generator -> rapidfem Bridge -> mesh.

Picks a couple of rapidpassives' RFIC layout generators (SpiralInductor,
SymmetricInductor), maps each rapidpassives layer onto the matching Sky130
stack layer (windings -> met5, crossings -> met4, vias -> via4, pgs -> li1),
extrudes them as 3D conductors, and runs the resulting GeometrySpec through
rapidfem.bridge.build_from_spec to produce (.msh, .toml) for the WASM demo.

Conductors get a refined maxh (one third of the global value) so the mesh
captures the trace edges. Boundary faces are auto-tagged in the bridge so
PEC BC matches the proper surface group.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "python_src"))

import rapidfem
from rapidfem.bridge import build_from_spec

RP_REPO = Path("C:/Repositories/rapidpassives")
sys.path.insert(0, str(RP_REPO))

from rapidpassives.spiralinductor import SpiralInductor
from rapidpassives.symmetricinductor import SymmetricInductor


# rapidpassives layer-name -> Sky130 stack layer
RP_LAYER_MAP = {
    "windings":     "met5",
    "windings_m4":  "met5",
    "windings_m2":  "met5",
    "crossings":    "met4",
    "crossings_m1": "met4",
    "vias":         "via4",
    "vias1":        "via4",
    "vias2":        "via4",
    "vias3":        "via4",
    "centertap":    "met5",
    "pgs":          "li1",
    "guard_ring":   "met5",
}


def _sky130_stack_dict() -> list[dict]:
    """Trimmed Sky130 stack covering the layers our rapidpassives demos use."""
    return [
        {"name": "substrate", "type": "substrate", "z": -3.018e-5, "thickness": 30e-6,
         "er": 11.9, "conductivity": 10},
        {"name": "oxide",     "type": "oxide",     "z": -1.8e-7,   "thickness": 5.805e-6, "er": 4.2},
        {"name": "li1",       "type": "metal",     "z":  0.0,      "thickness": 1.0e-7},
        {"name": "met4",      "type": "metal",     "z":  3.515e-6, "thickness": 0.65e-6},
        {"name": "via4",      "type": "via",       "z":  4.165e-6, "thickness": 0.20e-6},
        {"name": "met5",      "type": "metal",     "z":  4.365e-6, "thickness": 1.26e-6},
    ]


def _layers_to_polygons(layers: dict) -> list[dict]:
    """Project rapidpassives' per-layer polygons onto FEM stack layers via
    RP_LAYER_MAP. Layers absent from the map are silently dropped."""
    out: list[dict] = []
    UM = 1e-6
    for rp_layer, polys in layers.items():
        fem_layer = RP_LAYER_MAP.get(rp_layer)
        if not fem_layer or not polys:
            continue
        for xs, ys in polys:
            xy = []
            for x, y in zip(xs, ys):
                xy.extend([float(x) * UM, float(y) * UM])
            out.append({"layer": fem_layer, "xy": xy})
    return out


def build_spiral_spec() -> dict:
    sp = SpiralInductor(Dout=80, N=1, sides=8, width=8, spacing=4)
    polys = _layers_to_polygons(sp.layers)
    UM = 1e-6
    # rapidpassives port labels: P1 on windings (top metal), P2 on crossings
    # (lower metal), both at the outer edge of the spiral.
    x_lab = (sp.Dout / 2 + sp.width) * UM
    y_lab = (sp.width / 2 + sp.spacing / 2) * UM
    return {
        "name": "rp_spiral",
        "stack": _sky130_stack_dict(),
        "polygons": polys,
        "ports": [
            # P1 sits on met5 (windings); P2 on met4 (crossings under-pass)
            {"name": "p1", "x": +x_lab, "y": +y_lab, "layer": "met5", "gnd_layer": "li1",
             "size": 6e-6, "z0": 50},
            {"name": "p2", "x": +x_lab, "y": -y_lab, "layer": "met4", "gnd_layer": "li1",
             "size": 6e-6, "z0": 50},
        ],
        "boundary": {"air_padding_xy": 30e-6, "air_padding_z": 50e-6, "abc": "B"},
        "frequencies_hz": [1e9, 5e9, 10e9, 30e9, 60e9, 100e9],
        "maxh": 12e-6,
    }


def build_symmetric_spec() -> dict:
    si = SymmetricInductor(Dout=100, N=2, sides=8, width=6, spacing=3)
    polys = _layers_to_polygons(si.layers)
    UM = 1e-6
    # Symmetric inductor terminals: rapidpassives places P1/P2 at the two
    # outer windings on opposite sides. Both end on the top winding (met5);
    # crossings are internal under-passes.
    xt = (si.Dout / 2 + si.width) * UM
    return {
        "name": "rp_symmetric",
        "stack": _sky130_stack_dict(),
        "polygons": polys,
        "ports": [
            {"name": "p1", "x": -xt, "y": 0, "layer": "met5", "gnd_layer": "li1", "size": 6e-6, "z0": 50},
            {"name": "p2", "x": +xt, "y": 0, "layer": "met5", "gnd_layer": "li1", "size": 6e-6, "z0": 50},
        ],
        "boundary": {"air_padding_xy": 30e-6, "air_padding_z": 50e-6, "abc": "B"},
        "frequencies_hz": [1e9, 5e9, 10e9, 30e9, 60e9, 100e9],
        "maxh": 12e-6,
    }


def main() -> int:
    out_dir = REPO / "wasm-demo" / "app" / "static" / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec_fn in (build_spiral_spec, build_symmetric_spec):
        spec = spec_fn()
        name = spec["name"]
        print(f"== {name} ==")
        builder = build_from_spec(spec)
        msh = out_dir / f"{name}.msh"
        toml = out_dir / f"{name}.toml"
        builder.dump(str(msh), str(toml))
        sz = msh.stat().st_size / 1024
        print(f"  -> {msh.name}  ({sz:.0f} KB)")
        print(f"  -> {toml.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
