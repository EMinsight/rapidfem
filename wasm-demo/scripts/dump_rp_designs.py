"""End-to-end demo: rapidpassives Python generator -> rapidfem Bridge -> mesh.

Picks a couple of rapidpassives' canonical RFIC layout generators
(SpiralInductor, SymmetricInductor), maps their layers onto our Sky130 PDK,
builds a GeometrySpec, runs it through rapidfem.bridge.build_from_spec, and
dumps the resulting (.msh, .toml) into the WASM demo's static directory so
they show up as new examples in the browser.

The mapping is intentionally simplified: every metal-bearing layer
(windings/crossings/vias/centertap) lives on met5, ground shield on li1.
Real designs use multiple metals — that's a follow-up. This script's job
is to prove the rapidpassives->rapidfem pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "python_src"))

import rapidfem
from rapidfem.bridge import build_from_spec

# rapidpassives Python lives outside our repo
RP_REPO = Path("/c/Repositories/rapidpassives") if Path("/c/Repositories/rapidpassives").exists() \
    else Path("C:/Repositories/rapidpassives")
sys.path.insert(0, str(RP_REPO))

from rapidpassives.spiralinductor import SpiralInductor
from rapidpassives.symmetricinductor import SymmetricInductor


def _layers_to_polygons(layers: dict, fem_layer: str) -> list[dict]:
    """Take a rapidpassives `self.layers` dict (LayerName -> list of
    (x_arr, y_arr) polygon tuples) and project ALL polygons onto a single
    FEM layer. Returns SpecPolygon dicts in METERS."""
    out: list[dict] = []
    UM = 1e-6
    for layer_polys in layers.values():
        for xs, ys in layer_polys:
            xy = []
            for x, y in zip(xs, ys):
                xy.extend([float(x) * UM, float(y) * UM])
            out.append({"layer": fem_layer, "xy": xy})
    return out


def _sky130_stack_dict() -> list[dict]:
    """A trimmed Sky130 stack dict (substrate + oxide + li1 + met5 only)
    in the JSON format the bridge expects."""
    return [
        {"name": "substrate", "type": "substrate", "z": -3.018e-5, "thickness": 30e-6,
         "er": 11.9, "conductivity": 10},
        {"name": "oxide", "type": "oxide", "z": -1.8e-7, "thickness": 5.805e-6, "er": 4.2},
        {"name": "li1", "type": "metal", "z": 0.0, "thickness": 1e-7},
        {"name": "met5", "type": "metal", "z": 4.365e-6, "thickness": 1.26e-6},
    ]


def build_spiral_spec() -> dict:
    """Smaller-than-default spiral so the FEM stays fast in WASM."""
    sp = SpiralInductor(Dout=80, N=1, sides=8, width=8, spacing=4)
    polys = _layers_to_polygons(sp.layers, "met5")
    UM = 1e-6
    # rapidpassives places port labels near the outer end of the windings.
    x_lab = (sp.Dout / 2 + sp.width) * UM
    y_lab = (sp.width / 2 + sp.spacing / 2) * UM
    return {
        "name": "rp_spiral",
        "stack": _sky130_stack_dict(),
        "polygons": polys,
        "ports": [
            {"name": "p1", "x": +x_lab, "y": +y_lab, "layer": "met5", "gnd_layer": "li1",
             "size": 6e-6, "z0": 50},
            {"name": "p2", "x": +x_lab, "y": -y_lab, "layer": "met5", "gnd_layer": "li1",
             "size": 6e-6, "z0": 50},
        ],
        "boundary": {"air_padding_xy": 30e-6, "air_padding_z": 30e-6, "abc": "B"},
        "frequencies_hz": [1e9, 5e9, 10e9, 30e9, 60e9, 100e9],
        "maxh": 15e-6,
    }


def build_symmetric_spec() -> dict:
    si = SymmetricInductor(Dout=100, N=2, sides=8, width=6, spacing=3)
    polys = _layers_to_polygons(si.layers, "met5")
    UM = 1e-6
    # SymmetricInductor's two terminals straddle x=0; pick the outer-most
    # winding tangent points.
    xt = (si.Dout / 2 + si.width) * UM
    return {
        "name": "rp_symmetric",
        "stack": _sky130_stack_dict(),
        "polygons": polys,
        "ports": [
            {"name": "p1", "x": -xt, "y": 0, "layer": "met5", "gnd_layer": "li1", "size": 6e-6, "z0": 50},
            {"name": "p2", "x": +xt, "y": 0, "layer": "met5", "gnd_layer": "li1", "size": 6e-6, "z0": 50},
        ],
        "boundary": {"air_padding_xy": 30e-6, "air_padding_z": 30e-6, "abc": "B"},
        "frequencies_hz": [1e9, 5e9, 10e9, 30e9, 60e9, 100e9],
        "maxh": 15e-6,
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
