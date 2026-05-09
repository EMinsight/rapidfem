"""rapidfem mesher entry point for the rapidpassives → rapidfem bridge.

Consumes a GeometrySpec JSON (see ``wasm-demo/app/src/lib/bridge/spec.ts``)
and emits a (.msh, .toml) pair the WASM solver loads.

Workflow:

    rapidpassives (JS) → GeometrySpec (JSON)
                       ↓
                 rapidfem.bridge.build_from_spec(spec)
                       ↓
              .msh + config.toml  →  WASM solver

Run from CLI::

    python -m rapidfem.bridge spec.json -o build/
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rapidfem


def _layer_by_name(stack_dicts: list[dict], name: str) -> dict | None:
    for L in stack_dicts:
        if L["name"] == name:
            return L
    return None


def build_from_spec(spec: dict[str, Any]) -> rapidfem.SimulationBuilder:
    """Translate a parsed GeometrySpec dict into a `SimulationBuilder` ready
    to ``.dump()`` mesh + TOML or ``.build()`` and run."""
    g = rapidfem.Geometry()
    stack = spec["stack"]

    # ── Dielectric volumes (substrate, oxide, air) ─────────────────────────
    sub = _layer_by_name(stack, "substrate")
    oxide = _layer_by_name(stack, "oxide")
    if not sub or not oxide:
        raise ValueError("spec.stack must contain 'substrate' and 'oxide' layers")

    # Footprint from polygon bbox + boundary padding
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for poly in spec["polygons"]:
        for i in range(0, len(poly["xy"]), 2):
            x, y = poly["xy"][i], poly["xy"][i + 1]
            xmin = min(xmin, x); xmax = max(xmax, x)
            ymin = min(ymin, y); ymax = max(ymax, y)
    pad = spec["boundary"]["air_padding_xy"]
    fxy = (max(xmax - xmin, 1e-12) + 2 * pad,
           max(ymax - ymin, 1e-12) + 2 * pad)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    sub_box = g.box(
        fxy[0], fxy[1], sub["thickness"],
        position=(cx - fxy[0] / 2, cy - fxy[1] / 2, sub["z"]),
    )
    sub_box.name = "substrate"
    sub_box.material = "substrate"

    ox_box = g.box(
        fxy[0], fxy[1], oxide["thickness"],
        position=(cx - fxy[0] / 2, cy - fxy[1] / 2, oxide["z"]),
    )
    ox_box.name = "oxide"
    ox_box.material = "oxide"

    air_h = spec["boundary"]["air_padding_z"]
    air_top_z = oxide["z"] + oxide["thickness"]
    air = g.box(
        fxy[0], fxy[1], air_h,
        position=(cx - fxy[0] / 2, cy - fxy[1] / 2, air_top_z),
    )
    air.material = "air"

    # ── Metal polygons as 2D PEC plates ────────────────────────────────────
    # (thin-conductor approximation — see Geometry.from_gds(thin_conductors=True))
    plates: list = []
    for poly in spec["polygons"]:
        layer = _layer_by_name(stack, poly["layer"])
        if not layer or layer["type"] != "metal":
            continue
        pts = [(poly["xy"][i], poly["xy"][i + 1]) for i in range(0, len(poly["xy"]), 2)]
        import numpy as np
        arr = np.asarray(pts, dtype=np.float64)
        plate = g._plate_polygon(arr, z=layer["z"])
        plate.name = layer["name"]
        plates.append(plate)

    # ── Ports: extension pad on layer + ground patch on gnd_layer + port plate
    port_objs: list = []
    port_names: list[str] = []
    for port in spec["ports"]:
        layer = _layer_by_name(stack, port["layer"])
        gnd = _layer_by_name(stack, port["gnd_layer"])
        if not layer or not gnd:
            raise ValueError(f"port {port['name']!r} references unknown layer/gnd_layer")
        z_metal = layer["z"]
        z_gnd = gnd["z"] + gnd["thickness"]
        size = port.get("size") or 6e-6
        gnd_pad_size = 4 * size
        cxp, cyp = port["x"], port["y"]

        ext = g.xy_plate(size, size,
                         position=(cxp - size / 2, cyp - size / 2, z_metal))
        ext.name = layer["name"]
        gnd_pad = g.xy_plate(gnd_pad_size, gnd_pad_size,
                              position=(cxp - gnd_pad_size / 2, cyp - gnd_pad_size / 2, z_gnd))
        gnd_pad.name = f"{port['name']}_gnd"
        port_plate = g.plate(
            p0=(cxp - size / 2, cyp - size / 2, z_gnd),
            width=(size, 0, 0),
            height=(0, 0, z_metal - z_gnd),
        )
        port_plate.name = port["name"]
        port_objs += [ext, gnd_pad, port_plate]
        port_names.append(port["name"])

    # ── Single batched conformal fragment ────────────────────────────────
    g.fragment(sub_box, ox_box, *plates, *port_objs)

    # ABC on outer faces of air
    air.faces.where(lambda c, _, h=air_top_z + air_h: abs(c[2] - h) < 1e-12).name = "abc"
    for s in (-1, 1):
        air.faces.where(lambda c, _, s=s, f=fxy[0]: abs(c[0] - cx - s * f / 2) < 1e-12).name = "abc"
        air.faces.where(lambda c, _, s=s, f=fxy[1]: abs(c[1] - cy - s * f / 2) < 1e-12).name = "abc"

    # ── SimulationBuilder ─────────────────────────────────────────────────
    metal_pec_names = [L["name"] for L in stack if L["type"] == "metal"]
    gnd_names = [f"{p['name']}_gnd" for p in spec["ports"]]

    builder = (
        rapidfem.SimulationBuilder()
        .from_geometry(g, maxh=spec["maxh"])
        .frequencies(spec["frequencies_hz"])
        .pec(*metal_pec_names, *gnd_names)
        .abc("abc", order=1)
        .material("air", er=1.0)
    )
    for port in spec["ports"]:
        builder = builder.lumped_port(
            port["name"],
            direction=tuple(port.get("direction", (0, 0, 1))),
            z0=port.get("z0", 50.0),
        )
    # Materials from stack (substrate + oxide + any other dielectric)
    for L in stack:
        if L["type"] in ("substrate", "oxide", "air"):
            mat = dict(name=L["name"],
                       er=L.get("er", 1.0),
                       conductivity=L.get("conductivity", 0.0))
            if "tan_d" in L:
                mat["tand"] = L["tan_d"]
            builder = builder.material(**mat)

    builder._geometry = g  # pin so it's not gc'd before .dump()
    return builder


def main() -> int:
    parser = argparse.ArgumentParser(description="rapidfem GeometrySpec mesher")
    parser.add_argument("spec", type=Path, help="GeometrySpec JSON file")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("."),
                        help="output dir for the .msh and .toml")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    builder = build_from_spec(spec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = spec.get("name", args.spec.stem)
    msh_path = args.out_dir / f"{name}.msh"
    toml_path = args.out_dir / f"{name}.toml"
    builder.dump(str(msh_path), str(toml_path))
    print(f"wrote {msh_path}")
    print(f"wrote {toml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
