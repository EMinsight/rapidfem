#!/usr/bin/env python3
"""
RapidPassives → RapidFEM bridge.

Reads the FEM JSON export from RapidPassives and produces:
  - A 3D tetrahedral mesh (.msh) via gmsh
  - A matching RapidFEM config (.toml)

Usage:
  python passives_to_fem.py layout.json --output inductor
  # Produces: inductor.msh + inductor.toml

Then run:
  rapidfem inductor.toml
"""

import json
import argparse
import gmsh
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser(description="Convert RapidPassives export to RapidFEM mesh + config")
    parser.add_argument("input", help="FEM JSON export from RapidPassives")
    parser.add_argument("--output", "-o", default="passive", help="Output prefix (default: passive)")
    parser.add_argument("--lc", type=float, default=0, help="Mesh element size in um (0=auto)")
    parser.add_argument("--air-height", type=float, default=0, help="Air box height above metals in um (0=auto)")
    parser.add_argument("--boundary", choices=["pec", "abc"], default="pec", help="Outer boundary type")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    stack = data["stack"]
    layers = data["layers"]
    ports = data["ports"]
    sim = data["sim"]

    # Scale: everything in um, convert to meters at the end
    um = 1.0  # work in um, convert to meters for gmsh

    # Find bounding box of all metal polygons
    all_x, all_y = [], []
    for layer in layers:
        for poly in layer["polygons"]:
            for pt in poly:
                all_x.append(pt[0])
                all_y.append(pt[1])

    if not all_x:
        print("ERROR: No metal polygons found in export")
        sys.exit(1)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    extent = max(x_max - x_min, y_max - y_min)

    # Margins around the layout
    margin = extent * 0.3
    box_x_min = x_min - margin
    box_x_max = x_max + margin
    box_y_min = y_min - margin
    box_y_max = y_max + margin

    # Z extents
    metal_defs = {m["name"]: m for m in stack["metals"]}
    z_metals = [(m["z_um"] - m["thickness_um"]/2, m["z_um"] + m["thickness_um"]/2) for m in stack["metals"]]
    z_metal_min = min(z[0] for z in z_metals) if z_metals else 0
    z_metal_max = max(z[1] for z in z_metals) if z_metals else 1

    # Substrate: below the metals
    substrate_top = z_metal_min - 1  # 1um oxide gap below lowest metal
    substrate_thickness = 50  # um (simplified — real substrates are 300um but we truncate)
    substrate_bot = substrate_top - substrate_thickness

    # Air box above metals
    air_height = args.air_height if args.air_height > 0 else extent * 0.5
    z_top = z_metal_max + air_height

    # Mesh element size
    lc = args.lc if args.lc > 0 else extent / 15

    print(f"Layout: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}] um")
    print(f"Metals: z=[{z_metal_min:.2f}, {z_metal_max:.2f}] um")
    print(f"Domain: z=[{substrate_bot:.1f}, {z_top:.1f}] um, lc={lc:.2f} um")
    print(f"Metals: {len(stack['metals'])} layers, {sum(len(l['polygons']) for l in layers)} polygons")
    print(f"Ports: {len(ports)}")

    # Convert everything to meters
    scale = 1e-6  # um → m

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("passive")

    # Physical group tag counters
    tag_pec = 1
    tag_counter = 10  # start material/port tags from 10

    # Create the enclosing box (oxide + air)
    box = gmsh.model.occ.addBox(
        box_x_min * scale, box_y_min * scale, substrate_bot * scale,
        (box_x_max - box_x_min) * scale,
        (box_y_max - box_y_min) * scale,
        (z_top - substrate_bot) * scale,
    )

    # Create substrate volume
    substrate = gmsh.model.occ.addBox(
        box_x_min * scale, box_y_min * scale, substrate_bot * scale,
        (box_x_max - box_x_min) * scale,
        (box_y_max - box_y_min) * scale,
        substrate_thickness * scale,
    )

    # Create metal volumes by extruding polygons
    metal_volumes = {}  # metal_id → list of volume tags
    for layer in layers:
        metal_id = layer["metal"]
        if metal_id not in metal_defs:
            continue
        m = metal_defs[metal_id]
        z_bot = (m["z_um"] - m["thickness_um"] / 2) * scale
        thickness = m["thickness_um"] * scale

        if metal_id not in metal_volumes:
            metal_volumes[metal_id] = []

        for poly in layer["polygons"]:
            if len(poly) < 3:
                continue
            # Create polygon outline
            pts = []
            for pt in poly:
                p = gmsh.model.occ.addPoint(pt[0] * scale, pt[1] * scale, z_bot, lc * scale)
                pts.append(p)

            lines = []
            for i in range(len(pts)):
                l = gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                lines.append(l)

            loop = gmsh.model.occ.addCurveLoop(lines)
            surf = gmsh.model.occ.addPlaneSurface([loop])

            # Extrude to thickness
            ext = gmsh.model.occ.extrude([(2, surf)], 0, 0, thickness)
            # ext[1] is the extruded volume
            vol_tag = ext[1][1]
            metal_volumes[metal_id].append((3, vol_tag))

    # Fragment everything to get conformal mesh
    all_objects = [(3, box)]
    all_tools = [(3, substrate)]
    for vols in metal_volumes.values():
        all_tools.extend(vols)

    gmsh.model.occ.synchronize()

    # Instead of complex boolean, just fragment all volumes
    all_vols = [(3, box), (3, substrate)]
    for vols in metal_volumes.values():
        all_vols.extend(vols)

    if len(all_vols) > 1:
        try:
            result = gmsh.model.occ.fragment(all_vols[:1], all_vols[1:])
        except Exception as e:
            print(f"WARNING: Boolean fragment failed ({e}), using simple geometry")

    gmsh.model.occ.synchronize()

    # Get all volumes and surfaces
    all_final_vols = gmsh.model.getEntities(3)
    all_final_surfs = gmsh.model.getEntities(2)

    print(f"Geometry: {len(all_final_vols)} volumes, {len(all_final_surfs)} surfaces")

    # Classify volumes by position (metal / substrate / oxide+air)
    volume_assignments = {}  # tag → "metal_id" or "substrate" or "dielectric"

    for dim, tag in all_final_vols:
        bb = gmsh.model.getBoundingBox(dim, tag)
        z_center = (bb[2] + bb[5]) / 2 / scale  # back to um

        # Check if this volume is a metal
        assigned = False
        for metal_id, m in metal_defs.items():
            z_bot_m = m["z_um"] - m["thickness_um"] / 2
            z_top_m = m["z_um"] + m["thickness_um"] / 2
            if z_bot_m - 0.1 <= z_center <= z_top_m + 0.1:
                # Check if it's small enough to be a metal (not the surrounding dielectric)
                vol_x = (bb[3] - bb[0]) / scale
                vol_y = (bb[4] - bb[1]) / scale
                if vol_x < (box_x_max - box_x_min) * 0.9 and vol_y < (box_y_max - box_y_min) * 0.9:
                    volume_assignments[tag] = metal_id
                    assigned = True
                    break
        if not assigned:
            if z_center < substrate_top:
                volume_assignments[tag] = "substrate"
            else:
                volume_assignments[tag] = "dielectric"

    # Assign physical groups
    # Volume groups
    toml_materials = []
    tag_map = {}  # purpose → tag number

    # Group volumes by assignment
    from collections import defaultdict
    vol_groups = defaultdict(list)
    for tag, assignment in volume_assignments.items():
        vol_groups[assignment].append(tag)

    # If no metals were fragmented, just assign all as dielectric
    if not vol_groups:
        for dim, tag in all_final_vols:
            vol_groups["dielectric"].append(tag)

    for assignment, tags in vol_groups.items():
        phys_tag = tag_counter
        tag_counter += 1
        tag_map[assignment] = phys_tag
        gmsh.model.addPhysicalGroup(3, tags, phys_tag, assignment)

        if assignment == "substrate":
            toml_materials.append({
                "volume_tag": phys_tag,
                "er": stack["substrate_eps_r"],
                "ur": 1.0,
                "tand": 0.0,
                "conductivity": 1.0 / (stack["substrate_rho"] * 0.01),  # Ω·cm → S/m
            })
        elif assignment == "dielectric":
            toml_materials.append({
                "volume_tag": phys_tag,
                "er": stack["oxide_eps_r"],
                "ur": 1.0,
                "tand": 0.0,
                "conductivity": 0.0,
            })
        elif assignment in metal_defs:
            m = metal_defs[assignment]
            sigma = 1.0 / (m["rsh"] * m["thickness_um"] * 1e-6)  # S/m
            toml_materials.append({
                "volume_tag": phys_tag,
                "er": 1.0,
                "ur": 1.0,
                "tand": 0.0,
                "conductivity": sigma,
            })

    # Surface groups: outer boundary = PEC or ABC
    outer_surfs = []
    for dim, tag in all_final_surfs:
        bb = gmsh.model.getBoundingBox(dim, tag)
        # Check if surface is on the outer boundary
        on_boundary = (
            abs(bb[0] / scale - box_x_min) < 0.1 or abs(bb[3] / scale - box_x_min) < 0.1 or
            abs(bb[0] / scale - box_x_max) < 0.1 or abs(bb[3] / scale - box_x_max) < 0.1 or
            abs(bb[1] / scale - box_y_min) < 0.1 or abs(bb[4] / scale - box_y_min) < 0.1 or
            abs(bb[1] / scale - box_y_max) < 0.1 or abs(bb[4] / scale - box_y_max) < 0.1 or
            abs(bb[2] / scale - substrate_bot) < 0.1 or abs(bb[5] / scale - substrate_bot) < 0.1 or
            abs(bb[2] / scale - z_top) < 0.1 or abs(bb[5] / scale - z_top) < 0.1
        )
        if on_boundary:
            outer_surfs.append(tag)

    if outer_surfs:
        gmsh.model.addPhysicalGroup(2, outer_surfs, tag_pec, "PEC" if args.boundary == "pec" else "ABC")

    # Port surfaces: small rectangles at port locations
    toml_ports = []
    port_size = lc * 0.5  # port surface size

    for port in ports:
        metal_id = port["metal"]
        if metal_id not in metal_defs:
            continue
        m = metal_defs[metal_id]
        px = port["x_um"] * scale
        py = port["y_um"] * scale
        pz_bot = substrate_top * scale  # port spans from substrate to metal
        pz_top = (m["z_um"] + m["thickness_um"] / 2) * scale
        ps = port_size * scale

        # Create a small rectangular surface for the port
        p1 = gmsh.model.occ.addPoint(px - ps/2, py - ps/2, pz_bot)
        p2 = gmsh.model.occ.addPoint(px + ps/2, py - ps/2, pz_bot)
        p3 = gmsh.model.occ.addPoint(px + ps/2, py + ps/2, pz_bot)
        p4 = gmsh.model.occ.addPoint(px - ps/2, py + ps/2, pz_bot)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.occ.addPlaneSurface([loop])
        ext = gmsh.model.occ.extrude([(2, surf)], 0, 0, pz_top - pz_bot)

        gmsh.model.occ.synchronize()

        port_tag = tag_counter
        tag_counter += 1
        # Find the extruded surface (the side faces)
        port_surfs = [e[1] for e in ext if e[0] == 2]
        if port_surfs:
            gmsh.model.addPhysicalGroup(2, port_surfs[:1], port_tag, port["name"])

        toml_ports.append({
            "type": "lumped",
            "tag": port_tag,
            "z0": sim["z0"],
            "direction": [0, 0, 1],  # E-field along z (vertical)
            "width": port_size * scale,
            "height": (pz_top - pz_bot),
        })

    # Mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc * scale)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * scale * 0.1)

    print("Meshing...")
    gmsh.model.mesh.generate(3)

    # Count elements
    _, _, nt = gmsh.model.mesh.getElements(3)
    n_tets = len(nt[0]) // 4 if nt else 0
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    print(f"Mesh: {n_nodes} nodes, {n_tets} tets")

    # Write mesh
    msh_path = f"{args.output}.msh"
    gmsh.write(msh_path)
    print(f"Wrote {msh_path}")

    gmsh.finalize()

    # Write TOML config
    toml_path = f"{args.output}.toml"
    with open(toml_path, "w") as f:
        f.write(f'[mesh]\nfile = "{msh_path}"\n\n')
        f.write(f'[frequency]\n')
        if sim["n_points"] <= 1:
            f.write(f'values = [{sim["f_min"]}]\n\n')
        else:
            f.write(f'range = [{sim["f_min"]}, {sim["f_max"]}, {sim["n_points"]}]\n\n')

        for mat in toml_materials:
            f.write(f'[[materials]]\n')
            f.write(f'volume_tag = {mat["volume_tag"]}\n')
            f.write(f'er = {mat["er"]}\n')
            f.write(f'ur = {mat["ur"]}\n')
            f.write(f'tand = {mat["tand"]}\n')
            f.write(f'conductivity = {mat["conductivity"]:.6e}\n\n')

        for port in toml_ports:
            f.write(f'[[ports]]\n')
            f.write(f'type = "{port["type"]}"\n')
            f.write(f'tag = {port["tag"]}\n')
            f.write(f'z0 = {port["z0"]}\n')
            f.write(f'direction = [{port["direction"][0]}, {port["direction"][1]}, {port["direction"][2]}]\n')
            f.write(f'width = {port["width"]:.6e}\n')
            f.write(f'height = {port["height"]:.6e}\n\n')

        f.write(f'[pec]\ntags = [{tag_pec}]\n\n')

        f.write(f'[solver]\nprefer = "auto"\n\n')

        f.write(f'[output]\n')
        f.write(f'touchstone = "{args.output}.s{len(ports)}p"\n')
        f.write(f'z0 = {sim["z0"]}\n')
        f.write(f'vtk = "{args.output}_fields.vtk"\n')

    print(f"Wrote {toml_path}")
    print(f"\nRun: rapidfem {toml_path}")


if __name__ == "__main__":
    main()
