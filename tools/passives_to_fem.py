#!/usr/bin/env python3
"""
RapidPassives → RapidFEM bridge.

Reads the FEM JSON export from RapidPassives and produces:
  - A 3D tetrahedral mesh (.msh) via gmsh
  - A matching RapidFEM config (.toml)

Usage:
  python passives_to_fem.py layout.json --output inductor
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
    parser.add_argument("--output", "-o", default="passive", help="Output prefix")
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

    scale = 1e-6  # um → m

    # Bounding box of all metal polygons
    all_x, all_y = [], []
    for layer in layers:
        for poly in layer["polygons"]:
            for pt in poly:
                all_x.append(pt[0])
                all_y.append(pt[1])

    if not all_x:
        print("ERROR: No metal polygons found"); sys.exit(1)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    extent = max(x_max - x_min, y_max - y_min)
    margin = extent * 0.3

    bx0 = (x_min - margin) * scale
    bx1 = (x_max + margin) * scale
    by0 = (y_min - margin) * scale
    by1 = (y_max + margin) * scale

    metal_defs = {m["name"]: m for m in stack["metals"]}
    z_metals = [(m["z_um"] - m["thickness_um"]/2, m["z_um"] + m["thickness_um"]/2) for m in stack["metals"]]
    z_metal_min = min(z[0] for z in z_metals)
    z_metal_max = max(z[1] for z in z_metals)

    sub_top = (z_metal_min - 1) * scale  # 1um gap
    sub_thickness = 50  # um
    sub_bot = (z_metal_min - 1 - sub_thickness) * scale
    air_h = (args.air_height if args.air_height > 0 else extent * 0.5) * scale
    z_top = z_metal_max * scale + air_h

    lc = (args.lc if args.lc > 0 else extent / 15) * scale
    lc_metal = lc * 0.3  # finer mesh in metals

    print(f"Layout: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}] um")
    print(f"Metals: z=[{z_metal_min:.2f}, {z_metal_max:.2f}] um")
    print(f"Element size: {lc*1e6:.2f} um (metal: {lc_metal*1e6:.2f} um)")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("passive")

    # Step 1: Create all volumes
    # Enclosing box (full domain)
    box_tag = gmsh.model.occ.addBox(bx0, by0, sub_bot, bx1-bx0, by1-by0, z_top-sub_bot)

    # Substrate slab
    sub_tag = gmsh.model.occ.addBox(bx0, by0, sub_bot, bx1-bx0, by1-by0, sub_top-sub_bot)

    # Metal volumes (extruded polygons)
    metal_vol_tags = {}  # metal_id → list of (3, tag)
    for layer in layers:
        mid = layer["metal"]
        if mid not in metal_defs: continue
        m = metal_defs[mid]
        z_bot = (m["z_um"] - m["thickness_um"]/2) * scale
        t = m["thickness_um"] * scale

        if mid not in metal_vol_tags:
            metal_vol_tags[mid] = []

        for poly in layer["polygons"]:
            if len(poly) < 3: continue
            pts = [gmsh.model.occ.addPoint(p[0]*scale, p[1]*scale, z_bot, lc_metal) for p in poly]
            lines = [gmsh.model.occ.addLine(pts[i], pts[(i+1)%len(pts)]) for i in range(len(pts))]
            loop = gmsh.model.occ.addCurveLoop(lines)
            surf = gmsh.model.occ.addPlaneSurface([loop])
            ext = gmsh.model.occ.extrude([(2, surf)], 0, 0, t)
            vol = [e for e in ext if e[0] == 3][0]
            metal_vol_tags[mid].append(vol)

    # Step 2: Fragment ALL volumes for conformal mesh (ports handled post-mesh)
    all_vols = [(3, box_tag), (3, sub_tag)]
    for vols in metal_vol_tags.values():
        all_vols.extend(vols)

    # Track which original volume each input maps to
    # all_vols[0] = box, [1] = substrate, [2:] = metals per layer, then ports
    original_labels = ["box", "substrate"]
    for layer in layers:
        mid = layer["metal"]
        if mid not in metal_defs: continue
        for _ in layer["polygons"]:
            if len(_) >= 3:
                original_labels.append(mid)

    print(f"Fragmenting {len(all_vols)} volumes ({len(original_labels)} labels)...")
    result, result_map = gmsh.model.occ.fragment([all_vols[0]], all_vols[1:])
    gmsh.model.occ.synchronize()

    # result_map[i] = list of (dim, tag) that original all_vols[i] was split into
    # Use this to classify fragments by their ORIGINAL identity
    final_vols = gmsh.model.getEntities(3)
    print(f"After fragment: {len(final_vols)} volumes")

    vol_classification = {}  # tag → category string

    # First pass: mark volumes that came from metal/substrate originals
    for orig_idx, fragments in enumerate(result_map):
        label = original_labels[orig_idx] if orig_idx < len(original_labels) else "box"
        for dim, tag in fragments:
            if dim != 3: continue
            if label not in ("box",):
                vol_classification[tag] = label

    # Second pass: anything not yet classified is dielectric (leftover from box)
    for dim, tag in final_vols:
        if tag not in vol_classification:
            vol_classification[tag] = "dielectric"

    # Step 4: Assign physical groups
    tag_pec = 1
    tag_counter = 10
    from collections import defaultdict
    groups = defaultdict(list)
    for tag, cat in vol_classification.items():
        groups[cat].append(tag)

    toml_materials = []

    for cat, tags in groups.items():
        phys_tag = tag_counter; tag_counter += 1
        gmsh.model.addPhysicalGroup(3, tags, phys_tag, cat)
        print(f"  {cat}: {len(tags)} volumes, tag={phys_tag}")

        if cat == "substrate":
            toml_materials.append(dict(volume_tag=phys_tag, er=stack["substrate_eps_r"], ur=1.0,
                                       tand=0.0, conductivity=1.0/(stack["substrate_rho"]*0.01)))
        elif cat == "dielectric":
            toml_materials.append(dict(volume_tag=phys_tag, er=stack["oxide_eps_r"], ur=1.0,
                                       tand=0.0, conductivity=0.0))
        elif cat in metal_defs:
            m = metal_defs[cat]
            sigma = 1.0 / (m["rsh"] * m["thickness_um"] * 1e-6)
            toml_materials.append(dict(volume_tag=phys_tag, er=1.0, ur=1.0,
                                       tand=0.0, conductivity=sigma))

    # Outer boundary surfaces
    outer_surfs = []
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        eps = lc * 0.1
        on_bnd = (abs(bb[0]-bx0)<eps or abs(bb[3]-bx0)<eps or abs(bb[0]-bx1)<eps or abs(bb[3]-bx1)<eps or
                  abs(bb[1]-by0)<eps or abs(bb[4]-by0)<eps or abs(bb[1]-by1)<eps or abs(bb[4]-by1)<eps or
                  abs(bb[2]-sub_bot)<eps or abs(bb[5]-sub_bot)<eps or abs(bb[2]-z_top)<eps or abs(bb[5]-z_top)<eps)
        if on_bnd:
            outer_surfs.append(tag)

    if outer_surfs:
        gmsh.model.addPhysicalGroup(2, outer_surfs, tag_pec, "boundary")

    # Port: single lumped port between the two terminals (P1 and P2)
    # Find the end-face surfaces of each terminal's metal near the port coordinates
    toml_ports = []
    if len(ports) >= 2:
        p1 = ports[0]
        p2 = ports[1]

        # Collect end-face surfaces near both port locations
        port_surfs = []
        outer_set = set(outer_surfs)
        for port_def in [p1, p2]:
            mid = port_def["metal"]
            px = port_def["x_um"] * scale
            py = port_def["y_um"] * scale
            search_r = lc * 1.5

            for cat, tags in groups.items():
                if cat != mid: continue
                for vol_tag in tags:
                    bnd = gmsh.model.getBoundary([(3, vol_tag)], oriented=False)
                    for dim, stag in bnd:
                        if dim != 2 or stag in outer_set: continue
                        try:
                            cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, stag)
                        except:
                            bb = gmsh.model.getBoundingBox(dim, stag)
                            cx, cy, cz = (bb[0]+bb[3])/2, (bb[1]+bb[4])/2, (bb[2]+bb[5])/2
                        dist = ((cx - px)**2 + (cy - py)**2)**0.5
                        if dist < search_r:
                            port_surfs.append(stag)

        if port_surfs:
            port_tag = tag_counter; tag_counter += 1
            gmsh.model.addPhysicalGroup(2, port_surfs, port_tag, "Port")
            print(f"  Port: {len(port_surfs)} surfaces between {p1['name']} and {p2['name']}, tag={port_tag}")

            # Direction: from P2 to P1
            dx = p1["x_um"] - p2["x_um"]
            dy = p1["y_um"] - p2["y_um"]
            dn = (dx**2 + dy**2)**0.5
            if dn > 0:
                direction = [dx/dn, dy/dn, 0]
            else:
                direction = [0, 0, 1]

            toml_ports.append(dict(type="lumped", tag=port_tag, z0=sim["z0"],
                                   direction=direction, power=1.0))
        else:
            print(f"  WARNING: no port surfaces found")

    # Step 5: Mesh with refinement near metals
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_metal * 0.5)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay

    print("Meshing...")
    gmsh.model.mesh.generate(3)

    _, _, nt = gmsh.model.mesh.getElements(3)
    n_tets = sum(len(t)//4 for t in nt) if nt else 0
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    print(f"Mesh: {n_nodes} nodes, {n_tets} tets")

    msh_path = f"{args.output}.msh"
    gmsh.write(msh_path)
    print(f"Wrote {msh_path}")
    gmsh.finalize()

    # Step 6: Write TOML config
    toml_path = f"{args.output}.toml"
    with open(toml_path, "w") as f:
        f.write(f'[mesh]\nfile = "{msh_path}"\n\n')

        if sim["n_points"] <= 1:
            f.write(f'[frequency]\nvalues = [{sim["f_min"]}]\n\n')
        else:
            f.write(f'[frequency]\nrange = [{sim["f_min"]}, {sim["f_max"]}, {sim["n_points"]}]\n\n')

        for mat in toml_materials:
            f.write(f'[[materials]]\nvolume_tag = {mat["volume_tag"]}\n')
            f.write(f'er = {mat["er"]}\nur = {mat["ur"]}\ntand = {mat["tand"]}\n')
            f.write(f'conductivity = {mat["conductivity"]:.6e}\n\n')

        for p in toml_ports:
            f.write(f'[[ports]]\ntype = "{p["type"]}"\ntag = {p["tag"]}\n')
            f.write(f'z0 = {p["z0"]}\ndirection = {p["direction"]}\n\n')

        f.write(f'[pec]\ntags = [{tag_pec}]\n\n')
        f.write(f'[solver]\nprefer = "auto"\n\n')
        f.write(f'[output]\ntouchstone = "{args.output}.s{len(ports)}p"\nz0 = {sim["z0"]}\n')
        f.write(f'vtk = "{args.output}_fields.vtk"\n')

    print(f"Wrote {toml_path}")
    print(f"\nRun: rapidfem {toml_path}")


if __name__ == "__main__":
    main()
