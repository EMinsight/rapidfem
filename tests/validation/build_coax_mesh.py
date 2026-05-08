"""
Build a coaxial line mesh for rapidfem CoaxPort validation.

Geometry: outer cylinder (Ro), inner cylinder (Ri, PEC), length L. Air-filled.
Tags: 1 = PEC (inner conductor walls + outer conductor wall), 3 = port 1 (z=0 annulus),
4 = port 2 (z=L annulus).
"""
import os
import gmsh

mm = 1e-3
ri = 0.5 * mm
ro = 1.7 * mm
L = 30 * mm

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meshes")
os.makedirs(out_dir, exist_ok=True)
msh_path = os.path.join(out_dir, "coax.msh")

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("coax")
occ = gmsh.model.occ

# Outer cylinder (filled), then subtract inner cylinder → annular dielectric
outer_vol = occ.addCylinder(0, 0, 0, 0, 0, L, ro)
inner_vol = occ.addCylinder(0, 0, 0, 0, 0, L, ri)
out_dt, _ = occ.cut([(3, outer_vol)], [(3, inner_vol)])
occ.synchronize()

vol_tag = out_dt[0][1]
faces = gmsh.model.getEntities(dim=2)

# Classify faces by AABB and centroid
pec, port1, port2 = [], [], []
tol = 1e-6
for dim, ftag in faces:
    bb = gmsh.model.getBoundingBox(dim, ftag)
    xmin, ymin, zmin, xmax, ymax, zmax = bb
    dz = zmax - zmin
    # Annular endcap at z=0 or z=L: dz ~ 0
    if dz < tol and abs(zmin) < tol:
        port1.append(ftag)
    elif dz < tol and abs(zmin - L) < tol:
        port2.append(ftag)
    else:
        # Side walls (inner + outer cylinder)
        pec.append(ftag)

print(f"PEC walls: {pec}, port1 annulus: {port1}, port2 annulus: {port2}")

gmsh.model.addPhysicalGroup(2, pec, tag=1); gmsh.model.setPhysicalName(2, 1, "PEC")
gmsh.model.addPhysicalGroup(2, port1, tag=3); gmsh.model.setPhysicalName(2, 3, "Port1")
gmsh.model.addPhysicalGroup(2, port2, tag=4); gmsh.model.setPhysicalName(2, 4, "Port2")
gmsh.model.addPhysicalGroup(3, [vol_tag], tag=100); gmsh.model.setPhysicalName(3, 100, "Air")

gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5 * mm)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.08 * mm)
# Refine near inner conductor where 1/ρ field gradient is steep
inner_pts = [t for d, t in gmsh.model.getEntities(0) if abs(gmsh.model.getValue(0, t, [])[0]) < ri + 1e-6
             and abs(gmsh.model.getValue(0, t, [])[1]) < ri + 1e-6]
if inner_pts:
    gmsh.model.mesh.setSize([(0, t) for t in inner_pts], 0.1 * mm)
gmsh.model.mesh.generate(3)

n_nodes, _, _ = gmsh.model.mesh.getNodes()
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
n_tets = sum(len(t) for t in elem_tags)
print(f"Mesh: {len(n_nodes)} nodes, {n_tets} tets")

gmsh.write(msh_path)
gmsh.finalize()
print(f"Wrote {msh_path}")
