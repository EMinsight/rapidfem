"""
Build a parallel-plate box mesh for rapidfem validation.
Geometry: 20mm × 20mm × 50mm. z=0 and z=L are ports (tags 3, 4). y=±H/2 are PEC (tag 1).
x=±W/2 are PMC (tag 2).
"""
import os
import gmsh

mm = 1e-3
W = 20 * mm
H = 20 * mm
L = 50 * mm

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meshes")
os.makedirs(out_dir, exist_ok=True)
msh_path = os.path.join(out_dir, "parallel_plate.msh")

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("parallel_plate")
occ = gmsh.model.occ

box = occ.addBox(-W/2, -H/2, 0, W, H, L)
occ.synchronize()

# Identify faces by their normal/position
faces = gmsh.model.getEntities(dim=2)
pec, pmc, port1, port2 = [], [], [], []
tol = 1e-6
for dim, tag in faces:
    bb = gmsh.model.getBoundingBox(dim, tag)
    xmin, ymin, zmin, xmax, ymax, zmax = bb
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin

    # x=±W/2 face: PMC (small dx, x at ±W/2)
    if dx < tol and (abs(xmin + W/2) < tol or abs(xmin - W/2) < tol):
        pmc.append(tag)
    # y=±H/2 face: PEC
    elif dy < tol and (abs(ymin + H/2) < tol or abs(ymin - H/2) < tol):
        pec.append(tag)
    # z=0 face: port 1
    elif dz < tol and abs(zmin) < tol:
        port1.append(tag)
    # z=L face: port 2
    elif dz < tol and abs(zmin - L) < tol:
        port2.append(tag)

print(f"PEC: {pec}, PMC: {pmc}, port1: {port1}, port2: {port2}")

gmsh.model.addPhysicalGroup(2, pec, tag=1)
gmsh.model.setPhysicalName(2, 1, "PEC")
gmsh.model.addPhysicalGroup(2, pmc, tag=2)
gmsh.model.setPhysicalName(2, 2, "PMC")
gmsh.model.addPhysicalGroup(2, port1, tag=3)
gmsh.model.setPhysicalName(2, 3, "Port1")
gmsh.model.addPhysicalGroup(2, port2, tag=4)
gmsh.model.setPhysicalName(2, 4, "Port2")
gmsh.model.addPhysicalGroup(3, [1], tag=100)
gmsh.model.setPhysicalName(3, 100, "Air")

gmsh.option.setNumber("Mesh.MeshSizeMax", 4 * mm)
gmsh.option.setNumber("Mesh.MeshSizeMin", 2 * mm)
gmsh.model.mesh.generate(3)

# Stats
n_nodes, _, _ = gmsh.model.mesh.getNodes()
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
n_tets = sum(len(t) for t in elem_tags)
print(f"Mesh: {len(n_nodes)} nodes, {n_tets} tets")

gmsh.write(msh_path)
gmsh.finalize()
print(f"Wrote {msh_path}")
