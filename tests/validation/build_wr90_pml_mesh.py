"""
WR-90 waveguide with a PML layer at the +z end.
Tags: 1=PEC walls, 3=port1 (z=0), 100=air volume, 200=PML volume.
The PML inner face is at z=L_air; the PML extends from z=L_air to z=L_air+L_pml.
The far end (z=L_total) is also PEC (closes the box; PML absorbs before it reflects).
"""
import os
import gmsh

mm = 1e-3
a = 22.86 * mm
b = 10.16 * mm
L_air = 25.0 * mm   # main air section
L_pml = 15.0 * mm   # PML thickness
L_total = L_air + L_pml

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meshes")
os.makedirs(out_dir, exist_ok=True)
msh_path = os.path.join(out_dir, "wr90_pml.msh")

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("wr90_pml")
occ = gmsh.model.occ

air = occ.addBox(0, 0, 0, a, b, L_air)
pml = occ.addBox(0, 0, L_air, a, b, L_pml)
occ.synchronize()
occ.fragment([(3, air)], [(3, pml)])
occ.synchronize()

vols = gmsh.model.getEntities(dim=3)
faces = gmsh.model.getEntities(dim=2)
tol = 1e-6

# Classify volumes by their z extent (centroid)
air_vol = None
pml_vol = None
for d, t in vols:
    bb = gmsh.model.getBoundingBox(d, t)
    cz = 0.5 * (bb[2] + bb[5])
    if cz < L_air:
        air_vol = t
    else:
        pml_vol = t

# Classify faces
pec, port1, end_face = [], [], []
for d, t in faces:
    bb = gmsh.model.getBoundingBox(d, t)
    xmin, ymin, zmin, xmax, ymax, zmax = bb
    dz = zmax - zmin
    if dz < tol and abs(zmin) < tol:
        port1.append(t)
    elif dz < tol and abs(zmin - L_total) < tol:
        end_face.append(t)
    elif dz < tol and abs(zmin - L_air) < tol:
        # Internal interface between air and PML — leave as natural BC
        pass
    else:
        pec.append(t)

print(f"air_vol={air_vol}, pml_vol={pml_vol}, port1={port1}, end_face={end_face}, PEC walls={pec}")

# End face also PEC (closes the box, but PML absorbs the wave)
gmsh.model.addPhysicalGroup(2, pec + end_face, tag=1); gmsh.model.setPhysicalName(2, 1, "PEC")
gmsh.model.addPhysicalGroup(2, port1, tag=3); gmsh.model.setPhysicalName(2, 3, "Port1")
gmsh.model.addPhysicalGroup(3, [air_vol], tag=100); gmsh.model.setPhysicalName(3, 100, "Air")
gmsh.model.addPhysicalGroup(3, [pml_vol], tag=200); gmsh.model.setPhysicalName(3, 200, "PML")

gmsh.option.setNumber("Mesh.MeshSizeMax", 3 * mm)
gmsh.option.setNumber("Mesh.MeshSizeMin", 1 * mm)
gmsh.model.mesh.generate(3)

n_nodes, _, _ = gmsh.model.mesh.getNodes()
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
n_tets = sum(len(t) for t in elem_tags)
print(f"Mesh: {len(n_nodes)} nodes, {n_tets} tets")

gmsh.write(msh_path)
gmsh.finalize()
print(f"Wrote {msh_path}")
