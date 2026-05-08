"""Debug substrate/oxide tag drift after fragment."""
import gdstk
import gmsh
import tempfile
import os

import rapidfem
import rapidfem.rfic as rfic


def make_test_gds(path):
    lib = gdstk.Library(name="test_lib", unit=1e-6, precision=1e-9)
    cell = lib.new_cell("microstrip_test")
    cell.add(gdstk.rectangle((-100, -2.5), (100, 2.5), layer=72, datatype=20))
    cell.add(gdstk.rectangle((-110, -50), (-90, -10), layer=72, datatype=20))
    cell.add(gdstk.rectangle((90, -50), (110, -10), layer=72, datatype=20))
    lib.write_gds(path)


def dump_state(g, label):
    print(f"\n=== {label} ===")
    print(f"  g._entities count: {len(g._entities)}")
    for e in g._entities:
        if e.dim == 3:
            print(f"    dim=3 tag={e.tag} name={e.name!r} mat={e.material!r}")
            print(f"      cog={tuple(round(c*1e6, 2) for c in e.cog)} um")
            print(f"      bbox xyz min=({e.bbox[0]*1e6:.1f},{e.bbox[1]*1e6:.1f},{e.bbox[2]*1e6:.1f}) "
                  f"max=({e.bbox[3]*1e6:.1f},{e.bbox[4]*1e6:.1f},{e.bbox[5]*1e6:.1f}) um")
    print(f"  gmsh.model.getEntities(3): {gmsh.model.getEntities(3)}")
    for d, t in gmsh.model.getEntities(3):
        cog = tuple(round(c*1e6, 2) for c in gmsh.model.occ.getCenterOfMass(d, t))
        bbox = gmsh.model.getBoundingBox(d, t)
        print(f"    gmsh dim=3 tag={t} cog={cog} um  "
              f"bbox=({bbox[0]*1e6:.1f},{bbox[1]*1e6:.1f},{bbox[2]*1e6:.1f}) "
              f"-> ({bbox[3]*1e6:.1f},{bbox[4]*1e6:.1f},{bbox[5]*1e6:.1f}) um")


with tempfile.NamedTemporaryFile(suffix=".gds", delete=False) as f:
    gds_path = f.name
make_test_gds(gds_path)

um = 1e-6
stack = rfic.Stack.sky130()

g = rapidfem.Geometry.from_gds(gds_path, stack=stack, merge=True)
dump_state(g, "After from_gds (3 met5 polygons extruded)")

foot = (260*um, 120*um)
stack.create_substrate(g, footprint=foot, center=True)
dump_state(g, "After create_substrate (substrate + oxide + fragment)")

# Now write the mesh and check what physical groups gmsh registers
mesh_bytes, name_to_tag = g.mesh(maxh=20*um)
print(f"\n=== After g.mesh() ===")
print(f"name_to_tag: {name_to_tag}")
print(f"gmsh physical groups (3D):")
for d, t in gmsh.model.getPhysicalGroups(dim=3):
    name = gmsh.model.getPhysicalName(d, t)
    ents = gmsh.model.getEntitiesForPhysicalGroup(d, t)
    print(f"  dim={d} tag={t} name={name!r} entities={list(ents)}")
print(f"gmsh physical groups (2D):")
for d, t in gmsh.model.getPhysicalGroups(dim=2):
    name = gmsh.model.getPhysicalName(d, t)
    ents = gmsh.model.getEntitiesForPhysicalGroup(d, t)
    print(f"  dim={d} tag={t} name={name!r} entities={list(ents)}")

print()
print(f"=== Tets per gmsh entity (dim=3) ===")
for d, t in gmsh.model.getEntities(3):
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3, tag=t)
    n = sum(len(e) for e in elem_tags)
    name_phys = []
    for t_pg in gmsh.model.getPhysicalGroupsForEntity(d, t):
        name_phys.append((int(t_pg), gmsh.model.getPhysicalName(d, int(t_pg))))
    print(f"  entity dim=3 tag={t}: {n} tets, in physical groups: {name_phys}")

print()
print(f"=== Writing mesh.msh and checking ===")
with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
    msh_path = f.name
gmsh.write(msh_path)
print(f"  Mesh written to {msh_path} ({os.path.getsize(msh_path)} bytes)")

g.close()
os.unlink(gds_path)
os.unlink(msh_path)
