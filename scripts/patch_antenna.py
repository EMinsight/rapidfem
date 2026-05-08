"""
Build a rectangular microstrip patch antenna geometry,
mesh with gmsh, and write a RapidFEM config.

Geometry:
  - Ground plane (PEC) at z=0
  - Substrate (εr) from z=0 to z=h
  - Patch (PEC) at z=h, centered on substrate
  - Lumped port: vertical face in y-z plane from ground to patch
  - Air box enclosing everything, ABC on outer walls

Physical tags:
  1 = PEC (ground plane at z=0 + patch at z=h)
  2 = ABC (air box outer walls)
  3 = Lumped port face
  100 = Substrate volume
  101 = Air volume
"""

import gmsh
import sys
import os
import math

# ── Antenna parameters (FR-4 substrate, ~2.4 GHz design) ──
sub_w = 60e-3     # substrate width (x)
sub_l = 60e-3     # substrate length (y)
sub_h = 1.6e-3    # substrate height (z)
er_sub = 4.4

patch_w = 38e-3   # patch width (x)
patch_l = 29e-3   # patch length (y)

# Feed: a thin rectangle in the y-z plane at the patch edge
# The lumped port is a vertical face from ground to patch
feed_x = 0.0                    # centered in x
feed_y = -patch_l / 2           # at the -y edge of patch
feed_width = 1.5e-3             # width of feed strip in x

# Air box
air_pad_xy = 25e-3
air_pad_z_top = 25e-3
air_pad_z_bot = 0.0  # no air below ground (ground = PEC)

# Mesh sizes
lc_feed = 0.4e-3
lc_patch = 2.0e-3
lc_sub = 3.0e-3
lc_air = 8.0e-3

# ── Derived ──
total_w = sub_w + 2 * air_pad_xy
total_l = sub_l + 2 * air_pad_xy
total_h = sub_h + air_pad_z_top

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "meshes")
os.makedirs(out_dir, exist_ok=True)
msh_path = os.path.join(out_dir, "patch_antenna_cq.msh")
toml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "config_patch_cq.toml")

# ── Build with gmsh OCC ──
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("patch_antenna")
occ = gmsh.model.occ

# 1. Air box: from z=0 to z=total_h
air = occ.addBox(-total_w/2, -total_l/2, 0, total_w, total_l, total_h)

# 2. Substrate: from z=0 to z=sub_h
sub = occ.addBox(-sub_w/2, -sub_l/2, 0, sub_w, sub_l, sub_h)

# 3. Patch surface: rectangle at z=sub_h
patch_rect = occ.addRectangle(-patch_w/2, -patch_l/2, sub_h, patch_w, patch_l)

# 4. Feed port: rectangle in y-z plane at feed_y
# This is a vertical rectangle from z=0 to z=sub_h, width=feed_width in x
feed_rect = occ.addRectangle(
    feed_x - feed_width/2, 0, 0,       # corner in local 2D (maps to x, z)
    feed_width, sub_h                    # dx, dz
)
# Rotate from XY to XZ plane (y-normal face at y=feed_y)
occ.rotate([(2, feed_rect)], 0, 0, 0, 1, 0, 0, math.pi/2)
occ.translate([(2, feed_rect)], 0, feed_y, 0)

# 5. Fragment everything together
print("Boolean fragments...")
all_objs = [(3, air)]
all_tools = [(3, sub), (2, patch_rect), (2, feed_rect)]
out_map, out_map_parent = occ.fragment(all_objs, all_tools)
occ.synchronize()

# ── Identify entities ──
volumes = gmsh.model.getEntities(dim=3)
surfaces = gmsh.model.getEntities(dim=2)
print(f"{len(volumes)} volumes, {len(surfaces)} surfaces")

# Classify volumes
substrate_vols = []
air_vols = []

for dim, tag in volumes:
    bb = gmsh.model.getBoundingBox(dim, tag)
    xmin, ymin, zmin, xmax, ymax, zmax = bb
    dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin

    # Substrate: z goes from ~0 to ~sub_h, fits within substrate bounds
    if abs(zmin) < 1e-5 and abs(zmax - sub_h) < 1e-5 and dx <= sub_w + 1e-5:
        substrate_vols.append(tag)
        print(f"  Vol {tag}: substrate ({dx*1e3:.1f} x {dy*1e3:.1f} x {dz*1e3:.1f} mm)")
    else:
        air_vols.append(tag)
        print(f"  Vol {tag}: air ({dx*1e3:.1f} x {dy*1e3:.1f} x {dz*1e3:.1f} mm)")

# Classify surfaces
ground_faces = []
patch_faces = []
abc_faces = []
feed_faces = []

tol = 1e-5

for dim, tag in surfaces:
    bb = gmsh.model.getBoundingBox(dim, tag)
    xmin, ymin, zmin, xmax, ymax, zmax = bb
    dx, dy, dz = xmax-xmin, ymax-ymin, zmax-zmin

    # Feed port: y-normal face, small in x, spans z=0 to z=sub_h, at y=feed_y
    if (abs(dy) < tol and abs(ymin - feed_y) < tol and
        dx < 2*feed_width and abs(zmin) < tol and abs(zmax - sub_h) < tol):
        feed_faces.append(tag)
        print(f"  Surf {tag}: FEED PORT ({dx*1e3:.3f} x {dz*1e3:.3f} mm at y={ymin*1e3:.2f}mm)")
        continue

    # ABC: faces on the outer boundary of the air box
    on_outer = False
    # Top face: z = total_h
    if abs(dz) < tol and abs(zmin - total_h) < tol:
        on_outer = True
    # Bottom face: z = 0 — this is the GROUND plane, not ABC
    # Side faces: x = ±total_w/2
    if abs(dx) < tol and (abs(xmin + total_w/2) < tol or abs(xmin - total_w/2) < tol):
        on_outer = True
    # Side faces: y = ±total_l/2
    if abs(dy) < tol and (abs(ymin + total_l/2) < tol or abs(ymin - total_l/2) < tol):
        on_outer = True

    if on_outer:
        abc_faces.append(tag)
        continue

    # Ground plane: z=0, horizontal (not on the outer boundary)
    if abs(dz) < tol and abs(zmin) < tol and dx < total_w - 1e-3:
        ground_faces.append(tag)
        continue

    # Patch: z=sub_h face that matches patch dimensions
    if (abs(dz) < tol and abs(zmin - sub_h) < tol and
        abs(dx - patch_w) < 1e-5 and abs(dy - patch_l) < 1e-5):
        patch_faces.append(tag)
        print(f"  Surf {tag}: PATCH ({dx*1e3:.1f} x {dy*1e3:.1f} mm)")
        continue

# Physical groups
pec_all = ground_faces + patch_faces
if pec_all:
    gmsh.model.addPhysicalGroup(2, pec_all, tag=1)
    gmsh.model.setPhysicalName(2, 1, "PEC")
    print(f"PEC: {len(pec_all)} faces (ground={len(ground_faces)}, patch={len(patch_faces)})")

if abc_faces:
    gmsh.model.addPhysicalGroup(2, abc_faces, tag=2)
    gmsh.model.setPhysicalName(2, 2, "ABC")
    print(f"ABC: {len(abc_faces)} faces")

if feed_faces:
    gmsh.model.addPhysicalGroup(2, feed_faces, tag=3)
    gmsh.model.setPhysicalName(2, 3, "LumpedPort")
    print(f"Lumped port: {len(feed_faces)} faces")
else:
    print("WARNING: No feed port faces found!")
    # Debug: print all surfaces
    for dim, tag in surfaces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        xmin, ymin, zmin, xmax, ymax, zmax = bb
        print(f"  DEBUG surf {tag}: bb=({xmin*1e3:.2f},{ymin*1e3:.2f},{zmin*1e3:.2f})-({xmax*1e3:.2f},{ymax*1e3:.2f},{zmax*1e3:.2f})")

if substrate_vols:
    gmsh.model.addPhysicalGroup(3, substrate_vols, tag=100)
    gmsh.model.setPhysicalName(3, 100, "Substrate")
    print(f"Substrate: {len(substrate_vols)} volumes")

if air_vols:
    gmsh.model.addPhysicalGroup(3, air_vols, tag=101)
    gmsh.model.setPhysicalName(3, 101, "Air")
    print(f"Air: {len(air_vols)} volumes")

# ── Mesh sizing ──
# Default: coarse
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc_air)

# Substrate points: medium
for vol_tag in substrate_vols:
    bnd = gmsh.model.getBoundary([(3, vol_tag)], recursive=True, oriented=False)
    pts = list(set(t for d, t in bnd if d == 0))
    if pts:
        gmsh.model.mesh.setSize([(0, p) for p in pts], lc_sub)

# Patch points: fine
for face_tag in patch_faces:
    bnd = gmsh.model.getBoundary([(2, face_tag)], recursive=True, oriented=False)
    pts = list(set(t for d, t in bnd if d == 0))
    if pts:
        gmsh.model.mesh.setSize([(0, p) for p in pts], lc_patch)

# Feed points: very fine
for face_tag in feed_faces:
    bnd = gmsh.model.getBoundary([(2, face_tag)], recursive=True, oriented=False)
    pts = list(set(t for d, t in bnd if d == 0))
    if pts:
        gmsh.model.mesh.setSize([(0, p) for p in pts], lc_feed)

# Generate
print("\nGenerating 3D mesh...")
gmsh.model.mesh.generate(3)

node_tags, _, _ = gmsh.model.mesh.getNodes()
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
n_tets = sum(len(t) for t in elem_tags)
print(f"Mesh: {len(node_tags)} nodes, {n_tets} tets")

gmsh.write(msh_path)
print(f"Wrote: {msh_path}")

# Clean up temp files
for f in ["patch_sub.step", "patch_air.step"]:
    p = os.path.join(out_dir, f)
    if os.path.exists(p):
        os.remove(p)

gmsh.finalize()

# ── Write config ──
config = f"""[mesh]
file = "tests/meshes/patch_antenna_cq.msh"

[frequency]
range = [1.5e9, 3.5e9, 21]

[[ports]]
type = "lumped"
tag = 3
z0 = 50.0
direction = [0.0, 0.0, 1.0]

[[ports]]
type = "abc"
tag = 2
order = 1

[pec]
tags = [1]

[[materials]]
volume_tag = 100
er = {er_sub}

[[materials]]
volume_tag = 101
er = 1.0

[output]
touchstone = "tests/patch_antenna_cq.s1p"
vtk = "tests/patch_antenna_cq.vtk"
"""

with open(toml_path, "w") as f:
    f.write(config)
print(f"Wrote: {toml_path}")
print(f"\nRun: cargo run --release -- tests/config_patch_cq.toml")
