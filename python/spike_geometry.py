"""
Spike: prove COG-based name re-resolution survives gmsh OCC boolean ops.

Three test cases exercise the failure modes:
  1. fragment() of two abutting boxes — face stays put, tag may renumber
  2. cut() of a box minus a smaller box — interior faces appear, exterior faces survive
  3. fuse() of two boxes — shared interior face vanishes, others survive

For each: tag a face on the input shape, compute its COG, perform the boolean op,
synchronize, then look up the new face whose COG matches the stored value.
Print whether re-resolution succeeded with what tolerance.
"""
from __future__ import annotations
import math

import gmsh

TOL = 1e-9  # absolute COG tolerance (geometry is ~1m scale here, well within)


def cog(dim: int, tag: int) -> tuple[float, float, float]:
    """Center of mass for a (dim, tag) entity, via gmsh OCC."""
    return tuple(gmsh.model.occ.getCenterOfMass(dim, tag))


def find_by_cog(dim: int, target: tuple[float, float, float], tol: float) -> int | None:
    """After synchronize: find an entity in `dim` whose COG matches target within tol."""
    for d, t in gmsh.model.getEntities(dim):
        if d != dim:
            continue
        c = cog(dim, t)
        if math.dist(c, target) < tol:
            return t
    return None


def label_faces(box_tag: int, label: dict[int, str]) -> dict[str, tuple[float, float, float]]:
    """Snapshot COGs of named faces of a 3D entity. Returns name -> COG."""
    boundary = gmsh.model.getBoundary([(3, box_tag)], oriented=False)
    out = {}
    for i, (d, t) in enumerate(boundary):
        if d == 2 and i in label:
            out[label[i]] = cog(2, t)
    return out


def case_fragment_two_boxes() -> str:
    """Two abutting boxes; mark each box's outer faces; fragment; verify all survive."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("frag")
    occ = gmsh.model.occ

    a = occ.addBox(0, 0, 0, 1, 1, 1)
    b = occ.addBox(1, 0, 0, 1, 1, 1)  # abuts a on +x face of a / -x face of b
    occ.synchronize()

    # snapshot 6 outer-face COGs of each box (some are shared)
    bnd_a = gmsh.model.getBoundary([(3, a)], oriented=False)
    bnd_b = gmsh.model.getBoundary([(3, b)], oriented=False)
    cogs_a = {f"a_face_{i}": cog(d, t) for i, (d, t) in enumerate(bnd_a) if d == 2}
    cogs_b = {f"b_face_{i}": cog(d, t) for i, (d, t) in enumerate(bnd_b) if d == 2}

    occ.fragment([(3, a)], [(3, b)])
    occ.synchronize()

    # Now look up each named COG in the post-fragment topology
    misses_a = [n for n, c in cogs_a.items() if find_by_cog(2, c, TOL) is None]
    misses_b = [n for n, c in cogs_b.items() if find_by_cog(2, c, TOL) is None]
    gmsh.finalize()
    if misses_a or misses_b:
        return f"FAIL: missing {misses_a + misses_b}"
    return f"OK: {len(cogs_a) + len(cogs_b)} faces resolved across fragment()"


def case_cut_box_minus_box() -> str:
    """Box minus a smaller interior box. Outer faces of the big box must survive."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cut")
    occ = gmsh.model.occ

    big = occ.addBox(0, 0, 0, 2, 2, 2)
    occ.synchronize()
    big_face_cogs = {f"big_{i}": cog(d, t)
                     for i, (d, t) in enumerate(gmsh.model.getBoundary([(3, big)], oriented=False))
                     if d == 2}

    small = occ.addBox(0.5, 0.5, 0.5, 1, 1, 1)
    occ.cut([(3, big)], [(3, small)])
    occ.synchronize()

    misses = [n for n, c in big_face_cogs.items() if find_by_cog(2, c, TOL) is None]
    gmsh.finalize()
    if misses:
        return f"FAIL: missing {misses}"
    return f"OK: {len(big_face_cogs)} outer faces resolved across cut()"


def case_fuse_two_boxes() -> str:
    """Fuse two abutting boxes. Outer faces survive; the shared interior face vanishes (expected)."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("fuse")
    occ = gmsh.model.occ

    a = occ.addBox(0, 0, 0, 1, 1, 1)
    b = occ.addBox(1, 0, 0, 1, 1, 1)
    occ.synchronize()
    cogs_a = {f"a_{i}": cog(d, t) for i, (d, t) in enumerate(gmsh.model.getBoundary([(3, a)], oriented=False)) if d == 2}
    shared_x = (1.0, 0.5, 0.5)  # COG of the shared face at x=1, y∈[0,1], z∈[0,1]

    occ.fuse([(3, a)], [(3, b)])
    occ.synchronize()

    # Outer faces of a that are NOT the shared face must survive
    is_shared = lambda c: math.dist(c, shared_x) < TOL
    survived = {n: find_by_cog(2, c, TOL) for n, c in cogs_a.items() if not is_shared(c)}
    shared_lost = all(find_by_cog(2, c, TOL) is None for c in cogs_a.values() if is_shared(c))

    gmsh.finalize()
    n_outer = sum(1 for n in survived)
    if any(t is None for t in survived.values()):
        return f"FAIL: outer face missing"
    if not shared_lost:
        return f"NOTE: shared face survived fuse() — interior face lookup needs different policy"
    return f"OK: {n_outer} outer faces resolved across fuse(); shared interior face correctly absent"


def case_renumber_check() -> str:
    """Confirm that fragment() actually renumbers tags. Otherwise the whole exercise is moot."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("renum")
    occ = gmsh.model.occ

    a = occ.addBox(0, 0, 0, 1, 1, 1)
    b = occ.addBox(1, 0, 0, 1, 1, 1)
    occ.synchronize()
    pre_a_face_tags = sorted([t for d, t in gmsh.model.getBoundary([(3, a)], oriented=False) if d == 2])

    occ.fragment([(3, a)], [(3, b)])
    occ.synchronize()
    post_face_tags = sorted([t for d, t in gmsh.model.getEntities(2)])

    gmsh.finalize()
    overlap = set(pre_a_face_tags) & set(post_face_tags)
    return f"pre  faces: {pre_a_face_tags}\n  post faces: {post_face_tags}\n  overlap: {sorted(overlap)} ({'tags stable' if len(overlap) == len(pre_a_face_tags) else 'tags renumbered — re-resolve required'})"


def bbox_diag(dim: int, tag: int) -> tuple:
    """Bounding box of an entity. Returns (xmin, ymin, zmin, xmax, ymax, zmax)."""
    return tuple(gmsh.model.getBoundingBox(dim, tag))


def find_by_cog_bbox(
    dim: int,
    target_cog: tuple[float, float, float],
    target_bbox: tuple,
    tol: float,
) -> int | None:
    """Disambiguate co-located entities by also matching bounding box.
    For two faces with same COG (e.g. annulus + plate), bbox extents differ.
    """
    for d, t in gmsh.model.getEntities(dim):
        if d != dim:
            continue
        c = cog(dim, t)
        if math.dist(c, target_cog) >= tol:
            continue
        b = bbox_diag(dim, t)
        if all(abs(b[i] - target_bbox[i]) < tol for i in range(6)):
            return t
    return None


def case_plate_on_box_disambiguated() -> str:
    """Same as case_plate_on_box but using COG+bbox to disambiguate co-located faces."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate_on_box_disamb")
    occ = gmsh.model.occ

    sub = occ.addBox(0, 0, 0, 10, 10, 1)
    plate = occ.addRectangle(2, 2, 1, 6, 6)
    occ.synchronize()

    sub_top_cog = (5, 5, 1)
    sub_top_bbox = bbox_diag(2, [t for d, t in gmsh.model.getBoundary([(3, sub)], oriented=False)
                                  if d == 2 and abs(cog(2, t)[2] - 1) < TOL][0])
    plate_cog = cog(2, plate)
    plate_bbox = bbox_diag(2, plate)

    occ.fragment([(3, sub)], [(2, plate)])
    occ.synchronize()

    # Naive COG-only resolution
    naive_sub_top = find_by_cog(2, sub_top_cog, TOL)
    naive_plate = find_by_cog(2, plate_cog, TOL)

    # COG + bbox resolution
    smart_sub_top = find_by_cog_bbox(2, sub_top_cog, sub_top_bbox, TOL)
    smart_plate = find_by_cog_bbox(2, plate_cog, plate_bbox, TOL)

    gmsh.finalize()
    return (
        f"naive (COG only):   sub_top -> tag {naive_sub_top}, plate -> tag {naive_plate}\n"
        f"  smart (COG + bbox): sub_top -> tag {smart_sub_top}, plate -> tag {smart_plate}\n"
        f"  (smart should resolve both correctly to DIFFERENT tags)"
    )


def case_plate_on_box() -> str:
    """Practical: a Box + an XYPlate sitting on its top face. Fragment them.
    The plate is conceptually a 'sub-region' of the box's top face. We want:
    - Substrate top face (remaining after plate is cut out): re-findable
    - Plate face (was an independent 2D entity): re-findable
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate_on_box")
    occ = gmsh.model.occ

    sub = occ.addBox(0, 0, 0, 10, 10, 1)        # 10×10 substrate, 1 thick
    plate = occ.addRectangle(2, 2, 1, 6, 6)     # 6×6 plate sitting at z=1
    occ.synchronize()

    # Mark the 6 outer faces of the substrate by COG, and the plate face by COG
    sub_face_cogs = {f"sub_{i}": cog(d, t) for i, (d, t) in enumerate(gmsh.model.getBoundary([(3, sub)], oriented=False)) if d == 2}
    plate_cog = cog(2, plate)

    # Fragment: this should embed the plate into the substrate's top face,
    # splitting that top face into "plate region" and "remaining annulus" sub-faces.
    occ.fragment([(3, sub)], [(2, plate)])
    occ.synchronize()

    # Re-resolve: each substrate face by COG. The TOP face's COG (5, 5, 1)
    # might now be the merged region, OR a sub-region. Test what gmsh actually does.
    misses = []
    for n, c in sub_face_cogs.items():
        t = find_by_cog(2, c, TOL)
        # Also check looser tolerance in case fragment shifts COG of split faces
        if t is None:
            t = find_by_cog(2, c, 1e-6)
            if t is not None:
                misses.append(f"{n} (only loose match @ tol=1e-6)")
            else:
                misses.append(f"{n} (no match)")
    plate_resolved = find_by_cog(2, plate_cog, TOL)
    msg = f"sub face misses: {misses}, plate face resolved: {plate_resolved is not None}"

    # Bonus: enumerate post-fragment faces with their COGs for diagnosis
    post_faces = [(t, cog(2, t)) for d, t in gmsh.model.getEntities(2)]
    diag = "\n    " + "\n    ".join(f"face tag={t} cog=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})" for t, c in post_faces)

    gmsh.finalize()
    return msg + diag


if __name__ == "__main__":
    print("=" * 60)
    print("Spike: gmsh OCC boolean ops + COG-based name re-resolution")
    print("=" * 60)
    print()
    print(f"renumber check:")
    print(f"  {case_renumber_check()}")
    print()
    print(f"fragment(box, box): {case_fragment_two_boxes()}")
    print(f"cut(big, small):    {case_cut_box_minus_box()}")
    print(f"fuse(box, box):     {case_fuse_two_boxes()}")
    print()
    print(f"plate_on_box (practical patch antenna case):")
    print(f"  {case_plate_on_box()}")
    print()
    print(f"plate_on_box, COG ambiguity test:")
    print(f"  {case_plate_on_box_disambiguated()}")
    print()
    print("=" * 60)
    print("Conclusions for production layer:")
    print("=" * 60)
    print("  - fragment + cut: re-resolution by COG works")
    print("  - coplanar overlap (annulus + sub-region): COG-only is AMBIGUOUS")
    print("    -> need COG + bbox, with parent-volume disambiguator as fallback")
    print("  - fuse: face merging shifts COG -> names cannot be preserved through fuse")
    print("    -> document as user-facing limitation; warn at fuse() call site")
    print("  - tags can stay stable for unchanged faces (opportunistic optimization)")
