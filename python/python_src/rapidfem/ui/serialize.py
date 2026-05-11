"""Serialize rapidfem objects into JSON payloads the viewer can consume.

The bundled canvas3d viewer expects per-entity buffers in the form
``{ name, tag, color: [r,g,b], positions: number[], normals: number[] }``
where ``positions`` and ``normals`` are flat float arrays (3 components per
vertex, 3 vertices per triangle — flat-shaded, no indexing).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import gmsh


# ── Geometry → triangle payload ───────────────────────────────────────────────


def _color_from_name(name: str) -> list[float]:
    """Stable, evenly-distributed RGB color in [0,1] from a name."""
    if not name:
        name = "_unnamed"
    h = hashlib.md5(name.encode("utf-8")).digest()
    # golden-ratio hue, varying saturation/value just enough to separate
    hue = (h[0] / 255.0)
    sat = 0.45 + (h[1] / 255.0) * 0.30
    val = 0.65 + (h[2] / 255.0) * 0.25
    # HSV → RGB
    i = int(hue * 6)
    f = hue * 6 - i
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)
    table = [(val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q)]
    return list(table[i % 6])


def _surface_triangulation(dim_tag: tuple[int, int]) -> tuple[list[float], list[float]]:
    """Extract a triangulated surface for one 2D entity.

    Returns (positions, normals) as flat python lists in METERS.
    Each triangle contributes 3 × 3 floats; normals are flat-shaded.
    """
    dim, tag = dim_tag
    if dim != 2:
        return [], []
    # Gmsh mesh element type 2 == 3-node triangle.
    types, _elem_tags, node_tags = gmsh.model.mesh.getElements(dim=2, tag=tag)
    positions: list[float] = []
    normals: list[float] = []
    for et, nodes in zip(types, node_tags):
        if et != 2:
            continue
        # nodes is a flat list, 3 node ids per triangle
        for i in range(0, len(nodes), 3):
            a_id, b_id, c_id = nodes[i], nodes[i + 1], nodes[i + 2]
            ax, ay, az = gmsh.model.mesh.getNode(a_id)[0]
            bx, by, bz = gmsh.model.mesh.getNode(b_id)[0]
            cx, cy, cz = gmsh.model.mesh.getNode(c_id)[0]
            # flat normal = (b-a) x (c-a) normalized
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            nl = math.sqrt(nx * nx + ny * ny + nz * nz)
            if nl > 0:
                nx, ny, nz = nx / nl, ny / nl, nz / nl
            positions.extend((ax, ay, az, bx, by, bz, cx, cy, cz))
            normals.extend((nx, ny, nz) * 3)
    return positions, normals


def _ensure_surface_mesh(maxh: float) -> None:
    """Generate a coarse surface mesh suitable for visualization."""
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", maxh)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.model.mesh.generate(2)


def _global_bbox() -> dict[str, list[float]]:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    if not all(math.isfinite(v) for v in (xmin, ymin, zmin, xmax, ymax, zmax)):
        return {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    return {"min": [xmin, ymin, zmin], "max": [xmax, ymax, zmax]}


def _bbox_diag() -> float:
    bb = _global_bbox()
    dx = bb["max"][0] - bb["min"][0]
    dy = bb["max"][1] - bb["min"][1]
    dz = bb["max"][2] - bb["min"][2]
    return max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-9)


def geometry_to_payload(g: Any, *, target_tris: int = 4000) -> dict:
    """Tessellate a Geometry's OCC entities and produce a viewer payload.

    Uses gmsh's 2D mesher with a coarse size to keep this responsive on
    every save (``rapidfem serve`` calls this on Ctrl+S). ``target_tris``
    nudges the mesh size to land near a desired triangle budget.
    """
    gmsh.model.occ.synchronize()
    diag = _bbox_diag()
    maxh = diag / max(math.sqrt(target_tris / 6.0), 4.0)
    # If the user already meshed (explicit g.mesh() or via builder.from_geometry),
    # the gmsh state holds a full 3D tet mesh — return that as a Mesh payload
    # so the viewer can render the actual FEM discretization, not a coarse
    # preview tessellation.
    existing_node_tags, _, _ = gmsh.model.mesh.getNodes()
    if len(existing_node_tags) > 0:
        try:
            return mesh_to_payload(g, maxh=0.0)
        except Exception:
            pass  # fall through to coarse preview if extraction fails
    _ensure_surface_mesh(maxh)

    # Group OCC surfaces by their owning named entity. Process 2-D faces
    # FIRST so named ports/PEC/etc. claim their surfaces before the parent
    # volume sweeps up everything else.
    raw_ents = list(getattr(g, "_entities", []))
    def _ent_sort_key(e):
        # Named 2-D faces first, then unnamed 2-D, then 3-D volumes.
        dim_rank = 0 if e.dim == 2 else 1 if e.dim == 3 else 2
        name_rank = 0 if e.name else 1
        return (dim_rank, name_rank)
    raw_ents.sort(key=_ent_sort_key)

    entities: list[dict] = []
    seen_surface_tags: set[int] = set()

    for ent in raw_ents:
        name = ent.name or f"_{ 'face' if ent.dim == 2 else 'volume' }_{ent.tag}"
        surface_dim_tags: list[tuple[int, int]] = []
        if ent.dim == 3:
            for d, t in gmsh.model.getBoundary([(3, ent.tag)], oriented=False):
                if d == 2 and t not in seen_surface_tags:
                    surface_dim_tags.append((2, t))
                    seen_surface_tags.add(t)
        elif ent.dim == 2:
            if ent.tag not in seen_surface_tags:
                surface_dim_tags.append((2, ent.tag))
                seen_surface_tags.add(ent.tag)
        else:
            continue

        pos: list[float] = []
        nor: list[float] = []
        for dt in surface_dim_tags:
            p, n = _surface_triangulation(dt)
            pos.extend(p)
            nor.extend(n)

        if not pos:
            continue
        entities.append({
            "name": name,
            "tag": int(ent.tag),
            "dim": int(ent.dim),
            "color": _color_from_name(name),
            "positions": pos,
            "normals": nor,
            "material": ent.material,
        })

    # Also collect untracked surfaces (anything the user didn't name) so the
    # viewer still shows the full shape.
    untracked_pos: list[float] = []
    untracked_nor: list[float] = []
    for d, t in gmsh.model.getEntities(2):
        if t in seen_surface_tags:
            continue
        p, n = _surface_triangulation((d, t))
        untracked_pos.extend(p)
        untracked_nor.extend(n)
    if untracked_pos:
        entities.append({
            "name": "_untracked",
            "tag": 0,
            "dim": 2,
            "color": [0.55, 0.55, 0.55],
            "positions": untracked_pos,
            "normals": untracked_nor,
            "material": None,
        })

    return {
        "kind": "geometry",
        "bbox": _global_bbox(),
        "entities": entities,
        "stats": {
            "n_entities": len(entities),
            "n_triangles": sum(len(e["positions"]) // 9 for e in entities),
            "maxh": maxh,
        },
    }


# ── Full 3D mesh → viewer payload ─────────────────────────────────────────────


def mesh_to_payload(g: Any, *, maxh: float) -> dict:
    """Generate the full 3D mesh on the Geometry and extract a viewer payload.

    Calls ``g.mesh(maxh=maxh)`` which leaves the gmsh model populated with a
    tet mesh + named physical groups; we then read nodes, tets, and surface
    triangles back out and ship them to the canvas3d viewer in its expected
    MeshData layout.
    """
    import time
    t0 = time.perf_counter()
    # If the user already triggered meshing (e.g. via builder.from_geometry()
    # in the same script), gmsh holds the mesh + physical groups — calling
    # g.mesh() again collides on duplicate physical tags. Skip the re-mesh in
    # that case; otherwise generate now.
    name_to_tag: dict[str, int] = {}
    msh_bytes_len = 0
    existing_node_tags, _, _ = gmsh.model.mesh.getNodes()
    if len(existing_node_tags) == 0:
        mesh_bytes_local, name_to_tag = g.mesh(maxh=maxh)
        msh_bytes_len = len(mesh_bytes_local)
    else:
        # Recover name_to_tag from gmsh's physical groups (drop "_mat_" prefix).
        for dim, ptag in gmsh.model.getPhysicalGroups():
            n = gmsh.model.getPhysicalName(dim, ptag) or ""
            if n.startswith("_mat_"):
                n = n[len("_mat_"):]
            if n:
                name_to_tag[n] = ptag
    t_mesh = time.perf_counter() - t0

    # ── Nodes ────────────────────────────────────────────────────────────
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    # node_tags is 1-based and may not be contiguous after fragment ops.
    # Build a dense index map: gmsh_tag -> dense_idx
    max_tag = int(node_tags.max()) if len(node_tags) else 0
    idx_map = [0] * (max_tag + 1)
    for i, t in enumerate(node_tags):
        idx_map[int(t)] = i
    nodes_flat = list(coords)  # already flat xyz

    # ── Physical groups ──────────────────────────────────────────────────
    phys_names: dict[int, str] = {}
    phys_dim: dict[int, int] = {}
    phys_to_entities: dict[int, list[int]] = {}
    for dim, ptag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, ptag) or f"_phys_{dim}_{ptag}"
        if name.startswith("_mat_"):
            name = name[len("_mat_"):]
        phys_names[ptag] = name
        phys_dim[ptag] = dim
        phys_to_entities[ptag] = list(gmsh.model.getEntitiesForPhysicalGroup(dim, ptag))

    # Map each entity (dim, tag) → its physical-group tag (if any). Entities
    # may belong to multiple physical groups; we take the first hit.
    entity_to_phys: dict[tuple[int, int], int] = {}
    for ptag, ents in phys_to_entities.items():
        d = phys_dim[ptag]
        for e in ents:
            entity_to_phys.setdefault((d, int(e)), ptag)

    # ── Tets (3D elements) ───────────────────────────────────────────────
    tets_flat: list[int] = []
    tet_phys: list[int] = []
    for dim, etag in gmsh.model.getEntities(3):
        etypes, eelem_tags, enode_tags = gmsh.model.mesh.getElements(dim=3, tag=etag)
        ptag = entity_to_phys.get((3, etag), 0)
        for et, _tags, nodes_arr in zip(etypes, eelem_tags, enode_tags):
            if et != 4:  # 4-node tet
                continue
            n = len(nodes_arr) // 4
            for k in range(n):
                a = idx_map[int(nodes_arr[4 * k + 0])]
                b = idx_map[int(nodes_arr[4 * k + 1])]
                c = idx_map[int(nodes_arr[4 * k + 2])]
                d_ = idx_map[int(nodes_arr[4 * k + 3])]
                tets_flat.extend((a, b, c, d_))
                tet_phys.append(ptag)

    # ── Surface tris ─────────────────────────────────────────────────────
    tris_flat: list[int] = []
    tri_phys: list[int] = []
    for dim, etag in gmsh.model.getEntities(2):
        etypes, _eelem_tags, enode_tags = gmsh.model.mesh.getElements(dim=2, tag=etag)
        ptag = entity_to_phys.get((2, etag), 0)
        for et, nodes_arr in zip(etypes, enode_tags):
            if et != 2:  # 3-node triangle
                continue
            n = len(nodes_arr) // 3
            for k in range(n):
                a = idx_map[int(nodes_arr[3 * k + 0])]
                b = idx_map[int(nodes_arr[3 * k + 1])]
                c = idx_map[int(nodes_arr[3 * k + 2])]
                tris_flat.extend((a, b, c))
                tri_phys.append(ptag)

    return {
        "kind": "mesh",
        "bbox": _global_bbox(),
        "nodes": nodes_flat,
        "tris": tris_flat,
        "tri_phys": tri_phys,
        "tets": tets_flat,
        "tet_phys": tet_phys,
        "phys_names": phys_names,
        "phys_dim": phys_dim,
        "name_to_tag": name_to_tag,
        "stats": {
            "n_nodes": len(nodes_flat) // 3,
            "n_tets": len(tet_phys),
            "n_tris": len(tri_phys),
            "mesh_time_s": t_mesh,
            "msh_bytes": msh_bytes_len,
        },
    }
