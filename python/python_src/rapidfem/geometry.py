"""
Geometry + meshing builder for rapidfem, NGSolve-style.

Wraps gmsh's OpenCASCADE kernel with named entities that survive boolean ops:

    g = Geometry()
    sub = g.box(60e-3, 60e-3, 1.6e-3, position=(-30e-3, -30e-3, 0))
    patch = g.xy_plate(38e-3, 29e-3, position=(-19e-3, -14.5e-3, 1.6e-3))

    g.fragment(sub, patch)         # bool op, names survive

    sub.faces.min(axis="z").name = "ground"   # selector + attribute write
    patch.name = "patch_pec"
    sub.material = "fr4"

    sub.maxh = 5e-3
    patch.maxh = 1.5e-3

    mesh_bytes, name_to_tag = g.mesh(maxh=10e-3)

The `name_to_tag` dict feeds into rapidfem.SimulationBuilder so users never write
integer physical group tags by hand.

Tracking strategy (per spike findings, see python/spike_geometry.py):
- Each named entity stores (cog, bbox, dim) at registration time.
- After every boolean op, the geometry walks its registry and re-resolves each
  entity by matching (cog, bbox) against current gmsh entities. COG-only is
  ambiguous for coplanar overlapping faces (e.g. annulus + sub-region after
  embedding a plate); bbox disambiguates.
- `fuse` is supported but warned: face merging shifts COGs, names cannot survive.
"""
from __future__ import annotations

import io
import math
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable

import gmsh

# ── Tolerances ────────────────────────────────────────────────────────────────
_COG_TOL = 1e-9   # distance tol for matching center-of-mass (m)
_BBOX_TOL = 1e-9  # tol for matching bounding-box corners (m)


# ─────────────────────────────────────────────────────────────────────────────
# Internal entity tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Entity:
    """A tracked gmsh entity with stable identity through boolean ops.

    `tag` is the *current* gmsh tag — may be updated by Geometry._reresolve()
    after fragment/cut. The (cog, bbox, dim) triple is the stable identity used
    for re-resolution.
    """
    dim: int
    tag: int
    cog: tuple[float, float, float]
    bbox: tuple[float, float, float, float, float, float]
    name: str | None = None
    material: str | None = None
    maxh: float | None = None

    @staticmethod
    def from_dimtag(dim: int, tag: int) -> "_Entity":
        cog = tuple(gmsh.model.occ.getCenterOfMass(dim, tag))
        bbox = tuple(gmsh.model.getBoundingBox(dim, tag))
        return _Entity(dim=dim, tag=tag, cog=cog, bbox=bbox)


def _bbox_match(a: tuple, b: tuple, tol: float) -> bool:
    return all(abs(a[i] - b[i]) < tol for i in range(6))


def _resolve_entity(target: _Entity) -> int | None:
    """Find a gmsh entity in `target.dim` matching the stored (bbox, cog).

    Strategy: bbox is the primary identity (stable through fragment, even when
    sub-volumes are carved out — the outer bbox doesn't change). COG is a
    secondary discriminator with a tolerance that scales with the bbox extent.
    This handles two real failure modes:
      1. COG drift after carving a sub-volume out of a larger one (e.g. air
         box minus substrate) — bbox stays put, COG shifts by mass redistribution.
      2. Coplanar overlapping faces (annulus + sub-region after fragment) —
         both have the same COG, but different bboxes.
    """
    # bbox extent diagonal — sets an "internal scale" for COG drift tolerance.
    extent = math.sqrt(sum((target.bbox[3 + i] - target.bbox[i]) ** 2 for i in range(3)))
    cog_tol = max(_COG_TOL, 0.01 * extent)  # 1% of diagonal, or absolute floor

    candidates = []
    for d, t in gmsh.model.getEntities(target.dim):
        if d != target.dim:
            continue
        bbox = tuple(gmsh.model.getBoundingBox(d, t))
        if not _bbox_match(bbox, target.bbox, _BBOX_TOL):
            continue
        cog = tuple(gmsh.model.occ.getCenterOfMass(d, t))
        if math.dist(cog, target.cog) < cog_tol:
            candidates.append(t)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Tie-break by smallest COG distance.
        candidates.sort(key=lambda t: math.dist(
            tuple(gmsh.model.occ.getCenterOfMass(target.dim, t)), target.cog))
        return candidates[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Entity collection with selectors and attribute writes
# ─────────────────────────────────────────────────────────────────────────────

class EntityCollection:
    """A collection of sub-entities (faces or edges) with selectors and bulk
    attribute writes.

    Setting `.name = "..."` or `.maxh = ...` applies to all members. Selectors
    return *new* collections, so chaining composes:

        box.faces.where(lambda c, b: c[2] < 0.5).min(axis="x").name = "port"
        box.edges.where(lambda c, _: c[2] == 0).maxh = 1e-3
    """

    def __init__(self, geometry: "Geometry", entities: list[_Entity]):
        self._geometry = geometry
        self._entities = entities

    def __iter__(self):
        return iter(self._entities)

    def __len__(self):
        return len(self._entities)

    # Selectors
    def min(self, axis: str = "z") -> "EntityCollection":
        ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        if not self._entities:
            return EntityCollection(self._geometry, [])
        m = min(e.cog[ax] for e in self._entities)
        kept = [e for e in self._entities if abs(e.cog[ax] - m) < _COG_TOL]
        return EntityCollection(self._geometry, kept)

    def max(self, axis: str = "z") -> "EntityCollection":
        ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        if not self._entities:
            return EntityCollection(self._geometry, [])
        m = max(e.cog[ax] for e in self._entities)
        kept = [e for e in self._entities if abs(e.cog[ax] - m) < _COG_TOL]
        return EntityCollection(self._geometry, kept)

    def where(self, predicate: Callable[[tuple, tuple], bool]) -> "EntityCollection":
        kept = [e for e in self._entities if predicate(e.cog, e.bbox)]
        return EntityCollection(self._geometry, kept)

    # Bulk attribute setters (NGSolve idiom: `coll.name = "..."` writes to all)
    @property
    def name(self) -> str | None:
        names = {e.name for e in self._entities if e.name is not None}
        if len(names) == 1:
            return names.pop()
        return None

    @name.setter
    def name(self, value: str) -> None:
        for e in self._entities:
            e.name = value

    @property
    def maxh(self) -> float | None:
        vals = {e.maxh for e in self._entities if e.maxh is not None}
        if len(vals) == 1:
            return vals.pop()
        return None

    @maxh.setter
    def maxh(self, value: float) -> None:
        for e in self._entities:
            e.maxh = value


# Back-compat aliases (and clearer naming for users)
FaceCollection = EntityCollection
EdgeCollection = EntityCollection


# ─────────────────────────────────────────────────────────────────────────────
# Top-level geometric object
# ─────────────────────────────────────────────────────────────────────────────

class GeoObject:
    """A primitive (volume or 2D plate) in the geometry.

    Direct attribute writes set the entity's own name/material/maxh:

        substrate.name = "fr4_volume"
        substrate.material = "fr4"
        substrate.maxh = 5e-3

    Sub-entity collections expose selectors:

        substrate.faces.min(axis="z").name = "ground"
    """

    def __init__(self, geometry: "Geometry", entity: _Entity):
        self._geometry = geometry
        self._entity = entity

        # Discover faces (dim=2) and edges (dim=1) of the parent entity.
        # gmsh's getBoundary recursive=True returns vertices (lowest dim), not edges.
        # We walk faces ourselves to find edges, deduping shared edges.
        face_ents: list[_Entity] = []
        edge_ents: list[_Entity] = []
        seen_edge_tags: set[int] = set()
        if entity.dim == 3:
            for d, t in gmsh.model.getBoundary([(3, entity.tag)], oriented=False):
                if d != 2:
                    continue
                face_ents.append(_Entity.from_dimtag(d, t))
                for d_e, t_e in gmsh.model.getBoundary([(2, t)], oriented=False):
                    if d_e == 1 and t_e not in seen_edge_tags:
                        seen_edge_tags.add(t_e)
                        edge_ents.append(_Entity.from_dimtag(d_e, t_e))
        elif entity.dim == 2:
            for d, t in gmsh.model.getBoundary([(2, entity.tag)], oriented=False):
                if d == 1 and t not in seen_edge_tags:
                    seen_edge_tags.add(t)
                    edge_ents.append(_Entity.from_dimtag(d, t))

        self.faces = EntityCollection(geometry, face_ents)
        self.edges = EntityCollection(geometry, edge_ents)
        for e in face_ents + edge_ents:
            geometry._entities.append(e)

    @property
    def name(self) -> str | None:
        return self._entity.name

    @name.setter
    def name(self, value: str) -> None:
        self._entity.name = value

    @property
    def material(self) -> str | None:
        return self._entity.material

    @material.setter
    def material(self, value: str) -> None:
        self._entity.material = value

    @property
    def maxh(self) -> float | None:
        return self._entity.maxh

    @maxh.setter
    def maxh(self, value: float) -> None:
        self._entity.maxh = value

    @property
    def dim(self) -> int:
        return self._entity.dim


# ─────────────────────────────────────────────────────────────────────────────
# Geometry — top-level builder owning a gmsh session
# ─────────────────────────────────────────────────────────────────────────────

class Geometry:
    """Top-level geometry builder. Owns a gmsh model session for its lifetime."""

    def __init__(self, name: str = "rapidfem"):
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(name)
        self._objects: list[GeoObject] = []
        self._entities: list[_Entity] = []  # all named-or-trackable entities
        self._owns_gmsh = True  # we'll finalize on close

    # ── GDS-driven extrusion ────────────────────────────────────────────────

    @staticmethod
    def from_gds(
        path: str,
        stack,                          # rapidfem.rfic.Stack
        top_cell: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        flatten: bool = True,
        merge: bool = True,
    ) -> "Geometry":
        """Load a GDSII layout and extrude all matching polygons into 3D primitives.

        Each polygon on a (gds, datatype) tuple in `stack` becomes a 3D box
        (rectangles) or extruded prism (general polygons) at the layer's z with
        the layer's thickness. All polygons of one layer get the layer's name
        as a physical-group tag so the SimulationBuilder can wire them by name.

        Args:
            path: Path to the .gds(.gz) file.
            stack: A `rapidfem.rfic.Stack` mapping (gds, datatype) → PdkLayer.
            top_cell: Cell name to extrude. ``None`` ⇒ auto-pick the unique
                top-level cell.
            bbox: Optional (xmin, ymin, xmax, ymax) crop box in meters; polygons
                outside are skipped (faster iteration on large layouts).
            flatten: Resolve cell references before extrusion (default True).
                Set False only if your top cell has no references.
            merge: Merge co-layer polygons via gmsh fragment so adjacent traces
                produce a conformal mesh (default True).
        """
        try:
            import gdstk
        except ImportError as e:
            raise ImportError("gdstk is required for from_gds(). pip install gdstk") from e
        import numpy as np

        lib = gdstk.read_gds(path)
        cells_by_name = {c.name: c for c in lib.cells}

        # Resolve top cell
        if top_cell is None:
            tops = lib.top_level()
            if len(tops) != 1:
                names = ", ".join(c.name for c in tops)
                raise ValueError(
                    f"GDS has {len(tops)} top-level cells ({names}); "
                    f"specify top_cell=...")
            cell = tops[0]
        else:
            if top_cell not in cells_by_name:
                avail = ", ".join(cells_by_name.keys())
                raise ValueError(f"top_cell {top_cell!r} not in GDS; available: {avail}")
            cell = cells_by_name[top_cell]

        # GDS unit (typically 1e-6 = micron, sometimes 1e-9 = nm). gdstk reports
        # the working unit in lib.unit (meters per GDS unit).
        unit = lib.unit  # meters per logical unit in the GDS

        # Flatten the cell to expose all polygons including those in references.
        flat_polys = cell.get_polygons() if not flatten else cell.flatten().polygons

        # Optional bbox crop
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox

        g = Geometry(name=cell.name or "gds_import")
        # Group polygons by their PdkLayer (so we name + extrude per-layer).
        per_layer: dict[str, list[np.ndarray]] = {}
        for poly in flat_polys:
            pdk_layer = stack.by_gds(poly.layer, poly.datatype)
            if pdk_layer is None:
                continue
            pts_raw = np.asarray(poly.points, dtype=np.float64)
            if bbox is not None:
                if (pts_raw[:, 0].max() < xmin or pts_raw[:, 0].min() > xmax
                        or pts_raw[:, 1].max() < ymin or pts_raw[:, 1].min() > ymax):
                    continue
            pts_m = pts_raw * unit  # convert GDS coords → meters
            per_layer.setdefault(pdk_layer.name, []).append(pts_m)

        # Extrude each polygon into a 3D solid at the layer's z with its thickness.
        # Rectangular axis-aligned polygons take the fast addBox path;
        # everything else builds a wire → plane surface → extrude.
        per_layer_objs: dict[str, list[GeoObject]] = {}
        for layer_name, polys in per_layer.items():
            pdk = stack.by_name(layer_name)
            objs: list[GeoObject] = []
            for pts in polys:
                obj = g._extrude_polygon(pts, z=pdk.z, thickness=pdk.thickness)
                obj.name = layer_name
                objs.append(obj)
            per_layer_objs[layer_name] = objs

        # Optional: fragment co-layer polygons so adjacent traces share faces.
        if merge:
            for objs in per_layer_objs.values():
                if len(objs) >= 2:
                    g.fragment(objs[0], *objs[1:])

        return g

    def _extrude_polygon(
        self,
        pts: "np.ndarray",
        z: float,
        thickness: float,
    ) -> GeoObject:
        """Extrude a 2D polygon (Nx2 numpy array) vertically into a 3D solid.

        Rectangular axis-aligned polygons → addBox (fast path).
        Otherwise: build a wire → plane surface → extrude.
        """
        import numpy as np

        # Detect axis-aligned rectangle (4 points, 90° corners)
        if pts.shape[0] in (4, 5):
            p = pts[:4]
            xs, ys = sorted(set(p[:, 0])), sorted(set(p[:, 1]))
            if len(xs) == 2 and len(ys) == 2:
                tag = gmsh.model.occ.addBox(
                    xs[0], ys[0], z,
                    xs[1] - xs[0], ys[1] - ys[0], thickness,
                )
                return self._wrap_volume(tag)

        # General polygon: wire → plane → extrude
        # Drop duplicated last point (gdstk includes it sometimes)
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        pt_tags = [gmsh.model.occ.addPoint(p[0], p[1], z) for p in pts]
        line_tags = [
            gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i + 1) % len(pt_tags)])
            for i in range(len(pt_tags))
        ]
        loop = gmsh.model.occ.addCurveLoop(line_tags)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        # Extrude vertically by thickness; second elt of return is the top cap
        out = gmsh.model.occ.extrude([(2, surf)], 0, 0, thickness)
        # gmsh's extrude returns: [top_face, volume, side_faces...]
        vol_tag = next(t for d, t in out if d == 3)
        return self._wrap_volume(vol_tag)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._owns_gmsh and gmsh.isInitialized():
            gmsh.finalize()
            self._owns_gmsh = False

    # ── Primitives ──────────────────────────────────────────────────────────

    def _wrap_volume(self, tag: int) -> GeoObject:
        gmsh.model.occ.synchronize()
        ent = _Entity.from_dimtag(3, tag)
        obj = GeoObject(self, ent)
        self._objects.append(obj)
        self._entities.append(ent)
        return obj

    def _wrap_face(self, tag: int) -> GeoObject:
        gmsh.model.occ.synchronize()
        ent = _Entity.from_dimtag(2, tag)
        obj = GeoObject(self, ent)
        self._objects.append(obj)
        self._entities.append(ent)
        return obj

    def box(self, width: float, depth: float, height: float,
            position: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        """Axis-aligned box. `position` = lower (xmin, ymin, zmin) corner; the
        box extends `width`, `depth`, `height` along x, y, z respectively.
        Returns a `GeoObject` with 6 `.faces`, 12 `.edges`."""
        x, y, z = position
        tag = gmsh.model.occ.addBox(x, y, z, width, depth, height)
        return self._wrap_volume(tag)

    def cylinder(self, radius: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0),
                 axis: tuple[float, float, float] = (0, 0, 1),
                 angle: float = 2 * math.pi) -> GeoObject:
        """Cylinder along `axis`. `position` = base center, `height` is along `axis`."""
        x, y, z = position
        ax, ay, az = (axis[0] * height, axis[1] * height, axis[2] * height)
        tag = gmsh.model.occ.addCylinder(x, y, z, ax, ay, az, radius, angle=angle)
        return self._wrap_volume(tag)

    def sphere(self, radius: float, center: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        cx, cy, cz = center
        tag = gmsh.model.occ.addSphere(cx, cy, cz, radius)
        return self._wrap_volume(tag)

    def cone(self, r1: float, r2: float, height: float,
             position: tuple[float, float, float] = (0, 0, 0),
             axis: tuple[float, float, float] = (0, 0, 1),
             angle: float = 2 * math.pi) -> GeoObject:
        """Truncated cone (or cylinder if r1==r2). `position` = base center."""
        x, y, z = position
        ax, ay, az = (axis[0] * height, axis[1] * height, axis[2] * height)
        tag = gmsh.model.occ.addCone(x, y, z, ax, ay, az, r1, r2, angle=angle)
        return self._wrap_volume(tag)

    def wedge(self, dx: float, dy: float, dz: float,
              top_x: float = 0.0,
              position: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        """Rectangular-base prism. The base is dx×dy at z=0; the top edge runs
        from x=0 to x=top_x at height dz (parallel to y). top_x=0 ⇒ triangular
        wedge; top_x=dx ⇒ ordinary box.
        """
        x, y, z = position
        tag = gmsh.model.occ.addWedge(x, y, z, dx, dy, dz, ltx=top_x)
        return self._wrap_volume(tag)

    def torus(self, major_radius: float, minor_radius: float,
              center: tuple[float, float, float] = (0, 0, 0),
              angle: float = 2 * math.pi) -> GeoObject:
        """Torus with major (donut) and minor (tube) radii, centered on `center`,
        with axis along z. `angle` < 2π gives a partial torus."""
        cx, cy, cz = center
        tag = gmsh.model.occ.addTorus(cx, cy, cz, major_radius, minor_radius, angle=angle)
        return self._wrap_volume(tag)

    def xy_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        """Rectangle in the xy-plane (constant z). `width` along x, `height` along y."""
        x, y, z = position
        tag = gmsh.model.occ.addRectangle(x, y, z, width, height)
        return self._wrap_face(tag)

    def xz_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        """Rectangle in the xz-plane (constant y). `width` along x, `height` along z."""
        return self.plate(p0=position, width=(width, 0, 0), height=(0, 0, height))

    def yz_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        """Rectangle in the yz-plane (constant x). `width` along y, `height` along z."""
        return self.plate(p0=position, width=(0, width, 0), height=(0, 0, height))

    def plate(self, p0: tuple[float, float, float],
              width: tuple[float, float, float],
              height: tuple[float, float, float]) -> GeoObject:
        """Plate at arbitrary orientation. p0 = corner; width, height = edge vectors.

        gmsh OCC has no direct API; we build a 4-vertex wire and surface.
        """
        x0, y0, z0 = p0
        wx, wy, wz = width
        hx, hy, hz = height
        v1 = gmsh.model.occ.addPoint(x0, y0, z0)
        v2 = gmsh.model.occ.addPoint(x0 + wx, y0 + wy, z0 + wz)
        v3 = gmsh.model.occ.addPoint(x0 + wx + hx, y0 + wy + hy, z0 + wz + hz)
        v4 = gmsh.model.occ.addPoint(x0 + hx, y0 + hy, z0 + hz)
        l1 = gmsh.model.occ.addLine(v1, v2)
        l2 = gmsh.model.occ.addLine(v2, v3)
        l3 = gmsh.model.occ.addLine(v3, v4)
        l4 = gmsh.model.occ.addLine(v4, v1)
        loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        tag = gmsh.model.occ.addPlaneSurface([loop])
        return self._wrap_face(tag)

    # ── Boolean ops ─────────────────────────────────────────────────────────

    def fragment(self, target: GeoObject, *tools: GeoObject) -> None:
        """Boolean fragment: makes geometry conformal at interfaces. Survives names."""
        target_dt = [(target.dim, target._entity.tag)]
        tools_dt = [(t.dim, t._entity.tag) for t in tools]
        gmsh.model.occ.fragment(target_dt, tools_dt)
        gmsh.model.occ.synchronize()
        self._reresolve()

    def cut(self, target: GeoObject, *tools: GeoObject) -> None:
        """Boolean subtract."""
        target_dt = [(target.dim, target._entity.tag)]
        tools_dt = [(t.dim, t._entity.tag) for t in tools]
        gmsh.model.occ.cut(target_dt, tools_dt)
        gmsh.model.occ.synchronize()
        self._reresolve()

    def fuse(self, target: GeoObject, *tools: GeoObject) -> None:
        """Boolean union. WARNING: face names are NOT preserved (faces merge,
        COGs shift). Use only when names on faces don't matter."""
        warnings.warn(
            "fuse() merges faces and shifts their COGs; named faces on the "
            "operands will not be reliably re-resolvable. Set names AFTER fuse, "
            "or use fragment() if you need names preserved.",
            stacklevel=2,
        )
        target_dt = [(target.dim, target._entity.tag)]
        tools_dt = [(t.dim, t._entity.tag) for t in tools]
        gmsh.model.occ.fuse(target_dt, tools_dt)
        gmsh.model.occ.synchronize()
        self._reresolve()

    def _reresolve(self) -> None:
        """Re-find every tracked entity by (cog, bbox) after a boolean op."""
        survived = []
        for ent in self._entities:
            new_tag = _resolve_entity(ent)
            if new_tag is not None:
                ent.tag = new_tag
                survived.append(ent)
            elif ent.name or ent.material or ent.maxh:
                # Lost an entity that had user attributes — warn
                warnings.warn(
                    f"Tracked entity (dim={ent.dim}, name={ent.name!r}, "
                    f"cog={ent.cog}) lost during boolean op; attributes will be dropped.",
                    stacklevel=3,
                )
            # else: silently drop untracked entities (boundary helpers etc.)
        self._entities = survived

    # ── Mesh emit ───────────────────────────────────────────────────────────

    def mesh(self, maxh: float = 1.0, transition_distance: float | None = None) -> tuple[bytes, dict[str, int]]:
        """Generate the 3D mesh and return (msh_bytes, name_to_tag).

        Per-entity `obj.maxh = h` is honored via gmsh `Distance` + `Threshold`
        background fields, so refinement transitions are smooth instead of abrupt.
        Each refined entity contributes a Threshold field that grows from `h`
        right at the entity to the global `maxh` at `transition_distance`
        (default: 5*h). The combined background is the per-cell minimum of all
        Threshold fields, so the smallest size wins where regions overlap.

        `name_to_tag` maps each user-supplied name to its physical-group tag.
        """
        gmsh.model.occ.synchronize()

        # ── Per-entity mesh size: gmsh Distance + Threshold background fields ──
        threshold_field_ids: list[int] = []
        for ent in self._entities:
            if ent.maxh is None:
                continue
            dist_id = gmsh.model.mesh.field.add("Distance")
            if ent.dim == 0:
                gmsh.model.mesh.field.setNumbers(dist_id, "PointsList", [ent.tag])
            elif ent.dim == 1:
                gmsh.model.mesh.field.setNumbers(dist_id, "CurvesList", [ent.tag])
            elif ent.dim == 2:
                gmsh.model.mesh.field.setNumbers(dist_id, "SurfacesList", [ent.tag])
            elif ent.dim == 3:
                # Volumes: refine across their boundary surfaces
                boundary = gmsh.model.getBoundary([(3, ent.tag)], oriented=False)
                surf_tags = [t for d, t in boundary if d == 2]
                if not surf_tags:
                    continue
                gmsh.model.mesh.field.setNumbers(dist_id, "SurfacesList", surf_tags)
            else:
                continue

            thr_id = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thr_id, "InField", dist_id)
            gmsh.model.mesh.field.setNumber(thr_id, "SizeMin", ent.maxh)
            gmsh.model.mesh.field.setNumber(thr_id, "SizeMax", maxh)
            gmsh.model.mesh.field.setNumber(thr_id, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(
                thr_id, "DistMax",
                transition_distance if transition_distance is not None else 5 * ent.maxh,
            )
            threshold_field_ids.append(thr_id)

        if threshold_field_ids:
            min_id = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_id, "FieldsList", threshold_field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_id)
            # Disable competing default size sources so Field is authoritative
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # Assign physical groups by name. Group entities of the same name+dim.
        by_dim_name: dict[tuple[int, str], list[int]] = {}
        for ent in self._entities:
            if ent.name:
                by_dim_name.setdefault((ent.dim, ent.name), []).append(ent.tag)
        # Material → physical group on volumes (dim=3)
        for ent in self._entities:
            if ent.material and ent.dim == 3:
                key = (3, f"_mat_{ent.material}")
                by_dim_name.setdefault(key, []).append(ent.tag)

        name_to_tag: dict[str, int] = {}
        next_tag = 1
        for (dim, name), tags in by_dim_name.items():
            phys_tag = next_tag
            next_tag += 1
            gmsh.model.addPhysicalGroup(dim, tags, tag=phys_tag, name=name)
            display_name = name[len("_mat_"):] if name.startswith("_mat_") else name
            name_to_tag[display_name] = phys_tag

        # Generate. SaveAll=1 ensures volumes without explicit material/name still
        # land in the .msh (otherwise gmsh writes only physical-group elements).
        gmsh.option.setNumber("Mesh.MeshSizeMax", maxh)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.model.mesh.generate(3)

        # Write to a temp file, read bytes back
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            tmp_path = f.name
        try:
            gmsh.write(tmp_path)
            with open(tmp_path, "rb") as f:
                mesh_bytes = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return mesh_bytes, name_to_tag


__all__ = ["Geometry", "GeoObject", "FaceCollection"]
