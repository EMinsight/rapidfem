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

    Setting ``.name = "..."`` or ``.maxh = ...`` applies to all members.
    Selectors return *new* collections so chains compose.

    Examples
    --------
    >>> box.faces.where(lambda c, b: c[2] < 0.5).min(axis="x").name = "port"
    >>> box.edges.where(lambda c, _: c[2] == 0).maxh = 1e-3
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
        """Keep only entities whose centroid is at the minimum along ``axis``.

        Parameters
        ----------
        axis : {'x', 'y', 'z'}, optional
            Axis to compare centroids on. Default ``'z'``.

        Returns
        -------
        EntityCollection
            Subset of this collection at the min coordinate. Usually one
            face for a convex primitive.
        """
        ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        if not self._entities:
            return EntityCollection(self._geometry, [])
        m = min(e.cog[ax] for e in self._entities)
        kept = [e for e in self._entities if abs(e.cog[ax] - m) < _COG_TOL]
        return EntityCollection(self._geometry, kept)

    def max(self, axis: str = "z") -> "EntityCollection":
        """Keep only entities whose centroid is at the maximum along ``axis``.

        Mirror of :meth:`min`. See that method for parameters/returns.
        """
        ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        if not self._entities:
            return EntityCollection(self._geometry, [])
        m = max(e.cog[ax] for e in self._entities)
        kept = [e for e in self._entities if abs(e.cog[ax] - m) < _COG_TOL]
        return EntityCollection(self._geometry, kept)

    def where(self, predicate: Callable[[tuple, tuple], bool]) -> "EntityCollection":
        """Filter entities by an arbitrary predicate on centroid + bbox.

        Parameters
        ----------
        predicate : Callable[[tuple, tuple], bool]
            Function ``(centroid, bbox) -> bool``. Both arguments are
            3-tuples; ``bbox`` is ``(xmin, ymin, zmin, xmax, ymax, zmax)``.

        Returns
        -------
        EntityCollection
            Entities for which the predicate returned True.
        """
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

    Attributes
    ----------
    name : str | None
        Physical-group name the entity gets when meshed. Setting this
        makes the entity reachable through the builder's name resolver.
    material : str | None
        Material name (volume-only). Must be wired to a
        ``SimulationBuilder.material(...)`` call later.
    maxh : float | None
        Per-entity mesh size override in metres.
    dim : int
        Topological dimension: 3 for volumes, 2 for plates.
    faces : EntityCollection
        Bounding faces of a volume (or self, for a plate).
    edges : EntityCollection
        Bounding edges of the entity.

    Examples
    --------
    >>> substrate = g.box(60e-3, 60e-3, 1.6e-3)
    >>> substrate.name = "fr4_volume"
    >>> substrate.material = "fr4"
    >>> substrate.maxh = 5e-3
    >>> substrate.faces.min(axis="z").name = "ground"
    """

    def __init__(self, geometry: "Geometry", entity: _Entity):
        self._geometry = geometry
        self._entity = entity
        # Note: `.faces` and `.edges` are computed on-demand (properties) so
        # they always reflect the CURRENT gmsh topology. After a boolean op
        # splits a face into pieces, accessing `obj.faces` re-discovers them
        # all. Names set previously persist via the geometry's entity registry
        # (matched by cog+bbox).

    def _discover_subentities(self, target_dim: int) -> list[_Entity]:
        """Re-query gmsh for the current sub-entities of dimension `target_dim`.
        Existing entries in `self._geometry._entities` matching by (cog, bbox)
        keep their name/material/maxh; new entries get fresh blank metadata."""
        if self._entity.dim == 3 and target_dim == 2:
            children = [(d, t) for d, t in
                        gmsh.model.getBoundary([(3, self._entity.tag)], oriented=False)
                        if d == 2]
        elif self._entity.dim == 3 and target_dim == 1:
            children = []
            seen: set[int] = set()
            for d, t in gmsh.model.getBoundary([(3, self._entity.tag)], oriented=False):
                if d != 2:
                    continue
                for d_e, t_e in gmsh.model.getBoundary([(2, t)], oriented=False):
                    if d_e == 1 and t_e not in seen:
                        seen.add(t_e)
                        children.append((d_e, t_e))
        elif self._entity.dim == 2 and target_dim == 1:
            children = [(d, t) for d, t in
                        gmsh.model.getBoundary([(2, self._entity.tag)], oriented=False)
                        if d == 1]
        else:
            children = []

        # Build entries, reusing existing _Entity records if cog+bbox matches
        out: list[_Entity] = []
        for d, t in children:
            cog = tuple(gmsh.model.occ.getCenterOfMass(d, t))
            bbox = tuple(gmsh.model.getBoundingBox(d, t))
            existing = self._find_or_register_entity(d, t, cog, bbox)
            out.append(existing)
        return out

    def _find_or_register_entity(self, dim, tag, cog, bbox) -> _Entity:
        """Look up an existing _Entity in the geometry registry by (cog, bbox);
        register a new one if not found. Updates the tag to current."""
        for ent in self._geometry._entities:
            if ent.dim != dim:
                continue
            if math.dist(ent.cog, cog) < _COG_TOL and _bbox_match(ent.bbox, bbox, _BBOX_TOL):
                ent.tag = tag       # refresh tag (may have changed after fragment)
                return ent
        new_ent = _Entity(dim=dim, tag=tag, cog=cog, bbox=bbox)
        self._geometry._entities.append(new_ent)
        return new_ent

    @property
    def faces(self) -> EntityCollection:
        return EntityCollection(self._geometry, self._discover_subentities(2))

    @property
    def edges(self) -> EntityCollection:
        return EntityCollection(self._geometry, self._discover_subentities(1))

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
    """Top-level geometry builder. Owns a gmsh OCC model for its lifetime.

    Build with primitives (:meth:`box`, :meth:`cylinder`, ...), tag
    faces / edges / volumes with names, assign materials, then call
    :meth:`mesh` to produce the FEM mesh. Hand the meshed geometry to
    :class:`rapidfem.SimulationBuilder` to assemble a :class:`Simulation`.

    Examples
    --------
    >>> import rapidfem
    >>> g = rapidfem.Geometry()
    >>> air = g.box(22.86e-3, 10.16e-3, 30e-3,
    ...             position=(-11.43e-3, -5.08e-3, 0))
    >>> air.material = "air"
    >>> air.faces.min(axis="z").name = "port_in"
    >>> air.faces.max(axis="z").name = "port_out"
    >>> for f in air.faces:
    ...     if f.name is None:
    ...         f.name = "pec"
    >>> g.mesh(maxh=3e-3)
    """

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
        thin_conductors: bool = False,
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
            thin_conductors: If True, metal layers become 2D PEC plates at the
                layer's bottom z (thin-conductor approximation, t << w). The
                resulting plate carries the layer's name as a SURFACE physical
                group, which is what the simulator's `.pec(...)` BC expects.
                Recommended for RFIC-style metal with thicknesses ≤ wavelength/100.
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

        # Build each polygon at its layer's z. With thin_conductors=True (or
        # for non-metal types) we keep it as a 2D plate so PEC BC can attach
        # to a SURFACE group; otherwise we extrude to a 3D solid.
        per_layer_objs: dict[str, list[GeoObject]] = {}
        for layer_name, polys in per_layer.items():
            pdk = stack.by_name(layer_name)
            use_2d = thin_conductors and pdk.type == "metal"
            objs: list[GeoObject] = []
            for pts in polys:
                if use_2d:
                    obj = g._plate_polygon(pts, z=pdk.z)
                else:
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

    def _plate_polygon(self, pts: "np.ndarray", z: float) -> GeoObject:
        """Create a 2D plane surface from a closed polygon at height z.

        Used by `from_gds(thin_conductors=True)` so metals stay 2D and their
        layer-name physical group lands on FACES (PEC-compatible).
        """
        import numpy as np
        # Rectangle fast path
        if pts.shape[0] in (4, 5):
            p = pts[:4]
            xs, ys = sorted(set(p[:, 0])), sorted(set(p[:, 1]))
            if len(xs) == 2 and len(ys) == 2:
                tag = gmsh.model.occ.addRectangle(
                    xs[0], ys[0], z, xs[1] - xs[0], ys[1] - ys[0]
                )
                return self._wrap_face(tag)
        # General polygon
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        tol = 1e-9
        keep = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - keep[-1]) > tol:
                keep.append(p)
        if len(keep) > 1 and np.linalg.norm(keep[-1] - keep[0]) <= tol:
            keep = keep[:-1]
        pts = np.asarray(keep)
        if len(pts) < 3:
            raise ValueError(f"Polygon collapsed to {len(pts)} unique vertices")
        pt_tags = [gmsh.model.occ.addPoint(p[0], p[1], z) for p in pts]
        line_tags = [
            gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i + 1) % len(pt_tags)])
            for i in range(len(pt_tags))
        ]
        loop = gmsh.model.occ.addCurveLoop(line_tags)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        return self._wrap_face(surf)

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
        # Drop adjacent coincident vertices — gdstk boolean unions can leave
        # them at shared corners and they'd collapse OCC line endpoints into a
        # broken loop. Keep one of every coincident-pair (within ~1nm).
        tol = 1e-9
        keep = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - keep[-1]) > tol:
                keep.append(p)
        # Also check wraparound (last vs first)
        if len(keep) > 1 and np.linalg.norm(keep[-1] - keep[0]) <= tol:
            keep = keep[:-1]
        pts = np.asarray(keep)
        if len(pts) < 3:
            raise ValueError(f"Polygon collapsed to {len(pts)} unique vertices")
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
        """Add an axis-aligned box primitive.

        Parameters
        ----------
        width, depth, height : float
            Extents along x, y, z respectively (m).
        position : tuple[float, float, float], optional
            Lower corner ``(xmin, ymin, zmin)``. Default origin.

        Returns
        -------
        GeoObject
            Volume with 6 ``.faces`` and 12 ``.edges``.
        """
        x, y, z = position
        tag = gmsh.model.occ.addBox(x, y, z, width, depth, height)
        return self._wrap_volume(tag)

    def cylinder(self, radius: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0),
                 axis: tuple[float, float, float] = (0, 0, 1),
                 angle: float = 2 * math.pi) -> GeoObject:
        """Add a (partial-sweep) cylinder primitive.

        Parameters
        ----------
        radius : float
            Cylinder radius in metres.
        height : float
            Extent along ``axis``.
        position : tuple[float, float, float], optional
            Base centre. Default origin.
        axis : tuple[float, float, float], optional
            Cylinder axis direction. Default +z.
        angle : float, optional
            Sweep angle in radians. Default 2π (full cylinder).

        Returns
        -------
        GeoObject
            Volume.
        """
        x, y, z = position
        ax, ay, az = (axis[0] * height, axis[1] * height, axis[2] * height)
        tag = gmsh.model.occ.addCylinder(x, y, z, ax, ay, az, radius, angle=angle)
        return self._wrap_volume(tag)

    def sphere(self, radius: float, center: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        """Add a sphere primitive.

        Parameters
        ----------
        radius : float
            Sphere radius in metres.
        center : tuple[float, float, float], optional
            Sphere centre. Default origin.

        Returns
        -------
        GeoObject
            Volume.
        """
        cx, cy, cz = center
        tag = gmsh.model.occ.addSphere(cx, cy, cz, radius)
        return self._wrap_volume(tag)

    def cone(self, r1: float, r2: float, height: float,
             position: tuple[float, float, float] = (0, 0, 0),
             axis: tuple[float, float, float] = (0, 0, 1),
             angle: float = 2 * math.pi) -> GeoObject:
        """Add a truncated cone (or cylinder if ``r1 == r2``).

        Parameters
        ----------
        r1, r2 : float
            Base and top radii in metres.
        height : float
            Extent along ``axis``.
        position : tuple[float, float, float], optional
            Base centre. Default origin.
        axis : tuple[float, float, float], optional
            Cone axis direction. Default +z.
        angle : float, optional
            Sweep angle in radians. Default 2π.

        Returns
        -------
        GeoObject
            Volume.
        """
        x, y, z = position
        ax, ay, az = (axis[0] * height, axis[1] * height, axis[2] * height)
        tag = gmsh.model.occ.addCone(x, y, z, ax, ay, az, r1, r2, angle=angle)
        return self._wrap_volume(tag)

    def wedge(self, dx: float, dy: float, dz: float,
              top_x: float = 0.0,
              position: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        """Add a rectangular-base prism (wedge).

        The base is ``dx × dy`` at z = 0; the top edge runs from x = 0 to
        x = ``top_x`` at height ``dz``, parallel to y.

        Parameters
        ----------
        dx, dy, dz : float
            Base width, base depth, height in metres.
        top_x : float, optional
            x-extent of the top edge. ``0`` = triangular wedge; ``dx`` =
            ordinary box. Default 0.
        position : tuple[float, float, float], optional
            Lower-left corner of the base. Default origin.

        Returns
        -------
        GeoObject
            Volume.
        """
        x, y, z = position
        tag = gmsh.model.occ.addWedge(x, y, z, dx, dy, dz, ltx=top_x)
        return self._wrap_volume(tag)

    def torus(self, major_radius: float, minor_radius: float,
              center: tuple[float, float, float] = (0, 0, 0),
              angle: float = 2 * math.pi) -> GeoObject:
        """Add a torus primitive.

        Parameters
        ----------
        major_radius : float
            Donut radius (centre of the tube to torus axis).
        minor_radius : float
            Tube radius.
        center : tuple[float, float, float], optional
            Torus centre. Default origin. Axis is along +z.
        angle : float, optional
            Sweep angle in radians. ``< 2π`` gives a partial torus.

        Returns
        -------
        GeoObject
            Volume.
        """
        cx, cy, cz = center
        tag = gmsh.model.occ.addTorus(cx, cy, cz, major_radius, minor_radius, angle=angle)
        return self._wrap_volume(tag)

    def xy_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        """Add a thin rectangular plate in the xy-plane.

        Parameters
        ----------
        width : float
            x-extent in metres.
        height : float
            y-extent in metres (note: not a vertical extent).
        position : tuple[float, float, float], optional
            Lower corner. Default origin.

        Returns
        -------
        GeoObject
            2D face (dim=2). Typically used for thin conductors like
            patch antennas or microstrip traces.
        """
        x, y, z = position
        tag = gmsh.model.occ.addRectangle(x, y, z, width, height)
        return self._wrap_face(tag)

    def xz_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        """Add a thin rectangular plate in the xz-plane.

        See :meth:`xy_plate`. ``width`` runs along x, ``height`` along z.
        """
        return self.plate(p0=position, width=(width, 0, 0), height=(0, 0, height))

    def yz_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        """Add a thin rectangular plate in the yz-plane.

        See :meth:`xy_plate`. ``width`` runs along y, ``height`` along z.
        """
        return self.plate(p0=position, width=(0, width, 0), height=(0, 0, height))

    def plate(self, p0: tuple[float, float, float],
              width: tuple[float, float, float],
              height: tuple[float, float, float]) -> GeoObject:
        """Add a thin rectangular plate at arbitrary orientation.

        Parameters
        ----------
        p0 : tuple[float, float, float]
            One corner of the rectangle.
        width : tuple[float, float, float]
            Edge vector from ``p0`` defining one side.
        height : tuple[float, float, float]
            Edge vector from ``p0`` defining the perpendicular side.

        Returns
        -------
        GeoObject
            2D face. Used for vertical lumped-port sheets, oblique feed
            plates, etc.

        Notes
        -----
        gmsh OCC has no direct arbitrary-rectangle API; we build a
        four-vertex wire and plane-surface internally.
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
        """Boolean fragment: make all overlaps conformal.

        Splits each overlap into a shared face / volume that both
        operands keep, so meshing produces a single mesh across the
        interfaces. Names assigned on the operands survive.

        Parameters
        ----------
        target : GeoObject
            First operand.
        *tools : GeoObject
            Additional operands to fragment with ``target``.

        Notes
        -----
        Uses gmsh's ``occ.fragment`` ``out_map`` to update each input's
        tag directly — robust against the COG drift that can break naive
        name-tracking after boolean ops. Child faces / edges are
        re-resolved by ``(centroid, bbox)`` matching.

        Examples
        --------
        >>> g.fragment(air, substrate, patch, feed)
        """
        target_dt = [(target.dim, target._entity.tag)]
        tools_dt = [(t.dim, t._entity.tag) for t in tools]
        _, out_map = gmsh.model.occ.fragment(target_dt, tools_dt)
        gmsh.model.occ.synchronize()
        inputs = [target] + list(tools)
        self._apply_out_map(inputs, out_map)
        self._reresolve_children(top_level=set(id(o._entity) for o in inputs))

    def cut(self, target: GeoObject, *tools: GeoObject) -> None:
        """Boolean subtract ``tools`` from ``target``.

        Parameters
        ----------
        target : GeoObject
            Object to subtract from. Survives (possibly as several pieces).
        *tools : GeoObject
            Objects to subtract. **Consumed** by the operation — do not
            reference them afterwards.
        """
        target_dt = [(target.dim, target._entity.tag)]
        tools_dt = [(t.dim, t._entity.tag) for t in tools]
        _, out_map = gmsh.model.occ.cut(target_dt, tools_dt)
        gmsh.model.occ.synchronize()
        # Tools are consumed by `cut`; only the target survives (with possibly
        # multiple pieces). out_map[0] = target's new pieces.
        self._apply_out_map([target], out_map[:1] if out_map else [[]])
        self._reresolve_children(top_level={id(target._entity)})

    def _apply_out_map(self, inputs: list[GeoObject], out_map: list) -> None:
        """For each input GeoObject, update its tag/cog/bbox from gmsh's out_map.
        If an input was split into multiple pieces, the first piece keeps the
        original GeoObject; the others are registered as additional `_Entity`s
        carrying the same name/material/maxh.
        """
        for input_obj, new_dimtags in zip(inputs, out_map):
            if not new_dimtags:
                warnings.warn(
                    f"GeoObject (dim={input_obj.dim}, name={input_obj._entity.name!r}) "
                    f"vanished during boolean op",
                    stacklevel=3,
                )
                continue
            d0, t0 = new_dimtags[0]
            input_obj._entity.tag = t0
            input_obj._entity.cog = tuple(gmsh.model.occ.getCenterOfMass(d0, t0))
            input_obj._entity.bbox = tuple(gmsh.model.getBoundingBox(d0, t0))
            for d, t in new_dimtags[1:]:
                extra = _Entity.from_dimtag(d, t)
                extra.name = input_obj._entity.name
                extra.material = input_obj._entity.material
                extra.maxh = input_obj._entity.maxh
                self._entities.append(extra)

    def _reresolve_children(self, top_level: set[int]) -> None:
        """Re-resolve child entities (faces, edges) by COG+bbox. Top-level
        entities (already updated via out_map) are skipped via `top_level` set
        of `_Entity` ids."""
        survived = []
        for ent in self._entities:
            if id(ent) in top_level:
                survived.append(ent)
                continue
            new_tag = _resolve_entity(ent)
            if new_tag is not None:
                ent.tag = new_tag
                survived.append(ent)
            elif ent.name or ent.material or ent.maxh:
                warnings.warn(
                    f"Tracked entity (dim={ent.dim}, name={ent.name!r}, "
                    f"cog={ent.cog}) lost during boolean op; attributes dropped.",
                    stacklevel=3,
                )
        self._entities = survived

    def fuse(self, target: GeoObject, *tools: GeoObject) -> None:
        """Boolean union ``target ∪ tools``.

        Parameters
        ----------
        target : GeoObject
            First operand. Survives as the merged object.
        *tools : GeoObject
            Operands to merge in.

        Warnings
        --------
        Face names on the operands are NOT preserved (faces merge and
        centroids shift). Top-level volume names survive via the gmsh
        out_map, but **set face names AFTER fuse**, or use
        :meth:`fragment` if interface preservation matters.
        """
        warnings.warn(
            "fuse() merges faces and shifts their COGs; named faces on the "
            "operands will not be reliably re-resolvable. Set face names AFTER "
            "fuse, or use fragment() if you need them preserved.",
            stacklevel=2,
        )
        target_dt = [(target.dim, target._entity.tag)]
        tools_dt = [(t.dim, t._entity.tag) for t in tools]
        _, out_map = gmsh.model.occ.fuse(target_dt, tools_dt)
        gmsh.model.occ.synchronize()
        inputs = [target] + list(tools)
        self._apply_out_map(inputs, out_map)
        self._reresolve_children(top_level=set(id(o._entity) for o in inputs))

    # ── Mesh emit ───────────────────────────────────────────────────────────

    def mesh(self, maxh: float = 1.0, transition_distance: float | None = None) -> tuple[bytes, dict[str, int]]:
        """Generate the 3D tet mesh of the current geometry.

        Calls gmsh's OCC mesher with the configured per-entity sizes
        and global cap. Per-entity ``obj.maxh = h`` is honoured via
        gmsh ``Distance + Threshold`` background fields so refinement
        transitions are smooth, not abrupt.

        Parameters
        ----------
        maxh : float, optional
            Global maximum tet edge length in metres. Default 1.0
            (always pass a real value).
        transition_distance : float, optional
            Distance over which a refined region's element size grows
            from its local ``h`` to the global cap. Default ``5 · h``
            per-entity.

        Returns
        -------
        mesh_bytes : bytes
            gmsh ``.msh`` v4 file as bytes — feed to
            :meth:`SimulationBuilder.mesh` or use ``.mesh_from(g)`` to
            pick it up from the geometry's cache.
        name_to_tag : dict[str, int]
            Map from every user-supplied name (face / edge / volume,
            and ``material=``) to its physical-group integer tag.

        Notes
        -----
        Side effect: caches ``(mesh_bytes, name_to_tag)`` on
        ``self._last_mesh`` so ``SimulationBuilder.mesh_from(g)`` can
        find it without re-meshing.
        """
        gmsh.model.occ.synchronize()
        # Wipe any prior mesh state AND physical groups. Without the latter,
        # re-running this cell hits "Physical surface 1 already exists".
        # Without the former, gmsh reuses stale 1D/2D meshes and partially
        # ignores the new maxh.
        try:
            gmsh.model.mesh.clear()
        except Exception:
            pass
        try:
            for dim, ptag in gmsh.model.getPhysicalGroups():
                gmsh.model.removePhysicalGroups([(dim, ptag)])
        except Exception:
            pass

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
            # When threshold fields are active the user explicitly cares about
            # local size — keep `ExtendFromBoundary` off so global Max applies
            # away from refined regions, but leave Curvature on (combined via
            # Min) so curved features get resolved cleanly even if the user
            # only set per-volume sizes.
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        # Curvature-based sizing: gmsh disables this by default. Turning it
        # on gives curved primitives (cylinder, sphere, cone, torus) a
        # geometry-accurate facet count without the user having to refine
        # those surfaces by hand. Value = target elements per 2π radians.
        # 12 is a reasonable balance between fidelity and DoF count for
        # second-kind Nédélec-2 (high-order absorbs some discretisation
        # error already). User can override before calling .mesh().
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)

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

        self._last_mesh = (mesh_bytes, name_to_tag)
        return mesh_bytes, name_to_tag


__all__ = ["Geometry", "GeoObject", "FaceCollection"]
