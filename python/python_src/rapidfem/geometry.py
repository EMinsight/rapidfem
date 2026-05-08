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
# Face collection with selectors and attribute writes
# ─────────────────────────────────────────────────────────────────────────────

class FaceCollection:
    """Collection of faces with selectors and bulk attribute writes.

    Setting `.name = "..."` or `.maxh = ...` applies to all faces in the
    collection. Selectors return *new* FaceCollections, so chaining composes:

        box.faces.where(lambda c, b: c[2] < 0.5).min(axis="x").name = "port"
    """

    def __init__(self, geometry: "Geometry", entities: list[_Entity]):
        self._geometry = geometry
        self._entities = entities

    def __iter__(self):
        return iter(self._entities)

    def __len__(self):
        return len(self._entities)

    # Selectors
    def min(self, axis: str = "z") -> "FaceCollection":
        ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        if not self._entities:
            return FaceCollection(self._geometry, [])
        m = min(e.cog[ax] for e in self._entities)
        kept = [e for e in self._entities if abs(e.cog[ax] - m) < _COG_TOL]
        return FaceCollection(self._geometry, kept)

    def max(self, axis: str = "z") -> "FaceCollection":
        ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        if not self._entities:
            return FaceCollection(self._geometry, [])
        m = max(e.cog[ax] for e in self._entities)
        kept = [e for e in self._entities if abs(e.cog[ax] - m) < _COG_TOL]
        return FaceCollection(self._geometry, kept)

    def where(self, predicate: Callable[[tuple, tuple], bool]) -> "FaceCollection":
        kept = [e for e in self._entities if predicate(e.cog, e.bbox)]
        return FaceCollection(self._geometry, kept)

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
        # Build face collection from the entity's boundary
        if entity.dim == 3:
            self.faces = FaceCollection(
                geometry,
                [
                    _Entity.from_dimtag(d, t)
                    for d, t in gmsh.model.getBoundary([(3, entity.tag)], oriented=False)
                    if d == 2
                ],
            )
            # register all faces too
            for fe in self.faces._entities:
                geometry._entities.append(fe)
        else:
            # 2D plate has no faces collection; itself IS the surface
            self.faces = FaceCollection(geometry, [])

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
        """Axis-aligned box. `position` = lower-corner."""
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

    def xy_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0)) -> GeoObject:
        """Rectangle in the xy-plane (constant z)."""
        x, y, z = position
        tag = gmsh.model.occ.addRectangle(x, y, z, width, height)
        return self._wrap_face(tag)

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

    def mesh(self, maxh: float = 1.0) -> tuple[bytes, dict[str, int]]:
        """Generate the 3D mesh and return (msh_bytes, name_to_tag).

        `name_to_tag` maps each user-supplied name to its assigned integer
        physical-group tag. Pass `msh_bytes` to `Simulation.from_bytes` and
        the mapping to `SimulationBuilder` to wire ports/materials by name.
        """
        gmsh.model.occ.synchronize()

        # Apply per-entity mesh size hints by setting size at boundary points
        for ent in self._entities:
            if ent.maxh is None:
                continue
            try:
                # set size at all points adjacent to this entity
                pts = gmsh.model.getBoundary(
                    [(ent.dim, ent.tag)], oriented=False, recursive=True,
                )
                pt_dt = [(d, t) for d, t in pts if d == 0]
                if pt_dt:
                    gmsh.model.mesh.setSize(pt_dt, ent.maxh)
            except Exception:
                pass

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

        # Generate
        gmsh.option.setNumber("Mesh.MeshSizeMax", maxh)
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
