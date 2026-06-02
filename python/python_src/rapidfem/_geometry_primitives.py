"""Primitive solid and surface builders for :class:`rapidfem.geometry.Geometry`.

Split out of ``geometry.py`` as a mixin: the ``box`` / ``cylinder`` /
``sphere`` / ... constructors form a cohesive group that only needs the
scale helper and the wrap helpers (which stay on ``Geometry``). ``Geometry``
inherits :class:`_PrimitivesMixin`, so every builder below is a normal
``Geometry`` method at runtime.
"""
from __future__ import annotations

import math
import warnings

import gmsh


def _position_with_center_alias(position, center, *, what):
    """Back-compat shim: accept the legacy ``center=`` keyword for ``position=``.

    ``sphere`` and ``torus`` historically took ``center=``; every other
    primitive takes ``position=``. ``position`` is now canonical, ``center``
    stays as a deprecated alias.
    """
    if center is not None:
        warnings.warn(
            f"{what}: 'center' is deprecated, use 'position' instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return center
    return position


class _PrimitivesMixin:
    """Primitive builders, mixed into :class:`rapidfem.geometry.Geometry`."""

    def box(self, width: float, depth: float, height: float,
            position: tuple[float, float, float] = (0, 0, 0),
            *,
            material=None,
            maxh: float | None = None) -> GeoObject:
        """add an axis-aligned box primitive

        The workhorse volume primitive, used for substrates, air
        regions, waveguide cavities, and PML slabs. The returned
        :class:`GeoObject` has 6 ``.faces`` and 12 ``.edges`` selectable
        via :class:`EntityCollection`.


        Example
        -------
        .. code-block:: python

            air = g.box(22.86e-3, 10.16e-3, 30e-3,
                        position=(-11.43e-3, -5.08e-3, 0),
                        material=rf.Air())


        Parameters
        ----------
        width, depth, height : float
            extents along x, y, z respectively in metres
        position : tuple[float, float, float]
            lower corner ``(xmin, ymin, zmin)`` (defaults to origin)
        material : rapidfem.Material, optional
            volume material (``rf.Air()``, ``rf.Dielectric(er=...)``,
            ...)
        maxh : float, optional
            per-volume mesh size override in metres

        Returns
        -------
        GeoObject
            volume with 6 ``.faces`` and 12 ``.edges``
        """
        x, y, z = position
        s = self._s
        tag = gmsh.model.occ.addBox(s(x), s(y), s(z), s(width), s(depth), s(height))
        return self._wrap_volume(tag, material=material, maxh=maxh)

    def cylinder(self, radius: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0),
                 axis: tuple[float, float, float] = (0, 0, 1),
                 angle: float = 2 * math.pi,
                 *,
                 material=None,
                 maxh: float | None = None) -> GeoObject:
        """add a (partial-sweep) cylinder primitive

        Curved surfaces honour ``Mesh.MeshSizeFromCurvature`` so the
        cylinder side wall meshes into geometry-accurate facets without
        manual refinement.


        Example
        -------
        Outer dielectric of a coax line:

        .. code-block:: python

            air = g.cylinder(radius=ro, height=L,
                             position=(0, 0, 0),
                             material=rf.Air())


        Parameters
        ----------
        radius : float
            cylinder radius in metres
        height : float
            extent along ``axis``
        position : tuple[float, float, float]
            base centre (defaults to origin)
        axis : tuple[float, float, float]
            cylinder axis direction (defaults to +z)
        angle : float
            sweep angle in radians; defaults to :math:`2\\pi` (full
            cylinder), :math:`<2\\pi` gives a partial cylinder
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override

        Returns
        -------
        GeoObject
            volume
        """
        x, y, z = position
        ax, ay, az = (axis[0] * height, axis[1] * height, axis[2] * height)
        s = self._s
        tag = gmsh.model.occ.addCylinder(s(x), s(y), s(z),
                                         s(ax), s(ay), s(az), s(radius), angle=angle)
        return self._wrap_volume(tag, material=material, maxh=maxh)

    def sphere(self, radius: float, position: tuple[float, float, float] = (0, 0, 0),
               *,
               material=None,
               maxh: float | None = None,
               center: tuple[float, float, float] | None = None) -> GeoObject:
        """add a sphere primitive

        Parameters
        ----------
        radius : float
            sphere radius in metres
        position : tuple[float, float, float]
            sphere centre (defaults to origin)
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override
        center : tuple[float, float, float], optional
            deprecated alias for ``position``

        Returns
        -------
        GeoObject
            volume
        """
        position = _position_with_center_alias(position, center, what="sphere()")
        cx, cy, cz = position
        s = self._s
        tag = gmsh.model.occ.addSphere(s(cx), s(cy), s(cz), s(radius))
        return self._wrap_volume(tag, material=material, maxh=maxh)

    def cone(self, r1: float, r2: float, height: float,
             position: tuple[float, float, float] = (0, 0, 0),
             axis: tuple[float, float, float] = (0, 0, 1),
             angle: float = 2 * math.pi,
             *,
             material=None,
             maxh: float | None = None) -> GeoObject:
        """add a truncated cone (or cylinder if ``r1 == r2``)

        Parameters
        ----------
        r1, r2 : float
            base and top radii in metres
        height : float
            extent along ``axis``
        position : tuple[float, float, float]
            base centre (defaults to origin)
        axis : tuple[float, float, float]
            cone axis direction (defaults to +z)
        angle : float
            sweep angle in radians (defaults to :math:`2\\pi`)
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override

        Returns
        -------
        GeoObject
            volume
        """
        x, y, z = position
        ax, ay, az = (axis[0] * height, axis[1] * height, axis[2] * height)
        s = self._s
        tag = gmsh.model.occ.addCone(s(x), s(y), s(z),
                                     s(ax), s(ay), s(az), s(r1), s(r2), angle=angle)
        return self._wrap_volume(tag, material=material, maxh=maxh)

    def wedge(self, dx: float, dy: float, dz: float,
              top_x: float = 0.0,
              position: tuple[float, float, float] = (0, 0, 0),
              *,
              material=None,
              maxh: float | None = None) -> GeoObject:
        """add a rectangular-base prism (wedge)

        The base is ``dx × dy`` at z = 0; the top edge runs from x = 0
        to x = ``top_x`` at height ``dz``, parallel to y. Useful for
        symmetric horn walls and tapered ridge waveguides.


        Parameters
        ----------
        dx, dy, dz : float
            base width, base depth, height in metres
        top_x : float
            x-extent of the top edge; ``0`` gives a triangular wedge,
            ``dx`` an ordinary box
        position : tuple[float, float, float]
            lower-left corner of the base (defaults to origin)
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override

        Returns
        -------
        GeoObject
            volume
        """
        x, y, z = position
        s = self._s
        tag = gmsh.model.occ.addWedge(s(x), s(y), s(z), s(dx), s(dy), s(dz), ltx=s(top_x))
        return self._wrap_volume(tag, material=material, maxh=maxh)

    def torus(self, major_radius: float, minor_radius: float,
              position: tuple[float, float, float] = (0, 0, 0),
              angle: float = 2 * math.pi,
              *,
              material=None,
              maxh: float | None = None,
              center: tuple[float, float, float] | None = None) -> GeoObject:
        """add a torus primitive

        Parameters
        ----------
        major_radius : float
            donut radius (tube-centre to torus-axis distance) in metres
        minor_radius : float
            tube radius in metres
        position : tuple[float, float, float]
            torus centre (defaults to origin); axis is along +z
        angle : float
            sweep angle in radians; :math:`<2\\pi` gives a partial torus
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override
        center : tuple[float, float, float], optional
            deprecated alias for ``position``

        Returns
        -------
        GeoObject
            volume
        """
        position = _position_with_center_alias(position, center, what="torus()")
        cx, cy, cz = position
        s = self._s
        tag = gmsh.model.occ.addTorus(s(cx), s(cy), s(cz), s(major_radius), s(minor_radius),
                                      angle=angle)
        return self._wrap_volume(tag, material=material, maxh=maxh)

    def xy_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0),
                 *,
                 maxh: float | None = None) -> GeoObject:
        """add a thin rectangular plate in the xy-plane

        2-D primitive, used for thin conductors like patch antennas,
        microstrip traces, and lumped-port footprints. The returned
        object carries dim = 2 and a single ``.faces`` selector that
        points at itself.


        Note
        ----
        ``height`` here is the y-extent, *not* a vertical (z) extent.
        For an arbitrarily oriented plate (e.g. a vertical feed sheet)
        use :meth:`plate` with explicit width/height vectors.


        Example
        -------
        A patch antenna on top of a substrate:

        .. code-block:: python

            patch = g.xy_plate(38e-3, 29e-3,
                               position=(-19e-3, -14.5e-3, SUB_H))
            rf.PEC(patch)


        Parameters
        ----------
        width : float
            x-extent in metres
        height : float
            y-extent in metres
        position : tuple[float, float, float]
            lower corner (defaults to origin)
        maxh : float, optional
            per-plate mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        x, y, z = position
        s = self._s
        tag = gmsh.model.occ.addRectangle(
            s(x), s(y), s(z), s(width), s(height)
        )
        return self._wrap_face(tag, maxh=maxh)

    def xz_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0),
                 *,
                 maxh: float | None = None) -> GeoObject:
        """add a thin rectangular plate in the xz-plane

        Convenience wrapper around :meth:`plate` for the most common
        axis-aligned vertical plate. ``width`` runs along x,
        ``height`` along z.


        Parameters
        ----------
        width : float
            x-extent in metres
        height : float
            z-extent in metres
        position : tuple[float, float, float]
            lower corner (defaults to origin)
        maxh : float, optional
            per-plate mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        return self.plate(p0=position, width=(width, 0, 0), height=(0, 0, height), maxh=maxh)

    def yz_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0),
                 *,
                 maxh: float | None = None) -> GeoObject:
        """add a thin rectangular plate in the yz-plane

        Convenience wrapper around :meth:`plate` for the most common
        axis-aligned vertical plate. ``width`` runs along y,
        ``height`` along z.


        Parameters
        ----------
        width : float
            y-extent in metres
        height : float
            z-extent in metres
        position : tuple[float, float, float]
            lower corner (defaults to origin)
        maxh : float, optional
            per-plate mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        return self.plate(p0=position, width=(0, width, 0), height=(0, 0, height), maxh=maxh)

    def plate(self, p0: tuple[float, float, float],
              width: tuple[float, float, float],
              height: tuple[float, float, float],
              *,
              maxh: float | None = None) -> GeoObject:
        """add a thin rectangular plate at arbitrary orientation

        Used for vertical lumped-port sheets, oblique feed plates, and
        any flat 2-D region whose sides are not axis-aligned.


        Note
        ----
        gmsh OCC has no direct arbitrary-rectangle API; we build a
        four-vertex wire and plane surface internally. The four edge
        vectors ``width`` and ``height`` should be orthogonal, if
        they are not, you get a planar parallelogram, not a rectangle.


        Example
        -------
        Vertical lumped-port plate bridging substrate to a trace:

        .. code-block:: python

            port = g.plate(
                p0=(FEED_X - W/2, FEED_Y, 0),
                width=(W, 0, 0),
                height=(0, 0, SUB_H),
            )


        Parameters
        ----------
        p0 : tuple[float, float, float]
            one corner of the rectangle
        width : tuple[float, float, float]
            edge vector from ``p0`` defining one side
        height : tuple[float, float, float]
            edge vector from ``p0`` defining the perpendicular side
        maxh : float, optional
            per-plate mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        x0, y0, z0 = p0
        wx, wy, wz = width
        hx, hy, hz = height
        s = self._s
        v1 = gmsh.model.occ.addPoint(s(x0), s(y0), s(z0))
        v2 = gmsh.model.occ.addPoint(s(x0 + wx), s(y0 + wy), s(z0 + wz))
        v3 = gmsh.model.occ.addPoint(s(x0 + wx + hx), s(y0 + wy + hy), s(z0 + wz + hz))
        v4 = gmsh.model.occ.addPoint(s(x0 + hx), s(y0 + hy), s(z0 + hz))
        l1 = gmsh.model.occ.addLine(v1, v2)
        l2 = gmsh.model.occ.addLine(v2, v3)
        l3 = gmsh.model.occ.addLine(v3, v4)
        l4 = gmsh.model.occ.addLine(v4, v1)
        loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        tag = gmsh.model.occ.addPlaneSurface([loop])
        return self._wrap_face(tag, maxh=maxh)

    def polygon(self, points: Iterable[tuple[float, ...]],
                position: tuple[float, float, float] = (0, 0, 0),
                *,
                holes: "list[list[tuple]] | None" = None,
                maxh: float | None = None) -> GeoObject:
        """add a planar polygon face

        2-D primitive for arbitrary outlines, combine with
        :meth:`extrude` for a non-axis-aligned trace, :meth:`revolve`
        for an axisymmetric solid, or :meth:`loft` to bridge two
        profiles into a horn-style frustum.


        Note
        ----
        2-tuple vertices are placed in the xy-plane at ``z = 0`` plus
        the ``position`` offset; 3-tuple vertices must all be coplanar,
        gmsh OCC errors on non-planar input.


        Example
        -------
        Rectangular waveguide aperture (yz-plane at ``x = L``) for a
        horn loft:

        .. code-block:: python

            aperture = g.polygon([
                (L, -W/2, -H/2), (L,  W/2, -H/2),
                (L,  W/2,  H/2), (L, -W/2,  H/2),
            ])


        Parameters
        ----------
        points : iterable of (x, y) or (x, y, z) tuples
            vertices in CCW order; polygon closes automatically
        position : tuple[float, float, float]
            offset added to every vertex (defaults to origin)
        maxh : float, optional
            per-face mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        pts = list(points)
        if len(pts) < 3:
            raise ValueError("polygon needs at least 3 vertices")
        x0, y0, z0 = position
        s = self._s

        def _build_loop(loop_pts):
            vtags = []
            for p in loop_pts:
                if len(p) == 2:
                    vtags.append(gmsh.model.occ.addPoint(
                        s(p[0] + x0), s(p[1] + y0), s(z0)))
                elif len(p) == 3:
                    vtags.append(gmsh.model.occ.addPoint(
                        s(p[0] + x0), s(p[1] + y0), s(p[2] + z0)))
                else:
                    raise ValueError(
                        f"polygon point must be (x,y) or (x,y,z), got {p!r}")
            n = len(vtags)
            lt = [gmsh.model.occ.addLine(vtags[i], vtags[(i + 1) % n])
                  for i in range(n)]
            return gmsh.model.occ.addCurveLoop(lt)

        outer_loop = _build_loop(pts)
        if not holes:
            tag = gmsh.model.occ.addPlaneSurface([outer_loop])
            return self._wrap_face(tag, maxh=maxh)

        # For polygon-with-holes we go through Boolean cut: build the outer
        # disc and each hole as separate plane surfaces, then subtract.
        # gmsh's multi-loop addPlaneSurface form is brittle when followed by
        # extrude (PLC errors at facet intersections); cut delivers a clean
        # BREP that meshes reliably.
        outer_surf = gmsh.model.occ.addPlaneSurface([outer_loop])
        hole_surfs = []
        for h in holes:
            hp = list(h)
            if len(hp) < 3:
                continue
            hl = _build_loop(hp)
            hole_surfs.append(gmsh.model.occ.addPlaneSurface([hl]))
        if not hole_surfs:
            return self._wrap_face(outer_surf, maxh=maxh)
        out, _ = gmsh.model.occ.cut(
            [(2, outer_surf)],
            [(2, hs) for hs in hole_surfs],
        )
        gmsh.model.occ.synchronize()
        result_tag = next((t for d_, t in out if d_ == 2), None)
        if result_tag is None:
            raise RuntimeError("polygon-with-holes cut produced no surface")
        return self._wrap_face(result_tag, maxh=maxh)

    def disc(self, radius: float,
             position: tuple[float, float, float] = (0, 0, 0),
             *,
             axis: tuple[float, float, float] = (0, 0, 1),
             maxh: float | None = None) -> GeoObject:
        """add a circular face with an arbitrary normal

        Smooth NURBS circle (gmsh OCC ``addDisk``), meshes into curved
        triangles when ``MeshSizeFromCurvature`` is active. Pair with
        :meth:`extrude` for a circular post or :meth:`revolve` for a
        spherical cap.


        Parameters
        ----------
        radius : float
            disc radius in metres
        position : tuple[float, float, float]
            disc centre (defaults to origin)
        axis : tuple[float, float, float]
            disc normal (defaults to +z, i.e. the xy-plane). Any direction
            is allowed, e.g. ``(1, 0, 0)`` puts the disc in the yz-plane.
            Need not be unit length; gmsh normalises it.
        maxh : float, optional
            per-face mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        x, y, z = position
        s = self._s
        # gmsh treats an empty zAxis as the default +z (xy-plane); only pass a
        # normal when it actually differs, to avoid disturbing existing calls.
        z_axis = ([float(axis[0]), float(axis[1]), float(axis[2])]
                  if tuple(axis) != (0, 0, 1) else [])
        tag = gmsh.model.occ.addDisk(s(x), s(y), s(z), s(radius), s(radius),
                                     zAxis=z_axis)
        return self._wrap_face(tag, maxh=maxh)
