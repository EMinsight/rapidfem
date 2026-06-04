"""Geometry builder ops: translate / mirror / copy / array / fillet / chamfer.

Fast checks (no FEM solve): assert the OCC calls behave (centroids move
as requested, copies are independent, arrays are evenly placed) and that
filleted / chamfered bodies still mesh.
"""
from __future__ import annotations

import math

import pytest

import rapidfem as rf

MM = 1e-3


def test_translate_moves_centroid():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM)
    x0, y0, z0 = b._entity.cog
    g.translate(b, dx=2 * MM, dz=-1 * MM)
    x1, y1, z1 = b._entity.cog
    assert x1 - x0 == pytest.approx(2 * MM)
    assert y1 - y0 == pytest.approx(0.0, abs=1e-12)
    assert z1 - z0 == pytest.approx(-1 * MM)


def test_mirror_reflects_across_plane():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM, position=(1 * MM, 0, 0))
    x0 = b._entity.cog[0]
    g.mirror(b, normal=(1, 0, 0))
    assert b._entity.cog[0] == pytest.approx(-x0)


def test_copy_is_independent_and_inherits_material():
    g = rf.Geometry(maxh=5 * MM)
    mat = rf.Dielectric(er=4.4)
    b = g.box(1 * MM, 1 * MM, 1 * MM, material=mat)
    d = g.copy(b)
    assert d._entity.tag != b._entity.tag
    assert d.material is mat
    # Moving the copy must not disturb the source.
    src_x = b._entity.cog[0]
    g.translate(d, dx=2 * MM)
    assert b._entity.cog[0] == pytest.approx(src_x)


def test_copy_does_not_inherit_name():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM)
    b.name = "src"
    d = g.copy(b)
    assert d.name is None


def test_array_linear_even_spacing():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM)
    cells = g.array(b, 4, spacing=(2 * MM, 0, 0))
    assert len(cells) == 4
    assert cells[0] is b
    xs = sorted(c._entity.cog[0] for c in cells)
    deltas = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    for d in deltas:
        assert d == pytest.approx(2 * MM)


def test_array_polar_constant_radius():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM, position=(5 * MM, -0.5 * MM, -0.5 * MM))
    petals = g.array(b, 6, rotation=2 * math.pi / 6)
    assert len(petals) == 6
    radii = [math.hypot(p._entity.cog[0], p._entity.cog[1]) for p in petals]
    for r in radii:
        assert r == pytest.approx(radii[0], abs=1e-6)


def test_array_requires_exactly_one_mode():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM)
    with pytest.raises(ValueError):
        g.array(b, 3)
    with pytest.raises(ValueError):
        g.array(b, 3, spacing=(1 * MM, 0, 0), rotation=0.1)
    with pytest.raises(ValueError):
        g.array(b, 0, spacing=(1 * MM, 0, 0))


def test_fillet_meshes():
    g = rf.Geometry(maxh=2 * MM)
    b = g.box(4 * MM, 4 * MM, 4 * MM, material=rf.Air())
    g.fillet(b, 0.5 * MM)
    mesh_bytes, _ = g.mesh()
    assert len(mesh_bytes) > 0


def test_chamfer_meshes():
    g = rf.Geometry(maxh=2 * MM)
    b = g.box(4 * MM, 4 * MM, 4 * MM, material=rf.Air())
    g.chamfer(b, 0.4 * MM)
    mesh_bytes, _ = g.mesh()
    assert len(mesh_bytes) > 0


def test_fillet_rejects_face():
    g = rf.Geometry(maxh=5 * MM)
    f = g.xy_plate(1 * MM, 1 * MM)
    with pytest.raises(ValueError):
        g.fillet(f, 1e-4)
