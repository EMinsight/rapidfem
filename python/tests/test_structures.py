"""General RF structure builders in rapidfem.structures.

Fast checks: the builders compose valid geometry, return the documented
result fields, and (with add_ports) attach the canonical ports and still mesh.
"""
from __future__ import annotations

import pytest

import rapidfem as rf
from rapidfem import structures as st

MM = 1e-3


def test_coax_geometry_only():
    g = rf.Geometry(maxh=3 * MM)
    cx = st.coax(g, ri=1.5 * MM, ro=3.45 * MM, length=20 * MM)
    assert cx.dielectric is not None
    assert len(cx.port_a) >= 1
    assert len(cx.port_b) >= 1
    assert cx.ports == []  # no physics attached unless asked
    # Geometry-only result must still mesh.
    mesh_bytes, _ = g.mesh()
    assert len(mesh_bytes) > 0


def test_coax_add_ports_meshes():
    g = rf.Geometry(maxh=3 * MM)
    cx = st.coax(g, ri=1.5 * MM, ro=3.45 * MM, length=20 * MM, add_ports=True)
    assert len(cx.ports) == 2
    mesh_bytes, name_to_tag = g.mesh()
    assert len(mesh_bytes) > 0


def test_coax_dielectric_fill():
    g = rf.Geometry(maxh=3 * MM)
    cx = st.coax(g, ri=1.0 * MM, ro=2.3 * MM, length=10 * MM, er=2.1)
    mat = cx.dielectric.material
    assert getattr(mat, "er", None) == pytest.approx(2.1)


def test_coax_axis_x_meshes():
    g = rf.Geometry(maxh=3 * MM)
    cx = st.coax(g, ri=1.5 * MM, ro=3.45 * MM, length=15 * MM,
                 axis="x", add_ports=True)
    assert len(cx.ports) == 2
    mesh_bytes, _ = g.mesh()
    assert len(mesh_bytes) > 0


def test_coax_rejects_bad_radii():
    g = rf.Geometry(maxh=3 * MM)
    with pytest.raises(ValueError):
        st.coax(g, ri=3.0 * MM, ro=2.0 * MM, length=10 * MM)


def test_coax_rejects_bad_axis():
    g = rf.Geometry(maxh=3 * MM)
    with pytest.raises(ValueError):
        st.coax(g, ri=1.0 * MM, ro=2.0 * MM, length=10 * MM, axis="w")
