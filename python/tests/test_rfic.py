# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Regression tests for the rapidfem.rfic package.

The bundled ``*.fem.json`` example exports double as fixtures: every layout
that rapidpassives ships through ``exportForFEM()`` must keep building into
a valid geometry (conductors extruded, ports resolved, fragment clean).
Meshing/solving is covered by the geometry suite; here we pin the builder
contract itself, cheap enough to run on every push.
"""
import json
from importlib.resources import files

import pytest

import rapidfem.rfic as rfic

FIXTURES = [
    "fd_rfic_spiral_from_json.fem.json",
    "fd_rfic_symmetric_inductor_from_json.fem.json",
    "fd_rfic_symmetric_transformer_from_json.fem.json",
    "fd_rfic_stacked_transformer_from_json.fem.json",
    "fd_rfic_ratrace_coupler_from_json.fem.json",
]


def _fixture(name: str) -> dict:
    with (files("rapidfem.examples") / name).open() as f:
        return json.load(f)


# ── Stack model ─────────────────────────────────────────────────────────────

def test_stack_presets_resolve():
    for pdk in ("sky130", "sg13g2"):
        stack = rfic.Stack.from_pdk(pdk)
        assert stack.metals(), pdk
        assert stack.vias(), pdk
        assert stack.top_z > stack.bottom_z


def test_stack_json_roundtrip():
    a = rfic.Stack.sky130()
    b = rfic.Stack.from_dict(a.to_dict())
    assert b.name == a.name
    assert len(b.layers) == len(a.layers)
    for la, lb in zip(a.layers, b.layers):
        assert la.name == lb.name
        assert la.gds_key == lb.gds_key
        assert la.z == pytest.approx(lb.z)
        assert la.thickness == pytest.approx(lb.thickness)
        assert la.sigma == lb.sigma
    assert b.oxide_er == a.oxide_er
    assert b.substrate_sigma == a.substrate_sigma


def test_stack_lookup_errors():
    stack = rfic.Stack.sky130()
    with pytest.raises(KeyError):
        stack.by_name("nope")
    assert stack.by_gds(9999) is None


# ── FEM-JSON bridge ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", FIXTURES)
def test_from_fem_json_builds(fixture):
    doc = _fixture(fixture)
    layout = rfic.from_fem_json(doc)
    try:
        # every JSON conductor layer materialised at least one volume
        json_layers = {c["layer"] for c in doc["conductors"]
                       if any(l["thickness_um"] > 0 for l in doc["stack"]["layers"]
                              if l["id"] == c["layer"])}
        assert set(layout.conductors) == json_layers
        assert all(vols for vols in layout.conductors.values())
        # every JSON port resolved into a plate
        assert set(layout.ports) == {p["name"] for p in doc["ports"]}
        # enclosure handles exist and are 3D
        for obj in (layout.substrate, layout.oxide, layout.air):
            assert obj.dim == 3
        assert layout.doc is doc
    finally:
        layout.geometry.close()


def test_from_fem_json_rejects_unknown_schema():
    doc = _fixture(FIXTURES[0])
    doc["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        rfic.from_fem_json(doc)


def test_from_fem_json_meshes():
    """One representative layout through the full mesh path, with stats."""
    layout = rfic.from_fem_json(_fixture("fd_rfic_spiral_from_json.fem.json"))
    g = layout.geometry
    try:
        import rapidfem as rf
        all_conductors = [v for vs in layout.conductors.values() for v in vs]
        rf.PEC(*(v.faces for v in all_conductors), layout.ground)
        for port in layout.ports.values():
            rf.LumpedPort(port, direction=(0, 0, 1), z0=50.0)
        rf.ABC(*layout.air.faces.outer)
        g.mesh()
        s = g.mesh_stats
        assert s is not None
        assert s.n_tets > 0
        assert s.dofs_min == s.n_edges
        assert s.dofs_max == 2 * s.n_edges + 2 * s.n_tris
        assert any(name.startswith("port_") for name in s.groups)
    finally:
        g.close()
