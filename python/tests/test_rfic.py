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


def test_stack_from_xml_sg13g2():
    """The bundled IHP XML must reproduce the process-spec values that the
    SG13G2 measurement validation (muehlhaus) pinned down."""
    um = 1e-6
    s = rfic.Stack.sg13g2()

    m1 = s.by_name("Metal1")
    assert m1.z == pytest.approx(1.04 * um)
    assert m1.z_top == pytest.approx(1.46 * um)
    assert m1.sigma == 2.164e7

    tm2 = s.by_name("TopMetal2")
    assert tm2.z == pytest.approx(11.2303 * um)
    assert tm2.thickness == pytest.approx(3.0 * um)
    assert tm2.sigma == 3.03e7

    assert s.by_name("TopVia2").sigma == 3.143e6
    assert s.by_name("SUBGND").is_pec          # LOWLOSS 1e10 convention

    # background dielectric stack, anchored by the Substrate Offset
    assert [d.name for d in s.dielectrics] == [
        "Substrate", "EPI", "SiO2", "Passive", "AIR"]
    assert s.dielectric_at(-1 * um).name == "EPI"
    sio2 = s.dielectric_at(5 * um)
    assert sio2.z_top == pytest.approx(15.7303 * um)
    passive = s.dielectric_at(15.8 * um)
    assert s.materials[passive.material].er == 6.6

    # legacy scalars derived from the dielectric stack
    assert s.substrate_er == 11.9
    assert s.substrate_sigma == 2.0
    assert s.oxide_er == 4.1


def test_stack_from_xml_roundtrips_dielectrics():
    s = rfic.Stack.sg13g2()
    s2 = rfic.Stack.from_dict(s.to_dict())
    assert len(s2.dielectrics) == len(s.dielectrics)
    assert len(s2.layers) == len(s.layers)
    assert s2.materials["TopMetal2"].sigma == s.materials["TopMetal2"].sigma
    assert s2.by_name("Metal1").z == pytest.approx(s.by_name("Metal1").z)
    assert s2.oxide_er == s.oxide_er


def test_stack_from_xml_rejects_non_stackup():
    with pytest.raises(ValueError, match="root element"):
        rfic.Stack.from_xml("<NotAStackup/>")


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
