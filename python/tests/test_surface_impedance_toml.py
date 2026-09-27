# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Unit tests for the SurfaceImpedance face topologies (no solve).

Pins the finite-thickness config boundary: boundary faces emit no flag
(coth(γt)), conductor walls emit ``two_sided`` (coth(γt/2)), embedded sheets
emit ``sheet`` (coth(γt/2)/2). Also pins the warning for a BC on the complete
shell of a solid that is still meshed (issue #46).
"""
import warnings

import pytest

import rapidfem as rf


def _sibc(**kwargs):
    g = rf.Geometry()
    plate = g.xy_plate(1e-3, 1e-3, position=(0, 0, 0))
    return rf.SurfaceImpedance(plate, **kwargs)


def test_default_is_one_sided():
    toml = _sibc(conductivity=5.8e7, thickness=3e-6)._to_toml(tag=42)
    assert 'type = "surface_impedance"' in toml
    assert "thickness = " in toml
    assert "two_sided" not in toml


def test_two_sided_emits_flag():
    toml = _sibc(conductivity=5.8e7, thickness=3e-6, two_sided=True)._to_toml(tag=42)
    assert "thickness = " in toml
    assert "two_sided = true" in toml


def test_semi_infinite_has_no_thickness_terms():
    toml = _sibc(conductivity=5.8e7)._to_toml(tag=7)
    assert "thickness" not in toml
    assert "two_sided" not in toml


def test_sheet_emits_flag():
    toml = _sibc(conductivity=3e7, thickness=1e-6, sheet=True)._to_toml(tag=42)
    assert "sheet = true" in toml
    assert "two_sided" not in toml


def test_sheet_and_two_sided_exclude_each_other():
    with pytest.raises(ValueError, match="exclude"):
        _sibc(conductivity=3e7, thickness=1e-6, sheet=True, two_sided=True)


@pytest.mark.parametrize("choice", [None, True, False])
def test_shell_of_meshed_solid_warns(choice):
    g = rf.Geometry()
    trace = g.box(1e-3, 1e-4, 3e-6, position=(0, 0, 0))
    with pytest.warns(UserWarning, match="still meshed"):
        rf.SurfaceImpedance(trace.faces, conductivity=3e7, thickness=3e-6,
                            two_sided=choice)


def test_single_boundary_face_is_silent():
    g = rf.Geometry()
    sub = g.box(1e-3, 1e-3, 1e-4, position=(0, 0, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rf.SurfaceImpedance(sub.faces.min(axis="z"), conductivity=2e7,
                            thickness=4.2e-7)
