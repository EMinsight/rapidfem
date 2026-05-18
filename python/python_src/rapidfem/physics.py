"""Object-based physics — ports + BCs that attach directly to geometry entities.

Physics objects register themselves with the underlying :class:`Geometry`
on construction; no setter / no method-chaining::

    air = g.box(A, B, L, material=rf.Air())

    rf.RectWaveguidePort(air.faces.min(axis='z'))
    rf.RectWaveguidePort(air.faces.max(axis='z'))
    rf.PEC(*air.faces.unassigned)

Each physics object owns the entities it points at — the geometry's
``mesh()`` step creates a physical group per object and stores the
resulting tag in ``Geometry._physics_tags[id(obj)]``. :class:`Problem`
walks the registry to assemble the TOML config the Rust solver consumes.
"""
from __future__ import annotations

from typing import Sequence

from .geometry import EntityCollection, GeoObject, _Entity


def _f64(x: float) -> str:
    return f"{float(x):.10g}"


def _normalize(targets, *, expected_dim: int, cls_name: str):
    """Flatten variadic geometry args to a list of _Entity + their Geometry.

    Accepts ``GeoObject``, ``EntityCollection``, individual ``_Entity``,
    and any iterable of those. All resolved entities must belong to the
    same Geometry and (if ``expected_dim`` is set) carry that dim.
    """
    entities: list[_Entity] = []
    geom = None
    for t in targets:
        if isinstance(t, GeoObject):
            entities.append(t._entity)
            geom = t._geometry if geom is None else geom
            if geom is not t._geometry:
                raise ValueError(
                    f"{cls_name}: targets span multiple Geometry instances")
        elif isinstance(t, EntityCollection):
            entities.extend(t._entities)
            if geom is None:
                geom = t._geometry
            elif geom is not t._geometry:
                raise ValueError(
                    f"{cls_name}: targets span multiple Geometry instances")
        elif isinstance(t, _Entity):
            entities.append(t)
            if t._geometry is None:
                raise ValueError(
                    f"{cls_name}: bare _Entity without Geometry back-ref")
            if geom is None:
                geom = t._geometry
            elif geom is not t._geometry:
                raise ValueError(
                    f"{cls_name}: targets span multiple Geometry instances")
        else:
            raise TypeError(
                f"{cls_name}: cannot use {type(t).__name__} as a target")

    if not entities:
        raise ValueError(f"{cls_name}: no targets")
    if geom is None:
        raise ValueError(f"{cls_name}: could not resolve target Geometry")
    for e in entities:
        if e.dim != expected_dim:
            kind = {2: "face", 3: "volume"}.get(expected_dim, f"dim={expected_dim}")
            raise ValueError(
                f"{cls_name}: expected {kind} targets, got dim={e.dim}")
    return entities, geom


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class _Physics:
    """Common base for all ports and BCs.

    Subclasses set ``_expected_dim`` (2 for faces, 3 for volumes) and
    ``_section`` ("ports", "pec", or "pml") so the :class:`Problem`
    serialiser knows where to put the generated TOML.
    """
    _expected_dim: int = 2
    _section: str = "ports"

    def __init__(self, *targets):
        ents, geom = _normalize(targets,
                                expected_dim=self._expected_dim,
                                cls_name=type(self).__name__)
        self._entities = ents
        self._geometry = geom
        geom._physics.append(self)

    def _to_toml(self, tag: int) -> str:
        """Return the TOML block for this physics object, given its
        physical-group tag from the mesh step. ``[pec]`` aggregation is
        handled by :class:`Problem`; PEC overrides to return ``""``."""
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Driven ports (live on faces)
# ─────────────────────────────────────────────────────────────────────────────

class RectWaveguidePort(_Physics):
    """Analytic TE-mode driven port on a rectangular waveguide face."""

    def __init__(self, *targets,
                 mode: tuple[int, int] = (1, 0),
                 er: float = 1.0,
                 power: float = 1.0,
                 width: float = 0.0,
                 height: float = 0.0):
        super().__init__(*targets)
        self.mode = (int(mode[0]), int(mode[1]))
        self.er = float(er)
        self.power = float(power)
        self.width = float(width)
        self.height = float(height)

    def _to_toml(self, tag: int) -> str:
        return (
            f'[[ports]]\ntype = "rectangular"\ntag = {tag}\n'
            f'mode = [{self.mode[0]}, {self.mode[1]}]\n'
            f'er = {_f64(self.er)}\npower = {_f64(self.power)}\n'
            f'width = {_f64(self.width)}\nheight = {_f64(self.height)}\n'
        )


class LumpedPort(_Physics):
    """Lumped voltage-source driven port on a face."""

    def __init__(self, *targets,
                 direction: Sequence[float],
                 z0: float = 50.0,
                 power: float = 1.0,
                 width: float = 0.0,
                 height: float = 0.0):
        super().__init__(*targets)
        self.direction = tuple(float(v) for v in direction)
        self.z0 = float(z0)
        self.power = float(power)
        self.width = float(width)
        self.height = float(height)

    def _to_toml(self, tag: int) -> str:
        d = self.direction
        return (
            f'[[ports]]\ntype = "lumped"\ntag = {tag}\n'
            f'z0 = {_f64(self.z0)}\npower = {_f64(self.power)}\n'
            f'direction = [{_f64(d[0])}, {_f64(d[1])}, {_f64(d[2])}]\n'
            f'width = {_f64(self.width)}\nheight = {_f64(self.height)}\n'
        )


class CoaxPort(_Physics):
    """TEM-mode driven port on a coaxial annular face."""

    def __init__(self, *targets,
                 ri: float,
                 ro: float,
                 origin: Sequence[float] | None = None,
                 z_axis: Sequence[float] | None = None,
                 er: float = 1.0,
                 power: float = 1.0):
        super().__init__(*targets)
        self.ri = float(ri)
        self.ro = float(ro)
        self.origin = tuple(float(v) for v in origin) if origin is not None else None
        self.z_axis = tuple(float(v) for v in z_axis) if z_axis is not None else None
        self.er = float(er)
        self.power = float(power)

    def _to_toml(self, tag: int) -> str:
        s = (
            f'[[ports]]\ntype = "coax"\ntag = {tag}\n'
            f'ri = {_f64(self.ri)}\nro = {_f64(self.ro)}\n'
            f'er = {_f64(self.er)}\npower = {_f64(self.power)}\n'
        )
        if self.origin is not None:
            o = self.origin
            s += f'origin = [{_f64(o[0])}, {_f64(o[1])}, {_f64(o[2])}]\n'
        if self.z_axis is not None:
            z = self.z_axis
            s += f'z_axis = [{_f64(z[0])}, {_f64(z[1])}, {_f64(z[2])}]\n'
        return s


class UserDefinedPort(_Physics):
    """Driven port with user-supplied uniform E-field on the face."""

    def __init__(self, *targets,
                 e_field: Sequence[float],
                 power: float = 1.0):
        super().__init__(*targets)
        self.e_field = tuple(float(v) for v in e_field)
        self.power = float(power)

    def _to_toml(self, tag: int) -> str:
        e = self.e_field
        return (
            f'[[ports]]\ntype = "user_defined"\ntag = {tag}\n'
            f'e_field = [{_f64(e[0])}, {_f64(e[1])}, {_f64(e[2])}]\n'
            f'power = {_f64(self.power)}\n'
        )


class FloquetPort(_Physics):
    """Floquet plane-wave port for periodic unit cells."""

    def __init__(self, *targets,
                 scan_theta_deg: float = 0.0,
                 scan_phi_deg: float = 0.0,
                 mode_nr: int = 1,
                 er: float = 1.0,
                 power: float = 1.0):
        super().__init__(*targets)
        self.scan_theta_deg = float(scan_theta_deg)
        self.scan_phi_deg = float(scan_phi_deg)
        self.mode_nr = int(mode_nr)
        self.er = float(er)
        self.power = float(power)

    def _to_toml(self, tag: int) -> str:
        return (
            f'[[ports]]\ntype = "floquet"\ntag = {tag}\n'
            f'scan_theta_deg = {_f64(self.scan_theta_deg)}\n'
            f'scan_phi_deg = {_f64(self.scan_phi_deg)}\n'
            f'mode_nr = {self.mode_nr}\n'
            f'er = {_f64(self.er)}\npower = {_f64(self.power)}\n'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

class PEC(_Physics):
    """Perfect electric conductor on one or more faces."""
    _section = "pec"

    def _to_toml(self, tag: int) -> str:
        # PEC tags are aggregated by Problem into [pec] tags=[...]; emit nothing.
        return ""


class PMC(_Physics):
    """Perfect magnetic conductor — symmetry boundary."""

    def _to_toml(self, tag: int) -> str:
        return f'[[ports]]\ntype = "pmc"\ntag = {tag}\n'


class ABC(_Physics):
    """First/second-order absorbing boundary condition on a face."""

    def __init__(self, *targets,
                 order: int = 1,
                 abctype: str = "B"):
        super().__init__(*targets)
        self.order = int(order)
        self.abctype = str(abctype)

    def _to_toml(self, tag: int) -> str:
        return (
            f'[[ports]]\ntype = "abc"\ntag = {tag}\n'
            f'order = {self.order}\nabctype = "{self.abctype}"\n'
        )


class SurfaceImpedance(_Physics):
    """Surface-impedance BC. Pass either (conductivity, mur, er[, thickness])
    or an explicit ``zs`` tuple."""

    def __init__(self, *targets,
                 conductivity: float = 0.0,
                 mur: float = 1.0,
                 er: float = 1.0,
                 thickness: float | None = None,
                 zs: tuple[float, float] | None = None):
        super().__init__(*targets)
        self.conductivity = float(conductivity)
        self.mur = float(mur)
        self.er = float(er)
        self.thickness = float(thickness) if thickness is not None else None
        self.zs = (float(zs[0]), float(zs[1])) if zs is not None else None

    def _to_toml(self, tag: int) -> str:
        s = (
            f'[[ports]]\ntype = "surface_impedance"\ntag = {tag}\n'
            f'conductivity = {_f64(self.conductivity)}\n'
            f'mur = {_f64(self.mur)}\ner = {_f64(self.er)}\n'
        )
        if self.thickness is not None:
            s += f'thickness = {_f64(self.thickness)}\n'
        if self.zs is not None:
            s += f'zs = [{_f64(self.zs[0])}, {_f64(self.zs[1])}]\n'
        return s


class LumpedElement(_Physics):
    """Chip series-RLC element on a 2D footprint."""

    def __init__(self, *targets,
                 r: float = 0.0,
                 l: float = 0.0,
                 c: float | None = None,
                 direction: Sequence[float] = (0.0, 0.0, 1.0),
                 width: float = 0.0,
                 height: float = 0.0):
        super().__init__(*targets)
        self.r = float(r)
        self.l = float(l)
        self.c = float(c) if c is not None else None
        self.direction = tuple(float(v) for v in direction)
        self.width = float(width)
        self.height = float(height)

    def _to_toml(self, tag: int) -> str:
        s = (
            f'[[ports]]\ntype = "lumped_element"\ntag = {tag}\n'
            f'r = {_f64(self.r)}\nl = {_f64(self.l)}\n'
        )
        if self.c is not None:
            s += f'c = {_f64(self.c)}\n'
        d = self.direction
        s += (
            f'direction = [{_f64(d[0])}, {_f64(d[1])}, {_f64(d[2])}]\n'
            f'width = {_f64(self.width)}\nheight = {_f64(self.height)}\n'
        )
        return s


class PML(_Physics):
    """Coordinate-stretched anisotropic Perfectly Matched Layer on a volume."""
    _expected_dim = 3
    _section = "pml"

    def __init__(self, *targets,
                 direction: Sequence[float],
                 inner_face: float,
                 thickness: float,
                 er_base: float = 1.0,
                 ur_base: float = 1.0,
                 exponent: float = 1.5,
                 delta_max: float = 8.0):
        super().__init__(*targets)
        self.direction = tuple(float(v) for v in direction)
        self.inner_face = float(inner_face)
        self.thickness = float(thickness)
        self.er_base = float(er_base)
        self.ur_base = float(ur_base)
        self.exponent = float(exponent)
        self.delta_max = float(delta_max)

    def _to_toml(self, tag: int) -> str:
        d = self.direction
        return (
            f'[[pml]]\nvolume_tag = {tag}\n'
            f'direction = [{_f64(d[0])}, {_f64(d[1])}, {_f64(d[2])}]\n'
            f'inner_face = {_f64(self.inner_face)}\n'
            f'thickness = {_f64(self.thickness)}\n'
            f'er_base = {_f64(self.er_base)}\n'
            f'ur_base = {_f64(self.ur_base)}\n'
            f'exponent = {_f64(self.exponent)}\n'
            f'delta_max = {_f64(self.delta_max)}\n'
        )


__all__ = [
    "RectWaveguidePort", "LumpedPort", "CoaxPort", "UserDefinedPort", "FloquetPort",
    "PEC", "PMC", "ABC", "SurfaceImpedance", "LumpedElement", "PML",
]
