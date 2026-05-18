"""Volume materials for the object-based API.

A ``Material`` is attached to a primitive at construction time::

    sub = g.box(60e-3, 60e-3, 1.6e-3, material=rf.Dielectric(er=4.4, tand=0.02))
    air = g.box(...,                  material=rf.Air())

Multiple volumes can share one ``Material`` instance — they then end up in
the same physical group at mesh time.
"""
from __future__ import annotations

from typing import Sequence


def _f64(x: float) -> str:
    return f"{float(x):.10g}"


class Debye:
    """First-order Debye dispersion."""

    def __init__(self, *, er_inf: float, er_static: float, tau_s: float):
        self.er_inf = float(er_inf)
        self.er_static = float(er_static)
        self.tau_s = float(tau_s)

    def _to_toml(self) -> str:
        return (
            f"[materials.debye]\n"
            f"er_inf = {_f64(self.er_inf)}\n"
            f"er_static = {_f64(self.er_static)}\n"
            f"tau_s = {_f64(self.tau_s)}\n"
        )


class Drude:
    """Drude dispersion for metals."""

    def __init__(self, *, plasma_freq_hz: float, damping_freq_hz: float,
                 er_inf: float = 1.0):
        self.plasma_freq_hz = float(plasma_freq_hz)
        self.damping_freq_hz = float(damping_freq_hz)
        self.er_inf = float(er_inf)

    def _to_toml(self) -> str:
        return (
            f"[materials.drude]\n"
            f"er_inf = {_f64(self.er_inf)}\n"
            f"plasma_freq_hz = {_f64(self.plasma_freq_hz)}\n"
            f"damping_freq_hz = {_f64(self.damping_freq_hz)}\n"
        )


class Material:
    """Generic isotropic / anisotropic linear material.

    The named subclasses (``Air``, ``Dielectric``, ``Conductor``,
    ``Anisotropic``) just preset constructor defaults for readability.
    """

    def __init__(self, *,
                 er: float = 1.0,
                 ur: float = 1.0,
                 tand: float = 0.0,
                 conductivity: float = 0.0,
                 er_diag: Sequence[float] | None = None,
                 ur_diag: Sequence[float] | None = None,
                 debye: Debye | None = None,
                 drude: Drude | None = None):
        self.er = float(er)
        self.ur = float(ur)
        self.tand = float(tand)
        self.conductivity = float(conductivity)
        self.er_diag = tuple(float(v) for v in er_diag) if er_diag is not None else None
        self.ur_diag = tuple(float(v) for v in ur_diag) if ur_diag is not None else None
        self.debye = debye
        self.drude = drude

    def _to_toml(self, volume_tag: int) -> str:
        s = (
            f"[[materials]]\nvolume_tag = {volume_tag}\n"
            f"er = {_f64(self.er)}\nur = {_f64(self.ur)}\n"
            f"tand = {_f64(self.tand)}\nconductivity = {_f64(self.conductivity)}\n"
        )
        if self.er_diag is not None:
            s += f"er_diag = [{_f64(self.er_diag[0])}, {_f64(self.er_diag[1])}, {_f64(self.er_diag[2])}]\n"
        if self.ur_diag is not None:
            s += f"ur_diag = [{_f64(self.ur_diag[0])}, {_f64(self.ur_diag[1])}, {_f64(self.ur_diag[2])}]\n"
        if self.debye is not None:
            s += self.debye._to_toml()
        if self.drude is not None:
            s += self.drude._to_toml()
        return s


class Air(Material):
    """Vacuum / air — εr = μr = 1, lossless."""

    def __init__(self):
        super().__init__()


class Dielectric(Material):
    """Isotropic dielectric. ``er`` is required; loss via ``tand``."""

    def __init__(self, er: float, *,
                 tand: float = 0.0,
                 ur: float = 1.0,
                 conductivity: float = 0.0):
        super().__init__(er=er, ur=ur, tand=tand, conductivity=conductivity)


class Conductor(Material):
    """Bulk lossy conductor — σ in S/m."""

    def __init__(self, *, conductivity: float, ur: float = 1.0, er: float = 1.0):
        super().__init__(er=er, ur=ur, conductivity=conductivity)


class Anisotropic(Material):
    """Diagonal anisotropy via ``er_diag`` / ``ur_diag``."""

    def __init__(self, *,
                 er_diag: Sequence[float] | None = None,
                 ur_diag: Sequence[float] | None = None,
                 tand: float = 0.0,
                 conductivity: float = 0.0):
        super().__init__(tand=tand, conductivity=conductivity,
                         er_diag=er_diag, ur_diag=ur_diag)


__all__ = [
    "Material", "Air", "Dielectric", "Conductor", "Anisotropic",
    "Debye", "Drude",
]
