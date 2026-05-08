"""
Fluent SimulationBuilder — pair the geometry-layer's name → tag map with
typed port/material/frequency builders, no TOML strings or integer tags
in user code.

    sim = (
        rapidfem.SimulationBuilder()
        .from_geometry(g, maxh=10e-3)
        .frequencies(np.linspace(2.3e9, 2.5e9, 21))
        .pec("ground", "patch_pec")
        .lumped_port("feed", direction=(0, 0, 1), z0=50.0)
        .material("fr4", er=4.4)
        .material("air", er=1.0)
        .build()
    )

`build()` resolves every name through the geometry's `name_to_tag` map and
constructs a `Simulation` via the existing in-memory bytes API.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from rapidfem._native import Simulation


def _f64(x: float) -> str:
    return f"{float(x):.10g}"


class SimulationBuilder:
    def __init__(self):
        self._mesh_bytes: bytes | None = None
        self._name_to_tag: dict[str, int] = {}
        self._frequencies: list[float] = []
        self._ports: list[str] = []        # TOML [[ports]] blocks
        self._materials: list[str] = []    # TOML [[materials]] blocks
        self._pec_tags: list[int] = []
        self._z0_ref: float = 50.0
        self._mat_name_to_tag: dict[str, int] = {}

    # ── Mesh sources ────────────────────────────────────────────────────────

    def mesh(self, mesh_bytes: bytes, name_to_tag: dict[str, int]) -> "SimulationBuilder":
        self._mesh_bytes = mesh_bytes
        self._name_to_tag = dict(name_to_tag)
        # Also separate out the material-volume tags (gmsh physical groups for volumes).
        # Volumes will be wired to materials in the TOML.
        return self

    def from_geometry(self, geometry, maxh: float = 1.0) -> "SimulationBuilder":
        """Convenience: invokes geometry.mesh(maxh) and stores the bytes + name map."""
        mesh_bytes, name_to_tag = geometry.mesh(maxh=maxh)
        return self.mesh(mesh_bytes, name_to_tag)

    # ── Frequencies ────────────────────────────────────────────────────────

    def frequencies(self, values: Iterable[float]) -> "SimulationBuilder":
        self._frequencies = [float(v) for v in values]
        return self

    def frequency_range(self, start: float, stop: float, n: int) -> "SimulationBuilder":
        self._frequencies = list(np.linspace(start, stop, n))
        return self

    # ── PEC / PMC ──────────────────────────────────────────────────────────

    def pec(self, *names: str) -> "SimulationBuilder":
        for n in names:
            self._pec_tags.append(self._tag(n))
        return self

    def pmc(self, *names: str) -> "SimulationBuilder":
        for n in names:
            tag = self._tag(n)
            self._ports.append(f'[[ports]]\ntype = "pmc"\ntag = {tag}\n')
        return self

    # ── Driven / radiation ports ───────────────────────────────────────────

    def rect_waveguide(self, name: str, *,
                       mode: tuple[int, int] = (1, 0),
                       er: float = 1.0,
                       power: float = 1.0,
                       width: float = 0.0,
                       height: float = 0.0) -> "SimulationBuilder":
        tag = self._tag(name)
        self._ports.append(
            f'[[ports]]\ntype = "rectangular"\ntag = {tag}\n'
            f'mode = [{int(mode[0])}, {int(mode[1])}]\n'
            f'er = {_f64(er)}\npower = {_f64(power)}\n'
            f'width = {_f64(width)}\nheight = {_f64(height)}\n'
        )
        return self

    def lumped_port(self, name: str, *,
                    direction: Sequence[float],
                    z0: float = 50.0,
                    power: float = 1.0,
                    width: float = 0.0,
                    height: float = 0.0) -> "SimulationBuilder":
        tag = self._tag(name)
        d = [float(v) for v in direction]
        self._ports.append(
            f'[[ports]]\ntype = "lumped"\ntag = {tag}\n'
            f'z0 = {_f64(z0)}\npower = {_f64(power)}\n'
            f'direction = [{_f64(d[0])}, {_f64(d[1])}, {_f64(d[2])}]\n'
            f'width = {_f64(width)}\nheight = {_f64(height)}\n'
        )
        return self

    def coax_port(self, name: str, *,
                  ri: float,
                  ro: float,
                  origin: Sequence[float] | None = None,
                  z_axis: Sequence[float] | None = None,
                  er: float = 1.0,
                  power: float = 1.0) -> "SimulationBuilder":
        tag = self._tag(name)
        s = (
            f'[[ports]]\ntype = "coax"\ntag = {tag}\n'
            f'ri = {_f64(ri)}\nro = {_f64(ro)}\n'
            f'er = {_f64(er)}\npower = {_f64(power)}\n'
        )
        if origin is not None:
            o = [float(v) for v in origin]
            s += f'origin = [{_f64(o[0])}, {_f64(o[1])}, {_f64(o[2])}]\n'
        if z_axis is not None:
            z = [float(v) for v in z_axis]
            s += f'z_axis = [{_f64(z[0])}, {_f64(z[1])}, {_f64(z[2])}]\n'
        self._ports.append(s)
        return self

    def user_defined_port(self, name: str, *,
                          e_field: Sequence[float],
                          power: float = 1.0) -> "SimulationBuilder":
        tag = self._tag(name)
        e = [float(v) for v in e_field]
        self._ports.append(
            f'[[ports]]\ntype = "user_defined"\ntag = {tag}\n'
            f'e_field = [{_f64(e[0])}, {_f64(e[1])}, {_f64(e[2])}]\n'
            f'power = {_f64(power)}\n'
        )
        return self

    def floquet_port(self, name: str, *,
                     scan_theta_deg: float = 0.0,
                     scan_phi_deg: float = 0.0,
                     mode_nr: int = 1,
                     er: float = 1.0,
                     power: float = 1.0) -> "SimulationBuilder":
        tag = self._tag(name)
        self._ports.append(
            f'[[ports]]\ntype = "floquet"\ntag = {tag}\n'
            f'scan_theta_deg = {_f64(scan_theta_deg)}\n'
            f'scan_phi_deg = {_f64(scan_phi_deg)}\n'
            f'mode_nr = {int(mode_nr)}\n'
            f'er = {_f64(er)}\npower = {_f64(power)}\n'
        )
        return self

    def abc(self, name: str, *, order: int = 1, abctype: str = "B") -> "SimulationBuilder":
        tag = self._tag(name)
        self._ports.append(
            f'[[ports]]\ntype = "abc"\ntag = {tag}\n'
            f'order = {int(order)}\nabctype = "{abctype}"\n'
        )
        return self

    def surface_impedance(self, name: str, *,
                          conductivity: float = 0.0,
                          mur: float = 1.0,
                          er: float = 1.0,
                          thickness: float | None = None,
                          zs: tuple[float, float] | None = None) -> "SimulationBuilder":
        tag = self._tag(name)
        s = (
            f'[[ports]]\ntype = "surface_impedance"\ntag = {tag}\n'
            f'conductivity = {_f64(conductivity)}\n'
            f'mur = {_f64(mur)}\ner = {_f64(er)}\n'
        )
        if thickness is not None:
            s += f'thickness = {_f64(thickness)}\n'
        if zs is not None:
            s += f'zs = [{_f64(zs[0])}, {_f64(zs[1])}]\n'
        self._ports.append(s)
        return self

    def lumped_element(self, name: str, *,
                       r: float = 0.0,
                       l: float = 0.0,
                       c: float | None = None,
                       direction: Sequence[float] = (0.0, 0.0, 1.0),
                       width: float = 0.0,
                       height: float = 0.0) -> "SimulationBuilder":
        tag = self._tag(name)
        s = (
            f'[[ports]]\ntype = "lumped_element"\ntag = {tag}\n'
            f'r = {_f64(r)}\nl = {_f64(l)}\n'
        )
        if c is not None:
            s += f'c = {_f64(c)}\n'
        d = [float(v) for v in direction]
        s += (
            f'direction = [{_f64(d[0])}, {_f64(d[1])}, {_f64(d[2])}]\n'
            f'width = {_f64(width)}\nheight = {_f64(height)}\n'
        )
        self._ports.append(s)
        return self

    # ── Materials ──────────────────────────────────────────────────────────

    def material(self, name: str, *,
                 er: float = 1.0,
                 ur: float = 1.0,
                 tand: float = 0.0,
                 conductivity: float = 0.0,
                 er_diag: Sequence[float] | None = None,
                 ur_diag: Sequence[float] | None = None,
                 debye: dict | None = None,
                 drude: dict | None = None) -> "SimulationBuilder":
        tag = self._tag(name)
        s = (
            f'[[materials]]\nvolume_tag = {tag}\n'
            f'er = {_f64(er)}\nur = {_f64(ur)}\n'
            f'tand = {_f64(tand)}\nconductivity = {_f64(conductivity)}\n'
        )
        if er_diag is not None:
            v = [float(x) for x in er_diag]
            s += f'er_diag = [{_f64(v[0])}, {_f64(v[1])}, {_f64(v[2])}]\n'
        if ur_diag is not None:
            v = [float(x) for x in ur_diag]
            s += f'ur_diag = [{_f64(v[0])}, {_f64(v[1])}, {_f64(v[2])}]\n'
        if debye is not None:
            s += (
                f'[materials.debye]\n'
                f'er_inf = {_f64(debye["er_inf"])}\n'
                f'er_static = {_f64(debye["er_static"])}\n'
                f'tau_s = {_f64(debye["tau_s"])}\n'
            )
        if drude is not None:
            s += (
                f'[materials.drude]\n'
                f'er_inf = {_f64(drude.get("er_inf", 1.0))}\n'
                f'plasma_freq_hz = {_f64(drude["plasma_freq_hz"])}\n'
                f'damping_freq_hz = {_f64(drude["damping_freq_hz"])}\n'
            )
        self._materials.append(s)
        return self

    # ── Output / reference impedance ───────────────────────────────────────

    def reference_impedance(self, z0: float) -> "SimulationBuilder":
        self._z0_ref = float(z0)
        return self

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self) -> Simulation:
        if self._mesh_bytes is None:
            raise ValueError("call .mesh(...) or .from_geometry(...) before .build()")
        if not self._frequencies:
            raise ValueError("call .frequencies(...) before .build()")

        toml = ['[mesh]\nfile = "(in-memory)"\n']
        freqs_str = ", ".join(_f64(f) for f in self._frequencies)
        toml.append(f"[frequency]\nvalues = [{freqs_str}]\n")
        toml.extend(self._ports)
        toml.extend(self._materials)
        if self._pec_tags:
            tags_str = ", ".join(str(t) for t in self._pec_tags)
            toml.append(f"[pec]\ntags = [{tags_str}]\n")
        else:
            toml.append("[pec]\ntags = []\n")
        toml.append(f"[output]\nz0 = {_f64(self._z0_ref)}\n")

        config_toml = "\n".join(toml)
        return Simulation.from_bytes(self._mesh_bytes, config_toml)

    # ── Internals ──────────────────────────────────────────────────────────

    def _tag(self, name: str) -> int:
        if name not in self._name_to_tag:
            available = ", ".join(sorted(self._name_to_tag.keys()))
            raise KeyError(
                f"name {name!r} not found in geometry. Available: {available}"
            )
        return self._name_to_tag[name]


__all__ = ["SimulationBuilder"]
