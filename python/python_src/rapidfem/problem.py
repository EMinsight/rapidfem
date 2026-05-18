"""Problem — generic FEM problem definition (geometry + materials + ports + BCs).

Multiple analyses can run on the same problem::

    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
    air = g.box(A, B, L, position=(-A/2, -B/2, 0), material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis='z'))
    rf.RectWaveguidePort(air.faces.max(axis='z'))
    rf.PEC(*air.faces.unassigned)
    g.mesh()

    prob = rf.Problem(g)
    result  = prob.sweep(np.linspace(8e9, 12e9, 21))
    modes   = prob.eigenmode(target_frequency=10e9, n_modes=6)
    pattern = prob.farfield(result, freq_idx=10, port_idx=0)

Implementation: each analysis call re-assembles the TOML config from the
geometry's physics registry, then constructs an in-memory native
:class:`Simulation` and dispatches to its ``run_sweep`` /
``run_eigenmode`` / ``compute_farfield`` method. The native instance is
cached on the Problem so follow-ups like ``farfield`` can reuse it
without re-assembly.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ._native import Simulation as _NativeSimulation
from .geometry import Geometry
from .physics import PEC, PML


def _f64(x: float) -> str:
    return f"{float(x):.10g}"


class Adaptive:
    """Adaptive-mesh-refinement settings for ``prob.sweep(adaptive=...)``."""

    def __init__(self, *, theta: float = 0.5, refinement_ratio: float = 0.5):
        self.theta = float(theta)
        self.refinement_ratio = float(refinement_ratio)


class Problem:
    """A meshed problem ready for analysis.

    Parameters
    ----------
    geometry : rapidfem.Geometry
        A geometry on which ``g.mesh()`` has been called. The Problem
        snapshots the mesh bytes and a reference to the geometry so it
        can rebuild the TOML config per analysis call.
    """

    def __init__(self, geometry: Geometry):
        if geometry._last_mesh is None:
            raise ValueError(
                "geometry not meshed yet — call g.mesh() before constructing a Problem")
        self._geometry = geometry
        self._mesh_bytes, _ = geometry._last_mesh
        self._native: _NativeSimulation | None = None  # cached after first analysis

    # ── Analyses ──────────────────────────────────────────────────────────

    def sweep(self, frequencies: Iterable[float], *,
              z0: float = 50.0,
              adaptive: Adaptive | None = None):
        """Driven frequency sweep. Returns a :class:`SweepResult`."""
        freqs = [float(f) for f in frequencies]
        if not freqs:
            raise ValueError("sweep needs at least one frequency")
        toml = self._assemble_toml(frequencies=freqs, z0=z0, adaptive=adaptive)
        self._native = _NativeSimulation.from_bytes(self._mesh_bytes, toml)
        return self._native.run_sweep()

    def eigenmode(self, target_frequency: float, *,
                  n_modes: int = 6,
                  z0: float = 50.0):
        """Modal solve around ``target_frequency``. Returns list of Eigenmode."""
        toml = self._assemble_toml(
            frequencies=[float(target_frequency)],
            z0=z0,
            eigenmode=(float(target_frequency), int(n_modes)),
        )
        self._native = _NativeSimulation.from_bytes(self._mesh_bytes, toml)
        return self._native.run_eigenmode()

    def farfield(self, result, *,
                 freq_idx: int,
                 port_idx: int,
                 n_theta: int = 91,
                 n_phi: int = 72):
        """Far-field pattern derived from a prior :meth:`sweep` result."""
        if self._native is None:
            raise ValueError(
                "call .sweep(...) before .farfield(...) — far-field needs a solved problem")
        return self._native.compute_farfield(result, freq_idx, port_idx, n_theta, n_phi)

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def n_dofs(self) -> int:
        """DoF count of the last-assembled native simulation."""
        if self._native is None:
            raise ValueError("run an analysis first to assemble the FEM operator")
        return self._native.n_dofs

    @property
    def n_tets(self) -> int:
        """Tetrahedra in the mesh (assembled lazily)."""
        if self._native is None:
            raise ValueError("run an analysis first to assemble the FEM operator")
        return self._native.n_tets

    # ── TOML assembly ─────────────────────────────────────────────────────

    def _assemble_toml(self, *,
                       frequencies: list[float],
                       z0: float,
                       adaptive: Adaptive | None = None,
                       eigenmode: tuple[float, int] | None = None) -> str:
        g = self._geometry
        parts: list[str] = ['[mesh]\nfile = "(in-memory)"\n']

        freqs_str = ", ".join(_f64(f) for f in frequencies)
        parts.append(f"[frequency]\nvalues = [{freqs_str}]\n")

        # Collect volume entities targeted by PML — they get a [[pml]] block
        # and must NOT also generate a [[materials]] entry (the PML carries
        # its own er_base/ur_base, and double-tagging volumes confuses the
        # Rust solver). Mirrors the old builder workflow where a PML volume
        # had no .material at all.
        pml_volume_ids: set[int] = set()
        for phys in g._physics:
            if isinstance(phys, PML):
                for ent in phys._entities:
                    pml_volume_ids.add(id(ent))

        # Materials — group volumes by Material instance; tag came from mesh().
        # Skip Material instances whose every-volume is a PML target.
        seen_materials: set[int] = set()
        for ent in g._entities:
            mat = ent.material
            if mat is None or isinstance(mat, str) or ent.dim != 3:
                continue
            if id(ent) in pml_volume_ids:
                continue
            mat_id = id(mat)
            if mat_id in seen_materials:
                continue
            seen_materials.add(mat_id)
            tag = g._material_tags.get(mat_id)
            if tag is None:
                raise RuntimeError(
                    f"material {mat!r} has no tag — re-run g.mesh() after attaching it")
            parts.append(mat._to_toml(tag))

        # Physics — ports, BCs, PML. PEC tags get aggregated separately.
        pec_tags: list[int] = []
        for phys in g._physics:
            tag = g._physics_tags.get(id(phys))
            if tag is None:
                raise RuntimeError(
                    f"physics object {phys!r} has no tag — re-run g.mesh() "
                    f"after constructing it")
            if isinstance(phys, PEC):
                pec_tags.append(tag)
            else:
                block = phys._to_toml(tag)
                if block:
                    parts.append(block)

        if pec_tags:
            tags_str = ", ".join(str(t) for t in pec_tags)
            parts.append(f"[pec]\ntags = [{tags_str}]\n")
        else:
            parts.append("[pec]\ntags = []\n")

        if eigenmode is not None:
            f0, nm = eigenmode
            parts.append(f"[eigenmode]\ntarget_frequency = {_f64(f0)}\nn_modes = {nm}\n")

        if adaptive is not None:
            parts.append(
                f"[adaptive]\ntheta = {_f64(adaptive.theta)}\n"
                f"refinement_ratio = {_f64(adaptive.refinement_ratio)}\n"
            )

        parts.append(f"[output]\nz0 = {_f64(z0)}\n")
        return "\n".join(parts)


__all__ = ["Problem", "Adaptive"]
