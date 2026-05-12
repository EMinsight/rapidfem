"""Fluent builder for assembling a :class:`Simulation` from a meshed geometry.

The :class:`SimulationBuilder` lets you compose ports, materials, frequencies,
PEC walls, PML regions, and output options on top of an already-meshed
:class:`rapidfem.Geometry`. Names from the geometry layer are resolved to
gmsh physical-group tags internally, so user code never deals with TOML
strings or integer tags.

Example
-------
>>> import numpy as np
>>> import rapidfem
>>> g = rapidfem.Geometry()
>>> # ... build geometry, assign names + materials ...
>>> g.mesh(maxh=5e-3)
>>> sim = (
...     rapidfem.SimulationBuilder()
...     .mesh_from(g)
...     .frequencies(np.linspace(2.3e9, 2.5e9, 21))
...     .pec("ground", "patch_pec")
...     .lumped_port("feed", direction=(0, 0, 1), z0=50.0)
...     .material("fr4", er=4.4)
...     .material("air", er=1.0)
...     .build()
... )
>>> result = sim.run_sweep()
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from rapidfem._native import Simulation


def _f64(x: float) -> str:
    return f"{float(x):.10g}"


class SimulationBuilder:
    """Fluent builder for a frequency-domain Maxwell :class:`Simulation`.

    Methods return ``self`` so calls can be chained. ``build()`` consumes
    the accumulated state and returns a native :class:`Simulation` ready
    for ``run_sweep()`` or ``run_eigenmode()``.

    Every method that references geometry takes ``name`` strings instead
    of gmsh physical-group integers — names are resolved through the
    name→tag map captured by ``mesh()`` / ``from_geometry()`` /
    ``mesh_from()``.
    """

    def __init__(self):
        self._mesh_bytes: bytes | None = None
        self._name_to_tag: dict[str, int] = {}
        self._frequencies: list[float] = []
        self._ports: list[str] = []        # TOML [[ports]] blocks
        self._materials: list[str] = []    # TOML [[materials]] blocks
        self._pec_tags: list[int] = []
        self._z0_ref: float = 50.0
        self._mat_name_to_tag: dict[str, int] = {}
        self._eigenmode: tuple[float, int] | None = None
        self._adaptive: tuple[float, float] | None = None
        self._out_touchstone: str | None = None
        self._out_vtk: str | None = None
        self._out_farfield: str | None = None
        self._out_farfield_nfft_tag: int | None = None
        self._out_group_delay: str | None = None

    # ── Mesh sources ────────────────────────────────────────────────────────

    def mesh(self, mesh_bytes: bytes, name_to_tag: dict[str, int]) -> "SimulationBuilder":
        """Bind a pre-meshed .msh blob and its name → tag map.

        Use ``from_geometry`` or ``mesh_from`` for the typical workflow;
        this lower-level entry point exists for callers that produce
        gmsh bytes through some other path.

        Parameters
        ----------
        mesh_bytes : bytes
            A gmsh ``.msh`` v4 file as bytes.
        name_to_tag : dict[str, int]
            Mapping from geometry names (port faces, materials, etc.)
            to gmsh physical-group integer tags.

        Returns
        -------
        SimulationBuilder
            Self, for call chaining.
        """
        self._mesh_bytes = mesh_bytes
        self._name_to_tag = dict(name_to_tag)
        return self

    def from_geometry(self, geometry, maxh: float = 1.0) -> "SimulationBuilder":
        """Mesh a :class:`Geometry` and bind the result in one step.

        Equivalent to calling ``geometry.mesh(maxh)`` then ``mesh(...)``.

        Parameters
        ----------
        geometry : rapidfem.Geometry
            Geometry with names + materials assigned. Will be meshed
            in-place if not already meshed.
        maxh : float, optional
            Maximum tetrahedron edge length in metres. Default 1.0
            (which is enormous — always pass a value).

        Returns
        -------
        SimulationBuilder
            Self, for call chaining.
        """
        mesh_bytes, name_to_tag = geometry.mesh(maxh=maxh)
        return self.mesh(mesh_bytes, name_to_tag)

    def mesh_from(self, geometry) -> "SimulationBuilder":
        """Reuse a geometry that was meshed in an earlier cell.

        Reads the cached ``.msh`` bytes + name→tag map from
        ``geometry._last_mesh`` without re-meshing. Raises if the
        geometry was never meshed.

        Parameters
        ----------
        geometry : rapidfem.Geometry
            A geometry on which ``.mesh(maxh=...)`` has been called.

        Returns
        -------
        SimulationBuilder
            Self, for call chaining.

        Raises
        ------
        ValueError
            If the geometry has no cached mesh.
        """
        cached = getattr(geometry, "_last_mesh", None)
        if cached is None:
            raise ValueError(
                "geometry has no mesh yet — call g.mesh(maxh=...) first, "
                "or use builder.from_geometry(g, maxh=...) to mesh + store in one go."
            )
        mesh_bytes, name_to_tag = cached
        return self.mesh(mesh_bytes, name_to_tag)

    # ── Frequencies ────────────────────────────────────────────────────────

    def frequencies(self, values: Iterable[float]) -> "SimulationBuilder":
        """Set the driven-sweep frequency points.

        Parameters
        ----------
        values : Iterable[float]
            Frequencies in Hz. Order is preserved.

        Returns
        -------
        SimulationBuilder
            Self.

        Examples
        --------
        >>> b.frequencies(np.linspace(8e9, 12e9, 21))
        """
        self._frequencies = [float(v) for v in values]
        return self

    def frequency_range(self, start: float, stop: float, n: int) -> "SimulationBuilder":
        """Set frequencies as a linearly-spaced sweep.

        Parameters
        ----------
        start, stop : float
            Sweep endpoints in Hz (inclusive on both ends).
        n : int
            Number of points (≥ 2 for a real sweep).

        Returns
        -------
        SimulationBuilder
            Self.
        """
        self._frequencies = list(np.linspace(start, stop, n))
        return self

    # ── PEC / PMC ──────────────────────────────────────────────────────────

    def pec(self, *names: str) -> "SimulationBuilder":
        """Mark one or more named surfaces as perfect electric conductor.

        Parameters
        ----------
        *names : str
            Geometry names to treat as PEC. Each must exist in the
            geometry's name → tag map.

        Returns
        -------
        SimulationBuilder
            Self.

        Examples
        --------
        >>> b.pec("ground", "patch_pec", "pec")
        """
        for n in names:
            self._pec_tags.append(self._tag(n))
        return self

    def pmc(self, *names: str) -> "SimulationBuilder":
        """Mark one or more named surfaces as perfect magnetic conductor.

        Useful as a symmetry boundary for problems where the magnetic
        field is tangential to a plane.

        Parameters
        ----------
        *names : str
            Geometry names to treat as PMC.

        Returns
        -------
        SimulationBuilder
            Self.
        """
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
        """Drive or terminate a port with an analytic TE mode of a
        rectangular waveguide.

        Cross-section dimensions (``width``, ``height``) are auto-detected
        from the port face bounding-box when set to 0. Override only if
        the face is clipped or you want to drive a specific waveguide cross
        section.

        Parameters
        ----------
        name : str
            Geometry face name for the port plane.
        mode : tuple[int, int], optional
            (m, n) TE-mode indices. Default ``(1, 0)`` = TE₁₀.
        er : float, optional
            Relative permittivity inside the waveguide. Default 1.
        power : float, optional
            Incident power in watts. Default 1.
        width, height : float, optional
            Cross-section overrides in metres. 0 means auto-detect.

        Returns
        -------
        SimulationBuilder
            Self.
        """
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
        """Drive a port via a lumped 50-ohm (or arbitrary Z₀) voltage source.

        The port surface should bridge two PEC conductors (e.g. ground and
        a microstrip trace). ``direction`` is the voltage-integration
        direction — typically perpendicular to the two conductors.

        Parameters
        ----------
        name : str
            Geometry face name for the port surface.
        direction : Sequence[float]
            3-vector giving the voltage-integration direction.
        z0 : float, optional
            Reference port impedance in ohms. Default 50.
        power : float, optional
            Incident power in watts. Default 1.
        width, height : float, optional
            Port extent overrides in metres. 0 means auto-detect.

        Returns
        -------
        SimulationBuilder
            Self.

        Examples
        --------
        >>> b.lumped_port("feed", direction=(0, 0, 1), z0=50.0)
        """
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
        """Drive a port with the analytic TEM mode of a coaxial line.

        Parameters
        ----------
        name : str
            Geometry face name for the annular port surface.
        ri, ro : float
            Inner and outer coax radii in metres.
        origin : Sequence[float], optional
            Point on the coax axis. Defaults to the bbox centre of the
            port face.
        z_axis : Sequence[float], optional
            Direction of the coax axis. Defaults to +z.
        er : float, optional
            Relative permittivity of the coax dielectric. Default 1.
        power : float, optional
            Incident power in watts. Default 1.

        Returns
        -------
        SimulationBuilder
            Self.
        """
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
        """Drive a port with a user-supplied uniform E-field on the face.

        Parameters
        ----------
        name : str
            Geometry face name for the port.
        e_field : Sequence[float]
            3-vector of the imposed electric field on the port face.
        power : float, optional
            Normalisation power in watts. Default 1.

        Returns
        -------
        SimulationBuilder
            Self.
        """
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
        """Drive a port with a Floquet plane-wave mode for periodic unit cells.

        For frequency-selective surfaces, phased-array unit cells, and
        similar problems requiring oblique-incidence excitation.

        Parameters
        ----------
        name : str
            Geometry face name for the Floquet port (typically the top
            or bottom of a periodic unit cell).
        scan_theta_deg, scan_phi_deg : float, optional
            Scan angles in degrees (spherical coords). Default (0, 0)
            = normal incidence.
        mode_nr : int, optional
            Floquet mode index. Default 1 (fundamental).
        er : float, optional
            Relative permittivity of the port medium. Default 1.
        power : float, optional
            Incident power in watts. Default 1.

        Returns
        -------
        SimulationBuilder
            Self.
        """
        tag = self._tag(name)
        self._ports.append(
            f'[[ports]]\ntype = "floquet"\ntag = {tag}\n'
            f'scan_theta_deg = {_f64(scan_theta_deg)}\n'
            f'scan_phi_deg = {_f64(scan_phi_deg)}\n'
            f'mode_nr = {int(mode_nr)}\n'
            f'er = {_f64(er)}\npower = {_f64(power)}\n'
        )
        return self

    def pml(self, name: str, *,
            direction: Sequence[float],
            inner_face: float,
            thickness: float,
            er_base: float = 1.0,
            ur_base: float = 1.0,
            exponent: float = 1.5,
            delta_max: float = 8.0) -> "SimulationBuilder":
        """Apply a Perfectly Matched Layer to a *volume* in the geometry.

        Coordinate-stretched anisotropic PML following EMerge's formulation.
        For a closed PML enclosure around an antenna, use 5 non-overlapping
        slabs (top + 4 sides) with each pointing outward along its face
        normal — see ``examples/patch_antenna.py``.

        Parameters
        ----------
        name : str
            Geometry *volume* name. The volume must have its ``name``
            attribute set on the OCC primitive before meshing.
        direction : Sequence[float]
            Outward-pointing unit vector along the absorption axis. Must
            be axis-aligned (one of ±x̂, ±ŷ, ±ẑ).
        inner_face : float
            Coordinate of the PML's inner face along ``direction`` (m).
        thickness : float
            PML extent in metres beyond ``inner_face``.
        er_base, ur_base : float, optional
            Base relative permittivity / permeability inside the PML.
            Default 1 (air-PML).
        exponent : float, optional
            Polynomial profile exponent for the stretch. Typical 1.5–3.
        delta_max : float, optional
            Peak stretch magnitude δ_max at the outer face. Typical 5–12.

        Returns
        -------
        SimulationBuilder
            Self.

        Examples
        --------
        >>> b.pml("pml_top", direction=(0, 0, 1), inner_face=AIR_TOP,
        ...       thickness=PML_T, exponent=1.5, delta_max=8.0)
        """
        tag = self._tag(name)
        d = ", ".join(_f64(v) for v in direction)
        self._materials.append(
            f'[[pml]]\nvolume_tag = {tag}\ndirection = [{d}]\n'
            f'inner_face = {_f64(inner_face)}\n'
            f'thickness = {_f64(thickness)}\n'
            f'er_base = {_f64(er_base)}\n'
            f'ur_base = {_f64(ur_base)}\n'
            f'exponent = {_f64(exponent)}\n'
            f'delta_max = {_f64(delta_max)}\n'
        )
        return self

    def abc(self, name: str, *, order: int = 1, abctype: str = "B") -> "SimulationBuilder":
        """Apply an Absorbing Boundary Condition to a named surface.

        Lower-cost alternative to PML: a surface-level radiation
        boundary. Order 1 is the first-order Sommerfeld ABC; order 2 is
        higher-accuracy at the price of more matrix fill-in.

        Parameters
        ----------
        name : str
            Geometry face name for the ABC surface.
        order : int, optional
            ABC order. ``1`` or ``2``. Default 1.
        abctype : str, optional
            Coefficient family (A–E). Default ``"B"``.

        Returns
        -------
        SimulationBuilder
            Self.
        """
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
        """Apply a surface impedance boundary condition.

        Either give ``conductivity`` + ``mur`` + ``er`` (+ optionally
        ``thickness`` for a thin lossy sheet), or pass ``zs`` directly
        as a ``(real, imag)`` ohms-per-square tuple to override the
        analytic value.

        Parameters
        ----------
        name : str
            Geometry face name.
        conductivity : float, optional
            Bulk conductivity σ (S/m) of the lossy conductor.
        mur, er : float, optional
            Relative permeability / permittivity of the surface medium.
        thickness : float, optional
            Physical thickness of the lossy sheet in metres.
        zs : tuple[float, float], optional
            Explicit surface impedance ``(Re, Im)`` in Ω/□. Overrides
            the analytic computation.

        Returns
        -------
        SimulationBuilder
            Self.
        """
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
        """Embed a chip R / L / C element on a 2D surface.

        Models a series-RLC lumped element living on a named face. Use
        for isolation resistors (Wilkinson dividers), shunt caps, etc.

        Parameters
        ----------
        name : str
            Geometry face name for the element footprint.
        r : float, optional
            Series resistance in ohms.
        l : float, optional
            Series inductance in henries.
        c : float, optional
            Series capacitance in farads. ``None`` = no capacitor.
        direction : Sequence[float], optional
            Current-flow direction across the element. Default +z.
        width, height : float, optional
            Footprint dimensions in metres. 0 means auto-detect.

        Returns
        -------
        SimulationBuilder
            Self.
        """
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
        """Assign material properties to a named volume.

        Supports isotropic scalars (``er``, ``ur``, loss tangent,
        conductivity), diagonal anisotropy, and dispersive Debye / Drude
        models for frequency-dependent ε.

        Parameters
        ----------
        name : str
            Geometry volume name (must have ``volume.material = "..."``
            set in the geometry layer).
        er, ur : float, optional
            Relative permittivity / permeability. Default 1.
        tand : float, optional
            Loss tangent tan δ. Default 0.
        conductivity : float, optional
            Bulk conductivity σ (S/m). Default 0.
        er_diag, ur_diag : Sequence[float], optional
            Diagonal anisotropic (εxx, εyy, εzz) / (μxx, μyy, μzz).
            Overrides the scalar ``er``/``ur``.
        debye : dict, optional
            Debye dispersion. Keys: ``er_inf`` (ε∞), ``er_static`` (εs),
            ``tau_s`` (relaxation time in s).
        drude : dict, optional
            Drude dispersion. Keys: ``er_inf``, ``plasma_freq_hz``,
            ``damping_freq_hz``.

        Returns
        -------
        SimulationBuilder
            Self.

        Examples
        --------
        >>> b.material("fr4", er=4.4, tand=0.02)
        >>> b.material("gold", drude={
        ...     "er_inf": 1.0,
        ...     "plasma_freq_hz": 2.18e15,
        ...     "damping_freq_hz": 6.46e12,
        ... })
        """
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

    # ── Eigenmode + adaptive refinement ────────────────────────────────────

    def eigenmode(self, target_frequency: float, *,
                  n_modes: int = 6) -> "SimulationBuilder":
        """Configure an eigenmode solve around ``target_frequency``.

        After ``build()``, call :meth:`Simulation.run_eigenmode` to
        execute. Uses a shift-invert Lanczos iteration with PARDISO (or
        faer) as the inner direct factoriser.

        ``.frequencies(...)`` does **not** need to be called for an
        eigenmode-only build — ``target_frequency`` doubles as the
        single sample point the matrix assembly needs.

        Parameters
        ----------
        target_frequency : float
            Spectral shift in Hz. Modes nearest this frequency are
            returned.
        n_modes : int, optional
            Number of eigenpairs requested. Default 6.

        Returns
        -------
        SimulationBuilder
            Self.
        """
        self._eigenmode = (float(target_frequency), int(n_modes))
        return self

    def adaptive(self, *,
                 theta: float = 0.5,
                 refinement_ratio: float = 0.5) -> "SimulationBuilder":
        """Enable adaptive mesh refinement on the driven sweep.

        Parameters
        ----------
        theta : float, optional
            Dörfler-marking fraction — elements carrying the top θ of the
            residual error are marked. Default 0.5.
        refinement_ratio : float, optional
            Local size reduction applied to marked elements in the gmsh
            size field. Default 0.5.

        Returns
        -------
        SimulationBuilder
            Self.

        Notes
        -----
        The adaptive refinement loop (solve → estimate → mark → write
        size field → re-mesh → repeat) is driven by the ``rapidfem``
        CLI, not by :meth:`Simulation.run_sweep`. Combine with
        :meth:`dump` to produce a CLI-consumable config.
        """
        self._adaptive = (float(theta), float(refinement_ratio))
        return self

    # ── Output / reference impedance ───────────────────────────────────────

    def reference_impedance(self, z0: float) -> "SimulationBuilder":
        """Set the reference impedance used when reporting S-parameters.

        Default is 50 Ω. Has no effect on the assembly itself, only on
        the post-processed S-parameter normalisation.

        Parameters
        ----------
        z0 : float
            Reference impedance in ohms.

        Returns
        -------
        SimulationBuilder
            Self.
        """
        self._z0_ref = float(z0)
        return self

    def output_touchstone(self, path: str) -> "SimulationBuilder":
        """Request Touchstone (.sNp) export of the S-parameter sweep.

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        SimulationBuilder
            Self.

        Notes
        -----
        Consumed by the ``rapidfem`` CLI; pair with :meth:`dump` to
        write a config the CLI executes. From Python, write Touchstone
        post-solve via ``rapidfem.io.to_touchstone(result, path)``.
        """
        self._out_touchstone = str(path)
        return self

    def output_vtk(self, path: str) -> "SimulationBuilder":
        """Request VTK field-dump export (one file per frequency).

        Parameters
        ----------
        path : str
            Output file path or template.

        Returns
        -------
        SimulationBuilder
            Self.

        Notes
        -----
        CLI-driven. See :meth:`output_touchstone` for the workflow note.
        """
        self._out_vtk = str(path)
        return self

    def output_farfield(self, path: str, *,
                        nfft_tag: int | None = None) -> "SimulationBuilder":
        """Request a far-field CSV export from the NFFT surface.

        Parameters
        ----------
        path : str
            Output CSV path.
        nfft_tag : int, optional
            Physical-group tag of the NFFT surface. Defaults to the ABC
            tag if any.

        Returns
        -------
        SimulationBuilder
            Self.

        Notes
        -----
        CLI-driven. From Python, use :meth:`Simulation.compute_farfield`.
        """
        self._out_farfield = str(path)
        if nfft_tag is not None:
            self._out_farfield_nfft_tag = int(nfft_tag)
        return self

    def output_group_delay(self, path: str) -> "SimulationBuilder":
        """Request group-delay export τ_g = -dφ/dω as CSV.

        Parameters
        ----------
        path : str
            Output CSV path (one row per frequency).

        Returns
        -------
        SimulationBuilder
            Self.

        Notes
        -----
        CLI-driven.
        """
        self._out_group_delay = str(path)
        return self

    # ── Build ──────────────────────────────────────────────────────────────

    def _make_config_toml(self) -> str:
        # Frequencies are required for any *driven* run. An eigenmode-only
        # build can fall back to the target_frequency as a single shift
        # value — the Rust side just needs *some* number to seed the matrix
        # assembly with, which is what the target already is.
        if not self._frequencies:
            if self._eigenmode is None:
                raise ValueError("call .frequencies(...) before .build()/.dump()")
            f0, _ = self._eigenmode
            self._frequencies = [float(f0)]
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

        if self._eigenmode is not None:
            f0, nm = self._eigenmode
            toml.append(
                f"[eigenmode]\ntarget_frequency = {_f64(f0)}\nn_modes = {nm}\n"
            )
        if self._adaptive is not None:
            theta, ratio = self._adaptive
            toml.append(
                f"[adaptive]\ntheta = {_f64(theta)}\nrefinement_ratio = {_f64(ratio)}\n"
            )

        out_lines = [f"[output]\nz0 = {_f64(self._z0_ref)}"]
        if self._out_touchstone is not None:
            out_lines.append(f'touchstone = "{self._out_touchstone}"')
        if self._out_vtk is not None:
            out_lines.append(f'vtk = "{self._out_vtk}"')
        if self._out_farfield is not None:
            out_lines.append(f'farfield = "{self._out_farfield}"')
        if self._out_farfield_nfft_tag is not None:
            out_lines.append(f'nfft_tag = {self._out_farfield_nfft_tag}')
        if self._out_group_delay is not None:
            out_lines.append(f'group_delay = "{self._out_group_delay}"')
        toml.append("\n".join(out_lines) + "\n")
        return "\n".join(toml)

    def build(self) -> Simulation:
        """Construct a native :class:`Simulation` ready for solving.

        Returns
        -------
        Simulation
            The native FEM problem. Call ``.run_sweep()`` (driven) or
            ``.run_eigenmode()`` (modal) to compute results.

        Raises
        ------
        ValueError
            If no mesh has been bound or no frequencies set.
        """
        if self._mesh_bytes is None:
            raise ValueError("call .mesh(...) or .from_geometry(...) before .build()")
        return Simulation.from_bytes(self._mesh_bytes, self._make_config_toml())

    def dump(self, mesh_path: str, config_path: str) -> None:
        """Write the assembled mesh + TOML config to disk.

        Use this to feed the ``rapidfem`` CLI without going through
        Python's solver path. The output pair is exactly what the CLI
        expects.

        Parameters
        ----------
        mesh_path : str
            Destination for the binary ``.msh`` file.
        config_path : str
            Destination for the TOML config.

        Raises
        ------
        ValueError
            If no mesh has been bound.
        """
        if self._mesh_bytes is None:
            raise ValueError("call .mesh(...) or .from_geometry(...) before .dump()")
        with open(mesh_path, "wb") as f:
            f.write(self._mesh_bytes)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(self._make_config_toml())

    # ── Internals ──────────────────────────────────────────────────────────

    def _tag(self, name: str) -> int:
        if name not in self._name_to_tag:
            available = ", ".join(sorted(self._name_to_tag.keys()))
            raise KeyError(
                f"name {name!r} not found in geometry. Available: {available}"
            )
        return self._name_to_tag[name]


__all__ = ["SimulationBuilder"]
