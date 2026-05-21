"""Time-domain DGTD problem — :class:`ProblemTD`.

`ProblemTD` is the time-domain counterpart of :class:`ProblemFD`. Where
`ProblemFD` is an analysis tool (geometry in, S-parameters out), `ProblemTD`
is a *model-export* tool: it compiles a cavity into a linear ODE
``dy/dt = A·y`` and exposes it at every level of abstraction —

* :meth:`transient`           — turnkey: propagate an initial state,
* :meth:`step`                — advance the state one exponential step,
* :meth:`rhs` / :meth:`jacobian` — the ODE right-hand side / constant Jacobian,
* :meth:`state_space`         — the verbatim sparse operator ``A``.

The current backend meshes a structured box cavity with PEC walls; general
geometry support follows the frequency-domain ``(mesh, TOML)`` path.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from .._native import TdOperator
from ..excitation import GaussianPulse

_FLUX = {"upwind": 1.0, "central": 0.0}
_FIELD = {"E": 0, "H": 1}
_COMP = {"x": 0, "y": 1, "z": 2}

# Speed of light (m/s). The DG operator runs in normalised units (c = 1, time
# measured in metres); `c` maps operator results to physical SI units —
# `t_op = c·t_seconds`, `f_Hz = c·ω_op/(2π)`.
C_LIGHT = 299_792_458.0

# S-parameter extraction tolerances. A rectangular-window DFT of the recorded
# port signals stays leakage-free only once the transient has decayed; warn
# when the tail still carries more than this fraction of the peak amplitude.
_SPARAM_DECAY_FRAC = 0.08
# |S| above 1 by more than this is non-physical for a passive structure and
# flags an under-resolved or too-short transient window.
_SPARAM_PASSIVITY_TOL = 0.02
# Fraction of the driven-port peak a port signal must exceed to count as the
# transmitted-pulse arrival; twice that arrival time is the round-trip travel
# time after which the reflection (diagonal) DFT window must close.
_SPARAM_ARRIVAL_FRAC = 0.05

# Matched-absorber loss budget. An rf.PML region wires through to the TD
# backend as a graded impedance-matched absorbing layer. The loss rate `nu`
# ramps quadratically (`nu_max·frac²`) by depth into the layer; round-trip
# attenuation through a quadratically graded slab of thickness `t` is roughly
# `exp(-2·nu_max·t/3)`. Setting `nu_max = _ABSORBER_LOSS_BUDGET / thickness`
# fixes `nu_max·t` regardless of slab depth, so the layer absorbs equally well
# at any thickness. `_ABSORBER_LOSS_BUDGET = 24` gives a round-trip reflection
# of `exp(-2·24/3) ≈ 1e-7` — far below 1 %. (`rf.PML.delta_max` is the
# frequency-domain coordinate-stretch magnitude, a different quantity, and is
# deliberately NOT used as the TD loss rate.)
_ABSORBER_LOSS_BUDGET = 24.0


def _log(msg):
    """Progress logging for long TD runs — to stderr, like the FD solver."""
    print(f"  [rapidfem-td] {msg}", file=sys.stderr, flush=True)


def _arr(y):
    """A contiguous 1-D float64 array — the zero-copy form the native
    operator reads directly from its buffer (no Python-list round-trip)."""
    return np.ascontiguousarray(y, dtype=np.float64).ravel()


def _volume_materials(geometry):
    """Yield ``(material, tag)`` for each unique volume :class:`Material`
    carrying a physical-group tag. The single walk + ``id``-dedup that
    :func:`_collect_materials` and :func:`_collect_dispersive` share.
    """
    from ..materials import Material

    seen = set()
    for ent in getattr(geometry, "_entities", []):
        mat = getattr(ent, "material", None)
        if not isinstance(mat, Material) or getattr(ent, "dim", None) != 3:
            continue
        if id(mat) in seen:
            continue
        seen.add(id(mat))
        tag = geometry._material_tags.get(id(mat))
        if tag is None:
            continue
        yield mat, int(tag)


def _collect_materials(geometry):
    """Walk the geometry's volume materials.

    Returns ``[(tag, eps_diag, mu_diag, sigma)]`` for the native TD operator.
    A material's *non-dispersive* permittivity is reported here; a Debye
    material's non-dispersive permittivity is its ``er_inf`` (the
    high-frequency limit), since the dispersion above ``er_inf`` is supplied
    by the ADE polarisation machinery (see :func:`_collect_dispersive`), not
    as a constant permittivity. A loss tangent is a frequency-domain effect
    and is not turned into a constant conductivity here.
    """
    out = []
    for mat, tag in _volume_materials(geometry):
        eps = mat.er_diag if mat.er_diag is not None else (mat.er,) * 3
        # A Debye material's non-dispersive permittivity is its er_inf; the
        # dispersive operator forces eps = er_inf on these tets anyway, but
        # reporting it here keeps the material assignment self-consistent.
        if getattr(mat, "debye", None) is not None:
            eps = (mat.debye.er_inf,) * 3
        mu = mat.ur_diag if mat.ur_diag is not None else (mat.ur,) * 3
        out.append((
            tag,
            tuple(float(v) for v in eps),
            tuple(float(v) for v in mu),
            float(mat.conductivity),
        ))
    return out


def _collect_dispersive(geometry):
    """Walk the geometry's volume materials for Debye dispersive components.

    Returns ``[(tag, er_inf, er_static, tau_s)]`` for the native TD
    operator's ``dispersive`` argument. Each volume :class:`Material`
    carrying a :class:`rapidfem.Debye` component runs the time-domain
    auxiliary-differential-equation (ADE) update — the operator appends a
    per-element polarisation field and integrates ``dP/dt = a*P + g*E``.

    The Drude model is intentionally not collected here: the time-domain
    backend's ``dispersive.rs`` carries only the first-order Debye ADE.
    Drude (a second-order auxiliary equation) is a future extension.
    """
    out = []
    for mat, tag in _volume_materials(geometry):
        debye = getattr(mat, "debye", None)
        if debye is None:
            continue
        out.append((
            tag,
            float(debye.er_inf),
            float(debye.er_static),
            float(debye.tau_s),
        ))
    return out


def _collect_ports(geometry):
    """Walk the geometry's port physics — :class:`RectWaveguidePort` and
    :class:`LumpedPort`.

    Returns ``[(face_tag, mode_m, mode_n, direction)]`` for the native TD
    operator; the list order (geometry declaration order) fixes the port
    index used by :meth:`ProblemTD.sparams`. A lumped port maps to the
    ``(0, 0)`` sentinel mode — the operator's uniform-profile / TEM port
    (zero cutoff, flat impedance) — and forwards its voltage-integration
    ``direction`` as the port's transverse field axis. A waveguide port
    has ``direction`` ``None`` — its frame is auto-fit from the face.
    """
    from ..physics import LumpedPort, RectWaveguidePort

    out = []
    for phys in getattr(geometry, "_physics", []):
        if isinstance(phys, RectWaveguidePort):
            mode = (int(phys.mode[0]), int(phys.mode[1]))
            direction = None
        elif isinstance(phys, LumpedPort):
            mode = (0, 0)
            d = phys.direction
            direction = (float(d[0]), float(d[1]), float(d[2]))
        else:
            continue
        tag = geometry._physics_tags.get(id(phys))
        if tag is None:
            continue
        out.append((int(tag), mode[0], mode[1], direction))
    return out


def _collect_absorbers(geometry):
    """Walk the geometry's :class:`PML` physics regions.

    Each :class:`rapidfem.PML` terminates the domain with a volumetric
    absorbing slab. In time domain it wires through to the native operator
    as a graded impedance-matched absorbing layer — there is no separate
    coordinate-stretch PML in the TD backend; the matched absorber is the
    TD equivalent.

    Returns ``[(volume_tag, axis, inner_face, thickness, nu_max, is_low)]``
    for the native TD operator's ``absorbers`` argument. ``axis`` is the
    index (0/1/2) of the dominant component of the PML's outward
    ``direction``; ``is_low`` is true when that component points toward
    decreasing coordinate (the layer extends to the low-coordinate end).
    ``nu_max`` is derived from the slab thickness so the round-trip
    reflection stays well below 1 % — see :data:`_ABSORBER_LOSS_BUDGET`.
    """
    from ..physics import PML

    out = []
    for phys in getattr(geometry, "_physics", []):
        if not isinstance(phys, PML):
            continue
        tag = geometry._physics_tags.get(id(phys))
        if tag is None:
            continue
        d = phys.direction
        axis = int(np.argmax(np.abs(np.asarray(d, dtype=float))))
        is_low = bool(d[axis] < 0.0)
        thickness = float(phys.thickness)
        nu_max = _ABSORBER_LOSS_BUDGET / thickness if thickness > 0.0 else 0.0
        out.append((
            int(tag), axis, float(phys.inner_face), thickness,
            float(nu_max), is_low,
        ))
    return out


class TdODE:
    """The time-domain problem as an explicit linear ODE ``dy/dt = A·y``.

    A handoff object for external integrators (e.g.
    :func:`scipy.integrate.solve_ivp`): :meth:`rhs` carries the
    integrator's ``(t, y)`` signature and is evaluated matrix-free;
    :meth:`jacobian` returns the constant sparse ``A`` for implicit
    methods. Obtained from :meth:`ProblemTD.ode`.
    """

    def __init__(self, problem):
        self._p = problem
        self.n_dof = problem.n_dof

    def rhs(self, t, y):
        """``dy/dt`` at state ``y``. The ``t`` argument is ignored — the
        system is autonomous and linear — but kept for the integrator
        signature. Matrix-free, runs on all cores."""
        return self._p._op.apply(_arr(y))

    def jacobian(self, t=None, y=None):
        """The constant Jacobian ``A`` as a :class:`scipy.sparse.csr_matrix`."""
        return self._p.state_space()

    def __repr__(self):
        return f"TdODE(n_dof={self.n_dof})"


class TdStepper:
    """A reusable one-step exponential propagator bound to a fixed ``dt``.

    Call the stepper on a state to advance it by ``dt``; the exponential
    step is exact for the linear homogeneous system at any ``dt``.
    Obtained from :meth:`ProblemTD.stepper`.
    """

    def __init__(self, problem, dt, krylov_dim):
        self._p = problem
        self.dt = float(dt)
        self.krylov_dim = int(krylov_dim)

    def __call__(self, y):
        return self._p.step(y, self.dt, self.krylov_dim)

    def advance(self, y):
        """Advance ``y`` by one ``dt`` step — same as calling the stepper."""
        return self(y)

    def __repr__(self):
        return f"TdStepper(dt={self.dt:g}, krylov_dim={self.krylov_dim})"


class TdReducedModel:
    """A Krylov model-order-reduced view of a :class:`ProblemTD`.

    Wraps the native reduced model so :meth:`propagate` takes physical
    time — consistent with :meth:`ProblemTD.step`. Obtained from
    :meth:`ProblemTD.reduce`.
    """

    def __init__(self, native, c):
        self._m = native
        self._c = float(c)

    @property
    def r(self):
        """Reduced order — the Krylov subspace dimension actually used."""
        return self._m.r

    @property
    def n(self):
        """Full state dimension ``n_dof``."""
        return self._m.n

    @property
    def a_hat(self):
        """The reduced operator ``Â = VᵀAV`` — a dense ``r×r`` array."""
        return self._m.a_hat

    def project(self, y):
        """Project a full state into the reduced subspace — ``ŷ = Vᵀ·y``."""
        return self._m.project(_arr(y))

    def lift(self, yhat):
        """Lift a reduced state back to the full space — ``y = V·ŷ``."""
        return self._m.lift(_arr(yhat))

    def propagate(self, y0, t):
        """Propagate ``y0`` by physical time ``t`` through the reduced
        model — ``V·exp(t·Â)·Vᵀ·y₀``."""
        return self._m.propagate(_arr(y0), float(self._c * t))

    def __repr__(self):
        return f"TdReducedModel(r={self.r}, n={self.n})"


def _point_label(spec):
    """Human-readable label for a ``(point, field, component)`` probe/source
    spec — e.g. ``"E_z @ (0.25, 0.25, 0.5)"``."""
    p, f, c = spec
    coords = ", ".join(f"{v:g}" for v in np.asarray(p, dtype=float).ravel())
    return f"{f}_{c} @ ({coords})"


class TdScattering:
    """Modal-port scattering matrix from :meth:`ProblemTD.sparams`.

    Iterates as ``(frequencies, sparams)``, so the documented tuple
    unpacking — ``freqs, S = ptd.sparams(...)`` — keeps working unchanged;
    the named attributes additionally let :func:`rapidfem.show` plot the
    result in the UI.

    Attributes
    ----------
    frequencies : ndarray
        frequency axis, shape ``[n_freq]``
    sparams : ndarray of complex
        scattering matrix, shape ``[n_freq, n_port, n_port]``
    """

    def __init__(self, frequencies, sparams):
        self.frequencies = np.asarray(frequencies)
        self.sparams = np.asarray(sparams)

    @property
    def n_ports(self):
        """Number of ports — the side length of the S-matrix."""
        return self.sparams.shape[1] if self.sparams.ndim == 3 else 0

    def __iter__(self):
        return iter((self.frequencies, self.sparams))

    def __repr__(self):
        return (f"TdScattering(n_ports={self.n_ports}, "
                f"n_freq={self.frequencies.size})")


class TdResponse:
    """Probe time series from :meth:`ProblemTD.driven_transient`.

    Iterates as ``(times, responses)`` so the documented tuple unpacking
    keeps working; the stored source / probe labels let
    :func:`rapidfem.show` annotate the time-series plot.

    Attributes
    ----------
    times : ndarray
        time axis, shape ``[steps + 1]``
    responses : ndarray
        per-probe samples, shape ``[n_probes, steps + 1]``
    source_label, probe_labels : str, list of str
        human-readable point/field/component labels
    """

    def __init__(self, times, responses, *, source_label="", probe_labels=None):
        self.times = np.asarray(times)
        self.responses = np.asarray(responses)
        self.source_label = source_label
        self.probe_labels = list(probe_labels or [])

    def __iter__(self):
        return iter((self.times, self.responses))

    def __repr__(self):
        return (f"TdResponse(n_probes={self.responses.shape[0]}, "
                f"steps={self.times.size - 1})")


class TdTransfer:
    """Scalar field-to-field frequency response from
    :meth:`ProblemTD.transfer_function`.

    Iterates as ``(frequencies, H)`` so the documented tuple unpacking
    keeps working; the labels let :func:`rapidfem.show` annotate the plot.

    Attributes
    ----------
    frequencies : ndarray
        frequency axis, shape ``[steps // 2 + 1]``
    H : ndarray of complex
        the transfer function ``R(f) / G(f)``
    source_label, probe_label : str
        human-readable point/field/component labels
    """

    def __init__(self, frequencies, H, *, source_label="", probe_label=""):
        self.frequencies = np.asarray(frequencies)
        self.H = np.asarray(H)
        self.source_label = source_label
        self.probe_label = probe_label

    def __iter__(self):
        return iter((self.frequencies, self.H))

    def __repr__(self):
        return f"TdTransfer(n_freq={self.frequencies.size})"


class TdTrajectory(np.ndarray):
    """A time-domain field trajectory — ``[n_snapshot, n_dof]``.

    For every numerical purpose this *is* a :class:`numpy.ndarray` —
    indexing, slicing, ``.shape``, arithmetic and
    :meth:`ProblemTD.export_vtk` all behave exactly as before. It
    additionally carries a back-reference to the originating
    :class:`ProblemTD` and the time step, so :func:`rapidfem.show` can
    sample the DG state onto renderable geometry for the 3-D field
    animation in the UI.
    """

    def __new__(cls, data, *, problem=None, dt=None):
        obj = np.ascontiguousarray(data, dtype=np.float64).view(cls)
        obj._problem = problem
        obj._dt = dt
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._problem = getattr(obj, "_problem", None)
        self._dt = getattr(obj, "_dt", None)


def _fmt(a):
    """Whitespace-joined ascii of a numeric array — VTK DataArray payload."""
    return " ".join(f"{v:.9g}" for v in np.asarray(a).ravel())


def _write_vtu(path, points, connectivity, offsets, cell_types, point_data):
    """Write one VTK XML UnstructuredGrid (``.vtu``), ascii."""
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" '
        'byte_order="LittleEndian">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{len(points)}" '
        f'NumberOfCells="{len(offsets)}">',
        '      <Points>',
        '        <DataArray type="Float64" NumberOfComponents="3" '
        'format="ascii">',
        f'          {_fmt(points)}',
        '        </DataArray>',
        '      </Points>',
        '      <Cells>',
        '        <DataArray type="Int64" Name="connectivity" format="ascii">',
        f'          {_fmt(connectivity)}',
        '        </DataArray>',
        '        <DataArray type="Int64" Name="offsets" format="ascii">',
        f'          {_fmt(offsets)}',
        '        </DataArray>',
        '        <DataArray type="UInt8" Name="types" format="ascii">',
        f'          {_fmt(cell_types)}',
        '        </DataArray>',
        '      </Cells>',
        '      <PointData>',
    ]
    for name, data in point_data.items():
        lines.append(
            f'        <DataArray type="Float64" Name="{name}" '
            'NumberOfComponents="3" format="ascii">'
        )
        lines.append(f'          {_fmt(data)}')
        lines.append('        </DataArray>')
    lines += [
        '      </PointData>',
        '    </Piece>',
        '  </UnstructuredGrid>',
        '</VTKFile>',
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_pvd(path, entries):
    """Write a ParaView ``.pvd`` collection over ``(time, vtu-name)`` pairs."""
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" '
        'byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for t, fname in entries:
        lines.append(f'    <DataSet timestep="{t:.9g}" file="{fname}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class ProblemTD:
    """Time-domain DGTD Maxwell problem ready for analysis.

    A container around a meshed :class:`~rapidfem.Geometry` and its
    attached materials, ports and BCs — the time-domain counterpart of
    :class:`~rapidfem.ProblemFD`. The curl equations are discretised in
    space with a **nodal discontinuous Galerkin** method on tetrahedra,
    giving an explicit linear ODE ``dy/dt = A·y`` with a constant, sparse
    operator ``A``; a driven port adds a rank-1 source ``b(t)``.

    ``ProblemTD`` is a **model-export tool** — it hands back that ODE at
    every level of abstraction, so the verb to call is just the level of
    detail wanted:

    - :meth:`rhs` / :meth:`state_space` — the matrix-free right-hand side,
      or the verbatim sparse operator ``A``
    - :meth:`ode` — a handoff object for an external integrator
      (e.g. ``scipy.integrate.solve_ivp``)
    - :meth:`step` / :meth:`stepper` / :meth:`transient` — exact
      exponential time stepping (matrix-free Krylov / ETD)
    - :meth:`driven_transient` / :meth:`transfer_function` / :meth:`sparams`
      — soft-source or modal-port excitation, a scalar transfer function,
      and the modal-port scattering matrix
    - :meth:`reduce` — a Krylov model-order-reduced surrogate
    - :meth:`resonances` — cavity eigenfrequencies from the spectrum
    - :meth:`export_vtk` — a VTK field animation

    Because the semi-discrete system is linear with a constant ``A``, the
    exponential propagator is *exact* at any step size — the time step is
    set by the wanted output cadence, not by a CFL stability limit.

    Note
    ----
    The geometry must already be meshed (via ``g.mesh()``) before the
    ProblemTD is constructed — construction snapshots the mesh bytes.
    Re-meshing the geometry afterwards has no effect on an existing
    ProblemTD; construct a new one instead. :meth:`box` is a shortcut that
    builds directly on a structured box cavity, bypassing the geometry
    API — handy for validation.

    Example
    -------
    Build a waveguide problem, then read the model at three levels:

    .. code-block:: python

        g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
        air = g.box(22.86e-3, 10.16e-3, 30e-3, material=rf.Air())
        rf.RectWaveguidePort(air.faces.min(axis="z"))
        rf.RectWaveguidePort(air.faces.max(axis="z"))
        rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"),
               air.faces.min(axis="y"), air.faces.max(axis="y"))
        g.mesh()

        ptd = rf.ProblemTD(g, order=2, flux="upwind")
        A = ptd.state_space()                        # verbatim sparse A
        advance = ptd.stepper(dt=5e-12)              # exact exponential step
        freqs, S = ptd.sparams(np.linspace(8e9, 12e9, 21),
                               dt=3e-12, steps=820)  # modal-port S-matrix

    Attributes
    ----------
    n_dof : int
        state-vector length, ``6·Np·n_elem``
    order : int
        the DG polynomial order the operator was built at
    flux : str
        the numerical flux in use — ``"upwind"`` or ``"central"``
    c : float
        speed of light in the mesh's length units; sets the operator ↔
        physical time/frequency mapping
    """

    def __init__(self, geometry, *, order=2, flux="upwind", c=C_LIGHT):
        """
        Parameters
        ----------
        geometry : rapidfem.Geometry
            A geometry on which ``g.mesh()`` has already been called.
        order : int
            DG polynomial order.
        flux : {"upwind", "central"}
            Numerical flux. ``central`` is exactly energy-conserving;
            ``upwind`` additionally damps the discontinuous spurious modes.
        c : float
            Speed of light in the mesh's length units (default SI, metres);
            sets the operator↔physical time/frequency mapping.
        """
        if flux not in _FLUX:
            raise ValueError(f"flux must be one of {sorted(_FLUX)}")
        if getattr(geometry, "_last_mesh", None) is None:
            raise RuntimeError(
                "geometry not meshed yet — call g.mesh() before "
                "constructing a ProblemTD"
            )
        self.c = float(c)
        mesh_bytes = geometry._last_mesh[0]
        tag_materials = _collect_materials(geometry)
        tag_ports = _collect_ports(geometry)
        tag_absorbers = _collect_absorbers(geometry)
        # The TD operator runs in normalised units (c = 1, time measured in
        # the mesh's length units), so a Debye relaxation time given in
        # seconds is scaled to operator units: tau_op = c * tau_s. eps_inf /
        # eps_static are dimensionless and pass through unchanged.
        tag_dispersive = [
            (tag, er_inf, er_static, self.c * tau_s)
            for (tag, er_inf, er_static, tau_s) in _collect_dispersive(geometry)
        ]
        self._op = TdOperator.from_mesh_bytes(
            bytes(mesh_bytes), order, _FLUX[flux],
            tag_materials or None, tag_ports or None,
            tag_absorbers or None, tag_dispersive or None,
        )
        self._geometry = geometry
        self.order = order
        self.flux = flux
        _log(
            f"operator built - {self.n_dof} DOFs, order {order}, "
            f"flux={flux}, {len(tag_materials)} tagged materials, "
            f"{len(tag_ports)} ports, "
            f"{len(tag_absorbers)} matched-absorber (PML) regions, "
            f"{len(tag_dispersive)} Debye dispersive regions"
        )

    @classmethod
    def box(cls, *, size, cells, order=2, flux="upwind", c=1.0):
        """Build directly on a structured box cavity, bypassing the geometry
        API — handy for validation and quick experiments.

        Parameters
        ----------
        size : (lx, ly, lz)
            Cavity dimensions.
        cells : (nx, ny, nz)
            Structured-mesh cell counts per axis.
        c : float
            Speed of light in the box's length units (default 1, normalised).
        """
        if flux not in _FLUX:
            raise ValueError(f"flux must be one of {sorted(_FLUX)}")
        lx, ly, lz = size
        nx, ny, nz = cells
        obj = cls.__new__(cls)
        obj._op = TdOperator(nx, ny, nz, lx, ly, lz, order, _FLUX[flux])
        obj._geometry = None
        obj.order = order
        obj.flux = flux
        obj.c = float(c)
        obj.size = tuple(size)
        obj.cells = tuple(cells)
        _log(
            f"operator built (box) - {obj.n_dof} DOFs, order {order}, "
            f"flux={flux}"
        )
        return obj

    @property
    def n_dof(self):
        """State-vector length, ``6·Np·n_elem``."""
        return self._op.n_dof()

    # -- low level: the ODE -------------------------------------------------
    def rhs(self, y):
        """The ODE right-hand side ``dy/dt = A·y``."""
        return self._op.apply(_arr(y))

    def field_energy(self, state):
        """Instantaneous electromagnetic field energy of a state.

        Returns ``(1/2) * integral(eps*|E|^2 + mu*|H|^2) dV`` in the
        operator's units -- the material-weighted EM field energy carried
        by ``state``. This is a physically exact diagnostic: the DG
        energy-mass matrix is block-diagonal per element, so the energy is
        a cheap per-element quadratic-form sum, evaluated matrix-free with
        no n-by-n matrix ever materialised.

        Unlike a raw state norm ``numpy.dot(y, y)``, this weights each
        component by the local permittivity / permeability, so it is the
        quantity the central flux conserves exactly and the upwind flux
        leaves non-increasing.

        Parameters
        ----------
        state : array_like
            A state vector ``[n_dof]``. Trailing auxiliary DOFs beyond the
            ``6*Np*n_elem`` E,H block are ignored.

        Returns
        -------
        float
            The field energy, finite and non-negative for any real state.
        """
        return self._op.field_energy(_arr(state))

    def jacobian(self):
        """The (constant) Jacobian of the linear system — i.e. ``A`` itself,
        as a sparse matrix. See :meth:`state_space`."""
        return self.state_space()

    def state_space(self):
        """The verbatim operator ``A`` as a :class:`scipy.sparse.csr_matrix`."""
        from scipy.sparse import csr_matrix

        n, row_ptr, col_idx, values = self._op.state_space()
        return csr_matrix((values, col_idx, row_ptr), shape=(n, n))

    def ode(self):
        """Export the problem as an explicit linear ODE ``dy/dt = A·y``.

        Returns a :class:`TdODE` carrying everything an external
        integrator needs — ``n_dof``, a matrix-free ``rhs(t, y)`` with
        the :func:`scipy.integrate.solve_ivp` signature, and
        ``jacobian()``.
        """
        return TdODE(self)

    def resonances(self, *, n=8):
        """Cavity resonant frequencies (Hz) from the operator's spectrum.

        The DG Maxwell operator's eigenvalues are `±iω`; with the upwind flux
        the physical modes are the least-damped ones — `f = c·|ω|/(2π)`.
        Dense eigenvalue solve, so for modest meshes only.
        """
        a = self.state_space().toarray()
        ev = np.linalg.eigvals(a)
        omega = np.abs(ev.imag)
        phys = omega > 1e-3 * omega.max()  # drop the near-static modes
        ev_p = ev[phys]
        out = []
        for idx in np.argsort(-ev_p.real):  # least-damped first
            f = abs(ev_p[idx].imag) * self.c / (2.0 * np.pi)
            if any(abs(f - g) <= 1e-3 * g for g in out):
                continue
            out.append(f)
            if len(out) >= n:
                break
        return np.array(sorted(out))

    # -- mid level: stepping ------------------------------------------------
    def step(self, y, h, krylov_dim=40):
        """Advance the state by ``h`` (in the same time units as ``c``) with
        the matrix-free exponential propagator — exact for the linear
        homogeneous system at any ``h``."""
        return self._op.step(_arr(y), float(self.c * h), int(krylov_dim))

    def stepper(self, dt, *, krylov_dim=40):
        """A reusable one-step propagator bound to a fixed ``dt``.

        Returns a :class:`TdStepper` — call it repeatedly to advance a
        state without re-passing ``dt``/``krylov_dim`` each time.
        """
        return TdStepper(self, dt, krylov_dim)

    # -- model-order reduction ---------------------------------------------
    def reduce(self, start, *, dim=60):
        """Build a Krylov model-order-reduced model around ``start``.

        Runs ``dim``-step Arnoldi on the matrix-free operator from
        ``start``, projecting ``A`` onto the Krylov subspace
        ``span{start, A·start, A²·start, …}``. The returned
        :class:`TdReducedModel` propagates states *in that subspace* —
        ``start`` in particular — with a dense ``r×r`` exponential,
        orders of magnitude cheaper than the full operator.

        Parameters
        ----------
        start : array_like
            The state to reduce around — typically the initial condition
            or excitation vector you intend to propagate. The model is
            accurate for ``start`` and its Krylov orbit, not for
            arbitrary states.
        dim : int
            Krylov subspace dimension (the reduced order). The order
            actually used may be smaller on an early Arnoldi breakdown.

        Returns
        -------
        TdReducedModel
        """
        n = self.n_dof
        s = _arr(start)
        if s.size != n:
            raise ValueError(
                f"start must have length n_dof={n}, got {s.size}"
            )
        if not np.any(s):
            raise ValueError("reduce: start vector must be nonzero")
        _log(f"reduce - {dim}-step Arnoldi on {n} DOFs")
        rom = TdReducedModel(self._op.reduced_model(s, int(dim)), self.c)
        _log(f"reduce complete - reduced order r={rom.r}")
        return rom

    # -- ports: soft sources & field probes --------------------------------
    def probe_dof(self, point, *, field="E", component="z"):
        """Global DOF index for a field component at the node nearest
        ``point`` — used to place soft sources and field probes."""
        return self._op.nearest_node_dof(
            tuple(float(x) for x in point), _FIELD[field], _COMP[component]
        )

    def driven_transient(
        self, *, source, waveform, probes, dt, steps, krylov_dim=40,
        verbose=True,
    ):
        """Drive a soft point source and record field probes.

        Parameters
        ----------
        source : (point, field, component)
            Where and which field component to inject.
        waveform : callable
            ``g(t)`` — the excitation, e.g. a :class:`~rapidfem.GaussianPulse`.
        probes : list of (point, field, component)
            Field samples to record over the run.
        dt, steps : float, int
            Time step and step count.

        Returns
        -------
        TdResponse
            Iterates as ``(times, responses)`` — ``times`` of shape
            ``[steps+1]`` and ``responses`` of shape
            ``[n_probes, steps+1]`` — so ``times, resp = ...`` unpacking
            works unchanged; passing it to :func:`rapidfem.show` plots the
            probe signals.
        """
        sp, sf, sc = source
        sdof = self.probe_dof(sp, field=sf, component=sc)
        pdofs = [
            self.probe_dof(p, field=f, component=c) for (p, f, c) in probes
        ]
        n = self.n_dof
        y = np.zeros(n)
        times = np.arange(steps + 1) * dt
        resp = np.zeros((len(pdofs), steps + 1))
        for k, d in enumerate(pdofs):
            resp[k, 0] = y[d]
        t0 = time.time()
        every = max(1, steps // 10)
        for s in range(steps):
            g = float(waveform(s * dt))
            y = self._op.step_driven(
                _arr(y), sdof, g, float(self.c * dt), krylov_dim
            )
            for k, d in enumerate(pdofs):
                resp[k, s + 1] = y[d]
            if verbose and (s + 1) % every == 0:
                el = time.time() - t0
                eta = el / (s + 1) * (steps - s - 1)
                _log(
                    f"driven_transient {s + 1}/{steps}  "
                    f"({el:.1f}s elapsed, ETA {eta:.0f}s)"
                )
        if verbose:
            _log(
                f"driven_transient complete - {steps} steps "
                f"in {time.time() - t0:.1f}s"
            )
        return TdResponse(
            times, resp,
            source_label=_point_label(source),
            probe_labels=[_point_label(p) for p in probes],
        )

    def transfer_function(
        self, *, source, probe, pulse, dt, steps, krylov_dim=40,
        verbose=True,
    ):
        """Field-to-field frequency response by on-the-fly RFT.

        Drives a broadband ``pulse`` at ``source``, records ``probe``,
        then divides the probe spectrum by the source spectrum —
        ``H(f) = R(f) / G(f)`` — to recover the linear cavity's transfer
        function in one transient run. Peaks of ``|H(f)|`` mark the
        resonances.

        This is the scalar, on-the-fly-RFT observable. It is a transfer
        function between two *field points*, not a normalised port wave:
        true modal-port S-parameters need waveguide-mode injection /
        extraction.

        Parameters
        ----------
        source : (point, field, component)
            Soft-source location and field component to inject.
        probe : (point, field, component)
            Field sample to record.
        pulse : callable
            Broadband excitation ``g(t)`` — its spectrum sets the usable
            frequency band (a :class:`~rapidfem.GaussianPulse` is the
            typical choice).
        dt, steps : float, int
            Time step and step count; together they fix the frequency
            resolution ``1/(steps·dt)`` and the Nyquist limit
            ``1/(2·dt)``.

        Returns
        -------
        TdTransfer
            Iterates as ``(freqs, H)`` — ``freqs`` the frequency axis (Hz
            for an SI geometry, operator units for a :meth:`box`, length
            ``steps//2 + 1``) and ``H`` the complex transfer function
            ``R(f)/G(f)``, zero outside the pulse band. ``freqs, H = ...``
            unpacking works unchanged; :func:`rapidfem.show` plots it.
        """
        times, resp = self.driven_transient(
            source=source, waveform=pulse, probes=[probe],
            dt=dt, steps=steps, krylov_dim=krylov_dim, verbose=verbose,
        )
        g = np.asarray(pulse(times), dtype=float)
        spec_g = np.fft.rfft(g)
        spec_r = np.fft.rfft(resp[0])
        freqs = np.fft.rfftfreq(times.size, dt)
        # H = R/G only where the drive carries real energy. Outside the
        # pulse band G→0, and dividing by it amplifies pure numerical
        # noise — the classic deconvolution artefact — so H is held at
        # zero below 1 % of the peak source spectrum.
        h = np.zeros_like(spec_r)
        band = np.abs(spec_g) > 1e-2 * np.abs(spec_g).max()
        h[band] = spec_r[band] / spec_g[band]
        return TdTransfer(
            freqs, h,
            source_label=_point_label(source),
            probe_label=_point_label(probe),
        )

    # -- ports: scattering parameters --------------------------------------
    def _port_impedance(self, port_idx, f):
        """TE-mode wave impedance of a port at physical frequency ``f``."""
        wc = self._op.port_cutoff(port_idx)
        fc = self.c * wc / (2.0 * np.pi)
        r = fc / f
        return 1.0 / np.sqrt(1.0 - r * r)

    def sparams(self, freqs, *, dt, steps, pulse=None, krylov_dim=40,
                verbose=True):
        """Scattering matrix ``S(f)`` of the waveguide-port network.

        Drives each port in turn with a broadband pulse, extracts the
        modal amplitudes at every port by surface-integral projection,
        and assembles ``S_ij(f) = B_i(f)/A_j(f)`` — the wave leaving port
        ``i`` per unit wave incident at the driven port ``j``.

        Parameters
        ----------
        freqs : array_like
            Frequencies — Hz for a geometry, operator units for a
            :meth:`box`.
        dt, steps : float, int
            Time step and step count. The run must be long enough to
            capture the response; for a clean ``S₁₁`` it should stop
            before energy multiply-reflected by the (characteristic)
            ports returns.
        pulse : callable, optional
            Broadband excitation ``g(t)``; defaults to a modulated
            Gaussian spanning the frequency band.
        krylov_dim : int
            Krylov dimension of the exponential step.

        Returns
        -------
        TdScattering
            Iterates as ``(freqs, S)`` — ``freqs`` the frequency axis and
            ``S`` the complex scattering matrix of shape
            ``[n_freq, n_port, n_port]``, with ``S[f, i, j]`` in the same
            index order as the frequency-domain backend's
            ``SweepResult.sparams``. ``freqs, S = ...`` unpacking works
            unchanged; :func:`rapidfem.show` plots the S-parameters.
        """
        freqs = np.asarray(freqs, dtype=float).ravel()
        n_ports = self._op.n_ports()
        if n_ports == 0:
            raise RuntimeError(
                "ProblemTD has no ports — attach RectWaveguidePort(s) or "
                "LumpedPort(s) to the geometry before constructing it"
            )
        # Below a waveguide port's cutoff the modal wave impedance turns
        # imaginary and the scattering parameters are undefined; reject the
        # run up front rather than let NaN poison the whole S-matrix.
        f_min = float(freqs.min())
        for p in range(n_ports):
            f_cut = self.c * self._op.port_cutoff(p) / (2.0 * np.pi)
            if f_min <= f_cut:
                raise ValueError(
                    f"frequency {f_min:.4g} is at or below the cutoff "
                    f"{f_cut:.4g} of port {p}; restrict freqs to the "
                    f"propagating band"
                )
        n = self.n_dof
        times = np.arange(steps) * dt

        if pulse is None:
            fc = float(np.mean(freqs))
            fw = float(np.ptp(freqs)) or 0.5 * fc
            tau = 1.0 / (np.pi * max(fw, 0.25 * fc))
            pulse = GaussianPulse(t0=4.0 * tau, tau=tau, f0=fc)
        g = np.asarray(pulse(times), dtype=float)
        # DFT kernel — rows index frequency, columns index time sample.
        phase = np.exp(-2j * np.pi * np.outer(freqs, times)) * dt

        h_op = float(self.c * dt)
        a_inc = np.zeros((n_ports, freqs.size), dtype=complex)
        b_out = np.zeros((n_ports, n_ports, freqs.size), dtype=complex)
        every = max(1, steps // 5)
        for j in range(n_ports):
            src = self._op.port_source(j)
            y = np.zeros(n)
            pe = np.zeros((n_ports, steps))
            ph = np.zeros((n_ports, steps))
            t0 = time.time()
            for s in range(steps):
                y = self._op.step_with_source(
                    y, src * g[s], h_op, krylov_dim
                )
                for i in range(n_ports):
                    pe[i, s], ph[i, s] = self._op.port_projections(y, i)
                if verbose and (s + 1) % every == 0:
                    _log(
                        f"sparams drive {j + 1}/{n_ports}: "
                        f"step {s + 1}/{steps} ({time.time() - t0:.1f}s)"
                    )
            # Reflection (i == j) and transmission (i != j) terms need
            # opposite DFT windows. Transmission must capture the whole
            # slow, dispersive transmitted pulse, so it uses the full run.
            # Reflection must be read off before the imperfectly absorbed
            # port re-reflection makes its round trip back to the driven
            # port, so its window closes at twice the first-arrival time
            # at the nearest other port -- the round-trip travel time.
            peak = float(np.abs(pe).max())
            arrivals = []
            for i in range(n_ports):
                if i == j or peak <= 0.0:
                    continue
                env = np.abs(pe[i])
                above = np.flatnonzero(env > _SPARAM_ARRIVAL_FRAC * peak)
                if above.size:
                    arrivals.append(int(above[0]))
            refl_w = (
                int(np.clip(2 * min(arrivals), steps // 4, steps))
                if arrivals else steps
            )
            phase_r = (
                phase
                if refl_w >= steps
                else np.exp(-2j * np.pi * np.outer(freqs, times[:refl_w]))
                * dt
            )
            # Forward / backward modal split A,B = (P_e +- Z*P_h)/2 of the
            # recorded total field.
            for i in range(n_ports):
                z = np.array(
                    [self._port_impedance(i, f) for f in freqs]
                )
                if i == j:
                    pe_f = phase_r @ pe[i, :refl_w]
                    ph_f = phase_r @ ph[i, :refl_w]
                    a_inc[j] = 0.5 * (pe_f + z * ph_f)
                    b_out[j, i] = 0.5 * (pe_f - z * ph_f)
                else:
                    pe_f = phase @ pe[i]
                    ph_f = phase @ ph[i]
                    b_out[j, i] = 0.5 * (pe_f - z * ph_f)
            # Closing the loop: the transmission DFT stays leakage-free
            # only once the transient has decayed by the window end.
            tail = max(1, steps // 20)
            resid = float(np.abs(pe[:, -tail:]).max())
            if verbose and peak > 0.0 and resid > _SPARAM_DECAY_FRAC * peak:
                _log(
                    f"sparams drive {j + 1}: port signal still at "
                    f"{resid / peak:.0%} of peak at the window end, "
                    f"raise steps for a cleaner extraction"
                )

        s_mat = np.zeros((freqs.size, n_ports, n_ports), dtype=complex)
        for j in range(n_ports):
            for i in range(n_ports):
                s_mat[:, i, j] = b_out[j, i] / a_inc[j]
        smax = float(np.abs(s_mat).max())
        if verbose:
            _log(
                f"sparams complete - {n_ports}-port S-matrix at "
                f"{freqs.size} frequencies"
            )
            if smax > 1.0 + _SPARAM_PASSIVITY_TOL:
                _log(
                    f"sparams: |S| peaks at {smax:.3f} above unity, the "
                    f"transient window is too short or the mesh too coarse "
                    f"for a passive result"
                )
        return TdScattering(freqs, s_mat)

    # -- turnkey: a transient run ------------------------------------------
    def transient(self, y0=None, *, dt, steps, source=None, waveform=None,
                  krylov_dim=40, verbose=True):
        """Propagate the field for ``steps`` steps of size ``dt``.

        With ``y0`` only this is the free (homogeneous) evolution of an
        initial state. Passing ``source`` (a ``(point, field, component)``
        spec) together with ``waveform`` (a callable ``g(t)``) instead
        drives a soft point source every step -- a driven transient whose
        full field history is returned, so :func:`rapidfem.show` animates
        the driven problem (e.g. a pulse radiating into a PML-terminated
        domain).

        Parameters
        ----------
        y0 : array_like, optional
            Initial state; defaults to zero (the rest state for a driven
            run).
        dt, steps : float, int
            Time step and step count.
        source : (point, field, component), optional
            Soft-source location and field component. Driving needs both
            ``source`` and ``waveform``.
        waveform : callable, optional
            Excitation ``g(t)``, e.g. a :class:`~rapidfem.GaussianPulse`.

        Returns
        -------
        TdTrajectory
            The field trajectory, shape ``[steps + 1, n_dof]``. It *is* a
            :class:`numpy.ndarray` for every numerical purpose (indexing,
            slicing, :meth:`export_vtk`); passing it to
            :func:`rapidfem.show` plays it back as a 3-D field animation
            in the UI.
        """
        n = self.n_dof
        y = np.zeros(n) if y0 is None else _arr(y0)
        driven = source is not None and waveform is not None
        sdof = None
        if driven:
            sp, sf, sc = source
            sdof = self.probe_dof(sp, field=sf, component=sc)
        h_op = float(self.c * dt)
        traj = np.empty((steps + 1, n))
        traj[0] = y
        t0 = time.time()
        every = max(1, steps // 10)
        label = "driven transient" if driven else "transient"
        for k in range(steps):
            if driven:
                g = float(waveform(k * dt))
                y = self._op.step_driven(_arr(y), sdof, g, h_op, krylov_dim)
            else:
                y = self.step(y, dt, krylov_dim)
            traj[k + 1] = y
            if verbose and (k + 1) % every == 0:
                el = time.time() - t0
                eta = el / (k + 1) * (steps - k - 1)
                _log(
                    f"{label} {k + 1}/{steps}  "
                    f"({el:.1f}s elapsed, ETA {eta:.0f}s)"
                )
        if verbose:
            _log(f"{label} complete - {steps} steps "
                 f"in {time.time() - t0:.1f}s")
        return TdTrajectory(traj, problem=self, dt=dt)

    # -- field export ------------------------------------------------------
    def export_vtk(self, states, path, *, times=None):
        """Write a DG field trajectory as a ParaView-openable VTK series.

        Each snapshot in ``states`` becomes a ``.vtu`` file; a ``.pvd``
        collection ties them into a time animation. The field is exported
        on **discontinuous linear tetrahedra** — one cell per DG element,
        sampled at the element corners — carrying the ``E`` and ``H``
        vector fields as point data. Corner values are exact;
        sub-element high-order variation is not rendered.

        Parameters
        ----------
        states : ndarray
            A single state ``[n_dof]`` or a trajectory
            ``[n_snapshots, n_dof]`` — e.g. the return of
            :meth:`transient`.
        path : str or os.PathLike
            Output base path. ``<path>.pvd`` and ``<path>_NNNN.vtu`` are
            written (the parent directory is created if missing).
        times : array_like, optional
            Time value per snapshot for the ``.pvd`` timeline; defaults
            to the snapshot index.

        Returns
        -------
        str
            Path of the ``.pvd`` collection file.
        """
        states = np.ascontiguousarray(states, dtype=np.float64)
        if states.ndim == 1:
            states = states[None, :]
        n_snap, n_dof = states.shape
        if n_dof != self.n_dof:
            raise ValueError(
                f"states carry {n_dof} DOFs, expected {self.n_dof}"
            )

        o = self.order
        np_ = (o + 1) * (o + 2) * (o + 3) // 6
        n_elem = self.n_dof // (6 * np_)
        corners = np.array(self._op.corner_local_nodes(), dtype=np.int64)

        # Discontinuous linear tets: 4 corner points per element.
        coords = self._op.node_coords().reshape(n_elem, np_, 3)
        corner_xyz = coords[:, corners, :].reshape(-1, 3)
        conn = np.arange(n_elem * 4, dtype=np.int64)
        offsets = np.arange(4, n_elem * 4 + 1, 4, dtype=np.int64)
        cell_types = np.full(n_elem, 10, dtype=np.uint8)  # 10 = VTK_TETRA

        if times is None:
            times = np.arange(n_snap, dtype=float)
        else:
            times = np.asarray(times, dtype=float).ravel()
            if times.size != n_snap:
                raise ValueError(
                    f"times has {times.size} entries, expected {n_snap}"
                )

        base = os.fspath(path)
        parent = os.path.dirname(base)
        if parent:
            os.makedirs(parent, exist_ok=True)
        stem = os.path.basename(base)

        entries = []
        for s in range(n_snap):
            fields = states[s].reshape(n_elem, np_, 6)[:, corners, :]
            vtu = f"{base}_{s:04d}.vtu"
            _write_vtu(
                vtu, corner_xyz, conn, offsets, cell_types,
                {
                    "E": fields[..., 0:3].reshape(-1, 3),
                    "H": fields[..., 3:6].reshape(-1, 3),
                },
            )
            entries.append((float(times[s]), f"{stem}_{s:04d}.vtu"))

        pvd = f"{base}.pvd"
        _write_pvd(pvd, entries)
        _log(f"export_vtk - {n_snap} snapshot(s) -> {pvd}")
        return pvd
