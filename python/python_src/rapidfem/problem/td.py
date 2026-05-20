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

import sys
import time

import numpy as np

from .._native import TdOperator

_FLUX = {"upwind": 1.0, "central": 0.0}
_FIELD = {"E": 0, "H": 1}
_COMP = {"x": 0, "y": 1, "z": 2}

# Speed of light (m/s). The DG operator runs in normalised units (c = 1, time
# measured in metres); `c` maps operator results to physical SI units —
# `t_op = c·t_seconds`, `f_Hz = c·ω_op/(2π)`.
C_LIGHT = 299_792_458.0


def _log(msg):
    """Progress logging for long TD runs — to stderr, like the FD solver."""
    print(f"  [rapidfem-td] {msg}", file=sys.stderr, flush=True)


def _arr(y):
    """A contiguous 1-D float64 array — the zero-copy form the native
    operator reads directly from its buffer (no Python-list round-trip)."""
    return np.ascontiguousarray(y, dtype=np.float64).ravel()


def _collect_materials(geometry):
    """Walk the geometry's volume materials.

    Returns ``[(tag, eps_diag, mu_diag, sigma)]`` for the native TD operator.
    Non-dispersive materials only — loss-tangent dispersion is a frequency-
    domain effect and is handled by the ADE machinery, not as a constant
    conductivity.
    """
    from ..materials import Material

    out = []
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
        eps = mat.er_diag if mat.er_diag is not None else (mat.er,) * 3
        mu = mat.ur_diag if mat.ur_diag is not None else (mat.ur,) * 3
        out.append((
            int(tag),
            tuple(float(v) for v in eps),
            tuple(float(v) for v in mu),
            float(mat.conductivity),
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


class ProblemTD:
    """Time-domain DGTD Maxwell problem.

    Built from a meshed :class:`~rapidfem.Geometry` — arbitrary unstructured
    tetrahedral meshes. :meth:`box` is a shortcut for a structured box
    cavity, used for validation.
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
        mesh_bytes = geometry._last_mesh[0]
        tag_materials = _collect_materials(geometry)
        self._op = TdOperator.from_mesh_bytes(
            bytes(mesh_bytes), order, _FLUX[flux], tag_materials or None
        )
        self._geometry = geometry
        self.order = order
        self.flux = flux
        self.c = float(c)
        _log(
            f"operator built — {self.n_dof} DOFs, order {order}, "
            f"flux={flux}, {len(tag_materials)} tagged materials"
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
            f"operator built (box) — {obj.n_dof} DOFs, order {order}, "
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
        _log(f"reduce — {dim}-step Arnoldi on {n} DOFs")
        rom = TdReducedModel(self._op.reduced_model(s, int(dim)), self.c)
        _log(f"reduce complete — reduced order r={rom.r}")
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
        times : ndarray, shape ``[steps+1]``
        responses : ndarray, shape ``[n_probes, steps+1]``
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
                f"driven_transient complete — {steps} steps "
                f"in {time.time() - t0:.1f}s"
            )
        return times, resp

    # -- turnkey: a transient run ------------------------------------------
    def transient(self, y0, *, dt, steps, krylov_dim=40, verbose=True):
        """Propagate ``y0`` for ``steps`` steps of size ``dt``.

        Returns the trajectory as an array of shape ``[steps + 1, n_dof]``.
        """
        y = np.asarray(y0, dtype=float).ravel()
        traj = np.empty((steps + 1, y.size))
        traj[0] = y
        t0 = time.time()
        every = max(1, steps // 10)
        for k in range(steps):
            y = self.step(y, dt, krylov_dim)
            traj[k + 1] = y
            if verbose and (k + 1) % every == 0:
                el = time.time() - t0
                eta = el / (k + 1) * (steps - k - 1)
                _log(
                    f"transient {k + 1}/{steps}  "
                    f"({el:.1f}s elapsed, ETA {eta:.0f}s)"
                )
        if verbose:
            _log(f"transient complete — {steps} steps in {time.time() - t0:.1f}s")
        return traj
