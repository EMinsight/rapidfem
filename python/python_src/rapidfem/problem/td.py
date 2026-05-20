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

import numpy as np

from .._native import TdOperator

_FLUX = {"upwind": 1.0, "central": 0.0}
_FIELD = {"E": 0, "H": 1}
_COMP = {"x": 0, "y": 1, "z": 2}

# Speed of light (m/s). The DG operator runs in normalised units (c = 1, time
# measured in metres); `c` maps operator results to physical SI units —
# `t_op = c·t_seconds`, `f_Hz = c·ω_op/(2π)`.
C_LIGHT = 299_792_458.0


def _aslist(y):
    return np.asarray(y, dtype=float).ravel().tolist()


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
        return obj

    @property
    def n_dof(self):
        """State-vector length, ``6·Np·n_elem``."""
        return self._op.n_dof()

    # -- low level: the ODE -------------------------------------------------
    def rhs(self, y):
        """The ODE right-hand side ``dy/dt = A·y``."""
        return np.asarray(self._op.apply(_aslist(y)))

    def jacobian(self):
        """The (constant) Jacobian of the linear system — i.e. ``A`` itself,
        as a sparse matrix. See :meth:`state_space`."""
        return self.state_space()

    def state_space(self):
        """The verbatim operator ``A`` as a :class:`scipy.sparse.csr_matrix`."""
        from scipy.sparse import csr_matrix

        n, row_ptr, col_idx, values = self._op.state_space()
        return csr_matrix((values, col_idx, row_ptr), shape=(n, n))

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
        return np.asarray(
            self._op.step(_aslist(y), float(self.c * h), int(krylov_dim))
        )

    # -- ports: soft sources & field probes --------------------------------
    def probe_dof(self, point, *, field="E", component="z"):
        """Global DOF index for a field component at the node nearest
        ``point`` — used to place soft sources and field probes."""
        return self._op.nearest_node_dof(
            tuple(float(x) for x in point), _FIELD[field], _COMP[component]
        )

    def driven_transient(
        self, *, source, waveform, probes, dt, steps, krylov_dim=40
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
        for s in range(steps):
            g = float(waveform(s * dt))
            y = np.asarray(
                self._op.step_driven(
                    y.tolist(), sdof, g, float(self.c * dt), krylov_dim
                )
            )
            for k, d in enumerate(pdofs):
                resp[k, s + 1] = y[d]
        return times, resp

    # -- turnkey: a transient run ------------------------------------------
    def transient(self, y0, *, dt, steps, krylov_dim=40):
        """Propagate ``y0`` for ``steps`` steps of size ``dt``.

        Returns the trajectory as an array of shape ``[steps + 1, n_dof]``.
        """
        y = np.asarray(y0, dtype=float).ravel()
        traj = np.empty((steps + 1, y.size))
        traj[0] = y
        for k in range(steps):
            y = self.step(y, dt, krylov_dim)
            traj[k + 1] = y
        return traj
