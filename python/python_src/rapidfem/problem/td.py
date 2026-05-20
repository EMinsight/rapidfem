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


def _aslist(y):
    return np.asarray(y, dtype=float).ravel().tolist()


class ProblemTD:
    """Time-domain DGTD Maxwell problem.

    Built from a meshed :class:`~rapidfem.Geometry` — arbitrary unstructured
    tetrahedral meshes. :meth:`box` is a shortcut for a structured box
    cavity, used for validation.
    """

    def __init__(self, geometry, *, order=2, flux="upwind"):
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
        """
        if flux not in _FLUX:
            raise ValueError(f"flux must be one of {sorted(_FLUX)}")
        if getattr(geometry, "_last_mesh", None) is None:
            raise RuntimeError(
                "geometry not meshed yet — call g.mesh() before "
                "constructing a ProblemTD"
            )
        mesh_bytes = geometry._last_mesh[0]
        self._op = TdOperator.from_mesh_bytes(
            bytes(mesh_bytes), order, _FLUX[flux]
        )
        self._geometry = geometry
        self.order = order
        self.flux = flux

    @classmethod
    def box(cls, *, size, cells, order=2, flux="upwind"):
        """Build directly on a structured box cavity, bypassing the geometry
        API — handy for validation and quick experiments.

        Parameters
        ----------
        size : (lx, ly, lz)
            Cavity dimensions.
        cells : (nx, ny, nz)
            Structured-mesh cell counts per axis.
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

    # -- mid level: stepping ------------------------------------------------
    def step(self, y, h, krylov_dim=40):
        """Advance the state by ``h`` with the matrix-free exponential
        propagator — exact for the linear homogeneous system at any ``h``."""
        return np.asarray(self._op.step(_aslist(y), float(h), int(krylov_dim)))

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
