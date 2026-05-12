"""rapidfem — frequency-domain electromagnetic FEM solver in Rust.

Quick start::

    import rapidfem
    sim = rapidfem.Simulation.from_files("mesh.msh", "config.toml")
    result = sim.run_sweep()
    print(result.frequencies.shape, result.sparams.shape)

Solver backend
--------------
The PyPI wheel defaults to the pure-Rust ``faer`` LU solver (no native
dependencies). To opt in to MKL PARDISO (faster on large complex-symmetric
problems), set the env var **before** importing rapidfem::

    import os
    os.environ["RAPIDFEM_SOLVER"] = "pardiso"   # or "auto"
    import rapidfem

PARDISO additionally requires ``mkl_rt`` on the system PATH — see the
README for install options.
"""
import os
import sys

# Make MKL loadable for PARDISO. Rust's libloading uses LoadLibraryA which
# doesn't honour os.add_dll_directory — but if mkl_rt is already in the
# process's loaded-module table, LoadLibraryA returns that handle by name.
# So we pre-load via ctypes after extending the DLL search to common
# conda/anaconda locations.
if sys.platform == "win32":
    _added: list[str] = []
    _conda = os.environ.get("CONDA_PREFIX")
    _cands = [
        os.path.join(_conda, "Library", "bin") if _conda else None,
        os.path.join(sys.prefix, "Library", "bin"),
        os.path.join(sys.base_prefix, "Library", "bin"),
    ]
    for _p in _cands:
        if _p and os.path.isdir(_p) and os.path.exists(os.path.join(_p, "mkl_rt.2.dll")):
            try:
                os.add_dll_directory(_p)  # type: ignore[attr-defined]
                _added.append(_p)
            except (AttributeError, OSError):
                pass
    if _added:
        try:
            import ctypes
            ctypes.CDLL("mkl_rt.2.dll")  # pre-load so libloading picks up the handle
        except OSError:
            pass

from rapidfem._native import Simulation, SweepResult, Eigenmode, RadiationPattern
from rapidfem.geometry import Geometry, GeoObject, EntityCollection, FaceCollection, EdgeCollection
from rapidfem.builder import SimulationBuilder
from rapidfem import io  # registers .to_network/.to_touchstone/.to_hdf5 on SweepResult
from rapidfem import rfic  # RFIC builder helpers (Stack, microstrip, via, gsg_port, ...)
from rapidfem import _show_capture


_C0 = 299_792_458.0


def lambda_maxh(*, f_max: float, er_max: float = 1.0,
                mu_max: float = 1.0, per_lambda: int = 12) -> float:
    """Wavelength-based mesh size cap.

    Picks the largest tet edge length that still resolves the shortest
    wavelength in the model with ``per_lambda`` elements per wavelength::

        maxh = c0 / (sqrt(er_max · mu_max) · f_max · per_lambda)

    Parameters
    ----------
    f_max : float
        Highest frequency in the planned sweep, in Hz.
    er_max : float, optional
        Largest relative permittivity in the model. The smallest local
        wavelength lives in the highest-εᵣ material, so this is the one
        that bounds the mesh size. Default 1.0 (air only).
    mu_max : float, optional
        Largest relative permeability (rarely > 1 in microwave work).
        Default 1.0.
    per_lambda : int, optional
        Target elements per wavelength. For second-kind Nédélec-2,
        ``8–12`` is the usual range; raise to ``15`` for stringent
        accuracy, drop to ``6–8`` for fast preview meshes. Default 12.

    Returns
    -------
    float
        Mesh size cap in metres, ready to pass to ``g.mesh(maxh=...)``.

    Examples
    --------
    >>> import rapidfem
    >>> # Patch antenna on FR-4 at 2.8 GHz
    >>> maxh = rapidfem.lambda_maxh(f_max=2.8e9, er_max=4.4)
    >>> g.mesh(maxh=maxh)                                  # doctest: +SKIP
    """
    if f_max <= 0:
        raise ValueError(f"f_max must be positive, got {f_max}")
    if er_max <= 0 or mu_max <= 0:
        raise ValueError(f"er_max and mu_max must be positive, got {er_max}, {mu_max}")
    if per_lambda <= 0:
        raise ValueError(f"per_lambda must be positive, got {per_lambda}")
    return _C0 / ((er_max * mu_max) ** 0.5 * f_max * per_lambda)


def show(obj, name: str = "default"):
    """Hand an object to the rapidfem viewer (no-op outside ``rapidfem serve``).

    In a plain Python run, ``show`` prints a one-line summary and returns
    ``obj`` unchanged — scripts behave the same on the command line.
    Under ``rapidfem serve`` (or during a static-demo bake), the kernel
    activates a capture slot; ``show`` forwards the object to the live
    3D viewer / S-parameter plot.

    Parameters
    ----------
    obj : Geometry | SimulationBuilder | Simulation | SweepResult
        Anything renderable by the UI. Geometry pre-mesh shows a coarse
        OCC-surface preview; post-mesh shows the FEM tet mesh. Simulation
        and SweepResult render |E(t,r)|² point clouds + S-parameters.
    name : str, optional
        Display slot name. Repeated ``show`` calls with the same name
        overwrite earlier outputs; different names allocate separate
        viewers. Default ``"default"``.

    Returns
    -------
    The same ``obj``, so calls compose with assignment::

        result = rapidfem.show(sim.run_sweep())
    """
    kind = _show_capture.classify(obj)
    if _show_capture.is_capturing():
        _show_capture.capture(name=name, obj=obj, kind=kind)
    else:
        print(f"rapidfem.show({name}={type(obj).__name__}) [kind={kind}] — run via `rapidfem serve` to see it in the UI.")
    return obj


__all__ = [
    "Simulation", "SweepResult", "Eigenmode", "RadiationPattern",
    "Geometry", "GeoObject", "EntityCollection", "FaceCollection", "EdgeCollection",
    "SimulationBuilder", "io", "rfic", "show", "lambda_maxh",
]
__version__ = "0.4.0"
