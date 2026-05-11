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
from rapidfem._native import Simulation, SweepResult, Eigenmode, RadiationPattern
from rapidfem.geometry import Geometry, GeoObject, EntityCollection, FaceCollection, EdgeCollection
from rapidfem.builder import SimulationBuilder
from rapidfem import io  # registers .to_network/.to_touchstone/.to_hdf5 on SweepResult
from rapidfem import rfic  # RFIC builder helpers (Stack, microstrip, via, gsg_port, ...)
from rapidfem import _show_capture


def show(obj, name: str = "default"):
    """Send a Geometry, SimulationBuilder, Simulation, or SweepResult to the UI.

    In a normal Python run this is a print-only no-op — scripts work the
    same on the command line. When ``rapidfem serve`` executes the script
    it activates a capture slot, and the object is forwarded to the viewer.
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
    "SimulationBuilder", "io", "rfic", "show",
]
__version__ = "0.2.0"
