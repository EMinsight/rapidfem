"""rapidfem — frequency-domain electromagnetic FEM solver in Rust.

Quick start::

    import rapidfem
    sim = rapidfem.Simulation.from_files("mesh.msh", "config.toml")
    result = sim.run_sweep()
    print(result.frequencies.shape, result.sparams.shape)
"""
from rapidfem._native import Simulation, SweepResult, Eigenmode, RadiationPattern
from rapidfem.geometry import Geometry, GeoObject, EntityCollection, FaceCollection, EdgeCollection
from rapidfem.builder import SimulationBuilder
from rapidfem import io  # registers .to_network/.to_touchstone/.to_hdf5 on SweepResult
from rapidfem import rfic  # RFIC builder helpers (Stack, microstrip, via, gsg_port, ...)

__all__ = [
    "Simulation", "SweepResult", "Eigenmode", "RadiationPattern",
    "Geometry", "GeoObject", "EntityCollection", "FaceCollection", "EdgeCollection",
    "SimulationBuilder", "io", "rfic",
]
__version__ = "0.1.0"
