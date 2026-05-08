"""rapidfem — frequency-domain electromagnetic FEM solver in Rust.

Quick start::

    import rapidfem
    sim = rapidfem.Simulation.from_files("mesh.msh", "config.toml")
    result = sim.run_sweep()
    print(result.frequencies.shape, result.sparams.shape)
"""
from rapidfem._native import Simulation, SweepResult, Eigenmode, RadiationPattern

__all__ = ["Simulation", "SweepResult", "Eigenmode", "RadiationPattern"]
__version__ = "0.1.0"
