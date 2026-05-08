"""
Patch antenna example: run a sweep + far-field pattern computation entirely from Python.

Demonstrates the v2 API: Simulation.run_sweep, Simulation.compute_farfield with
gain / directivity / axial ratio / LCP/RCP arrays as numpy.
"""
import os
import sys

import numpy as np
import rapidfem

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MESH = os.path.join(REPO, "tests", "meshes", "patch_antenna_cq.msh")
CONFIG = os.path.join(REPO, "tests", "config_patch_ff.toml")


def main() -> int:
    sim = rapidfem.Simulation.from_files(MESH, CONFIG)
    print(f"Mesh: {sim.n_tets} tets, {sim.n_dofs} DOFs, {sim.n_driven_ports} driven ports")

    result = sim.run_sweep()
    print(f"Sweep: {result.solve_time_s:.2f}s, {len(result.frequencies)} freq points")
    s11 = abs(result.sparams[0, 0, 0])
    print(f"|S11| at f={result.frequencies[0]/1e9:.3f} GHz: {s11:.4f}")

    # Far-field at the first (and only) sweep frequency
    pattern = sim.compute_farfield(result, freq_idx=0, port_idx=0, n_theta=91, n_phi=72)
    if pattern is None:
        print("Far-field surface empty; skipping pattern")
        return 1

    print()
    print(f"Peak directivity: {pattern.peak_directivity_dbi:.2f} dBi")
    print(f"Peak gain:        {pattern.peak_gain_dbi:.2f} dBi")

    # Pattern arrays — shape [n_phi, n_theta]
    D = pattern.directivity_dbi
    G = pattern.gain_dbi
    AR = pattern.axial_ratio_db
    print(f"D shape:  {D.shape}, dtype {D.dtype}, range [{D.min():.2f}, {D.max():.2f}] dBi")
    print(f"G shape:  {G.shape}, range [{G.min():.2f}, {G.max():.2f}] dBi")
    print(f"AR shape: {AR.shape}, range [{AR.min():.2f}, {AR.max():.2f}] dB")

    # Find broadside direction (closest to theta=0)
    theta = pattern.theta_rad
    phi = pattern.phi_rad
    it_broadside = int(np.argmin(np.abs(theta)))
    ip_broadside = int(np.argmin(np.abs(phi)))
    print(f"\nBroadside (theta=0, phi=0):  D = {D[ip_broadside, it_broadside]:.2f} dBi, "
          f"G = {G[ip_broadside, it_broadside]:.2f} dBi, "
          f"AR = {AR[ip_broadside, it_broadside]:.1f} dB")

    # Sanity: peak directivity matches the .peak_directivity_dbi attribute
    peak_check = float(D.max())
    if abs(peak_check - pattern.peak_directivity_dbi) > 0.01:
        print(f"ERROR: peak_directivity_dbi ({pattern.peak_directivity_dbi:.4f}) does not match D.max() ({peak_check:.4f})")
        return 1

    print("\nOK — Python sweep + far-field pipeline working end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
