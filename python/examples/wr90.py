"""
WR-90 straight waveguide example using rapidfem's Python API.

Reproduces what `tests/validation/compare_wr90.py` does, but with the rapidfem
Rust core called directly from Python (no subprocess, no Touchstone files).
"""
import os
import sys

import numpy as np
import rapidfem

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MESH = os.path.join(REPO, "tests", "meshes", "wr90_straight.msh")
CONFIG = os.path.join(REPO, "tests", "validation", "rapidfem_wr90.toml")


def main() -> int:
    sim = rapidfem.Simulation.from_files(MESH, CONFIG)
    print(f"Mesh: {sim.n_tets} tets, {sim.n_dofs} DOFs, {sim.n_driven_ports} driven ports")

    result = sim.run_sweep()
    f = result.frequencies
    s = result.sparams  # complex128, shape [n_freq, n_driven, n_driven]
    print(f"Sweep done in {result.solve_time_s:.3f}s, {len(f)} freq points")
    print()
    print(f"{'freq[GHz]':>10} {'|S11|':>10} {'|S21|':>10} {'arg(S21)deg':>12}")
    for k in range(len(f)):
        s11 = abs(s[k, 0, 0])
        s21 = abs(s[k, 1, 0])
        arg = np.angle(s[k, 1, 0], deg=True)
        print(f"{f[k]/1e9:10.4f} {s11:10.4f} {s21:10.4f} {arg:12.2f}")

    # Sanity checks for WR-90 straight section
    s11_max = np.abs(s[:, 0, 0]).max()
    s21_min = np.abs(s[:, 1, 0]).min()
    s21_max = np.abs(s[:, 1, 0]).max()
    print()
    print(f"max |S11| = {s11_max:.5f}  (expected << 1)")
    print(f"|S21| range = [{s21_min:.5f}, {s21_max:.5f}]  (expected ~1)")

    ok_s11 = s11_max < 0.01
    ok_s21 = abs(s21_min - 1.0) < 0.01 and abs(s21_max - 1.0) < 0.01
    if ok_s11 and ok_s21:
        print("OK")
        return 0
    else:
        print("FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
