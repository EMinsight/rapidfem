"""
FloquetPort smoke test: TE plane wave at normal incidence through air.
PMC side walls keep the mode purely TEM (proxy for periodic BC at θ=0).
Expected: |S11|≈0, |S21|≈1 for the matched line.
"""
from __future__ import annotations
import os
import subprocess
import sys

import numpy as np
from compare import load_touchstone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))


def main() -> int:
    rf_toml = os.path.join(HERE, "rapidfem_floquet.toml")
    rf_s2p = os.path.join(HERE, "rapidfem_floquet.s2p")
    mesh = os.path.join(HERE, "..", "meshes", "parallel_plate.msh")

    if not os.path.exists(mesh):
        subprocess.check_call([sys.executable, os.path.join(HERE, "build_parallel_plate_mesh.py")], cwd=HERE)

    rc = subprocess.call(
        ["cargo", "run", "--release", "--quiet", "--", os.path.relpath(rf_toml, REPO)],
        cwd=REPO,
    )
    if rc != 0:
        return rc

    f, s = load_touchstone(rf_s2p)
    print(f"\n=== FloquetPort theta=0 normal incidence (PMC walls, parallel plate) ===")
    print(f"{'freq[GHz]':>10} {'|S11|':>8} {'|S21|':>8}  status")
    fail = 0
    for k in range(len(f)):
        s11 = abs(s[k, 0, 0])
        s21 = abs(s[k, 1, 0])
        ok = s11 < 0.01 and abs(s21 - 1.0) < 0.005
        if not ok:
            fail += 1
        print(f"{f[k]/1e9:10.3f} {s11:8.4f} {s21:8.4f}  {'OK' if ok else 'FAIL'}")
    print(f"\nFails: {fail}/{len(f)}")
    print(f"Tolerance: |S11|<0.01, ||S21|-1|<0.005")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
