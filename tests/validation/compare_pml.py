"""
PML self-validation: WR-90 with PML termination at +z.
Without PML, end-PEC would reflect ~100%. With PML, |S11| should be small (<<1).
A 1.5 wavelength thick uniaxial PML with delta_max=8, n=1.5 typically gives |S11|<0.01.
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
    rf_toml = os.path.join(HERE, "rapidfem_wr90_pml.toml")
    rf_s2p = os.path.join(HERE, "rapidfem_wr90_pml.s1p")
    mesh = os.path.join(HERE, "..", "meshes", "wr90_pml.msh")

    if not os.path.exists(mesh):
        subprocess.check_call([sys.executable, os.path.join(HERE, "build_wr90_pml_mesh.py")], cwd=HERE)

    rc = subprocess.call(
        ["cargo", "run", "--release", "--quiet", "--", os.path.relpath(rf_toml, REPO)],
        cwd=REPO,
    )
    if rc != 0:
        return rc

    f, s = load_touchstone(rf_s2p)
    print(f"\n=== PML termination of WR-90: |S11| should be << 1 ===")
    print(f"{'freq[GHz]':>10} {'|S11|':>8}  status")
    fail = 0
    for k in range(len(f)):
        s11 = abs(s[k, 0, 0])
        ok = s11 < 0.01  # PML target
        if not ok:
            fail += 1
        print(f"{f[k]/1e9:10.3f} {s11:8.4f}  {'OK' if ok else 'FAIL'}")
    print(f"\nFails: {fail}/{len(f)}")
    print(f"Tolerance: |S11|<0.01 (typical PML target)")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
