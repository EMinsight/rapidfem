"""
Drive the EMerge WR-90 sim, then run rapidfem on a matching mesh, then compare.

Usage:  python compare_wr90.py
"""
from __future__ import annotations
import os
import subprocess
import sys

import numpy as np
from compare import load_csv, load_touchstone, compare

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))


def main() -> int:
    emerge_csv = os.path.join(HERE, "emerge_wr90.csv")
    rapidfem_s2p = os.path.join(HERE, "rapidfem_wr90.s2p")
    rapidfem_toml = os.path.join(HERE, "rapidfem_wr90.toml")

    # Step 1: EMerge (skip if csv already exists and is fresh enough)
    if not os.path.exists(emerge_csv):
        print(">>> Running EMerge WR-90...")
        rc = subprocess.call([sys.executable, os.path.join(HERE, "run_emerge_wr90.py")],
                             cwd=HERE)
        if rc != 0:
            print(f"EMerge run failed (rc={rc})")
            return rc

    # Step 2: rapidfem
    print(">>> Running rapidfem WR-90...")
    rc = subprocess.call(
        ["cargo", "run", "--release", "--quiet", "--", os.path.relpath(rapidfem_toml, REPO)],
        cwd=REPO,
    )
    if rc != 0:
        print(f"rapidfem run failed (rc={rc})")
        return rc

    # Step 3: load both
    f_em, s_em = load_csv(emerge_csv)
    f_rf, s_rf = load_touchstone(rapidfem_s2p)

    # Step 4: compare
    return compare("EMerge", f_em, s_em, "rapidfem", f_rf, s_rf,
                   tol_abs=0.01, tol_rel=0.05)


if __name__ == "__main__":
    sys.exit(main())
