"""End-to-end driver: parallel-plate UserDefinedPort EMerge ↔ rapidfem comparison."""
from __future__ import annotations
import os
import subprocess
import sys

from compare import load_csv, load_touchstone, compare

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))


def main() -> int:
    em_csv = os.path.join(HERE, "emerge_parallel_plate.csv")
    rf_s2p = os.path.join(HERE, "rapidfem_parallel_plate.s2p")
    rf_toml = os.path.join(HERE, "rapidfem_parallel_plate.toml")
    mesh = os.path.join(HERE, "..", "meshes", "parallel_plate.msh")

    if not os.path.exists(mesh):
        print(">>> Building parallel plate mesh...")
        subprocess.check_call([sys.executable, os.path.join(HERE, "build_parallel_plate_mesh.py")],
                              cwd=HERE)

    if not os.path.exists(em_csv):
        print(">>> Running EMerge parallel plate...")
        rc = subprocess.call([sys.executable, os.path.join(HERE, "run_emerge_parallel_plate.py")],
                             cwd=HERE)
        if rc != 0:
            return rc

    print(">>> Running rapidfem parallel plate...")
    rc = subprocess.call(
        ["cargo", "run", "--release", "--quiet", "--", os.path.relpath(rf_toml, REPO)],
        cwd=REPO,
    )
    if rc != 0:
        return rc

    f_em, s_em = load_csv(em_csv)
    f_rf, s_rf = load_touchstone(rf_s2p)
    return compare("EMerge", f_em, s_em, "rapidfem", f_rf, s_rf,
                   tol_abs=0.01, tol_rel=0.05)


if __name__ == "__main__":
    sys.exit(main())
