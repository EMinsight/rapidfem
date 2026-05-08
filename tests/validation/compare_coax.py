"""
Coax CoaxPort self-validation against analytical TEM transmission line.

EMerge's own CoaxPort is buggy (gives |S11|=1.0; no demos use it), so we cannot
cross-validate. Instead we check that rapidfem reproduces the analytical answer
for a matched coax line: |S11| → 0, |S21| → 1, arg(S21) = -k·√εr·L.
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
    rf_s2p = os.path.join(HERE, "rapidfem_coax.s2p")
    rf_toml = os.path.join(HERE, "rapidfem_coax.toml")
    mesh = os.path.join(HERE, "..", "meshes", "coax.msh")

    if not os.path.exists(mesh):
        subprocess.check_call([sys.executable, os.path.join(HERE, "build_coax_mesh.py")], cwd=HERE)

    rc = subprocess.call(
        ["cargo", "run", "--release", "--quiet", "--", os.path.relpath(rf_toml, REPO)],
        cwd=REPO,
    )
    if rc != 0:
        return rc

    f, s = load_touchstone(rf_s2p)
    L = 30e-3
    c0 = 299_792_458.0
    er = 1.0
    print(f"\n=== rapidfem coax vs analytical TEM line ===")
    print(f"{'freq[GHz]':>10} {'|S11|':>8} {'|S21|':>8} {'arg(S21)deg':>12} {'expected':>10} {'diff':>8}")
    fail = 0
    for k in range(len(f)):
        s11 = abs(s[k, 0, 0])
        s21 = abs(s[k, 1, 0])
        arg_meas = np.angle(s[k, 1, 0], deg=True)
        beta = 2 * np.pi * f[k] * np.sqrt(er) / c0
        arg_expected = np.degrees(-beta * L) % 360
        if arg_expected > 180:
            arg_expected -= 360
        arg_diff = ((arg_meas - arg_expected + 180) % 360) - 180
        ok_s11 = s11 < 0.01
        ok_s21 = abs(s21 - 1.0) < 0.005
        ok_arg = abs(arg_diff) < 5.0
        ok = ok_s11 and ok_s21 and ok_arg
        if not ok:
            fail += 1
        print(f"{f[k]/1e9:10.3f} {s11:8.4f} {s21:8.4f} {arg_meas:10.2f} {arg_expected:10.2f} {arg_diff:8.2f}  {'OK' if ok else 'FAIL'}")

    print(f"\nFails: {fail}/{len(f)}")
    print(f"Tolerance: |S11|<0.01, ||S21|-1|<0.005, |arg diff|<5deg")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
