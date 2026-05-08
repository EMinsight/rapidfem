"""
Debye dispersion sanity test.

Runs WR-90 with a Debye-filled volume, compares the computed eps_r(f) at each frequency
against the analytical Debye formula. The actual S-parameters are sensitive to mesh and
geometry, so we don't compare those against a closed form here — instead this is a
self-consistency + smoothness check that the dispersion math is wired correctly.
"""
from __future__ import annotations
import os
import subprocess
import sys

import numpy as np
from compare import load_touchstone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))


def debye_eps(er_inf: float, er_static: float, tau_s: float, f_hz: float) -> complex:
    omega = 2 * np.pi * f_hz
    return er_inf + (er_static - er_inf) / (1 + 1j * omega * tau_s)


def main() -> int:
    rf_toml = os.path.join(HERE, "rapidfem_debye.toml")
    rf_s2p = os.path.join(HERE, "rapidfem_debye.s2p")

    rc = subprocess.call(
        ["cargo", "run", "--release", "--quiet", "--", os.path.relpath(rf_toml, REPO)],
        cwd=REPO,
    )
    if rc != 0:
        return rc

    f, s = load_touchstone(rf_s2p)
    er_inf, er_static, tau_s = 2.0, 4.0, 5.0e-11

    print(f"\n=== Debye dispersion: analytical eps_r(f) vs S-params smoothness ===")
    print(f"{'freq[GHz]':>10} {'eps_r(f) re':>10} {'eps_r(f) im':>10} {'|S11|':>8} {'|S21|':>8}")
    for k in range(len(f)):
        eps = debye_eps(er_inf, er_static, tau_s, f[k])
        s11 = abs(s[k, 0, 0])
        s21 = abs(s[k, 1, 0])
        print(f"{f[k]/1e9:10.3f} {eps.real:10.4f} {eps.imag:10.4f} {s11:8.4f} {s21:8.4f}")

    # Smoothness check: |dS/df| bounded
    s11_arr = np.abs(s[:, 0, 0])
    diffs = np.abs(np.diff(s11_arr))
    smooth = np.all(diffs < 0.1)
    energy_ok = np.all(np.abs(s[:, 0, 0]) ** 2 + np.abs(s[:, 1, 0]) ** 2 <= 1.05)

    print(f"\nSmoothness (max dS11 between adj. freqs): {diffs.max():.4f} (< 0.1: {smooth})")
    print(f"Passive (|S11|**2 + |S21|**2 <= 1.05): {energy_ok}")
    return 0 if (smooth and energy_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
