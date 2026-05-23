"""Smoke test for the TD macromodel Python verb.

Builds a tiny matched WR-90 two-port guide, calls
``ProblemTD.macromodel(...)``, evaluates a band, writes Touchstone,
re-reads the file. Verifies the end-to-end Python pipeline.
"""
from pathlib import Path

import numpy as np

import rapidfem as rf

mm = 1e-3
A_WG, B_WG = 22.86 * mm, 10.16 * mm    # WR-90 cross-section
L = 5.0 * mm                           # short guide (<= one wavelength), fast smoke test

g = rf.Geometry(maxh=6.0 * mm)
air = g.box(A_WG, B_WG, L, material=rf.Air())
rf.RectWaveguidePort(air.faces.min(axis="z"))
rf.RectWaveguidePort(air.faces.max(axis="z"))
rf.PEC(
    air.faces.min(axis="x"), air.faces.max(axis="x"),
    air.faces.min(axis="y"), air.faces.max(axis="y"),
)
g.mesh()

ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"smoke: {ptd.n_dof} DOFs, {ptd._op.n_ports()} ports")

mac = ptd.macromodel(r=300, sprim=True)
print(f"  macromodel built: r={mac.r}, n_ports={mac.n_ports} (sprim=True)")

freqs = np.linspace(9.0e9, 11.0e9, 5)
S = mac.sweep(freqs)
S_p = np.stack([mac.evaluate(f, passive=True) for f in freqs], axis=0)
print(f"  sweep shape: {S.shape}, passive shape: {S_p.shape}")

for k, f in enumerate(freqs):
    sig = float(np.linalg.svd(S[k], compute_uv=False).max())
    sig_p = float(np.linalg.svd(S_p[k], compute_uv=False).max())
    print(
        f"    f={f/1e9:5.2f} GHz  |S11|={abs(S[k,0,0]):.3f}  "
        f"|S21|={abs(S[k,1,0]):.3f}  sigma_max raw={sig:.3f}  pas={sig_p:.3f}"
    )

out = Path("td_macromodel_smoke.s2p")
mac.to_touchstone(out, freqs, format="RI", z_ref=50.0)
print(f"  Touchstone written: {out.resolve()}")

text = out.read_text().splitlines()
data_lines = [line for line in text if line and not line.startswith(("!", "#"))]
print(f"  Touchstone data lines: {len(data_lines)}")
print(f"  first data row: {data_lines[0]}")

out.unlink()
print("OK - macromodel Python pipeline works end-to-end")
