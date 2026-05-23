"""Spiral inductor S-parameters via TD block-Krylov macromodel.

A direct counterpart to ``fd_rfic_spiral_from_json``: same JSON layout
load, same PEC / port wiring, but the per-frequency direct solve is
replaced by a single block-Krylov projection of the matrix-free TD
operator. The macromodel builds once (one cost), then evaluates the
full S-matrix sweep in milliseconds — no per-frequency solver call.

Method
------
``ProblemTD.macromodel(r=..., sprim=True)`` runs a SPRIM-style
structure-preserving block-Krylov projection seeded by the
port-injection vectors. The result is a compact reduced-order MIMO
state-space ``(A_hat, B_hat, C_hat)``; evaluating the S-matrix is then
a small dense complex LU per frequency. See
``docs/td-macromodel-plan.md`` for the M1-M3 method and gates.

Touchstone export at the end produces a ``.s2p`` file that any
circuit simulator (e.g. ``scikit-rf``) can ingest.
"""

# %% Load + build the geometry from JSON
import math
from importlib.resources import files
from pathlib import Path

import numpy as np

import rapidfem as rf
import rapidfem.rfic as rfic

JSON_PATH = Path(str(files("rapidfem.examples") / "fd_rfic_spiral_from_json.fem.json"))

layout = rfic.from_fem_json(JSON_PATH)

print(f"loaded {layout.doc['metadata']['generator']} layout")
print(f"  params: {layout.doc['metadata']['params']}")
print(f"  ports:  {list(layout.ports)}")


# %% Wire BCs: PEC conductors, lumped ports, PEC outer air
# The FD example uses a first-order ABC on the outer air faces; the TD
# backend exposes PML matched-absorber slabs (`rf.PML`) instead. For a
# small RFIC structure (Dout ~ 130 um, electrically tiny at the 1-50
# GHz band of interest), closing the air box with PEC is enough: the
# lowest box-resonance lies far above the band, and the macromodel's
# block-Krylov projection captures the dominant near-DC inductive
# response cleanly.
all_volumes = [v for vols in layout.conductors.values() for v in vols]
rf.PEC(*(v.faces for v in all_volumes), *layout.ground_patches)

for port in layout.ports.values():
    rf.LumpedPort(port, direction=(0, 0, 1), z0=50.0)

# Close the air box with PEC on every outer face.
rf.PEC(*layout.air.faces.outer)


# %% Mesh + ProblemTD
layout.geometry.mesh()

ptd = rf.ProblemTD(layout.geometry, order=2, flux="upwind")
print(f"\nProblemTD: {ptd.n_dof} DOFs, {ptd._op.n_ports()} ports")


# %% Macromodel build
# SPRIM-style structure-preserving build (the M3 path): builds a
# block-diagonal V = blockdiag(V_E, V_H), preserving the curl
# operator's E <-> H coupling structure and the resulting S-matrix's
# bounded-real property. Composes with the passivity-enforcement
# perturbation (`passive=True` at evaluate-time) for hard
# sigma_max(S) <= 1.
R_TOTAL = 240   # block-Krylov dimension, split half/half across E and H
mm = ptd.macromodel(r=R_TOTAL, sprim=True)
print(f"macromodel built: r={mm.r}, n_ports={mm.n_ports}")


# %% Frequency sweep
# 50 points across 1-50 GHz; the macromodel evaluates each one in
# microseconds, so a thousand-point sweep would still be milliseconds.
FREQS_HZ = np.linspace(1e9, 50e9, 50)
S = mm.sweep(FREQS_HZ)            # [n_freq, 2, 2] complex
S_passive = np.stack(
    [mm.evaluate(f, passive=True) for f in FREQS_HZ],
    axis=0,
)


# %% Touchstone export — `.s2p` in MA format at 50 ohm reference
out_path = Path("td_rfic_spiral.s2p")
mm.to_touchstone(out_path, FREQS_HZ, format="MA", z_ref=50.0)
print(f"wrote Touchstone: {out_path.resolve()}")


# %% Series-L extraction (same de-embedding as the FD example)
n_ports = mm.n_ports
I = np.eye(n_ports)
z0 = 50.0

print(f"\n{'f [GHz]':>9} {'|S21|':>7} {'|S21| pas':>10} {'L [nH]':>9}")
for k, f in enumerate(FREQS_HZ):
    s = S[k]
    Z = math.sqrt(z0) * (I + s) @ np.linalg.inv(I - s) * math.sqrt(z0)
    omega = 2 * math.pi * f
    L_series = (Z[0, 0].imag - Z[1, 0].imag) / omega
    print(
        f"{f * 1e-9:>9.1f} {abs(S[k, 1, 0]):>7.3f} "
        f"{abs(S_passive[k, 1, 0]):>10.3f} {L_series * 1e9:>9.3f}"
    )


# %% Passivity check: sigma_max(S) for the raw vs the passivity-clipped path
def sigma_max(s):
    return float(np.linalg.svd(s, compute_uv=False).max())


sigma_raw = max(sigma_max(s) for s in S)
sigma_pas = max(sigma_max(s) for s in S_passive)
print(
    f"\nsigma_max across the band: "
    f"raw={sigma_raw:.4f}, passivity-clipped={sigma_pas:.4f}"
)
print(
    "The clipped sweep is the file you would hand to a circuit "
    "simulator; both forms agree at every frequency where the raw "
    "sigma_max is already <= 1."
)
