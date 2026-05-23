"""TD macromodel performance on a RFIC symmetric inductor (the right
target).

A propagating waveguide is the wrong workload for compression — the
in-band content is a continuous dispersion of modes, the Krylov has
to span them all, and the per-evaluate cost stays large because `r`
stays large.

A RFIC inductor is the right workload: electrically tiny, a handful
of in-band resonances, the Krylov collapses onto those few modes and
`r` ~ 30-60 is enough. This is where the macromodel wins.

Compares: build cost, per-evaluate cost, 1000-point sweep cost - and
the same against the frequency-domain backend's direct sweep at a
matched frequency count, so the speed-up is concrete.
"""
import math
import time
from importlib.resources import files
from pathlib import Path

import numpy as np

import rapidfem as rf
import rapidfem.rfic as rfic

JSON_PATH = Path(
    str(files("rapidfem.examples")
        / "fd_rfic_symmetric_inductor_from_json.fem.json")
)

# %% Load + wire the geometry. Re-uses the FD example's JSON layout.
t0 = time.perf_counter()
layout = rfic.from_fem_json(JSON_PATH)

all_volumes = [v for vols in layout.conductors.values() for v in vols]
rf.PEC(*(v.faces for v in all_volumes), *layout.ground_patches)
for port in layout.ports.values():
    rf.LumpedPort(port, direction=(0, 0, 1), z0=50.0)
# First-order ABC on the air-box outer faces - the TD backend now
# treats `rf.ABC` as a Silver-Mueller characteristic-absorbing flux
# (port-with-no-mode), mirroring what the FD example does. This
# replaces the earlier PEC closure that was reflecting cavity
# resonances into the macromodel.
rf.ABC(*layout.air.faces.outer, order=1)

layout.geometry.mesh()
ptd = rf.ProblemTD(layout.geometry, order=2, flux="upwind")
t_setup = time.perf_counter() - t0
print(f"setup (load + mesh + ProblemTD): {t_setup:.2f} s")
print(f"  DOFs: {ptd.n_dof}, ports: {ptd._op.n_ports()}")

# %% Macromodel build at a *small* r appropriate for a resonant
# structure - the whole point of using a compact macromodel here.
R = 60
print(f"\nbuilding macromodel at r = {R} (plain block-Krylov) ...")
t0 = time.perf_counter()
mac = ptd.macromodel(r=R, sprim=False)
t_build = time.perf_counter() - t0
print(f"  build: {t_build:.2f} s")
print(f"  realised r = {mac.r}, n_ports = {mac.n_ports}")

# %% Per-frequency cost
f0 = 10e9
t0 = time.perf_counter()
N_PROBE = 200
for _ in range(N_PROBE):
    _ = mac.evaluate(f0)
t_eval = (time.perf_counter() - t0) / N_PROBE
print(f"  per-evaluate at 10 GHz (avg of {N_PROBE}): {t_eval*1e6:.1f} us")

# %% 1000-point sweep cost
freqs_1k = np.linspace(1e9, 30e9, 1000)
t0 = time.perf_counter()
S_sweep = mac.sweep(freqs_1k)
t_sweep_1k = time.perf_counter() - t0
print(f"  1000-point sweep (1-30 GHz): {t_sweep_1k*1e3:.0f} ms "
      f"({t_sweep_1k*1e6/1000:.1f} us/point)")

# %% Touchstone write
out = Path("td_rfic_inductor.s2p")
t0 = time.perf_counter()
mac.to_touchstone(out, freqs_1k, format="MA")
t_ts = time.perf_counter() - t0
print(f"  Touchstone (1000 points): {t_ts*1e3:.0f} ms")

# %% Frequency-domain reference for comparison: same setup, 16-point sweep
print("\nfor comparison: FD direct sweep on the same JSON setup ...")
# Need to rebuild the FD-side wiring (separate path: ABC, not PEC outer air).
layout_fd = rfic.from_fem_json(JSON_PATH)
all_volumes_fd = [v for vols in layout_fd.conductors.values() for v in vols]
rf.PEC(*(v.faces for v in all_volumes_fd), *layout_fd.ground_patches)
for port in layout_fd.ports.values():
    rf.LumpedPort(port, direction=(0, 0, 1), z0=50.0)
rf.ABC(*layout_fd.air.faces.outer, order=1)
layout_fd.geometry.mesh()
prob_fd = rf.Problem(layout_fd.geometry)
freqs_fd = np.linspace(1e9, 30e9, 16)
t0 = time.perf_counter()
result_fd = prob_fd.sweep(freqs_fd)
t_fd = time.perf_counter() - t0
print(f"  FD 16-point sweep: {t_fd:.1f} s ({t_fd/16*1e3:.0f} ms/point)")

# %% Numerical sanity: compare the macromodel S at the FD points
print(f"\n{'f [GHz]':>9} {'|S11| TD':>10} {'|S11| FD':>10} "
      f"{'|S21| TD':>10} {'|S21| FD':>10}")
S_td_at_fd = mac.sweep(freqs_fd)
for k, f in enumerate(freqs_fd):
    s_td = S_td_at_fd[k]
    s_fd = result_fd.sparams[k]
    print(
        f"{f/1e9:>9.1f} {abs(s_td[0,0]):>10.3f} {abs(s_fd[0,0]):>10.3f} "
        f"{abs(s_td[1,0]):>10.3f} {abs(s_fd[1,0]):>10.3f}"
    )

# %% Headline performance summary
print("\n--- summary ---")
print(f"setup           : {t_setup:>7.2f} s")
print(f"TD build (r={R}) : {t_build:>7.2f} s")
print(f"FD 16-pt sweep  : {t_fd:>7.2f} s  ({t_fd/16*1e3:.0f} ms/point)")
print(f"TD per-evaluate : {t_eval*1e6:>7.1f} us")
print(f"TD 1000-pt sweep: {t_sweep_1k*1e3:>7.0f} ms")
print()
speedup_pt = (t_fd / 16) / t_eval
print(f"TD vs FD per-point speed-up at evaluate: {speedup_pt:.0f}x")
print(
    f"TD breaks even with FD at {t_build / (t_fd / 16):.0f} sweep "
    f"frequencies (build cost / FD per-point cost)"
)

out.unlink()
