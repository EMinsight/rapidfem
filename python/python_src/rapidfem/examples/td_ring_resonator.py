"""Time-domain DGTD: a dielectric ring resonator, evanescently driven.

A high-permittivity ceramic ring sits in an air-filled PEC cavity. A soft
source placed in the air just inside the ring hole drives a continuous
sinusoid at one of the cavity's resonances: its near (evanescent) field
couples across the gap into the high-er ring, and over many cycles the
field builds up and runs around the ring as a circulating
whispering-gallery-style mode. The 3-D animation shows the orbiting field;
the probe plot shows the resonant buildup envelope.

The drive lands on a ring whispering-gallery resonance (found once with a
broadband sweep of this geometry), so the narrowband sinusoid rings the
ring up over many cycles. A continuous (narrowband) drive instead of an
impulse, and near-field gap coupling instead of a direct on-ring kick.
"""

# %% Parameters
import numpy as np

import rapidfem as rf

mm = 1e-3
R_MAJ = 11.0 * mm        # ring radius (tube centre to torus axis)
R_MIN = 2.6 * mm         # ring tube radius
ER = 10.0                # high-er ceramic ring
BOX = 38.0 * mm          # cubic air-cavity edge

# %% Geometry: a dielectric ring embedded in an air-filled PEC cavity
g = rf.Geometry(maxh=3.5 * mm)
air = g.box(BOX, BOX, BOX, position=(-BOX / 2, -BOX / 2, -BOX / 2),
            material=rf.Air())
ring = g.torus(R_MAJ, R_MIN, material=rf.Dielectric(er=ER), maxh=1.7 * mm)
g.fragment(air, ring)
# Only the 6 axis-aligned cavity walls are PEC; the air-ring interface
# (exposed by fragment) stays interior so the field couples into the ring.
rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"),
       air.faces.min(axis="y"), air.faces.max(axis="y"),
       air.faces.min(axis="z"), air.faces.max(axis="z"))
g.mesh()
rf.show(g)

# %% Build the time-domain problem
ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"DGTD ring resonator - {ptd.n_dof // 60} tets, {ptd.n_dof} state DOFs, "
      f"ring er = {ER}")
# A whispering-gallery resonance of this ring, located once by a broadband
# sweep (a clean, isolated mode in the 8 GHz region). Driving a narrowband
# tone here rings the high-er ring up over many cycles.
f0 = 8.32e9
print(f"driving the ring WGM at {f0 / 1e9:.2f} GHz")

# %% Drive a narrowband sinusoid in the ring hole; the field couples in
# A point just inside the inner edge of the ring tube (in the hole), so the
# evanescent near field reaches across the gap into the high-er ring.
src_point = (R_MAJ - R_MIN - 2.0 * mm, 0.0, 0.0)
# Narrowband (large tau) Gaussian-modulated sinusoid -> a long tone burst
# at the resonance, so the ring rings up over many cycles.
drive = rf.GaussianPulse(t0=600e-12, tau=180e-12, f0=f0)
traj = ptd.transient(
    source=(src_point, "E", "z"), waveform=drive,
    dt=5e-12, steps=320, method="explicit", device="gpu",
)

amp = np.linalg.norm(traj, axis=1)
assert np.all(np.isfinite(traj)), "transient must stay finite"
print(f"ring transient - {traj.shape[0]} snapshots, "
      f"amplitude {amp[0]:.3f} -> peak {amp.max():.3f} -> {amp[-1]:.3f}")

# %% Visualise: the field circulating in the dielectric ring
rf.show(traj)

# %% Probe on the ring tube: the resonant buildup of the circulating field
probe = ptd.probe_dof((R_MAJ, 0.0, 0.0), field="E", component="z")
times = np.arange(traj.shape[0]) * 5e-12
from rapidfem.problem.td import TdResponse
rf.show(TdResponse(times, np.array([traj[:, probe]]),
                   probe_labels=["E_z on the ring"]))
