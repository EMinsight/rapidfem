"""Time-domain DGTD: a driven waveguide T-junction power divider.

A soft pulse is driven into the stem of a waveguide tee; it travels up to
the junction and splits into the two crossbar arms. All three arms end in
an `rf.PML` matched-absorber slab, so the divided pulses leave cleanly
with no cavity echo — the 3-D transient shows the split happening in time
against matched terminations.

Driven source + PML termination, both honoured by `ProblemTD` straight
from the geometry API. The T-shaped air guide is tiled by face-adjacent
boxes so `g.fragment` alone stitches it conformally — no boolean union
needed. PEC walls are the TD operator's default on every non-port face.
"""

# %% Parameters
import numpy as np

import rapidfem as rf

mm = 1e-3
W = 16.0 * mm            # square guide cross-section
L_ARM = 44.0 * mm        # each crossbar output arm
L_STEM = 44.0 * mm       # the input stem
PML_T = 20.0 * mm        # matched-absorber slab thickness

# %% Geometry: a T-junction air guide, PML-terminated on all three arms.
# The T is tiled by non-overlapping, face-adjacent boxes (left arm, centre
# junction, right arm, stem); g.fragment stitches them and the PML slabs
# into one conformal mesh.
g = rf.Geometry(maxh=W / 3.5)
air = rf.Air()
left = g.box(L_ARM, W, W, position=(-W / 2 - L_ARM, -W / 2, -W / 2),
             material=air)
hub = g.box(W, W, W, position=(-W / 2, -W / 2, -W / 2), material=air)
right = g.box(L_ARM, W, W, position=(W / 2, -W / 2, -W / 2), material=air)
stem = g.box(W, L_STEM, W, position=(-W / 2, -W / 2 - L_STEM, -W / 2),
             material=air)

x_in = W / 2 + L_ARM                 # crossbar arm-end coordinate
y_in = -W / 2 - L_STEM               # stem-end coordinate
pml_xm = g.box(PML_T, W, W, position=(-x_in - PML_T, -W / 2, -W / 2),
               material=air)
pml_xp = g.box(PML_T, W, W, position=(x_in, -W / 2, -W / 2), material=air)
pml_ys = g.box(W, PML_T, W, position=(-W / 2, y_in - PML_T, -W / 2),
               material=air)
g.fragment(left, hub, right, stem, pml_xm, pml_xp, pml_ys)

rf.PML(pml_xm, direction=(-1, 0, 0), inner_face=-x_in, thickness=PML_T)
rf.PML(pml_xp, direction=(1, 0, 0), inner_face=x_in, thickness=PML_T)
rf.PML(pml_ys, direction=(0, -1, 0), inner_face=y_in, thickness=PML_T)
g.mesh()
rf.show(g)

# %% Build the time-domain problem on the T-junction mesh
ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"DGTD power divider - {ptd.n_dof // 60} tets, "
      f"{ptd.n_dof} state DOFs")

# %% Drive a pulse in the stem and watch it split at the junction
pulse = rf.GaussianPulse(t0=100e-12, tau=26e-12, f0=14e9)
traj = ptd.transient(
    source=((0.0, y_in + 8 * mm, 0.0), "E", "z"),
    waveform=pulse, dt=3e-12, steps=260,
)

amp = np.linalg.norm(traj, axis=1)
assert np.all(np.isfinite(traj)), "transient must stay finite"
print(f"driven transient - {traj.shape[0]} snapshots, "
      f"amplitude {amp[0]:.3f} -> peak {amp.max():.3f} -> {amp[-1]:.3f}")

# %% Visualise: the pulse dividing into the two PML-terminated arms
rf.show(traj)
