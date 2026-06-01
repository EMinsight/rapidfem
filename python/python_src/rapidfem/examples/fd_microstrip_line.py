"""50 ohm microstrip line on RO4003C — Z0 sweet-spot extraction.

A straight 30 mm microstrip section on 0.508 mm Rogers RO4003C
(er = 3.55), with lumped ports at both ends. The sweep is centred on the
half-wavelength resonance near 3 GHz where lumped ports see a clean
travelling-wave impedance.
"""

# %% Parameters
import math
import numpy as np
import rapidfem as rf

mm = 1e-3

SUB_H  = 0.508 * mm
ER_SUB = 3.55
TAND   = 0.0027

LINE_W = 1.13 * mm
LINE_L = 30.0 * mm

SUB_W  = 20.0 * mm
AIR_H  = 10.0 * mm

FREQUENCIES = np.linspace(2.85e9, 3.30e9, 21)
MAXH = rf.lambda_maxh(f_max=3.3e9, er_max=ER_SUB)


# %% Geometry + Materials
g = rf.Geometry(maxh=MAXH)

# Substrate is the thin feature — fix its mesh at ~1.5× its own thickness so
# the dielectric carries 3-4 cells through. Air can stay on the global
# wavelength cap.
fr4 = rf.Dielectric(er=ER_SUB, tand=TAND, maxh=1.5 * SUB_H)

sub = g.box(SUB_W, LINE_L, SUB_H, position=(-SUB_W / 2, 0, 0), material=fr4)
air = g.box(SUB_W, LINE_L, AIR_H, position=(-SUB_W / 2, 0, SUB_H),
            material=rf.Air())

trace = g.xy_plate(LINE_W, LINE_L, position=(-LINE_W / 2, 0, SUB_H))

port_in = g.plate(
    p0=(-LINE_W / 2, 0, 0),
    width=(LINE_W, 0, 0),
    height=(0, 0, SUB_H),
)
port_out = g.plate(
    p0=(-LINE_W / 2, LINE_L, 0),
    width=(LINE_W, 0, 0),
    height=(0, 0, SUB_H),
)

g.fragment(sub, air, trace, port_in, port_out)

rf.LumpedPort(port_in,  direction=(0, 0, 1), z0=50.0)
rf.LumpedPort(port_out, direction=(0, 0, 1), z0=50.0)
rf.PEC(trace, sub.faces.min(axis="z"))   # trace + ground plane
rf.ABC(*air.faces.outer)

rf.show(g)


# %% Mesh
g.auto_refine_features(base_maxh=MAXH)
g.mesh()
rf.show(g)


# %% Problem + Sweep
prob = rf.Problem(g)
result = prob.sweep(FREQUENCIES)
rf.show(prob)
rf.show(result)

s11_db = [20 * math.log10(max(abs(result.sparams[i, 0, 0]), 1e-12))
          for i in range(len(FREQUENCIES))]
s21_db = [20 * math.log10(max(abs(result.sparams[i, 1, 0]), 1e-12))
          for i in range(len(FREQUENCIES))]
print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")
print(f"|S11| across band: {min(s11_db):.1f} to {max(s11_db):.1f} dB")
print(f"|S21| across band: {min(s21_db):.2f} to {max(s21_db):.2f} dB")
