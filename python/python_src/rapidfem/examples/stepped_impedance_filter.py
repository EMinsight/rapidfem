"""Microstrip stepped-impedance low-pass filter.

A 7-section trace of alternating wide / narrow strips on a thin substrate
acts as a classic Richards-style LPF. Cutoff around 2 GHz. Adapted from
EMerge's ``demo1_stepped_imp_filter.py``.
"""

# %% Parameters
import math
import numpy as np
import rapidfem as rf

mm = 1e-3
mil = 0.0254 * mm

LENGTHS_MIL = [400, 660, 660, 660, 660, 660, 400]
WIDTHS_MIL  = [ 50, 128,   8, 224,   8, 128,  50]

SUB_H = 62 * mil
ER_SUB = 2.2
AIR_H = 15 * mm
PAD_Y = 12 * mm

FREQUENCIES = np.linspace(0.2e9, 8.0e9, 41)
MAXH = rf.lambda_maxh(f_max=8.0e9)


# %% Geometry + Materials
LENGTHS = [L * mil for L in LENGTHS_MIL]
WIDTHS  = [W * mil for W in WIDTHS_MIL]
total_L = sum(LENGTHS)
sub_W   = max(WIDTHS) + 2 * PAD_Y

g = rf.Geometry(maxh=MAXH)

sub = g.box(total_L, sub_W, SUB_H, position=(-total_L / 2, -sub_W / 2, 0),
            material=rf.Dielectric(er=ER_SUB),
            maxh=rf.lambda_maxh(f_max=8.0e9, er_max=ER_SUB))
air = g.box(total_L, sub_W, AIR_H, position=(-total_L / 2, -sub_W / 2, SUB_H),
            material=rf.Air())

x_cursor = -total_L / 2
trace_plates: list = []
for L_seg, W_seg in zip(LENGTHS, WIDTHS):
    plate = g.xy_plate(L_seg, W_seg, position=(x_cursor, -W_seg / 2, SUB_H))
    trace_plates.append(plate)
    x_cursor += L_seg

W_IN = WIDTHS[0]
port_in = g.plate(
    p0=(-total_L / 2, -W_IN / 2, 0),
    width=(0, W_IN, 0),
    height=(0, 0, SUB_H),
)
port_out = g.plate(
    p0=(total_L / 2, -W_IN / 2, 0),
    width=(0, W_IN, 0),
    height=(0, 0, SUB_H),
)

g.fragment(sub, air, *trace_plates, port_in, port_out)

rf.LumpedPort(port_in,  direction=(0, 0, 1), z0=50.0)
rf.LumpedPort(port_out, direction=(0, 0, 1), z0=50.0)
rf.PEC(sub.faces.min(axis="z"), *trace_plates)
rf.ABC(*air.faces.outer, order=2)

rf.show(g)


# %% Mesh
g.auto_refine_features(base_maxh=MAXH, min_maxh=0.3e-3)
g.mesh()
rf.show(g)


# %% Problem + Sweep
prob = rf.Problem(g)
result = prob.sweep(FREQUENCIES)
rf.show(prob)
rf.show(result)

s21_db = [20 * math.log10(max(abs(result.sparams[i, 1, 0]), 1e-12))
          for i in range(len(FREQUENCIES))]
print(f"DOFs: {prob.n_dofs}, tets: {prob.n_tets}")
cutoff_idx = next((i for i, db in enumerate(s21_db) if db < -3), len(FREQUENCIES) - 1)
print(f"|S21| 3-dB cutoff near {FREQUENCIES[cutoff_idx]/1e9:.2f} GHz "
      f"(stop-band floor {min(s21_db):.1f} dB)")
