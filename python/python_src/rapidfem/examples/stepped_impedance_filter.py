"""Microstrip stepped-impedance low-pass filter.

A 7-section trace of alternating wide / narrow strips on a thin substrate
acts as a classic Richards-style LPF: wide sections look capacitive, narrow
sections inductive. The cutoff is around 2 GHz; the upper stop-band has
the characteristic ripples + harmonic re-entry.

Adapted from EMerge's ``demo1_stepped_imp_filter.py`` (PCB-layouter
section dimensions kept verbatim — except we build the trace as plain
xy_plates instead of going through a routing DSL):
https://github.com/FennisRobert/EMerge/blob/main/examples/demo1_stepped_imp_filter.py

Notebook-style:  Parameters -> Geometry -> Mesh -> Simulation
"""

# %% Parameters
import numpy as np
import rapidfem

mm = 1e-3
mil = 0.0254 * mm

# Section lengths (mil) and widths (mil) — EMerge dimensions.
LENGTHS_MIL = [400, 660, 660, 660, 660, 660, 400]
WIDTHS_MIL  = [ 50, 128,   8, 224,   8, 128,  50]

SUB_H = 62 * mil            # PCB substrate thickness
ER_SUB = 2.2                # low-loss laminate (Rogers-like)
AIR_H = 60 * mil            # air-box height above substrate
PAD_Y = 200 * mil           # lateral margin around the trace

FREQUENCIES = np.linspace(0.2e9, 8.0e9, 41)


# %% Geometry + Materials
# Trace runs along +x. Centre the assembly at x = 0 so the lumped ports
# sit at the substrate's ±x ends.
LENGTHS = [L * mil for L in LENGTHS_MIL]
WIDTHS  = [W * mil for W in WIDTHS_MIL]
total_L = sum(LENGTHS)
sub_W   = max(WIDTHS) + 2 * PAD_Y

g = rapidfem.Geometry()

# Substrate slab (centred on the trace axis).
sub = g.box(total_L, sub_W, SUB_H,
            position=(-total_L / 2, -sub_W / 2, 0))
sub.material = "ro4003"

# Air box on top.
air = g.box(total_L, sub_W, AIR_H,
            position=(-total_L / 2, -sub_W / 2, SUB_H))
air.material = "air"

# Trace segments — one xy_plate per section, butted edge-to-edge.
x_cursor = -total_L / 2
trace_plates: list[rapidfem.GeoObject] = []
for L_seg, W_seg in zip(LENGTHS, WIDTHS):
    plate = g.xy_plate(L_seg, W_seg,
                       position=(x_cursor, -W_seg / 2, SUB_H))
    trace_plates.append(plate)
    x_cursor += L_seg

# Lumped-port plates at each end — vertical rectangles from ground to trace.
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

# Conformal cut so the trace + ports + substrate share faces cleanly.
g.fragment(sub, air, *trace_plates, port_in, port_out)

# Conductive surfaces.
sub.faces.min(axis="z").name = "ground_pec"   # bottom = ground plane
for plate in trace_plates:
    plate.name = "trace_pec"
port_in.name = "port_in"
port_out.name = "port_out"

# Lateral + top walls of the air box → ABC (open boundary).
for face in air.faces:
    if face.name is None:
        face.name = "abc"

# Mesh density: λ_air/12 globally, with substrate getting its own size to
# resolve the higher-εᵣ wavelength.
MAXH = rapidfem.lambda_maxh(f_max=8.0e9)
sub.maxh = rapidfem.lambda_maxh(f_max=8.0e9, er_max=ER_SUB)

rapidfem.show(g)


# %% Mesh
# Auto-refine for the very narrow W2=8 mil section (≈0.2 mm) — that strip
# defines the high-impedance behaviour and must be resolved.
g.auto_refine_features(base_maxh=MAXH, min_maxh=0.3e-3)
g.mesh(maxh=MAXH)
rapidfem.show(g)


# %% Simulation
sim = (
    rapidfem.SimulationBuilder()
    .mesh_from(g)
    .frequencies(FREQUENCIES)
    .pec("ground_pec", "trace_pec")
    .lumped_port("port_in",  direction=(0, 0, 1), z0=50.0)
    .lumped_port("port_out", direction=(0, 0, 1), z0=50.0)
    .abc("abc", order=2)
    .material("ro4003", er=ER_SUB)
    .material("air", er=1.0)
    .build()
)
result = sim.run_sweep()
rapidfem.show(sim)
rapidfem.show(result)

import math
s21_db = [20 * math.log10(max(abs(result.sparams[i, 1, 0]), 1e-12))
          for i in range(len(FREQUENCIES))]
print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")
# A stepped-impedance LPF should show clear roll-off above ~2 GHz.
cutoff_idx = next((i for i, db in enumerate(s21_db) if db < -3), len(FREQUENCIES) - 1)
print(f"|S21| 3-dB cutoff near {FREQUENCIES[cutoff_idx]/1e9:.2f} GHz "
      f"(stop-band floor {min(s21_db):.1f} dB)")
