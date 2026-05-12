"""50 ohm microstrip line on RO4003C — Z0 extraction via S-parameters.

A straight 50 mm microstrip section with lumped ports at both ends. The
substrate is 0.508 mm Rogers RO4003C (er = 3.55). The trace width is
sized for ~50 ohm characteristic impedance using the standard W/H ratio
for thin-substrate microstrip.

Run the sweep, then check |S11| << 1 across the band — that confirms
the line is well-matched, which only happens if the trace is genuinely
~50 ohm. A mismatched line would show pronounced |S11| ripples.

Notebook-style:  Parameters -> Geometry -> Mesh -> Simulation
"""

# %% Parameters
import numpy as np
import rapidfem

mm = 1e-3

# Substrate (RO4003C laminate)
SUB_H  = 0.508 * mm           # dielectric thickness
ER_SUB = 3.55                 # Rogers RO4003C
TAND   = 0.0027

# Trace — 50 ohm @ 0.508 mm RO4003 lands near W = 1.13 mm.
LINE_W = 1.13 * mm
LINE_L = 30.0 * mm                # ~1 λg at 6 GHz — enough to see the match

# Substrate + air-box extents. Lateral PEC walls are far enough away
# (~10 W) that they don't perturb the characteristic impedance.
SUB_W  = 12.0 * mm
AIR_H  = 6.0 * mm                 # air headroom above the trace

# Driven sweep across the L–S band.
FREQUENCIES = np.linspace(1.0e9, 6.0e9, 21)

# Mesh density — substrate gets ~1 element through (W/H ratio is the
# Z0-defining feature, not the bulk dielectric resolution).
MAXH = 1.5 * mm


# %% Geometry + Materials
g = rapidfem.Geometry()

# Substrate slab + air above.
sub = g.box(SUB_W, LINE_L, SUB_H,
            position=(-SUB_W / 2, 0, 0))
air = g.box(SUB_W, LINE_L, AIR_H,
            position=(-SUB_W / 2, 0, SUB_H))

# Conducting trace on top of the substrate.
trace = g.xy_plate(LINE_W, LINE_L,
                   position=(-LINE_W / 2, 0, SUB_H))

# Vertical lumped-port rectangles at each end of the trace, spanning the
# substrate gap from ground (z=0) to trace (z=SUB_H).
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

# Boundaries
sub.faces.min(axis="z").name = "ground_pec"   # ground plane
trace.name = "trace_pec"
port_in.name = "port_in"
port_out.name = "port_out"

# Lateral + top walls of the air box → ABC, so the line is "open"
# (radiation-free in the model, but no box-cavity resonances).
for face in air.faces:
    if face.name is None:
        face.name = "abc"

sub.material = "ro4003"
air.material = "air"

rapidfem.show(g)


# %% Mesh
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
    .material("ro4003", er=ER_SUB, tand=TAND)
    .material("air", er=1.0)
    .build()
)
result = sim.run_sweep()
rapidfem.show(sim)
rapidfem.show(result)

import math
s11_db = [20 * math.log10(max(abs(result.sparams[i, 0, 0]), 1e-12))
          for i in range(len(FREQUENCIES))]
s21_db = [20 * math.log10(max(abs(result.sparams[i, 1, 0]), 1e-12))
          for i in range(len(FREQUENCIES))]
print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")
print(f"|S11| across band: {min(s11_db):.1f} to {max(s11_db):.1f} dB")
print(f"|S21| across band: {min(s21_db):.2f} to {max(s21_db):.2f} dB")
print("(a well-matched 50 ohm line shows |S11| below -20 dB and |S21| near 0)")
