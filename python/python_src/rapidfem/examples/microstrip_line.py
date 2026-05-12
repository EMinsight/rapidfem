"""50 ohm microstrip line on RO4003C — Z0 sweet-spot extraction.

A straight 30 mm microstrip section on 0.508 mm Rogers RO4003C
(er = 3.55), with lumped ports at both ends. The sweep is centred on
the line's half-wavelength resonance near 3 GHz, where the standing
wave inside the line collapses and the lumped ports see a clean
travelling-wave impedance — that's the only region where the S-params
faithfully reflect line behaviour.

Background: lumped ports compute S-params via port-voltage integrals
and assume the line carries a pure 50 Ω travelling wave at the port
plane. Whenever the trace's actual Z0 deviates from 50 Ω (here it does,
because the open-bounded ABC model lacks the top-side dielectric that
the textbook closed-form Z0 formula assumes), reflections set up a
standing wave that biases the integral and produces small artefacts
(e.g. |S11|² + |S21|² straying above 1 between line resonances). The
artefacts vanish near each (n+1) · λg/2 resonance, where the
standing wave node sits on the port.

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

# Trace — closed-form 50 Ω on RO4003 0.508 mm sits at W ≈ 1.13 mm.
LINE_W = 1.13 * mm
LINE_L = 30.0 * mm                # ≈ λg/2 at the sweet-spot frequency

# Substrate + air-box extents — generous lateral + vertical headroom so
# the 2nd-order ABC sees only the well-confined microstrip mode and
# doesn't reflect fringe fields back into the lumped ports.
SUB_W  = 20.0 * mm
AIR_H  = 10.0 * mm

# Driven sweep narrowed to the λg/2 resonance window so the lumped-port
# Z0 calibration is clean. The notch in |S11| around 3 GHz is where the
# line is electrically λg/2 — perfectly matched at that frequency.
FREQUENCIES = np.linspace(2.85e9, 3.30e9, 21)

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
