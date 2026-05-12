"""Dielectric resonator filter — eigenmode analysis.

A high-εᵣ ceramic puck sits on a low-εᵣ alumina support inside a metal
cavity. The DR's first few TE/TM-like modes are what an RF filter would
exploit; here we solve directly for them with the shift-invert Lanczos
eigenmode driver.

The Metal box is what *makes* this a resonator — without the PEC walls
the modes would radiate away and no bound eigenmodes would exist. Tip
when viewing the result in the UI: click the ``pec`` entry in the legend
to hide the cavity walls; the mode field is concentrated inside the
high-εᵣ puck (with weak coupling into the air gap and support).

Adapted from EMerge's ``demo9_dielectric_resonator.py``:
https://github.com/FennisRobert/EMerge/blob/main/examples/demo9_dielectric_resonator.py

Notebook-style:  Parameters -> Geometry -> Mesh -> Eigenmode
"""

# %% Parameters
import numpy as np
import rapidfem

mm = 1e-3
inch = 25.4 * mm

# Cavity (PEC enclosure)
W = 2.0 * inch       # square base
S = 2.03 * inch      # height

# Support cylinder (alumina-like)
D_SUP, L_SUP = 0.56 * inch, 0.80 * inch
ER_SUP = 10.0

# Resonator puck (high-εᵣ ceramic)
D_RES, L_RES = 1.176 * inch, 0.481 * inch
ER_RES = 34.0

# Eigenmode search — first 5 modes near 2 GHz.
F_TARGET = 2.0e9
N_MODES = 5


# %% Geometry + Materials
g = rapidfem.Geometry()

# Air-filled cavity, centered on the xy-origin.
air = g.box(W, W, S, position=(-W / 2, -W / 2, 0))
air.material = "air"

# Support cylinder on the floor.
support = g.cylinder(radius=D_SUP / 2, height=L_SUP)
support.material = "alumina"

# Puck sits on top of the support.
resonator = g.cylinder(radius=D_RES / 2, height=L_RES, position=(0, 0, L_SUP))
resonator.material = "ceramic"

# Conformal cuts so the three volumes share faces.
g.fragment(air, support, resonator)

# Only the 6 axis-aligned cavity walls are PEC — selecting via min/max
# avoids tagging the cylinder-air interface faces that `fragment` exposes
# on `air.faces`. Tagging those would wall the dielectric off from the
# air and the eigenmode solver would find isolated-dielectric modes
# instead of cavity-resonator modes.
air.faces.min(axis="z").name = "pec"   # cavity floor
air.faces.max(axis="z").name = "pec"   # cavity ceiling
air.faces.min(axis="x").name = "pec"
air.faces.max(axis="x").name = "pec"
air.faces.min(axis="y").name = "pec"
air.faces.max(axis="y").name = "pec"

# Mesh — wavelength in the ceramic is √34 ≈ 5.8× shorter than air, so
# resolve there. For the cavity bulk λ_air/12 ≈ 12.5 mm at 2 GHz is fine.
resonator.maxh = rapidfem.lambda_maxh(f_max=3.0e9, er_max=ER_RES)  # ~ 1.4 mm
support.maxh = rapidfem.lambda_maxh(f_max=3.0e9, er_max=ER_SUP)    # ~ 2.6 mm

rapidfem.show(g)


# %% Mesh
g.mesh(maxh=rapidfem.lambda_maxh(f_max=3.0e9))  # air bulk
rapidfem.show(g)


# %% Eigenmode
sim = (
    rapidfem.SimulationBuilder()
    .mesh_from(g)
    .eigenmode(F_TARGET, n_modes=N_MODES)
    .pec("pec")
    .material("air", er=1.0)
    .material("alumina", er=ER_SUP)
    .material("ceramic", er=ER_RES)
    .build()
)
modes = sim.run_eigenmode()
rapidfem.show(sim)
rapidfem.show(modes)   # Mode slider + per-mode field viewer in the UI.

print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")
print(f"Found {len(modes)} modes near {F_TARGET/1e9:.2f} GHz:")
for i, m in enumerate(modes):
    f_real = m.frequency_hz / 1e9
    f_imag = m.frequency_imag_hz / 1e9
    q = m.q_factor
    q_str = f"Q={q:.1f}" if np.isfinite(q) else "Q=∞"
    print(f"  mode {i+1}: f = {f_real:.4f} GHz   (imag {f_imag:+.2e} GHz, {q_str})")
