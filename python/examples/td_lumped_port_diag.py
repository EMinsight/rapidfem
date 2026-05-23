"""Diagnose why the TD macromodel sees zero coupling between two
lumped ports on the microstrip example.

Inspects port_source / port_modal_projections directly, then walks
the impulse-Krylov chain `b, A*b, A^2*b, ...` to check whether the
field propagated from port 0 ever reaches port 1's modal-projection
support.

If port_modal_projections(A^k * port_source(0), port=1) is zero for
all k up to some depth that physically spans the line, then either:
  - port_source produces a vector that does not couple to the
    propagating modes (lumped-port wiring bug), or
  - port_modal_projections at port 1 reads a different field
    component than what port_source at port 0 excites (B / C
    mismatch on lumped ports), or
  - the operator drops the signal in transit (flux dissipation,
    geometric isolation between ports).
"""
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

MAXH = rf.lambda_maxh(f_max=3.3e9, er_max=ER_SUB)


def build_geometry():
    g = rf.Geometry(maxh=MAXH)
    fr4 = rf.Dielectric(er=ER_SUB, tand=TAND, maxh=1.5 * SUB_H)
    sub = g.box(SUB_W, LINE_L, SUB_H, position=(-SUB_W / 2, 0, 0), material=fr4)
    air = g.box(SUB_W, LINE_L, AIR_H, position=(-SUB_W / 2, 0, SUB_H),
                material=rf.Air())
    trace = g.xy_plate(LINE_W, LINE_L, position=(-LINE_W / 2, 0, SUB_H))
    port_in = g.plate(
        p0=(-LINE_W / 2, 0, 0), width=(LINE_W, 0, 0), height=(0, 0, SUB_H),
    )
    port_out = g.plate(
        p0=(-LINE_W / 2, LINE_L, 0), width=(LINE_W, 0, 0), height=(0, 0, SUB_H),
    )
    g.fragment(sub, air, trace, port_in, port_out)
    rf.LumpedPort(port_in,  direction=(0, 0, 1), z0=50.0)
    rf.LumpedPort(port_out, direction=(0, 0, 1), z0=50.0)
    rf.PEC(trace, sub.faces.min(axis="z"))
    rf.ABC(*air.faces.outer, order=1)
    g.auto_refine_features(base_maxh=MAXH)
    g.mesh()
    return g


g = build_geometry()
ptd = rf.ProblemTD(g, order=2, flux="central")
op = ptd._op
n_ports = op.n_ports()
print(f"DOFs: {ptd.n_dof}, total ports (incl. ABC): {n_ports}")
# Modal ports are the two lumped ports declared first in the geometry;
# the third entry is the ABC face which has no mode.
MODAL_PORTS = [0, 1]
for k in MODAL_PORTS:
    print(f"  port {k} cutoff = {op.port_cutoff(k):.3g}")

# Step 1: inspect port_source for each modal port.
print("\n[1] port_source magnitudes")
for k in MODAL_PORTS:
    b = op.port_source(k)
    print(f"  port {k}: ||b|| = {np.linalg.norm(b):.4e}, "
          f"non-zero entries = {int((b != 0).sum())} / {b.size}")

# Step 2: self-projection — port_modal_projections at port k of its
# own source vector. Tells us how port_source couples back into the
# modal-projection space.
print("\n[2] port_modal_projections(port_source(j), port=i)")
print(f"     {'P_e':>12} {'P_h':>12}")
for i in MODAL_PORTS:
    for j in MODAL_PORTS:
        b = op.port_source(j)
        p_e, p_h = op.port_projections(b, i)
        print(f"  i={i} j={j}: {p_e:>12.4e} {p_h:>12.4e}")

# Step 3: walk the impulse Krylov from port 0's source vector and
# read out the modal projection at port 1 every K matvecs.
# Normalise between steps - the raw Krylov chain explodes by ~rho(A)
# per matvec (~1e11 in physical units), so we re-normalise to unit
# norm after each apply and track the projection ratio against the
# normalised vector, which is what the macromodel actually uses.
print("\n[3] modal projection at port 1 after k matvecs of port 0's source")
print("    (vector re-normalised to unit norm each step)")
print(f"     {'k':>4} {'P_e(p1)/||v||':>16} {'P_h(p1)/||v||':>16}")
v = op.port_source(0)
v = v / np.linalg.norm(v)
p_e, p_h = op.port_projections(v, 1)
print(f"     {0:>4} {p_e:>16.4e} {p_h:>16.4e}")
K_MAX = 200
STRIDE = 5
for k in range(1, K_MAX + 1):
    v = op.apply(v)
    nv = np.linalg.norm(v)
    if nv > 0:
        v = v / nv
    if k <= 10 or k % STRIDE == 0:
        p_e, p_h = op.port_projections(v, 1)
        print(f"     {k:>4} {p_e:>16.4e} {p_h:>16.4e}")
        if k >= 50 and abs(p_e) < 1e-15 and abs(p_h) < 1e-15:
            # Skip ahead if still exactly zero - clearer signal.
            pass
