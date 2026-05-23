"""Minimal Python reproduction of the validated Rust two-lumped-port
test: an empty box with two lumped ports on opposite faces. If this
gives |S21| > 0, the issue is specifically with the microstrip
geometry (trace, substrate, fragmentation). If it also gives
|S21| = 0, the Python pipeline is dropping something the Rust test
keeps intact.
"""
import time
import numpy as np
import rapidfem as rf

# Box dimensions: same shape as the rust test (2 x 0.5 x 6 in
# operator units). Pick physical sizes so the c=1 normalisation
# the rust test uses matches a free-space band of order 1 GHz.
mm = 1e-3
A, B, L = 200 * mm, 50 * mm, 600 * mm

g = rf.Geometry(maxh=L / 16)
air = g.box(A, B, L, position=(-A / 2, -B / 2, 0), material=rf.Air())
port_in = g.xy_plate(A, B, position=(-A / 2, -B / 2, 0))
port_out = g.xy_plate(A, B, position=(-A / 2, -B / 2, L))
g.fragment(air, port_in, port_out)
rf.LumpedPort(port_in,  direction=(0, 1, 0), z0=50.0)
rf.LumpedPort(port_out, direction=(0, 1, 0), z0=50.0)
# ABC on the side walls of the box (matches the rust validated test,
# which uses absorbing-only mode=None ports on the side walls).
rf.ABC(
    air.faces.min(axis="x"), air.faces.max(axis="x"),
    air.faces.min(axis="y"), air.faces.max(axis="y"),
    order=1,
)
g.mesh()

ptd = rf.ProblemTD(g, order=2, flux="central")
op = ptd._op
print(f"DOFs: {ptd.n_dof}, ports: {op.n_ports()}")
for k in range(op.n_ports()):
    print(
        f"  port {k}: cutoff = {op.port_cutoff(k):.3g}, "
        f"face pairs = {op.port_n_faces(k)}, "
        f"interior = {op.port_n_interior_faces(k)}, "
        f"||b|| = {np.linalg.norm(op.port_source(k)):.3e}"
    )

# Drive port 0 with a Gaussian pulse, probe modal projections at
# BOTH ports over time. Use the documented transient sparams flow.
FREQS = np.linspace(0.3e9, 0.6e9, 5)
print("\nrunning transient sparams ...")
t0 = time.perf_counter()
scattering = ptd.sparams(
    FREQS, dt=1.0e-12, steps=2500, krylov_dim=30, verbose=True,
)
t_td = time.perf_counter() - t0
_, S = scattering
print(f"transient: {t_td:.1f} s")

print(f"\n{'f [GHz]':>9} {'|S11|':>8} {'|S21|':>8} {'|S12|':>8} {'|S22|':>8}")
for k, f in enumerate(FREQS):
    print(
        f"{f/1e9:>9.3f} {abs(S[k,0,0]):>8.3f} {abs(S[k,1,0]):>8.3f} "
        f"{abs(S[k,0,1]):>8.3f} {abs(S[k,1,1]):>8.3f}"
    )
