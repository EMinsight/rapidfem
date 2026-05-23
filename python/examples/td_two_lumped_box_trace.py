"""Step-by-step trace of port modal projections during a transient."""
import numpy as np
import rapidfem as rf

mm = 1e-3
A, B, L = 200 * mm, 50 * mm, 600 * mm

g = rf.Geometry(maxh=L / 16)
air = g.box(A, B, L, position=(-A / 2, -B / 2, 0), material=rf.Air())
port_in = g.xy_plate(A, B, position=(-A / 2, -B / 2, 0))
port_out = g.xy_plate(A, B, position=(-A / 2, -B / 2, L))
g.fragment(air, port_in, port_out)
rf.LumpedPort(port_in,  direction=(0, 1, 0), z0=50.0)
rf.LumpedPort(port_out, direction=(0, 1, 0), z0=50.0)
rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"),
       air.faces.min(axis="y"), air.faces.max(axis="y"))
g.mesh()

ptd = rf.ProblemTD(g, order=2, flux="central")
op = ptd._op
n = ptd.n_dof
print(f"DOFs: {n}, ports: {op.n_ports()}")

# Drive port 0 with a smooth Gaussian-modulated cosine.
c = 299792458.0
dt = 1e-12
steps = 1500
src = op.port_source(0)
g_pulse = lambda t: np.exp(-((t - 1e-9) / 3e-10) ** 2) * np.cos(2 * np.pi * 0.5e9 * t)
h_op = float(c * dt)
y = np.zeros(n)
print(f"\n  {'step':>5} {'t [ns]':>8} {'P_e p0':>11} {'P_e p1':>11} {'P_h p0':>11} {'P_h p1':>11}")
for s in range(steps):
    t = s * dt
    y = op.step_with_source(y, src * g_pulse(t), h_op, 30)
    if s % 100 == 0 or s == steps - 1:
        pe0, ph0 = op.port_projections(y, 0)
        pe1, ph1 = op.port_projections(y, 1)
        print(f"  {s:>5} {t*1e9:>8.2f} {pe0:>11.3e} {pe1:>11.3e} {ph0:>11.3e} {ph1:>11.3e}")
