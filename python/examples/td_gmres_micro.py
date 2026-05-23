"""Microsecond-level timing of one shift-invert GMRES call.

Builds the smallest possible TD operator that still has lumped ports,
then triggers shift-invert with r = 4 and times it. If a single shift-
invert build on a tiny problem takes more than a few seconds, the
GMRES inner loop is buggy.
"""
import time
import numpy as np
import rapidfem as rf

mm = 1e-3

# Tiny straight microstrip: 5mm long, coarser mesh.
SUB_H, ER, TAND = 0.508 * mm, 3.55, 0.0027
LINE_W, LINE_L = 1.13 * mm, 5.0 * mm
SUB_W, AIR_H = 8.0 * mm, 4.0 * mm

g = rf.Geometry(maxh=2.0 * mm)
fr4 = rf.Dielectric(er=ER, tand=TAND, maxh=2.0 * SUB_H)
sub = g.box(SUB_W, LINE_L, SUB_H, position=(-SUB_W / 2, 0, 0), material=fr4)
air = g.box(SUB_W, LINE_L, AIR_H, position=(-SUB_W / 2, 0, SUB_H),
            material=rf.Air())
trace = g.xy_plate(LINE_W, LINE_L, position=(-LINE_W / 2, 0, SUB_H))
p_in  = g.plate(p0=(-LINE_W / 2, 0, 0), width=(LINE_W, 0, 0), height=(0, 0, SUB_H))
p_out = g.plate(p0=(-LINE_W / 2, LINE_L, 0), width=(LINE_W, 0, 0), height=(0, 0, SUB_H))
g.fragment(sub, air, trace, p_in, p_out)
rf.LumpedPort(p_in,  direction=(0, 0, 1), z0=50.0)
rf.LumpedPort(p_out, direction=(0, 0, 1), z0=50.0)
rf.PEC(trace, sub.faces.min(axis="z"))
rf.ABC(*air.faces.outer, order=1)
g.mesh()

ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"DOFs: {ptd.n_dof}, ports: {ptd._op.n_ports()}")

# Time a single apply on this size.
y = np.random.randn(ptd.n_dof)
t0 = time.perf_counter()
N = 20
for _ in range(N):
    _ = ptd._op.apply(y)
t_apply = (time.perf_counter() - t0) / N
print(f"one apply (n_dof={ptd.n_dof}): {t_apply*1e3:.1f} ms")

# Now try shift-invert at small r and see total wall time.
for r in (2, 4, 8, 20, 40):
    t0 = time.perf_counter()
    try:
        mac = ptd.macromodel(r=r, shift_freq_hz=3.0e9)
        t_si = time.perf_counter() - t0
        print(f"shift-invert r={r}: {t_si:.2f} s "
              f"(expected ~{r*60*2*t_apply:.2f} s if GMRES uses all 60 iters)")
    except Exception as e:
        t_si = time.perf_counter() - t0
        print(f"shift-invert r={r} FAILED after {t_si:.2f} s: {e}")
