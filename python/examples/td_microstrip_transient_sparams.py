"""Validates the TD port machinery on the microstrip by running the
existing transient-pulse `sparams` method (the FD-validated path).

If this gives |S21| ~ 0.88 matching FD, the port machinery is sound
and the macromodel's |S21|=0 issue is basis-specific (Krylov
truncation). If this also gives |S21|=0 or wildly off values, the
issue is in port_source / port_modal_projections for lumped ports
on this geometry.
"""
import time
import numpy as np
import rapidfem as rf

mm = 1e-3

SUB_H, ER, TAND = 0.508 * mm, 3.55, 0.0027
LINE_W, LINE_L = 1.13 * mm, 30.0 * mm
SUB_W, AIR_H = 20.0 * mm, 10.0 * mm

MAXH = rf.lambda_maxh(f_max=3.3e9, er_max=ER)


def build_geometry():
    g = rf.Geometry(maxh=MAXH)
    fr4 = rf.Dielectric(er=ER, tand=TAND, maxh=1.5 * SUB_H)
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


print("=== Microstrip: TD transient sparams vs FD ===\n")

print("[1] FD reference ...")
g_fd = build_geometry()
prob_fd = rf.Problem(g_fd)
FREQS = np.linspace(2.85e9, 3.30e9, 9)
res_fd = prob_fd.sweep(FREQS)
s_fd = res_fd.sparams

print("\n[2] TD transient sparams (drive each port with a broadband pulse) ...")
g_td = build_geometry()
ptd = rf.ProblemTD(g_td, order=2, flux="upwind")
print(f"    DOFs: {ptd.n_dof}, ports: {ptd._op.n_ports()}")

# Time-domain S-parameters with the documented transient path.
# A 4 ns window with 0.5 ps step captures one round-trip on a 30 mm
# line in air (~200 ps each way) plus several wavelengths at 3 GHz.
t0 = time.perf_counter()
scattering = ptd.sparams(
    FREQS, dt=2.0e-12, steps=1500, krylov_dim=30, verbose=True,
)
t_td = time.perf_counter() - t0
_, s_td = scattering
print(f"    transient sparams: {t_td:.1f} s")

print(f"\n{'f [GHz]':>8} {'|S11| TD':>10} {'|S11| FD':>10} "
      f"{'|S21| TD':>10} {'|S21| FD':>10}")
for k, f in enumerate(FREQS):
    print(
        f"{f/1e9:>8.2f} {abs(s_td[k,0,0]):>10.3f} "
        f"{abs(s_fd[k,0,0]):>10.3f} {abs(s_td[k,1,0]):>10.3f} "
        f"{abs(s_fd[k,1,0]):>10.3f}"
    )

d11 = float(np.max(np.abs(np.abs(s_td[:, 0, 0]) - np.abs(s_fd[:, 0, 0]))))
d21 = float(np.max(np.abs(np.abs(s_td[:, 1, 0]) - np.abs(s_fd[:, 1, 0]))))
print(f"\nmax |S11| deviation: {d11:.3f}")
print(f"max |S21| deviation: {d21:.3f}")

if d21 < 0.1 and abs(s_td[:, 1, 0]).min() > 0.5:
    print("\n=> Port machinery is sound. Macromodel |S21|=0 is basis-specific.")
elif abs(s_td[:, 1, 0]).max() < 0.1:
    print(
        "\n=> Transient ALSO returns |S21|~0. Port wiring (port_source / "
        "port_modal_projections) is broken on this lumped-port setup."
    )
else:
    print("\n=> Transient is partially right, partially off. Mixed problem.")
