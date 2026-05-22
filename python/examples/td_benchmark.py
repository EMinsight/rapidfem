"""Time-domain DGTD performance benchmark on a real unstructured mesh.

Where the Rust `bench` example measures the DGTD hot path on a structured
box, this measures it on the geometry a production run actually sees: the
dielectric ring resonator of ``td_ring_resonator.py``, a torus embedded
in an air cavity, meshed unstructured by gmsh. That matters for the flux
term's neighbour gather: on a structured box neighbouring elements sit
close in memory, on a real gmsh mesh the element numbering scatters them,
which is the access pattern the operator hits in practice.

Reported per mesh resolution:

* matvec throughput, ``rhs`` (one ``A.y``), in Mdof/s;
* step cost, one Krylov step, both the adaptive production path
  (``KRYLOV_TOL``) and the fixed-dimension worst case (``tol=0``);
* the matvec / orthogonalisation split, where the worst-case step's time
  goes, the breakdown that decides where a tune (or an accelerator) pays.

Run after ``maturin develop --release``:

    python python/examples/td_benchmark.py
"""

import time

import numpy as np

import rapidfem as rf

mm = 1e-3

# Ring-resonator geometry, identical to examples/td_ring_resonator.py.
R_MAJ = 11.0 * mm        # ring radius, tube-centre to torus axis
R_MIN = 2.6 * mm         # ring tube (cross-section) radius
ER = 10.0                # high-permittivity ceramic ring
BOX = 38.0 * mm          # cubic air-cavity edge

KRYLOV_DIM = 40          # Krylov-dimension cap, the solver default
DT = 4e-12               # transient step of the ring-resonator example
PROPAGATE_STEPS = 12     # steps run to reach a representative mid-transient
                         # state (a smooth field, not a raw delta pulse)


def build(maxh_air, maxh_ring):
    """A meshed ProblemTD for the ring resonator at the given resolution."""
    g = rf.Geometry(maxh=maxh_air)
    air = g.box(BOX, BOX, BOX, position=(-BOX / 2, -BOX / 2, -BOX / 2),
                material=rf.Air())
    ring = g.torus(R_MAJ, R_MIN, material=rf.Dielectric(er=ER),
                   maxh=maxh_ring)
    g.fragment(air, ring)
    rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"),
           air.faces.min(axis="y"), air.faces.max(axis="y"),
           air.faces.min(axis="z"), air.faces.max(axis="z"))
    g.mesh()
    return rf.ProblemTD(g, order=2, flux="upwind")


def median(reps, f):
    """Median wall-clock seconds of `reps` runs of `f`."""
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


def representative_state(ptd):
    """A mid-transient field state: a probe-point pulse propagated a few
    steps, so the Krylov benchmark sees a smooth field rather than the
    delta pulse the adaptive subspace would find artificially hard."""
    y = np.zeros(ptd.n_dof)
    y[ptd.probe_dof((R_MAJ, 0.0, 0.0), field="E", component="z")] = 1.0
    for _ in range(PROPAGATE_STEPS):
        y = ptd.step(y, DT, KRYLOV_DIM)
    return y


def main():
    print("rapidfem-td performance benchmark on a real unstructured mesh")
    print("geometry: dielectric ring resonator (torus in a PEC air cavity)")
    print(f"krylov-dim cap {KRYLOV_DIM}, dt {DT:g} s\n")

    # Coarse to fine, spanning the 1e5 to 1e6 state-DOF production regime.
    resolutions = [
        (6.0 * mm, 3.0 * mm),
        (4.5 * mm, 2.3 * mm),
        (3.4 * mm, 1.8 * mm),
        (2.7 * mm, 1.5 * mm),
    ]

    print(f"{'tets':>8} {'n_dof':>10} {'rhs [ms]':>10} {'Mdof/s':>9} "
          f"{'adaptive':>11} {'fixed [ms]':>11} {'matvec':>9} "
          f"{'ortho':>9} {'ortho%':>7}")

    for maxh_air, maxh_ring in resolutions:
        ptd = build(maxh_air, maxh_ring)
        n = ptd.n_dof
        y = representative_state(ptd)

        ptd.rhs(y)  # warm
        t_rhs = median(50, lambda: ptd.rhs(y))
        mdofs = n / t_rhs / 1e6

        # Adaptive step, the production path (KRYLOV_TOL); an easy step
        # converges in far fewer than KRYLOV_DIM matvecs.
        ptd.step(y, DT, KRYLOV_DIM)
        t_adaptive = median(15, lambda: ptd.step(y, DT, KRYLOV_DIM))

        # Fixed-dimension worst case: tol=0 runs the full KRYLOV_DIM, so
        # the matvec share is exactly KRYLOV_DIM matvecs and the remainder
        # is the CGS2 orthogonalisation.
        ptd.step(y, DT, KRYLOV_DIM, tol=0.0)
        t_fixed = median(15, lambda: ptd.step(y, DT, KRYLOV_DIM, tol=0.0))
        matvec = t_rhs * KRYLOV_DIM
        ortho = max(t_fixed - matvec, 0.0)

        print(f"{n // 60:>8} {n:>10} {t_rhs * 1e3:>10.3f} {mdofs:>9.1f} "
              f"{t_adaptive * 1e3:>10.2f} {t_fixed * 1e3:>11.2f} "
              f"{matvec * 1e3:>9.2f} {ortho * 1e3:>9.2f} "
              f"{100 * ortho / t_fixed:>6.0f}%")


if __name__ == "__main__":
    main()
