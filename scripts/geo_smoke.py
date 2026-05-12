"""Smoke-test the new geometry-builder API: polygon/disc/extrude/revolve
plus rotate/stretch/intersect. Builds three quick geometries, meshes
each, and prints tet count + wall time. No solver.

Run from the repo root:

    python scripts/geo_smoke.py
"""
from __future__ import annotations

import math
import time
import sys

import rapidfem


def case_revolved_horn() -> dict:
    """Conical horn from a revolved 4-point profile (EMerge demo5 shape).
    Exercises polygon + revolve.
    """
    cm = 0.01
    waveguide_radius = 2.779 / 2 * cm
    waveguide_length = 2.872 * cm
    aperture_radius = 10.334 / 2 * cm
    aperture_length = 7.809 * cm

    g = rapidfem.Geometry()
    # Profile in xy-plane, will revolve around x-axis (sweep through y-z).
    profile = g.polygon([
        (0, 0),
        (waveguide_length, 0),
        (waveguide_length + aperture_length, 0),
        (waveguide_length + aperture_length, aperture_radius),
        (waveguide_length, waveguide_radius),
        (0, waveguide_radius),
    ])
    horn = g.revolve(profile, axis_point=(0, 0, 0), axis_dir=(1, 0, 0))
    horn.material = "air"

    t0 = time.perf_counter()
    g.mesh(maxh=rapidfem.lambda_maxh(f_max=12e9))
    return {"name": "revolved_horn", "dt": time.perf_counter() - t0}


def case_extruded_hex_trace() -> dict:
    """Hex-section microstrip-style trace: polygon + extrude + stretch."""
    g = rapidfem.Geometry()
    R = 5e-3
    pts = [(R * math.cos(t), R * math.sin(t))
           for t in (i * math.pi / 3 for i in range(6))]
    hex_face = g.polygon(pts)
    trace = g.extrude(hex_face, height=50e-3)
    trace.material = "air"
    # Stretch test: squash by 10% along y.
    g.stretch(trace, fy=0.9)

    t0 = time.perf_counter()
    g.mesh(maxh=5e-3)
    return {"name": "extruded_hex_trace", "dt": time.perf_counter() - t0}


def case_clipped_disc() -> dict:
    """disc extruded + intersected with a half-space-ish box.
    Exercises disc + extrude + intersect + rotate.
    """
    g = rapidfem.Geometry()
    d = g.disc(radius=20e-3)
    cyl = g.extrude(d, height=40e-3)
    cyl.material = "air"
    # Rotate cylinder 30 deg around y-axis.
    g.rotate(cyl, angle=math.radians(30), axis=(0, 1, 0))
    clip = g.box(60e-3, 60e-3, 30e-3, position=(-30e-3, -30e-3, 0))
    g.intersect(cyl, clip)

    t0 = time.perf_counter()
    g.mesh(maxh=4e-3)
    return {"name": "clipped_disc", "dt": time.perf_counter() - t0}


def main() -> int:
    cases = [case_revolved_horn, case_extruded_hex_trace, case_clipped_disc]
    failed = 0
    for fn in cases:
        try:
            r = fn()
            print(f"  {r['name']:<22}  mesh {r['dt']*1000:7.1f} ms  OK")
        except Exception as e:
            print(f"  {fn.__name__:<22}  FAIL  {type(e).__name__}: {e}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
