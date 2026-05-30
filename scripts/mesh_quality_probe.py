"""Probe mesh quality with and without the WP1 mesher changes.

Meshes the coax-monopole geometry (the stiff `td_coax_open` demo) two
ways:
    1. Legacy: Mesh.Algorithm3D = 1 (serial Delaunay), no optimize pass.
    2. New default: HXT + ``optimize="Netgen"`` post-pass.

Reports tet-quality histograms and the smallest tet's `cfl_dt`-relevant
edge-length ratio so the sliver-killing benefit is measurable, not just
a vibes-based "feels cleaner".

Run with:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/mesh_quality_probe.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

import rapidfem as rf
from rapidfem._native import TdOperator  # noqa: F401  (warms native import)


def build_coax_monopole():
    """Same geometry as ``examples/td_coax_open.py``."""
    mm = 1e-3
    RI, RO = 1.50e-3, 3.45e-3
    LIN = 25.0e-3
    LPROT = 10.0e-3
    BW = 44.0e-3
    BL = 70.0e-3
    Z0 = -BL / 2
    Z1 = Z0 + LIN
    Z2 = Z1 + LPROT
    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=10.0e9))
    box = g.box(BW, BW, BL, position=(-BW / 2, -BW / 2, Z0), material=rf.Air())
    coax = g.cylinder(
        radius=RO, height=LIN, position=(0, 0, Z0), axis=(0, 0, 1),
        material=rf.Air(), maxh=RO / 3,
    )
    inner = g.cylinder(
        radius=RI, height=LIN + LPROT, position=(0, 0, Z0), axis=(0, 0, 1),
        material=rf.Air(), maxh=RO / 3,
    )
    g.fragment(box, coax, inner)
    rf.CoaxPort(coax.faces.min(axis="z"), ri=RI, ro=RO, origin=(0, 0, Z0))
    rf.PEC(*coax.faces.where(lambda c, b: Z0 + 1e-4 < c[2] < Z1 - 1e-4))
    rf.PEC(*inner.faces.where(lambda c, b: Z0 + 1e-4 < c[2] < Z2 - 1e-4))
    rf.PEC(inner.faces.max(axis="z"))
    rf.ABC(*box.faces.outer, order=1)
    return g, mm, LPROT


def tet_quality_stats(g):
    """Quality histogram of the meshed `g`. Reads tet coordinates from the
    gmsh model and computes the **radius ratio** `R = 3·r_in / r_out` —
    the canonical Hesthaven sliver indicator (1 for regular, 0 for
    degenerate). Returns (min, p1, p5, p50, n_tets)."""
    import gmsh

    _, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    # Tet elements (type 4 in gmsh).
    _, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    if not elem_node_tags:
        return None
    node_tags = np.asarray(elem_node_tags[0], dtype=np.int64).reshape(-1, 4)
    # gmsh node tags are 1-based and dense. Build a tag → index lookup.
    all_node_tags, _, _ = gmsh.model.mesh.getNodes()
    tag_to_idx = {int(t): i for i, t in enumerate(all_node_tags)}
    idx = np.array(
        [[tag_to_idx[int(t)] for t in row] for row in node_tags],
        dtype=np.int64,
    )
    p = coords[idx]                # (n_tets, 4, 3)
    # Edge lengths.
    edges = np.array(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        dtype=np.int64,
    )
    e = p[:, edges[:, 0]] - p[:, edges[:, 1]]  # (n_tets, 6, 3)
    elen = np.linalg.norm(e, axis=2)           # (n_tets, 6)
    # Volume per tet via the scalar triple product.
    v0 = p[:, 1] - p[:, 0]
    v1 = p[:, 2] - p[:, 0]
    v2 = p[:, 3] - p[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(v0, v1), v2)) / 6.0
    # Surface area = sum of triangle areas. The four faces of a tet (each a
    # combination of 3 of the 4 vertices).
    tri = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    area = np.zeros(p.shape[0])
    for (a, b, c) in tri:
        u = p[:, b] - p[:, a]
        v = p[:, c] - p[:, a]
        area += 0.5 * np.linalg.norm(np.cross(u, v), axis=1)
    # Inscribed sphere radius r_in = 3V / A.
    r_in = np.where(area > 0, 3.0 * vol / area, 0.0)
    # Circumscribed sphere: use the longest-edge upper bound (cheap proxy).
    # For sliver hunting we care about ratios, not absolute accuracy.
    r_out_max = elen.max(axis=1)
    radius_ratio = np.where(r_out_max > 0, 3.0 * r_in / r_out_max, 0.0)
    return {
        "n_tets": int(p.shape[0]),
        "min": float(radius_ratio.min()),
        "p01": float(np.percentile(radius_ratio, 1)),
        "p05": float(np.percentile(radius_ratio, 5)),
        "p50": float(np.percentile(radius_ratio, 50)),
        "edge_min": float(elen.min()),
        "edge_max": float(elen.max()),
        "edge_min_p01": float(np.percentile(elen.min(axis=1), 1)),
    }


def probe(algorithm, optimize, label):
    t0 = time.time()
    g, _, _ = build_coax_monopole()
    g.mesh(algorithm=algorithm, optimize=optimize)
    stats = tet_quality_stats(g)
    elapsed = time.time() - t0
    print(f"\n=== {label} ===")
    print(f"  mesher: algorithm={algorithm!r}, optimize={optimize}")
    print(f"  meshing time: {elapsed:.2f}s")
    print(f"  n_tets: {stats['n_tets']}")
    print(f"  radius ratio: min={stats['min']:.4f}, "
          f"p01={stats['p01']:.4f}, p05={stats['p05']:.4f}, "
          f"p50={stats['p50']:.4f}")
    print(f"  edge length: min={stats['edge_min']*1e3:.3f} mm, "
          f"max={stats['edge_max']*1e3:.3f} mm")
    return stats


if __name__ == "__main__":
    legacy = probe("delaunay", False, "Legacy (Delaunay, no optimize)")
    new = probe("hxt", True, "New default (HXT + Netgen optimize)")
    print("\n=== Delta ===")
    print(f"  min radius ratio:  {legacy['min']:.4f} -> {new['min']:.4f}  "
          f"({new['min'] / max(legacy['min'], 1e-12):.2f}x)")
    print(f"  p01 radius ratio:  {legacy['p01']:.4f} -> {new['p01']:.4f}")
    print(f"  smallest edge:     {legacy['edge_min']*1e6:.1f} um -> "
          f"{new['edge_min']*1e6:.1f} um")
    sys.exit(0)
