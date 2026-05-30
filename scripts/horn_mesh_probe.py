"""Mesh-only probe of the FD horn to size up tet count and memory.

Runs only the geometry + mesh stages of `fd_pyramidal_horn.py`, so we can
gauge how big the system will be before paying for a full sweep. Prints
tet count, n_dofs estimate (per Nedelec-2: ~20 edge DOFs / tet), and the
sparse-direct RAM ballpark.
"""
from __future__ import annotations

import sys

import gmsh

import rapidfem as rf


def main():
    # ── Inline the geometry / mesh stages from fd_pyramidal_horn.py ──────
    mm = 1e-3
    wga, wgb = 22.86 * mm, 10.16 * mm
    Lfeed = 15.0 * mm
    Lhorn = 50.0 * mm
    WH, HH = 30.0 * mm, 22.0 * mm
    LPAD_BEAM = 88.0 * mm
    LPAD_SIDE = 30.0 * mm
    PML_T = 15.0 * mm
    F0 = 10.0e9
    MAXH = rf.lambda_maxh(f_max=11.0e9, per_lambda=8)
    MAXH_AIR = rf.lambda_maxh(f_max=11.0e9, per_lambda=2)

    AIR_X0, AIR_X1 = -Lfeed, Lhorn + LPAD_BEAM
    AIR_Y0, AIR_Y1 = -WH / 2 - LPAD_SIDE, WH / 2 + LPAD_SIDE
    AIR_Z0, AIR_Z1 = -HH / 2 - LPAD_SIDE, HH / 2 + LPAD_SIDE

    g = rf.Geometry(maxh=MAXH_AIR)
    air = g.box(AIR_X1 - AIR_X0, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
                position=(AIR_X0, AIR_Y0, AIR_Z0), material=rf.Air())
    pml_xp = g.box(PML_T, AIR_Y1 - AIR_Y0, AIR_Z1 - AIR_Z0,
                   position=(AIR_X1, AIR_Y0, AIR_Z0),
                   material=rf.Air(), maxh=2 * MAXH)
    feed = g.box(Lfeed, wga, wgb, position=(-Lfeed, -wga / 2, -wgb / 2),
                 material=rf.Air(), maxh=wgb / 3)
    throat = g.polygon([
        (0, -wga / 2, -wgb / 2), (0,  wga / 2, -wgb / 2),
        (0,  wga / 2,  wgb / 2), (0, -wga / 2,  wgb / 2),
    ])
    aperture = g.polygon([
        (Lhorn, -WH / 2, -HH / 2), (Lhorn,  WH / 2, -HH / 2),
        (Lhorn,  WH / 2,  HH / 2), (Lhorn, -WH / 2,  HH / 2),
    ])
    horn = g.loft(throat, aperture, material=rf.Air(), maxh=MAXH)
    g.fragment(air, feed, horn, pml_xp)

    rf.RectWaveguidePort(feed.faces.min(axis="x"), mode=(1, 0), power=1.0)
    rf.PEC(feed.faces.min(axis="y"), feed.faces.max(axis="y"),
           feed.faces.min(axis="z"), feed.faces.max(axis="z"))
    rf.PEC(*horn.faces.where(lambda c, b: 1e-6 < c[0] < Lhorn - 1e-6))
    rf.PML(pml_xp, direction=(1, 0, 0), inner_face=Lhorn + LPAD_BEAM,
           thickness=PML_T)
    rf.PEC(*pml_xp.faces.outer)
    rf.ABC(*air.faces.outer.unassigned, order=1)

    g.mesh()

    # ── Stats ────────────────────────────────────────────────────────────
    _, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    n_tets = len(elem_node_tags[0]) // 4 if elem_node_tags else 0
    n_dofs = 20 * n_tets        # rough Nedelec-2 edge-DOF count
    # Sparse-direct LU RAM: O(N^1.5) for 3-D; constant ~50 bytes per nnz of L,
    # nnz(L) ~ N^1.5 for unstructured 3-D meshes. Roughly:
    ram_gb = 50e-9 * n_dofs ** 1.5

    print(f"BOX volume:  ({AIR_X1 - AIR_X0:.3f}, "
          f"{AIR_Y1 - AIR_Y0:.3f}, {AIR_Z1 - AIR_Z0:.3f}) m")
    print(f"LPAD_BEAM:   {LPAD_BEAM * 1000:.0f} mm  (+x for PML)")
    print(f"LPAD_SIDE:   {LPAD_SIDE * 1000:.0f} mm  (y/z for ABC)")
    print(f"MAXH:        {MAXH * 1000:.2f} mm  (horn / feed)")
    print(f"MAXH_AIR:    {MAXH_AIR * 1000:.2f} mm  (outer air)")
    print(f"n_tets:      {n_tets:,}")
    print(f"n_dofs (~):  {n_dofs:,}")
    print(f"sparse-LU RAM ballpark: {ram_gb:.1f} GB")
    sys.exit(0)


if __name__ == "__main__":
    main()
