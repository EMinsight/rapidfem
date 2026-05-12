"""Mesh-stats baseline for the smart-meshing work.

Runs each bundled example *up to and including* `g.mesh(...)` only — no solver —
and prints tet count, vertex count, edge count, and mesh wall-time. Use this to
compare the impact of meshing changes without paying for the FEM solve.

Usage:
    python scripts/mesh_bench.py             # all examples
    python scripts/mesh_bench.py wr90 patch  # only matching examples
"""
from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

EX_DIR = Path(__file__).resolve().parents[1] / "python" / "python_src" / "rapidfem" / "examples"


def run_until_mesh(path: Path) -> dict:
    """Execute the example script's cells up to (and including) g.mesh(...).

    Skips everything after the `# %% Simulation` cell so we don't try to load
    the native solver. Imports rapidfem.show as a no-op so `rapidfem.show(g)`
    inside the script doesn't try to push to a UI slot.
    """
    src = path.read_text(encoding="utf-8")
    cells = re.split(r"^# %%.*$", src, flags=re.M)
    pre_cells = []
    for cell in cells:
        pre_cells.append(cell)
        if "g.mesh(" in cell:
            break
    code = "\n".join(pre_cells)

    import rapidfem
    orig_show = rapidfem.show
    rapidfem.show = lambda *a, **k: None
    glb = {"__name__": "__main__"}
    t0 = time.perf_counter()
    try:
        exec(compile(code, str(path), "exec"), glb)
    finally:
        rapidfem.show = orig_show
    dt = time.perf_counter() - t0

    # The example calls g.mesh() which returns (bytes, name_to_tag) and stores
    # them on g._last_mesh — but the local `g` lives only in `glb`. Grab it.
    g = glb.get("g")
    if g is None:
        return {"name": path.stem, "error": "no g in script"}
    import gmsh
    n_tets = len(gmsh.model.mesh.getElementsByType(4)[0])
    n_verts = len(gmsh.model.mesh.getNodes()[0])
    # Edge count: gmsh.model.mesh.getElementEdgeNodes(4) returns 6 edges per tet,
    # each as a pair of node tags — unique pairs give global edges.
    edge_nodes = gmsh.model.mesh.getElementEdgeNodes(4)
    pairs = set()
    for i in range(0, len(edge_nodes), 2):
        a, b = edge_nodes[i], edge_nodes[i + 1]
        pairs.add((min(a, b), max(a, b)))
    n_edges = len(pairs)
    # ND-2 second-kind has 2 DoFs/edge + 2 DoFs/face. Faces:
    face_nodes = gmsh.model.mesh.getElementFaceNodes(4, 3)  # 4 faces × 3 nodes each
    face_set = set()
    for i in range(0, len(face_nodes), 3):
        tri = tuple(sorted([face_nodes[i], face_nodes[i + 1], face_nodes[i + 2]]))
        face_set.add(tri)
    n_faces = len(face_set)
    n_dofs_nd2 = 2 * n_edges + 2 * n_faces

    return {
        "name": path.stem,
        "tets": n_tets,
        "verts": n_verts,
        "edges": n_edges,
        "faces": n_faces,
        "dofs_nd2": n_dofs_nd2,
        "mesh_time_s": dt,
    }


def main(argv):
    filter_terms = [a.lower() for a in argv]
    rows = []
    examples = sorted(EX_DIR.glob("*.py"))
    examples = [p for p in examples if not p.name.startswith("_")]
    for path in examples:
        if filter_terms and not any(t in path.stem.lower() for t in filter_terms):
            continue
        # Reset gmsh between examples so old meshes don't leak in
        import gmsh
        if gmsh.isInitialized():
            gmsh.finalize()
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            row = run_until_mesh(path)
        except Exception as e:
            row = {"name": path.stem, "error": repr(e)[:80]}
        rows.append(row)

    # Pretty print
    print(f"{'example':<22} {'tets':>8} {'verts':>8} {'edges':>9} {'faces':>9} {'ND-2 DoFs':>11} {'mesh [s]':>10}")
    print("-" * 80)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<22} ERROR: {r['error']}")
            continue
        print(f"{r['name']:<22} {r['tets']:>8d} {r['verts']:>8d} {r['edges']:>9d} {r['faces']:>9d} {r['dofs_nd2']:>11d} {r['mesh_time_s']:>10.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
