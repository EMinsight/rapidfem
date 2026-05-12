"""End-to-end HTTP smoke test for the subprocess-based runner.

Hits /api/cell/run + /api/cell/poll, runs the same 4 WR-90 cells the WS
smoke used to. Reports messages received and overall success.
"""
from __future__ import annotations

import json
import sys
import time

import urllib.request


BASE = "http://127.0.0.1:5174"
FILE_KEY = "smoke/wr90.py"


WR90_CELLS = [
    """\
import numpy as np
import rapidfem
A, B, L = 22.86e-3, 10.16e-3, 30.0e-3
FREQUENCIES = np.linspace(8.0e9, 12.0e9, 21)
MAXH = 5.0e-3
""",
    """\
g = rapidfem.Geometry()
air = g.box(A, B, L, position=(-A / 2, -B / 2, 0))
air.material = "air"
air.faces.min(axis="z").name = "port_in"
air.faces.max(axis="z").name = "port_out"
for face in air.faces:
    if face.name is None:
        face.name = "pec"
rapidfem.show(g)
""",
    """\
g.mesh(maxh=MAXH)
rapidfem.show(g)
""",
    """\
sim = (
    rapidfem.SimulationBuilder()
    .mesh_from(g)
    .frequencies(FREQUENCIES)
    .rect_waveguide("port_in")
    .rect_waveguide("port_out")
    .pec("pec")
    .material("air", er=1.0)
    .build()
)
result = sim.run_sweep()
rapidfem.show(sim)
rapidfem.show(result)
print(f"DOFs: {sim.n_dofs}, tets: {sim.n_tets}")
""",
]


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def run_one_cell(idx: int, code: str, reset: bool) -> bool:
    print(f"\n=== cell {idx} ===")
    t0 = time.perf_counter()
    resp = post("/api/cell/run", {"file": FILE_KEY, "code": code, "reset": reset})
    cell_id = resp.get("cell_id")
    print(f"  started, cell_id={cell_id}")
    n_events = 0
    total_bytes = 0
    while True:
        poll = post("/api/cell/poll", {"file": FILE_KEY})
        msgs = poll.get("messages") or []
        n_events += len(msgs)
        for m in msgs:
            total_bytes += len(json.dumps(m))
            t = m.get("type")
            if t == "stream":
                pass  # quiet
            elif t == "display":
                size = len(json.dumps(m))
                print(f"  [display kind={m.get('kind')} {size//1024} KiB]")
            elif t == "error":
                print(f"  [error] {m.get('error')[:200]}")
                return False
            elif t == "done":
                dt = time.perf_counter() - t0
                print(f"  done ok={m.get('ok')} after {n_events} events, "
                      f"total {total_bytes/1024:.1f} KiB, {dt:.2f}s")
                return bool(m.get("ok"))
            elif t == "worker-exit":
                print("  worker exited unexpectedly!")
                return False
        if not msgs:
            # No new messages — give the worker a moment, then continue polling.
            # The server's poll endpoint already long-polls 100 ms.
            pass
        if time.perf_counter() - t0 > 60:
            print("  TIMEOUT after 60s")
            return False


def main() -> int:
    # Health check first
    try:
        with urllib.request.urlopen(BASE + "/api/health", timeout=2) as r:
            h = json.loads(r.read())
        print(f"server up: {h}")
    except Exception as e:
        print(f"cannot reach {BASE}: {e}")
        return 2

    ok_all = True
    for i, code in enumerate(WR90_CELLS):
        if not run_one_cell(i, code, reset=(i == 0)):
            ok_all = False
            break

    if ok_all:
        print("\nALL CELLS COMPLETED SUCCESSFULLY")
        return 0
    print("\nFAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
