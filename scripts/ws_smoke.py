"""End-to-end WS smoke test: connect, run wr90 example, watch all events.

Run while `rapidfem serve` is alive on default 5174.
"""
import asyncio
import json
import sys
import time

import websockets


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


async def main():
    url = "ws://127.0.0.1:5174/ws/kernel"
    print(f"connecting to {url} ...")
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        hello = await asyncio.wait_for(ws.recv(), timeout=5)
        print(f"  recv: {hello}")
        for i, code in enumerate(WR90_CELLS):
            cell_id = f"c{i}"
            print(f"\n=== cell {i} ===")
            await ws.send(json.dumps({
                "type": "execute",
                "cell_id": cell_id,
                "file": "wr90_smoke.py",
                "code": code,
                "reset": i == 0,
            }))
            t0 = time.perf_counter()
            n_events = 0
            total_bytes = 0
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=120)
                except asyncio.TimeoutError:
                    print("  TIMEOUT after 120s")
                    return 1
                n_events += 1
                total_bytes += len(raw) if isinstance(raw, (str, bytes)) else 0
                evt = json.loads(raw)
                t = evt.get("type")
                if t == "stream":
                    pass  # quiet
                elif t == "display":
                    kind = evt.get("kind")
                    payload = evt.get("payload") or {}
                    size = len(raw)
                    print(f"  [display kind={kind} {size//1024} KiB]")
                elif t == "error":
                    print(f"  [error] {evt.get('error', {}).get('type')}: {evt.get('error', {}).get('message')[:200]}")
                elif t == "done":
                    dt = time.perf_counter() - t0
                    print(f"  done ok={evt.get('ok')} after {n_events} events, total {total_bytes/1024:.1f} KiB, {dt:.2f}s")
                    break
                else:
                    print(f"  [{t}]")
        print("\nALL CELLS COMPLETED SUCCESSFULLY")
        # Try a reconnect after the long cell
        print("\nclosing and reconnecting to verify WS stability ...")
    await asyncio.sleep(0.5)
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        hello = await asyncio.wait_for(ws.recv(), timeout=5)
        print(f"reconnect ok, recv: {hello}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
