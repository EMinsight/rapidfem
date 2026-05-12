"""HTTP smoke for dielectric_resonator.py — runs the example via /api/cell/run."""
import json
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:5174"
FILE_KEY = "smoke/dr.py"


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def main():
    src = open("python/python_src/rapidfem/examples/dielectric_resonator.py", encoding="utf-8").read()
    cells = [c for c in re.split(r"^# %%.*$", src, flags=re.M) if c.strip()]
    print(f"running {len(cells)} cells")
    for i, code in enumerate(cells):
        cell_id = f"c{i}"
        print(f"\n=== cell {i} ===")
        r = post("/api/cell/run", {"file": FILE_KEY, "code": code, "reset": i == 0, "cell_id": cell_id})
        if not r.get("ok"):
            print(f"  run failed: {r}"); return 1
        t0 = time.perf_counter()
        events = 0
        last_error = None
        while True:
            poll = post("/api/cell/poll", {"file": FILE_KEY})
            msgs = poll.get("messages") or []
            events += len(msgs)
            for m in msgs:
                t = m.get("type")
                if t == "stream":
                    v = m.get("value", "").rstrip().encode("ascii", "replace").decode("ascii")
                    if v: print(f"  [{m.get('stream')}] {v[:200]}")
                elif t == "display":
                    size = len(json.dumps(m))
                    print(f"  [display kind={m.get('kind')} {size//1024} KiB name={m.get('name')}]")
                    if m.get("kind") == "result":
                        payload = m.get("payload", {})
                        print(f"     eigenmode={payload.get('eigenmode')} n_freq={payload.get('n_freq')} n_driven={payload.get('n_driven')}")
                        print(f"     frequencies={[f'{f/1e9:.4f}' for f in payload.get('frequencies', [])][:5]}")
                        fields = payload.get('fields') or []
                        print(f"     fields shape: outer={len(fields)} inner_first={len(fields[0]) if fields else None}")
                elif t == "error":
                    print(f"  [ERROR] {m.get('error')[:200]}")
                    last_error = m
                elif t == "done":
                    if m.get("id") == cell_id:
                        print(f"  done ok={m.get('ok')} after {events} events in {time.perf_counter()-t0:.1f}s")
                        if not m.get("ok"):
                            return 1
                        break
                elif t == "worker-exit":
                    print(f"  [WORKER EXIT]"); return 1
            else:
                if time.perf_counter() - t0 > 60:
                    print("  TIMEOUT"); return 1
                continue
            break
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
