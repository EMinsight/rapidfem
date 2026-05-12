"""Run dielectric_resonator.py through the actual worker subprocess.

This bypasses Flask/HTTP entirely — pure JSON over stdin/stdout against the
same worker.py the live serve uses. Reports each message the worker emits
so we can see exactly where a 'cell failed' originates.
"""
import json
import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "python/python_src/rapidfem/examples/dielectric_resonator.py"
WORKER = Path(__file__).resolve().parents[1] / "python/python_src/rapidfem/ui/worker.py"


def send(p, msg):
    p.stdin.write(json.dumps(msg) + "\n")
    p.stdin.flush()


def read_until_done(p, cell_id):
    while True:
        line = p.stdout.readline()
        if not line:
            print("WORKER EXITED")
            err = p.stderr.read()
            if err:
                print("STDERR:", err)
            return False
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            print(" RAW:", line[:200])
            continue
        t = msg.get("type")
        if t == "stream":
            v = msg.get("value", "").rstrip()
            if v:
                safe = v.encode("ascii", errors="replace").decode("ascii")
                print(f"  [{msg.get('stream')}] {safe[:200]}")
        elif t == "display":
            kind = msg.get("kind")
            size = len(json.dumps(msg))
            print(f"  [display kind={kind} {size//1024} KiB]")
        elif t == "error":
            print(f"  [ERROR id={msg.get('id')}] {msg.get('error')[:200]}")
            if msg.get("traceback"):
                print("   TB:", msg.get("traceback")[:500])
            return False
        elif t == "done":
            if msg.get("id") == cell_id:
                print(f"  done ok={msg.get('ok')}")
                return msg.get("ok", False)


def main():
    src = EXAMPLE.read_text(encoding="utf-8")
    # Split on '# %%' markers like the notebook does
    import re
    parts = re.split(r"^# %%.*$", src, flags=re.M)
    cells = [p for p in parts if p.strip()]
    print(f"running {len(cells)} cells through worker")

    p = subprocess.Popen(
        [sys.executable, "-u", str(WORKER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    # init
    send(p, {"type": "init"})
    # wait for ready
    while True:
        line = p.stdout.readline()
        if not line:
            print("worker died during init"); return 1
        msg = json.loads(line.strip())
        if msg.get("type") == "ready":
            break
        if msg.get("type") == "error":
            print("init error:", msg); return 1

    for i, code in enumerate(cells):
        cid = f"c{i}"
        print(f"\n=== cell {i} ===")
        send(p, {"type": "cell-run", "id": cid, "code": code})
        ok = read_until_done(p, cid)
        if not ok:
            print(f"\nFAILED at cell {i}")
            return 1

    p.stdin.close()
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
