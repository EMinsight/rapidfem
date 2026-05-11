"""Pre-bake bundled examples for the static web demo.

Reads each `python/python_src/rapidfem/examples/*.py`, splits it into
notebook cells along `# %%` markers, runs each cell through the same
kernel pipeline the live server uses, captures every `rapidfem.show()`
display payload + stdout/stderr, and writes the recordings to
``python/python_src/rapidfem/ui/frontend-src/static/demo/``.

Run from the repo root:

    python scripts/bake_demo.py

Reused machinery (single source of truth for runtime behaviour):
- gmsh / capture lifecycle    → rapidfem.ui.kernel
- show() collector            → rapidfem._show_capture
- serializer (display events) → rapidfem.ui.serialize / api._serialize_captures_for_protocol
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Windows console defaults to cp1252 — print() falls over on the unicode
# box-drawing chars we use in summaries. Force UTF-8 on the std streams.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# Matches a line that, after optional whitespace, starts with `# %%`.
# Same predicate as Notebook.svelte's `parse()` (regex /^\s*#\s*%%/).
_MARKER_RE = re.compile(r"^\s*#\s*%%")


@dataclass
class Cell:
    """One notebook cell — body text plus its marker line (if any)."""

    text: str
    marker: str | None  # the `# %%` line that introduces this cell, or None
                       # for the implicit first cell when the file does not
                       # start with a marker.

    @property
    def has_code(self) -> bool:
        """True if the cell body has anything other than whitespace."""
        return self.text.strip() != ""


def split_cells(source: str) -> list[Cell]:
    """Split Python source into cells along `# %%` marker lines.

    Mirrors the parsing logic in ``Notebook.svelte:parse`` so cell
    boundaries in the bake step match what users see in the live UI:

    - A marker is any line matching ``^\\s*#\\s*%%``.
    - Text before the first marker becomes an implicit first cell with
      ``marker=None``. If that cell is empty (file starts with a marker),
      it is dropped.
    - The marker line itself is stored on the cell that follows it, NOT
      included in ``cell.text``, so executing ``cell.text`` is safe and
      matches what the live editor sends to the kernel.
    """
    lines = source.split("\n")
    out: list[Cell] = []
    buf: list[str] = []
    marker: str | None = None

    def flush() -> None:
        out.append(Cell(text="\n".join(buf), marker=marker))

    for line in lines:
        if _MARKER_RE.match(line):
            flush()
            marker = line
            buf = []
        else:
            buf.append(line)
    flush()

    # Drop leading empty cell if the file starts with a marker.
    if len(out) > 1 and out[0].marker is None and out[0].text.strip() == "":
        out.pop(0)

    return out


def serialize_cells(cells: Iterable[Cell]) -> str:
    """Inverse of split_cells — round-trip-safe within a tolerance.

    Reassembles the source from cells; intended as a sanity check, not
    as a canonical formatter. Reproduces the same shape Notebook.svelte's
    ``serialize()`` produces.
    """
    parts: list[str] = []
    for c in cells:
        if c.marker is not None:
            parts.append(c.marker)
        parts.append(c.text)
    return "\n".join(parts)


# ── Per-cell exec via the live kernel pipeline ───────────────────────────


def _bake_cell(cell: Cell, kernel) -> dict:
    """Run a single cell through the same pipeline the WS server uses and
    return a JSON-serialisable record of what happened.

    Reuses the production helpers — ``_show_capture`` for ``rapidfem.show()``
    payloads, ``_capture_streams`` for native fd-level stdout/stderr,
    ``_serialize_captures_for_protocol`` for the display-event shape — so
    a baked run cannot diverge from a live run by accident.
    """
    # Imported lazily so a `python scripts/bake_demo.py` that only runs the
    # cell-splitter smoke test does not require the [ui] extra (Flask, etc.).
    from rapidfem import _show_capture
    from rapidfem.ui.api import (
        _capture_streams,
        _format_exception,
        _serialize_captures_for_protocol,
    )

    stream_lines: list[dict] = []

    def on_line(stream: str, line: str) -> None:
        stream_lines.append({"stream": stream, "line": line})

    err: BaseException | None = None
    _show_capture.start_capture()
    try:
        with _capture_streams(on_line=on_line, stage="cell"):
            try:
                exec(compile(cell.text, kernel.file_path or "<cell>", "exec"),
                     kernel.namespace)
            except BaseException as e:  # noqa: BLE001 — same shape as kernel_ws
                err = e
    finally:
        captured = _show_capture.stop_capture()

    record: dict = {
        "marker": cell.marker,
        "code": cell.text,
        "stream_lines": stream_lines,
    }

    if err is not None:
        record["status"] = "error"
        record["error"] = _format_exception(err)
        record["display_events"] = []
        return record

    try:
        display_events = _serialize_captures_for_protocol(captured)
    except Exception as e:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = _format_exception(e)
        record["display_events"] = []
        return record

    record["status"] = "ok"
    record["display_events"] = display_events
    return record


# ── Field payload binarisation ───────────────────────────────────────────

# Magic + version for the field-bin format. Anything reading these files must
# verify both before slicing — float32 byte-soup with the wrong stride is a
# pain to debug after the fact.
_FIELD_BIN_MAGIC = 0x52464D46  # "RFMF" little-endian
_FIELD_BIN_VERSION = 1


def _binarise_fields(fields) -> tuple[bytes, dict] | None:
    """Pack a ``[n_freq][n_port]`` nested list of per-node field arrays into
    a single binary buffer, returning ``(bytes, replacement_dict)``.

    Layout (little-endian):
        u32 magic, u32 version, u32 n_freq, u32 n_port, u32 stride
        u8[n_freq*n_port]                    presence mask (0 = null, 1 = present)
        float32[stride] * sum(mask)          concatenated payloads, row-major (f,p)

    ``stride`` is the per-(f,p) array length. We assert it is uniform across
    every non-null entry — the FEM mesh is global, so this holds in practice;
    if it ever doesn't we want to fail loudly rather than silently truncate.

    Returns ``None`` if the input is missing/empty/all-null (nothing to bake).
    """
    import struct

    if fields is None:
        return None
    n_freq = len(fields)
    if n_freq == 0:
        return None
    n_port = len(fields[0]) if isinstance(fields[0], list) else 0
    if n_port == 0:
        return None

    # Find stride from the first non-null entry; verify uniformity.
    stride: int | None = None
    for row in fields:
        for x in row:
            if x is not None:
                if stride is None:
                    stride = len(x)
                elif len(x) != stride:
                    raise ValueError(
                        f"non-uniform field stride: {len(x)} vs {stride}"
                    )
    if stride is None:
        return None  # all-null, nothing to bake

    mask_bytes = bytearray(n_freq * n_port)
    payload_floats: list[float] = []
    for fi, row in enumerate(fields):
        for pi, x in enumerate(row):
            if x is None:
                continue
            mask_bytes[fi * n_port + pi] = 1
            payload_floats.extend(x)

    header = struct.pack("<5I", _FIELD_BIN_MAGIC, _FIELD_BIN_VERSION,
                         n_freq, n_port, stride)
    payload = struct.pack(f"<{len(payload_floats)}f", *payload_floats)
    buf = bytes(header) + bytes(mask_bytes) + payload

    return buf, {
        "$bin": True,
        "magic": _FIELD_BIN_MAGIC,
        "version": _FIELD_BIN_VERSION,
        "n_freq": n_freq,
        "n_port": n_port,
        "stride": stride,
    }


def _extract_fields_to_bin(record: dict, bin_dir: Path) -> list[Path]:
    """Walk a baked record, pull field arrays out of every result event,
    write them as .bin sidecars, and replace the array in-place with a
    binary-reference stub.

    Returns the list of bin files written, in deterministic order.
    """
    written: list[Path] = []
    name = record["name"]

    for ci, cell in enumerate(record.get("cells", [])):
        for di, ev in enumerate(cell.get("display_events", [])):
            if ev.get("kind") != "result":
                continue
            payload = ev.get("payload") or {}
            fields = payload.get("fields")
            res = _binarise_fields(fields)
            if res is None:
                continue
            buf, stub = res
            bin_name = f"{name}_c{ci}_d{di}.bin"
            stub["url"] = bin_name
            (bin_dir / bin_name).write_bytes(buf)
            written.append(bin_dir / bin_name)
            payload["fields"] = stub

    return written


def _reset_gmsh() -> None:
    """Wipe gmsh state between examples so OCC geometry from one file
    doesn't leak into the next. Same call the WS kernel uses on reset."""
    try:
        import gmsh
        if gmsh.isInitialized():
            gmsh.clear()
    except Exception:
        pass


def bake_example(path: Path) -> dict:
    """Bake one example file end-to-end.

    Returns ``{name, source, cells: [<cell records>]}``. Cells run
    sequentially against a single persistent namespace — the same model
    the live notebook uses, so variables defined in cell N are visible
    in cell N+1.
    """
    from rapidfem.ui.kernel import Kernel

    source = path.read_text(encoding="utf-8")
    cells = split_cells(source)

    _reset_gmsh()
    kernel = Kernel(file_path=path.name)

    cell_records: list[dict] = []
    for c in cells:
        if not c.has_code:
            # Pure-whitespace cell (e.g. the implicit one ahead of a
            # docstring-only file). Preserve the marker but skip exec.
            cell_records.append({
                "marker": c.marker,
                "code": c.text,
                "status": "ok",
                "stream_lines": [],
                "display_events": [],
            })
            continue
        cell_records.append(_bake_cell(c, kernel))

    return {
        "name": path.stem,
        "filename": path.name,
        "source": source,
        "cells": cell_records,
    }


# ── Smoke test / dev entry point ─────────────────────────────────────────
def _smoke() -> int:
    """Walk the bundled examples and print a cell summary for each."""
    here = Path(__file__).resolve().parent.parent
    examples_dir = here / "python" / "python_src" / "rapidfem" / "examples"
    files = sorted(p for p in examples_dir.glob("*.py") if not p.name.startswith("_"))
    if not files:
        print(f"no examples found under {examples_dir}")
        return 1

    for path in files:
        src = path.read_text(encoding="utf-8")
        cells = split_cells(src)
        print(f"\n── {path.name}  ({len(cells)} cells)")
        for i, c in enumerate(cells):
            head = (c.marker or "").strip() or "(implicit first cell)"
            n_lines = c.text.count("\n") + (1 if c.text else 0)
            print(f"  [{i}] {head}  · {n_lines} line(s), {len(c.text)} chars")
        # Round-trip check
        if serialize_cells(cells) != src:
            # Tolerated only for trailing-newline edge cases.
            a = serialize_cells(cells).rstrip("\n")
            b = src.rstrip("\n")
            if a != b:
                print(f"  WARNING: round-trip diverges for {path.name}")
    return 0


def _output_dir() -> Path:
    """Where baked artefacts land: the SvelteKit static dir, copied to
    ``dist/demo/`` at build time."""
    here = Path(__file__).resolve().parent.parent
    return here / "python" / "python_src" / "rapidfem" / "ui" / "frontend-src" / "static" / "demo"


def _examples_dir() -> Path:
    here = Path(__file__).resolve().parent.parent
    return here / "python" / "python_src" / "rapidfem" / "examples"


def bake_all() -> dict:
    """Bake every example under ``rapidfem/examples/`` into ``static/demo/``.

    Writes:
      - ``<name>.json``         per example (cells + display events, field
                                arrays replaced by bin-ref stubs)
      - ``<name>_c<i>_d<j>.bin`` per field result event
      - ``manifest.json``       index: ``{examples: [{name, filename,
                                json: "<name>.json", bytes: ...}], ...}``

    Returns the manifest dict.
    """
    import json
    import time

    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wipe previous artefacts so renames / removed examples don't linger.
    for old in list(out_dir.glob("*.json")) + list(out_dir.glob("*.bin")):
        try:
            old.unlink()
        except OSError:
            pass

    examples_dir = _examples_dir()
    py_files = sorted(p for p in examples_dir.glob("*.py")
                      if not p.name.startswith("_"))
    if not py_files:
        raise FileNotFoundError(f"no .py examples under {examples_dir}")

    entries: list[dict] = []
    for path in py_files:
        t0 = time.perf_counter()
        print(f"\n── baking {path.name}", file=sys.stderr)
        record = bake_example(path)

        bin_files = _extract_fields_to_bin(record, out_dir)
        json_path = out_dir / f"{record['name']}.json"
        json_path.write_text(json.dumps(record), encoding="utf-8")

        dt = time.perf_counter() - t0
        json_bytes = json_path.stat().st_size
        bin_bytes = sum(p.stat().st_size for p in bin_files)
        print(
            f"   {json_bytes:>9,} B json  +  {bin_bytes:>9,} B bin"
            f"   ({len(bin_files)} bin)   [{dt:.1f}s]",
            file=sys.stderr,
        )

        # Per-cell status summary for the manifest (lets the FE show error
        # markers without parsing the full JSON eagerly).
        cell_summaries = [
            {
                "marker": c.get("marker"),
                "status": c.get("status"),
                "kinds": [d.get("kind") for d in c.get("display_events", [])],
                "stream_lines": len(c.get("stream_lines", [])),
            }
            for c in record["cells"]
        ]

        entries.append({
            "name": record["name"],
            "filename": record["filename"],
            "json": f"{record['name']}.json",
            "bin_files": [p.name for p in bin_files],
            "n_cells": len(record["cells"]),
            "cells": cell_summaries,
            "json_bytes": json_bytes,
            "bin_bytes": bin_bytes,
            "bake_seconds": round(dt, 2),
        })

    manifest = {
        "version": 1,
        "baked_at": int(time.time()),
        "examples": entries,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_json = sum(e["json_bytes"] for e in entries)
    total_bin = sum(e["bin_bytes"] for e in entries)
    print(
        f"\nwrote manifest.json with {len(entries)} example(s);"
        f" total {total_json:,} B json + {total_bin:,} B bin"
        f" = {total_json + total_bin:,} B",
        file=sys.stderr,
    )
    return manifest


def _bake_one(name: str) -> int:
    """Bake a single example end-to-end, write JSON + field .bin sidecars
    to ``static/demo/``, and print a size report."""
    import json
    here = Path(__file__).resolve().parent.parent
    target = here / "python" / "python_src" / "rapidfem" / "examples" / name
    if not target.exists() and target.suffix != ".py":
        target = target.with_suffix(".py")
    if not target.exists():
        print(f"example not found: {target}", file=sys.stderr)
        return 1

    print(f"baking {target.name}...", file=sys.stderr)
    record = bake_example(target)

    for i, c in enumerate(record["cells"]):
        head = (c["marker"] or "").strip() or "(implicit)"
        ds = c.get("display_events", [])
        kinds = ",".join(d.get("kind", "?") for d in ds) or "—"
        sl = len(c.get("stream_lines", []))
        status = c["status"]
        print(f"  [{i}] {head:30s}  status={status:5s}  display=[{kinds}]  stream={sl}",
              file=sys.stderr)

    out_dir = _output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_json_size = len(json.dumps(record))

    bin_files = _extract_fields_to_bin(record, out_dir)
    json_path = out_dir / f"{record['name']}.json"
    json_path.write_text(json.dumps(record), encoding="utf-8")

    post_json_size = json_path.stat().st_size
    bin_total = sum(p.stat().st_size for p in bin_files)
    print(
        f"\nwrote {json_path.name} ({post_json_size:,} B)"
        f" + {len(bin_files)} bin file(s) totalling {bin_total:,} B",
        file=sys.stderr,
    )
    print(
        f"  JSON savings: {raw_json_size:,} -> {post_json_size:,} B  "
        f"({100 * (1 - post_json_size / raw_json_size):.1f}% reduction)",
        file=sys.stderr,
    )
    print(
        f"  combined on-disk: {post_json_size + bin_total:,} B "
        f"({(post_json_size + bin_total) / raw_json_size * 100:.1f}% of raw JSON)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--bake-one":
        if len(sys.argv) < 3:
            print("usage: bake_demo.py --bake-one <example_name>", file=sys.stderr)
            raise SystemExit(2)
        raise SystemExit(_bake_one(sys.argv[2]))
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        raise SystemExit(_smoke())
    # Default: bake everything.
    bake_all()
    raise SystemExit(0)
