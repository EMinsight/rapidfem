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

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Each example is baked in its own subprocess (see `_bake_subprocess`) so a
# hang — most often gmsh's OpenCASCADE boolean kernel deadlocking on a dense
# geometry — takes down only its own attempt, not the whole bake.
BAKE_TIMEOUT_S = 600   # kill + retry an example that runs longer than this
BAKE_ATTEMPTS = 3      # attempts per example before it is skipped

# Bake subprocesses run OpenMP / MKL single-threaded. A threading deadlock
# needs at least two threads — pinning OpenMP and MKL to one thread removes
# the gmsh-OCC boolean-kernel deadlock at the source. rayon (the time-domain
# stepper) keeps its pool: the TD examples do not hang and stay fast.
_BAKE_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MKL_THREADING_LAYER": "SEQUENTIAL",
    "OPENBLAS_NUM_THREADS": "1",
}

# The curated web-demo set — only these examples are baked into the static
# demo. Every other script under `examples/` stays a package example
# (runnable, listed in the live `rapidfem serve` notebook) but is left out
# of the web demo: numbers-only validation/benchmark scripts and near-
# duplicate geometries do not earn a card on the landing page.
DEMO_EXAMPLES = frozenset({
    # frequency-domain
    "fd_wr90",
    "fd_coax_step",
    "fd_microstrip_line",
    "fd_iris_filter",
    "fd_patch_antenna",
    "fd_inverted_f_antenna",
    "fd_pyramidal_horn",
    "fd_dielectric_resonator",
    # RFIC layout import
    "fd_rfic_spiral_from_json",
    # time-domain — driven modal-port families + complex structures, each a
    # field animation plus a transient port/probe plot
    "td_rect_waveguide",
    "td_coax_line",
    "td_wave_port",
    "td_dielectric_step",
    "td_power_divider",
    "td_horn_radiation",
    "td_coax_open",
})

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


# ── Bulk-payload binarisation ────────────────────────────────────────────


def _extract_payloads_to_bin(record: dict, bin_dir: Path) -> list[Path]:
    """Lift every bulk numeric array out of a baked record into the two
    binary sidecars — ``<name>.geo.bin`` (mesh / geometry) and
    ``<name>.field.bin`` (field / trajectory data) — replacing each array
    in the record with a ``$bin`` reference.

    Reuses :mod:`rapidfem.ui.binpack`, the single packer the live
    WebSocket protocol shares. Returns the bin files actually written; a
    buffer holding nothing but its 8-byte header (an example with no mesh,
    or no field data) is skipped.
    """
    from rapidfem.ui import binpack

    events = [
        ev
        for cell in record.get("cells", [])
        for ev in cell.get("display_events", [])
    ]
    geo, field = binpack.pack(events)

    name = record["name"]
    written: list[Path] = []
    for blob, suffix in ((geo, "geo"), (field, "field")):
        if len(blob) > 8:  # more than the bare header
            path = bin_dir / f"{name}.{suffix}.bin"
            path.write_bytes(blob)
            written.append(path)
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
    # The bake runs in-process — we don't need the subprocess kernel from
    # rapidfem.ui.runner, just a tiny namespace + the show-capture +
    # serialize pipeline. Live serving uses a real worker subprocess.
    import rapidfem

    source = path.read_text(encoding="utf-8")
    cells = split_cells(source)

    _reset_gmsh()

    class _Kernel:
        file_path = path.name
        namespace = {
            "__name__": "__rapidfem_bake__",
            "__file__": path.name,
            "rapidfem": rapidfem,
        }

    kernel = _Kernel()

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


def _source_fingerprint_mtime() -> float:
    """Newest mtime across files whose change should invalidate every bake:
    this script, plus everything in the rapidfem package (``.py`` and the
    compiled ``_native.pyd``). An example output is considered stale
    whenever it's older than its source ``.py`` *or* this fingerprint.
    """
    here = Path(__file__).resolve().parent.parent
    pkg = here / "python" / "python_src" / "rapidfem"
    files = [Path(__file__)]
    files.extend(pkg.rglob("*.py"))
    files.extend(pkg.rglob("*.pyd"))
    files.extend(pkg.rglob("*.so"))
    return max((p.stat().st_mtime for p in files if p.is_file()), default=0.0)


def _is_fresh(json_path: Path, src_path: Path, fingerprint_mtime: float) -> bool:
    """``True`` iff ``json_path`` exists and is newer than both the source
    ``.py`` and the global source fingerprint. Stale otherwise."""
    if not json_path.is_file():
        return False
    out_mtime = json_path.stat().st_mtime
    return out_mtime > src_path.stat().st_mtime and out_mtime > fingerprint_mtime


def _bake_subprocess(name: str, log: Path) -> float | None:
    """Bake one example in a fresh ``--bake-one`` subprocess, with a timeout
    and up to :data:`BAKE_ATTEMPTS` retries. Returns the wall-clock seconds
    of the successful attempt, or ``None`` if every attempt failed.

    Isolating each example means a hang — typically gmsh's OpenCASCADE
    boolean kernel deadlocking on a dense geometry — only takes down its
    own attempt: the subprocess is killed and retried, and the bake moves
    on. Every attempt's full output is appended to `log` for later
    diagnosis; a one-line verdict goes to the console.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), "--bake-one", name]
    for attempt in range(1, BAKE_ATTEMPTS + 1):
        t0 = time.perf_counter()
        outcome: str
        detail = ""
        try:
            proc = subprocess.run(
                cmd, timeout=BAKE_TIMEOUT_S, capture_output=True, text=True,
                env=_BAKE_ENV,
            )
            dt = time.perf_counter() - t0
            if proc.returncode == 0:
                print(f"   baked {name} in {dt:.1f}s (attempt {attempt})",
                      file=sys.stderr)
                _append_log(log, name, attempt, "ok", proc.stdout, proc.stderr)
                return dt
            outcome = f"exit {proc.returncode} after {dt:.1f}s"
            detail = (proc.stderr or "")
            _append_log(log, name, attempt, outcome, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired as e:
            dt = time.perf_counter() - t0
            outcome = f"TIMED OUT after {BAKE_TIMEOUT_S}s — killed"
            detail = (e.stderr or b"").decode("utf-8", "replace") \
                if isinstance(e.stderr, bytes) else (e.stderr or "")
            _append_log(log, name, attempt, outcome, "", detail)
        print(f"   attempt {attempt}/{BAKE_ATTEMPTS} for {name}: {outcome}",
              file=sys.stderr)
        for line in detail.strip().splitlines()[-6:]:
            print(f"     | {line}", file=sys.stderr)
    return None


def _append_log(log: Path, name: str, attempt: int, outcome: str,
                out: str, err: str) -> None:
    """Append one bake attempt's full output to the diagnostics log."""
    try:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n{'='*70}\n{name}  attempt {attempt}  -> {outcome}\n")
            if out.strip():
                fh.write(f"--- stdout ---\n{out}\n")
            if err.strip():
                fh.write(f"--- stderr ---\n{err}\n")
    except OSError:
        pass


def bake_all(force: bool = False) -> dict:
    """Bake every example under ``rapidfem/examples/`` into ``static/demo/``.

    Lazy by default — examples whose ``.json`` output is newer than both
    the source ``.py`` and the rapidfem package fingerprint are reused.
    Pass ``force=True`` to bake everything regardless.

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

    examples_dir = _examples_dir()
    # Only the curated demo set is baked — see DEMO_EXAMPLES.
    py_files = sorted(p for p in examples_dir.glob("*.py")
                      if p.stem in DEMO_EXAMPLES)
    if not py_files:
        raise FileNotFoundError(
            f"no DEMO_EXAMPLES .py files under {examples_dir}")
    missing = DEMO_EXAMPLES - {p.stem for p in py_files}
    if missing:
        print(f"warning: DEMO_EXAMPLES not found on disk: "
              f"{', '.join(sorted(missing))}", file=sys.stderr)

    fingerprint = _source_fingerprint_mtime()
    expected_names = {p.stem for p in py_files}

    # Prune orphan artefacts (an example was renamed or removed). We only
    # touch files that match an output naming pattern; user-dropped files
    # in static/demo/ get left alone.
    for old in list(out_dir.glob("*.json")):
        if old.name == "manifest.json":
            continue
        if old.stem not in expected_names:
            try:
                old.unlink()
            except OSError:
                pass
    for old in list(out_dir.glob("*.bin")):
        # bin names: <example>.geo.bin / <example>.field.bin
        prefix = old.name.rsplit(".", 2)[0]
        if prefix not in expected_names:
            try:
                old.unlink()
            except OSError:
                pass

    # Per-attempt diagnostics for every subprocess bake — start fresh.
    bake_log = out_dir / "_bake.log"
    bake_log.write_text(
        f"bake_demo run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )

    entries: list[dict] = []
    baked = 0
    reused = 0
    skipped: list[str] = []
    for path in py_files:
        json_path = out_dir / f"{path.stem}.json"
        fresh = (not force) and _is_fresh(json_path, path, fingerprint)

        if fresh:
            # Re-use existing artefacts; just rebuild the manifest entry
            # from the on-disk record so summaries stay in sync.
            record = json.loads(json_path.read_text(encoding="utf-8"))
            bin_files = sorted(out_dir.glob(f"{path.stem}.*.bin"))
            json_bytes = json_path.stat().st_size
            bin_bytes = sum(p.stat().st_size for p in bin_files)
            dt = 0.0
            print(f"── reusing {path.name}  ({json_bytes:,} B json"
                  f" + {bin_bytes:,} B bin)", file=sys.stderr)
            reused += 1
        else:
            print(f"\n── baking {path.name}", file=sys.stderr)
            # Each example runs in its own subprocess: a gmsh hang takes
            # down only its own attempt, not the whole bake.
            dt_val = _bake_subprocess(path.stem, bake_log)
            if dt_val is None:
                print(f"!! SKIPPED {path.name} — failed all "
                      f"{BAKE_ATTEMPTS} attempts (see {bake_log.name})",
                      file=sys.stderr)
                skipped.append(path.name)
                continue
            dt = dt_val
            record = json.loads(json_path.read_text(encoding="utf-8"))
            bin_files = sorted(out_dir.glob(f"{path.stem}.*.bin"))
            json_bytes = json_path.stat().st_size
            bin_bytes = sum(p.stat().st_size for p in bin_files)
            print(
                f"   {json_bytes:>9,} B json  +  {bin_bytes:>9,} B bin"
                f"   ({len(bin_files)} bin)",
                file=sys.stderr,
            )
            baked += 1

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
        f"\nwrote manifest.json with {len(entries)} example(s)"
        f" ({baked} baked, {reused} reused, {len(skipped)} skipped);"
        f" total {total_json:,} B json + {total_bin:,} B bin"
        f" = {total_json + total_bin:,} B",
        file=sys.stderr,
    )
    if skipped:
        print(f"!! {len(skipped)} example(s) skipped after {BAKE_ATTEMPTS} "
              f"attempts each: {', '.join(skipped)}", file=sys.stderr)
        print(f"   per-attempt diagnostics: {bake_log}", file=sys.stderr)
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

    bin_files = _extract_payloads_to_bin(record, out_dir)
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
    # Default: lazy bake (only stale examples). --force re-bakes everything.
    force = "--force" in sys.argv[1:]
    bake_all(force=force)
    raise SystemExit(0)
