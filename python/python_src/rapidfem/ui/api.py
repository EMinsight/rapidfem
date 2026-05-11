"""JSON API endpoints for the rapidfem UI.

Registered onto the Flask app by ``rapidfem.ui.server.create_app``.
"""
from __future__ import annotations

import io
import sys
import threading
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

import rapidfem
from rapidfem import _show_capture
from rapidfem.ui.bus import BUS


# Long-running operations (mesh, solve) hold this lock so the gmsh model
# isn't mutated by two requests at once. The capture slot is thread-local,
# so the lock also pins them to a single request at a time.
_pipeline_lock = threading.Lock()


def _format_exception(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _exec_user_code(code: str, workdir: Path) -> dict[str, Any]:
    """Run a piece of user code with capture active. Returns the response payload."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    namespace: dict[str, Any] = {
        "__name__": "__rapidfem_user__",
        "__file__": str(workdir / "<editor>"),
        "rapidfem": rapidfem,
    }

    _show_capture.start_capture()
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            compiled = compile(code, "<editor>", "exec")
            exec(compiled, namespace)
    except SystemExit as e:
        return {
            "ok": False,
            "error": _format_exception(e),
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "captures": [],
        }
    except BaseException as e:  # noqa: BLE001 — surface everything from user code
        return {
            "ok": False,
            "error": _format_exception(e),
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "captures": [c.name for c in _show_capture.get_captured()],
        }
    finally:
        captured = _show_capture.stop_capture()

    # Serialize each captured object. Currently only Geometry is rendered;
    # other kinds (Simulation, SweepResult) are forwarded as metadata and
    # picked up by separate endpoints (/api/mesh, /api/solve).
    from rapidfem.ui.serialize import geometry_to_payload

    rendered: list[dict[str, Any]] = []
    for item in captured:
        entry: dict[str, Any] = {"name": item.name, "kind": item.kind}
        if item.kind == "geometry":
            try:
                entry["payload"] = geometry_to_payload(item.obj)
            except Exception as e:  # noqa: BLE001 — bad geometry, surface it
                entry["error"] = _format_exception(e)
        rendered.append(entry)

    return {
        "ok": True,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "captures": rendered,
    }


def _exec_for_pipeline(code: str, workdir: Path) -> tuple[dict[str, Any], list]:
    """Run user code with capture, return (response-shell, raw captures).

    The shell is the standard /api/run-style payload (ok/error/stdout/stderr),
    minus the rendered "captures" — those are returned alongside as raw objects
    so /api/mesh and /api/solve can act on them.
    """
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    namespace: dict[str, Any] = {
        "__name__": "__rapidfem_user__",
        "__file__": str(workdir / "<editor>"),
        "rapidfem": rapidfem,
    }
    _show_capture.start_capture()
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(code, "<editor>", "exec"), namespace)
    except BaseException as e:  # noqa: BLE001
        captured = _show_capture.stop_capture()
        return {
            "ok": False,
            "error": _format_exception(e),
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
        }, captured
    captured = _show_capture.stop_capture()
    return {
        "ok": True,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
    }, captured


def _find_capture(captures: list, kind: str, name: str | None):
    for c in captures:
        if c.kind != kind:
            continue
        if name is None or c.name == name:
            return c
    return None


def register(app: Flask) -> None:
    workdir: Path = app.config["RAPIDFEM_WORKDIR"]

    @app.post("/api/run")
    def api_run():
        body = request.get_json(silent=True) or {}
        code = body.get("code", "")
        if not isinstance(code, str):
            return jsonify({"ok": False, "error": {"type": "ValueError", "message": "code must be string", "traceback": ""}}), 400
        BUS.publish({"kind": "stage_start", "stage": "run"})
        with _pipeline_lock:
            result = _exec_user_code(code, workdir)
        BUS.publish({"kind": "stage_end", "stage": "run", "ok": result["ok"]})
        return jsonify(result), 200

    @app.post("/api/mesh")
    def api_mesh():
        body = request.get_json(silent=True) or {}
        code = body.get("code", "")
        maxh = float(body.get("maxh", 0.0)) or None
        geometry_name = body.get("geometry_name")  # optional; None → first geometry
        if not isinstance(code, str):
            return jsonify({"ok": False, "error": {"type": "ValueError", "message": "code must be string", "traceback": ""}}), 400

        BUS.publish({"kind": "stage_start", "stage": "mesh"})
        with _pipeline_lock:
            shell, captures = _exec_for_pipeline(code, workdir)
            if not shell["ok"]:
                BUS.publish({"kind": "stage_end", "stage": "mesh", "ok": False})
                return jsonify(shell), 200

            cap = _find_capture(captures, "geometry", geometry_name)
            if cap is None:
                return jsonify({**shell, "ok": False, "error": {
                    "type": "LookupError",
                    "message": f"no Geometry captured (looked for name={geometry_name!r}). Did you forget rapidfem.show(g)?",
                    "traceback": "",
                }}), 200

            from rapidfem.ui.serialize import mesh_to_payload
            if maxh is None:
                from rapidfem.ui.serialize import _bbox_diag
                maxh = _bbox_diag() / 20.0
            try:
                payload = mesh_to_payload(cap.obj, maxh=maxh)
            except Exception as e:  # noqa: BLE001
                return jsonify({**shell, "ok": False, "error": _format_exception(e)}), 200

        BUS.publish({"kind": "stage_end", "stage": "mesh", "ok": True, "stats": payload["stats"]})
        return jsonify({**shell, "mesh": payload, "name": cap.name}), 200

    @app.post("/api/solve")
    def api_solve():
        body = request.get_json(silent=True) or {}
        code = body.get("code", "")
        builder_name = body.get("builder_name")  # optional; None → first builder
        if not isinstance(code, str):
            return jsonify({"ok": False, "error": {"type": "ValueError", "message": "code must be string", "traceback": ""}}), 400

        BUS.publish({"kind": "stage_start", "stage": "solve"})
        with _pipeline_lock:
            shell, captures = _exec_for_pipeline(code, workdir)
            if not shell["ok"]:
                BUS.publish({"kind": "stage_end", "stage": "solve", "ok": False})
                return jsonify(shell), 200

            cap = _find_capture(captures, "builder", builder_name)
            if cap is None:
                return jsonify({**shell, "ok": False, "error": {
                    "type": "LookupError",
                    "message": f"no SimulationBuilder captured (looked for name={builder_name!r}). Pass it through rapidfem.show(builder).",
                    "traceback": "",
                }}), 200

            try:
                import time
                t0 = time.perf_counter()
                sim = cap.obj.build()
                result = sim.run_sweep()
                t_solve = time.perf_counter() - t0
                freqs = result.frequencies.tolist()
                s = result.sparams
                n_freq, n_p, _ = s.shape
                sparams_payload = []
                for fi in range(n_freq):
                    f_mat = []
                    for r in range(n_p):
                        row = []
                        for c in range(n_p):
                            v = s[fi, r, c]
                            row.append([float(v.real), float(v.imag)])
                        f_mat.append(row)
                    sparams_payload.append(f_mat)
            except Exception as e:  # noqa: BLE001
                return jsonify({**shell, "ok": False, "error": _format_exception(e)}), 200

        BUS.publish({"kind": "stage_end", "stage": "solve", "ok": True, "solve_time_s": t_solve, "n_freq": n_freq, "n_driven": n_p})
        # Continued below.
        return jsonify({
            **shell,
            "result": {
                "frequencies": freqs,
                "sparams": sparams_payload,
                "n_driven": n_p,
                "n_freq": n_freq,
                "n_dofs": sim.n_dofs,
                "n_tets": sim.n_tets,
                "solve_time_s": t_solve,
            },
            "name": cap.name,
        }), 200

    # ── File endpoints ────────────────────────────────────────────────────────

    def _safe_path(rel: str) -> Path | None:
        """Resolve `rel` inside workdir; reject path traversal."""
        if not rel or "\x00" in rel:
            return None
        try:
            target = (workdir / rel).resolve()
        except (OSError, ValueError):
            return None
        try:
            target.relative_to(workdir)
        except ValueError:
            return None
        return target

    @app.get("/api/files")
    def api_files_list():
        out: list[dict[str, Any]] = []
        for p in sorted(workdir.rglob("*.py")):
            if any(part.startswith(".") or part in {"__pycache__", "node_modules", "target"} for part in p.relative_to(workdir).parts):
                continue
            try:
                rel = p.relative_to(workdir).as_posix()
                st = p.stat()
            except OSError:
                continue
            out.append({"path": rel, "size": st.st_size, "mtime": st.st_mtime})
        return jsonify({"workdir": str(workdir), "files": out})

    @app.get("/api/files/<path:rel>")
    def api_files_get(rel: str):
        target = _safe_path(rel)
        if target is None or not target.is_file():
            return jsonify({"ok": False, "error": "not found"}), 404
        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "path": rel, "content": content})

    @app.put("/api/files/<path:rel>")
    def api_files_put(rel: str):
        target = _safe_path(rel)
        if target is None:
            return jsonify({"ok": False, "error": "invalid path"}), 400
        body = request.get_json(silent=True) or {}
        content = body.get("content", "")
        if not isinstance(content, str):
            return jsonify({"ok": False, "error": "content must be string"}), 400
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
        except OSError as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "path": rel, "size": target.stat().st_size})
