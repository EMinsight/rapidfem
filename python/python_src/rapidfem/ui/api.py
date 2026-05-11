"""JSON API endpoints for the rapidfem UI.

Registered onto the Flask app by ``rapidfem.ui.server.create_app``.
"""
from __future__ import annotations

import io
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

import rapidfem
from rapidfem import _show_capture
from rapidfem.ui.bus import BUS
from rapidfem.ui.kernel import Kernel, get_kernel, reset_kernel


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


_WIN = sys.platform == "win32"
if _WIN:
    import ctypes
    import msvcrt
    _STD_OUTPUT_HANDLE = -11
    _STD_ERROR_HANDLE = -12
    _GetStdHandle = ctypes.windll.kernel32.GetStdHandle
    _GetStdHandle.restype = ctypes.c_void_p
    _SetStdHandle = ctypes.windll.kernel32.SetStdHandle
    _SetStdHandle.argtypes = [ctypes.c_int, ctypes.c_void_p]


@contextmanager
def _capture_native_streams(stage: str):
    """OS-level fd capture so Rust eprintln! and gmsh output reach the UI.

    On Windows Rust's stderr uses GetStdHandle(STD_ERROR_HANDLE) directly,
    not the C-runtime fd 2, so we also have to call SetStdHandle alongside
    the POSIX-style dup2 to actually redirect it.
    """
    sys.stdout.flush(); sys.stderr.flush()
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    os.dup2(out_w, 1)
    os.dup2(err_w, 2)

    saved_win_out = saved_win_err = None
    if _WIN:
        saved_win_out = _GetStdHandle(_STD_OUTPUT_HANDLE)
        saved_win_err = _GetStdHandle(_STD_ERROR_HANDLE)
        _SetStdHandle(_STD_OUTPUT_HANDLE, msvcrt.get_osfhandle(1))
        _SetStdHandle(_STD_ERROR_HANDLE, msvcrt.get_osfhandle(2))

    os.close(out_w)
    os.close(err_w)

    lines_out: list[str] = []
    lines_err: list[str] = []

    def reader(fd: int, kind: str, accum: list[str]) -> None:
        # Use raw os.read so Python's TextIOWrapper / BufferedReader don't
        # batch multiple lines before delivering them. Each os.read call
        # returns as soon as the pipe has any bytes, so lines from Rust
        # `eprintln!` stream out at solver speed, not at chunk-boundary.
        buf = b""
        try:
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break  # EOF — pipe writer closed
                buf += chunk
                while b"\n" in buf:
                    raw, _, buf = buf.partition(b"\n")
                    s = raw.rstrip(b"\r").decode("utf-8", errors="replace")
                    if not s:
                        continue
                    accum.append(s)
                    BUS.publish({"kind": "log", "stage": stage, "stream": kind, "line": s, "t": time.time()})
            if buf:
                tail = buf.decode("utf-8", errors="replace").rstrip()
                if tail:
                    accum.append(tail)
                    BUS.publish({"kind": "log", "stage": stage, "stream": kind, "line": tail, "t": time.time()})
        except Exception:
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    t_out = threading.Thread(target=reader, args=(out_r, "stdout", lines_out), daemon=True)
    t_err = threading.Thread(target=reader, args=(err_r, "stderr", lines_err), daemon=True)
    t_out.start()
    t_err.start()
    # sys.stdout/stderr in Python wrap the C fd via a TextIOWrapper with a
    # locale-derived encoding (cp1252 on Windows). User code printing a non-
    # ASCII char (e.g. subscript) would crash on encode. Force UTF-8 for the
    # duration of the cell so prints with Unicode work.
    prior_out_enc = getattr(sys.stdout, "encoding", None)
    prior_err_enc = getattr(sys.stderr, "encoding", None)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", write_through=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", write_through=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        yield lines_out, lines_err
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        try:
            if prior_out_enc:
                sys.stdout.reconfigure(encoding=prior_out_enc)  # type: ignore[attr-defined]
            if prior_err_enc:
                sys.stderr.reconfigure(encoding=prior_err_enc)  # type: ignore[attr-defined]
        except Exception:
            pass
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        if _WIN and saved_win_out is not None and saved_win_err is not None:
            _SetStdHandle(_STD_OUTPUT_HANDLE, saved_win_out)
            _SetStdHandle(_STD_ERROR_HANDLE, saved_win_err)
        os.close(saved_out)
        os.close(saved_err)
        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)


def _bbox_for_nodes(nodes_np) -> dict[str, list[float]]:
    """Bounding box from a (n_nodes, 3) array."""
    if nodes_np is None or len(nodes_np) == 0:
        return {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    mn = nodes_np.min(axis=0).tolist()
    mx = nodes_np.max(axis=0).tolist()
    return {"min": mn, "max": mx}


def _reset_gmsh() -> None:
    """Wipe gmsh model state so a fresh Geometry() doesn't collide with the
    last request's leftover model. Safe to call even if gmsh isn't initialized."""
    try:
        import gmsh
        if gmsh.isInitialized():
            gmsh.clear()
    except Exception:  # noqa: BLE001
        pass


def _exec_user_code(code: str, workdir: Path, stage: str = "run") -> dict[str, Any]:
    """Run a piece of user code with capture active. Returns the response payload."""
    _reset_gmsh()
    namespace: dict[str, Any] = {
        "__name__": "__rapidfem_user__",
        "__file__": str(workdir / "<editor>"),
        "rapidfem": rapidfem,
    }

    _show_capture.start_capture()
    err_state: BaseException | None = None
    try:
        with _capture_native_streams(stage) as (lines_out, lines_err):
            try:
                compiled = compile(code, "<editor>", "exec")
                exec(compiled, namespace)
            except SystemExit as e:
                err_state = e
            except BaseException as e:  # noqa: BLE001 — surface everything
                err_state = e
        stdout_text = "\n".join(lines_out)
        stderr_text = "\n".join(lines_err)
        if err_state is not None:
            return {
                "ok": False,
                "error": _format_exception(err_state),
                "stdout": stdout_text,
                "stderr": stderr_text,
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
        "stdout": stdout_text,
        "stderr": stderr_text,
        "captures": rendered,
    }


def _exec_for_pipeline(code: str, workdir: Path, stage: str = "pipeline") -> tuple[dict[str, Any], list]:
    """Run user code with capture, return (response-shell, raw captures)."""
    _reset_gmsh()
    namespace: dict[str, Any] = {
        "__name__": "__rapidfem_user__",
        "__file__": str(workdir / "<editor>"),
        "rapidfem": rapidfem,
    }
    _show_capture.start_capture()
    err: BaseException | None = None
    try:
        with _capture_native_streams(stage) as (lines_out, lines_err):
            try:
                exec(compile(code, "<editor>", "exec"), namespace)
            except BaseException as e:  # noqa: BLE001
                err = e
    finally:
        captured = _show_capture.stop_capture()
    stdout_text = "\n".join(lines_out)
    stderr_text = "\n".join(lines_err)
    if err is not None:
        return {
            "ok": False,
            "error": _format_exception(err),
            "stdout": stdout_text,
            "stderr": stderr_text,
        }, captured
    return {
        "ok": True,
        "stdout": stdout_text,
        "stderr": stderr_text,
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
            shell, captures = _exec_for_pipeline(code, workdir, stage="mesh")
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
                with _capture_native_streams("mesh") as (m_out, m_err):
                    payload = mesh_to_payload(cap.obj, maxh=maxh)
                if m_out or m_err:
                    if m_out: shell["stdout"] = (shell["stdout"] + "\n" + "\n".join(m_out)).strip()
                    if m_err: shell["stderr"] = (shell["stderr"] + "\n" + "\n".join(m_err)).strip()
            except Exception as e:  # noqa: BLE001
                return jsonify({**shell, "ok": False, "error": _format_exception(e)}), 200

        BUS.publish({"kind": "stage_end", "stage": "mesh", "ok": True, "stats": payload["stats"]})
        return jsonify({**shell, "mesh": payload, "name": cap.name}), 200

    @app.post("/api/solve")
    def api_solve():
        body = request.get_json(silent=True) or {}
        code = body.get("code", "")
        builder_name = body.get("builder_name")  # optional; None → first builder
        include_fields = bool(body.get("include_fields", True))
        if not isinstance(code, str):
            return jsonify({"ok": False, "error": {"type": "ValueError", "message": "code must be string", "traceback": ""}}), 400

        BUS.publish({"kind": "stage_start", "stage": "solve"})
        with _pipeline_lock:
            shell, captures = _exec_for_pipeline(code, workdir, stage="solve")
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
                import numpy as np
                t0 = time.perf_counter()
                with _capture_native_streams("solve") as (solver_out, solver_err):
                    sim = cap.obj.build()
                    result = sim.run_sweep()
                t_solve = time.perf_counter() - t0
                # Fold the solver's native-stream output into the response so
                # the user sees PARDISO/faer timings, frequency breakdown, etc.
                if solver_out or solver_err:
                    extra_out = "\n".join(solver_out)
                    extra_err = "\n".join(solver_err)
                    if extra_out:
                        shell["stdout"] = (shell["stdout"] + "\n" + extra_out).strip()
                    if extra_err:
                        shell["stderr"] = (shell["stderr"] + "\n" + extra_err).strip()
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

                # ── Nodal field as A/B/C phasor terms ─────────────────────
                # |E(t)|² = A cos²(ωt) + B sin²(ωt) − 2 C cos·sin
                # with A = |E_re|², B = |E_im|², C = E_re · E_im
                fields_payload = None
                if include_fields:
                    n_nodes = sim.mesh_nodes.shape[0]
                    fields_payload = []
                    for fi in range(n_freq):
                        per_freq = []
                        for pi in range(n_p):
                            E = sim.field_at_nodes(result, fi, pi)
                            if E is None:
                                per_freq.append(None)
                                continue
                            re = np.asarray(E.real)
                            im = np.asarray(E.imag)
                            A = np.sum(re * re, axis=1)
                            B = np.sum(im * im, axis=1)
                            C = np.sum(re * im, axis=1)
                            abc = np.stack([A, B, C], axis=1).astype(np.float32).ravel().tolist()
                            per_freq.append(abc)
                        fields_payload.append(per_freq)

                # ── Mesh that the solver actually used ────────────────────
                from rapidfem.ui.serialize import mesh_to_payload
                mesh_payload = None
                try:
                    cap_geo = _find_capture(captures, "geometry", None)
                    if cap_geo is not None:
                        mesh_payload = mesh_to_payload(cap_geo.obj, maxh=0.0)
                except Exception:
                    mesh_payload = None
            except Exception as e:  # noqa: BLE001
                return jsonify({**shell, "ok": False, "error": _format_exception(e)}), 200

        BUS.publish({"kind": "stage_end", "stage": "solve", "ok": True, "solve_time_s": t_solve, "n_freq": n_freq, "n_driven": n_p})
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
                "fields": fields_payload,
            },
            "mesh": mesh_payload,
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

    # ── Notebook cell execution ───────────────────────────────────────────

    def _serialize_captures(captures: list) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
        """Render captured Geometry/Builder/Simulation/Result objects."""
        from rapidfem.ui.serialize import geometry_to_payload, mesh_to_payload
        import numpy as np

        rendered: list[dict[str, Any]] = []
        mesh_payload: dict[str, Any] | None = None
        result_payload: dict[str, Any] | None = None
        last_sim = None
        last_result = None
        last_geo = None

        for item in captures:
            entry: dict[str, Any] = {"name": item.name, "kind": item.kind}
            if item.kind == "geometry":
                last_geo = item.obj
                try:
                    out = geometry_to_payload(item.obj)
                    # geometry_to_payload now returns mesh_to_payload when the
                    # gmsh state already holds a tet mesh. Route those to the
                    # dedicated mesh_payload slot — the frontend's geo-handler
                    # only knows the OCC wireframe / triangle-entities format.
                    if out.get("kind") == "mesh":
                        if mesh_payload is None:
                            mesh_payload = out
                        entry["meshed"] = True
                    else:
                        entry["payload"] = out
                except Exception as e:  # noqa: BLE001
                    entry["error"] = _format_exception(e)
            elif item.kind == "builder":
                pass
            elif item.kind == "simulation":
                last_sim = item.obj
            elif item.kind == "result":
                last_result = item.obj
            rendered.append(entry)

        # If a Simulation was shown, prefer the full gmsh-state mesh
        # (named physical groups, surface tris, dielectric volumes) over
        # the bare nodes/tets we'd get from sim.mesh_nodes.
        if last_sim is not None:
            try:
                mesh_payload = mesh_to_payload(last_geo, maxh=0.0)
            except Exception:
                # Fallback: nodes+tets from the simulation directly.
                try:
                    nodes_np = np.asarray(last_sim.mesh_nodes)
                    mesh_payload = {
                        "kind": "mesh",
                        "nodes": nodes_np.ravel().tolist(),
                        "tets": np.asarray(last_sim.mesh_tets).ravel().tolist(),
                        "tris": [], "tri_phys": [], "tet_phys": [1] * int(last_sim.mesh_tets.shape[0]),
                        "phys_names": {"1": "mesh"}, "phys_dim": {"1": 3}, "name_to_tag": {"mesh": 1},
                        "bbox": _bbox_for_nodes(nodes_np),
                        "stats": {"n_nodes": int(nodes_np.shape[0]),
                                  "n_tets": int(last_sim.mesh_tets.shape[0]),
                                  "n_tris": 0, "mesh_time_s": 0.0, "msh_bytes": 0},
                    }
                except Exception:
                    mesh_payload = None

        # Combine Simulation + SweepResult into full payload with mesh + fields.
        if last_sim is not None and last_result is not None:
            try:
                s = last_result.sparams
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
                fields_payload = []
                for fi in range(n_freq):
                    per_freq = []
                    for pi in range(n_p):
                        E = last_sim.field_at_nodes(last_result, fi, pi)
                        if E is None:
                            per_freq.append(None)
                            continue
                        re = np.asarray(E.real); im = np.asarray(E.imag)
                        A = np.sum(re * re, axis=1)
                        B = np.sum(im * im, axis=1)
                        C = np.sum(re * im, axis=1)
                        per_freq.append(np.stack([A, B, C], axis=1).astype(np.float32).ravel().tolist())
                    fields_payload.append(per_freq)
                result_payload = {
                    "frequencies": last_result.frequencies.tolist(),
                    "sparams": sparams_payload,
                    "n_driven": n_p, "n_freq": n_freq,
                    "n_dofs": last_sim.n_dofs, "n_tets": last_sim.n_tets,
                    "solve_time_s": last_result.solve_time_s,
                    "fields": fields_payload,
                }
            except Exception as e:  # noqa: BLE001
                result_payload = {"error": _format_exception(e)}

        return rendered, mesh_payload, result_payload

    @app.post("/api/cell/run")
    def api_cell_run():
        body = request.get_json(silent=True) or {}
        file_path = body.get("file", "<unnamed>")
        code = body.get("code", "")
        reset_first = bool(body.get("reset", False))
        if not isinstance(code, str):
            return jsonify({"ok": False, "error": {"type": "ValueError", "message": "code must be string", "traceback": ""}}), 400

        kernel = get_kernel(file_path)
        if reset_first:
            kernel.reset()

        BUS.publish({"kind": "stage_start", "stage": "cell", "file": file_path})
        err: BaseException | None = None
        with _pipeline_lock, kernel.lock:
            _show_capture.start_capture()
            try:
                with _capture_native_streams("cell") as (lines_out, lines_err):
                    try:
                        exec(compile(code, file_path or "<cell>", "exec"), kernel.namespace)
                    except BaseException as e:  # noqa: BLE001
                        err = e
            finally:
                captured = _show_capture.stop_capture()

        stdout_text = "\n".join(lines_out)
        stderr_text = "\n".join(lines_err)

        if err is not None:
            BUS.publish({"kind": "stage_end", "stage": "cell", "ok": False})
            return jsonify({
                "ok": False,
                "error": _format_exception(err),
                "stdout": stdout_text,
                "stderr": stderr_text,
                "captures": [], "mesh": None, "result": None,
            }), 200

        rendered, mesh_payload, result_payload = _serialize_captures(captured)
        BUS.publish({"kind": "stage_end", "stage": "cell", "ok": True})
        return jsonify({
            "ok": True,
            "stdout": stdout_text, "stderr": stderr_text,
            "captures": rendered, "mesh": mesh_payload, "result": result_payload,
        }), 200

    @app.post("/api/cell/reset")
    def api_cell_reset():
        body = request.get_json(silent=True) or {}
        file_path = body.get("file", "")
        reset_kernel(file_path)
        return jsonify({"ok": True})

    # ── Examples (shipped with the package) ───────────────────────────────

    @app.get("/api/examples")
    def api_examples_list():
        from importlib import resources
        try:
            root = resources.files("rapidfem.examples")
        except (ModuleNotFoundError, FileNotFoundError):
            return jsonify({"examples": []})
        items: list[dict[str, Any]] = []
        for entry in root.iterdir():  # type: ignore[attr-defined]
            if not entry.is_file():
                continue
            name = entry.name
            if not name.endswith(".py") or name.startswith("_"):
                continue
            items.append({"name": name})
        items.sort(key=lambda i: i["name"])
        return jsonify({"examples": items})

    @app.get("/api/examples/<name>")
    def api_examples_get(name: str):
        if not name.endswith(".py") or "/" in name or "\\" in name or ".." in name:
            return jsonify({"ok": False, "error": "invalid"}), 400
        from importlib import resources
        try:
            content = (resources.files("rapidfem.examples") / name).read_text(encoding="utf-8")
        except Exception:
            return jsonify({"ok": False, "error": "not found"}), 404
        return jsonify({"ok": True, "name": name, "content": content})

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

    @app.delete("/api/files/<path:rel>")
    def api_files_delete(rel: str):
        target = _safe_path(rel)
        if target is None or not target.is_file():
            return jsonify({"ok": False, "error": "not found"}), 404
        try:
            target.unlink()
        except OSError as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        # Drop the kernel so a future file at the same path starts fresh.
        try:
            from rapidfem.ui.kernel import drop_kernel
            drop_kernel(str(target))
        except Exception:
            pass
        return jsonify({"ok": True, "path": rel})

    @app.post("/api/files/rename")
    def api_files_rename():
        body = request.get_json(silent=True) or {}
        old_rel = body.get("from", "")
        new_rel = body.get("to", "")
        old = _safe_path(old_rel)
        new = _safe_path(new_rel)
        if old is None or new is None or not old.is_file():
            return jsonify({"ok": False, "error": "invalid path"}), 400
        if new.exists():
            return jsonify({"ok": False, "error": "destination exists"}), 409
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
        except OSError as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "from": old_rel, "to": new_rel})
