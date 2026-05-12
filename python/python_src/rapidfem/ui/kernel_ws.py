"""WebSocket-based kernel protocol.

One WS endpoint, one ordered event stream per cell execution. Replaces the
split (HTTP /api/cell/run + WS /ws for logs) so captures and log lines
arrive in a single channel without race conditions.

Message protocol
----------------

Client → server:
    {"type": "execute", "cell_id": str, "file": str, "code": str, "reset": bool}
    {"type": "reset",   "file": str}
    {"type": "interrupt", "cell_id": str}

Server → client (every event carries cell_id when applicable):
    {"type": "hello"}
    {"type": "started",  "cell_id": ...}
    {"type": "stream",   "cell_id": ..., "stream": "stdout"|"stderr", "line": str}
    {"type": "display",  "cell_id": ..., "kind": "geometry"|"mesh"|"result", "payload": ...}
    {"type": "error",    "cell_id": ..., "error": {"type", "message", "traceback"}}
    {"type": "done",     "cell_id": ..., "ok": bool}
"""
from __future__ import annotations

import json
import threading
import traceback
from typing import Any

import rapidfem
from rapidfem import _show_capture
from rapidfem.ui.kernel import get_kernel, reset_kernel


# fd-capture is process-global (dup2 on fds 1/2) so two simultaneous cell
# executions on different kernels would interleave their stdout/stderr.
# This lock serializes all cell exec across kernels — single-user UI, the
# loss of parallelism is fine.
_GLOBAL_EXEC_LOCK = threading.Lock()


def _format_exception(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def register_kernel_ws(sock) -> None:
    from rapidfem.ui.api import _capture_streams, _serialize_captures_for_protocol  # type: ignore

    @sock.route("/ws/kernel")
    def kernel_ws(ws):  # pragma: no cover — exercised end-to-end
        # Per-connection lock that serialises every `ws.send`. Without it,
        # the stdout/stderr reader threads spawned by `_capture_streams`
        # write to the same socket as this handler thread — `socket.send`
        # is not atomic for multi-byte writes, so concurrent calls produce
        # interleaved bytes that destroy the WS frame stream. Symptom:
        # "Invalid frame header" / "reserved bits must be 0" in clients,
        # plus Werkzeug dumping `HTTP/1.1 500` onto the now-corrupted WS.
        send_lock = threading.Lock()

        def send(obj: dict[str, Any]) -> None:
            payload = json.dumps(obj, default=str)
            with send_lock:
                try:
                    ws.send(payload)
                except Exception:
                    pass

        send({"type": "hello"})

        while True:
            raw = ws.receive(timeout=None)
            if raw is None:
                break
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            t = msg.get("type")
            if t == "execute":
                _handle_execute(msg, send, _capture_streams, _serialize_captures_for_protocol)
            elif t == "reset":
                file_path = msg.get("file", "")
                reset_kernel(file_path)
                send({"type": "reset_ack", "file": file_path})
            elif t == "interrupt":
                # Soft-interrupt not implemented yet; client should fall back
                # to "reset" if it wants to stop a running cell.
                send({"type": "interrupt_ack", "cell_id": msg.get("cell_id"), "ok": False})


def _handle_execute(msg, send, capture_streams_cm, serialize_for_protocol):
    cell_id = msg.get("cell_id") or ""
    file_path = msg.get("file", "<unnamed>")
    code = msg.get("code", "")
    reset_first = bool(msg.get("reset", False))

    if not isinstance(code, str):
        send({"type": "error", "cell_id": cell_id, "error": {
            "type": "ValueError", "message": "code must be a string", "traceback": "",
        }})
        send({"type": "done", "cell_id": cell_id, "ok": False})
        return

    kernel = get_kernel(file_path)
    if reset_first:
        kernel.reset()

    send({"type": "started", "cell_id": cell_id, "file": file_path})

    def on_line(stream: str, line: str) -> None:
        send({"type": "stream", "cell_id": cell_id, "stream": stream, "line": line})

    err: BaseException | None = None
    with _GLOBAL_EXEC_LOCK, kernel.lock:
        _show_capture.start_capture()
        try:
            with capture_streams_cm(on_line=on_line, stage="cell"):
                try:
                    exec(compile(code, file_path or "<cell>", "exec"), kernel.namespace)
                except BaseException as e:  # noqa: BLE001
                    err = e
        finally:
            captured = _show_capture.stop_capture()

    if err is not None:
        send({"type": "error", "cell_id": cell_id, "error": _format_exception(err)})
        send({"type": "done", "cell_id": cell_id, "ok": False})
        return

    try:
        for display in serialize_for_protocol(captured):
            send({"type": "display", "cell_id": cell_id, **display})
    except Exception as e:  # noqa: BLE001
        send({"type": "error", "cell_id": cell_id, "error": _format_exception(e)})
        send({"type": "done", "cell_id": cell_id, "ok": False})
        return

    send({"type": "done", "cell_id": cell_id, "ok": True})
