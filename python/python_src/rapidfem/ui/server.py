"""Flask server for `rapidfem serve`.

Exposes the bundled SvelteKit frontend on `/` and a small JSON/WebSocket API
on `/api/*` and `/ws`. All endpoints are local-only by convention — there is
no authentication.

The actual endpoints are registered in `rapidfem.ui.api` (split out so the
server module stays small and testable).
"""
from __future__ import annotations

import os
import threading
import webbrowser
from importlib import resources
from pathlib import Path

from flask import Flask, jsonify, send_from_directory


_FRONTEND_PACKAGE = "rapidfem.ui.frontend"


def _frontend_dist_path() -> Path | None:
    """Locate the bundled frontend's dist/ directory.

    Returns None when the wheel was built without a frontend bundle (e.g.
    a from-source install that did not run `scripts/build_frontend`).
    Callers should surface a helpful message in that case.
    """
    try:
        root = resources.files(_FRONTEND_PACKAGE)
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    dist = root / "dist"
    try:
        if dist.is_dir():  # type: ignore[union-attr]
            return Path(str(dist))
    except (AttributeError, NotADirectoryError):
        pass
    return None


def create_app(workdir: Path, debug: bool = False) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["RAPIDFEM_WORKDIR"] = Path(workdir).resolve()
    app.config["RAPIDFEM_DEBUG"] = debug

    dist = _frontend_dist_path()
    app.config["RAPIDFEM_FRONTEND_DIST"] = dist

    # CORS for the localhost dev case where the SvelteKit dev server (vite,
    # port 5173) talks to this Flask process (port 5174). Strict-localhost
    # check rather than wildcard.
    @app.after_request
    def _cors(resp):
        origin = resp.headers.get("Origin") or ""
        # Allow any localhost origin (vite dev server, custom ports).
        # Reflected only, not "*", so credentials remain blocked anyway.
        from flask import request
        req_origin = request.headers.get("Origin", "")
        if req_origin.startswith("http://127.0.0.1") or req_origin.startswith("http://localhost"):
            resp.headers["Access-Control-Allow-Origin"] = req_origin
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.get("/api/health")
    def health():
        return jsonify({
            "ok": True,
            "workdir": str(app.config["RAPIDFEM_WORKDIR"]),
            "frontend_bundled": dist is not None,
        })

    # Frontend static serving — only registered when dist/ is present.
    if dist is not None:
        @app.get("/", defaults={"path": ""})
        @app.get("/<path:path>")
        def _spa(path: str):
            target = dist / path
            if path and target.exists() and target.is_file():
                return send_from_directory(dist, path)
            # SPA fallback: serve index.html so client-side routes resolve.
            return send_from_directory(dist, "index.html")
    else:
        @app.get("/")
        def _no_frontend():
            return (
                "<h1>rapidfem ui</h1>"
                "<p>Frontend bundle not found. Build it with "
                "<code>scripts/build_frontend.{ps1,sh}</code> and reinstall, "
                "or install a release wheel.</p>"
            ), 503

    try:
        from rapidfem.ui import api  # noqa: F401
        api.register(app)
    except ImportError:
        pass

    try:
        # simple_websocket (under flask-sock) negotiates `permessage-deflate`
        # and sends frames with RSV1 set — but its deflate implementation
        # produces frames that browsers and Python's `websockets` client
        # reject as "Invalid frame header" / "reserved bits must be 0".
        # Locally we have no bandwidth pressure; deflate is pure cost+risk.
        # Belt-and-suspenders defense in three layers:
        #   1. PerMessageDeflate.accept → return None so the extension isn't
        #      negotiated; `_enabled` stays False and FrameProtocol filters it
        #      out at Connection-construction time.
        #   2. PerMessageDeflate.enabled → always False, in case some state
        #      mutation flips `_enabled` later. With enabled=False, FrameProtocol
        #      drops the extension from `self.extensions` on construction.
        #   3. PerMessageDeflate.frame_outbound → pass-through (no compression,
        #      no RSV bits) — the last-line guarantee that nothing slips through
        #      even if both upper layers were bypassed.
        # AcceptConnection extensions=[] is *also* patched for completeness,
        # so the handshake response carries no `Sec-WebSocket-Extensions`.
        import wsproto.extensions as _wsx
        if not getattr(_wsx, "_rapidfem_no_deflate", False):
            _wsx.PerMessageDeflate.accept = lambda self, offer: None
            _wsx.PerMessageDeflate.enabled = lambda self: False
            _wsx.PerMessageDeflate.frame_outbound = (
                lambda self, proto, opcode, rsv, data, fin: (rsv, data)
            )
            _wsx._rapidfem_no_deflate = True

        import simple_websocket.ws as _sw_ws
        if not getattr(_sw_ws, "_rapidfem_no_deflate", False):
            _orig_accept = _sw_ws.AcceptConnection

            def _no_deflate_accept(*args, **kwargs):
                kwargs["extensions"] = []
                return _orig_accept(*args, **kwargs)

            _sw_ws.AcceptConnection = _no_deflate_accept
            _sw_ws._rapidfem_no_deflate = True

        # simple_websocket.Base.send calls `self.sock.send(out_data)` —
        # plain socket.send, which on Windows returns short when the OS TCP
        # buffer (~64 KB by default) can't accept the whole frame. The
        # remaining bytes are silently dropped, the frame is truncated, and
        # the next bytes the client decodes from the middle of the payload
        # look like a malformed WS frame ("reserved bits must be 0" / "Invalid
        # frame header"). Swap the implementation for one that uses sendall.
        import simple_websocket as _sw
        if not getattr(_sw, "_rapidfem_sendall", False):
            _orig_send = _sw.Base.send

            def _sendall_send(self, data):
                from simple_websocket import ConnectionClosed
                from wsproto.events import Message, TextMessage
                if not self.connected:
                    raise ConnectionClosed(self.close_reason, self.close_message)
                if isinstance(data, bytes):
                    out_data = self.ws.send(Message(data=data))
                else:
                    out_data = self.ws.send(TextMessage(data=str(data)))
                # sendall retries until every byte is written, which is what
                # WS framing requires — anything less and the next frame
                # starts in the middle of a truncated one.
                self.sock.sendall(out_data)

            _sw.Base.send = _sendall_send
            _sw._rapidfem_sendall = True

        from flask_sock import Sock
        from rapidfem.ui.bus import BUS
        from rapidfem.ui.kernel_ws import register_kernel_ws

        sock = Sock(app)
        register_kernel_ws(sock)

        @sock.route("/ws")
        def _ws(ws):  # pragma: no cover — exercised end-to-end
            q = BUS.subscribe()
            try:
                ws.send('{"kind":"hello","ok":true}')
                while True:
                    payload = q.get()
                    ws.send(payload)
            except Exception:
                pass
            finally:
                BUS.unsubscribe(q)
    except ImportError:
        pass  # flask-sock optional dependency

    return app


def run(app: Flask, host: str = "127.0.0.1", port: int = 5174, open_browser: bool = True) -> None:
    url = f"http://{host}:{port}/"
    print(f"rapidfem serve — workdir: {app.config['RAPIDFEM_WORKDIR']}")
    print(f"rapidfem serve — listening on {url}")
    if app.config["RAPIDFEM_FRONTEND_DIST"] is None:
        print("rapidfem serve — WARNING: no frontend bundle found (run scripts/build_frontend).")

    # gmsh.initialize installs a SIGINT handler, which only the main thread
    # can do. Eagerly init here so Geometry() calls from worker request
    # threads skip the init via gmsh.isInitialized().
    try:
        import gmsh
        if not gmsh.isInitialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
    except Exception as e:  # noqa: BLE001
        print(f"rapidfem serve — gmsh pre-init failed: {e}")

    if open_browser and not os.environ.get("RAPIDFEM_NO_BROWSER"):
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=app.config["RAPIDFEM_DEBUG"], use_reloader=False)
