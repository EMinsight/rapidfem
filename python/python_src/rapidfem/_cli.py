"""rapidfem CLI — `rapidfem serve` and friends.

Entry point registered via pyproject.toml [project.scripts].
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rapidfem import __version__


_WELCOME_SCRIPT = '''\
"""Welcome to rapidfem!

This is the script you opened in the editor. The UI workflow:

  1. Edit this file and press Ctrl+S — the right-hand 3D view refreshes
     automatically (it calls /api/run on the Flask backend, which exec's
     this script and renders whatever you pass to rapidfem.show()).

  2. Click "Generate Mesh" to run gmsh and see the full tet mesh.

  3. Click "Run Simulation" to build a SimulationBuilder + run the FEM
     frequency sweep. S-params appear in the second tab.

The example below is a tiny WR-90 rectangular waveguide section — change
its dimensions, materials, or ports, then save to see updates.
"""
import numpy as np
import rapidfem


# ── Geometry ─────────────────────────────────────────────────────────────
# WR-90 = 22.86 mm × 10.16 mm; 30 mm long section.
A, B, L = 22.86e-3, 10.16e-3, 30.0e-3

g = rapidfem.Geometry()
air = g.box(A, B, L, position=(-A / 2, -B / 2, 0))
air.material = "air"

# Name the two faces at z = 0 and z = L as ports.
air.faces.min(axis="z").name = "port_in"
air.faces.max(axis="z").name = "port_out"
# All other faces are PEC walls.
for face in air.faces:
    if face.name is None:
        face.name = "pec"

# Mesh a bit smaller on the ports so they get resolved properly.
air.maxh = 5e-3

# ── Send the geometry to the viewer ──────────────────────────────────────
rapidfem.show(g)


# ── Build a SimulationBuilder (used by "Run Simulation") ─────────────────
# 21-point sweep across the WR-90 single-mode band.
builder = (
    rapidfem.SimulationBuilder()
    .from_geometry(g, maxh=5e-3)
    .frequencies(np.linspace(8.0e9, 12.0e9, 21))
    .rect_waveguide("port_in")
    .rect_waveguide("port_out")
    .pec("pec")
    .material("air", er=1.0)
)
rapidfem.show(builder)
'''


def _maybe_write_welcome(workdir: Path) -> None:
    """Drop a welcome.py the first time someone serves an empty workdir."""
    has_py = any(p.is_file() for p in workdir.glob("*.py"))
    if has_py:
        return
    target = workdir / "welcome.py"
    if target.exists():
        return
    try:
        target.write_text(_WELCOME_SCRIPT, encoding="utf-8", newline="\n")
        print(f"rapidfem serve — wrote starter script to {target}")
    except OSError as e:
        print(f"rapidfem serve — could not write welcome.py: {e}", file=sys.stderr)


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from rapidfem.ui.server import create_app, run
    except ImportError as e:
        print(
            "error: rapidfem was installed without the UI extra.\n"
            "       Install with:  pip install 'rapidfem[ui]'\n"
            f"       (import failure: {e})",
            file=sys.stderr,
        )
        return 2

    workdir = Path(args.workdir).resolve()
    if not workdir.exists():
        print(f"error: workdir does not exist: {workdir}", file=sys.stderr)
        return 2
    if not workdir.is_dir():
        print(f"error: workdir is not a directory: {workdir}", file=sys.stderr)
        return 2

    _maybe_write_welcome(workdir)

    app = create_app(workdir=workdir, debug=args.debug)
    run(app, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rapidfem",
        description="rapidfem — frequency-domain EM FEM solver.",
    )
    p.add_argument("--version", action="version", version=f"rapidfem {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    s = sub.add_parser(
        "serve",
        help="launch the local UI (Flask + bundled SvelteKit frontend)",
        description="Start the rapidfem UI on a local web server.",
    )
    s.add_argument(
        "workdir",
        nargs="?",
        default=".",
        help="project working directory (default: current directory)",
    )
    s.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    s.add_argument("--port", type=int, default=5174, help="bind port (default: 5174)")
    s.add_argument("--debug", action="store_true", help="Flask debug mode + hot reload")
    s.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    s.set_defaults(func=_cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
