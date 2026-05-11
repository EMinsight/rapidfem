"""rapidfem CLI — `rapidfem serve` and friends.

Entry point registered via pyproject.toml [project.scripts].
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rapidfem import __version__


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
