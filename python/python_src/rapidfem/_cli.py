"""rapidfem CLI — `rapidfem serve` and friends.

Entry point registered via pyproject.toml [project.scripts].
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rapidfem import __version__


def _default_workdir() -> Path:
    """Default workdir: ``~/rapidfem-workspace/``.

    Lives outside any source tree so example copies never pollute the
    rapidfem checkout, and stays stable across shell sessions so users
    can run ``rapidfem serve`` from anywhere and find their edits.
    """
    return Path.home() / "rapidfem-workspace"


def _populate_examples(workdir: Path) -> int:
    """Copy bundled examples into ``workdir`` if no ``.py`` is there yet.

    Idempotent: never overwrites an existing file. Returns the count of
    files actually copied (0 if the workdir already had Python in it).
    """
    if any(p.is_file() for p in workdir.glob("*.py")):
        return 0
    try:
        from importlib import resources
        root = resources.files("rapidfem.examples")
    except (ModuleNotFoundError, FileNotFoundError):
        return 0
    n = 0
    for entry in root.iterdir():  # type: ignore[attr-defined]
        if not entry.is_file():
            continue
        name = entry.name
        if not name.endswith(".py") or name.startswith("_"):
            continue
        target = workdir / name
        if target.exists():
            continue
        try:
            content = entry.read_text(encoding="utf-8")
            target.write_text(content, encoding="utf-8", newline="\n")
            n += 1
        except OSError as e:
            print(f"rapidfem serve — could not copy {name}: {e}", file=sys.stderr)
    return n


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

    if args.workdir is None:
        workdir = _default_workdir()
        if not workdir.exists():
            try:
                workdir.mkdir(parents=True, exist_ok=True)
                print(f"rapidfem serve — created workdir {workdir}")
            except OSError as e:
                print(f"error: could not create default workdir {workdir}: {e}", file=sys.stderr)
                return 2
    else:
        workdir = Path(args.workdir).resolve()
        if not workdir.exists():
            print(f"error: workdir does not exist: {workdir}", file=sys.stderr)
            return 2
        if not workdir.is_dir():
            print(f"error: workdir is not a directory: {workdir}", file=sys.stderr)
            return 2

    # Populate bundled examples on a fresh workdir so users see the demos
    # the first time they open the UI. Skipped if any *.py is already
    # present — never overwrites user edits.
    n_copied = _populate_examples(workdir)
    if n_copied:
        print(f"rapidfem serve — populated {n_copied} example scripts in {workdir}")

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
        default=None,
        help="project working directory (default: ~/rapidfem-workspace/)",
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
