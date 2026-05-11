"""Capture-slot for rapidfem.show().

When the UI server runs user code, it calls `start_capture()` first, exec's
the user script, and then collects whatever was passed to `rapidfem.show()`
via `get_captured()`.

Outside UI mode the capture slot is inactive and `show()` is a print-only
no-op, so scripts behave the same on the command line and in the UI.
"""
from __future__ import annotations

import threading
from typing import Any, NamedTuple


class CapturedItem(NamedTuple):
    name: str
    obj: Any
    kind: str  # "geometry" | "builder" | "simulation" | "result" | "unknown"


_state = threading.local()


def start_capture() -> None:
    _state.active = True
    _state.items = []


def stop_capture() -> list[CapturedItem]:
    items: list[CapturedItem] = list(getattr(_state, "items", []))
    _state.active = False
    _state.items = []
    return items


def get_captured() -> list[CapturedItem]:
    return list(getattr(_state, "items", []))


def is_capturing() -> bool:
    return bool(getattr(_state, "active", False))


def capture(name: str, obj: Any, kind: str) -> None:
    if not is_capturing():
        return
    _state.items.append(CapturedItem(name=name, obj=obj, kind=kind))


def classify(obj: Any) -> str:
    cls = type(obj).__name__
    mod = getattr(type(obj), "__module__", "") or ""
    if cls == "Geometry" and mod.startswith("rapidfem"):
        return "geometry"
    if cls == "SimulationBuilder" and mod.startswith("rapidfem"):
        return "builder"
    if cls == "Simulation":
        return "simulation"
    if cls == "SweepResult":
        return "result"
    return "unknown"
