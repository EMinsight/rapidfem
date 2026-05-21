"""Shared formatting helpers for spec / TOML emission."""
from __future__ import annotations


def _f64(x: float) -> str:
    """Format a float with 10 significant digits for spec emission."""
    return f"{float(x):.10g}"
