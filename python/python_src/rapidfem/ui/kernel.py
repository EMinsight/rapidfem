"""Per-file persistent Python namespace for notebook-style cell execution.

Each open file gets a long-lived `Kernel`. Cells are exec'd into the same
namespace so variables carry across cells (`g = Geometry()` in one cell,
`g.mesh(...)` in the next). Reset wipes the namespace and the underlying
gmsh state so a fresh "Run All" starts clean.
"""
from __future__ import annotations

import threading
from typing import Any

import rapidfem


class Kernel:
    """One persistent execution context per file path."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.namespace: dict[str, Any] = {}
        self.lock = threading.Lock()
        self._reset_namespace()

    def _reset_namespace(self) -> None:
        self.namespace = {
            "__name__": "__rapidfem_kernel__",
            "__file__": self.file_path,
            "rapidfem": rapidfem,
        }

    def reset(self) -> None:
        with self.lock:
            self._reset_namespace()
            try:
                import gmsh
                if gmsh.isInitialized():
                    gmsh.clear()
            except Exception:
                pass


_kernels: dict[str, Kernel] = {}
_kernels_lock = threading.Lock()


def get_kernel(file_path: str) -> Kernel:
    with _kernels_lock:
        if file_path not in _kernels:
            _kernels[file_path] = Kernel(file_path)
        return _kernels[file_path]


def reset_kernel(file_path: str) -> None:
    get_kernel(file_path).reset()


def drop_kernel(file_path: str) -> None:
    """Forget a kernel entirely (e.g. when the file is renamed/deleted)."""
    with _kernels_lock:
        _kernels.pop(file_path, None)
