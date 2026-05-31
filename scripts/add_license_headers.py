"""Add SPDX license headers to every .rs source file in rapidfem.

EMerge-derived files (full or substantial port of an EMerge Python module)
get a header crediting Robert Fennis as original copyright holder; all
other Rust source files get a rapidfem-only header. Both headers point at
LICENSE for the full GPL-3.0+ terms with the Gmsh additional permission.

Idempotent: skips files that already start with `// SPDX-License-Identifier:`.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# EMerge-derived files — per the per-file `//! Exact port of` / `//! Mirrors`
# headers and the file-by-file attribution in NOTICE.
EMERGE_DERIVED = {
    "crates/rapidfem-core/src/constants.rs",
    "crates/rapidfem-core/src/mesh.rs",
    "crates/rapidfem-core/src/quadrature.rs",
    "crates/rapidfem-core/src/materials.rs",
    "crates/rapidfem-fd/src/abc_order2.rs",
    "crates/rapidfem-fd/src/assembly.rs",
    "crates/rapidfem-fd/src/basis.rs",
    "crates/rapidfem-fd/src/coefficients.rs",
    "crates/rapidfem-fd/src/interp.rs",
    "crates/rapidfem-fd/src/sparam.rs",
    "crates/rapidfem-fd/src/tet_assembly.rs",
    "crates/rapidfem-fd/src/touchstone.rs",
    "crates/rapidfem-fd/src/tri_assembly.rs",
    "crates/rapidfem-fd/src/waveguide.rs",
}

HEADER_EMERGE = """\
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
// Copyright (C) Robert Fennis (original EMerge source)
//
// This file is part of rapidfem and contains code ported from EMerge
// (https://github.com/FennisRobert/EMerge), originally licensed under
// GPL-2.0-or-later with the Gmsh additional permission; redistributed
// here under GPL-3.0-or-later with that permission preserved.
// See LICENSE and NOTICE for the full terms.

"""

HEADER_RAPIDFEM = """\
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

"""

SKIP_MARKER = "// SPDX-License-Identifier:"


def patch(path: Path, header: str) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith(SKIP_MARKER):
        return "skip"
    path.write_text(header + text, encoding="utf-8")
    return "patched"


def main() -> int:
    rs_files: list[Path] = []
    for crate in ("crates/rapidfem-core", "crates/rapidfem-fd", "crates/rapidfem-td", "python"):
        src = ROOT / crate / "src"
        if not src.exists():
            continue
        rs_files.extend(src.glob("**/*.rs"))

    n_emerge = n_rf = n_skip = 0
    for f in sorted(rs_files):
        rel = f.relative_to(ROOT).as_posix()
        header = HEADER_EMERGE if rel in EMERGE_DERIVED else HEADER_RAPIDFEM
        status = patch(f, header)
        flag = "EMG" if rel in EMERGE_DERIVED else "RFM"
        print(f"  {status:7s} [{flag}] {rel}")
        if status == "skip":
            n_skip += 1
        elif rel in EMERGE_DERIVED:
            n_emerge += 1
        else:
            n_rf += 1
    print(f"\nPatched {n_emerge} EMerge-derived + {n_rf} rapidfem-only files; "
          f"skipped {n_skip} already-headered files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
