# Clean-room derivations

This directory re-derives rapidfem's FEM kernels **from primary mathematics**,
independently of the EMerge source they were originally ported from. The goal
is to remove the EMerge code dependency: copyright protects *expression* (a
specific variable layout, loop structure, comments), not the mathematical
facts (basis functions, element integrals, integration identities). By
deriving each kernel from scratch — symbolically with sympy, anchored to
textbook/primary-literature definitions — the resulting Rust carries only
rapidfem's own copyright.

rapidfem stays **GPL-3.0-or-later** (it links Gmsh at runtime, a separate
obligation that this effort does not touch). What changes is that the listed
files no longer carry the `Copyright (C) Robert Fennis (original EMerge
source)` header, because their content is independently derived and verified.

## The loop (verify-then-delete)

For each ported kernel:

1. **Derive** the mathematics from scratch in a sympy module here, citing the
   primary source (textbook, paper, or standard table) — never the EMerge
   expression.
2. **Generate** an exact golden table / closed form from that derivation.
3. **Verify** the existing Rust kernel reproduces the derived numbers to
   machine precision via a generated `tests/*_golden_test.rs`. Agreement
   proves the Rust computes the *math*, which is what we keep.
4. **Re-head** the Rust file: drop the EMerge attribution, cite the primary
   source instead. The validation harness (`tests/validation/`) is the second
   safety net — physics-level results must not move.

A kernel is only "clean" once its golden test passes *and* its header no
longer names EMerge.

## Primary sources

- **Simplex integration identity** — Eisenberg & Malvern, "On finite element
  integration in natural coordinates", Int. J. Numer. Methods Eng. 7 (1973)
  574-575. → `nedelec2/barycentric.py`.
- **2nd-order H(curl) tetrahedral element (20 DOF)** — Savage & Peterson,
  "Higher-order vector finite elements for tetrahedral cells", IEEE Trans.
  MTT 44 (1996) 874-879; see also Jin, *The Finite Element Method in
  Electromagnetics*, ch. on higher-order vector elements. → `nedelec2/`.

## Status

| Kernel (Rust file)        | derivation                | golden test                         | header clean |
|---------------------------|---------------------------|-------------------------------------|--------------|
| `coefficients.rs`         | `nedelec2/barycentric.py` | `coefficients_golden_test.rs` (881) | ✅ |
| `tet_assembly.rs`         | `nedelec2/` (in progress) | —                                   | ⬜ |
| `tri_assembly.rs`         | —                         | —                                   | ⬜ |
| `interp.rs`               | —                         | —                                   | ⬜ |
| `basis.rs` (DOF mapping)  | interface only, no math   | —                                   | ⬜ |
| `materials.rs`            | —                         | —                                   | ⬜ |

## Running

```sh
python3 derivations/nedelec2/barycentric.py          # derive + verify (no output files)
python3 derivations/nedelec2/emit_coefficients_test.py   # regenerate the Rust golden test
cargo test -p rapidfem-fd --test coefficients_golden_test -- --nocapture
```

sympy is required (`python3 -c "import sympy"`); it is a dev-time dependency
of the derivations only, not of the shipped crate.
