# Validation Harness

EMerge ↔ rapidfem numerical comparison framework.

## Goal

Verify rapidfem reproduces EMerge's results to defined tolerances on representative
problems. Foundation for any "parity" claim.

## Layout

- `compare.py` — shared utilities. Loads Touchstone (.s1p/.s2p) and CSV S-parameter
  files, performs frequency-aligned comparison with absolute and relative tolerances.
- `run_emerge_<case>.py` — Python script that builds a problem in EMerge, runs the
  sweep, writes `<case>.csv`.
- `run_rapidfem_<case>.py` — driver that subprocess-calls `cargo run` with the
  matching TOML config and reads the resulting Touchstone output.
- `compare_<case>.py` — top-level driver: run both, compare, print pass/fail.

## Status

- [x] `compare.py` — Touchstone + CSV loaders, freq-interpolated comparison.
- [ ] `patch_antenna` — EMerge geometry currently fails gmsh (PLC intersection).
      Needs demo4-style inset feed topology to be mesh-stable.
- [ ] `wr90_straight` — Pending.
- [ ] `parallel_plate` — Pending. Blocked on UserDefinedPort in rapidfem.

## Adding a new case

1. Build the problem in EMerge, save S-params to CSV via `compare.save_csv`.
2. Build matching `rapidfem` config (gmsh script + TOML).
3. Write a comparison driver that calls both and `compare.compare(...)`.
4. Document tolerance choice (FEM discretization, ABC reflection floor, etc.).

## Tolerances

Two independent meshes will not agree to machine precision. Reasonable defaults:
- `tol_abs = 0.05` on |S| (FEM discretization at moderate refinement)
- `tol_rel = 0.10` for off-resonance regions where |S| is large

For same-mesh comparison (future), tighten to `tol_abs = 1e-4`.
