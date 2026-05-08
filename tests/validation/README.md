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
- [x] `wr90_straight` — Both tools agree on |S| to <0.005 absolute across 9–11 GHz. Tests rect waveguide ports + general FEM pipeline. `python compare_wr90.py`.
- [x] `parallel_plate` — UserDefinedPort case (demo0 layout). Both tools agree on |S| with rel diff <2% across 8–12 GHz. Tests UserDefinedPort + PMC. `python compare_parallel_plate.py`.
- [ ] `patch_antenna` — EMerge geometry fails gmsh PLC intersection (port plate touches patch edge); needs inset feed or geometry rework.
- [ ] `coax` — EMerge's own CoaxPort gives unphysical |S11|=1.0 (no EMerge demos use it). rapidfem CoaxPort matches EMerge's math but cross-validation blocked.

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
