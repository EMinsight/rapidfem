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
- [x] `coax` — Self-validation against analytical TEM transmission line. Matched line of length L: |S11|<0.001, |S21|=0.9994, phase agrees with -k·√εr·L within 0.1°. EMerge's own CoaxPort gives |S11|=1.0 (broken), so direct cross-validation is impossible; the analytical comparison is equivalent. `python compare_coax.py`. Note: requires fine mesh near the inner conductor (mesh script refines to 0.08mm there) to resolve the 1/ρ field singularity.
- [x] `floquet` — FloquetPort smoke test at normal incidence (θ=0) with PMC side walls (proxy for periodic BC at θ=0). 9/9 freq points, |S11|<0.003, |S21|=1.0. `python compare_floquet.py`. **Limitations**: oblique incidence drops the transverse phase factor (real-only mode field API), and there are no Periodic BCs on side walls yet (task #28) — for true phased-array unit cell sims at θ≠0 both pieces are needed.
- [x] `pml` — Uniaxial PML termination of WR-90 waveguide (instead of an ABC). With δmax=8, n=1.5, thickness 15mm: |S11|<0.004 across 9–11 GHz, 11/11 OK. `python compare_pml.py`. Runs with PARDISO (≈10× faster than faer, ~0.5s for 11 freqs).
- [x] `debye` — Debye dispersion in WR-90 (ε∞=2, εs=4, τ=50ps). Analytical εr(f) matches expectation, S-params smooth and passive across 9–11 GHz. `python compare_debye.py`. Validates that the per-frequency K rebuild path works correctly when materials are dispersive.
- [x] `antenna_metrics` — Edge-fed patch antenna at 2.4 GHz with **closed NFFT surface** (ABC walls + PEC ground, J=n×H, M=0 on PEC). Peak now correctly at θ≈0 broadside (θ=2°, φ=185°), D=1.82 dBi, G=0.64 dBi, AR=47.9 dB at peak (strongly linear). `python compare_antenna_metrics.py`. The phantom back lobe is gone. Quantitative pattern validation against EMerge demo4 is still blocked by the gmsh-PLC environment issue on this Windows install.
- [x] `surface_impedance` — `rf.SurfaceImpedance` lossy walls validated against the analytic TE₁₀ conductor attenuation (Pozar eq. 3.96) in a WR-90 guide: α from |S21| agrees to ~2 %. Also gates the explicit-`zs` vs. σ-derived `zs` branches (exact) and the finite-thickness `tanh` correction against its thin-sheet (Re Zs → 1/σt, α → α∞·δ/t) and thick-sheet (α → α∞) limits (<0.3 %). Pure-Python pytest, no EMerge dependency: `pytest -m slow python/tests/test_metals_validation.py`.
- [x] `volume_conductivity` — `rf.Material(conductivity=σ)` / `rf.Conductor` validated against the *exact* complex TE₁₀ propagation constant γ=√(kc²−k₀²εr*) in a lossy-air WR-90 guide (no perturbation approximation): α from |S21| agrees to ~0.1 %. A second gate checks bulk σ vs. the equivalent loss tangent tanδ=σ/(ωε₀εr) at one frequency (agreement to ~1e-10), pinning the `εr*=εr(1−j·tanδ)−j·σ/(ωε₀)` assembly formula. Same pytest module as `surface_impedance`.

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
