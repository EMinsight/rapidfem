# Local conditioning under slivers — derivation & findings

`sliver_conditioning.py` studies how the canonical R2 element (from
`derivations/nedelec2/element.py`) conditions as a tetrahedron flattens into a
sliver, and which solver-side remedies actually help. Run:

```sh
python3 derivations/conditioning/sliver_conditioning.py
```

## What the math says (the honest picture)

Let `q = 6V / h_e³` be the normalized volume (1 ≈ regular, → 0 ≈ sliver).

1. **Blowup is intrinsic and ~quadratic.** `cond(D+F) ≈ 3.8e2 · (1/q)²`.
   A sliver's ill-conditioning comes from the curl–curl term scaling like
   `1/V` while the mass scales like `V`; it is *geometric*, not a unit/scale
   artifact. Numbers (unit tensors):

   | q (norm. vol) | cond(D) | cond(F) | cond(D+F) |
   |---------------|---------|---------|-----------|
   | 6.3e-1        | 2.0e1   | 1.3e2   | 6.0e2     |
   | 1.4e-2        | 3.3e4   | 2.0e5   | 4.1e6     |
   | 1.4e-4        | 3.3e8   | 2.0e9   | 4.1e10    |
   | 1.4e-6        | 3.3e12  | 2.0e13  | 4.0e14    |
   | 1.4e-7        | —       | 2.0e15  | 2.0e17 ✗  |

   Below `q ≈ 1e-7` float64 can no longer represent the gradients and the
   element matrix is numerical noise.

2. **Floor threshold (lever ③).** From the fitted `cond(q)`:
   `cond = 1e8` at `q ≈ 2.6e-3`, `cond = 1e12` at `q ≈ 3.3e-5`,
   `cond = 1/u ≈ 4.5e15` at `q ≈ 5.9e-7`. → **Guard `6V` so that
   `q = 6V/h_e³ ≳ 1e-9`**; below that the tet is numerically dead — floor it
   and warn instead of emitting NaN/Inf into the global system.

3. **Diagonal equilibration does NOT cure a single sliver (lever ①).**
   `A → S A S` on a sliver only moved `cond` 4.05e8 → 2.86e8 (Jacobi *and*
   Ruiz). The spread is intrinsic, not a diagonal mismatch. Equilibration is
   still worth doing — it is solution-preserving (verified to 4e-9) and helps
   the *global* matrix when elements/materials mix scales — but it is general
   hygiene, **not** a sliver fix.

4. **Iterative refinement recovers accuracy only up to a ceiling (lever ②).**
   It contracts the error by `ρ = cond·u_factor` per step *iff* `ρ < 1`. On a
   moderate sliver (cond 4e6) a single-precision factorization (ρ=0.48)
   refines 8e-3 → 9e-11 in four cheap re-solves. For a true sliver with
   `cond > 1/u` it **diverges** — no solver-side trick rescues it.

## Consequence for the implementation

Without touching the mesh you cannot make a true sliver well-conditioned —
the geometry is bad. The realistic, solution-safe levers, by value:

- **③ Volume/area floor + warning** — the real protection: stop one dead tet
  from poisoning the whole factorization with NaN. Threshold `q ≳ 1e-9`.
- **⑥ Per-tet quality diagnostic** — measure `q`, warn at load, make ①–③
  measurable.
- **② Iterative refinement** — cheap accuracy recovery for moderate cases
  (`cond·u < 1`); a no-op when well conditioned.
- **① Symmetric diagonal equilibration** — solution-safe global hygiene,
  modest gain; matters most when materials/PML mix scales.
