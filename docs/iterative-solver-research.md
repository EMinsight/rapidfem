# Iterative Solver Research — Auxiliary-Space Preconditioning for Second-Kind Nédélec-2 Time-Harmonic Maxwell

Scoping document for an iterative solver to push rapidfem past the ~500k-DOF wall where
PARDISO / faer LU become RAM-bound.

**Target problem.** Sparse complex-symmetric (not Hermitian) indefinite system
`K = E − k₀² B` over second-kind Nédélec elements of order 2 (20 DOFs/tet, 8 DOFs/tri).
PML, ABC (Robin order 2), waveguide-port and lumped-port boundary terms supported.
Pure Rust shipping path via PyPI wheel; **HYPRE/MFEM/PETSc cannot be runtime deps**
(only benchmarks/reference). System is held native complex throughout — no 2×2 real
block expansion currently.

**Code reference points in rapidfem.**

- `src/basis.rs` — `Nedelec2Basis`: DOF layout `[edge·m1 | face·m1 | edge·m2 | face·m2]`,
  total `n_field = 2·n_edges + 2·n_tris`.
- `src/assembly.rs` — `assemble_and_solve_with_pml`: builds `K = E − k₀²B`, applies
  PEC elimination, hands `K, b` to faer LU (default) or PARDISO (opt-in).
- `src/tet_assembly.rs`, `src/tri_assembly.rs`, `src/abc_order2.rs`, `src/waveguide.rs`,
  `src/csym_ldlt.rs` (verified but unused), `src/pardiso.rs`.

---

## 1. Theoretical foundation — auxiliary-space preconditioning for H(curl) at higher order

### 1.1 The Hiptmair–Xu (HX) framework (foundation)

**Paper.** R. Hiptmair, J. Xu, *Nodal Auxiliary Space Preconditioning in H(curl) and H(div) Spaces*,
SIAM J. Numer. Anal. **45**(6), 2483–2509, 2007.
<https://epubs.siam.org/doi/10.1137/060660588>

**Key result.** For the H(curl)-elliptic problem `(α curl u, curl v) + (β u, v) = (f,v)`
with α, β > 0, a spectrally equivalent preconditioner is

```
B_HX = D⁻¹ + Π (Π_curl)⁻¹ Πᵀ + G (L_h)⁻¹ Gᵀ
```

where
- `D` is a Jacobi/Gauss–Seidel smoother on the H(curl) matrix itself;
- `G : V_h^scalar → V_h^edge` is the **discrete gradient** (one row per edge, two ±1
  entries on the endpoint vertices, in the lowest-order ND_0 case);
- `Π : (V_h^scalar)³ → V_h^edge` is the **Nédélec interpolation** of vector P1 fields
  into the edge space;
- `Π_curl` is the SPD `(P1)³` curl-curl + mass operator (block-diagonal vector
  Laplacian-like);
- `L_h` is the **scalar P1 Laplacian + mass** on vertex DOFs.

The condition number bound is **independent of mesh size**, and `B_HX` is robust under
jumps in α (less so in β).

**The higher-order question.** The 2007 paper is *stated* for ND_0, but the proof
relies only on (i) a regular decomposition `u = v + grad ψ` with `v ∈ (H¹)³`, and
(ii) the existence of a stable commuting interpolation `Π_h : (H¹)³ → V_h`. Both
extend to higher-order Nédélec spaces, **first or second kind**, on simplices. The
*practical* difficulty is constructing `Π` and `G` for the higher-order DOF layout
and proving their cost stays bounded — which the literature has now mostly handled.

**Why it matters.** This is exactly the preconditioner we need, but every nontrivial
piece (Π, G, AMG on vector / scalar Poisson) must be rebuilt for second-kind ND_2.
The HX construction does *not* care that the system is complex-symmetric or even
indefinite per se — the spectral equivalence is for the absolute-value /
real-positive-definite "preconditioner operator", not the system itself
(see section 1.5 and section 7).

### 1.2 Kolev–Vassilevski — parallel implementation (HYPRE AMS)

**Paper.** Tz. Kolev, P. Vassilevski, *Parallel Auxiliary Space AMG for H(curl) Problems*,
J. Comp. Math. **27**, 604–623, 2009. PDF: <https://www.osti.gov/servlets/purl/1670552>.
Earlier: *Parallel H¹-based auxiliary space AMG solver for H(curl) problems*,
LLNL-TR-238695, 2007: <https://www.osti.gov/biblio/897951>.

**Key result.** Implemented HX as the **AMS** solver inside HYPRE, with BoomerAMG
(classical Ruge–Stüben in HYPRE) used as the black-box AMG on the auxiliary Poisson
problems. The paper documents the "high-order interface" of AMS:

> "the user does not need to provide the coordinates of the vertices … but instead
> should construct and pass the Nedelec interpolation matrix Π which maps (high-order)
> vector nodal finite elements into the (high-order) Nedelec space."
> ([HYPRE AMS docs](https://hypre.readthedocs.io/en/latest/solvers-ams.html))

The matrix Π has columns ordered node-by-node `(x,y,z components interleaved)` and
each row represents a Nédélec DOF expressed as a linear combination of the
P_k³/Q_k³ basis evaluated at the appropriate point/moment.

**Why it matters.** Two things: (a) the *theoretical* extension to higher order is
already in production use here; (b) AMS itself is the *operational* template — even
in pure Rust we want the same algebraic structure, only with our own AMG and
Π, G constructions.

### 1.3 The Wikipedia summary / status

<https://en.wikipedia.org/wiki/Hiptmair%E2%80%93Xu_preconditioner> — useful one-page
overview noting HX was named one of the DOE "top ten computational breakthroughs."

### 1.4 Newer / refined HX theory (2015+)

- **Hiptmair, Li, Xu, *Nodal Auxiliary Space Preconditioning for the Surface
  de Rham Complex***, Found. Comput. Math., 2023 (arXiv:2107.07978).
  <https://arxiv.org/abs/2107.07978> — generalises HX to the surface de Rham
  complex; mainly relevant if we ever need a 2-D / shell variant.
- **Substructuring the HX preconditioner for positive definite H(curl) problems**,
  BIT Num. Math., 2024.
  <https://link.springer.com/article/10.1007/s10543-024-01031-y> — Schur-complement
  variant, useful for substructuring / DD layering if we go multi-region later;
  not on the critical path for v1.
- **Auxiliary Space AMG for H(curl) Problems on Hexahedral DG Meshes**, 2019
  <https://link.springer.com/chapter/10.1007/978-3-319-93873-8_20> — confirms HX
  carries to non-conforming DG settings; we use conforming tets so this is just
  reassuring.

### 1.5 Complex-symmetric / indefinite curl-curl preconditioning

HX as originally stated targets the **positive definite** `curl-curl + mass`. For
`(curl α curl − k₀² β) E = J` (our case), the operator is indefinite once
`k₀² β > σ_min(curl α curl)` on some mode. The pragmatic approaches in the
literature are:

1. **Preconditioner on `|K|`, not on K.** Build the auxiliary preconditioner from
   `curl α curl + k₀² β` (positive shift, absolute-value of the mass term) and apply
   it as a preconditioner to GMRES/FGMRES solving with the true `K`. This is what
   MFEM's **`ex25p`** PML example does (see `examples/ex25p.cpp`, lines 524–586 in
   the local copy at `C:\Repositories\TEMP\mfem\examples\ex25p.cpp`):

   ```cpp
   ParBilinearForm prec(fespace);
   prec.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv));
   prec.AddDomainIntegrator(new VectorFEMassIntegrator(restr_absomeg));   // |ω²ε|, NOT −ω²ε
   // ... plus PML "absolute value" coefficients
   pc_r.reset(new HypreAMS(*PCOpAh.As<HypreParMatrix>(), fespace));
   // Block-diagonal real/imag preconditioner; GMRES with kdim=200 on the true K
   ```

   This is the de-facto standard for indefinite curl-curl + AMS.

2. **Ledger et al., Preconditioners for the indefinite linear system arising from
   the hp discretization of Maxwell's equations**, Commun. Numer. Methods Eng., 2009.
   <https://onlinelibrary.wiley.com/doi/10.1002/cnm.1131> — directly on point for
   "what to do in the indefinite hp case", recommends the absolute-value
   preconditioner approach.

3. **Hu, Li, Zou, *An Effective Preconditioner for a PML System for the
   Time-Harmonic Maxwell Equation***, M2AN, 2014.
   <https://www.math.cuhk.edu.hk/~zou/publication/m2an14hu.pdf> — closest analogue
   to our setting: PML + indefinite, develops block preconditioners and analyses
   their effectivity.

4. **Bonazzoli et al., GenEO-type DD preconditioners for H(curl)** (Springer JoSciComp 2025)
   <https://link.springer.com/article/10.1007/s10915-025-03061-2> — robust DD; high
   cost (coarse space from local eigenproblems), only worth it if HX-AMS stalls.

**Bottom line for rapidfem.** Build the **positive-definite "shifted" surrogate** `M = curl α curl + k₀² β` and use HX-AMS preconditioning on `M`, apply to Krylov on the true `K`. This is the well-validated path. The 2007 paper is theoretically silent on indefinite K, but the literature plus MFEM `ex25p` confirms it works in practice provided the Krylov method is the "right" one (Section 5).

### 1.6 Schöberl & Zaglmayr — high-order Nédélec construction

**Paper.** J. Schöberl, S. Zaglmayr, *High order Nédélec elements with local complete
sequence properties*, COMPEL **24**(2), 374–384, 2005.
<https://www.emerald.com/insight/content/doi/10.1108/03321640510586015/full/html>

**Key result.** Hierarchical basis where each edge/face/cell block separately satisfies
a local exact sequence. The crucial consequence:

> "a second advantage of this construction is that simple block-diagonal preconditioning gets efficient"

i.e. one can build an HX-like decomposition **block-by-block** without an explicit
auxiliary mesh, because the high-order blocks split cleanly along the de Rham complex
into gradient and non-gradient parts. NGSolve's `hcurlamg` preconditioner
(based on Reitzinger–Schöberl) uses this.

**Why it matters for us (only partially).** Our basis is *second-kind ND_2*, not the
Schöberl–Zaglmayr hierarchical first-kind basis, and we don't have a hierarchical
gradient sub-basis baked in. The construction is informative but not directly
applicable; building Π onto vector P1 is still the simplest path.

---

## 2. Static-condensation / Low-Order-Refined (LOR) approach

**The core insight.** Instead of constructing Π and G analytically for the high-order
space, refine the *mesh* so that the lowest-order Nédélec space on the refined mesh
has the *same DOF count* as the high-order space on the coarse mesh, and is
spectrally equivalent. Apply HYPRE AMS / classical HX to that low-order refined
operator. You get high-order accuracy with a low-order, well-studied preconditioner.

### 2.1 Key papers

1. **W. Pazner, T. Kolev, *Efficient Low-Order Refined Preconditioners for High-Order
   Matrix-Free Continuous and Discontinuous Galerkin Methods***, SIAM J. Sci. Comput.
   **43**(5), S475–S498, 2021. <https://epubs.siam.org/doi/10.1137/19M1282052>
   — establishes spectral equivalence for the **H¹** case and the matrix-free framework.

2. **W. Pazner, T. Kolev, C. R. Dohrmann, *Low-Order Preconditioning for the High-Order
   Finite Element de Rham Complex***, SIAM J. Sci. Comput. **45**(2), A675–A702, 2023.
   arXiv:2203.02465 <https://arxiv.org/abs/2203.02465>
   <https://epubs.siam.org/doi/10.1137/22M1486534>
   — **the** paper for our case. Unifies LOR preconditioning across H¹, H(curl), H(div)
   for Nédélec / Raviart–Thomas. Uses polynomial **histopolation** (interpolation by
   prescribed mean values over sub-cell regions) to define the LOR-to-HO basis change.
   Spectral equivalence **independent of polynomial degree p and mesh size h**.
   This is the most generally-applicable theoretical foundation for "AMS on
   higher-order curl-curl".

3. **A. T. Barker, T. Kolev, *Matrix-free preconditioning for high-order H(curl)
   discretizations***, Numer. Linear Algebra Appl. **28**(2), e2348, 2021.
   <https://doi.org/10.1002/nla.2348> — the predecessor / 2-D companion to (2).
   Establishes that a *sparsified* H¹ solver on the high-order nodal lattice gives
   a sufficient auxiliary-space preconditioner with `O(p^{d+1})` flops, `O(p^d)`
   memory. Important for the cost analysis.

4. **W. Pazner, T. Kolev, J.-S. Camier, *End-to-end GPU acceleration of LOR
   preconditioning for high-order FE***, Int. J. HPC Appl., 2023.
   <https://journals.sagepub.com/doi/10.1177/10943420231175462>
   — engineering paper on the GPU implementation. Not directly applicable to our
   CPU-Rust setting but the assembly pattern (batched per-element, no temp matrices)
   maps cleanly to rayon-parallel loops.

### 2.2 Trade-offs: full Π operator vs. LOR

| Aspect | Direct higher-order Π | LOR + AMS on refined mesh |
| ---- | ---- | ---- |
| Theory needed | Π and G explicit, per element, second-kind specific | Polynomial histopolation table only |
| Implementation pain | Each new basis = new derivation | One LOR refiner that respects de Rham, then reuse AMS |
| Memory | Π is `n_field × 3·n_vert` sparse | LOR matrix has roughly 6×—9× more rows than the HO matrix |
| Smoother quality | HX smoother on HO matrix directly (sharp) | Smoother on LOR matrix; HO matrix used only for matvec & residual |
| Best when | Order ≤ 2–3, fixed basis | Order ≥ 3 or moving to other H(curl) bases later |
| Reference impl | None for ND-2 second kind (we'd be first) | MFEM `LORSolver<HypreAMS>`, well-tested |

For **second-kind ND_2 fixed**, building Π directly is competitive in effort and
much cheaper at runtime. LOR becomes attractive only if rapidfem later supports
multiple orders or first/second-kind switching.

### 2.3 Reference code in MFEM

Local clone: `C:\Repositories\TEMP\mfem`.

- `fem/lor/lor.hpp`, `lor.cpp` — `LORBase`, `LORDiscretization`, `ParLORDiscretization`,
  including `GetDofPermutation()` (LOR-to-HO DOF mapping — nontrivial for ND/RT).
- `fem/lor/lor_nd.hpp`, `lor_nd_impl.hpp` — `BatchedLOR_ND` template kernels
  `Assemble2D<ORDER,SDIM>()` and `Assemble3D<ORDER>()`. Per-tet sub-element loop,
  precomputed sparsity (`nnz_per_row = 7` in 2D).
- `fem/lor/lor_ams.hpp`, `lor_ams.cpp` — `BatchedLOR_AMS` constructs the discrete
  gradient `G` (LOR vertices → LOR edges) and coordinate vectors for AMS. Note
  `Form3DEdgeToVertex` (line 105 of `lor_ams.cpp`) is the part that needs an
  ND-specific edge↔vertex incidence pattern on the LOR-refined element.
- `linalg/hypre.hpp` lines 1984–2058 — `HypreAMS` wrapper, including the
  high-order constructor signature
  `HypreAMS(const HypreParMatrix &A, HypreParMatrix *G, HypreParVector *x, *y, *z)`
  used by `LORSolver<HypreAMS>` (line 363 of `lor_ams.cpp`).

---

## 3. Direct higher-order Π-interpolation for second-kind Nédélec-2

This is the path with the most up-front math but the cleanest runtime artifact:
a single sparse `Π` matrix and a single sparse `G` matrix that the rest of the
preconditioner code uses uniformly.

### 3.1 What Π must do

`Π : (P1)³_h → ND_2^{II}_h` — given a piecewise-linear vector field on the mesh
vertices (one (x,y,z) triple per vertex, `3·n_vert` DOFs total), return its
projection onto our 20-DOF/tet second-kind Nédélec-2 space.

For second-kind ND, **DOFs are point-evaluation moments at well-chosen edge and
face nodes** (not edge averages as in first-kind). Specifically, for ND_2 second
kind on a tet (per Nédélec 1986 and the EMerge port — confirm against
`src/basis.rs` and `python/python_src/.../elements/nedelec2.py`):

- per edge: 2 moments — tangential value at two Gauss-Legendre points on the edge;
- per face: 2 vector-tangential moments at two points on the face;
- 20 = 6·2 (edge) + 4·2 (face); no interior cell DOFs at order 2.

Then column `(v, c)` of Π (vertex `v`, component `c ∈ {x,y,z}`) is built as follows:

> For each tet T containing v, for each DOF i of T, evaluate the basis function
> `φ_v^c` (linear in T, equal to `eᶜ` at v, 0 at other vertices) at the DOF nodal
> point, dot with the DOF tangent / face direction, accumulate into row
> `tet_to_field[T][i]`.

This is **purely local per-tet**, easily parallel with rayon, and produces a sparse
matrix with `≈ 20 × 12 = 240` nonzeros per tet (4 vertices × 3 components × 20 DOFs,
but most are shared with neighbours, so the final NNZ is much smaller).

### 3.2 Discrete gradient G

`G : P1_h → ND_2^{II}_h` (scalar P1 on vertices → ND_2 edge field). Same structure as Π
but the input is a scalar ψ with `eᶜ ↦ ∇ψ`. For each DOF i of T, the value is
`∇ψ · t_i` evaluated at the DOF point, where `∇ψ` is constant on T and given by the
shape-function gradients we already cache in `src/coefficients.rs`.

For second-kind ND_2 in particular, `range(G) = grad(P1)` is the **kernel of the
curl-curl operator**, which is what makes HX work in the first place — this is a
verifiable invariant: `K · G · ψ` should be 0 (up to roundoff and PEC elimination)
for any vertex potential ψ on a homogeneous problem.

### 3.3 Open-source examples to study

- **MFEM** has the closest analogue, but only for **first-kind** ND_k via HYPRE AMS:
  see `fem/fespace.cpp` / `fem/transfer.cpp` and `linalg/hypre.cpp::HypreAMS::MakeGradientAndInterpolation()`
  (function declared in `linalg/hypre.hpp`). Not directly portable but the algorithmic
  pattern (loop over elements, write per-element block of Π) is exactly what we need.
- **NGSolve**: `ngsolve/fem/hcurlfe.cpp` and `comp/hcurlhdivfes.cpp` build commuting
  interpolators for the Schöberl–Zaglmayr basis. Different basis, but the **structure**
  of constructing per-element Π matrices using DOF functionals is transferable.
- **Firedrake**: `firedrake/preconditioners/hypre_ams.py`
  ([source](https://www.firedrakeproject.org/_modules/firedrake/preconditioners/hypre_ams.html))
  shows the clean Python interface — builds G via
  `assemble(interpolate(grad(TrialFunction(P1)), V))`, which is FIAT-driven. Two
  important caveats it documents:
  - **"Hypre AMS requires lowest order Nedelec elements!"** (hard error if
    `formdegree != 1 or degree != 1`).
  - **"HypreAMS preconditioner not yet implemented in complex mode"** — and
    Firedrake's complex mode mostly bypasses HypreAMS for our class of problem.

  So Firedrake, despite being the most "spiritually similar" project, is **not** a
  proven reference for our combination (higher-order + complex).

### 3.4 Clean derivations in papers

- The **EMerge / EMagPy** source we already mirror has explicit second-kind ND_2 DOF
  definitions; the algebra carries over.
- Nédélec's original *Mixed finite elements in R³* (Numer. Math. 1980, second-kind
  paper 1986) gives the definitive DOF functionals.
- For a worked-out higher-order Π formula in code, the cleanest pedagogical reference
  is **Demkowicz et al., Computing with hp-Adaptive Finite Elements, vol. 2 (2007)**,
  ch. 4 — but it's for projection-based interpolation, not nodal, which gives a slightly
  different (and pricier per-element) Π.

---

## 4. Reference implementations — concrete file paths and what to learn

### 4.1 MFEM (most directly useful)

Local: `C:\Repositories\TEMP\mfem`. Repo: <https://github.com/mfem/mfem>.

| File | What to study |
| ---- | ---- |
| `examples/ex22p.cpp` | Damped harmonic oscillator H(curl) variant (`prob=1`). Lines 365–476 are **the template** for "indefinite curl-curl preconditioner": absolute-value mass on PC, real/imag block-diagonal HypreAMS, FGMRES. Order arbitrary (set with `-o`), uses standard `HypreAMS(matrix, fespace)` constructor — for HO MFEM handles the Π internally. |
| `examples/ex25p.cpp` | The PML version. Lines 522–586. Switches to GMRES with `kdim=200`, `rel_tol=1e-5`. Uses *absolute value* of the PML jacobian coefficients in the preconditioner bilinear form. This is the closest published recipe to what rapidfem needs. |
| `examples/ex31p.cpp` | Definite Maxwell with anisotropy — sanity reference for HX with non-trivial α. |
| `linalg/hypre.hpp` lines 1984–2058 | `HypreAMS` C++ wrapper: shows minimal data needed by AMS: matrix A, discrete gradient G, x/y/z coordinate vectors (or for HO: Π_full + Π_x/y/z components). |
| `linalg/hypre.cpp::HypreAMS::MakeGradientAndInterpolation()` | The actual construction code — read for ND_k Π/G algorithm reference. |
| `fem/lor/lor_nd.hpp` + `lor_nd_impl.hpp` | Batched LOR-ND assembly: per-macroelement, per-sub-element loops with precomputed sparsity. **Direct pattern to port to Rust + rayon.** |
| `fem/lor/lor_ams.cpp` | `BatchedLOR_AMS::Form3DEdgeToVertex` and `FormGradientMatrix` — gives the LOR discrete gradient from element-restriction operators. |
| `fem/lor/lor.hpp` | `LORBase::GetDofPermutation()` — the LOR-to-HO DOF permutation for ND/RT spaces (nontrivial because edge/face DOFs reorder). |

### 4.2 HYPRE

Repo: <https://github.com/hypre-space/hypre>. The internal AMS solver:

- `src/parcsr_ls/ams.c` — the actual algorithmic loop. About 4000 lines but the kernel
  is small: SOR/L1-Jacobi smoother on K, BoomerAMG on `Πᵀ K Π` and on `Gᵀ K G`,
  combine additively or multiplicatively (cycle types 1–14).
- `src/parcsr_ls/ams.h` — public interface declarations.
- Read for the **smoother choice on the indefinite case**: HYPRE AMS uses ℓ¹-smoothing
  by default for the outer Hiptmair iteration, which is robust under indefiniteness.

### 4.3 Firedrake

Repo: <https://github.com/firedrakeproject/firedrake>. Mostly *negative* lessons:

- `firedrake/preconditioners/hypre_ams.py` — see Section 3.3. **Hard-codes ND_1**.
- `firedrake/preconditioners/hiptmair.py` — the `HiptmairPC` class. Two-level method
  for H(curl) with H¹ auxiliary space, internally a PCMG. Order-agnostic in principle
  but the smoother is patch-relaxation (vertex/edge stars), which is expensive to
  port. Read for the **auxiliary-space mathematics** at the PETSc level
  ([docs](https://www.firedrakeproject.org/firedrake.preconditioners.html)).
- `firedrake/preconditioners/asm.py` — auxiliary-space patch preconditioners.

### 4.4 deal.II

Repo: <https://github.com/dealii/dealii>. Direct quote from their wiki:

> "Time-harmonic Maxwell's equations lack the usual notion of local smoothing
> properties, which renders the usual suspects, such as a geometric multigrid,
> largely useless."
> ([Electromagnetic problem wiki](https://github.com/dealii/dealii/wiki/Electromagnetic-problem))

deal.II does **not** ship a native HX/AMS. Tutorials use:
- `step-81` — time-harmonic Maxwell with **direct solver** (UMFPACK), explicitly
  saying iterative preconditioning is not provided.
- `PreconditionAMG` is the wrapper around **Trilinos ML** (smoothed-aggregation) /
  hypre BoomerAMG — but only used for scalar/H¹ in the tutorials.

**Lesson: deal.II is *not* a useful reference for our problem.**

### 4.5 NGSolve

Repo: <https://github.com/NGSolve/ngsolve>. Heavy hitter for high-order Maxwell:

- `comp/preconditioner.hpp/cpp`, `comp/hcurlfes.cpp`, `fem/hcurlfe.cpp` — high-order
  hierarchical (first-kind) bases with built-in commuting projector.
- `linalg/amg.cpp` and the separate `hcurlamg` add-on package — Reitzinger–Schöberl
  edge-AMG. Different algorithm from HX (does not need a vertex-space AMG); the
  prolongation is built from the discrete gradient by hand.
- `comp/bddc.cpp` — BDDC preconditioner; good for very high order, but BDDC requires
  static condensation infrastructure we don't have. Out of scope for v1.

**Lesson.** Confirms that high-order H(curl) preconditioning is mature for SPD/definite
problems and shows two viable families (HX with vertex AMG vs. Reitzinger–Schöberl
direct edge AMG). For our complex-symmetric indefinite case the NGSolve community
typically still uses direct solvers (or BDDC) for moderate sizes — they don't have
a published "high-order AMS-equivalent for indefinite Maxwell" that we could mirror.

### 4.6 PETSc

`src/ksp/pc/impls/hypre/hypre.c` (in PETSc) is the user-side wrapper for AMS:
`PCHYPRESetDiscreteGradient`, `PCHYPRESetInterpolations(dim, RT_PiFull, RT_Pi[], ND_PiFull, ND_Pi[])`
([docs](https://petsc.org/main/manualpages/PC/PCHYPRESetInterpolations/)).
This is the API any "bring your own Π/G" framework targets. If we ever want to
make rapidfem talk to an external HYPRE benchmark, **this is the exact set of
matrices we'd need to expose** — and once we have them, our own pure-Rust HX
implementation has the same artifacts.

### 4.7 Pure-Rust ecosystem audit

What exists today (May 2026):

| Crate | Status | Use to us |
| ---- | ---- | ---- |
| `faer` | Production-quality, complex support, sparse LU/QR/Cholesky (incl. `faer-sparse`). Authored by Sarah El Kazdadi. | Already our default LU. Provides the building blocks for matvec, sparse linear algebra, partial pivoting — but **no AMG**. |
| `sprs` | Mature, lightweight CSR/CSC, basic iterative solvers (CG, BiCGStab). | Useful for CSR plumbing; not enough on its own. |
| `nalgebra-sparse` | Sparse extension to nalgebra; small ecosystem; less performant than faer for large. | Avoid. |
| `russell_sparse` | Wraps MUMPS / UMFPACK via FFI. | Cross-platform wheel-shipping is the same problem as HYPRE — system Fortran/BLAS deps. Useful only as a benchmark, not a default. |
| `rsparse` | Pure-Rust port of CSparse (cholsol/lusol/qrsol). | Reference / verification for direct solves; no AMG. |
| `scirs2-sparse` | New (2025), SciPy-port project, claims `SmoothedAggregationAMG`. | Worth a look — but the project is young and "production-ready" is not established. Crates.io page is sparse, no published benchmarks vs. HYPRE/AMGX, broad scope reduces depth. **Don't depend on it for the AMG kernel.** |
| `spsolve` | Wraps SuperLU/MUMPS — same FFI issue. | Benchmark only. |
| (none) | A pure-Rust BoomerAMG or smoothed-aggregation implementation of production quality. | **Gap. We would be building this.** |

**Conclusion.** No off-the-shelf pure-Rust AMG of production quality exists for our
purpose as of mid-2026. Either we build one or we wrap.

---

## 5. Krylov method choice for complex-symmetric + (likely) non-symmetric preconditioner

### 5.1 Methods to consider

| Method | Storage | Optimality | Notes |
| ---- | ---- | ---- | ---- |
| **GMRES (restarted)** | O(m·N), m = restart | Yes (over Krylov_m) | Works for any prec. Restart needed for memory. Standard fallback. |
| **FGMRES** | O(m·N) for two bases | Yes (flexible) | Allows **prec to vary across iterations** (e.g. inner GMRES on the auxiliary AMG). MFEM `ex22p` uses FGMRES. |
| **COCG** | O(N) (short recurrence) | "Galerkin" only | Exploits complex symmetry of `K`. **Requires symmetric preconditioner** in the same bilinear form sense (M^T = M, with transpose, not conjugate). HX-AMS is *not* symmetric in this sense (it has a smoother + nonsymmetric combinations). |
| **COCR** | O(N) | Smoother residual than COCG | Same restrictions. Sogabe–Zhang 2007. |
| **QMR** (Freund) | O(N) | Quasi-minimal residual | The original choice for complex symmetric. Long-lookback variants exist. |
| **BiCGStab** | O(N) | Heuristic stabilisation | Generic, robust, but no symmetry exploitation. |
| **TFQMR / IDR(s)** | O(N)–O(s·N) | IDR is best non-restart short-recurrence option for general complex matrices. | Less standard for this exact problem class. |

### 5.2 What the literature recommends, and what reference codes do

- **MFEM `ex22p`** (`examples/ex22p.cpp` line 470): **FGMRES** + block-diagonal
  preconditioner with HypreAMS on the real block. The block-diagonal preconditioner
  is built on a *different* (absolute-value) bilinear form, so the preconditioner
  is **not** complex-symmetric ⇒ COCG/COCR are out of the question; we *must* use
  GMRES/FGMRES/BiCGStab.
- **MFEM `ex25p`** (PML): **GMRES**, kdim=200, max=2000, rtol=1e-5.
- **Sogabe et al., QMR / COCG / COCR for complex-symmetric**: COCG/COCR shine when
  the preconditioner is itself complex-symmetric (e.g. ILU with the symmetric pattern,
  or shifted-Laplacian SPD prec). For HX-AMS-style preconditioners (mixed Jacobi +
  Π solve + G solve) the preconditioner is **not** complex-symmetric, so the COCG/COCR
  short-recurrence breaks down or stagnates.
- **Mardal et al., Iterative methods for Maxwell** — generally favour FGMRES when the
  inner preconditioner is itself iterative (AMG V-cycle ≈ varying op), MINRES only
  for genuinely symmetric definite cases.
- **Freund's original QMR paper** is still the right reference if we ever build a
  matched (complex-symmetric) preconditioner — but that's far from where HX-AMS lives.

### 5.3 Recommendation for rapidfem

1. **Default**: restarted **FGMRES**(60) with HX-AMS preconditioner, real/imag block
   diagonal pattern of MFEM `ex22p`/`ex25p`. Tolerance 1e-8 absolute or 1e-6 relative.
2. **Fallback**: **BiCGStab(2)** with the same preconditioner — useful when GMRES
   memory cost on large 3-D problems hurts; loses optimality but short recurrence.
3. **Only if we ever build a complex-symmetric preconditioner** (e.g. a shifted-Laplacian
   ILDLᵀ): switch to **COCG** with that preconditioner. Defer.

For the eventual GPU / WASM port, FGMRES Arnoldi is more memory-bandwidth-bound per
iteration than BiCGStab. For pure CPU + rayon, FGMRES is fine.

---

## 6. AMG for the auxiliary scalar / vector Poisson problems

This is the **single biggest piece of infrastructure** to build.

### 6.1 What we need

For 3-D HX on second-kind ND_2:

- **`A_vec`**: vector P1 Laplacian + mass: `3·n_vert × 3·n_vert`, block-diagonal in
  components (each component is the standard scalar P1 Laplacian). Used by
  `(Π_curl)⁻¹ ≈ AMG(A_vec)`.
- **`A_scal`**: scalar P1 Laplacian: `n_vert × n_vert`. Used by `L_h⁻¹ ≈ AMG(A_scal)`.

Both are real, SPD, the classic AMG target.

### 6.2 SOTA: classical Ruge–Stüben vs. smoothed aggregation

| Approach | Strengths | Weaknesses | Library exemplars |
| ---- | ---- | ---- | ---- |
| **Classical Ruge–Stüben (RS)** | Best on M-matrix-like scalar elliptic; sharp theory; great on SPD H¹ Laplacian; default in HYPRE BoomerAMG. | More expensive setup; poorer on anisotropic/jumps. | HYPRE BoomerAMG, AmgCL `amg<runtime>`. |
| **Smoothed Aggregation (SA)** | Naturally handles vector / elasticity problems via "near-null-space modes" (rigid-body translations). Cheaper setup. | Less theoretical sharpness for pure scalar Laplacian; needs near-null-space input for vector case. | Trilinos ML/MueLu, AmgCL `relaxation::spai0` + aggregates, PyAMG. |

For our case **both A_scal and A_vec are SPD scalar/vector Laplacians from a vertex
P1 discretization** — the *easiest* case for AMG.

Practical recommendation: **classical Ruge–Stüben for A_scal, smoothed aggregation
with constant near-null-space modes for A_vec**. SA is also slightly easier to
implement correctly in pure Rust because the prolongation construction is more
algebraically straightforward (aggregate → constant prolongator → smooth by Jacobi).

### 6.3 Cost of building AMG from scratch in pure Rust — honest estimate

A *production-quality* AMG that competes with HYPRE BoomerAMG is a multi-person-year
project. A *workable* SA-AMG sufficient for our preconditioner is much smaller:

| Component | Difficulty | Effort (engineer-weeks) |
| ---- | ---- | ---- |
| Strong-connection graph (C-style) | low | 1 |
| Aggregation (greedy + max-independent-set) | low | 1 |
| Tentative prolongator from near-null-space modes | medium | 1 |
| Jacobi smoothing of prolongator | low | 0.5 |
| Galerkin coarse op `R A P` (= `Pᵀ A P`) | low (sparse triple product) | 1 |
| Smoother (symmetric Gauss–Seidel, ℓ¹-Jacobi) | low | 1 |
| V-cycle and W-cycle drivers + setup/solve API | medium | 1 |
| Multilevel hierarchy with coarsest-level direct solve | low (re-use faer) | 0.5 |
| Robustness on jumps / anisotropy / boundary modifications | high | 2 |
| Unit / integration tests vs. PyAMG or AmgCL on canned problems | medium | 1 |
| **Total realistic baseline** | | **≈ 10 weeks** |

This is a real cost. The alternative is a **conditional opt-in HYPRE FFI** for
power users (off the wheel-shipping critical path), used as a benchmark target.

### 6.4 Wrap option: AmgCL via FFI

[AmgCL](https://github.com/ddemidov/amgcl) is a header-only C++ AMG library with
runtime-configurable backends. Smaller footprint than HYPRE, no MPI dependency,
licenses MIT. Could be wrapped via `cc` crate + `cxx`. Wheel-shipping concern is
the C++ stdlib symbol resolution on Linux — solvable but adds maintenance cost.

Hybrid play: **build our own SA-AMG to MVP quality, validate against AmgCL via
FFI in the test harness only.** Keeps the production wheel pure-Rust.

---

## 7. Complex shift / shifted-Laplacian for high-frequency Maxwell

### 7.1 Helmholtz origin

Erlangga, Vuik, Oosterlee (2004/2006) introduced the **complex shifted Laplacian**
`(−Δ − (1 + iβ)k²)` as a preconditioner for `(−Δ − k²)` in scalar Helmholtz.
The complex shift moves all eigenvalues off the real axis, so multigrid actually
converges on the preconditioner system. β = 0.5 is the canonical default.

Reference: Y. A. Erlangga, C. W. Oosterlee, C. Vuik, *A novel multigrid based
preconditioner for heterogeneous Helmholtz problems*, SIAM J. Sci. Comput., 2006.
<https://homepages.cwi.nl/~barry/chapter10.pdf>

Convergence theory: Gander, Graham, Spence, *Applying GMRES to the Helmholtz
equation with shifted Laplacian preconditioning*, Numer. Math., 2015.
<https://link.springer.com/article/10.1007/s00211-015-0700-2>

### 7.2 What's been done for curl-curl

Direct extensions to time-harmonic Maxwell with `curl curl − k²` exist but are
**less mature**:

- The MFEM `ex25p` "absolute value" trick is essentially a shifted-Laplacian
  preconditioner with shift β chosen to make the auxiliary problem positive
  definite. Real shift, not complex.
- van Dijk et al. and follow-ons (~2010s) tried complex shift for full curl-curl
  + ε terms; results show convergence rate improves with shift but the wavenumber
  dependence (`# GMRES iterations ∝ k^α`) is not fully eliminated.
- The **multilevel Krylov–multigrid** approach (Erlangga–Nabben) extends to
  electromagnetics; not yet a community standard for engineering codes.

### 7.3 Recommendation for rapidfem

- For **v1 of the iterative solver** stick with the *real* "absolute-value"
  shift as in MFEM `ex25p`. This is what's validated and what every reference
  implementation does. Don't try to be more clever.
- **Future v2** experiment: parameterise the preconditioner shift as
  `α curl curl + (β_real + i·β_imag) k² ε`, with `β_real ≈ 1` (so we have the
  abs-value baseline as `β_imag=0`) and tune `β_imag ∈ [0.2, 0.7]`. Compare
  GMRES iteration counts on the WR-90 PML benchmark and the patch-antenna case.
  This is a research direction; flag as **open**.

---

## 8. Implementation building blocks — concrete sub-problems and difficulty

Estimates are engineer-weeks for **one moderately experienced numerical Rust
developer** (you), assuming the rest of rapidfem is already there. Calendar
duration can be 1.5–2× of effort weeks given context switching and testing.

| # | Sub-problem | Where it lives | Difficulty | Effort | Open / Standard |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 1 | **Π construction for second-kind ND_2** (per-tet, parallel via rayon) | new `src/aux/interp_pi.rs` | medium | 1.5w | Standard (DOF functionals well-defined); no public ND_2-2K reference, we'd be first. |
| 2 | **Discrete gradient G** for ND_2 | new `src/aux/grad.rs` | low–medium | 0.5w | Standard. |
| 3 | **Vertex P1 scalar Laplacian `A_scal`** assembly (already need vertex coords) | new `src/aux/p1_lap.rs` | low | 0.5w | Trivial. |
| 4 | **Vertex P1 vector Laplacian `A_vec`** (block-diag, 3 copies of A_scal in standard P1) | reuse #3 | low | 0.25w | Trivial. |
| 5 | **Smoothed-aggregation AMG** (real SPD) | new `src/aux/amg.rs` (or workspace crate) | **high** | 8–10w | Standard algorithm, big quality bar to reach to compete with HYPRE. **Single biggest item.** |
| 6 | **HX preconditioner driver** (smoother on K + Π·AMG(A_vec)·Πᵀ + G·AMG(A_scal)·Gᵀ) | new `src/iterative/hx_ams.rs` | medium | 2w | Standard once #1–#5 work. |
| 7 | **Complex-symmetric / indefinite handling** — build positive-definite surrogate `M = curl α curl + k² |β|` for the preconditioner only | new `src/iterative/abs_value_prec.rs` | medium | 1w | Standard (MFEM ex25p pattern). |
| 8 | **FGMRES(m) Krylov solver** for complex sparse, block-diag real/imag preconditioner | new `src/iterative/fgmres.rs` | medium | 1.5w | Standard; ensure complex arithmetic is consistent (some refs use complex conjugation in inner products, careful: for complex-symmetric matvec preconditioning we use the bilinear, not sesquilinear, inner product on residuals only at output). |
| 9 | **PEC elimination consistency** with auxiliary spaces (Π and G must respect PEC DOFs in K) | refactor of #1, #2 + `assembly.rs` | medium | 1w | Standard but easy to get wrong; HYPRE AMS docs warn: *"AMS expects a matrix defined on the whole mesh with no boundary edges/nodes excluded."* Implication: we apply PEC at K *after* applying the preconditioner on the un-eliminated G/Π. |
| 10 | **Benchmark harness** (iteration count vs. DOFs, wallclock, memory) vs. PARDISO/faer baselines on patch antenna, WR-90 PML, microstrip, iris filter | `tests/validation/` extension | medium | 1.5w | Standard. |
| 11 | **Optional: BiCGStab fallback** | extend #8 | low | 0.5w | Standard. |
| 12 | **Optional: AmgCL wrap as a benchmark/comparison** (gated behind a feature flag, not in default wheel) | `crates/amgcl-sys`? | medium | 1.5w | Standard but cross-platform build pain. |
| 13 | **Research: complex shift in #7** | extend #7 | high | 1–2w to first numbers, then open-ended | **Open research.** |

**Critical path (MVP).** Items 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ≈ **≥ 18 engineer-weeks**.
Item 5 (AMG) is ~50% of the total. Items 1–4 + 6–10 without 5 (i.e. using AmgCL via
FFI as the AMG kernel for an MVP) ≈ **~10 weeks**, gated by item 12 maintenance.

### Risk register

- **AMG quality is the load-bearing assumption.** If our home-grown SA-AMG is 2× slower
  per V-cycle than BoomerAMG, the whole iterative solver is 2× slower than the
  literature predicts, and PARDISO may still win up to ~1M DOFs. Mitigation: validate
  early against PyAMG (Python harness in `tests/validation/`).
- **Second-kind ND_2 Π has no published reference implementation.** Verification needed:
  test that `K · G · ψ ≈ 0` (kernel property) and `|| Π v − v ||` is small for smooth v.
  Both are checkable in unit tests against EMerge's known smooth field problems.
- **Complex-symmetric inner products in FGMRES**: the bilinear form `(x,y) = xᵀy`
  (no conjugate) breaks Krylov optimality theory; the sesquilinear form `(x,y) = x*y`
  preserves it but the preconditioner is not Hermitian. Use the sesquilinear form,
  follow MFEM convention. This is a one-liner choice but matters.
- **Indefiniteness near resonance.** At k near a discrete eigenvalue of `curl α curl`,
  K is *nearly singular* and any preconditioner stalls. This is inherent to the
  physics, not a solver bug. Document in user-facing notes; suggest frequency
  step refinement near resonances.

### Recommended phasing

**Phase A (4–6 weeks).** Wrap AmgCL behind FFI, build items 1, 2, 3, 4, 6, 7, 8, 9
(everything except own AMG and the optional BiCGStab/benchmark/research). Get an
end-to-end FGMRES + HX-AMS solver working in rapidfem against PARDISO on the
existing validation suite. This is the **fastest path to numbers**.

**Phase B (8–10 weeks).** Build the pure-Rust SA-AMG (item 5), validate on canned
SPD Laplacian problems against PyAMG, swap into HX-AMS, deprecate the AmgCL FFI
path or keep it as a feature-flagged benchmark.

**Phase C (open-ended).** Item 13, GPU/WASM port via faer's GPU backend if/when
available, BDDC/DD layering for very large problems (>10M DOFs).

---

## Appendix A — full citations

- R. Hiptmair, J. Xu, "Nodal Auxiliary Space Preconditioning in H(curl) and H(div) Spaces," SIAM J. Numer. Anal. **45**(6), 2007. <https://epubs.siam.org/doi/10.1137/060660588>
- Tz. Kolev, P. Vassilevski, "Parallel Auxiliary Space AMG for H(curl) Problems," J. Comp. Math. **27**, 2009. <https://www.osti.gov/servlets/purl/1670552>
- J. Schöberl, S. Zaglmayr, "High order Nédélec elements with local complete sequence properties," COMPEL **24**(2), 2005. <https://www.emerald.com/insight/content/doi/10.1108/03321640510586015>
- W. Pazner, T. Kolev, C. R. Dohrmann, "Low-Order Preconditioning for the High-Order Finite Element de Rham Complex," SIAM J. Sci. Comput. **45**(2), 2023. <https://arxiv.org/abs/2203.02465>
- W. Pazner, T. Kolev, "Efficient Low-Order Refined Preconditioners for High-Order Matrix-Free CG/DG Methods," SIAM J. Sci. Comput. **43**(5), 2021. <https://epubs.siam.org/doi/10.1137/19M1282052>
- A. T. Barker, T. Kolev, "Matrix-free preconditioning for high-order H(curl) discretizations," Numer. Linear Algebra Appl. **28**(2), 2021. <https://doi.org/10.1002/nla.2348>
- P. D. Ledger, K. Morgan, O. Hassan, "Preconditioners for the indefinite linear system arising from the hp discretization of Maxwell's equations," Commun. Numer. Methods Eng., 2009. <https://onlinelibrary.wiley.com/doi/10.1002/cnm.1131>
- Q. Hu, R. Li, J. Zou, "An Effective Preconditioner for a PML System For the Time-Harmonic Maxwell Equation," M2AN, 2014. <https://www.math.cuhk.edu.hk/~zou/publication/m2an14hu.pdf>
- Y. A. Erlangga, C. W. Oosterlee, C. Vuik, "A novel multigrid based preconditioner for heterogeneous Helmholtz problems," SIAM J. Sci. Comput., 2006. <https://homepages.cwi.nl/~barry/chapter10.pdf>
- M. J. Gander, I. G. Graham, E. A. Spence, "Applying GMRES to the Helmholtz equation with shifted Laplacian preconditioning," Numer. Math., 2015. <https://link.springer.com/article/10.1007/s00211-015-0700-2>
- T. Sogabe, S. L. Zhang, "A COCR method for solving complex symmetric linear systems," J. Comput. Appl. Math., 2007. <https://www.sciencedirect.com/science/article/pii/S0377042705007648>
- HYPRE AMS user manual: <https://hypre.readthedocs.io/en/latest/solvers-ams.html>
- HYPRE source: <https://github.com/hypre-space/hypre>
- MFEM source: <https://github.com/mfem/mfem> (local mirror `C:\Repositories\TEMP\mfem`)
- Firedrake HypreAMS: <https://www.firedrakeproject.org/_modules/firedrake/preconditioners/hypre_ams.html>
- PETSc `PCHYPRESetInterpolations`: <https://petsc.org/main/manualpages/PC/PCHYPRESetInterpolations/>
- NGSolve docs: <https://docu.ngsolve.org/latest/>

---

## Appendix B — opinionated TL;DR

1. **Use HX directly with a direct Π for ND_2 second-kind.** Don't go LOR. Effort
   savings are illusory at fixed order 2; LOR pays off at order ≥ 3 or for a multi-order codebase.
2. **The preconditioner is on `M = curl α curl + k²|β|`, not on K.** MFEM `ex25p` recipe.
3. **FGMRES(60) is the right Krylov.** COCG/COCR only become relevant if we ever build
   a complex-symmetric preconditioner, which we won't in v1.
4. **The single biggest item is real SPD AMG.** Either build pure-Rust SA-AMG
   (~10 weeks, real risk on quality) or wrap AmgCL behind FFI as a stepping stone
   (~1.5 weeks, but no longer pure-Rust).
5. **Phase A = wrap AmgCL + everything else. Phase B = replace AmgCL with our own AMG.**
   This is the lowest-risk path to actually getting iteration-count numbers we can
   defend in two months instead of six.
6. **The complex shift question is open research; skip in v1.**
