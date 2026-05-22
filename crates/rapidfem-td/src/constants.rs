//! Centralised numerical constants for the time-domain backend.
//!
//! Every tolerance and algorithmic threshold lives here — never inline,
//! never hardcoded at a call site. A solver constant that needs tuning is
//! tuned in one place. The numerical precision is centralised here too:
//! [`Field`] and [`Accum`] are the two scalar types the solver is written
//! against, so a precision change is a one-line edit, not a sweep.

// ── Numerical precision ───────────────────────────────────────────────────

/// Scalar type of the electromagnetic field state and the matrix-free
/// spatial operator ([`crate::rhs::MaxwellOperator`] and its `apply`).
///
/// This is the precision a future GPU / mixed-precision path may drop to
/// `f32`: the per-element curl + flux is the throughput-bound hot path and
/// the part a consumer GPU runs fastest in single precision. On the CPU it
/// is `f64` today. The operator-side data path — fields, fluxes, geometric
/// factors, material tensors — is meant to be typed `Field` end to end.
///
/// `Field` and [`Accum`] meet at the matvec boundary inside
/// [`crate::propagator::KrylovWorkspace::expmv_into`]: once they differ,
/// the matvec closure passed in owns the down/up-cast.
pub type Field = f64;

/// Scalar type of the Krylov accumulator: the Arnoldi basis, the CGS2
/// orthogonalisation, the Hessenberg matrix and the dense `expm`.
///
/// Stays `f64` — the scaling-and-squaring propagator and the
/// [`KRYLOV_TOL`]` = 1e-10` a-posteriori estimate depend on double
/// precision, so this is *not* a flip target. Typing the propagator
/// against `Accum` documents that boundary rather than enabling a change.
pub type Accum = f64;

// ── Krylov / ETD exponential propagator ───────────────────────────────────

/// Relative a-posteriori error tolerance for the Krylov exponential
/// propagator. After each Arnoldi vector the step's error estimate is
/// checked against this; the subspace stops growing once it converges, so
/// an easy step costs far fewer matvecs than the `krylov_dim` cap.
pub const KRYLOV_TOL: Accum = 1e-10;

/// Arnoldi lucky-breakdown threshold. When the next basis vector's norm
/// falls below this the Krylov subspace already spans the relevant
/// invariant subspace exactly and the iteration stops.
pub const ARNOLDI_BREAKDOWN: Accum = 1e-12;

/// Minimum working-vector slice length per rayon task in the parallel CGS2
/// orthogonalisation. The chunk is sized from `n` so every core gets work
/// (see [`ARNOLDI_TASKS_PER_THREAD`]); this floor keeps a task above the
/// rayon dispatch overhead and clear of cache-line false sharing on `w`.
pub const ARNOLDI_MIN_CHUNK: usize = 256;

/// Target rayon tasks per worker thread for the CGS2 `w -= V·c` update — a
/// few-fold over-subscription so the work-stealing scheduler balances the
/// pass even at modest `n`. A fixed chunk size starved the pass on small
/// and medium meshes (only `⌈n/chunk⌉` tasks regardless of core count);
/// deriving the chunk from `n` and the thread count fixes that.
pub const ARNOLDI_TASKS_PER_THREAD: usize = 4;

// ── Dense matrix exponential (scaling-and-squaring) ───────────────────────

/// Scaling-and-squaring threshold: the matrix is halved until its
/// ∞-norm is at or below this before the Taylor core is summed.
pub const EXPM_SCALE_THRESHOLD: Accum = 0.5;

/// Taylor-series order for the dense `expm` core — the number of terms
/// summed after scaling. About 18 terms reach double precision once the
/// scaled norm is within [`EXPM_SCALE_THRESHOLD`].
pub const EXPM_TAYLOR_TERMS: usize = 18;

// ── Explicit low-storage Runge-Kutta (LSERK4) ─────────────────────────────

/// `a` coefficients of the Carpenter-Kennedy 5-stage 4th-order low-storage
/// Runge-Kutta scheme (LSERK4). The explicit alternative to the Krylov
/// exponential propagator: five matvecs per step, two state registers, and
/// — unlike the unconditionally stable exponential integrator — a CFL step
/// limit set by the spectral radius of the operator. Standard nodal-DG
/// integrator (Hesthaven & Warburton, *Nodal DG Methods*).
pub const LSERK4_A: [Field; 5] = [
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
];

/// `b` coefficients of LSERK4 — the per-stage update weights. Paired with
/// [`LSERK4_A`]; see there.
pub const LSERK4_B: [Field; 5] = [
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
];

/// LSERK4 stage count — five stages reaching fourth order.
pub const LSERK4_STAGES: usize = 5;

// ── Matrix-free operator (apply) ──────────────────────────────────────────

/// Target rayon tasks per worker thread for the matrix-free `apply`'s
/// element loop. Elements are dealt out in this many contiguous runs per
/// thread: enough to let the work-stealing scheduler balance an uneven
/// mesh, few enough that neighbouring threads do not false-share the `dy`
/// cache lines at the run boundaries. A per-element chunk regressed past
/// roughly six cores; a contiguous run per task restores the scaling.
pub const APPLY_TASKS_PER_THREAD: usize = 4;
