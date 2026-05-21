//! Centralised numerical constants for the time-domain backend.
//!
//! Every tolerance and algorithmic threshold lives here — never inline,
//! never hardcoded at a call site. A solver constant that needs tuning is
//! tuned in one place.

// ── Krylov / ETD exponential propagator ───────────────────────────────────

/// Relative a-posteriori error tolerance for the Krylov exponential
/// propagator. After each Arnoldi vector the step's error estimate is
/// checked against this; the subspace stops growing once it converges, so
/// an easy step costs far fewer matvecs than the `krylov_dim` cap.
pub const KRYLOV_TOL: f64 = 1e-10;

/// Arnoldi lucky-breakdown threshold. When the next basis vector's norm
/// falls below this the Krylov subspace already spans the relevant
/// invariant subspace exactly and the iteration stops.
pub const ARNOLDI_BREAKDOWN: f64 = 1e-12;

/// Working-vector slice length per rayon task in the parallel CGS2
/// orthogonalisation — the granularity at which `w -= V·c` is fanned out.
pub const ARNOLDI_CHUNK: usize = 8192;

// ── Dense matrix exponential (scaling-and-squaring) ───────────────────────

/// Scaling-and-squaring threshold: the matrix is halved until its
/// ∞-norm is at or below this before the Taylor core is summed.
pub const EXPM_SCALE_THRESHOLD: f64 = 0.5;

/// Taylor-series order for the dense `expm` core — the number of terms
/// summed after scaling. About 18 terms reach double precision once the
/// scaled norm is within [`EXPM_SCALE_THRESHOLD`].
pub const EXPM_TAYLOR_TERMS: usize = 18;
