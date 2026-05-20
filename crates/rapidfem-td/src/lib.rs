//! rapidfem-td — time-domain DGTD backend.
//!
//! The DG spatial operator, the Krylov/ETD exponential propagator, the
//! state-space export and model-order reduction land here. See
//! `docs/td-backend-plan.md` for the work-package breakdown.

pub mod dg_basis;
pub mod geom_factors;
