//! rapidfem-td — time-domain DGTD backend.
//!
//! The DG spatial operator, the Krylov/ETD exponential propagator, the
//! state-space export and model-order reduction land here. See
//! `docs/td-backend-plan.md` for the work-package breakdown.

pub mod absorber;
pub mod constants;
pub mod dg_basis;
pub mod dispersive;
pub mod explicit;
pub mod geom_factors;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod mesh_gen;
pub mod mor;
pub mod propagator;
pub mod rhs;
pub mod waveguide;
