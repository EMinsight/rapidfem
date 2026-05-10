//! 2.5D tetrahedral mesher for layered EM geometries.
//!
//! Pure Rust, WASM-friendly, no gmsh / OCC / Python in the loop.
//!
//! # Pipeline
//! ```text
//!   MeshSpec (slabs + conductor polygons + ports + tags)
//!       │
//!       ▼
//!   slab_levels  ─►  per-slab triangulate (spade CDT)
//!       │
//!       ▼
//!   prism extrusion → 3 tets per prism  (deterministic split)
//!       │
//!       ▼
//!   tag propagation (volume / surface)
//!       │
//!       ▼
//!   MeshOutput (nodes, tets, tris, per-elem tags) — feeds the FEM solver
//! ```
//!
//! # Scope (initial)
//! Handles 2.5D layered geometries: stacked z-slabs with axis-aligned
//! polygon footprints. RFIC, microstrip, patch antennas, hollow waveguides
//! — anything where conductors can be expressed as 2D polygons at fixed z.
//!
//! Tilted / curved conductors that aren't 2.5D are out of scope here;
//! a future general-3D fallback (csgrs Brep + Delaunay-refinement tet
//! generation) lives separately.

#![forbid(unsafe_code)]

pub mod spec;
pub mod output;
pub mod mesher;

pub use spec::*;
pub use output::*;
pub use mesher::{mesh, MeshError};
