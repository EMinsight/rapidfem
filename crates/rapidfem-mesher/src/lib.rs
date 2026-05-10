//! 2.5D tetrahedral mesher for layered EM geometries.
//!
//! Takes a `rapidfem_geom::Scene` of tagged solids and produces a tagged
//! tetrahedral mesh suitable for the rapidfem FEM solver. Pure Rust, no
//! gmsh / OCC / Python in the loop, WASM-friendly.
//!
//! # Approach (initial)
//! - Slice the scene's z-extent at every layer boundary present in any
//!   solid (substrate top, metal bottom/top, oxide top, …).
//! - Per slab: union the xy-projections of all solids that occupy that
//!   slab, run constrained Delaunay 2D (spade) with polygon edges as
//!   constraints, get a triangulated 2D domain tagged by region.
//! - Per slab triangle × slab thickness = prism → split into 3 tets
//!   deterministically (one of the 3 valid prism→tet decompositions).
//! - Carry tags from solids to tets (volume tags) and to faces on
//!   solid boundaries (surface tags for PEC / port / ABC).
//!
//! Inputs that aren't 2.5D (tilted vias, curved conductors) get a clear
//! error — to be lifted later by adding general-3D fallback.

#![forbid(unsafe_code)]
