//! `MeshSpec` — declarative input the mesher consumes.
//!
//! Captures everything a 2.5D layered EM geometry needs:
//!   - z-stacked dielectric slabs (substrate, oxide, air, …)
//!   - 2D polygons per conductor / via with their host layer
//!   - port plates (vertical or horizontal) with tags
//!   - the simulation footprint (xy bounding box)
//!
//! Coordinates are in meters. The mesher produces tagged tets such that
//! every region in the mesh maps back to a slab or polygon here.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DielectricSlab {
	/// Tag used by the FEM solver to look up material properties.
	pub name: String,
	pub z_bottom: f64,
	pub z_top: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConductorPolygon {
	/// PEC physical-group tag, propagated to all faces of this polygon's
	/// extruded volume.
	pub name: String,
	/// 2D outline in xy [meters], CCW. Must be a simple polygon.
	pub xy: Vec<[f64; 2]>,
	pub z_bottom: f64,
	pub z_top: f64,
}

/// A vertical thin plate spanning two z values. Used for lumped-port
/// excitation surfaces (E ⊥ plate, integrated as voltage gap).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerticalPort {
	pub name: String,
	/// Two endpoints of the plate's xy footprint. The plate is a quad with
	/// these xy endpoints at z_bottom and z_top.
	pub xy_a: [f64; 2],
	pub xy_b: [f64; 2],
	pub z_bottom: f64,
	pub z_top: f64,
}

/// Optional PML wrap around the inner simulation domain. When present, the
/// mesher extends the footprint by `thickness` in xy and adds extra dielectric
/// slabs of `thickness` above and below the existing stack — the FEM solver
/// then applies stretched-coordinate Maxwell to the extra cells. Replaces (or
/// complements) the 1st-order ABC and absorbs near-field reflections from
/// directions where the inner padding is too thin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PmlSpec {
	/// PML layer thickness in meters, applied uniformly to all 6 outer faces
	/// of the inner domain (footprint ± thickness in xy, dielectric stack ±
	/// thickness in z).
	pub thickness: f64,
	/// Number of tet layers across the PML thickness. The stretched-coordinate
	/// material is evaluated at each tet centroid, so multiple layers are
	/// needed to discretize the polynomially-graded σ profile — too few
	/// layers and the inner-face transition reflects strongly. Default 4.
	#[serde(default)]
	pub n_layers: Option<usize>,
	/// Base ε_r used inside the PML cells (before stretching). Default 1.
	#[serde(default)]
	pub er_base: Option<f64>,
	/// Base μ_r used inside the PML cells (before stretching). Default 1.
	#[serde(default)]
	pub ur_base: Option<f64>,
	/// Polynomial grading exponent for the conductivity profile σ ~ uⁿ.
	/// Default 1.5.
	#[serde(default)]
	pub exponent: Option<f64>,
	/// Peak imaginary stretch parameter δ_max at the outer face. Default 8.
	#[serde(default)]
	pub delta_max: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshSpec {
	pub footprint_min: [f64; 2],
	pub footprint_max: [f64; 2],
	pub dielectrics: Vec<DielectricSlab>,
	pub conductors: Vec<ConductorPolygon>,
	pub ports: Vec<VerticalPort>,
	/// Tag for the outer absorbing-boundary faces of the (extended) air
	/// domain. Still applied at the PML's outermost faces to terminate any
	/// residual fields, so it stays mandatory.
	pub abc_tag: String,
	/// Target tet edge length [meters], used by the 2D CDT refinement (so
	/// effectively the in-plane resolution).
	pub maxh: f64,
	/// Optional separate z-direction step. Each dielectric slab gets
	/// subdivided so no z-layer is taller than this. Useful for layered
	/// stacks where in-plane features (trace width, port spacing) want a
	/// big `maxh`, but vertical field gradients (across thin metal/oxide
	/// layers) want fine z-resolution. Defaults to `maxh` when None.
	#[serde(default)]
	pub z_maxh: Option<f64>,
	/// PML wrap. None = legacy ABC-only behaviour.
	#[serde(default)]
	pub pml: Option<PmlSpec>,
}
