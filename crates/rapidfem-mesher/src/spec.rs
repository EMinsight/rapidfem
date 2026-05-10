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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshSpec {
	pub footprint_min: [f64; 2],
	pub footprint_max: [f64; 2],
	pub dielectrics: Vec<DielectricSlab>,
	pub conductors: Vec<ConductorPolygon>,
	pub ports: Vec<VerticalPort>,
	/// Tag for the outer absorbing-boundary faces of the air domain.
	pub abc_tag: String,
	/// Target tet edge length [meters]. The mesher takes it as a hint.
	pub maxh: f64,
}
