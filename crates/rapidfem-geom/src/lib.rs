//! 3D geometry kernel for rapidfem.
//!
//! Wraps `csgrs` (pure-Rust CSG on BSP trees) to provide a tagged geometry
//! API the rapidfem-mesher consumes. The kernel is application-neutral —
//! anything you can express as Boolean ops on solids works (RFIC stacks,
//! microstrip, patch antennas, hollow waveguides, lossy bricks, …).
//!
//! Compiles to wasm32-unknown-unknown so the browser pipeline can construct
//! geometry without a Python/gmsh round-trip.

#![forbid(unsafe_code)]

use csgrs::csg::CSG;
use csgrs::mesh::Mesh;
use csgrs::sketch::Sketch;

/// Tag identifying a region in the geometry. The mesher propagates it to
/// every tet / face it produces from this region. Use it to wire materials
/// (volumes), PEC surfaces, ports, and ABC.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tag(pub String);

impl<S: Into<String>> From<S> for Tag {
	fn from(s: S) -> Self { Tag(s.into()) }
}

/// A solid (volumetric region) in the geometry. Holds a csgrs mesh + a
/// list of tags applied to all of its faces / its volume.
#[derive(Clone)]
pub struct Solid {
	pub(crate) mesh: Mesh<()>,
	pub volume_tag: Option<Tag>,
	pub surface_tag: Option<Tag>,
}

impl Solid {
	/// Axis-aligned box at `(x, y, z)` (lower corner) with given side lengths.
	pub fn box_(x: f64, y: f64, z: f64, dx: f64, dy: f64, dz: f64) -> Self {
		let m = Mesh::cuboid(dx, dy, dz, None).translate(x, y, z);
		Solid { mesh: m, volume_tag: None, surface_tag: None }
	}

	/// Z-axis cylinder of given radius + height, base at z=0 (axis along +z).
	pub fn cylinder(radius: f64, height: f64, segments: usize) -> Self {
		let m = Mesh::cylinder(radius, height, segments, None);
		Solid { mesh: m, volume_tag: None, surface_tag: None }
	}

	/// Polygon (CCW xy vertices) extruded vertically by `height`, starting
	/// at z=0. Translate afterward to position. Must be a simple polygon.
	pub fn extruded_polygon(xy: &[[f64; 2]], height: f64) -> Self {
		let sketch: Sketch<()> = Sketch::polygon(xy, None);
		let m = sketch.extrude(height);
		Solid { mesh: m, volume_tag: None, surface_tag: None }
	}

	pub fn translate(self, dx: f64, dy: f64, dz: f64) -> Self {
		Solid { mesh: self.mesh.translate(dx, dy, dz), ..self }
	}

	pub fn with_volume_tag(mut self, tag: impl Into<Tag>) -> Self {
		self.volume_tag = Some(tag.into());
		self
	}

	pub fn with_surface_tag(mut self, tag: impl Into<Tag>) -> Self {
		self.surface_tag = Some(tag.into());
		self
	}

	pub fn union(self, other: Solid) -> Self {
		Solid {
			mesh: self.mesh.union(&other.mesh),
			volume_tag: self.volume_tag,
			surface_tag: self.surface_tag,
		}
	}

	pub fn difference(self, other: Solid) -> Self {
		Solid {
			mesh: self.mesh.difference(&other.mesh),
			volume_tag: self.volume_tag,
			surface_tag: self.surface_tag,
		}
	}

	pub fn intersection(self, other: Solid) -> Self {
		Solid {
			mesh: self.mesh.intersection(&other.mesh),
			volume_tag: self.volume_tag,
			surface_tag: self.surface_tag,
		}
	}
}

/// A scene = collection of tagged solids, ready for the mesher.
#[derive(Default)]
pub struct Scene {
	pub solids: Vec<Solid>,
}

impl Scene {
	pub fn new() -> Self { Self::default() }
	pub fn add(&mut self, s: Solid) -> &mut Self {
		self.solids.push(s);
		self
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn build_box() {
		let s = Solid::box_(0.0, 0.0, 0.0, 1.0, 1.0, 1.0).with_volume_tag("substrate");
		assert!(s.mesh.polygons.len() > 0);
		assert_eq!(s.volume_tag, Some(Tag("substrate".into())));
	}

	#[test]
	fn boolean_difference() {
		let a = Solid::box_(0.0, 0.0, 0.0, 2.0, 2.0, 2.0);
		let b = Solid::box_(0.5, 0.5, 0.5, 1.0, 1.0, 1.0);
		let c = a.difference(b);
		assert!(c.mesh.polygons.len() > 0);
	}
}
