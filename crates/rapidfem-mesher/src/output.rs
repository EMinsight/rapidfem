//! Mesher output: tet mesh + tag tables.
//!
//! Designed to map cleanly onto `rapidfem::mesh::Mesh` (the FEM solver's
//! input). Names match where it matters; conversion via `into_rapidfem_mesh`
//! is the only place that depends on the upstream solver crate.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshOutput {
	/// Vertex positions [m], length 3 × n_nodes.
	pub nodes: Vec<f64>,
	/// Tetrahedra, 4 node indices each (length 4 × n_tets).
	pub tets: Vec<u32>,
	/// Surface triangles, 3 node indices each (length 3 × n_tris).
	pub tris: Vec<u32>,
	/// Per-tet physical group tag (length n_tets).
	pub tet_tag: Vec<i32>,
	/// Per-tri physical group tag (length n_tris).
	pub tri_tag: Vec<i32>,
	/// Tag → name lookup. Tag 0 is reserved for "untagged".
	pub tag_names: Vec<(i32, String)>,
	/// Tag → dimension lookup (2 for surface groups, 3 for volume groups).
	pub tag_dim: Vec<(i32, u8)>,
}

impl MeshOutput {
	pub fn n_nodes(&self) -> usize { self.nodes.len() / 3 }
	pub fn n_tets(&self) -> usize { self.tets.len() / 4 }
	pub fn n_tris(&self) -> usize { self.tris.len() / 3 }
}
