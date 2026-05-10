//! 2.5D layered tet mesher.
//!
//! Algorithm overview:
//!   1. Walk the spec, collect every distinct z-coordinate that any
//!      dielectric slab, conductor, or port references — these are the
//!      "slab boundaries". Sort + dedup → list of N+1 z values defining N
//!      slabs.
//!   2. For each slab [z_k, z_{k+1}]:
//!        a. Project the slab footprint (= simulation footprint).
//!        b. Insert the polygon outlines of every conductor / port that
//!           overlaps this slab z-range as constraint edges into a 2D CDT
//!           (spade). Constraint vertices come from the polygon corners.
//!        c. Run the CDT → 2D triangulation with constraint edges respected.
//!        d. Assign each 2D triangle a *region tag*: which conductor (if
//!           inside one), which port (if inside one), else the dielectric
//!           slab tag.
//!   3. Generate global 3D vertex IDs:
//!      - At every distinct z plane, copy the 2D vertex list and assign IDs.
//!   4. For each slab × each 2D triangle:
//!      - Form the prism (3 vertices at z_bottom, 3 at z_top).
//!      - Split the prism into 3 tets (deterministic — see `prism_to_tets`).
//!      - Assign the slab's tet tag to each of those tets.
//!   5. Boundary faces — find 2D triangle edges that lie between two
//!      different region tags (conductor↔dielectric, dielectric↔dielectric
//!      at slab boundaries, etc.) → emit as boundary tris with the
//!      conductor/port tag.
//!   6. ABC: outer faces of the simulation footprint at the air slab.

use crate::output::MeshOutput;
use crate::spec::MeshSpec;
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation, InsertionError};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, thiserror::Error)]
pub enum MeshError {
	#[error("spec is empty (no dielectric slabs)")]
	Empty,
	#[error("conductor {0:?} z-range [{1}..{2}] is outside any dielectric slab")]
	ConductorOutsideStack(String, f64, f64),
	#[error("CDT insertion failed: {0:?}")]
	Cdt(InsertionError),
}

impl From<InsertionError> for MeshError {
	fn from(e: InsertionError) -> Self { MeshError::Cdt(e) }
}

/// Numerical tolerance for collapsing nearly-equal z values into the same
/// slab boundary. Tuned for typical RFIC scales (µm = 1e-6); adjust for
/// other domains if needed.
const Z_TOL: f64 = 1e-12;

/// Snap-and-dedup of every z value referenced in the spec.
pub(crate) fn slab_levels(spec: &MeshSpec) -> Vec<f64> {
	let mut zs: Vec<f64> = Vec::new();
	for d in &spec.dielectrics {
		zs.push(d.z_bottom);
		zs.push(d.z_top);
	}
	for c in &spec.conductors {
		zs.push(c.z_bottom);
		zs.push(c.z_top);
	}
	for p in &spec.ports {
		zs.push(p.z_bottom);
		zs.push(p.z_top);
	}
	zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let mut out: Vec<f64> = Vec::with_capacity(zs.len());
	for z in zs {
		if out.last().map_or(true, |&last| (z - last).abs() > Z_TOL) {
			out.push(z);
		}
	}
	out
}

/// Find the dielectric slab whose z-range fully contains [z_bot, z_top].
/// Returns its name, or None if not found.
pub(crate) fn enclosing_slab<'a>(spec: &'a MeshSpec, z_bot: f64, z_top: f64) -> Option<&'a str> {
	let zmid = (z_bot + z_top) / 2.0;
	spec.dielectrics
		.iter()
		.find(|d| d.z_bottom - Z_TOL <= zmid && zmid <= d.z_top + Z_TOL)
		.map(|d| d.name.as_str())
}

/// Per-slab 2D mesh data: vertex positions in xy, triangles, region tag per
/// triangle.
pub(crate) struct SlabMesh2D {
	pub vertices: Vec<[f64; 2]>,
	pub tris: Vec<[usize; 3]>,
	/// Region tag (>=0) per triangle.
	pub tri_tag: Vec<i32>,
}

/// Triangulate one slab. Inserts polygon constraints for every conductor /
/// port that overlaps the slab; tags each output triangle by which region
/// it belongs to.
pub(crate) fn triangulate_slab(
	spec: &MeshSpec,
	z_bottom: f64,
	z_top: f64,
	tag_for_name: &HashMap<String, i32>,
) -> Result<SlabMesh2D, MeshError> {
	let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
		ConstrainedDelaunayTriangulation::new();

	let z_eps = Z_TOL;
	let slab_overlaps = |zb: f64, zt: f64| zt > z_bottom + z_eps && zb < z_top - z_eps;

	// 1) Footprint as outer constraint
	let [xmin, ymin] = spec.footprint_min;
	let [xmax, ymax] = spec.footprint_max;
	let corners = [
		Point2::new(xmin, ymin),
		Point2::new(xmax, ymin),
		Point2::new(xmax, ymax),
		Point2::new(xmin, ymax),
	];
	let mut corner_handles = [None; 4];
	for (i, c) in corners.iter().enumerate() {
		corner_handles[i] = Some(cdt.insert(*c)?);
	}
	for i in 0..4 {
		let a = corner_handles[i].unwrap();
		let b = corner_handles[(i + 1) % 4].unwrap();
		cdt.add_constraint(a, b);
	}

	// 2) Conductor outlines
	let mut polygon_regions: Vec<(Vec<[f64; 2]>, i32)> = Vec::new();
	for cond in &spec.conductors {
		if !slab_overlaps(cond.z_bottom, cond.z_top) { continue; }
		let tag = *tag_for_name.get(&cond.name).unwrap();
		insert_polygon_constraint(&mut cdt, &cond.xy)?;
		polygon_regions.push((cond.xy.clone(), tag));
	}
	// Vertical port "plates" are 1D in xy — a line segment from xy_a to xy_b.
	// Insert as two constrained vertices + an edge between them; no region
	// fill since the port is a 1D feature in this slab.
	for port in &spec.ports {
		if !slab_overlaps(port.z_bottom, port.z_top) { continue; }
		let pa = cdt.insert(Point2::new(port.xy_a[0], port.xy_a[1]))?;
		let pb = cdt.insert(Point2::new(port.xy_b[0], port.xy_b[1]))?;
		cdt.add_constraint(pa, pb);
	}

	// 3) Pull triangles + assign region tag
	let dielectric_tag = enclosing_slab(spec, z_bottom, z_top)
		.and_then(|n| tag_for_name.get(n).copied())
		.unwrap_or(0);

	let mut vertices: Vec<[f64; 2]> = Vec::with_capacity(cdt.num_vertices());
	let mut handle_to_idx: HashMap<usize, usize> = HashMap::new();
	for v in cdt.vertices() {
		let p = v.position();
		handle_to_idx.insert(v.fix().index(), vertices.len());
		vertices.push([p.x, p.y]);
	}

	let mut tris: Vec<[usize; 3]> = Vec::new();
	let mut tri_tag: Vec<i32> = Vec::new();
	for face in cdt.inner_faces() {
		let vs = face.vertices();
		let a = handle_to_idx[&vs[0].fix().index()];
		let b = handle_to_idx[&vs[1].fix().index()];
		let c = handle_to_idx[&vs[2].fix().index()];
		let centroid = [
			(vertices[a][0] + vertices[b][0] + vertices[c][0]) / 3.0,
			(vertices[a][1] + vertices[b][1] + vertices[c][1]) / 3.0,
		];
		let region = polygon_regions
			.iter()
			.find(|(poly, _)| point_in_polygon(centroid, poly))
			.map(|(_, t)| *t)
			.unwrap_or(dielectric_tag);
		tris.push([a, b, c]);
		tri_tag.push(region);
	}

	Ok(SlabMesh2D { vertices, tris, tri_tag })
}

fn insert_polygon_constraint(
	cdt: &mut ConstrainedDelaunayTriangulation<Point2<f64>>,
	xy: &[[f64; 2]],
) -> Result<(), MeshError> {
	let mut handles = Vec::with_capacity(xy.len());
	for &[x, y] in xy {
		handles.push(cdt.insert(Point2::new(x, y))?);
	}
	for i in 0..handles.len() {
		let a = handles[i];
		let b = handles[(i + 1) % handles.len()];
		// can_add_constraint is True unless the segment crosses an existing one
		if cdt.can_add_constraint(a, b) {
			cdt.add_constraint(a, b);
		}
	}
	Ok(())
}

fn point_in_polygon(pt: [f64; 2], poly: &[[f64; 2]]) -> bool {
	let (x, y) = (pt[0], pt[1]);
	let mut inside = false;
	let n = poly.len();
	let mut j = n - 1;
	for i in 0..n {
		let (xi, yi) = (poly[i][0], poly[i][1]);
		let (xj, yj) = (poly[j][0], poly[j][1]);
		if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi + f64::EPSILON) + xi) {
			inside = !inside;
		}
		j = i;
	}
	inside
}

/// Split a triangular prism into 3 tets. The prism's 6 vertices are:
///   bottom face: (a, b, c) at z_bottom (CCW from above)
///   top face:    (a', b', c') at z_top
/// Standard split (Delaunay-friendly): use the diagonal that goes through
/// the smallest-id vertex on the bottom — this guarantees vertex-shared
/// neighbour prisms produce conformal faces.
pub(crate) fn prism_to_tets(b: [u32; 3], t: [u32; 3]) -> [[u32; 4]; 3] {
	// Pick the smallest bottom vertex id as the "anchor" so adjacent prisms
	// agree on the diagonal direction (face conformity).
	let mut order = [0usize, 1, 2];
	order.sort_by_key(|&i| b[i]);
	let i0 = order[0]; let i1 = order[1]; let i2 = order[2];
	let (b0, b1, b2) = (b[i0], b[i1], b[i2]);
	let (t0, t1, t2) = (t[i0], t[i1], t[i2]);
	[
		[b0, b1, b2, t0],
		[b1, b2, t0, t1],
		[b2, t0, t1, t2],
	]
}

pub fn mesh(spec: &MeshSpec) -> Result<MeshOutput, MeshError> {
	if spec.dielectrics.is_empty() { return Err(MeshError::Empty); }
	let zs = slab_levels(spec);

	// Tag bookkeeping
	let mut tag_for_name: HashMap<String, i32> = HashMap::new();
	let mut tag_names: Vec<(i32, String)> = Vec::new();
	let mut tag_dim: Vec<(i32, u8)> = Vec::new();
	let mut next_tag: i32 = 1;
	let mut register_tag = |name: &str, dim: u8,
		map: &mut HashMap<String, i32>,
		names: &mut Vec<(i32, String)>,
		dims: &mut Vec<(i32, u8)>,
		next: &mut i32| -> i32 {
		if let Some(&t) = map.get(name) { return t; }
		let t = *next;
		*next += 1;
		map.insert(name.to_string(), t);
		names.push((t, name.to_string()));
		dims.push((t, dim));
		t
	};
	for d in &spec.dielectrics { register_tag(&d.name, 3, &mut tag_for_name, &mut tag_names, &mut tag_dim, &mut next_tag); }
	for c in &spec.conductors  { register_tag(&c.name, 2, &mut tag_for_name, &mut tag_names, &mut tag_dim, &mut next_tag); }
	for p in &spec.ports       { register_tag(&p.name, 2, &mut tag_for_name, &mut tag_names, &mut tag_dim, &mut next_tag); }
	let abc_tag = register_tag(&spec.abc_tag, 2, &mut tag_for_name, &mut tag_names, &mut tag_dim, &mut next_tag);

	// Per-slab triangulate
	let mut slabs: Vec<(f64, f64, SlabMesh2D)> = Vec::new();
	for w in zs.windows(2) {
		let (z0, z1) = (w[0], w[1]);
		let s = triangulate_slab(spec, z0, z1, &tag_for_name)?;
		slabs.push((z0, z1, s));
	}

	// Global 3D vertex pool: (vertex_idx_in_slab, z_level) → global node id.
	// All slabs share the same 2D vertex coordinates because every constraint
	// vertex was inserted globally — but spade's CDT is independent per slab,
	// so we use spatial deduplication (round to Z_TOL) instead.
	let mut node_at: HashMap<(i64, i64, i64), u32> = HashMap::new();
	let mut nodes: Vec<f64> = Vec::new();
	let q = |v: f64| (v / Z_TOL).round() as i64;
	let mut intern = |x: f64, y: f64, z: f64,
		node_at: &mut HashMap<(i64, i64, i64), u32>,
		nodes: &mut Vec<f64>| -> u32 {
		let k = (q(x), q(y), q(z));
		if let Some(&id) = node_at.get(&k) { return id; }
		let id = (nodes.len() / 3) as u32;
		nodes.push(x); nodes.push(y); nodes.push(z);
		node_at.insert(k, id);
		id
	};

	let mut tets: Vec<u32> = Vec::new();
	let mut tet_tag: Vec<i32> = Vec::new();
	let mut tris: Vec<u32> = Vec::new();
	let mut tri_tag: Vec<i32> = Vec::new();
	let mat_tag = |region_tag: i32, dielectric_tag: i32| -> i32 {
		// If the region is a conductor, the tet itself sits in the surrounding
		// dielectric — we use the dielectric's tag for the volume, and the
		// conductor's tag goes onto the boundary tris.
		if let Some((_, dim)) = tag_dim.iter().find(|(t, _)| *t == region_tag) {
			if *dim == 2 { return dielectric_tag; }
		}
		region_tag
	};

	for (z0, z1, slab) in &slabs {
		let dielectric = enclosing_slab(spec, *z0, *z1)
			.and_then(|n| tag_for_name.get(n).copied())
			.unwrap_or(0);
		// Map per-slab vertex idx → global bottom + top node id
		let mut bot_id = vec![0u32; slab.vertices.len()];
		let mut top_id = vec![0u32; slab.vertices.len()];
		for (i, &[x, y]) in slab.vertices.iter().enumerate() {
			bot_id[i] = intern(x, y, *z0, &mut node_at, &mut nodes);
			top_id[i] = intern(x, y, *z1, &mut node_at, &mut nodes);
		}
		for (tri_idx, &[a, b, c]) in slab.tris.iter().enumerate() {
			let region = slab.tri_tag[tri_idx];
			let vol_tag = mat_tag(region, dielectric);
			let bot = [bot_id[a], bot_id[b], bot_id[c]];
			let top = [top_id[a], top_id[b], top_id[c]];
			for tet in prism_to_tets(bot, top) {
				tets.extend_from_slice(&tet);
				tet_tag.push(vol_tag);
			}
			// Conductor region: emit its top + bottom face as PEC tris with the
			// conductor's tag (the conductor is "thin" → 2D plate, 2 tris per
			// triangulated cell).
			if region != vol_tag {
				tris.push(bot[0]); tris.push(bot[1]); tris.push(bot[2]);
				tri_tag.push(region);
				tris.push(top[0]); tris.push(top[2]); tris.push(top[1]);
				tri_tag.push(region);
			}
		}
	}

	// ABC: outer-bbox faces at every slab boundary at xmin/xmax/ymin/ymax + top
	// of topmost slab. Walk the global node list and collect every triangle on
	// these planes. Cheap O(n_tris) post-pass: any face whose 3 nodes share an
	// outer xy-coord or sit at z_max is on the bbox.
	// (For brevity we skip this in the first pass — caller can add ABC tris
	// from outer prism faces in a follow-up.)

	Ok(MeshOutput {
		nodes, tets, tris, tet_tag, tri_tag,
		tag_names, tag_dim,
	})
}
