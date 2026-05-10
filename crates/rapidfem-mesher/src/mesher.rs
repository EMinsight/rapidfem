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
use crate::spec::{DielectricSlab, MeshSpec};
use spade::{
	AngleLimit, ConstrainedDelaunayTriangulation, InsertionError, Point2,
	RefinementParameters, Triangulation,
};
use std::collections::HashMap;

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

/// Snap-and-dedup of every z value referenced in the spec, then sub-divide
/// any slab whose height exceeds `z_max_step` so vertical tets aren't
/// dramatically taller than the in-plane refinement scale.
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
	let mut snapped: Vec<f64> = Vec::with_capacity(zs.len());
	for z in zs {
		if snapped.last().map_or(true, |&last| (z - last).abs() > Z_TOL) {
			snapped.push(z);
		}
	}
	// Subdivide thick slabs so no z-layer is taller than `z_maxh` (falls back
	// to `maxh` if not set). Thin slabs (e.g. 1µm metal layers) stay
	// single-layer when their height is already below the step.
	let z_step_raw = spec.z_maxh.unwrap_or(spec.maxh);
	let z_step = if z_step_raw.is_finite() && z_step_raw > 0.0 { z_step_raw } else { f64::INFINITY };
	let mut out: Vec<f64> = Vec::with_capacity(snapped.len() * 2);
	out.push(snapped[0]);
	for w in snapped.windows(2) {
		let (a, b) = (w[0], w[1]);
		let h = b - a;
		let n = ((h / z_step).ceil() as usize).max(1);
		for k in 1..=n {
			out.push(a + h * (k as f64) / (n as f64));
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

/// Triangulate one slab. Inserts polygon constraints for every conductor /
/// port that overlaps the slab; tags each output triangle by which region
/// it belongs to.
/// Build one shared 2D mesh from the union of every conductor + port outline
/// across all z-layers, refine it once, then return vertices + tris (no tags).
/// Slab-specific region tags come from `slab_region_tags()` on the same tris.
pub(crate) fn triangulate_global(
	spec: &MeshSpec,
	maxh: f64,
	pml_layer_lines: &[((f64, f64), (f64, f64))],
) -> Result<(Vec<[f64; 2]>, Vec<[usize; 3]>), MeshError> {
	let mut cdt: ConstrainedDelaunayTriangulation<Point2<f64>> =
		ConstrainedDelaunayTriangulation::new();

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

	for cond in &spec.conductors {
		insert_polygon_constraint(&mut cdt, &cond.xy)?;
	}
	for port in &spec.ports {
		let pa = cdt.insert(Point2::new(port.xy_a[0], port.xy_a[1]))?;
		let pb = cdt.insert(Point2::new(port.xy_b[0], port.xy_b[1]))?;
		cdt.add_constraint(pa, pb);
	}

	// Layer lines (used by PML to force the CDT to discretize each PML wrap
	// into n_layers strips). These are simple line constraints; refinement
	// then propagates inside the strips.
	for &((ax, ay), (bx, by)) in pml_layer_lines {
		let pa = cdt.insert(Point2::new(ax, ay))?;
		let pb = cdt.insert(Point2::new(bx, by))?;
		if cdt.can_add_constraint(pa, pb) {
			cdt.add_constraint(pa, pb);
		}
	}

	if maxh.is_finite() && maxh > 0.0 {
		let max_area = 0.5 * maxh * maxh;
		let params = RefinementParameters::<f64>::new()
			.with_angle_limit(AngleLimit::from_deg(20.0))
			.with_max_allowed_area(max_area)
			.with_max_additional_vertices(20_000);
		cdt.refine(params);
	}

	let mut vertices: Vec<[f64; 2]> = Vec::with_capacity(cdt.num_vertices());
	let mut handle_to_idx: HashMap<usize, usize> = HashMap::new();
	for v in cdt.vertices() {
		let p = v.position();
		handle_to_idx.insert(v.fix().index(), vertices.len());
		vertices.push([p.x, p.y]);
	}
	let mut tris: Vec<[usize; 3]> = Vec::new();
	for face in cdt.inner_faces() {
		let vs = face.vertices();
		tris.push([
			handle_to_idx[&vs[0].fix().index()],
			handle_to_idx[&vs[1].fix().index()],
			handle_to_idx[&vs[2].fix().index()],
		]);
	}
	Ok((vertices, tris))
}

/// For a given slab, classify each global tri as conductor / port / dielectric
/// based on whether the conductor or port overlaps this slab's z-range.
pub(crate) fn slab_region_tags(
	spec: &MeshSpec,
	z_bottom: f64,
	z_top: f64,
	vertices: &[[f64; 2]],
	tris: &[[usize; 3]],
	tag_for_name: &HashMap<String, i32>,
) -> Vec<i32> {
	let z_eps = Z_TOL;
	let slab_overlaps = |zb: f64, zt: f64| zt > z_bottom + z_eps && zb < z_top - z_eps;
	let polygon_regions: Vec<(&[[f64; 2]], i32)> = spec.conductors.iter()
		.filter(|c| slab_overlaps(c.z_bottom, c.z_top))
		.map(|c| (c.xy.as_slice(), *tag_for_name.get(&c.name).unwrap()))
		.collect();
	let dielectric_tag = enclosing_slab(spec, z_bottom, z_top)
		.and_then(|n| tag_for_name.get(n).copied())
		.unwrap_or(0);
	let mut out = Vec::with_capacity(tris.len());
	for &[a, b, c] in tris {
		let centroid = [
			(vertices[a][0] + vertices[b][0] + vertices[c][0]) / 3.0,
			(vertices[a][1] + vertices[b][1] + vertices[c][1]) / 3.0,
		];
		let region = polygon_regions
			.iter()
			.find(|(poly, _)| point_in_polygon(centroid, *poly))
			.map(|(_, t)| *t)
			.unwrap_or(dielectric_tag);
		out.push(region);
	}
	out
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

	// PML wrap: build an "effective spec" with the footprint expanded by
	// pml.thickness in xy and two extra dielectric slabs (pml.thickness each)
	// above/below the original z stack. Tets that fall in those extension
	// regions get re-tagged below as pml_{x,y,z}{min,max}. The inner-domain
	// footprint corners are remembered so we know where the original boundary
	// was (= each PML's inner_face).
	let inner_xmin = spec.footprint_min[0];
	let inner_xmax = spec.footprint_max[0];
	let inner_ymin = spec.footprint_min[1];
	let inner_ymax = spec.footprint_max[1];
	let inner_zmin = spec.dielectrics.iter().map(|d| d.z_bottom).fold(f64::INFINITY, f64::min);
	let inner_zmax = spec.dielectrics.iter().map(|d| d.z_top).fold(f64::NEG_INFINITY, f64::max);

	// Default 1 layer — minimum that fits the WASM heap budget. The PML's
	// strong δ_max=8 still absorbs effectively even with a single cell;
	// you'd want 2-4 layers for production accuracy on bigger machines.
	let pml_n_layers: usize = spec.pml.as_ref().and_then(|p| p.n_layers).unwrap_or(1).max(1);
	let effective_spec = if let Some(p) = &spec.pml {
		let mut s = spec.clone();
		let t = p.thickness;
		s.footprint_min = [inner_xmin - t, inner_ymin - t];
		s.footprint_max = [inner_xmax + t, inner_ymax + t];
		// Sandwich PML slabs on top and bottom of the dielectric stack. We
		// emit `n_layers` sub-slabs of thickness t/n_layers each so the
		// stretched-coordinate σ(u) = u^n profile gets discretized with
		// multiple cells across the PML thickness — one cell would only see
		// the centroid value and reflect badly at the inner face.
		let dz = t / (pml_n_layers as f64);
		// Below (insert at position 0 to keep z-ordering).
		for k in (0..pml_n_layers).rev() {
			let z0 = inner_zmin - t + (k as f64) * dz;
			let z1 = inner_zmin - t + ((k + 1) as f64) * dz;
			s.dielectrics.insert(0, DielectricSlab {
				name: "_pml_z_below".to_string(), z_bottom: z0, z_top: z1,
			});
		}
		// Above.
		for k in 0..pml_n_layers {
			let z0 = inner_zmax + (k as f64) * dz;
			let z1 = inner_zmax + ((k + 1) as f64) * dz;
			s.dielectrics.push(DielectricSlab {
				name: "_pml_z_above".to_string(), z_bottom: z0, z_top: z1,
			});
		}
		s
	} else {
		spec.clone()
	};
	let spec_eff = &effective_spec;
	let zs = slab_levels(spec_eff);

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
	for d in &spec_eff.dielectrics { register_tag(&d.name, 3, &mut tag_for_name, &mut tag_names, &mut tag_dim, &mut next_tag); }
	// Conductors are PEC surfaces (interior of conductor is excluded from FEM
	// volume), so we register them as dim=2.
	for c in &spec_eff.conductors { register_tag(&c.name, 2, &mut tag_for_name, &mut tag_names, &mut tag_dim, &mut next_tag); }
	for p in &spec_eff.ports      { register_tag(&p.name, 2, &mut tag_for_name, &mut tag_names, &mut tag_dim, &mut next_tag); }
	let abc_tag = register_tag(&spec_eff.abc_tag, 2, &mut tag_for_name, &mut tag_names, &mut tag_dim, &mut next_tag);
	// PML region tags (dim=3) — one per absorbing direction. Only registered
	// when `spec.pml` is present; we allocate them up-front so the override
	// below has the integers ready.
	let pml_tags: Option<[(i32, [f64; 3], f64); 6]> = spec.pml.as_ref().map(|_| {
		let names = [
			("_pml_xmin", [-1.0, 0.0, 0.0], inner_xmin),
			("_pml_xmax", [ 1.0, 0.0, 0.0], inner_xmax),
			("_pml_ymin", [ 0.0,-1.0, 0.0], inner_ymin),
			("_pml_ymax", [ 0.0, 1.0, 0.0], inner_ymax),
			("_pml_zmin", [ 0.0, 0.0,-1.0], inner_zmin),
			("_pml_zmax", [ 0.0, 0.0, 1.0], inner_zmax),
		];
		let mut out: [(i32, [f64; 3], f64); 6] = Default::default();
		for (i, (name, dir, inner)) in names.iter().enumerate() {
			let t = register_tag(name, 3, &mut tag_for_name, &mut tag_names, &mut tag_dim, &mut next_tag);
			out[i] = (t, *dir, *inner);
		}
		out
	});

	// One shared 2D mesh used by every slab — guarantees vertex-conformity
	// across slab boundaries (essential for prism extrusion to give a manifold
	// 3D mesh). Refinement runs once.
	//
	// PML wrap: insert layer-parallel constraint lines on each of the 4 lateral
	// sides. spade is constraint-respecting, so the CDT is forced to slice
	// each PML wrap into `n_layers` strips of equal thickness (matching the
	// z-PML stratification). Without this the lateral PML would be a single
	// tet thick — same reflection-at-inner-face problem we just fixed for z.
	let mut pml_layer_lines: Vec<((f64, f64), (f64, f64))> = Vec::new();
	if spec.pml.is_some() && pml_n_layers > 1 {
		let t = spec.pml.as_ref().unwrap().thickness;
		let dx = t / (pml_n_layers as f64);
		// Vertical lines (xmin / xmax PML)
		for k in 1..pml_n_layers {
			let xl = inner_xmin - (k as f64) * dx;
			let xr = inner_xmax + (k as f64) * dx;
			let [_, fyminx] = spec_eff.footprint_min;
			let [_, fymaxx] = spec_eff.footprint_max;
			pml_layer_lines.push(((xl, fyminx), (xl, fymaxx)));
			pml_layer_lines.push(((xr, fyminx), (xr, fymaxx)));
		}
		// Horizontal lines (ymin / ymax PML)
		for k in 1..pml_n_layers {
			let yl = inner_ymin - (k as f64) * dx;
			let yr = inner_ymax + (k as f64) * dx;
			let [fxminy, _] = spec_eff.footprint_min;
			let [fxmaxy, _] = spec_eff.footprint_max;
			pml_layer_lines.push(((fxminy, yl), (fxmaxy, yl)));
			pml_layer_lines.push(((fxminy, yr), (fxmaxy, yr)));
		}
	}
	let (vertices2d, tris2d) = triangulate_global(spec_eff, spec_eff.maxh, &pml_layer_lines)?;

	// Per-slab region tags, all referring to the SAME global 2D tris.
	let mut slab_tags: Vec<(f64, f64, Vec<i32>)> = Vec::new();
	for w in zs.windows(2) {
		let (z0, z1) = (w[0], w[1]);
		let tags = slab_region_tags(spec_eff, z0, z1, &vertices2d, &tris2d, &tag_for_name);
		slab_tags.push((z0, z1, tags));
	}

	// Helper: a tri is in a "conductor region" if its tag is dim=2 and the
	// name maps to a conductor (vs port — which we leave as dielectric, since
	// ports are 1D edge constraints, not 2D area regions).
	let conductor_tags: std::collections::HashSet<i32> = spec_eff.conductors.iter()
		.map(|c| *tag_for_name.get(&c.name).unwrap()).collect();
	let is_conductor = |tag: i32| conductor_tags.contains(&tag);

	// Global 3D vertex pool. All vertices come from the shared 2D mesh; we
	// emit them at every distinct z plane.
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

	// Collect cells we DIDN'T tetrahedralize — those are conductor interiors.
	// Their boundary faces (top, bottom, 3 vertical sidewalls) all need PEC
	// tris with the conductor's tag. We emit every candidate face for them
	// in both diagonal triangulations; the FEM-side `inv_tris` lookup keeps
	// only the ones that match an actual neighbour-tet face, so internal
	// (conductor↔conductor) faces drop out automatically.
	let emit_quad_both_diags_local = |a0: u32, b0: u32, b1: u32, a1: u32, tag: i32,
		tris: &mut Vec<u32>, tri_tag: &mut Vec<i32>| {
		tris.push(a0); tris.push(b0); tris.push(b1); tri_tag.push(tag);
		tris.push(a0); tris.push(b1); tris.push(a1); tri_tag.push(tag);
		tris.push(a0); tris.push(b0); tris.push(a1); tri_tag.push(tag);
		tris.push(b0); tris.push(b1); tris.push(a1); tri_tag.push(tag);
	};

	for (z0, z1, tags) in &slab_tags {
		let dielectric = enclosing_slab(spec_eff, *z0, *z1)
			.and_then(|n| tag_for_name.get(n).copied())
			.unwrap_or(0);
		// Per-slab: every shared 2D vertex gets a global node at z0 and z1.
		let mut bot_id = vec![0u32; vertices2d.len()];
		let mut top_id = vec![0u32; vertices2d.len()];
		for (i, &[x, y]) in vertices2d.iter().enumerate() {
			bot_id[i] = intern(x, y, *z0, &mut node_at, &mut nodes);
			top_id[i] = intern(x, y, *z1, &mut node_at, &mut nodes);
		}
		for (tri_idx, &[a, b, c]) in tris2d.iter().enumerate() {
			let region = tags[tri_idx];
			let bot = [bot_id[a], bot_id[b], bot_id[c]];
			let top = [top_id[a], top_id[b], top_id[c]];

			if is_conductor(region) {
				// 3D-extruded conductor: skip tets; emit all 5 candidate
				// boundary faces with the conductor's tag.
				let cap_tag = region;
				// Top + bottom cap (already a tet face in neighbouring slab/cell)
				tris.push(bot[0]); tris.push(bot[1]); tris.push(bot[2]); tri_tag.push(cap_tag);
				tris.push(bot[0]); tris.push(bot[2]); tris.push(bot[1]); tri_tag.push(cap_tag);
				tris.push(top[0]); tris.push(top[1]); tris.push(top[2]); tri_tag.push(cap_tag);
				tris.push(top[0]); tris.push(top[2]); tris.push(top[1]); tri_tag.push(cap_tag);
				// 3 vertical sidewalls — emit both diagonals each.
				for (i, j) in [(0, 1), (1, 2), (2, 0)] {
					emit_quad_both_diags_local(bot[i], bot[j], top[j], top[i], cap_tag, &mut tris, &mut tri_tag);
				}
				continue;
			}

			for tet in prism_to_tets(bot, top) {
				tets.extend_from_slice(&tet);
				tet_tag.push(dielectric);
			}
		}
	}

	// Quad-on-tet-boundary emitter. The prism→3-tets split picks a specific
	// diagonal we don't easily replicate from outside, so we emit BOTH possible
	// triangulations of the quad — the FEM-side `inv_tris` lookup keeps only the
	// 2 that actually match a tet face, the other 2 are silently dropped.
	let emit_quad_both_diags = |a0: u32, b0: u32, b1: u32, a1: u32, tag: i32,
		tris: &mut Vec<u32>, tri_tag: &mut Vec<i32>| {
		// Diagonal a0—b1
		tris.push(a0); tris.push(b0); tris.push(b1); tri_tag.push(tag);
		tris.push(a0); tris.push(b1); tris.push(a1); tri_tag.push(tag);
		// Diagonal b0—a1
		tris.push(a0); tris.push(b0); tris.push(a1); tri_tag.push(tag);
		tris.push(b0); tris.push(b1); tris.push(a1); tri_tag.push(tag);
	};

	// Build the set of all edges in the global 2D mesh — refinement may have
	// subdivided port lines and footprint walls, so we walk the actual CDT
	// edges instead of assuming end-to-end emit.
	let mut edges2d: std::collections::HashSet<(usize, usize)> =
		std::collections::HashSet::with_capacity(tris2d.len() * 3);
	for &[a, b, c] in &tris2d {
		for (p, q) in [(a, b), (b, c), (c, a)] {
			let key = if p < q { (p, q) } else { (q, p) };
			edges2d.insert(key);
		}
	}
	// Test: is point (px,py) on segment (xa,ya)→(xb,yb)?
	let on_segment = |px: f64, py: f64, xa: f64, ya: f64, xb: f64, yb: f64| -> bool {
		let dx = xb - xa; let dy = yb - ya;
		let len2 = dx * dx + dy * dy;
		if len2 < 1e-30 { return false; }
		// Parameter t along the segment
		let t = ((px - xa) * dx + (py - ya) * dy) / len2;
		if t < -1e-9 || t > 1.0 + 1e-9 { return false; }
		// Perpendicular distance squared (scaled): treat as on-line if tiny.
		let cx = (px - xa) - t * dx;
		let cy = (py - ya) - t * dy;
		(cx * cx + cy * cy) < 1e-18
	};

	// Port plates: for every CDT edge whose two endpoints lie on the port
	// line, emit a vertical wall in each slab covered by the port's z-range.
	for port in &spec_eff.ports {
		let port_tag = *tag_for_name.get(&port.name).unwrap();
		for &(p, q) in &edges2d {
			let pp = vertices2d[p]; let qq = vertices2d[q];
			if !on_segment(pp[0], pp[1], port.xy_a[0], port.xy_a[1], port.xy_b[0], port.xy_b[1]) { continue; }
			if !on_segment(qq[0], qq[1], port.xy_a[0], port.xy_a[1], port.xy_b[0], port.xy_b[1]) { continue; }
			for (z0, z1, _) in &slab_tags {
				if !(port.z_top > *z0 + Z_TOL && port.z_bottom < *z1 - Z_TOL) { continue; }
				let a0 = intern(pp[0], pp[1], *z0, &mut node_at, &mut nodes);
				let b0 = intern(qq[0], qq[1], *z0, &mut node_at, &mut nodes);
				let a1 = intern(pp[0], pp[1], *z1, &mut node_at, &mut nodes);
				let b1 = intern(qq[0], qq[1], *z1, &mut node_at, &mut nodes);
				emit_quad_both_diags(a0, b0, b1, a1, port_tag, &mut tris, &mut tri_tag);
			}
		}
	}

	// ABC: outer bbox walls + top + bottom caps.
	let abc_tag = *tag_for_name.get(&spec_eff.abc_tag).unwrap();
	let [xmin, ymin] = spec_eff.footprint_min;
	let [xmax, ymax] = spec_eff.footprint_max;
	let footprint_walls = [
		(xmin, ymin, xmax, ymin),
		(xmax, ymin, xmax, ymax),
		(xmax, ymax, xmin, ymax),
		(xmin, ymax, xmin, ymin),
	];
	for (xa, ya, xb, yb) in footprint_walls {
		for &(p, q) in &edges2d {
			let pp = vertices2d[p]; let qq = vertices2d[q];
			if !on_segment(pp[0], pp[1], xa, ya, xb, yb) { continue; }
			if !on_segment(qq[0], qq[1], xa, ya, xb, yb) { continue; }
			for (z0, z1, _) in &slab_tags {
				let a0 = intern(pp[0], pp[1], *z0, &mut node_at, &mut nodes);
				let b0 = intern(qq[0], qq[1], *z0, &mut node_at, &mut nodes);
				let a1 = intern(pp[0], pp[1], *z1, &mut node_at, &mut nodes);
				let b1 = intern(qq[0], qq[1], *z1, &mut node_at, &mut nodes);
				emit_quad_both_diags(a0, b0, b1, a1, abc_tag, &mut tris, &mut tri_tag);
			}
		}
	}
	// Top + bottom caps: walk the shared 2D tris.
	if let (Some(first), Some(last)) = (slab_tags.first(), slab_tags.last()) {
		for (z, is_top) in [(first.0, false), (last.1, true)] {
			for &[a, b, c] in &tris2d {
				let na = intern(vertices2d[a][0], vertices2d[a][1], z, &mut node_at, &mut nodes);
				let nb = intern(vertices2d[b][0], vertices2d[b][1], z, &mut node_at, &mut nodes);
				let nc = intern(vertices2d[c][0], vertices2d[c][1], z, &mut node_at, &mut nodes);
				if is_top {
					tris.push(na); tris.push(nc); tris.push(nb);
				} else {
					tris.push(na); tris.push(nb); tris.push(nc);
				}
				tri_tag.push(abc_tag);
			}
		}
	}

	// PML region override: re-tag any tet whose centroid sits outside the
	// inner domain to the matching pml_{x,y,z}{min,max} tag. Corner tets
	// (overlap of two PML directions) currently get the last-matching tag —
	// the FEM applies that single direction's stretch to them, which is an
	// acceptable approximation for v1. Order: x then y then z so z PML wins
	// if a tet sits in both lateral and vertical PML regions.
	if let Some(ptags) = &pml_tags {
		let n_tets = tets.len() / 4;
		for ti in 0..n_tets {
			let n0 = tets[4 * ti] as usize;
			let n1 = tets[4 * ti + 1] as usize;
			let n2 = tets[4 * ti + 2] as usize;
			let n3 = tets[4 * ti + 3] as usize;
			let cx = (nodes[3 * n0] + nodes[3 * n1] + nodes[3 * n2] + nodes[3 * n3]) * 0.25;
			let cy = (nodes[3 * n0 + 1] + nodes[3 * n1 + 1] + nodes[3 * n2 + 1] + nodes[3 * n3 + 1]) * 0.25;
			let cz = (nodes[3 * n0 + 2] + nodes[3 * n1 + 2] + nodes[3 * n2 + 2] + nodes[3 * n3 + 2]) * 0.25;
			let mut new_tag: Option<i32> = None;
			if cx < inner_xmin - Z_TOL { new_tag = Some(ptags[0].0); }
			else if cx > inner_xmax + Z_TOL { new_tag = Some(ptags[1].0); }
			if cy < inner_ymin - Z_TOL { new_tag = Some(ptags[2].0); }
			else if cy > inner_ymax + Z_TOL { new_tag = Some(ptags[3].0); }
			if cz < inner_zmin - Z_TOL { new_tag = Some(ptags[4].0); }
			else if cz > inner_zmax + Z_TOL { new_tag = Some(ptags[5].0); }
			if let Some(t) = new_tag { tet_tag[ti] = t; }
		}
	}

	// Post-process: collapse the speculative tri-emission to actual boundary
	// faces. We emit each conductor side-wall in both diagonal triangulations
	// and each cap with both windings to be robust to prism_to_tets' diagonal
	// choice — but downstream consumers (FEM, viewer) want one tri per face.
	// Build a key→(tag, idx) map keyed by the sorted face vertex triple. Only
	// keep tris whose key matches an actual tet face.
	let mut tet_faces: std::collections::HashSet<(u32, u32, u32)> =
		std::collections::HashSet::with_capacity(tets.len());
	for chunk in tets.chunks_exact(4) {
		let v = [chunk[0], chunk[1], chunk[2], chunk[3]];
		for &[i, j, k] in &[[0u8,1,2],[0,1,3],[0,2,3],[1,2,3]] {
			let mut f = [v[i as usize], v[j as usize], v[k as usize]];
			f.sort();
			tet_faces.insert((f[0], f[1], f[2]));
		}
	}
	let mut seen: std::collections::HashMap<(u32, u32, u32), usize> =
		std::collections::HashMap::with_capacity(tris.len() / 4);
	let mut tris_filtered: Vec<u32> = Vec::with_capacity(tris.len() / 4);
	let mut tri_tag_filtered: Vec<i32> = Vec::with_capacity(tri_tag.len() / 4);
	for (i, chunk) in tris.chunks_exact(3).enumerate() {
		let mut f = [chunk[0], chunk[1], chunk[2]];
		f.sort();
		let key = (f[0], f[1], f[2]);
		if !tet_faces.contains(&key) { continue; }
		if let Some(&prev_i) = seen.get(&key) {
			// Already kept once. If the previous tag was 0 (untagged) and this
			// one isn't, overwrite — otherwise keep the first.
			if tri_tag_filtered[prev_i] == 0 && tri_tag[i] != 0 {
				tri_tag_filtered[prev_i] = tri_tag[i];
			}
			continue;
		}
		seen.insert(key, tris_filtered.len() / 3);
		tris_filtered.extend_from_slice(chunk);
		tri_tag_filtered.push(tri_tag[i]);
	}

	let pml_regions: Vec<crate::output::PmlRegionInfo> = if let (Some(p), Some(ptags)) = (&spec.pml, &pml_tags) {
		ptags.iter().map(|(tag, dir, inner)| crate::output::PmlRegionInfo {
			volume_tag: *tag,
			direction: *dir,
			inner_face: *inner,
			thickness: p.thickness,
		}).collect()
	} else {
		Vec::new()
	};

	Ok(MeshOutput {
		nodes, tets, tris: tris_filtered, tet_tag,
		tri_tag: tri_tag_filtered,
		tag_names, tag_dim,
		pml_regions,
	})
}
