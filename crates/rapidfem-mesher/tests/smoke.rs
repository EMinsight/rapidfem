//! End-to-end smoke test: build a minimal microstrip MeshSpec, mesh it,
//! verify the output makes sense.

use rapidfem_mesher::{
	mesh, ConductorPolygon, DielectricSlab, MeshSpec, VerticalPort,
};

fn microstrip_spec() -> MeshSpec {
	let um = 1e-6;
	let trace_l = 200.0 * um;
	let trace_w = 5.0 * um;
	let pad = 50.0 * um;
	MeshSpec {
		footprint_min: [-trace_l / 2.0 - pad, -trace_w / 2.0 - pad],
		footprint_max: [trace_l / 2.0 + pad, trace_w / 2.0 + pad],
		dielectrics: vec![
			DielectricSlab {
				name: "substrate".into(),
				z_bottom: -30.0 * um,
				z_top: 0.0,
			},
			DielectricSlab {
				name: "oxide".into(),
				z_bottom: 0.0,
				z_top: 5.625 * um,
			},
			DielectricSlab {
				name: "air".into(),
				z_bottom: 5.625 * um,
				z_top: 35.625 * um,
			},
		],
		conductors: vec![
			ConductorPolygon {
				name: "met5".into(),
				xy: vec![
					[-trace_l / 2.0, -trace_w / 2.0],
					[trace_l / 2.0, -trace_w / 2.0],
					[trace_l / 2.0, trace_w / 2.0],
					[-trace_l / 2.0, trace_w / 2.0],
				],
				z_bottom: 4.365 * um,
				z_top: 5.625 * um,
			},
			ConductorPolygon {
				name: "li1_gnd".into(),
				xy: vec![
					[-trace_l * 0.6, -trace_w * 5.0],
					[trace_l * 0.6, -trace_w * 5.0],
					[trace_l * 0.6, trace_w * 5.0],
					[-trace_l * 0.6, trace_w * 5.0],
				],
				z_bottom: 0.0,
				z_top: 0.1 * um,
			},
		],
		ports: vec![
			VerticalPort {
				name: "p1".into(),
				xy_a: [-trace_l / 2.0, -trace_w / 2.0],
				xy_b: [-trace_l / 2.0, trace_w / 2.0],
				z_bottom: 0.1 * um,
				z_top: 4.365 * um,
			},
			VerticalPort {
				name: "p2".into(),
				xy_a: [trace_l / 2.0, -trace_w / 2.0],
				xy_b: [trace_l / 2.0, trace_w / 2.0],
				z_bottom: 0.1 * um,
				z_top: 4.365 * um,
			},
		],
		abc_tag: "abc".into(),
		maxh: 15.0 * um,
	}
}

#[test]
fn microstrip_meshes() {
	let spec = microstrip_spec();
	let m = mesh(&spec).expect("mesher should succeed");
	assert!(m.n_nodes() > 0, "no nodes");
	assert!(m.n_tets() > 0, "no tets");
	assert!(m.tet_tag.len() == m.n_tets(), "tet_tag length mismatch");
	// Tag table includes all the named entities + abc + 3 dielectrics
	let names: Vec<&str> = m.tag_names.iter().map(|(_, n)| n.as_str()).collect();
	for expected in ["substrate", "oxide", "air", "met5", "li1_gnd", "p1", "p2", "abc"] {
		assert!(names.contains(&expected), "missing tag {expected:?}");
	}
	println!(
		"microstrip mesh: {} nodes, {} tets, {} tris",
		m.n_nodes(), m.n_tets(), m.n_tris()
	);
}
