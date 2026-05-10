//! Reproduce the live_microstrip end-to-end pipeline natively so panics
//! show up with line numbers instead of WASM `unreachable`.

use rapidfem::simulation::Simulation;

fn microstrip_spec_json() -> String {
    let um = 1e-6;
    let trace_l = 200.0 * um;
    let trace_w = 5.0 * um;
    let pad = 30.0 * um;
    let air_h = 30.0 * um;
    let sub_h = 15.0 * um;
    let pml_t = 15.0 * um;
    let _ = (air_h, sub_h);
    let s = serde_json::json!({
        "footprint_min": [-trace_l / 2.0 - pad, -trace_w / 2.0 - pad],
        "footprint_max": [trace_l / 2.0 + pad, trace_w / 2.0 + pad],
        "dielectrics": [
            {"name": "substrate", "z_bottom": -sub_h, "z_top": 0.0},
            {"name": "oxide", "z_bottom": 0.0, "z_top": 5.625 * um},
            {"name": "air", "z_bottom": 5.625 * um, "z_top": 5.625 * um + air_h},
        ],
        "conductors": [
            {"name": "met5",
             "xy": [[-trace_l/2.0, -trace_w/2.0], [trace_l/2.0, -trace_w/2.0],
                    [trace_l/2.0, trace_w/2.0], [-trace_l/2.0, trace_w/2.0]],
             "z_bottom": 4.365 * um, "z_top": 5.625 * um},
            {"name": "li1_gnd",
             "xy": [[-trace_l*0.6, -trace_w*5.0], [trace_l*0.6, -trace_w*5.0],
                    [trace_l*0.6, trace_w*5.0], [-trace_l*0.6, trace_w*5.0]],
             "z_bottom": 0.0, "z_top": 0.1 * um},
        ],
        "ports": [
            {"name": "p1",
             "xy_a": [-trace_l/2.0, -trace_w/2.0],
             "xy_b": [-trace_l/2.0, trace_w/2.0],
             "z_bottom": 0.1 * um, "z_top": 4.365 * um},
            {"name": "p2",
             "xy_a": [trace_l/2.0, -trace_w/2.0],
             "xy_b": [trace_l/2.0, trace_w/2.0],
             "z_bottom": 0.1 * um, "z_top": 4.365 * um},
        ],
        "abc_tag": "abc",
        "maxh": 35.0 * um,
        "z_maxh": 15.0 * um,
        "pml": { "thickness": pml_t },
    });
    s.to_string()
}

#[test]
fn live_microstrip_end_to_end() {
    let spec_json = microstrip_spec_json();
    let spec: rapidfem_mesher::MeshSpec = serde_json::from_str(&spec_json).unwrap();
    let mout = rapidfem_mesher::mesh(&spec).expect("mesh");
    eprintln!("mesh: {} nodes, {} tets, {} tris", mout.n_nodes(), mout.n_tets(), mout.n_tris());
    let tag_of = |name: &str| -> Option<i32> {
        mout.tag_names.iter().find(|(_, n)| n == name).map(|(t, _)| *t)
    };
    eprintln!("tags: {:?}", mout.tag_names);

    // Build the same TOML wasm-demo/src/lib.rs builds
    let mut toml = String::from("[mesh]\nfile = \"(in-memory)\"\n\n");
    toml.push_str("[frequency]\nvalues = [1e9]\n\n");
    for port in &spec.ports {
        if let Some(t) = tag_of(&port.name) {
            toml.push_str(&format!(
                "[[ports]]\ntype = \"lumped\"\ntag = {t}\nz0 = 50\npower = 1\ndirection = [0, 0, 1]\nwidth = 0\nheight = 0\n\n"
            ));
        }
    }
    if let Some(t) = tag_of("abc") {
        toml.push_str(&format!("[[ports]]\ntype = \"abc\"\ntag = {t}\norder = 1\nabctype = \"B\"\n\n"));
    }
    let pec_tags: Vec<String> = spec.conductors.iter()
        .filter_map(|c| tag_of(&c.name)).map(|t| t.to_string()).collect();
    toml.push_str(&format!("[pec]\ntags = [{}]\n\n", pec_tags.join(", ")));
    for d in &spec.dielectrics {
        if let Some(t) = tag_of(&d.name) {
            let er = if d.name == "substrate" { 11.9 } else if d.name == "oxide" { 4.2 } else { 1.0 };
            let sigma = if d.name == "substrate" { 10.0 } else { 0.0 };
            toml.push_str(&format!(
                "[[materials]]\nvolume_tag = {t}\ner = {er}\nur = 1\ntand = 0\nconductivity = {sigma}\n\n"
            ));
        }
    }
    // PML materials + regions, mirroring wasm-demo/src/lib.rs.
    for region in &mout.pml_regions {
        toml.push_str(&format!(
            "[[materials]]\nvolume_tag = {t}\ner = 1\nur = 1\ntand = 0\nconductivity = 0\n\n",
            t = region.volume_tag
        ));
        toml.push_str(&format!(
            "[[pml]]\nvolume_tag = {t}\ndirection = [{:e}, {:e}, {:e}]\n\
             inner_face = {:e}\nthickness = {:e}\ner_base = 1\nur_base = 1\nexponent = 1.5\ndelta_max = 8\n\n",
            region.direction[0], region.direction[1], region.direction[2],
            region.inner_face, region.thickness, t = region.volume_tag
        ));
    }
    toml.push_str("[output]\nz0 = 50\n");
    eprintln!("--- TOML ---\n{toml}\n--- END ---");

    // Convert MeshOutput to rapidfem::Mesh exactly like wasm-demo/src/lib.rs does
    let nodes: Vec<[f64; 3]> = (0..mout.n_nodes())
        .map(|i| [mout.nodes[i*3], mout.nodes[i*3+1], mout.nodes[i*3+2]]).collect();
    let tets: Vec<[usize; 4]> = (0..mout.n_tets())
        .map(|i| [mout.tets[i*4] as usize, mout.tets[i*4+1] as usize,
                  mout.tets[i*4+2] as usize, mout.tets[i*4+3] as usize]).collect();
    let mut mesh = rapidfem::mesh::Mesh::from_tets(nodes, tets);
    for (ti, &tag) in mout.tet_tag.iter().enumerate() {
        if tag != 0 { mesh.vtag_to_tet.entry(tag).or_default().push(ti); }
    }
    for (i, &tag) in mout.tri_tag.iter().enumerate() {
        if tag == 0 { continue; }
        let mut s = [mout.tris[i*3] as usize, mout.tris[i*3+1] as usize, mout.tris[i*3+2] as usize];
        s.sort();
        if let Some(&ti) = mesh.inv_tris.get(&(s[0], s[1], s[2])) {
            mesh.ftag_to_tri.entry(tag).or_default().push(ti);
        }
    }
    eprintln!("vtag_to_tet keys: {:?}", mesh.vtag_to_tet.keys().collect::<Vec<_>>());
    eprintln!("ftag_to_tri keys + counts: {:?}",
              mesh.ftag_to_tri.iter().map(|(k, v)| (*k, v.len())).collect::<Vec<_>>());

    let config = rapidfem::config::parse_config(&toml).expect("config");
    let sim = Simulation::new(mesh, config);
    let _result = sim.run_sweep();
    eprintln!("solve OK");
}
