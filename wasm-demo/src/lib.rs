//! WASM bindings for rapidfem. Exposes a single `run_sweep` function that takes
//! a mesh file (bytes) + TOML config (string) and returns a flat S-parameter array
//! plus frequencies as a JSON-serializable struct.
//!
//! Build: `wasm-pack build --target web --release`. Output goes to `pkg/`.

use rapidfem::simulation::Simulation;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
pub struct SweepResultJs {
    pub frequencies_hz: Vec<f64>,
    pub n_driven: usize,
    pub n_nodes: usize,
    /// Flattened complex S-params, [freq][obs][exc] in row-major order.
    /// Length = `n_freq * n_driven * n_driven * 2` (interleaved real,imag).
    pub sparams_flat: Vec<f64>,
    /// Per-node phasor terms (A, B, C) per excitation. Three f32 per node
    /// encode the time-domain magnitude evolution via
    ///   |E(t)|² = A·cos²(ωt) + B·sin²(ωt) − 2·C·cos·sin
    /// Static |E|² is recoverable as (A + B) when phase doesn't matter.
    /// Layout: [freq][exc][node][0..2], flat length = n_freq · n_driven · n_nodes · 3.
    pub fields_abc_flat: Vec<f32>,
    pub solve_time_s: f64,
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// FEM-side options pairing with a MeshSpec. JSON-friendly; everything
/// references regions by NAME (the WASM side resolves to integer tags
/// using the mesh output's tag table).
#[derive(serde::Deserialize)]
pub struct SolveOptions {
    pub frequencies_hz: Vec<f64>,
    #[serde(default = "default_z0")]
    pub port_z0: f64,
    /// Per-dielectric material properties keyed by slab name. Defaults to
    /// εr=1, σ=0 if a slab name isn't in the map.
    #[serde(default)]
    pub materials: std::collections::HashMap<String, MaterialOpts>,
}
fn default_z0() -> f64 { 50.0 }

#[derive(serde::Deserialize, Clone, Copy)]
pub struct MaterialOpts {
    #[serde(default = "default_one")]
    pub er: f64,
    #[serde(default)]
    pub conductivity: f64,
    #[serde(default)]
    pub tand: f64,
}
fn default_one() -> f64 { 1.0 }

/// Mesh-only entry: take a MeshSpec, run the mesher, return raw mesh data
/// in a viewer-friendly shape (nodes/tets/tris + tag tables). Used by the
/// JS side to populate the 3D viewer without solving anything.
#[derive(Serialize)]
pub struct MeshDataJs {
    pub nodes: Vec<f64>,           // [x,y,z, ...]
    pub tets: Vec<u32>,            // 4 indices per tet
    pub tris: Vec<u32>,            // 3 indices per tri
    pub tri_tag: Vec<i32>,
    pub tet_tag: Vec<i32>,
    pub tag_names: Vec<(i32, String)>,
    pub tag_dim: Vec<(i32, u8)>,
}

#[wasm_bindgen]
pub fn mesh_from_spec(spec_json: &str) -> Result<JsValue, JsValue> {
    let spec: rapidfem_mesher::MeshSpec = serde_json::from_str(spec_json)
        .map_err(|e| JsValue::from_str(&format!("spec parse: {e}")))?;
    let m = rapidfem_mesher::mesh(&spec)
        .map_err(|e| JsValue::from_str(&format!("mesh: {e}")))?;

    // For the in-browser viewer we want PML cells to look like part of the
    // surrounding dielectric (no separate "PML" category). Remap every
    // `_pml_*` tet tag to whichever real dielectric encloses its centroid.
    // - lateral PML (xmin/xmax/ymin/ymax): centroid z is inside the original
    //   dielectric stack → use the enclosing dielectric.
    // - vertical PML (zmin/zmax) + auxiliary `_pml_z_below/above` slabs:
    //   centroid z sits outside the stack → fallback to lowest/highest
    //   dielectric (same material, just stretched coordinates in the FEM).
    let lowest = spec.dielectrics.iter().min_by(|a, b| a.z_bottom.partial_cmp(&b.z_bottom).unwrap());
    let highest = spec.dielectrics.iter().max_by(|a, b| a.z_top.partial_cmp(&b.z_top).unwrap());
    let tag_of = |name: &str| -> i32 {
        m.tag_names.iter().find(|(_, n)| n == name).map(|(t, _)| *t).unwrap_or(0)
    };
    let name_of = |tag: i32| -> &str {
        m.tag_names.iter().find(|(t, _)| *t == tag).map(|(_, n)| n.as_str()).unwrap_or("")
    };

    let mut tet_tag_view = m.tet_tag.clone();
    for ti in 0..m.n_tets() {
        let name = name_of(m.tet_tag[ti]);
        if !name.starts_with("_pml_") { continue; }
        let n0 = m.tets[4 * ti] as usize;
        let n1 = m.tets[4 * ti + 1] as usize;
        let n2 = m.tets[4 * ti + 2] as usize;
        let n3 = m.tets[4 * ti + 3] as usize;
        let cz = (m.nodes[3 * n0 + 2] + m.nodes[3 * n1 + 2] + m.nodes[3 * n2 + 2] + m.nodes[3 * n3 + 2]) * 0.25;
        let mut new_tag: Option<i32> = None;
        for d in &spec.dielectrics {
            if cz >= d.z_bottom - 1e-12 && cz <= d.z_top + 1e-12 {
                new_tag = Some(tag_of(&d.name));
                break;
            }
        }
        if new_tag.is_none() {
            if let (Some(low), Some(high)) = (lowest, highest) {
                new_tag = Some(if cz < low.z_bottom { tag_of(&low.name) } else { tag_of(&high.name) });
            }
        }
        if let Some(t) = new_tag { tet_tag_view[ti] = t; }
    }

    // Strip `_pml_*` entries from the tag tables so the viewer never sees
    // them as a category at all.
    let tag_names_view: Vec<(i32, String)> = m.tag_names.iter()
        .filter(|(_, n)| !n.starts_with("_pml_"))
        .cloned().collect();
    let tag_dim_view: Vec<(i32, u8)> = m.tag_dim.iter()
        .filter(|(t, _)| tag_names_view.iter().any(|(t2, _)| t2 == t))
        .cloned().collect();

    let out = MeshDataJs {
        nodes: m.nodes,
        tets: m.tets,
        tris: m.tris,
        tri_tag: m.tri_tag,
        tet_tag: tet_tag_view,
        tag_names: tag_names_view,
        tag_dim: tag_dim_view,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("serialize: {e}")))
}

/// Mesh from a `MeshSpec` JSON, build the FEM config internally, and run
/// a sweep — all client-side, no `.msh` round-trip.
#[wasm_bindgen]
pub fn solve_from_spec(spec_json: &str, options_json: &str) -> Result<JsValue, JsValue> {
    let spec: rapidfem_mesher::MeshSpec = serde_json::from_str(spec_json)
        .map_err(|e| JsValue::from_str(&format!("spec parse: {e}")))?;
    let opts: SolveOptions = serde_json::from_str(options_json)
        .map_err(|e| JsValue::from_str(&format!("options parse: {e}")))?;
    let mout = rapidfem_mesher::mesh(&spec)
        .map_err(|e| JsValue::from_str(&format!("mesh: {e}")))?;

    // Name → integer tag from the mesher's tag table
    let tag_of = |name: &str| -> Option<i32> {
        mout.tag_names.iter().find(|(_, n)| n == name).map(|(t, _)| *t)
    };

    // Build the config TOML using integer tags
    let mut toml = String::from("[mesh]\nfile = \"(in-memory)\"\n\n");
    let freqs = opts.frequencies_hz.iter()
        .map(|v| format!("{v:e}")).collect::<Vec<_>>().join(", ");
    toml.push_str(&format!("[frequency]\nvalues = [{freqs}]\n\n"));
    for port in &spec.ports {
        if let Some(t) = tag_of(&port.name) {
            toml.push_str(&format!(
                "[[ports]]\ntype = \"lumped\"\ntag = {t}\nz0 = {z0}\n\
                 power = 1\ndirection = [0, 0, 1]\nwidth = 0\nheight = 0\n\n",
                z0 = opts.port_z0
            ));
        }
    }
    if let Some(t) = tag_of(&spec.abc_tag) {
        toml.push_str(&format!(
            "[[ports]]\ntype = \"abc\"\ntag = {t}\norder = 1\nabctype = \"B\"\n\n"
        ));
    }
    let pec_tags: Vec<String> = spec.conductors.iter()
        .filter_map(|c| tag_of(&c.name)).map(|t| t.to_string()).collect();
    toml.push_str(&format!("[pec]\ntags = [{}]\n\n", pec_tags.join(", ")));
    for d in &spec.dielectrics {
        if let Some(t) = tag_of(&d.name) {
            let m = opts.materials.get(&d.name).copied().unwrap_or(MaterialOpts {
                er: 1.0, conductivity: 0.0, tand: 0.0,
            });
            toml.push_str(&format!(
                "[[materials]]\nvolume_tag = {t}\ner = {er}\nur = 1\ntand = {tand}\nconductivity = {sigma}\n\n",
                er = m.er, tand = m.tand, sigma = m.conductivity
            ));
        }
    }

    // PML: for every region the mesher emitted, register a base [[materials]]
    // entry (so `build_material_tensors` doesn't choke on an unknown vol tag)
    // plus a [[pml]] entry that the FEM picks up to apply stretched-coordinate
    // Maxwell at run time.
    if let Some(p) = &spec.pml {
        let er_base = p.er_base.unwrap_or(1.0);
        let ur_base = p.ur_base.unwrap_or(1.0);
        let exponent = p.exponent.unwrap_or(1.5);
        let delta_max = p.delta_max.unwrap_or(8.0);
        for region in &mout.pml_regions {
            toml.push_str(&format!(
                "[[materials]]\nvolume_tag = {t}\ner = {er}\nur = {ur}\ntand = 0\nconductivity = 0\n\n",
                t = region.volume_tag, er = er_base, ur = ur_base
            ));
            toml.push_str(&format!(
                "[[pml]]\nvolume_tag = {t}\ndirection = [{dx:e}, {dy:e}, {dz:e}]\n\
                 inner_face = {inner:e}\nthickness = {thick:e}\n\
                 er_base = {er}\nur_base = {ur}\nexponent = {exp}\ndelta_max = {dmax}\n\n",
                t = region.volume_tag,
                dx = region.direction[0], dy = region.direction[1], dz = region.direction[2],
                inner = region.inner_face, thick = region.thickness,
                er = er_base, ur = ur_base, exp = exponent, dmax = delta_max
            ));
        }
    }

    toml.push_str(&format!("[output]\nz0 = {}\n", opts.port_z0));

    let config = rapidfem::config::parse_config(&toml)
        .map_err(|e| JsValue::from_str(&format!("config: {e}\n--- TOML ---\n{toml}")))?;
    let mesh = mesh_output_to_rapidfem(mout);
    let sim = Simulation::new(mesh, config);
    let result = sim.run_sweep();
    serialize_result(&sim, result)
}

/// Convert our mesher's `MeshOutput` to `rapidfem::Mesh`.
fn mesh_output_to_rapidfem(m: rapidfem_mesher::MeshOutput) -> rapidfem::mesh::Mesh {
    let nodes: Vec<[f64; 3]> = (0..m.n_nodes())
        .map(|i| [m.nodes[i * 3], m.nodes[i * 3 + 1], m.nodes[i * 3 + 2]])
        .collect();
    let tets: Vec<[usize; 4]> = (0..m.n_tets())
        .map(|i| {
            [
                m.tets[i * 4] as usize,
                m.tets[i * 4 + 1] as usize,
                m.tets[i * 4 + 2] as usize,
                m.tets[i * 4 + 3] as usize,
            ]
        })
        .collect();
    let mut mesh = rapidfem::mesh::Mesh::from_tets(nodes, tets);
    // Volume tags: per-tet → vtag_to_tet
    for (ti, &tag) in m.tet_tag.iter().enumerate() {
        if tag != 0 {
            mesh.vtag_to_tet.entry(tag).or_default().push(ti);
        }
    }
    // Surface tags: each output tri → look up its triangle index in mesh
    for (i, &tag) in m.tri_tag.iter().enumerate() {
        if tag == 0 { continue; }
        let a = m.tris[i * 3] as usize;
        let b = m.tris[i * 3 + 1] as usize;
        let c = m.tris[i * 3 + 2] as usize;
        let mut s = [a, b, c];
        s.sort();
        if let Some(&ti) = mesh.inv_tris.get(&(s[0], s[1], s[2])) {
            mesh.ftag_to_tri.entry(tag).or_default().push(ti);
        }
    }
    mesh
}

fn serialize_result(sim: &Simulation, result: rapidfem::simulation::SweepResult)
    -> Result<JsValue, JsValue>
{
    let n_freq = result.frequencies.len();
    let n_driven = result.n_driven;
    let n_nodes = sim.mesh.n_nodes();
    let mut flat = Vec::with_capacity(n_freq * n_driven * n_driven * 2);
    for f_mat in &result.sparams {
        for row in f_mat {
            for c in row {
                flat.push(c.re);
                flat.push(c.im);
            }
        }
    }
    let mut fields_abc = Vec::with_capacity(n_freq * n_driven * n_nodes * 3);
    for sols_at_freq in &result.solutions {
        for sol in sols_at_freq {
            let abc = sim.nodal_field_phasor_terms(sol);
            for t in abc { fields_abc.push(t[0]); fields_abc.push(t[1]); fields_abc.push(t[2]); }
        }
    }
    let out = SweepResultJs {
        frequencies_hz: result.frequencies,
        n_driven,
        n_nodes,
        sparams_flat: flat,
        fields_abc_flat: fields_abc,
        solve_time_s: result.solve_time_s,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("serialize: {e}")))
}
