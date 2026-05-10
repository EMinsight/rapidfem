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
    /// Nodal |E| magnitudes per excitation (V/m).
    /// Layout: [freq][exc][node], flat length = n_freq * n_driven * n_nodes.
    pub fields_flat: Vec<f32>,
    pub solve_time_s: f64,
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Run a frequency sweep on an in-memory mesh + TOML config.
/// Returns SweepResultJs as a JS object.
#[wasm_bindgen]
pub fn run_sweep(mesh_bytes: &[u8], config_toml: &str) -> Result<JsValue, JsValue> {
    let sim = Simulation::from_bytes(mesh_bytes, config_toml)
        .map_err(|e| JsValue::from_str(&format!("setup: {}", e)))?;
    let result = sim.run_sweep();
    serialize_result(&sim, result)
}

/// Mesh from a `MeshSpec` JSON, build the FEM config from a TOML string,
/// and run a single sweep — all client-side, no `.msh` round-trip.
/// `spec_json`: serialized `rapidfem_mesher::MeshSpec`.
#[wasm_bindgen]
pub fn solve_from_spec(spec_json: &str, config_toml: &str) -> Result<JsValue, JsValue> {
    let spec: rapidfem_mesher::MeshSpec = serde_json::from_str(spec_json)
        .map_err(|e| JsValue::from_str(&format!("spec parse: {e}")))?;
    let mout = rapidfem_mesher::mesh(&spec)
        .map_err(|e| JsValue::from_str(&format!("mesh: {e}")))?;
    let mesh = mesh_output_to_rapidfem(mout);
    let config = rapidfem::config::parse_config(config_toml)
        .map_err(|e| JsValue::from_str(&format!("config: {e}")))?;
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
    let mut fields = Vec::with_capacity(n_freq * n_driven * n_nodes);
    for sols_at_freq in &result.solutions {
        for sol in sols_at_freq {
            let nodal = sim.nodal_field_magnitudes(sol);
            fields.extend(nodal);
        }
    }
    let out = SweepResultJs {
        frequencies_hz: result.frequencies,
        n_driven,
        n_nodes,
        sparams_flat: flat,
        fields_flat: fields,
        solve_time_s: result.solve_time_s,
    };
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("serialize: {e}")))
}
