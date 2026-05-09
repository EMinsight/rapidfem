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

    // Nodal |E| per (freq, excitation). result.solutions[freq][exc] is the
    // DOF vector for each driven excitation at each frequency.
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
        .map_err(|e| JsValue::from_str(&format!("serialize: {}", e)))
}
