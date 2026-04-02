/// End-to-end test: straight WR-90 waveguide section.
/// Expected: |S11| ≈ 0 (matched), |S21| ≈ 1 (full transmission).

use num_complex::Complex64 as C64;
use rapidfem::mesh_io::load_mesh;
use rapidfem::basis::Nedelec2Basis;
use rapidfem::waveguide::{RectWaveguide, detect_port_cs};
use rapidfem::assembly::assemble_and_solve;
use rapidfem::sparam::extract_s_parameter;
use rapidfem::constants::*;

#[test]
fn test_straight_waveguide_s_params() {
    let mesh = load_mesh("tests/meshes/wr90_straight.msh").expect("Load mesh");
    let basis = Nedelec2Basis::new(&mesh);

    let a = 22.86e-3;
    let b = 10.16e-3;
    let freq = 10.0e9;
    let k0 = 2.0 * PI * freq / C0;

    // Get port triangle indices
    let port1_tris = mesh.tris_for_tag(3).to_vec();
    let port2_tris = mesh.tris_for_tag(4).to_vec();
    let pec_tris = mesh.tris_for_tag(1).to_vec();

    eprintln!("Port1: {} tris, Port2: {} tris, PEC: {} tris",
        port1_tris.len(), port2_tris.len(), pec_tris.len());

    assert!(!port1_tris.is_empty(), "No port1 triangles found");
    assert!(!port2_tris.is_empty(), "No port2 triangles found");

    // Create port definitions
    let cs1 = detect_port_cs(&mesh.nodes, &port1_tris, &mesh.tris, a, b);
    let cs2 = detect_port_cs(&mesh.nodes, &port2_tris, &mesh.tris, a, b);

    let port1 = RectWaveguide {
        width: a, height: b, mode: (1, 0), er: 1.0,
        cs: cs1, port_number: 1,
    };
    let port2 = RectWaveguide {
        width: a, height: b, mode: (1, 0), er: 1.0,
        cs: cs2, port_number: 2,
    };

    let ports = vec![port1, port2];
    let port_tris = vec![port1_tris.clone(), port2_tris.clone()];

    // Assemble and solve
    let result = assemble_and_solve(
        &mesh, &basis, &ports, &port_tris, &pec_tris, freq, None,
    );

    eprintln!("\nS-parameter extraction:");

    // For S-parameter extraction, we need a field interpolation function.
    // For now, use a simple DOF-based evaluation at port face points.
    // This is a placeholder — proper Nedelec-2 interpolation will come in interp.rs.

    // Just check that solutions are non-zero
    for (pi, sol) in result.solutions.iter().enumerate() {
        let norm: f64 = sol.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        eprintln!("  Port {} solution ||x|| = {:.6e}", pi, norm);
        assert!(norm > 0.0, "Solution {} is zero", pi);
    }

    eprintln!("\n=== End-to-end test passed (solutions non-zero) ===");
    eprintln!("TODO: Add Nedelec-2 field interpolation for proper S-param extraction");
}
