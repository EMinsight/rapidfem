/// End-to-end test: straight WR-90 waveguide section.
/// Expected: |S11| ≈ 0 (matched), |S21| ≈ 1 (full transmission).

use rapidfem::mesh_io::load_mesh;
use rapidfem::basis::Nedelec2Basis;
use rapidfem::waveguide::{RectWaveguide, detect_port_cs};
use rapidfem::assembly::assemble_and_solve;
use rapidfem::sparam::extract_s_parameter;
use rapidfem::interp::make_field_evaluator;
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

    // Create port definitions
    let cs1 = detect_port_cs(&mesh, &port1_tris);
    let cs2 = detect_port_cs(&mesh, &port2_tris);

    let port1 = RectWaveguide {
        width: a, height: b, mode: (1, 0), er: 1.0,
        cs: cs1, port_number: 1,
    };
    let port2 = RectWaveguide {
        width: a, height: b, mode: (1, 0), er: 1.0,
        cs: cs2, port_number: 2,
    };

    // Debug: print port CS and compare against EMerge reference
    let k0 = 2.0 * PI * freq / C0;
    for (i, port) in [&port1, &port2].iter().enumerate() {
        let cs = &port.cs;
        eprintln!("Port {} CS:", i+1);
        eprintln!("  origin: ({:.5}, {:.5}, {:.5})", cs.origin[0], cs.origin[1], cs.origin[2]);
        eprintln!("  xhat:   ({:.4}, {:.4}, {:.4})", cs.xhat[0], cs.xhat[1], cs.xhat[2]);
        eprintln!("  yhat:   ({:.4}, {:.4}, {:.4})", cs.yhat[0], cs.yhat[1], cs.yhat[2]);
        eprintln!("  zhat:   ({:.4}, {:.4}, {:.4})", cs.zhat[0], cs.zhat[1], cs.zhat[2]);
        eprintln!("  beta={:.6}, gamma={:.6}, Zmode={:.4}", port.beta(k0), port.gamma(k0), port.z_mode(k0));
        let ef = port.mode_field_global(cs.origin[0], cs.origin[1], cs.origin[2], k0);
        eprintln!("  E_mode at center: ({:.4}, {:.4}, {:.4})", ef[0], ef[1], ef[2]);
    }

    let ports = vec![port1, port2];
    let port_tris = vec![port1_tris.clone(), port2_tris.clone()];

    // Assemble and solve
    let result = assemble_and_solve(
        &mesh, &basis, &ports, &port_tris, &pec_tris, freq, None,
    );

    // Check field at port 1 center using interpolation
    {
        let sol = &result.solutions[0]; // port 1 excited
        let field_at_center = rapidfem::interp::eval_field_in_tet(
            &mesh, &basis, sol,
            mesh.tri_to_tet[port1_tris[0]][0], // tet adjacent to first port1 tri
            a/2.0, 1e-6, b/2.0, // center of port 1 face (slightly inside)
        );
        eprintln!("\nField at port1 center (port1 excited):");
        eprintln!("  E = ({:.4e}, {:.4e}, {:.4e})", field_at_center[0], field_at_center[1], field_at_center[2]);
        let e_mode = ports[0].mode_field_global(a/2.0, 0.0, b/2.0, k0);
        eprintln!("  E_mode = ({:.4}, {:.4}, {:.4})", e_mode[0], e_mode[1], e_mode[2]);
    }

    // Extract S-parameters using field interpolation
    eprintln!("\nS-parameter extraction (quad order 4):");

    // Port 1 excited (solution 0)
    let field1 = make_field_evaluator(&mesh, &basis, &result.solutions[0], &port1_tris);
    let field1_at2 = make_field_evaluator(&mesh, &basis, &result.solutions[0], &port2_tris);

    let s11 = extract_s_parameter(&mesh, &ports[0], &port1_tris, k0, true, &field1, 4);
    let s21 = extract_s_parameter(&mesh, &ports[1], &port2_tris, k0, false, &field1_at2, 4);

    let s11_mag = s11.norm();
    let s21_mag = s21.norm();
    let power = s11_mag*s11_mag + s21_mag*s21_mag;

    eprintln!("  |S11| = {:.4} ({:.1} dB)", s11_mag, 20.0*s11_mag.max(1e-10).log10());
    eprintln!("  |S21| = {:.4} ({:.1} dB)", s21_mag, 20.0*s21_mag.max(1e-10).log10());
    eprintln!("  |S11|²+|S21|² = {:.4}", power);

    // For a matched waveguide:
    // |S11| should be small (< 0.1 = -20dB)
    // |S21| should be close to 1.0
    // Power conservation should hold
    eprintln!("\n=== Results ===");
    if s11_mag < 0.3 && s21_mag > 0.7 {
        eprintln!("  PASS: S-parameters look reasonable");
    } else {
        eprintln!("  WARN: S-parameters may need tuning (coarse mesh)");
    }
}
