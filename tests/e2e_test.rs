/// End-to-end test: straight WR-90 waveguide.
/// EMerge reference: S11=0.001496, S21=0.999824.

use num_complex::Complex64 as C64;
use rapidfem::mesh_io::load_mesh;
use rapidfem::basis::Nedelec2Basis;
use rapidfem::waveguide::{RectWaveguide, CoordinateSystem};
use rapidfem::assembly::assemble_and_solve;
use rapidfem::sparam::sparam_waveport;
use rapidfem::interp;
use rapidfem::constants::*;

#[test]
fn test_straight_waveguide_sparams() {
    let mesh = load_mesh("tests/meshes/wr90_straight.msh").expect("Load mesh");
    let basis = Nedelec2Basis::new(&mesh);

    let freq = 10.0e9;
    let k0 = 2.0 * PI * freq / C0;

    let port1_tris = mesh.tris_for_tag(3).to_vec();
    let port2_tris = mesh.tris_for_tag(4).to_vec();
    let pec_tris = mesh.tris_for_tag(1).to_vec();

    // Port CS from EMerge reference
    let cs1 = CoordinateSystem::new(
        [0.01143, 0.0, 0.00508], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0],
    );
    let cs2 = CoordinateSystem::new(
        [0.01143, 0.03, 0.00508], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0],
    );

    let port1 = RectWaveguide {
        port_number: 1, power: 1.0, mode: (1, 0), er: 1.0,
        polarization: 1.0, dims: (22.86e-3, 10.16e-3), cs: cs1,
    };
    let port2 = RectWaveguide {
        port_number: 2, power: 1.0, mode: (1, 0), er: 1.0,
        polarization: 1.0, dims: (22.86e-3, 10.16e-3), cs: cs2,
    };

    let ports: Vec<&RectWaveguide> = vec![&port1, &port2];
    let port_tris: Vec<&[usize]> = vec![&port1_tris, &port2_tris];

    let result = assemble_and_solve(&mesh, &basis, &ports, &port_tris, &pec_tris, freq);

    // S-param extraction: exact port of EMerge's approach
    // EMerge: fieldf = self.basis.interpolate_Ef(solution, tetids=port_tets)
    // which calls MATHLIB.ned2_tet_interp — 3D tet interpolation searching all tets.
    let sol0 = &result.solutions[0];

    // Build field evaluator using find_containing_tet (same as EMerge's ned2_tet_interp)
    let fieldf = |x: f64, y: f64, z: f64| -> (C64, C64, C64) {
        match interp::find_containing_tet(&mesh, x, y, z) {
            Some(tet) => interp::eval_field_in_tet(&mesh, &basis, sol0, tet, x, y, z),
            None => (C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)),
        }
    };

    // Test: check if find_containing_tet works for port face points
    let qpts = rapidfem::quadrature::gaus_quad_tri(4);
    let test_tri = mesh.tris[port1_tris[0]];
    let v0 = mesh.nodes[test_tri[0]];
    let v1 = mesh.nodes[test_tri[1]];
    let v2 = mesh.nodes[test_tri[2]];
    let mut found = 0;
    let mut missed = 0;
    for qp in &qpts {
        let x = v0[0]*qp[1] + v1[0]*qp[2] + v2[0]*qp[3];
        let y = v0[1]*qp[1] + v1[1]*qp[2] + v2[1]*qp[3];
        let z = v0[2]*qp[1] + v1[2]*qp[2] + v2[2]*qp[3];
        if interp::find_containing_tet(&mesh, x, y, z).is_some() {
            found += 1;
        } else {
            missed += 1;
        }
    }
    eprintln!("  Port1 tri[0] quad points: found={}, missed={}", found, missed);

    // Check local edge mapping for tet 0
    {
        let tet = &mesh.tets[0];
        let tet_edges = &mesh.tet_to_edge[0];
        let global_edge_nodes: [[usize; 2]; 6] = std::array::from_fn(|i| mesh.edges[tet_edges[i]]);
        let l_edge = rapidfem::basis::local_mapping(tet, &global_edge_nodes);
        eprintln!("  Tet 0: nodes={:?}", tet);
        eprintln!("  local_edge_map = {:?}", l_edge);
        eprintln!("  EMerge ref: [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]");
    }

    // Check a few DOF values
    eprintln!("  sol0[0..5] = {:?}", &sol0[0..5].iter().map(|x| format!("{:.4e}", x)).collect::<Vec<_>>());

    // Check field at test point
    let test_x = 0.01;
    let test_y = 0.001;
    let test_z = 0.005;
    if let Some(tet) = interp::find_containing_tet(&mesh, test_x, test_y, test_z) {
        let (ex, ey, ez) = interp::eval_field_in_tet(&mesh, &basis, sol0, tet, test_x, test_y, test_z);
        eprintln!("  Field at ({},{},{}): Ex={:.6e}, Ey={:.6e}, Ez={:.6e}", test_x, test_y, test_z, ex, ey, ez);
        eprintln!("  EMerge ref: Ex=9.39e0-4.59ej, Ey=-1.22e0-1.81ej, Ez=2.82e3-4.51e2j");
    } else {
        eprintln!("  FAILED: could not find tet for test point");
    }

    // S-param extraction
    let port1_tri_verts: Vec<[usize; 3]> = port1_tris.iter().map(|&ti| mesh.tris[ti]).collect();
    let port2_tri_verts: Vec<[usize; 3]> = port2_tris.iter().map(|&ti| mesh.tris[ti]).collect();

    let s11 = sparam_waveport(&mesh.nodes, &port1_tri_verts, &port1, k0, true, &fieldf, 4);
    let s21 = sparam_waveport(&mesh.nodes, &port2_tri_verts, &port2, k0, false, &fieldf, 4);

    eprintln!("\n=== S-parameters ===");
    eprintln!("  |S11| = {:.6} ({:.1} dB)", s11.norm(), 20.0 * s11.norm().max(1e-10).log10());
    eprintln!("  |S21| = {:.6} ({:.1} dB)", s21.norm(), 20.0 * s21.norm().max(1e-10).log10());
    eprintln!("  |S11|²+|S21|² = {:.6}", s11.norm_sqr() + s21.norm_sqr());
    eprintln!("  EMerge ref: |S11|=0.001496, |S21|=0.999824");
}
