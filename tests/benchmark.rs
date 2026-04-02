/// Benchmark: RapidFEM vs EMerge timing comparison.
use num_complex::Complex64 as C64;
use rapidfem::mesh_io::load_mesh;
use rapidfem::basis::Nedelec2Basis;
use rapidfem::waveguide::{RectWaveguide, CoordinateSystem};
use rapidfem::assembly::assemble_and_solve;
use rapidfem::sparam::sparam_waveport;
use rapidfem::interp;
use rapidfem::constants::*;

#[test]
fn benchmark_straight_waveguide() {
    let mesh = load_mesh("tests/meshes/wr90_straight.msh").expect("Load mesh");
    let basis = Nedelec2Basis::new(&mesh);
    let freq = 10.0e9;
    let k0 = 2.0 * PI * freq / C0;

    let port1_tris = mesh.tris_for_tag(3).to_vec();
    let port2_tris = mesh.tris_for_tag(4).to_vec();
    let pec_tris = mesh.tris_for_tag(1).to_vec();

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

    let ports: Vec<&dyn rapidfem::port::Port> = vec![&port1, &port2];
    let port_tris: Vec<&[usize]> = vec![&port1_tris, &port2_tris];

    let t_total = std::time::Instant::now();
    let result = assemble_and_solve(&mesh, &basis, &ports, &port_tris, &pec_tris, freq, None);
    let solve_time = t_total.elapsed().as_secs_f64();

    // S-param extraction
    let sol0 = &result.solutions[0];
    let fieldf = |x: f64, y: f64, z: f64| -> (C64, C64, C64) {
        match interp::find_containing_tet(&mesh, x, y, z) {
            Some(tet) => interp::eval_field_in_tet(&mesh, &basis, sol0, tet, x, y, z),
            None => (C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)),
        }
    };
    let p1_verts: Vec<[usize; 3]> = port1_tris.iter().map(|&ti| mesh.tris[ti]).collect();
    let p2_verts: Vec<[usize; 3]> = port2_tris.iter().map(|&ti| mesh.tris[ti]).collect();
    let s11 = sparam_waveport(&mesh.nodes, &p1_verts, &port1, k0, true, &fieldf, 4);
    let s21 = sparam_waveport(&mesh.nodes, &p2_verts, &port2, k0, false, &fieldf, 4);
    let total_time = t_total.elapsed().as_secs_f64();

    eprintln!("\n=== BENCHMARK: RapidFEM ===");
    eprintln!("  {} tets, {} DOFs", mesh.n_tets(), basis.n_field);
    eprintln!("  Solve time:  {:.3}s", solve_time);
    eprintln!("  Total time:  {:.3}s", total_time);
    eprintln!("  |S11| = {:.6} ({:.1} dB)", s11.norm(), 20.0*s11.norm().max(1e-10).log10());
    eprintln!("  |S21| = {:.6} ({:.1} dB)", s21.norm(), 20.0*s21.norm().max(1e-10).log10());
    eprintln!();
    eprintln!("=== EMerge reference (same mesh, SuperLU) ===");
    eprintln!("  600 tets, 4858 DOFs, 0.058s");
    eprintln!("  |S11| = 0.001496, |S21| = 0.999824");
}
