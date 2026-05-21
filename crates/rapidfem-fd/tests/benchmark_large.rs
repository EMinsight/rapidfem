/// Large-scale benchmarks: multiple geometries and mesh sizes.
use num_complex::Complex64 as C64;
use rapidfem_fd::mesh_io::load_mesh;
use rapidfem_fd::basis::Nedelec2Basis;
use rapidfem_fd::waveguide::{RectWaveguide, detect_rect_port, CoordinateSystem};
use rapidfem_fd::assembly::assemble_and_solve;
use rapidfem_fd::sparam::sparam_waveport;
use rapidfem_fd::interp::{self, TetGrid};
use rapidfem_fd::constants::*;

fn run_bench(mesh_path: &str, label: &str, a: f64, b: f64) {
    let mesh = match load_mesh(mesh_path) {
        Ok(m) => m,
        Err(e) => { eprintln!("  {}: SKIP ({})", label, e); return; }
    };
    let basis = Nedelec2Basis::new(&mesh);
    let freq = 10.0e9;
    let k0 = 2.0 * PI * freq / C0;

    let port1_tris = mesh.tris_for_tag(3).to_vec();
    let port2_tris = mesh.tris_for_tag(4).to_vec();
    let pec_tris = mesh.tris_for_tag(1).to_vec();

    if port1_tris.is_empty() || port2_tris.is_empty() {
        eprintln!("  {}: SKIP (no port tris)", label);
        return;
    }

    let (cs1, w1, h1) = detect_rect_port(&mesh, &port1_tris);
    let (cs2, _, _) = detect_rect_port(&mesh, &port2_tris);

    let port1 = RectWaveguide {
        port_number: 1, power: 1.0, mode: (1, 0), er: 1.0,
        polarization: 1.0, dims: (a, b), cs: cs1,
    };
    let port2 = RectWaveguide {
        port_number: 2, power: 1.0, mode: (1, 0), er: 1.0,
        polarization: 1.0, dims: (a, b), cs: cs2,
    };

    let ports: Vec<&dyn rapidfem_fd::port::Port> = vec![&port1, &port2];
    let port_tris: Vec<&[usize]> = vec![&port1_tris, &port2_tris];

    let t0 = std::time::Instant::now();
    let result = assemble_and_solve(&mesh, &basis, &ports, &port_tris, &pec_tris, freq, None)
        .expect("assemble_and_solve failed");
    let solve_time = t0.elapsed().as_secs_f64();

    // S-param extraction with TetGrid
    let grid = TetGrid::new(&mesh);
    let sol0 = &result.solutions[0];
    let fieldf = |x: f64, y: f64, z: f64| -> (C64, C64, C64) {
        match grid.find_containing_tet(&mesh, x, y, z) {
            Some(tet) => interp::eval_field_in_tet(&mesh, &basis, sol0, tet, x, y, z),
            None => (C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)),
        }
    };
    let p1v: Vec<[usize; 3]> = port1_tris.iter().map(|&ti| mesh.tris[ti]).collect();
    let p2v: Vec<[usize; 3]> = port2_tris.iter().map(|&ti| mesh.tris[ti]).collect();
    let p1_ref: &dyn rapidfem_fd::port::Port = &port1;
    let p2_ref: &dyn rapidfem_fd::port::Port = &port2;
    let s11 = sparam_waveport(&mesh.nodes, &p1v, p1_ref, k0, true, &fieldf, 4);
    let s21 = sparam_waveport(&mesh.nodes, &p2v, p2_ref, k0, false, &fieldf, 4);
    let total = t0.elapsed().as_secs_f64();

    eprintln!("  {}: {} tets, {} DOFs, solve={:.3}s, total={:.3}s, |S11|={:.4}, |S21|={:.4}",
        label, mesh.n_tets(), basis.n_field, solve_time, total, s11.norm(), s21.norm());
}

#[test]
#[ignore = "long-running benchmark; run with `cargo test --test benchmark_large -- --ignored`"]
fn benchmark_large_models() {
    let a = 22.86e-3;
    let b = 10.16e-3;

    eprintln!("\n=== LARGE-SCALE BENCHMARKS ===\n");

    eprintln!("Two-iris bandpass filter:");
    run_bench("tests/meshes/two_iris_coarse.msh", "coarse", a, b);
    run_bench("tests/meshes/two_iris_medium.msh", "medium", a, b);
    run_bench("tests/meshes/two_iris_fine.msh", "fine", a, b);

    eprintln!("\nStepped impedance filter:");
    run_bench("tests/meshes/stepped_coarse.msh", "coarse", a, b);
    run_bench("tests/meshes/stepped_medium.msh", "medium", a, b);
    run_bench("tests/meshes/stepped_fine.msh", "fine", a, b);

    eprintln!("\nLarge straight waveguide (scaling test):");
    run_bench("tests/meshes/wg_50k_50k.msh", "17K tets", a, b);
    run_bench("tests/meshes/wg_50k_100k.msh", "55K tets", a, b);
}
