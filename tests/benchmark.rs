/// Benchmark suite: RapidFEM performance across mesh sizes and geometries.
use num_complex::Complex64 as C64;
use rapidfem::mesh_io::load_mesh;
use rapidfem::basis::Nedelec2Basis;
use rapidfem::waveguide::{RectWaveguide, CoordinateSystem};
use rapidfem::assembly::assemble_and_solve;
use rapidfem::sparam::sparam_waveport;
use rapidfem::interp;
use rapidfem::constants::*;

fn run_waveguide_benchmark(mesh_path: &str, label: &str) {
    let mesh = load_mesh(mesh_path).expect("Load mesh");
    let basis = Nedelec2Basis::new(&mesh);
    let freq = 10.0e9;
    let k0 = 2.0 * PI * freq / C0;

    let port1_tris = mesh.tris_for_tag(3).to_vec();
    let port2_tris = mesh.tris_for_tag(4).to_vec();
    let pec_tris = mesh.tris_for_tag(1).to_vec();

    // Detect port CS from mesh
    let cs1 = detect_cs(&mesh, &port1_tris);
    let cs2 = detect_cs(&mesh, &port2_tris);

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

    let t0 = std::time::Instant::now();
    let result = assemble_and_solve(&mesh, &basis, &ports, &port_tris, &pec_tris, freq, None);
    let solve_time = t0.elapsed().as_secs_f64();

    let sol0 = &result.solutions[0];
    let fieldf = |x: f64, y: f64, z: f64| -> (C64, C64, C64) {
        match interp::find_containing_tet(&mesh, x, y, z) {
            Some(tet) => interp::eval_field_in_tet(&mesh, &basis, sol0, tet, x, y, z),
            None => (C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)),
        }
    };
    let p1v: Vec<[usize; 3]> = port1_tris.iter().map(|&ti| mesh.tris[ti]).collect();
    let p2v: Vec<[usize; 3]> = port2_tris.iter().map(|&ti| mesh.tris[ti]).collect();
    let p1_ref: &dyn rapidfem::port::Port = &port1;
    let p2_ref: &dyn rapidfem::port::Port = &port2;
    let s11 = sparam_waveport(&mesh.nodes, &p1v, p1_ref, k0, true, &fieldf, 4);
    let s21 = sparam_waveport(&mesh.nodes, &p2v, p2_ref, k0, false, &fieldf, 4);
    let total = t0.elapsed().as_secs_f64();

    eprintln!("  {label}: {tets} tets, {dofs} DOFs, solve={solve:.3}s, total={total:.3}s, |S11|={s11:.4}, |S21|={s21:.4}",
        tets=mesh.n_tets(), dofs=basis.n_field,
        solve=solve_time, total=total,
        s11=s11.norm(), s21=s21.norm());
}

fn detect_cs(mesh: &rapidfem::mesh::Mesh, tri_ids: &[usize]) -> CoordinateSystem {
    // Compute center, normal, broad axis from port face
    let mut center = [0.0; 3];
    let mut count = 0.0;
    let mut verts_set = std::collections::HashSet::new();
    for &ti in tri_ids {
        for &vi in &mesh.tris[ti] {
            if verts_set.insert(vi) {
                for k in 0..3 { center[k] += mesh.nodes[vi][k]; }
                count += 1.0;
            }
        }
    }
    for k in 0..3 { center[k] /= count; }

    let first_tri = mesh.tris[tri_ids[0]];
    let v0 = mesh.nodes[first_tri[0]];
    let v1 = mesh.nodes[first_tri[1]];
    let v2 = mesh.nodes[first_tri[2]];
    let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
    let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
    let n = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
    let nn = (n[0]*n[0]+n[1]*n[1]+n[2]*n[2]).sqrt();
    let mut normal = [n[0]/nn, n[1]/nn, n[2]/nn];

    // Orient outward using adjacent tet
    let adj_tet = mesh.tri_to_tet[tri_ids[0]][0];
    let tet = &mesh.tets[adj_tet];
    let tc = [(mesh.nodes[tet[0]][0]+mesh.nodes[tet[1]][0]+mesh.nodes[tet[2]][0]+mesh.nodes[tet[3]][0])/4.0,
              (mesh.nodes[tet[0]][1]+mesh.nodes[tet[1]][1]+mesh.nodes[tet[2]][1]+mesh.nodes[tet[3]][1])/4.0,
              (mesh.nodes[tet[0]][2]+mesh.nodes[tet[1]][2]+mesh.nodes[tet[2]][2]+mesh.nodes[tet[3]][2])/4.0];
    let to_tet = [tc[0]-center[0], tc[1]-center[1], tc[2]-center[2]];
    if normal[0]*to_tet[0]+normal[1]*to_tet[1]+normal[2]*to_tet[2] > 0.0 {
        normal = [-normal[0], -normal[1], -normal[2]];
    }

    // Find extents
    let mut min_c = [f64::INFINITY; 3];
    let mut max_c = [f64::NEG_INFINITY; 3];
    for &vi in &verts_set {
        for k in 0..3 { min_c[k] = min_c[k].min(mesh.nodes[vi][k]); max_c[k] = max_c[k].max(mesh.nodes[vi][k]); }
    }
    let ext = [max_c[0]-min_c[0], max_c[1]-min_c[1], max_c[2]-min_c[2]];
    let na = if normal[0].abs() > normal[1].abs() && normal[0].abs() > normal[2].abs() { 0 }
        else if normal[1].abs() > normal[2].abs() { 1 } else { 2 };
    let mut face_axes: Vec<(usize, f64)> = (0..3).filter(|&k| k != na).map(|k| (k, ext[k])).collect();
    face_axes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut xhat = [0.0; 3];
    xhat[face_axes[0].0] = 1.0;

    // Build CS using the same method as waveguide.rs
    let zn = (normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]).sqrt();
    let zhat = [normal[0]/zn, normal[1]/zn, normal[2]/zn];
    let dot_xz = xhat[0]*zhat[0]+xhat[1]*zhat[1]+xhat[2]*zhat[2];
    let xraw = [xhat[0]-dot_xz*zhat[0], xhat[1]-dot_xz*zhat[1], xhat[2]-dot_xz*zhat[2]];
    let xn = (xraw[0]*xraw[0]+xraw[1]*xraw[1]+xraw[2]*xraw[2]).sqrt();
    let xhat_f = [xraw[0]/xn, xraw[1]/xn, xraw[2]/xn];
    let yhat = [zhat[1]*xhat_f[2]-zhat[2]*xhat_f[1], zhat[2]*xhat_f[0]-zhat[0]*xhat_f[2], zhat[0]*xhat_f[1]-zhat[1]*xhat_f[0]];

    CoordinateSystem::new(center, xhat_f, yhat, zhat)
}

#[test]
fn benchmark_scaling() {
    eprintln!("\n=== RapidFEM Benchmark Suite (faer sparse LU) ===\n");

    eprintln!("Straight WR-90 waveguide:");
    for (path, label) in [
        ("tests/meshes/wg_tiny.msh", "tiny"),
        ("tests/meshes/wg_large.msh", "large"),
        ("tests/meshes/wg_xlarge.msh", "xlarge"),
    ] {
        run_waveguide_benchmark(path, label);
    }

    eprintln!("\nIris waveguide (d=15mm):");
    for (path, label) in [
        ("tests/meshes/iris_coarse.msh", "coarse"),
        ("tests/meshes/iris_fine.msh", "fine"),
    ] {
        run_waveguide_benchmark(path, label);
    }

    eprintln!("\n=== EMerge reference (SuperLU, same geometry) ===");
    eprintln!("  Straight WR-90 medium (600 tets, 4858 DOFs): 0.058s");
    eprintln!("  Straight WR-90 coarse (378 tets, 3198 DOFs): 1.304s (includes JIT warmup)");
}
