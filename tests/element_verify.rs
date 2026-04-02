/// Verify per-tet element matrices against EMerge reference values.
use num_complex::Complex64 as C64;
use rapidfem::tet_assembly::ned2_tet_stiff_mass;
use rapidfem::coefficients::VolumeCoeffCache;

#[test]
fn test_unit_tet_element_matrices() {
    let xs = [0.0, 1.0, 0.0, 0.0];
    let ys = [0.0, 0.0, 1.0, 0.0];
    let zs = [0.0, 0.0, 0.0, 1.0];

    let edge_lengths = [1.0, 1.0, 1.0, 2.0_f64.sqrt(), 2.0_f64.sqrt(), 2.0_f64.sqrt()];
    // Local edge map: (0,1),(0,2),(0,3),(1,2),(3,1),(2,3)
    let local_edge_map = [[0,1],[0,2],[0,3],[1,2],[3,1],[2,3]];
    // Local tri map: (0,1,2),(0,2,3),(0,3,1),(1,2,3)
    let local_tri_map = [[0,1,2],[0,2,3],[0,3,1],[1,2,3]];

    let identity: [[C64; 3]; 3] = [
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
    ];

    let vc = VolumeCoeffCache::new();
    let (dmat, fmat) = ned2_tet_stiff_mass(&xs, &ys, &zs, &edge_lengths, &local_edge_map, &local_tri_map, &identity, &identity, &vc);

    // EMerge reference values
    eprintln!("Dmat[0:6,0:6] (edge-edge stiffness):");
    for i in 0..6 {
        let row: Vec<String> = (0..6).map(|j| format!("{:10.6}", dmat[i][j].re)).collect();
        eprintln!("  [{}]", row.join(", "));
    }
    eprintln!("\nFmat[0:6,0:6] (edge-edge mass):");
    for i in 0..6 {
        let row: Vec<String> = (0..6).map(|j| format!("{:10.6}", fmat[i][j].re)).collect();
        eprintln!("  [{}]", row.join(", "));
    }

    // Compute Frobenius norms
    let d_norm: f64 = dmat.iter().flat_map(|r| r.iter()).map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    let f_norm: f64 = fmat.iter().flat_map(|r| r.iter()).map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    eprintln!("\n||Dmat||_F = {:.10e}", d_norm);
    eprintln!("||Fmat||_F = {:.10e}", f_norm);

    // Compare against EMerge reference
    let d_ref = 1.9311050377e+00;
    let f_ref = 6.4345700016e-02;
    let d_err = (d_norm - d_ref).abs() / d_ref;
    let f_err = (f_norm - f_ref).abs() / f_ref;
    eprintln!("\nDmat error: {:.2e}", d_err);
    eprintln!("Fmat error: {:.2e}", f_err);

    assert!(d_err < 1e-6, "Dmat Frobenius norm error too large: {}", d_err);
    assert!(f_err < 1e-6, "Fmat Frobenius norm error too large: {}", f_err);

    // Check specific entries against EMerge
    assert!((dmat[0][0].re - 0.300000).abs() < 1e-5, "Dmat[0,0] wrong: {}", dmat[0][0].re);
    assert!((dmat[0][1].re - (-0.150000)).abs() < 1e-5, "Dmat[0,1] wrong: {}", dmat[0][1].re);
    assert!((fmat[0][0].re - 0.009524).abs() < 1e-5, "Fmat[0,0] wrong: {}", fmat[0][0].re);
}
