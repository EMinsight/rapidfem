/// Verify Robin BC triangle stiffness against EMerge reference.
use num_complex::Complex64 as C64;
use rapidfem::tri_assembly::ned2_tri_stiff;
use rapidfem::coefficients::AreaCoeffCache;

#[test]
fn test_robin_bc_triangle_stiffness() {
    // Test triangle: right triangle in XZ plane at y=0
    // Vertices: (0,0,0), (22.86mm,0,0), (0,0,10.16mm)
    let verts = [
        [0.0, 0.0, 0.0],
        [22.86e-3, 0.0, 0.0],
        [0.0, 0.0, 10.16e-3],
    ];

    let gamma = C64::new(0.0, 158.238);
    let ac = AreaCoeffCache::new();
    let bmat = ned2_tri_stiff(&verts, gamma, &ac);

    // Frobenius norm
    let norm: f64 = bmat.iter().flat_map(|r| r.iter()).map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    eprintln!("||Bmat||_F = {:.10e}", norm);

    // EMerge reference: ||Bmat||_F = 1.7795983140e-02
    let ref_norm = 1.7795983140e-02;
    let err = (norm - ref_norm).abs() / ref_norm;
    eprintln!("Error vs EMerge: {:.2e}", err);

    eprintln!("\nBmat[0:4,0:4]:");
    for i in 0..4 {
        let row: Vec<String> = (0..4).map(|j| format!("{:.6e}+{:.6e}j", bmat[i][j].re, bmat[i][j].im)).collect();
        eprintln!("  [{}]", row.join(", "));
    }

    // EMerge reference entries
    assert!((bmat[0][0].im - 8.243673e-03).abs() < 1e-6,
        "Bmat[0,0] wrong: {:.6e}+{:.6e}j", bmat[0][0].re, bmat[0][0].im);

    assert!(err < 1e-4, "Robin BC norm error too large: {:.2e}", err);
}
