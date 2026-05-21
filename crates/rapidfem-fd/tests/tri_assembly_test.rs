/// Unit tests for tri_assembly.rs against EMerge reference values.
/// Tests ned2_tri_stiff and ned2_tri_force with known inputs.

use num_complex::Complex64 as C64;
use rapidfem_fd::tri_assembly::{ned2_tri_stiff, ned2_tri_force};
use rapidfem_fd::coefficients::AreaCoeffCache;
use rapidfem_fd::quadrature::gaus_quad_tri;

/// Reference values from EMerge for vertices:
/// v0 = (0, 0, 0), v1 = (0.02, 0, 0), v2 = (0, 0, 0.01)
/// gamma = 158.238j
#[test]
fn test_ned2_tri_stiff_vs_emerge() {
    let verts = [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.01]];
    let gamma = C64::new(0.0, 158.238);
    let ac = AreaCoeffCache::new();

    let bmat = ned2_tri_stiff(&verts, gamma, &ac);

    let norm: f64 = bmat.iter().flat_map(|r| r.iter()).map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    eprintln!("||Bmat||_F = {:.15e}", norm);

    // EMerge reference: 1.287056141671664e-02
    let ref_norm = 1.287056141671664e-02;
    let err = (norm - ref_norm).abs() / ref_norm;
    eprintln!("Norm error: {:.2e}", err);
    assert!(err < 1e-10, "Bmat Frobenius norm error: {:.2e}", err);

    // Spot check specific entries
    let check = |i: usize, j: usize, re: f64, im: f64| {
        let expected = C64::new(re, im);
        let got = bmat[i][j];
        let e = (got - expected).norm();
        if e > 1e-12 {
            eprintln!("MISMATCH Bmat[{},{}]: got {:.6e}+{:.6e}j, expected {:.6e}+{:.6e}j, err={:.2e}",
                i, j, got.re, got.im, re, im, e);
        }
        assert!(e < 1e-10, "Bmat[{},{}] error: {:.2e}", i, j, e);
    };

    check(0, 0, 0.0, 2.461480000000000e-03);
    check(0, 3, 0.0, -6.153700000000002e-04);
    check(3, 3, 0.0, 5.714150000000001e-04);
    check(4, 4, 0.0, 5.977880000000000e-03);
    check(7, 7, 0.0, 1.230740000000000e-03);

    eprintln!("ned2_tri_stiff: PASS");
}

/// Reference values from EMerge for same triangle with uniform Ez=1000j
#[test]
fn test_ned2_tri_force_vs_emerge() {
    let verts = [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.01]];
    let quad_pts = gaus_quad_tri(4);

    // Uniform U_inc = (0, 0, 1000j) at all quad points
    let glob_uinc: Vec<[C64; 3]> = quad_pts.iter()
        .map(|_| [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 1000.0)])
        .collect();

    let bvec = ned2_tri_force(&verts, &glob_uinc, &quad_pts);

    let norm: f64 = bvec.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    eprintln!("||bvec|| = {:.15e}", norm);

    // EMerge reference: 7.021791477646963e-02
    let ref_norm = 7.021791477646963e-02;
    let err = (norm - ref_norm).abs() / ref_norm;
    eprintln!("Norm error: {:.2e}", err);
    assert!(err < 1e-10, "bvec norm error: {:.2e}", err);

    // Check each entry against EMerge
    let ref_vals: [f64; 8] = [
        1.666666666666664e-02,
        3.726779962499651e-02,
        2.499999999999997e-02,
       -1.666666666666664e-02,
        3.333333333333335e-02,
        1.863389981249823e-02,
        2.499999999999999e-02,
        1.666666666666666e-02,
    ];

    for (i, &ref_im) in ref_vals.iter().enumerate() {
        let expected = C64::new(0.0, ref_im);
        let got = bvec[i];
        let e = (got - expected).norm();
        if e > 1e-12 {
            eprintln!("MISMATCH bvec[{}]: got {:.6e}+{:.6e}j, expected 0+{:.6e}j, err={:.2e}",
                i, got.re, got.im, ref_im, e);
        }
        assert!(e < 1e-10, "bvec[{}] error: {:.2e}", i, e);
    }

    eprintln!("ned2_tri_force: PASS");
}
