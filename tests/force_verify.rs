use num_complex::Complex64 as C64;
use rapidfem::tri_assembly::ned2_tri_force;
use rapidfem::quadrature::gaus_quad_tri;

#[test]
fn test_force_vector_uniform_field() {
    // Right triangle in XZ plane: (0,0,0), (0.02,0,0), (0,0,0.01)
    let verts = [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.01]];
    let quad_pts = gaus_quad_tri(4);

    // Uniform U_inc = (0, 0, 1000j) at all quad points
    let u_inc_at_qp: Vec<[C64; 3]> = quad_pts.iter()
        .map(|_| [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 1000.0)])
        .collect();

    let bvec = ned2_tri_force(&verts, &u_inc_at_qp, &quad_pts);

    let norm: f64 = bvec.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    eprintln!("||bvec|| = {:.10e}", norm);
    for (i, v) in bvec.iter().enumerate() {
        eprintln!("  bvec[{}] = {:.8e} + {:.8e}j", i, v.re, v.im);
    }

    // EMerge reference: ||bvec|| = 7.0710678119e-02
    let ref_norm = 7.0710678119e-02;
    let err = (norm - ref_norm).abs() / ref_norm;
    eprintln!("Error vs EMerge: {:.2e}", err);
    assert!(err < 0.01, "Force vector norm error: {:.2e}", err);
}
