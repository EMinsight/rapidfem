use rapidfem_fd::abc_order2::abc_order_2_terms;

#[test]
fn test_abc_order_2_terms_vs_emerge() {
    // `abc_order_2_terms` now returns the *real* Lengths*(Curl-Div)*|Area|
    // operator without the `cf = j*c2/k0` scale (the caller applies it while
    // combining with the first-order mass and projecting to PSD). EMerge's
    // reference Frobenius norm was taken on `cf*R`, so divide it by |cf| to get
    // the expected norm of the bare real operator. This still validates the
    // ported curl/divergence basis against EMerge.
    let verts = [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.01]];
    let local_edge_map = [[0, 1], [1, 2], [0, 2]];
    let cf_mag = (-0.51555f64 / 209.58).abs();

    let mat = abc_order_2_terms(&verts, &local_edge_map);

    let norm: f64 = mat.iter().flat_map(|r| r.iter()).map(|x| x * x).sum::<f64>().sqrt();
    eprintln!("||R||_F = {:.15e}", norm);

    // EMerge reference for ||cf*R||_F was 1.518683085185049e-02.
    let ref_norm = 1.518683085185049e-02 / cf_mag;
    let err = (norm - ref_norm).abs() / ref_norm;
    eprintln!("Norm error vs EMerge: {:.2e}", err);

    // Edge ordering may permute individual entries, but the Frobenius norm is
    // order-invariant and must match EMerge to within 1%.
    assert!(err < 0.01, "ABC order-2 matrix norm error: {:.2e}", err);
    eprintln!("abc_order_2_terms: PASS (norm within 1%)");
}
