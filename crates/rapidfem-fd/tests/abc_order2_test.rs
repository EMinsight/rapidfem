use num_complex::Complex64 as C64;
use rapidfem_fd::abc_order2::abc_order_2_terms;

#[test]
fn test_abc_order_2_terms_vs_emerge() {
    let verts = [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.01]];
    let local_edge_map = [[0, 1], [1, 2], [0, 2]];
    let cf = C64::new(0.0, -0.51555 / 209.58);

    let mat = abc_order_2_terms(&verts, &local_edge_map, cf);

    let norm: f64 = mat.iter().flat_map(|r| r.iter()).map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    eprintln!("||Mat||_F = {:.15e}", norm);

    // EMerge reference: 1.518683085185049e-02
    let ref_norm = 1.518683085185049e-02;
    let err = (norm - ref_norm).abs() / ref_norm;
    eprintln!("Norm error: {:.2e}", err);

    // Note: our edge ordering may differ from EMerge's local_edge_map convention,
    // so individual entries may be permuted. But the Frobenius norm should match.
    assert!(err < 0.01, "ABC order-2 matrix norm error: {:.2e}", err);
    eprintln!("abc_order_2_terms: PASS (norm within 1%)");
}
