/// Unit test for tet_assembly.rs against EMerge reference values.
use num_complex::Complex64 as C64;
use rapidfem::tet_assembly::ned2_tet_stiff_mass;
use rapidfem::coefficients::VolumeCoeffCache;

#[test]
fn test_ned2_tet_stiff_mass_vs_emerge() {
    // Unit tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    let xs = [0.0, 1.0, 0.0, 0.0];
    let ys = [0.0, 0.0, 1.0, 0.0];
    let zs = [0.0, 0.0, 0.0, 1.0];
    let s2 = 2.0_f64.sqrt();
    let edge_lengths = [1.0, 1.0, 1.0, s2, s2, s2];
    let local_edge_map = [[0,1],[0,2],[0,3],[1,2],[3,1],[2,3]];
    let local_tri_map = [[0,1,2],[0,2,3],[0,3,1],[1,2,3]];
    let identity: [[C64; 3]; 3] = [
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
    ];
    let vc = VolumeCoeffCache::new();
    let (dmat, fmat) = ned2_tet_stiff_mass(&xs, &ys, &zs, &edge_lengths,
        &local_edge_map, &local_tri_map, &identity, &identity, &vc);

    let d_norm: f64 = dmat.iter().flat_map(|r| r.iter()).map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    let f_norm: f64 = fmat.iter().flat_map(|r| r.iter()).map(|x| x.norm_sqr()).sum::<f64>().sqrt();

    let d_ref = 1.931105037709411e+00;
    let f_ref = 6.434570001645180e-02;

    let d_err = (d_norm - d_ref).abs() / d_ref;
    let f_err = (f_norm - f_ref).abs() / f_ref;
    eprintln!("||Dmat||_F = {:.15e}, err = {:.2e}", d_norm, d_err);
    eprintln!("||Fmat||_F = {:.15e}, err = {:.2e}", f_norm, f_err);

    assert!(d_err < 1e-10, "Dmat error: {:.2e}", d_err);
    assert!(f_err < 1e-10, "Fmat error: {:.2e}", f_err);
    eprintln!("ned2_tet_stiff_mass: PASS");
}
