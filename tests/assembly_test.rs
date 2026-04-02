use num_complex::Complex64 as C64;
use rapidfem::mesh_io::load_mesh;
use rapidfem::basis::Nedelec2Basis;
use rapidfem::tet_assembly::assemble_global_matrices;

#[test]
fn test_global_assembly_produces_matrices() {
    let mesh = load_mesh("tests/meshes/wr90_straight.msh").expect("Load mesh");
    let basis = Nedelec2Basis::new(&mesh);

    // Air-filled: εr = I, μr = I for all tets
    let identity: [[C64; 3]; 3] = [
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
    ];
    let n_tets = mesh.n_tets();
    let er: Vec<_> = vec![identity; n_tets];
    let ur: Vec<_> = vec![identity; n_tets];

    let (rows, cols, data_e, data_b) = assemble_global_matrices(&mesh, &basis, &er, &ur);

    let expected_nnz = n_tets * 400;
    assert_eq!(rows.len(), expected_nnz);
    assert_eq!(cols.len(), expected_nnz);
    assert_eq!(data_e.len(), expected_nnz);
    assert_eq!(data_b.len(), expected_nnz);

    // All DOF indices should be within range
    let n_field = basis.n_field;
    for &r in &rows { assert!(r < n_field, "Row {} >= n_field {}", r, n_field); }
    for &c in &cols { assert!(c < n_field, "Col {} >= n_field {}", c, n_field); }

    // Stiffness matrix should have non-zero entries
    let e_norm: f64 = data_e.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    let b_norm: f64 = data_b.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    assert!(e_norm > 0.0, "Stiffness matrix is zero");
    assert!(b_norm > 0.0, "Mass matrix is zero");

    eprintln!("Global assembly: {} DOFs, {} nnz entries", n_field, expected_nnz);
    eprintln!("  ||E||_F = {:.6e}", e_norm);
    eprintln!("  ||B||_F = {:.6e}", b_norm);
}
