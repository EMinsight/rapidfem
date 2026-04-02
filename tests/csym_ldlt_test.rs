use faer::c64;
use rapidfem::csym_ldlt::CSymLdlt;

#[test]
fn test_csym_ldlt_2x2() {
    // A = [[2+1j, 1+0j], [1+0j, 3+2j]]  (complex-symmetric, NOT Hermitian)
    // Upper triangle CSC: col 0: (0, 2+1j), col 1: (0, 1+0j), (1, 3+2j)
    let col_ptr = vec![0, 1, 3];
    let row_idx = vec![0, 0, 1];
    let values = vec![
        c64 { re: 2.0, im: 1.0 },  // A[0,0]
        c64 { re: 1.0, im: 0.0 },  // A[0,1]
        c64 { re: 3.0, im: 2.0 },  // A[1,1]
    ];

    let ldlt = CSymLdlt::factorize(2, &col_ptr, &row_idx, &values).expect("LDLt failed");

    // Solve Ax = [1, 2]
    let rhs = vec![c64 { re: 1.0, im: 0.0 }, c64 { re: 2.0, im: 0.0 }];
    let x = ldlt.solve(&rhs);

    // Verify: Ax should equal rhs
    let ax0 = values[0] * x[0] + values[1] * x[1]; // A[0,0]*x[0] + A[0,1]*x[1]
    let ax1 = values[1] * x[0] + values[2] * x[1]; // A[1,0]*x[0] + A[1,1]*x[1]

    eprintln!("x = [{:.6}, {:.6}]", x[0], x[1]);
    eprintln!("Ax = [{:.6}, {:.6}]", ax0, ax1);
    eprintln!("rhs = [{:.6}, {:.6}]", rhs[0], rhs[1]);

    let err0 = (ax0 - rhs[0]).norm();
    let err1 = (ax1 - rhs[1]).norm();
    eprintln!("Residual: [{:.2e}, {:.2e}]", err0, err1);

    assert!(err0 < 1e-10, "Residual[0] too large: {}", err0);
    assert!(err1 < 1e-10, "Residual[1] too large: {}", err1);
    eprintln!("csym_ldlt 2x2: PASS");
}

#[test]
fn test_csym_ldlt_3x3() {
    // A = [[4+1j, 1-1j, 0], [1-1j, 5+2j, 2+1j], [0, 2+1j, 3-1j]]
    // Upper triangle CSC:
    // col 0: (0, 4+1j)
    // col 1: (0, 1-1j), (1, 5+2j)
    // col 2: (1, 2+1j), (2, 3-1j)
    let col_ptr = vec![0, 1, 3, 5];
    let row_idx = vec![0, 0, 1, 1, 2];
    let values = vec![
        c64 { re: 4.0, im: 1.0 },
        c64 { re: 1.0, im: -1.0 },
        c64 { re: 5.0, im: 2.0 },
        c64 { re: 2.0, im: 1.0 },
        c64 { re: 3.0, im: -1.0 },
    ];

    let ldlt = CSymLdlt::factorize(3, &col_ptr, &row_idx, &values).expect("LDLt failed");

    let rhs = vec![
        c64 { re: 1.0, im: 0.0 },
        c64 { re: 0.0, im: 1.0 },
        c64 { re: -1.0, im: 0.0 },
    ];
    let x = ldlt.solve(&rhs);

    // Verify Ax = rhs
    // A is symmetric, so we can compute full matrix-vector product
    let a = [
        [c64 { re: 4.0, im: 1.0 }, c64 { re: 1.0, im: -1.0 }, c64 { re: 0.0, im: 0.0 }],
        [c64 { re: 1.0, im: -1.0 }, c64 { re: 5.0, im: 2.0 }, c64 { re: 2.0, im: 1.0 }],
        [c64 { re: 0.0, im: 0.0 }, c64 { re: 2.0, im: 1.0 }, c64 { re: 3.0, im: -1.0 }],
    ];

    for i in 0..3 {
        let ax_i = a[i][0]*x[0] + a[i][1]*x[1] + a[i][2]*x[2];
        let err = (ax_i - rhs[i]).norm();
        eprintln!("  Residual[{}] = {:.2e}", i, err);
        assert!(err < 1e-10, "Residual[{}] too large: {}", i, err);
    }
    eprintln!("csym_ldlt 3x3: PASS");
}
