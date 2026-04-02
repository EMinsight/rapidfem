//! Eigenmode solver: find resonant frequencies and Q-factors of cavities.
//!
//! Solves the generalized eigenvalue problem E*x = λ*B*x where:
//! - E = curl-curl stiffness matrix
//! - B = mass matrix
//! - λ = k₀² (squared wavenumber)
//! - f = c₀*√λ / (2π) (resonant frequency)
//!
//! Uses shift-invert Lanczos iteration with PARDISO or faer for the shift-invert solve.

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::tet_assembly::assemble_global_matrices;
use crate::constants::*;
use std::collections::HashSet;

/// Result of eigenmode analysis.
pub struct Eigenmode {
    /// Complex resonant frequency (Hz). Imaginary part indicates loss.
    pub frequency: C64,
    /// Quality factor Q = 0.5 * Re(f) / Im(f). Infinite for lossless modes.
    pub q_factor: f64,
    /// Eigenvalue λ = k₀²
    pub eigenvalue: C64,
    /// Eigenvector (field DOF values)
    pub field: Vec<C64>,
}

/// Solve eigenmode problem using shift-invert Lanczos.
///
/// Finds eigenvalues of E*x = λ*B*x near target_freq.
pub fn solve_eigenmode(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    pec_tri_indices: &[usize],
    materials: Option<&[crate::materials::Material]>,
    target_freq: f64,
    n_modes: usize,
) -> Vec<Eigenmode> {
    let n_tets = mesh.n_tets();
    let n_field = basis.n_field;

    // Assemble E and B matrices
    let (er, ur) = if let Some(mats) = materials {
        crate::materials::build_material_tensors(n_tets, mats, target_freq)
    } else {
        let identity: [[C64; 3]; 3] = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
        ];
        (vec![identity; n_tets], vec![identity; n_tets])
    };

    let t0 = std::time::Instant::now();
    let (rows, cols, data_e, data_b) = assemble_global_matrices(mesh, basis, &er, &ur);
    eprintln!("  Eigenmode: assembled E,B in {:.1}ms", t0.elapsed().as_secs_f64()*1e3);

    // PEC DOFs
    let mut pec_ids: HashSet<usize> = HashSet::new();
    for &ti in pec_tri_indices {
        for &ei in &mesh.tri_to_edge[ti] {
            for &d in &basis.edge_to_field[ei] { pec_ids.insert(d); }
        }
        for &d in &basis.tri_to_field[ti] { pec_ids.insert(d); }
    }

    let free_dofs: Vec<usize> = (0..n_field).filter(|d| !pec_ids.contains(d)).collect();
    let n_free = free_dofs.len();
    let mut dof_to_free = vec![usize::MAX; n_field];
    for (fi, &d) in free_dofs.iter().enumerate() { dof_to_free[d] = fi; }

    eprintln!("  Eigenmode: {} free DOFs, target={:.4e} Hz", n_free, target_freq);

    // Shift: σ = (2πf/c)²
    let k0_target = 2.0 * PI * target_freq / C0;
    let sigma = C64::from(k0_target * k0_target);

    // Build (E - σB) as faer sparse for shift-invert
    let t1 = std::time::Instant::now();
    let mut triplets: Vec<faer::sparse::Triplet<usize, usize, faer::c64>> = Vec::new();
    for i in 0..rows.len() {
        let r = rows[i]; let c = cols[i];
        if pec_ids.contains(&r) || pec_ids.contains(&c) { continue; }
        let val = data_e[i] - sigma * data_b[i];
        triplets.push(faer::sparse::Triplet {
            row: dof_to_free[r], col: dof_to_free[c],
            val: faer::c64 { re: val.re, im: val.im },
        });
    }

    let shift_mat = faer::sparse::SparseColMat::<usize, faer::c64>::try_new_from_triplets(
        n_free, n_free, &triplets,
    ).expect("faer matrix");
    let lu = shift_mat.sp_lu().expect("LU factorization failed");
    eprintln!("  Eigenmode: shift-invert LU in {:.1}ms", t1.elapsed().as_secs_f64()*1e3);

    // Also need B matrix for the Lanczos iteration: y = (E-σB)⁻¹ * B * x
    // Build B as a sparse operator
    let mut b_triplets: Vec<faer::sparse::Triplet<usize, usize, faer::c64>> = Vec::new();
    for i in 0..rows.len() {
        let r = rows[i]; let c = cols[i];
        if pec_ids.contains(&r) || pec_ids.contains(&c) { continue; }
        if data_b[i].re == 0.0 && data_b[i].im == 0.0 { continue; }
        b_triplets.push(faer::sparse::Triplet {
            row: dof_to_free[r], col: dof_to_free[c],
            val: faer::c64 { re: data_b[i].re, im: data_b[i].im },
        });
    }
    let b_mat = faer::sparse::SparseColMat::<usize, faer::c64>::try_new_from_triplets(
        n_free, n_free, &b_triplets,
    ).expect("B matrix");

    // Shift-invert Lanczos iteration
    let t2 = std::time::Instant::now();
    let n_lanczos = (2 * n_modes + 10).min(n_free);

    // Random starting vector
    let mut v = faer::Mat::<faer::c64>::from_fn(n_free, 1, |i, _| {
        faer::c64 { re: ((i * 7 + 13) % 97) as f64 / 97.0, im: 0.0 }
    });
    // Normalize
    let norm: f64 = (0..n_free).map(|i| { let x = v[(i,0)]; x.re*x.re + x.im*x.im }).sum::<f64>().sqrt();
    for i in 0..n_free { let s = 1.0/norm; v[(i,0)] = faer::c64 { re: v[(i,0)].re * s, im: 0.0 }; }

    // Lanczos vectors and tridiagonal
    let mut vecs: Vec<Vec<faer::c64>> = Vec::new();
    let mut alphas: Vec<faer::c64> = Vec::new();
    let mut betas: Vec<faer::c64> = Vec::new();

    let mut v_prev = faer::Mat::<faer::c64>::zeros(n_free, 1);
    let mut beta_prev = faer::c64 { re: 0.0, im: 0.0 };

    for j in 0..n_lanczos {
        // Store current vector
        let v_j: Vec<faer::c64> = (0..n_free).map(|i| v[(i,0)]).collect();
        vecs.push(v_j);

        // w = (E - σB)⁻¹ * B * v
        // Step 1: Bv = B * v
        let mut bv = faer::Mat::<faer::c64>::zeros(n_free, 1);
        // Sparse mat-vec: bv = B * v using sprs-style iteration
        {
            let b_ref = b_mat.as_ref();
            let col_ptrs = b_ref.symbolic().col_ptr();
            let row_indices = b_ref.symbolic().row_idx();
            let values = b_ref.val();
            for j in 0..n_free {
                let vj = v[(j, 0)];
                if vj.re == 0.0 && vj.im == 0.0 { continue; }
                let start = col_ptrs[j];
                let end = col_ptrs[j + 1];
                for p in start..end {
                    let i = row_indices[p];
                    let val = values[p];
                    bv[(i, 0)] = faer::c64 {
                        re: bv[(i, 0)].re + val.re * vj.re - val.im * vj.im,
                        im: bv[(i, 0)].im + val.re * vj.im + val.im * vj.re,
                    };
                }
            }
        }

        // Step 2: w = (E - σB)⁻¹ * Bv
        let mut w = bv;
        faer::linalg::solvers::SolveCore::solve_in_place_with_conj(&lu, faer::Conj::No, w.as_mut());

        // α = v^H * w (Hermitian inner product for Lanczos)
        let mut alpha = faer::c64 { re: 0.0, im: 0.0 };
        for i in 0..n_free {
            let vi = v[(i,0)];
            let wi = w[(i,0)];
            // v^H * w = conj(v) * w
            alpha.re += vi.re * wi.re + vi.im * wi.im;
            alpha.im += vi.re * wi.im - vi.im * wi.re;
        }
        alphas.push(alpha);

        // w = w - α*v - β*v_prev
        for i in 0..n_free {
            let vi = v[(i,0)];
            let vpi = v_prev[(i,0)];
            w[(i,0)] = faer::c64 {
                re: w[(i,0)].re - (alpha.re * vi.re - alpha.im * vi.im) - (beta_prev.re * vpi.re - beta_prev.im * vpi.im),
                im: w[(i,0)].im - (alpha.re * vi.im + alpha.im * vi.re) - (beta_prev.re * vpi.im + beta_prev.im * vpi.re),
            };
        }

        // β = ||w||
        let w_norm: f64 = (0..n_free).map(|i| { let x = w[(i,0)]; x.re*x.re + x.im*x.im }).sum::<f64>().sqrt();
        let beta = faer::c64 { re: w_norm, im: 0.0 };
        betas.push(beta);

        if w_norm < 1e-14 { break; }

        // v_prev = v, v = w / β
        v_prev = v.clone();
        beta_prev = beta;
        v = faer::Mat::<faer::c64>::from_fn(n_free, 1, |i, _| {
            faer::c64 { re: w[(i,0)].re / w_norm, im: w[(i,0)].im / w_norm }
        });
    }

    let m = alphas.len();
    eprintln!("  Eigenmode: {} Lanczos iterations in {:.1}ms", m, t2.elapsed().as_secs_f64()*1e3);

    // Build tridiagonal matrix T and solve dense eigenvalue problem
    // T[i,i] = alphas[i], T[i,i+1] = betas[i], T[i+1,i] = betas[i]
    // The eigenvalues μ of T approximate eigenvalues of (E-σB)⁻¹B
    // Then λ = σ + 1/μ
    let mut t_dense = vec![C64::new(0.0, 0.0); m * m];
    for i in 0..m {
        t_dense[i * m + i] = C64::new(alphas[i].re, alphas[i].im);
        if i + 1 < m {
            t_dense[i * m + (i+1)] = C64::new(betas[i].re, betas[i].im);
            t_dense[(i+1) * m + i] = C64::new(betas[i].re, betas[i].im);
        }
    }

    // Dense eigenvalue decomposition using faer
    let t_mat = faer::Mat::<faer::c64>::from_fn(m, m, |i, j| {
        faer::c64 { re: t_dense[i * m + j].re, im: t_dense[i * m + j].im }
    });

    let eig = t_mat.eigen().expect("Eigendecomposition failed");
    let eigenvalues = eig.S().column_vector();
    let eigenvectors = eig.U();

    // Convert shift-invert eigenvalues to original eigenvalues
    // μ_i are eigenvalues of (E-σB)⁻¹B, so λ_i = σ + 1/μ_i
    let mut modes: Vec<(C64, Vec<C64>)> = Vec::new();

    for k in 0..m {
        let mu = C64::new(eigenvalues[k].re, eigenvalues[k].im);
        if mu.norm() < 1e-30 { continue; }
        let lambda = sigma + C64::new(1.0, 0.0) / mu;

        // Only keep eigenvalues with positive real part (physical modes)
        if lambda.re <= 0.0 { continue; }

        // Compute Ritz vector: x = V * y where y is the eigenvector of T
        let mut x_free = vec![C64::new(0.0, 0.0); n_free];
        for j in 0..m.min(vecs.len()) {
            let coeff = C64::new(eigenvectors[(j, k)].re, eigenvectors[(j, k)].im);
            for i in 0..n_free {
                let vji = C64::new(vecs[j][i].re, vecs[j][i].im);
                x_free[i] += coeff * vji;
            }
        }

        // Map back to full DOF vector
        let mut x_full = vec![C64::new(0.0, 0.0); n_field];
        for (fi, &d) in free_dofs.iter().enumerate() {
            x_full[d] = x_free[fi];
        }

        modes.push((lambda, x_full));
    }

    // Sort by distance to target eigenvalue
    modes.sort_by(|a, b| {
        let da = (a.0 - sigma).norm();
        let db = (b.0 - sigma).norm();
        da.partial_cmp(&db).unwrap()
    });

    // Convert to Eigenmode structs
    modes.into_iter().take(n_modes).map(|(lambda, field)| {
        let k0 = lambda.sqrt();
        let freq = k0 * C64::from(C0 / (2.0 * PI));
        let q = if freq.im.abs() > 1e-30 { 0.5 * freq.re / freq.im.abs() } else { f64::INFINITY };
        Eigenmode { frequency: freq, q_factor: q, eigenvalue: lambda, field }
    }).collect()
}
