//! Top-level system assembly and solve.
//! Mirrors assembler.py: assemble_freq_matrix + solve.

use num_complex::Complex64 as C64;
use sprs::{CsMat, TriMat};
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::waveguide::RectWaveguide;
use crate::tet_assembly::assemble_global_matrices;
use crate::tri_assembly::{ned2_tri_stiff, ned2_tri_force};
use crate::coefficients::AreaCoeffCache;
use crate::quadrature::gaus_quad_tri;
use crate::constants::PI;

/// Result of a frequency-domain solve.
pub struct SolveResult {
    /// Solution vectors: one per port excitation.
    pub solutions: Vec<Vec<C64>>,
    /// System matrix K (for field interpolation later).
    pub n_field: usize,
}

/// Assemble and solve the frequency-domain system at one frequency.
///
/// Steps:
/// 1. Assemble E (curl-curl) and B (mass) matrices
/// 2. Form K = E - k₀²·B
/// 3. Add Robin BC (port impedance) to K
/// 4. Build port excitation vectors
/// 5. Apply PEC boundary conditions
/// 6. Solve K·x = b for each port
pub fn assemble_and_solve(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    ports: &[RectWaveguide],
    port_tri_indices: &[Vec<usize>], // per-port list of triangle indices
    pec_tri_indices: &[usize],       // PEC surface triangle indices
    freq: f64,
    _er_global: Option<&[[[C64; 3]; 3]]>,
) -> SolveResult {
    let c0 = crate::constants::C0;
    let k0 = 2.0 * PI * freq / c0;
    let n_field = basis.n_field;
    let n_tets = mesh.n_tets();

    eprintln!("Assembly at f={:.3e} Hz, k0={:.4}, n_field={}", freq, k0, n_field);

    // Material tensors (default: air)
    let identity: [[C64; 3]; 3] = [
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
    ];
    let er: Vec<_> = vec![identity; n_tets];
    let ur: Vec<_> = vec![identity; n_tets];

    // Step 1: Assemble E and B
    let t0 = std::time::Instant::now();
    let (rows, cols, data_e, data_b) = assemble_global_matrices(mesh, basis, &er, &ur);
    eprintln!("  Assembled E,B in {:.1}ms ({} nnz)", t0.elapsed().as_secs_f64()*1e3, rows.len());

    // Step 2: K = E - k0² * B
    let k0_sq = C64::from(k0 * k0);
    let mut tri_k = TriMat::new((n_field, n_field));
    for i in 0..rows.len() {
        tri_k.add_triplet(rows[i], cols[i], data_e[i] - k0_sq * data_b[i]);
    }

    // Step 3: Add Robin BC for each port
    let ac_base = AreaCoeffCache::new();
    let mut port_excitations: Vec<Vec<C64>> = Vec::new();
    let quad_pts = gaus_quad_tri(4); // order 4 = 6 points

    for (pi, port) in ports.iter().enumerate() {
        let gamma = port.gamma(k0);
        let tri_ids = &port_tri_indices[pi];

        // Robin BC stiffness: add 8×8 per triangle to K
        for &ti in tri_ids {
            let tri = &mesh.tris[ti];
            let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];
            let bmat = ned2_tri_stiff(&verts, gamma, &ac_base);
            let dofs = &basis.tri_to_field[ti];
            for ii in 0..8 {
                for jj in 0..8 {
                    tri_k.add_triplet(dofs[ii], dofs[jj], bmat[ii][jj]);
                }
            }
        }

        // Port excitation vector
        let mut bvec = vec![C64::new(0.0, 0.0); n_field];
        for &ti in tri_ids {
            let tri = &mesh.tris[ti];
            let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];

            // Evaluate U_inc at triangle vertices
            let u_inc: [[C64; 3]; 3] = std::array::from_fn(|i| {
                port.u_inc_global(verts[i][0], verts[i][1], verts[i][2], k0)
            });

            let force = ned2_tri_force(&verts, &u_inc, &quad_pts);
            let dofs = &basis.tri_to_field[ti];
            for i in 0..8 {
                bvec[dofs[i]] += force[i];
            }
        }
        let bvec_norm: f64 = bvec.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        port_excitations.push(bvec);
        eprintln!("  Port {}: gamma={:.4e}, {} tris, ||b||={:.4e}",
            pi, gamma, tri_ids.len(), bvec_norm);
    }

    // Step 5: Identify PEC DOFs
    let mut pec_dofs = std::collections::HashSet::new();
    for &ti in pec_tri_indices {
        let dofs = &basis.tri_to_field[ti];
        for &d in dofs {
            pec_dofs.insert(d);
        }
    }
    // Also add edge DOFs on PEC faces
    for &ti in pec_tri_indices {
        let edges = &mesh.tri_to_edge[ti];
        for &ei in edges {
            let edofs = &basis.edge_to_field[ei];
            for &d in edofs {
                pec_dofs.insert(d);
            }
        }
    }
    eprintln!("  PEC DOFs: {} of {}", pec_dofs.len(), n_field);
    for (pi, bvec) in port_excitations.iter().enumerate() {
        let b_free_norm: f64 = bvec.iter().enumerate()
            .filter(|(i, _)| !pec_dofs.contains(i))
            .map(|(_, x)| x.norm_sqr()).sum::<f64>().sqrt();
        eprintln!("  Port {} ||b_free|| = {:.4e}", pi, b_free_norm);
    }

    // Apply PEC: zero rows/cols, set diagonal to 1
    // First build CSR, then modify
    let k_csr: CsMat<C64> = tri_k.to_csr();

    // For the direct solve, we'll build a dense system (for now — small problems)
    // TODO: Use sparse LU via faer for larger problems
    let free_dofs: Vec<usize> = (0..n_field).filter(|d| !pec_dofs.contains(d)).collect();
    let n_free = free_dofs.len();
    eprintln!("  Free DOFs: {}", n_free);

    // Build reduced dense system
    let mut dof_to_free = vec![usize::MAX; n_field];
    for (fi, &d) in free_dofs.iter().enumerate() {
        dof_to_free[d] = fi;
    }

    // Extract reduced K matrix (dense)
    let mut k_dense = vec![C64::new(0.0, 0.0); n_free * n_free];
    for (row_idx, row_vec) in k_csr.outer_iterator().enumerate() {
        if pec_dofs.contains(&row_idx) { continue; }
        let fi = dof_to_free[row_idx];
        for (col_idx, &val) in row_vec.iter() {
            if pec_dofs.contains(&col_idx) { continue; }
            let fj = dof_to_free[col_idx];
            k_dense[fi * n_free + fj] += val;
        }
    }

    // Step 6: Solve for each port excitation
    // Dense LU factorization
    let t_solve = std::time::Instant::now();

    // Use simple Gaussian elimination with partial pivoting
    let mut solutions = Vec::new();
    let lu = dense_lu_factor(&k_dense, n_free);

    for (pi, bvec) in port_excitations.iter().enumerate() {
        // Extract reduced RHS
        let b_free: Vec<C64> = free_dofs.iter().map(|&d| bvec[d]).collect();
        let x_free = dense_lu_solve(&lu, &b_free, n_free);

        // Scatter back to full solution
        let mut x_full = vec![C64::new(0.0, 0.0); n_field];
        for (fi, &d) in free_dofs.iter().enumerate() {
            x_full[d] = x_free[fi];
        }
        solutions.push(x_full);
        eprintln!("  Solved port {} in {:.1}ms", pi, t_solve.elapsed().as_secs_f64()*1e3);
    }

    SolveResult { solutions, n_field }
}

// ========================================================================
// Dense LU solver (temporary — will be replaced with sparse LU)
// ========================================================================

struct DenseLU {
    a: Vec<C64>,
    piv: Vec<usize>,
    n: usize,
}

fn dense_lu_factor(a: &[C64], n: usize) -> DenseLU {
    let mut a = a.to_vec();
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Partial pivoting
        let mut max_val = a[k * n + k].norm();
        let mut max_row = k;
        for i in (k+1)..n {
            let v = a[i * n + k].norm();
            if v > max_val { max_val = v; max_row = i; }
        }
        if max_row != k {
            piv.swap(k, max_row);
            for j in 0..n { a.swap(k * n + j, max_row * n + j); }
        }

        let akk = a[k * n + k];
        if akk.norm() < 1e-30 { continue; }

        for i in (k+1)..n {
            let factor = a[i * n + k] / akk;
            a[i * n + k] = factor;
            for j in (k+1)..n {
                let akj = a[k * n + j];
                a[i * n + j] -= factor * akj;
            }
        }
    }

    DenseLU { a, piv, n }
}

fn dense_lu_solve(lu: &DenseLU, b: &[C64], n: usize) -> Vec<C64> {
    let mut x: Vec<C64> = lu.piv.iter().map(|&i| b[i]).collect();

    // Forward substitution (L)
    for i in 0..n {
        for j in 0..i {
            let lij = lu.a[i * n + j];
            let xj = x[j];
            x[i] -= lij * xj;
        }
    }

    // Back substitution (U)
    for i in (0..n).rev() {
        for j in (i+1)..n {
            let uij = lu.a[i * n + j];
            let xj = x[j];
            x[i] -= uij * xj;
        }
        x[i] /= lu.a[i * n + i];
    }

    x
}
