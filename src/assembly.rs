//! Exact port of assembler.py: assemble_freq_matrix + solve pipeline.
//!
//! Follows EMerge's assembly order exactly:
//! 1. E, B = tet_mass_stiffness_matrices
//! 2. K = (E - k0² * B).tocsr()
//! 3. PEC: collect DOFs from edge_to_field and tri_to_field for PEC faces
//! 4. Robin: Bempty = empty_tri_matrix(); compute_bc_entries; K += generate_csr(Bempty)
//! 5. Port vectors: assemble_robin_bc_bvec (generate_points_3d + compute_force_entries)
//! 6. Eliminate PEC DOFs, solve K*x = b

use num_complex::Complex64 as C64;
use sprs::{CsMat, CsMatView};
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::port::Port;
use crate::tet_assembly::assemble_global_matrices;
use crate::tri_assembly::{ned2_tri_stiff, ned2_tri_force};
use crate::coefficients::AreaCoeffCache;
use crate::quadrature::gaus_quad_tri;
use crate::constants::PI;
use std::collections::HashSet;

pub struct SolveResult {
    pub solutions: Vec<Vec<C64>>,
    pub n_field: usize,
}

/// Exact port of assembler.py:assemble_freq_matrix + solve.
/// Now accepts any Port type via trait objects.
pub fn assemble_and_solve(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    ports: &[&dyn Port],
    port_tri_indices: &[&[usize]],
    pec_tri_indices: &[usize],
    freq: f64,
    materials: Option<&[crate::materials::Material]>,
) -> SolveResult {
    let c0 = crate::constants::C0;
    let k0 = 2.0 * PI * freq / c0;
    let n_field = basis.n_field;
    let n_tets = mesh.n_tets();

    // Step 1: Build material tensors (exact port of assembler.py lines 280-303)
    let (er, ur) = if let Some(mats) = materials {
        crate::materials::build_material_tensors(n_tets, mats, freq)
    } else {
        // Default: air (identity tensors)
        let identity: [[C64; 3]; 3] = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
        ];
        (vec![identity; n_tets], vec![identity; n_tets])
    };

    let t0 = std::time::Instant::now();
    let (rows, cols, data_e, data_b) = assemble_global_matrices(mesh, basis, &er, &ur);
    eprintln!("  Assembled E,B in {:.1}ms ({} entries)", t0.elapsed().as_secs_f64()*1e3, rows.len());

    // Step 2: K = (E - B * k0²).tocsr()
    let k0_sq = C64::from(k0 * k0);
    let data_k: Vec<C64> = data_e.iter().zip(data_b.iter())
        .map(|(e, b)| e - k0_sq * b)
        .collect();

    // Build CSR from COO
    let mut k_csr: CsMat<C64> = {
        let mut tri_mat = sprs::TriMat::new((n_field, n_field));
        for i in 0..rows.len() {
            tri_mat.add_triplet(rows[i], cols[i], data_k[i]);
        }
        tri_mat.to_csr()
    };

    // Step 3: PEC DOFs — exact port of assembler.py lines 356-373
    let mut pec_ids: HashSet<usize> = HashSet::new();

    for &ti in pec_tri_indices {
        // edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
        let edges = &mesh.tri_to_edge[ti];
        for &ei in edges {
            // eids = field.edge_to_field[:, ii]
            let edofs = &basis.edge_to_field[ei];
            for &d in edofs {
                pec_ids.insert(d);
            }
        }
        // tids = field.tri_to_field[:, ii]
        let tdofs = &basis.tri_to_field[ti];
        for &d in tdofs {
            pec_ids.insert(d);
        }
    }
    eprintln!("  PEC DOFs: {} of {}", pec_ids.len(), n_field);

    // Step 4: Robin BC — exact port of assembler.py lines 380-413
    // Uses EMerge's flat array mechanism: Bempty + compute_bc_entries + generate_csr
    let ac_base = AreaCoeffCache::new();
    let gauss_points = gaus_quad_tri(4);

    // Bempty = field.empty_tri_matrix()
    let mut bempty = basis.empty_tri_matrix();

    for (pi, (port, tri_ids)) in ports.iter().zip(port_tri_indices.iter()).enumerate() {
        let gamma = port.get_gamma(k0);

        // Robin BC stiffness: for each port tri, compute 8x8 and write into flat array
        for &ti in *tri_ids {
            let tri = &mesh.tris[ti];
            let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];
            let bsub = ned2_tri_stiff(&verts, gamma, &ac_base);
            let p = ti * 64;
            for ii in 0..8 {
                for jj in 0..8 {
                    bempty[p + ii * 8 + jj] += bsub[ii][jj];
                }
            }
        }

        // ABC order-2 correction: Bempty += abc_order_2_matrix(...)
        if port.is_abc_order2() {
            if let Some(coeff) = port.abc_o2_coeff(k0) {
                let abc_correction = crate::abc_order2::abc_order_2_matrix(
                    mesh, basis, tri_ids, coeff,
                );
                for (i, &v) in abc_correction.iter().enumerate() {
                    bempty[i] += v;
                }
            }
        }

        eprintln!("  Port {} Robin: gamma={:.4e}, {} tris, driven={}", pi, gamma, tri_ids.len(), port.is_driven());
    }

    // K += field.generate_csr(Bempty)
    let robin_csr = basis.generate_csr(&bempty);
    k_csr = &k_csr + &robin_csr;

    // Step 5: Port excitation vectors — only for driven ports
    let mut port_vectors: Vec<Vec<C64>> = Vec::new();
    let mut driven_port_indices: Vec<usize> = Vec::new();

    for (pi, (port, tri_ids)) in ports.iter().zip(port_tri_indices.iter()).enumerate() {
        if !port.is_driven() {
            continue; // ABC: no excitation vector
        }
        driven_port_indices.push(pi);

        let mut bvec = vec![C64::new(0.0, 0.0); n_field];

        for &ti in *tri_ids {
            let tri = &mesh.tris[ti];
            let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];

            let u_inc_at_qp: Vec<[C64; 3]> = gauss_points.iter().filter_map(|qp| {
                let (l1, l2, l3) = (qp[1], qp[2], qp[3]);
                let x = verts[0][0]*l1 + verts[1][0]*l2 + verts[2][0]*l3;
                let y = verts[0][1]*l1 + verts[1][1]*l2 + verts[2][1]*l3;
                let z = verts[0][2]*l1 + verts[1][2]*l2 + verts[2][2]*l3;
                port.get_uinc(x, y, z, k0)
            }).collect();

            if u_inc_at_qp.len() == gauss_points.len() {
                let b_tri = ned2_tri_force(&verts, &u_inc_at_qp, &gauss_points);
                let dofs = &basis.tri_to_field[ti];
                for i in 0..8 {
                    bvec[dofs[i]] += b_tri[i];
                }
            }
        }

        let bnorm: f64 = bvec.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        eprintln!("  Port {} ||b|| = {:.6e}", pi, bnorm);
        port_vectors.push(bvec);
    }

    // Step 6: Eliminate PEC DOFs, solve
    let free_dofs: Vec<usize> = (0..n_field).filter(|d| !pec_ids.contains(d)).collect();
    let n_free = free_dofs.len();
    eprintln!("  Free DOFs: {}", n_free);

    // Build reduced dense K from CSR
    let mut dof_to_free = vec![usize::MAX; n_field];
    for (fi, &d) in free_dofs.iter().enumerate() {
        dof_to_free[d] = fi;
    }

    let mut k_dense = vec![C64::new(0.0, 0.0); n_free * n_free];
    for (row_idx, row_vec) in k_csr.outer_iterator().enumerate() {
        if pec_ids.contains(&row_idx) { continue; }
        let fi = dof_to_free[row_idx];
        for (col_idx, &val) in row_vec.iter() {
            if pec_ids.contains(&col_idx) { continue; }
            let fj = dof_to_free[col_idx];
            k_dense[fi * n_free + fj] += val;
        }
    }

    // LU factorize
    let t_solve = std::time::Instant::now();
    let lu = dense_lu_factor(&mut k_dense, n_free);

    // Solve for each port
    let mut solutions = Vec::new();
    for (pi, bvec) in port_vectors.iter().enumerate() {
        let b_free: Vec<C64> = free_dofs.iter().map(|&d| bvec[d]).collect();
        let x_free = dense_lu_solve(&lu, &b_free, n_free);

        let mut x_full = vec![C64::new(0.0, 0.0); n_field];
        for (fi, &d) in free_dofs.iter().enumerate() {
            x_full[d] = x_free[fi];
        }
        let xnorm: f64 = x_full.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        eprintln!("  Port {} solved in {:.1}ms, ||x|| = {:.6e}",
            pi, t_solve.elapsed().as_secs_f64()*1e3, xnorm);
        solutions.push(x_full);
    }

    SolveResult { solutions, n_field }
}

// Dense LU with partial pivoting
fn dense_lu_factor(a: &mut [C64], n: usize) -> (Vec<C64>, Vec<usize>) {
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
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
    (a.to_vec(), piv)
}

fn dense_lu_solve(lu: &(Vec<C64>, Vec<usize>), b: &[C64], n: usize) -> Vec<C64> {
    let (a, piv) = lu;
    let mut x: Vec<C64> = piv.iter().map(|&i| b[i]).collect();
    for i in 0..n {
        for j in 0..i {
            let lij = a[i * n + j];
            let xj = x[j];
            x[i] -= lij * xj;
        }
    }
    for i in (0..n).rev() {
        for j in (i+1)..n {
            let uij = a[i * n + j];
            let xj = x[j];
            x[i] -= uij * xj;
        }
        x[i] /= a[i * n + i];
    }
    x
}
