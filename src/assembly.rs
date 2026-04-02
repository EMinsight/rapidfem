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

    // Step 2: K = E - k0² * B (defer CSR construction — build faer triplets directly later)
    let t1 = std::time::Instant::now();
    let k0_sq = C64::from(k0 * k0);

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

    eprintln!("  Robin BC assembled in {:.1}ms", t1.elapsed().as_secs_f64()*1e3);

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

    // Step 6: Eliminate PEC DOFs, build reduced system, solve
    let free_dofs: Vec<usize> = (0..n_field).filter(|d| !pec_ids.contains(d)).collect();
    let n_free = free_dofs.len();
    eprintln!("  Free DOFs: {}", n_free);

    let mut dof_to_free = vec![usize::MAX; n_field];
    for (fi, &d) in free_dofs.iter().enumerate() {
        dof_to_free[d] = fi;
    }

    // Build COO triplets for reduced system: K = (E - k0²*B) + Robin
    let t2 = std::time::Instant::now();
    let mut coo_rows: Vec<usize> = Vec::new();
    let mut coo_cols: Vec<usize> = Vec::new();
    let mut coo_vals: Vec<C64> = Vec::new();

    for i in 0..rows.len() {
        let r = rows[i];
        let c = cols[i];
        if pec_ids.contains(&r) || pec_ids.contains(&c) { continue; }
        coo_rows.push(dof_to_free[r]);
        coo_cols.push(dof_to_free[c]);
        coo_vals.push(data_e[i] - k0_sq * data_b[i]);
    }
    for (idx, &val) in bempty.iter().enumerate() {
        if val.norm() == 0.0 { continue; }
        let r = basis.tri_rows[idx];
        let c = basis.tri_cols[idx];
        if pec_ids.contains(&r) || pec_ids.contains(&c) { continue; }
        coo_rows.push(dof_to_free[r]);
        coo_cols.push(dof_to_free[c]);
        coo_vals.push(val);
    }
    eprintln!("  COO: {} entries, built in {:.1}ms", coo_rows.len(), t2.elapsed().as_secs_f64()*1e3);

    // Try PARDISO first, fall back to faer
    let solutions = if let Some(ref mut pardiso) = crate::pardiso::PardisoSolver::try_new() {
        // Build upper-triangle CSR for PARDISO (mtype=6: complex symmetric)
        let t_par = std::time::Instant::now();
        let (ia, ja, a) = crate::pardiso::build_upper_csr(n_free, &coo_rows, &coo_cols, &coo_vals);
        eprintln!("  PARDISO: upper CSR {} nnz, built in {:.1}ms", a.len(), t_par.elapsed().as_secs_f64()*1e3);

        pardiso.analyze_and_factorize(n_free as i32, &ia, &ja, &a)
            .expect("PARDISO analyze+factorize failed");
        eprintln!("  PARDISO: factorized in {:.1}ms", t_par.elapsed().as_secs_f64()*1e3);

        let mut solutions = Vec::new();
        for (pi, bvec) in port_vectors.iter().enumerate() {
            let b_free: Vec<C64> = free_dofs.iter().map(|&d| bvec[d]).collect();
            let x_free = pardiso.solve(n_free as i32, &ia, &ja, &a, &b_free)
                .expect("PARDISO solve failed");

            let mut x_full = vec![C64::new(0.0, 0.0); n_field];
            for (fi, &d) in free_dofs.iter().enumerate() {
                x_full[d] = x_free[fi];
            }
            let xnorm: f64 = x_full.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            eprintln!("  Port {} solved (PARDISO) in {:.1}ms, ||x|| = {:.6e}",
                pi, t_par.elapsed().as_secs_f64()*1e3, xnorm);
            solutions.push(x_full);
        }
        solutions
    } else {
        // Fallback: faer sparse LU
        let mut triplets: Vec<faer::sparse::Triplet<usize, usize, faer::c64>> = Vec::new();
        for i in 0..coo_rows.len() {
            triplets.push(faer::sparse::Triplet {
                row: coo_rows[i], col: coo_cols[i],
                val: faer::c64 { re: coo_vals[i].re, im: coo_vals[i].im },
            });
        }
        let k_faer = faer::sparse::SparseColMat::<usize, faer::c64>::try_new_from_triplets(
            n_free, n_free, &triplets,
        ).expect("faer matrix");

        let t_solve = std::time::Instant::now();
        let lu = k_faer.sp_lu().expect("Sparse LU factorization failed");
        eprintln!("  faer LU factorized in {:.1}ms", t_solve.elapsed().as_secs_f64()*1e3);

        let mut solutions = Vec::new();
        for (pi, bvec) in port_vectors.iter().enumerate() {
            let mut x_mat = faer::Mat::<faer::c64>::from_fn(n_free, 1, |i, _| {
                let d = free_dofs[i];
                faer::c64 { re: bvec[d].re, im: bvec[d].im }
            });
            faer::linalg::solvers::SolveCore::solve_in_place_with_conj(&lu, faer::Conj::No, x_mat.as_mut());

            let mut x_full = vec![C64::new(0.0, 0.0); n_field];
            for (fi, &d) in free_dofs.iter().enumerate() {
                let v = x_mat[(fi, 0)];
                x_full[d] = C64::new(v.re, v.im);
            }
            let xnorm: f64 = x_full.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            eprintln!("  Port {} solved (faer) in {:.1}ms, ||x|| = {:.6e}",
                pi, t_solve.elapsed().as_secs_f64()*1e3, xnorm);
            solutions.push(x_full);
        }
        solutions
    };

    SolveResult { solutions, n_field }
}

/// Frequency sweep: solve at multiple frequencies.
///
/// For frequency-independent materials, caches E and B matrices.
/// Returns solutions per frequency: Vec<SolveResult>.
pub fn frequency_sweep(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    ports: &[&dyn Port],
    port_tri_indices: &[&[usize]],
    pec_tri_indices: &[usize],
    frequencies: &[f64],
    materials: Option<&[crate::materials::Material]>,
) -> Vec<SolveResult> {
    // Cache E, B for frequency-independent materials
    let n_tets = mesh.n_tets();
    let (er, ur) = if let Some(mats) = materials {
        crate::materials::build_material_tensors(n_tets, mats, frequencies[0])
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
    eprintln!("  Assembled E,B in {:.1}ms (cached for sweep)", t0.elapsed().as_secs_f64()*1e3);

    // PEC DOFs (frequency-independent)
    let mut pec_ids: HashSet<usize> = HashSet::new();
    for &ti in pec_tri_indices {
        for &ei in &mesh.tri_to_edge[ti] {
            for &d in &basis.edge_to_field[ei] { pec_ids.insert(d); }
        }
        for &d in &basis.tri_to_field[ti] { pec_ids.insert(d); }
    }

    let free_dofs: Vec<usize> = (0..basis.n_field).filter(|d| !pec_ids.contains(d)).collect();
    let n_free = free_dofs.len();
    let mut dof_to_free = vec![usize::MAX; basis.n_field];
    for (fi, &d) in free_dofs.iter().enumerate() { dof_to_free[d] = fi; }

    let ac_base = crate::coefficients::AreaCoeffCache::new();
    let gauss_points = crate::quadrature::gaus_quad_tri(4);

    let mut results = Vec::with_capacity(frequencies.len());

    // Precompute non-PEC COO indices for K entries (reused every frequency)
    let k_free_indices: Vec<usize> = (0..rows.len())
        .filter(|&i| !pec_ids.contains(&rows[i]) && !pec_ids.contains(&cols[i]))
        .collect();
    let k_free_rows: Vec<usize> = k_free_indices.iter().map(|&i| dof_to_free[rows[i]]).collect();
    let k_free_cols: Vec<usize> = k_free_indices.iter().map(|&i| dof_to_free[cols[i]]).collect();

    // Precompute non-PEC Robin indices (reused every frequency)
    let robin_free_indices: Vec<usize> = (0..basis.n_tris * 64)
        .filter(|&idx| {
            let r = basis.tri_rows[idx];
            let c = basis.tri_cols[idx];
            !pec_ids.contains(&r) && !pec_ids.contains(&c)
        })
        .collect();

    // Symbolic factorization: compute once at first frequency, reuse for all
    let mut symbolic_lu: Option<faer::sparse::linalg::solvers::SymbolicLu<usize>> = None;

    for (fi, &freq) in frequencies.iter().enumerate() {
        let t_freq = std::time::Instant::now();
        let k0 = 2.0 * PI * freq / crate::constants::C0;
        let k0_sq = C64::from(k0 * k0);
        let n_field = basis.n_field;

        // Robin BC (γ frequency-dependent)
        let mut bempty = basis.empty_tri_matrix();
        for (_, (port, tri_ids)) in ports.iter().zip(port_tri_indices.iter()).enumerate() {
            let gamma = port.get_gamma(k0);
            for &ti in *tri_ids {
                let tri = &mesh.tris[ti];
                let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];
                let bsub = ned2_tri_stiff(&verts, gamma, &ac_base);
                let p = ti * 64;
                for ii in 0..8 { for jj in 0..8 { bempty[p + ii*8 + jj] += bsub[ii][jj]; } }
            }
            if port.is_abc_order2() {
                if let Some(coeff) = port.abc_o2_coeff(k0) {
                    let abc_corr = crate::abc_order2::abc_order_2_matrix(mesh, basis, tri_ids, coeff);
                    for (i, &v) in abc_corr.iter().enumerate() { bempty[i] += v; }
                }
            }
        }

        // Port excitation vectors
        let mut port_bvecs: Vec<Vec<C64>> = Vec::new();
        for (port, tri_ids) in ports.iter().zip(port_tri_indices.iter()) {
            if !port.is_driven() { continue; }
            let mut bvec = vec![C64::new(0.0, 0.0); n_field];
            for &ti in *tri_ids {
                let tri = &mesh.tris[ti];
                let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];
                let u_at_qp: Vec<[C64; 3]> = gauss_points.iter()
                    .filter_map(|qp| {
                        let (l1,l2,l3) = (qp[1],qp[2],qp[3]);
                        port.get_uinc(
                            verts[0][0]*l1+verts[1][0]*l2+verts[2][0]*l3,
                            verts[0][1]*l1+verts[1][1]*l2+verts[2][1]*l3,
                            verts[0][2]*l1+verts[1][2]*l2+verts[2][2]*l3, k0)
                    }).collect();
                if u_at_qp.len() == gauss_points.len() {
                    let b_tri = ned2_tri_force(&verts, &u_at_qp, &gauss_points);
                    let dofs = &basis.tri_to_field[ti];
                    for i in 0..8 { bvec[dofs[i]] += b_tri[i]; }
                }
            }
            port_bvecs.push(bvec);
        }

        // Build faer triplets: K = (E - k0²*B) + Robin
        let mut triplets: Vec<faer::sparse::Triplet<usize, usize, faer::c64>> = Vec::new();
        triplets.reserve(k_free_indices.len() + robin_free_indices.len());

        for (ti, &orig_i) in k_free_indices.iter().enumerate() {
            let val = data_e[orig_i] - k0_sq * data_b[orig_i];
            triplets.push(faer::sparse::Triplet {
                row: k_free_rows[ti], col: k_free_cols[ti],
                val: faer::c64 { re: val.re, im: val.im },
            });
        }
        for &idx in &robin_free_indices {
            let val = bempty[idx];
            if val.norm() == 0.0 { continue; }
            triplets.push(faer::sparse::Triplet {
                row: dof_to_free[basis.tri_rows[idx]],
                col: dof_to_free[basis.tri_cols[idx]],
                val: faer::c64 { re: val.re, im: val.im },
            });
        }

        let k_faer = faer::sparse::SparseColMat::<usize, faer::c64>::try_new_from_triplets(
            n_free, n_free, &triplets,
        ).expect("faer matrix");

        // Symbolic reuse: compute symbolic once, reuse for subsequent frequencies
        let lu = if symbolic_lu.is_none() {
            let t_sym = std::time::Instant::now();
            let sym = faer::sparse::linalg::solvers::SymbolicLu::try_new(k_faer.as_ref().symbolic())
                .expect("symbolic LU");
            eprintln!("  Symbolic LU in {:.1}ms (computed once)", t_sym.elapsed().as_secs_f64()*1e3);
            let lu = faer::sparse::linalg::solvers::Lu::try_new_with_symbolic(sym.clone(), k_faer.as_ref())
                .expect("numeric LU");
            symbolic_lu = Some(sym);
            lu
        } else {
            faer::sparse::linalg::solvers::Lu::try_new_with_symbolic(
                symbolic_lu.as_ref().unwrap().clone(), k_faer.as_ref(),
            ).expect("numeric LU (reused symbolic)")
        };

        let mut solutions = Vec::new();
        for bvec in &port_bvecs {
            let mut x_mat = faer::Mat::<faer::c64>::from_fn(n_free, 1, |i, _| {
                let d = free_dofs[i];
                faer::c64 { re: bvec[d].re, im: bvec[d].im }
            });
            faer::linalg::solvers::SolveCore::solve_in_place_with_conj(&lu, faer::Conj::No, x_mat.as_mut());
            let mut x_full = vec![C64::new(0.0, 0.0); n_field];
            for (fi_d, &d) in free_dofs.iter().enumerate() {
                let v = x_mat[(fi_d, 0)];
                x_full[d] = C64::new(v.re, v.im);
            }
            solutions.push(x_full);
        }

        eprintln!("  f={:.4e} Hz: {:.1}ms (freq {}/{})", freq, t_freq.elapsed().as_secs_f64()*1e3, fi+1, frequencies.len());
        results.push(SolveResult { solutions, n_field });
    }

    results
}