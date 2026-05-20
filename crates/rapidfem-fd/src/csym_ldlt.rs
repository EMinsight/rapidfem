//! Complex-symmetric LDLᵀ factorization (NOT Hermitian LDLᴴ).
//!
//! For complex-symmetric matrices where A = Aᵀ (not A = A*).
//! Uses faer's symbolic Cholesky for fill-reducing ordering and elimination tree,
//! but performs the numeric factorization without conjugation.
//!
//! Key difference from faer's LDLᴴ:
//!   Hermitian: x[i] -= lij.conj() * xj,  d -= (lkj * xj.conj()).real()
//!   Symmetric: x[i] -= lij * xj,          d -= lkj * xj  (D is complex)

use faer::c64;

/// Simplicial complex-symmetric LDLᵀ factorization result.
pub struct CSymLdlt {
    /// L factor values (lower triangular, unit diagonal implied)
    pub l_values: Vec<c64>,
    /// D diagonal values (complex, NOT necessarily real)
    pub d_values: Vec<c64>,
    /// Row indices for L (from symbolic factorization)
    pub l_row_idx: Vec<usize>,
    /// Column pointers for L (from symbolic factorization)
    pub l_col_ptr: Vec<usize>,
    /// Permutation: original → reordered
    pub perm: Vec<usize>,
    /// Inverse permutation: reordered → original
    pub perm_inv: Vec<usize>,
    /// Matrix dimension
    pub n: usize,
}

impl CSymLdlt {
    /// Factorize a complex-symmetric sparse matrix.
    ///
    /// `col_ptr`, `row_idx`, `values`: CSC format of the UPPER triangle of A.
    /// Uses identity ordering (no fill reduction yet — TODO: add AMD).
    pub fn factorize(
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[c64],
    ) -> Result<Self, String> {
        // Identity ordering (no fill reduction)
        // TODO: integrate AMD ordering for better performance on large problems
        let perm: Vec<usize> = (0..n).collect();
        let perm_inv: Vec<usize> = (0..n).collect();

        // Compute elimination tree and column counts
        // Simple up-looking LDLᵀ without supernodes
        // Based on Tim Davis's CSparse ldl_symbolic + ldl_numeric

        // Symbolic: compute elimination tree and L column counts
        let mut etree = vec![usize::MAX; n]; // parent in elimination tree
        let mut l_counts = vec![0usize; n];  // nonzeros per column of L
        let mut flags = vec![false; n];

        for k in 0..n {
            flags[k] = true;
            // Walk up the elimination tree from each row index in column k of A
            let a_start = col_ptr[perm[k]];
            let a_end = col_ptr[perm[k] + 1];
            for idx in a_start..a_end {
                let mut i = perm_inv[row_idx[idx]];
                if i >= k { continue; } // only upper triangle
                // Walk from i to k in the elimination tree
                while !flags[i] {
                    flags[i] = true;
                    l_counts[i] += 1;
                    if etree[i] == usize::MAX {
                        etree[i] = k;
                    }
                    i = etree[i];
                }
            }
            // Reset flags
            let a_start2 = col_ptr[perm[k]];
            let a_end2 = col_ptr[perm[k] + 1];
            for idx in a_start2..a_end2 {
                let mut i = perm_inv[row_idx[idx]];
                if i >= k { continue; }
                while flags[i] {
                    flags[i] = false;
                    i = etree[i];
                    if i == usize::MAX { break; }
                }
            }
            flags[k] = false;
        }

        // Build L column pointers from counts
        let mut l_col_ptr = vec![0usize; n + 1];
        for i in 0..n {
            l_col_ptr[i + 1] = l_col_ptr[i] + l_counts[i];
        }
        let l_nnz = l_col_ptr[n];
        let mut l_row_idx_out = vec![0usize; l_nnz];
        let mut l_values_out = vec![c64 { re: 0.0, im: 0.0 }; l_nnz];
        let mut d_values_out = vec![c64 { re: 0.0, im: 0.0 }; n];

        // Numeric: up-looking LDLᵀ (complex-symmetric, NO conjugation)
        let mut x = vec![c64 { re: 0.0, im: 0.0 }; n];
        let mut curr_ptr = l_col_ptr[..n].to_vec(); // current write position per column

        for k in 0..n {
            // Scatter A[:,perm[k]] into x
            let a_start = col_ptr[perm[k]];
            let a_end = col_ptr[perm[k] + 1];
            for idx in a_start..a_end {
                let i = perm_inv[row_idx[idx]];
                // NO conjugation — complex-symmetric
                x[i] = x[i] + values[idx];
            }

            let mut dk = x[k];
            x[k] = c64 { re: 0.0, im: 0.0 };

            // Walk up elimination tree
            let mut i = 0;
            // Collect the reach of column k
            let mut reach = Vec::new();
            {
                let a_start = col_ptr[perm[k]];
                let a_end = col_ptr[perm[k] + 1];
                let mut flags2 = vec![false; n];
                flags2[k] = true;
                for idx in a_start..a_end {
                    let mut j = perm_inv[row_idx[idx]];
                    if j >= k { continue; }
                    while !flags2[j] {
                        flags2[j] = true;
                        reach.push(j);
                        if etree[j] == usize::MAX { break; }
                        j = etree[j];
                    }
                }
            }
            reach.sort();

            for &j in &reach {
                let j_start = l_col_ptr[j];
                let j_end = curr_ptr[j];

                let xj = x[j];
                x[j] = c64 { re: 0.0, im: 0.0 };

                // lkj = xj / d[j]  (NO conjugation)
                let dj = d_values_out[j];
                if dj.re == 0.0 && dj.im == 0.0 {
                    return Err(format!("Zero pivot at column {}", j));
                }
                let lkj = xj / dj;

                // Update x[i] -= L[i,j] * xj for all i > k in column j of L
                for p in j_start..j_end {
                    let li = l_row_idx_out[p];
                    // NO conjugation — complex-symmetric
                    x[li] = x[li] - l_values_out[p] * xj;
                }

                // d[k] -= lkj * xj  (NO .real() — D is complex)
                dk = dk - lkj * xj;

                // Store L[k,j] = lkj
                let pos = curr_ptr[j];
                l_row_idx_out[pos] = k;
                l_values_out[pos] = lkj;
                curr_ptr[j] = pos + 1;
            }

            if dk.re == 0.0 && dk.im == 0.0 {
                return Err(format!("Zero pivot at column {}", k));
            }
            d_values_out[k] = dk;
        }

        Ok(CSymLdlt {
            l_values: l_values_out,
            d_values: d_values_out,
            l_row_idx: l_row_idx_out,
            l_col_ptr,
            perm,
            perm_inv,
            n,
        })
    }

    /// Solve Ax = b where A = LDLᵀ.
    /// Steps: permute b, forward solve L, diagonal solve D, backward solve Lᵀ, unpermute.
    pub fn solve(&self, rhs: &[c64]) -> Vec<c64> {
        let n = self.n;
        let zero = c64 { re: 0.0, im: 0.0 };

        // Permute RHS: b_perm[i] = rhs[perm[i]]
        let mut x: Vec<c64> = self.perm.iter().map(|&p| rhs[p]).collect();

        // Forward solve: L * y = b_perm
        for j in 0..n {
            let xj = x[j];
            let start = self.l_col_ptr[j];
            let end = self.l_col_ptr[j + 1];
            for p in start..end {
                let i = self.l_row_idx[p];
                x[i] = x[i] - self.l_values[p] * xj;
            }
        }

        // Diagonal solve: D * z = y
        for i in 0..n {
            x[i] = x[i] / self.d_values[i];
        }

        // Backward solve: Lᵀ * x = z (transpose, NOT conjugate transpose)
        for j in (0..n).rev() {
            let start = self.l_col_ptr[j];
            let end = self.l_col_ptr[j + 1];
            for p in start..end {
                let i = self.l_row_idx[p];
                // Lᵀ[j,i] = L[i,j] (NO conjugation)
                x[j] = x[j] - self.l_values[p] * x[i];
            }
        }

        // Unpermute: result[perm[i]] = x[i]
        let mut result = vec![zero; n];
        for i in 0..n {
            result[self.perm[i]] = x[i];
        }
        result
    }
}
