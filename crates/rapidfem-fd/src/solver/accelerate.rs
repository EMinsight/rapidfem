// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! `SparseSolver` impl backed by Apple Accelerate's sparse Bunch-Kaufman.
//!
//! macOS-only. Uses the real-block reformulation
//!
//!   M = [[ Re(A), -Im(A) ],
//!        [-Im(A), -Re(A) ]]
//!
//! which turns the complex-symmetric `A` (size N) into a real-symmetric
//! INDEFINITE matrix of size 2N. The corresponding RHS is `[b_re; -b_im]`
//! and the solution decomposes as `x = x_re + j·x_im`. `M` is real symmetric
//! indefinite, factored by Apple's `SparseFactorizationLDLTSBK` (supernodal
//! Bunch-Kaufman) via the C shim in `accelerate_shim.c`.
//!
//! Why a shim: Apple's `SparseFactor` / `SparseSolve` are C++-overloaded; the
//! exported mangled symbols are awkward to bind directly from Rust. The shim
//! gives us three stable C entry points that we link to via `libloading`-free
//! FFI (compiled by `build.rs`).

use num_complex::Complex64 as C64;
use std::ffi::c_void;
use super::SparseSolver;

unsafe extern "C" {
    fn accel_ldlt_factorize(
        n: i32,
        col_starts: *const i64,
        row_idx: *const i32,
        values: *const f64,
    ) -> *mut c_void;
    fn accel_ldlt_solve(handle: *mut c_void, b: *const f64, x: *mut f64) -> i32;
    fn accel_ldlt_destroy(handle: *mut c_void);
}

pub struct AccelerateSolver {
    handle: *mut c_void,
    /// Complex dimension N (real-block dim is 2N).
    n: usize,
}

// The shim and Apple's factor are not thread-shared, but a single owner can
// move the solver across threads — same contract as `PardisoSolver`.
unsafe impl Send for AccelerateSolver {}

impl AccelerateSolver {
    /// Probe whether the Accelerate path is available. The shim is linked
    /// unconditionally on macOS so this always succeeds — kept for parity
    /// with `PardisoSolver::try_new`.
    pub fn try_new() -> Option<Self> {
        Some(Self { handle: std::ptr::null_mut(), n: 0 })
    }
}

impl Drop for AccelerateSolver {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { accel_ldlt_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl SparseSolver for AccelerateSolver {
    fn factorize(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[C64],
    ) -> Result<(), String> {
        // Free any prior factor before replacing.
        if !self.handle.is_null() {
            unsafe { accel_ldlt_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
        let (col_starts, row_idx, values) = build_real_block_upper_csc(n, rows, cols, vals);
        let two_n = 2 * n;
        debug_assert_eq!(col_starts.len(), two_n + 1);
        if two_n > i32::MAX as usize {
            return Err(format!("Accelerate: real-block dim 2N={two_n} exceeds i32"));
        }
        let h = unsafe {
            accel_ldlt_factorize(
                two_n as i32,
                col_starts.as_ptr(),
                row_idx.as_ptr(),
                values.as_ptr(),
            )
        };
        if h.is_null() {
            return Err("Accelerate: LDLᵀ factorisation failed".to_string());
        }
        self.handle = h;
        self.n = n;
        Ok(())
    }

    fn solve(&mut self, b: &[C64]) -> Result<Vec<C64>, String> {
        if self.handle.is_null() {
            return Err("Accelerate: solve before factorize".to_string());
        }
        if b.len() != self.n {
            return Err(format!("Accelerate: RHS length {} ≠ N = {}", b.len(), self.n));
        }
        let two_n = 2 * self.n;
        // Build real-block RHS = [b_re; -b_im].
        let mut rhs = vec![0.0f64; two_n];
        for i in 0..self.n {
            rhs[i]              = b[i].re;
            rhs[self.n + i]     = -b[i].im;
        }
        let mut sol = vec![0.0f64; two_n];
        let rc = unsafe {
            accel_ldlt_solve(self.handle, rhs.as_ptr(), sol.as_mut_ptr())
        };
        if rc != 0 {
            return Err(format!("Accelerate: SparseSolve returned {rc}"));
        }
        // Reconstruct x = x_re + j·x_im. Solution layout is [x_re; x_im].
        let out: Vec<C64> = (0..self.n)
            .map(|i| C64::new(sol[i], sol[self.n + i]))
            .collect();
        Ok(out)
    }

    fn name(&self) -> &'static str { "Apple Accelerate (Bunch-Kaufman)" }
}

/// Build the upper-triangle CSC representation of the real-block matrix
///
///   M = [[ Re(A), -Im(A) ],
///        [-Im(A), -Re(A) ]]
///
/// directly from the complex `A`'s full COO triplets. `A` is assumed complex
/// symmetric (both halves may appear in the input; duplicates are summed).
/// The returned arrays use Apple's column-major sparse layout:
///   - `col_starts: [i64; 2N+1]`
///   - `row_idx:    [i32; nnz]`  with `row_idx[k] ≤ col` for every entry
///   - `values:     [f64; nnz]`
fn build_real_block_upper_csc(
    n: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[C64],
) -> (Vec<i64>, Vec<i32>, Vec<f64>) {
    // 1) Dedup upper-triangle of A: collect (r, c, sum_v) with r ≤ c.
    let mut entries: Vec<(usize, usize, C64)> = Vec::with_capacity(rows.len());
    for i in 0..rows.len() {
        let (r, c) = (rows[i], cols[i]);
        if r <= c {
            entries.push((r, c, vals[i]));
        }
    }
    entries.sort_unstable_by_key(|&(r, c, _)| (r, c));

    let mut a_upper: Vec<(usize, usize, C64)> = Vec::with_capacity(entries.len());
    let mut i = 0;
    while i < entries.len() {
        let (r, c, mut v) = entries[i];
        i += 1;
        while i < entries.len() && entries[i].0 == r && entries[i].1 == c {
            v += entries[i].2;
            i += 1;
        }
        a_upper.push((r, c, v));
    }

    // 2) Expand each A_upper entry into M's upper-triangle entries.
    //    For (r, c, v) with r ≤ c:
    //      (r,        c,        Re(v))    block (0,0)
    //      (r+N,      c+N,     -Re(v))    block (1,1)
    //      (r,        c+N,     -Im(v))    block (0,1)
    //      if r < c:
    //      (c,        r+N,     -Im(v))    block (0,1), mirror
    //
    // Output COO collected column-first for the CSC build below.
    let two_n = 2 * n;
    let approx_nnz = a_upper.len() * 4;
    let mut col_of: Vec<usize> = Vec::with_capacity(approx_nnz);
    let mut row_of: Vec<usize> = Vec::with_capacity(approx_nnz);
    let mut val_of: Vec<f64>   = Vec::with_capacity(approx_nnz);

    for &(r, c, v) in &a_upper {
        let re = v.re;
        let im = v.im;
        // block(0,0): (r, c, Re)
        if re != 0.0 {
            row_of.push(r);     col_of.push(c);     val_of.push(re);
        }
        // block(1,1): (r+N, c+N, -Re)
        if re != 0.0 {
            row_of.push(r + n); col_of.push(c + n); val_of.push(-re);
        }
        // block(0,1): (r, c+N, -Im) — always upper since c+N > r
        if im != 0.0 {
            row_of.push(r);     col_of.push(c + n); val_of.push(-im);
            if r < c {
                // mirror: (c, r+N, -Im) — also upper since r+N > c
                row_of.push(c); col_of.push(r + n); val_of.push(-im);
            }
        }
    }

    // 3) Sort COO by (col, row) and pack into CSC.
    let mut order: Vec<usize> = (0..col_of.len()).collect();
    order.sort_unstable_by_key(|&k| (col_of[k], row_of[k]));

    let mut col_starts = vec![0i64; two_n + 1];
    let mut row_idx: Vec<i32> = Vec::with_capacity(order.len());
    let mut values:  Vec<f64> = Vec::with_capacity(order.len());

    let mut cur_col = 0usize;
    let mut nnz_so_far = 0i64;
    for k in order {
        let c = col_of[k];
        let r = row_of[k];
        let v = val_of[k];
        while cur_col <= c {
            col_starts[cur_col] = nnz_so_far;
            cur_col += 1;
        }
        row_idx.push(r as i32);
        values.push(v);
        nnz_so_far += 1;
    }
    while cur_col <= two_n {
        col_starts[cur_col] = nnz_so_far;
        cur_col += 1;
    }

    (col_starts, row_idx, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test on a tiny 3×3 complex-symmetric matrix — verifies the
    /// real-block construction and Apple's solver round-trip.
    #[test]
    fn solve_3x3_round_trip() {
        // A = [[ 2+0j,  1+0.5j, 0     ],
        //      [ 1+0.5j, 4-1j,  0.3j  ],
        //      [ 0,      0.3j,  3+0.2j]]
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let vals = vec![
            C64::new(2.0, 0.0),  C64::new(1.0, 0.5),
            C64::new(1.0, 0.5),  C64::new(4.0, -1.0), C64::new(0.0, 0.3),
            C64::new(0.0, 0.3),  C64::new(3.0, 0.2),
        ];
        let mut solver = AccelerateSolver::try_new().expect("Accelerate available");
        solver.factorize(3, &rows, &cols, &vals).unwrap();

        // Pick an x, compute b = A·x, solve, check ‖x_recovered − x‖ ≪ ‖x‖.
        let x = [C64::new(1.0, 0.0), C64::new(0.5, -0.7), C64::new(-0.3, 0.1)];
        let mut b = [C64::new(0.0, 0.0); 3];
        for i in 0..3 {
            for k in 0..rows.len() {
                if rows[k] == i {
                    b[i] += vals[k] * x[cols[k]];
                }
            }
        }
        let x_back = solver.solve(&b).unwrap();
        let err: f64 = x_back.iter().zip(x.iter()).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt();
        let xn: f64  = x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(err / xn < 1e-10, "rel err {} too large; got x = {:?}", err / xn, x_back);
    }
}
