// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! `SparseSolver` impl using faer's pure-Rust sparse LU.
//!
//! The general-LU path is the cross-platform fallback. Complex symmetric is
//! handled as a general matrix here — slower than a dedicated LDLᵀ but works
//! on every platform with no native deps. PARDISO and Apple Accelerate are
//! the fast lane on their respective systems.

use faer::sparse::{SparseColMat, Triplet};
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::c64;
use faer::Conj;
use faer::linalg::solvers::SolveCore;
use num_complex::Complex64 as C64;
use super::SparseSolver;

pub struct FaerLuSolver {
    n: usize,
    symbolic: Option<SymbolicLu<usize>>,
    lu: Option<Lu<usize, c64>>,
}

impl FaerLuSolver {
    pub fn new() -> Self { Self { n: 0, symbolic: None, lu: None } }

    fn build_matrix(n: usize, rows: &[usize], cols: &[usize], vals: &[C64])
        -> Result<SparseColMat<usize, c64>, String>
    {
        let triplets: Vec<Triplet<usize, usize, c64>> = rows.iter().zip(cols).zip(vals)
            .map(|((&r, &c), v)| Triplet { row: r, col: c, val: c64 { re: v.re, im: v.im } })
            .collect();
        SparseColMat::<usize, c64>::try_new_from_triplets(n, n, &triplets)
            .map_err(|e| format!("faer matrix build: {e:?}"))
    }
}

impl Default for FaerLuSolver {
    fn default() -> Self { Self::new() }
}

impl SparseSolver for FaerLuSolver {
    fn factorize(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[C64],
    ) -> Result<(), String> {
        let mat = Self::build_matrix(n, rows, cols, vals)?;
        let sym = SymbolicLu::try_new(mat.as_ref().symbolic())
            .map_err(|e| format!("faer symbolic LU: {e:?}"))?;
        let lu = Lu::try_new_with_symbolic(sym.clone(), mat.as_ref())
            .map_err(|e| format!("faer numeric LU: {e:?}"))?;
        self.n = n;
        self.symbolic = Some(sym);
        self.lu = Some(lu);
        Ok(())
    }

    /// Numeric refactor with the cached symbolic pattern — saves the AMD
    /// reordering and symbolic factorisation work across frequency sweeps.
    fn refactorize(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[C64],
    ) -> Result<(), String> {
        let Some(sym) = self.symbolic.clone() else {
            return self.factorize(n, rows, cols, vals);
        };
        let mat = Self::build_matrix(n, rows, cols, vals)?;
        let lu = Lu::try_new_with_symbolic(sym, mat.as_ref())
            .map_err(|e| format!("faer numeric LU: {e:?}"))?;
        self.n = n;
        self.lu = Some(lu);
        Ok(())
    }

    fn solve(&mut self, b: &[C64]) -> Result<Vec<C64>, String> {
        let lu = self.lu.as_ref()
            .ok_or_else(|| "faer LU: solve before factorize".to_string())?;
        if b.len() != self.n {
            return Err(format!("faer LU: RHS length {} ≠ n = {}", b.len(), self.n));
        }
        let mut x = faer::Mat::<c64>::from_fn(self.n, 1, |i, _| c64 { re: b[i].re, im: b[i].im });
        lu.solve_in_place_with_conj(Conj::No, x.as_mut());
        Ok((0..self.n).map(|i| { let v = x[(i, 0)]; C64::new(v.re, v.im) }).collect())
    }

    fn name(&self) -> &'static str { "faer LU" }
}
