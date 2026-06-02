// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! `SparseSolver` impl wrapping the low-level MKL PARDISO FFI in `crate::pardiso`.
//!
//! PARDISO expects an upper-triangle CSR with i32 indices for mtype=6
//! (complex symmetric). We accept the trait's full COO triplet input,
//! build that representation once during `factorize`, and replay it on
//! every `solve` (PARDISO's phase-33 wants the matrix again for iterative
//! refinement, see iparm[7]).

use num_complex::Complex64 as C64;
use super::SparseSolver;

pub struct PardisoSolver {
    inner: crate::pardiso::PardisoSolver,
    n: usize,
    ia: Vec<i32>,
    ja: Vec<i32>,
    a: Vec<C64>,
}

impl PardisoSolver {
    /// Try to load MKL and create a solver. Returns `None` if `mkl_rt`
    /// (`.dll` / `.so` / `.dylib`) isn't available on the system.
    pub fn try_new() -> Option<Self> {
        crate::pardiso::PardisoSolver::try_new().map(|inner| Self {
            inner,
            n: 0,
            ia: Vec::new(),
            ja: Vec::new(),
            a: Vec::new(),
        })
    }
}

impl SparseSolver for PardisoSolver {
    fn factorize(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[C64],
    ) -> Result<(), String> {
        let (ia, ja, a) = crate::pardiso::build_upper_csr(n, rows, cols, vals);
        // Phase 12, combined symbolic + numeric. Replaces whatever was
        // factored previously.
        self.inner.analyze_and_factorize(n as i32, &ia, &ja, &a)?;
        self.n = n;
        self.ia = ia;
        self.ja = ja;
        self.a = a;
        Ok(())
    }

    /// Reuse the cached symbolic factorisation across frequencies (PARDISO
    /// phase 22 only). Assumes the sparsity pattern is unchanged.
    fn refactorize(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[C64],
    ) -> Result<(), String> {
        if self.ia.is_empty() {
            return self.factorize(n, rows, cols, vals);
        }
        let (ia, ja, a) = crate::pardiso::build_upper_csr(n, rows, cols, vals);
        self.inner.factorize(n as i32, &ia, &ja, &a)?;
        self.n = n;
        self.ia = ia;
        self.ja = ja;
        self.a = a;
        Ok(())
    }

    fn solve(&mut self, b: &[C64]) -> Result<Vec<C64>, String> {
        self.inner.solve(self.n as i32, &self.ia, &self.ja, &self.a, b)
    }

    fn name(&self) -> &'static str { "PARDISO" }
}
