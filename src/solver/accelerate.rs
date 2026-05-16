//! `SparseSolver` impl backed by Apple Accelerate's sparse Bunch-Kaufman.
//!
//! macOS-only. Uses the real-block reformulation
//!
//!   M = [[ Re(A), -Im(A) ],
//!        [-Im(A), -Re(A) ]]
//!
//! which turns our complex-symmetric `A` (size N) into a real-symmetric
//! INDEFINITE 2N×2N system suitable for Accelerate's
//! `SparseFactorizationLDLTSBK` (supernodal Bunch-Kaufman). Solution
//! `[x_re; x_im]` reconstructs `x = x_re + j·x_im`.
//!
//! STATUS: stub — `try_new()` currently returns `None` so the auto-pick
//! falls through to faer. Real FFI lands in a follow-up commit.

use num_complex::Complex64 as C64;
use super::SparseSolver;

pub struct AccelerateSolver {
    // Real-block factorisation handles + buffers will live here once the
    // FFI lands. The struct stays the same so the trait wiring is stable.
    _phantom: (),
}

impl AccelerateSolver {
    /// Probe whether Accelerate's sparse solvers are usable. Currently
    /// returns `None` (stub) so the auto path silently falls through to faer.
    pub fn try_new() -> Option<Self> {
        None
    }
}

impl SparseSolver for AccelerateSolver {
    fn factorize(
        &mut self,
        _n: usize,
        _rows: &[usize],
        _cols: &[usize],
        _vals: &[C64],
    ) -> Result<(), String> {
        Err("Accelerate backend not yet implemented".to_string())
    }

    fn solve(&mut self, _b: &[C64]) -> Result<Vec<C64>, String> {
        Err("Accelerate backend not yet implemented".to_string())
    }

    fn name(&self) -> &'static str { "Apple Accelerate (Bunch-Kaufman)" }
}
