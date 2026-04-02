//! Exact port of EMerge material handling from assembler.py lines 280-305.
//!
//! Builds per-tet εr and μr tensors with loss tangent and conductivity:
//!   er_complex = er * (1 - j*tand) - j*cond / (ω*ε₀)

use num_complex::Complex64 as C64;
use crate::constants::EPS0;

/// Material definition for a region of the mesh.
/// Mirrors EMerge's Material class interface.
pub struct Material {
    /// Relative permittivity (scalar, isotropic)
    pub er: f64,
    /// Relative permeability (scalar, isotropic)
    pub ur: f64,
    /// Loss tangent tan(δ)
    pub tand: f64,
    /// Conductivity σ (S/m)
    pub cond: f64,
    /// Which tets this material applies to (indices into mesh.tets)
    pub tet_indices: Vec<usize>,
}

/// Build per-tet εr and μr tensors from material definitions.
/// Exact port of assembler.py lines 280-303.
///
/// Returns (er_tensors, ur_tensors) where each is Vec of 3x3 complex tensors.
pub fn build_material_tensors(
    n_tets: usize,
    materials: &[Material],
    frequency: f64,
) -> (Vec<[[C64; 3]; 3]>, Vec<[[C64; 3]; 3]>) {
    let w0 = 2.0 * std::f64::consts::PI * frequency;

    // Initialize as zeros (matches EMerge: np.zeros((3,3,n_tets)))
    let zero3x3 = [[C64::new(0.0, 0.0); 3]; 3];
    let mut er = vec![zero3x3; n_tets];
    let mut ur = vec![zero3x3; n_tets];
    let mut tand = vec![zero3x3; n_tets];
    let mut cond = vec![zero3x3; n_tets];

    // Accumulate material properties per tet (matches EMerge's mat.er(frequency, er) pattern)
    for mat in materials {
        for &ti in &mat.tet_indices {
            // Isotropic: set diagonal elements
            for k in 0..3 {
                er[ti][k][k] += C64::from(mat.er);
                ur[ti][k][k] += C64::from(mat.ur);
                tand[ti][k][k] += C64::from(mat.tand);
                cond[ti][k][k] += C64::from(mat.cond);
            }
        }
    }

    // Apply loss formula: er = er*(1 - 1j*tand) - 1j*cond/(W0*EPS0)
    // Exact port of assembler.py line 303
    for ti in 0..n_tets {
        for i in 0..3 {
            for j in 0..3 {
                let er_val = er[ti][i][j];
                let tand_val = tand[ti][i][j];
                let cond_val = cond[ti][i][j];
                er[ti][i][j] = er_val * (C64::new(1.0, 0.0) - C64::new(0.0, 1.0) * tand_val)
                    - C64::new(0.0, 1.0) * cond_val / C64::from(w0 * EPS0);
            }
        }
    }

    (er, ur)
}
