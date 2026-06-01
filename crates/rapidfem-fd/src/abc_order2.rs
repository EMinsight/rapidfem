// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
// Copyright (C) Robert Fennis (original EMerge source)
//
// This file is part of rapidfem and contains code ported from EMerge
// (https://github.com/FennisRobert/EMerge), originally licensed under
// GPL-2.0-or-later with the Gmsh additional permission; redistributed
// here under GPL-3.0-or-later with that permission preserved.
// See LICENSE and NOTICE for the full terms.

//! Passivity-projected second-order ABC correction matrix.
//!
//! The raw second-order term is `(c2/k0) * Lengths * (Curl - Div) * |Area|`
//! (ported from EMerge's robin_abc_order2.py). `Curl` and `Div` are each
//! symmetric positive-semidefinite, so their difference is indefinite, and the
//! second-order term can drive the boundary operator's imaginary part negative
//! — the boundary then *injects* energy and `|S|` exceeds 0 dB on high-Q /
//! near-conductor faces (a known Bayliss-Turkel instability).
//!
//! We restore passivity element-by-element. For each boundary triangle we form
//! the full imaginary boundary operator that the assembly applies,
//!
//! ```text
//!     H = k0*c1 * B1  +  (c2/k0) * (Curl - Div)
//! ```
//!
//! where `B1 = ∫ (n̂×W_i)·(n̂×W_j) dS` is the first-order surface mass (the same
//! matrix the Robin term uses, via `ned2_tri_stiff`), and project `H` onto the
//! nearest symmetric positive-semidefinite matrix (eigen-clamp the negatives to
//! zero). Summing PSD element operators keeps the global `Im(A) ⪰ 0`, so the
//! discrete system is dissipative and `|S| ≤ 1` is guaranteed, while the full
//! second-order accuracy survives in every eigendirection that is already
//! passivity-compatible (only the offending directions degrade, to a perfect
//! reflector rather than an energy source). The correction we hand back is
//! `j*(H_psd − k0*c1*B1)`, so `first-order (j*k0*c1*B1) + correction = j*H_psd`.

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::tri_assembly::ned2_tri_stiff;
use crate::coefficients::AreaCoeffCache;

const NQP: usize = 6;

// Port of _gqi: weighted inner product
fn gqi(v1: &[C64; NQP], v2: &[C64; NQP], w: &[f64; NQP]) -> C64 {
    let mut s = C64::new(0.0, 0.0);
    for i in 0..NQP {
        s += v1[i] * v2[i] * C64::from(w[i]);
    }
    s
}

// Port of _curl_edge_1(coeff, coords)
fn curl_edge_1(coeff: &[[f64; 2]; 3], xs: &[f64; NQP], ys: &[f64; NQP]) -> [C64; NQP] {
    let (a1, b1, c1) = (coeff[0][0], coeff[1][0], coeff[2][0]);
    let (a2, b2, c2) = (coeff[0][1], coeff[1][1], coeff[2][1]);
    let mut out = [C64::new(0.0, 0.0); NQP];
    for i in 0..NQP {
        out[i] = C64::from(-3.0*a1*b1*c2 + 3.0*a1*b2*c1 - 3.0*b1*b1*c2*xs[i] + 3.0*b1*b2*c1*xs[i] - 3.0*b1*c1*c2*ys[i] + 3.0*b2*c1*c1*ys[i]);
    }
    out
}

// Port of _curl_edge_2(coeff, coords)
fn curl_edge_2(coeff: &[[f64; 2]; 3], xs: &[f64; NQP], ys: &[f64; NQP]) -> [C64; NQP] {
    let (a1, b1, c1) = (coeff[0][0], coeff[1][0], coeff[2][0]);
    let (a2, b2, c2) = (coeff[0][1], coeff[1][1], coeff[2][1]);
    let _ = a1;
    let mut out = [C64::new(0.0, 0.0); NQP];
    for i in 0..NQP {
        out[i] = C64::from(-3.0*a2*b1*c2 + 3.0*a2*b2*c1 - 3.0*b1*b2*c2*xs[i] - 3.0*b1*c2*c2*ys[i] + 3.0*b2*b2*c1*xs[i] + 3.0*b2*c1*c2*ys[i]);
    }
    out
}

// Port of _curl_face_1(coeff, coords)
fn curl_face_1(coeff: &[[f64; 3]; 3], xs: &[f64; NQP], ys: &[f64; NQP]) -> [C64; NQP] {
    let (a1, b1, c1) = (coeff[0][0], coeff[1][0], coeff[2][0]);
    let (a2, b2, c2) = (coeff[0][1], coeff[1][1], coeff[2][1]);
    let (a3, b3, c3) = (coeff[0][2], coeff[1][2], coeff[2][2]);
    let mut out = [C64::new(0.0, 0.0); NQP];
    for i in 0..NQP {
        let x = xs[i]; let y = ys[i];
        out[i] = C64::from(
            -b2*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))
            + c2*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))
            + 2.0*(b1*c3 - b3*c1)*(a2 + b2*x + c2*y)
        );
    }
    out
}

// Port of _curl_face_2(coeff, coords)
fn curl_face_2(coeff: &[[f64; 3]; 3], xs: &[f64; NQP], ys: &[f64; NQP]) -> [C64; NQP] {
    let (a1, b1, c1) = (coeff[0][0], coeff[1][0], coeff[2][0]);
    let (a2, b2, c2) = (coeff[0][1], coeff[1][1], coeff[2][1]);
    let (a3, b3, c3) = (coeff[0][2], coeff[1][2], coeff[2][2]);
    let mut out = [C64::new(0.0, 0.0); NQP];
    for i in 0..NQP {
        let x = xs[i]; let y = ys[i];
        out[i] = C64::from(
            b3*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))
            - c3*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))
            - 2.0*(b1*c2 - b2*c1)*(a3 + b3*x + c3*y)
        );
    }
    out
}

// Port of _divergence_edge_1(coeff, coords)
fn divergence_edge_1(coeff: &[[f64; 2]; 3], xs: &[f64; NQP], ys: &[f64; NQP]) -> [C64; NQP] {
    let (a1, b1, c1) = (coeff[0][0], coeff[1][0], coeff[2][0]);
    let (a2, b2, c2) = (coeff[0][1], coeff[1][1], coeff[2][1]);
    let mut out = [C64::new(0.0, 0.0); NQP];
    for i in 0..NQP {
        let x = xs[i]; let y = ys[i];
        out[i] = C64::from(
            b1*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))
            + c1*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))
        );
    }
    out
}

// Port of _divergence_edge_2(coeff, coords)
fn divergence_edge_2(coeff: &[[f64; 2]; 3], xs: &[f64; NQP], ys: &[f64; NQP]) -> [C64; NQP] {
    let (a1, b1, c1) = (coeff[0][0], coeff[1][0], coeff[2][0]);
    let (a2, b2, c2) = (coeff[0][1], coeff[1][1], coeff[2][1]);
    let _ = a1;
    let mut out = [C64::new(0.0, 0.0); NQP];
    for i in 0..NQP {
        let x = xs[i]; let y = ys[i];
        out[i] = C64::from(
            b2*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))
            + c2*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))
        );
    }
    out
}

// Port of _divergence_face_1(coeff, coords)
fn divergence_face_1(coeff: &[[f64; 3]; 3], xs: &[f64; NQP], ys: &[f64; NQP]) -> [C64; NQP] {
    let (a1, b1, c1) = (coeff[0][0], coeff[1][0], coeff[2][0]);
    let (a2, b2, c2) = (coeff[0][1], coeff[1][1], coeff[2][1]);
    let (a3, b3, c3) = (coeff[0][2], coeff[1][2], coeff[2][2]);
    let _ = a2;
    let mut out = [C64::new(0.0, 0.0); NQP];
    for i in 0..NQP {
        let x = xs[i]; let y = ys[i];
        out[i] = C64::from(
            -b2*(b1*(a3 + b3*x + c3*y) - b3*(a1 + b1*x + c1*y))
            - c2*(c1*(a3 + b3*x + c3*y) - c3*(a1 + b1*x + c1*y))
        );
    }
    out
}

// Port of _divergence_face_2(coeff, coords)
fn divergence_face_2(coeff: &[[f64; 3]; 3], xs: &[f64; NQP], ys: &[f64; NQP]) -> [C64; NQP] {
    let (a1, b1, c1) = (coeff[0][0], coeff[1][0], coeff[2][0]);
    let (a2, b2, c2) = (coeff[0][1], coeff[1][1], coeff[2][1]);
    let (a3, b3, c3) = (coeff[0][2], coeff[1][2], coeff[2][2]);
    let _ = a3;
    let mut out = [C64::new(0.0, 0.0); NQP];
    for i in 0..NQP {
        let x = xs[i]; let y = ys[i];
        out[i] = C64::from(
            b3*(b1*(a2 + b2*x + c2*y) - b2*(a1 + b1*x + c1*y))
            + c3*(c1*(a2 + b2*x + c2*y) - c2*(a1 + b1*x + c1*y))
        );
    }
    out
}

// Port of tri_coefficients(vxs, vys) from robin_abc_order2.py lines 152-174
fn tri_coefficients(vxs: &[f64; 3], vys: &[f64; 3]) -> ([f64; 3], [f64; 3], [f64; 3], f64) {
    let (x1, x2, x3) = (vxs[0], vxs[1], vxs[2]);
    let (y1, y2, y3) = (vys[0], vys[1], vys[2]);

    let a1 = x2*y3 - y2*x3;
    let a2 = x3*y1 - y3*x1;
    let a3 = x1*y2 - y1*x2;
    let b1 = y2 - y3;
    let b2 = y3 - y1;
    let b3 = y1 - y2;
    let c1 = x3 - x2;
    let c2 = x1 - x3;
    let c3 = x2 - x1;

    let sa = 0.5 * ((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1));
    let sign = sa.signum();
    let area = sa.abs();

    ([a1*sign, a2*sign, a3*sign],
     [b1*sign, b2*sign, b3*sign],
     [c1*sign, c2*sign, c3*sign],
     area)
}

fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt();
    [a[0]/n, a[1]/n, a[2]/n]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn compute_distances_3(xs: &[f64; 3], ys: &[f64; 3], zs: &[f64; 3]) -> [[f64; 3]; 3] {
    let mut ds = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in i..3 {
            let d = ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2) + (zs[i]-zs[j]).powi(2)).sqrt();
            ds[i][j] = d; ds[j][i] = d;
        }
    }
    ds
}

/// Per-triangle real `Lengths * (Curl - Div) * |Area|` operator (the raw
/// second-order term without its `c2/k0` scaling). Symmetric 8x8. The caller
/// scales it, adds the first-order surface mass, and projects to PSD.
pub fn abc_order_2_terms(
    glob_vertices: &[[f64; 3]; 3],
    local_edge_map: &[[usize; 2]; 3],
) -> [[f64; 8]; 8] {
    let zero = C64::new(0.0, 0.0);

    // Local coordinate system (same as tri_assembly)
    let origin = glob_vertices[0];
    let v2 = glob_vertices[1];
    let v3 = glob_vertices[2];
    let e1 = [v2[0]-origin[0], v2[1]-origin[1], v2[2]-origin[2]];
    let e2 = [v3[0]-origin[0], v3[1]-origin[1], v3[2]-origin[2]];
    let zhat = normalize3(cross3(e1, e2));
    let xhat = normalize3(e1);
    let yhat = normalize3(cross3(zhat, xhat));

    // Project vertices to local 2D
    let mut xpts = [0.0; 3];
    let mut ypts = [0.0; 3];
    for i in 0..3 {
        let dx = glob_vertices[i][0] - origin[0];
        let dy = glob_vertices[i][1] - origin[1];
        let dz = glob_vertices[i][2] - origin[2];
        xpts[i] = xhat[0]*dx + xhat[1]*dy + xhat[2]*dz;
        ypts[i] = yhat[0]*dx + yhat[1]*dy + yhat[2]*dz;
    }

    let mut curl_mat = [[zero; 8]; 8];
    let mut div_mat = [[zero; 8]; 8];
    let mut lengths = [[1.0f64; 8]; 8];

    let distances = compute_distances_3(&xpts, &ypts, &[0.0; 3]);

    // Order-4 triangle quadrature - the canonical rule from `quadrature`,
    // not a hardcoded copy. Point order is irrelevant (the rule enters as a
    // weighted sum), so weight i pairs with barycentric coords i.
    let qpts = crate::quadrature::gaus_quad_tri(4);
    let mut dpts_w = [0.0; NQP];
    let mut dpts_l1 = [0.0; NQP];
    let mut dpts_l2 = [0.0; NQP];
    let mut dpts_l3 = [0.0; NQP];
    for i in 0..NQP {
        dpts_w[i] = qpts[i][0];
        dpts_l1[i] = qpts[i][1];
        dpts_l2[i] = qpts[i][2];
        dpts_l3[i] = qpts[i][3];
    }

    // Quadrature point coordinates
    let mut xs = [0.0; NQP];
    let mut ys = [0.0; NQP];
    for i in 0..NQP {
        xs[i] = xpts[0]*dpts_l1[i] + xpts[1]*dpts_l2[i] + xpts[2]*dpts_l3[i];
        ys[i] = ypts[0]*dpts_l1[i] + ypts[1]*dpts_l2[i] + ypts[2]*dpts_l3[i];
    }

    // Barycentric coefficients
    let (aas, bbs, ccs, area) = tri_coefficients(&xpts, &ypts);
    let a2 = 2.0 * area;
    let bary_coeff: [[f64; 3]; 3] = [
        [aas[0]/a2, aas[1]/a2, aas[2]/a2],
        [bbs[0]/a2, bbs[1]/a2, bbs[2]/a2],
        [ccs[0]/a2, ccs[1]/a2, ccs[2]/a2],
    ];

    // Face length factors
    lengths[3][0] *= distances[0][2]; lengths[3][1] *= distances[0][2]; lengths[3][2] *= distances[0][2];
    lengths[3][3] *= distances[0][2]; lengths[3][4] *= distances[0][2]; lengths[3][5] *= distances[0][2];
    lengths[3][6] *= distances[0][2]; lengths[3][7] *= distances[0][2];
    lengths[7][0] *= distances[0][1]; lengths[7][1] *= distances[0][1]; lengths[7][2] *= distances[0][1];
    lengths[7][3] *= distances[0][1]; lengths[7][4] *= distances[0][1]; lengths[7][5] *= distances[0][1];
    lengths[7][6] *= distances[0][1]; lengths[7][7] *= distances[0][1];
    for j in 0..8 {
        lengths[j][3] *= distances[0][2];
        lengths[j][7] *= distances[0][1];
    }

    // Face basis curl and divergence
    let ff1c = curl_face_1(&bary_coeff, &xs, &ys);
    let ff2c = curl_face_2(&bary_coeff, &xs, &ys);
    let ff1d = divergence_face_1(&bary_coeff, &xs, &ys);
    let ff2d = divergence_face_2(&bary_coeff, &xs, &ys);

    for iv1 in 0..3 {
        let ie1 = local_edge_map[iv1];
        let le = distances[ie1[0]][ie1[1]];
        for j in 0..8 { lengths[iv1][j] *= le; lengths[iv1+4][j] *= le; }
        for i in 0..8 { lengths[i][iv1] *= le; lengths[i][iv1+4] *= le; }

        // coeff[:,ie1] in EMerge = columns ie1[0],ie1[1] from 3x3 bary_coeff
        // Our layout: [row][col], so column extraction gives [3 rows][2 cols]
        let coeff_e1: [[f64; 2]; 3] = [
            [bary_coeff[0][ie1[0]], bary_coeff[0][ie1[1]]],
            [bary_coeff[1][ie1[0]], bary_coeff[1][ie1[1]]],
            [bary_coeff[2][ie1[0]], bary_coeff[2][ie1[1]]],
        ];
        let fe1c_1 = curl_edge_1(&coeff_e1, &xs, &ys);
        let fe2c_1 = curl_edge_2(&coeff_e1, &xs, &ys);
        let fe1d_1 = divergence_edge_1(&coeff_e1, &xs, &ys);
        let fe2d_1 = divergence_edge_2(&coeff_e1, &xs, &ys);

        for iv2 in 0..3 {
            let ie2 = local_edge_map[iv2];
            let coeff_e2: [[f64; 2]; 3] = [
                [bary_coeff[0][ie2[0]], bary_coeff[0][ie2[1]]],
                [bary_coeff[1][ie2[0]], bary_coeff[1][ie2[1]]],
                [bary_coeff[2][ie2[0]], bary_coeff[2][ie2[1]]],
            ];
            let fe1c_2 = curl_edge_1(&coeff_e2, &xs, &ys);
            let fe2c_2 = curl_edge_2(&coeff_e2, &xs, &ys);
            let fe1d_2 = divergence_edge_1(&coeff_e2, &xs, &ys);
            let fe2d_2 = divergence_edge_2(&coeff_e2, &xs, &ys);

            curl_mat[iv1][iv2]     = gqi(&fe1c_1, &fe1c_2, &dpts_w);
            curl_mat[iv1][iv2+4]   = gqi(&fe1c_1, &fe2c_2, &dpts_w);
            curl_mat[iv1+4][iv2]   = gqi(&fe2c_1, &fe1c_2, &dpts_w);
            curl_mat[iv1+4][iv2+4] = gqi(&fe2c_1, &fe2c_2, &dpts_w);

            div_mat[iv1][iv2]     = gqi(&fe1d_1, &fe1d_2, &dpts_w);
            div_mat[iv1][iv2+4]   = gqi(&fe1d_1, &fe2d_2, &dpts_w);
            div_mat[iv1+4][iv2]   = gqi(&fe2d_1, &fe1d_2, &dpts_w);
            div_mat[iv1+4][iv2+4] = gqi(&fe2d_1, &fe2d_2, &dpts_w);
        }

        // Edge-face interactions
        curl_mat[iv1][3]     = gqi(&fe1c_1, &ff1c, &dpts_w);
        curl_mat[iv1+4][3]   = gqi(&fe2c_1, &ff1c, &dpts_w);
        curl_mat[iv1][7]     = gqi(&fe1c_1, &ff2c, &dpts_w);
        curl_mat[iv1+4][7]   = gqi(&fe2c_1, &ff2c, &dpts_w);
        curl_mat[3][iv1]     = curl_mat[iv1][3];
        curl_mat[3][iv1+4]   = curl_mat[iv1+4][3];
        curl_mat[7][iv1]     = curl_mat[iv1][7];
        curl_mat[7][iv1+4]   = curl_mat[iv1+4][7];

        div_mat[iv1][3]     = gqi(&fe1d_1, &ff1d, &dpts_w);
        div_mat[iv1+4][3]   = gqi(&fe2d_1, &ff1d, &dpts_w);
        div_mat[iv1][7]     = gqi(&fe1d_1, &ff2d, &dpts_w);
        div_mat[iv1+4][7]   = gqi(&fe2d_1, &ff2d, &dpts_w);
        div_mat[3][iv1]     = div_mat[iv1][3];
        div_mat[3][iv1+4]   = div_mat[iv1+4][3];
        div_mat[7][iv1]     = div_mat[iv1][7];
        div_mat[7][iv1+4]   = div_mat[iv1+4][7];
    }

    // Face-face interactions
    curl_mat[3][3] = gqi(&ff1c, &ff1c, &dpts_w);
    curl_mat[3][7] = gqi(&ff1c, &ff2c, &dpts_w);
    curl_mat[7][3] = gqi(&ff2c, &ff1c, &dpts_w);
    curl_mat[7][7] = gqi(&ff2c, &ff2c, &dpts_w);
    div_mat[3][3] = gqi(&ff1d, &ff1d, &dpts_w);
    div_mat[3][7] = gqi(&ff1d, &ff2d, &dpts_w);
    div_mat[7][3] = gqi(&ff2d, &ff1d, &dpts_w);
    div_mat[7][7] = gqi(&ff2d, &ff2d, &dpts_w);

    // R = Lengths * (CurlMatrix - DivMatrix) * |Area|, real (the gqi inner
    // products of real polynomials carry no imaginary part).
    let mut r = [[0.0f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            r[i][j] = lengths[i][j] * (curl_mat[i][j] - div_mat[i][j]).re * area;
        }
    }
    r
}

/// Project a symmetric 8x8 onto the nearest positive-semidefinite matrix:
/// eigen-decompose (cyclic Jacobi), clamp negative eigenvalues to zero, and
/// reconstruct. Reduces to the input when it is already PSD.
fn psd_project_8(a: &[[f64; 8]; 8]) -> [[f64; 8]; 8] {
    // Work on a symmetrized copy; accumulate eigenvectors in `v`.
    let mut m = [[0.0f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            m[i][j] = 0.5 * (a[i][j] + a[j][i]);
        }
    }
    let mut v = [[0.0f64; 8]; 8];
    for i in 0..8 {
        v[i][i] = 1.0;
    }
    // Cyclic Jacobi sweeps. 8x8 converges in a handful of sweeps; 60 is a hard
    // cap that is never reached in practice.
    for _ in 0..60 {
        let mut off = 0.0f64;
        for p in 0..8 {
            for q in (p + 1)..8 {
                off += m[p][q] * m[p][q];
            }
        }
        if off <= 1e-28 {
            break;
        }
        for p in 0..8 {
            for q in (p + 1)..8 {
                let apq = m[p][q];
                if apq == 0.0 {
                    continue;
                }
                // Rotation that zeroes m[p][q] (Golub & Van Loan, sym. Schur).
                let tau = (m[q][q] - m[p][p]) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                // M <- Jᵀ M J  (apply to columns then rows of m, both sides).
                for k in 0..8 {
                    let g = m[k][p];
                    let h = m[k][q];
                    m[k][p] = c * g - s * h;
                    m[k][q] = s * g + c * h;
                }
                for k in 0..8 {
                    let g = m[p][k];
                    let h = m[q][k];
                    m[p][k] = c * g - s * h;
                    m[q][k] = s * g + c * h;
                }
                for k in 0..8 {
                    let g = v[k][p];
                    let h = v[k][q];
                    v[k][p] = c * g - s * h;
                    v[k][q] = s * g + c * h;
                }
            }
        }
    }
    // Reconstruct V * clamp(Λ, 0) * Vᵀ.
    let mut out = [[0.0f64; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            let mut sum = 0.0f64;
            for k in 0..8 {
                let lam = m[k][k].max(0.0);
                sum += v[i][k] * lam * v[j][k];
            }
            out[i][j] = sum;
        }
    }
    out
}

/// Assemble the passivity-projected order-2 ABC correction into the flat tri
/// matrix (same layout as Bempty). For each boundary triangle it combines the
/// raw second-order term with the first-order surface mass `B1`, projects the
/// resulting imaginary boundary operator onto the PSD cone, and returns the
/// correction `j*(H_psd − k0*c1*B1)` so that, added to the first-order Robin
/// term `j*k0*c1*B1` already in Bempty, the total boundary contribution is the
/// guaranteed-passive `j*H_psd`.
///
/// `c1`, `c2` are the order-2 coefficients (the same pair used by `get_gamma`
/// and the raw correction); `ac_base` is the shared area-coefficient cache the
/// Robin assembly already builds.
pub fn abc_order_2_matrix(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    tri_ids: &[usize],
    k0: f64,
    c1: f64,
    c2: f64,
    ac_base: &AreaCoeffCache,
) -> Vec<C64> {
    let mut mat = vec![C64::new(0.0, 0.0); basis.n_tris * 64];
    let kc1 = k0 * c1;       // first-order scale
    let c2k = c2 / k0;       // second-order scale

    for &itri in tri_ids {
        let tri = &mesh.tris[itri];
        let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];

        // Local edge mapping (same as robin BC: edges from tri_to_edge)
        let edges = &mesh.tri_to_edge[itri];
        let global_edge_nodes: [[usize; 2]; 3] = std::array::from_fn(|i| mesh.edges[edges[i]]);
        let mut local_edge_map = [[0usize; 2]; 3];
        for (i, pair) in global_edge_nodes.iter().enumerate() {
            for (j, &gid) in pair.iter().enumerate() {
                for k in 0..3 {
                    if tri[k] == gid {
                        local_edge_map[i][j] = k;
                        break;
                    }
                }
            }
        }

        // Raw second-order operator (real) and first-order surface mass B1.
        // `ned2_tri_stiff(.., gamma=1)` returns B1 in the same 8-DOF layout, so
        // the two combine directly.
        let r = abc_order_2_terms(&verts, &local_edge_map);
        let b1 = ned2_tri_stiff(&verts, C64::new(1.0, 0.0), ac_base);

        // Combined imaginary boundary operator, then PSD projection.
        let mut h = [[0.0f64; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                h[i][j] = kc1 * b1[i][j].re + c2k * r[i][j];
            }
        }
        let h_psd = psd_project_8(&h);

        // correction = j*(H_psd − k0*c1*B1)
        let p = itri * 64;
        for i in 0..8 {
            for j in 0..8 {
                let corr = h_psd[i][j] - kc1 * b1[i][j].re;
                mat[p + i * 8 + j] += C64::new(0.0, corr);
            }
        }
    }

    mat
}

#[cfg(test)]
mod tests {
    use super::psd_project_8;

    /// Minimum quadratic form x'Mx / x'x over the rows of a probe basis —
    /// a cheap negative-definiteness detector (a PSD matrix gives >= 0).
    fn min_rayleigh(m: &[[f64; 8]; 8], probes: &[[f64; 8]]) -> f64 {
        let mut lo = f64::INFINITY;
        for x in probes {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..8 {
                den += x[i] * x[i];
                for j in 0..8 {
                    num += x[i] * m[i][j] * x[j];
                }
            }
            if den > 0.0 {
                lo = lo.min(num / den);
            }
        }
        lo
    }

    #[test]
    fn psd_project_clamps_diagonal() {
        // In its own eigenbasis the projection must clamp negatives to zero.
        let diag = [3.0, -2.0, 0.0, 5.0, -0.1, 1.0, -7.0, 2.0];
        let mut a = [[0.0f64; 8]; 8];
        for i in 0..8 {
            a[i][i] = diag[i];
        }
        let out = psd_project_8(&a);
        for i in 0..8 {
            for j in 0..8 {
                let want = if i == j { diag[i].max(0.0) } else { 0.0 };
                assert!((out[i][j] - want).abs() < 1e-9, "[{i}][{j}]={} want {want}", out[i][j]);
            }
        }
    }

    #[test]
    fn psd_project_is_identity_on_psd() {
        // B = M Mᵀ is PSD; projecting it changes nothing.
        let mut m = [[0.0f64; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                m[i][j] = ((i * 7 + j * 3) % 5) as f64 - 2.0;
            }
        }
        let mut b = [[0.0f64; 8]; 8];
        for i in 0..8 {
            for j in 0..8 {
                let mut s = 0.0;
                for k in 0..8 {
                    s += m[i][k] * m[j][k];
                }
                b[i][j] = s;
            }
        }
        let out = psd_project_8(&b);
        for i in 0..8 {
            for j in 0..8 {
                assert!((out[i][j] - b[i][j]).abs() < 1e-6, "[{i}][{j}] {} vs {}", out[i][j], b[i][j]);
            }
        }
    }

    #[test]
    fn psd_project_output_is_psd_when_rotated() {
        // Indefinite A = R D Rᵀ (D has a -3 eigenvalue), rotated in plane (0,3).
        // The projection must reproduce R clamp(D) Rᵀ and be PSD.
        let (c, s) = (0.6f64.cos(), 0.6f64.sin());
        let d = [4.0, 1.0, 2.0, -3.0, 0.5, 1.5, 2.5, 0.7];
        let mut a = [[0.0f64; 8]; 8];
        for i in 0..8 {
            a[i][i] = d[i];
        }
        a[0][0] = c * c * d[0] + s * s * d[3];
        a[3][3] = s * s * d[0] + c * c * d[3];
        a[0][3] = c * s * (d[0] - d[3]);
        a[3][0] = a[0][3];

        let out = psd_project_8(&a);

        // Expected: same construction with d[3] clamped to 0.
        let dc = [4.0, 1.0, 2.0, 0.0, 0.5, 1.5, 2.5, 0.7];
        let mut exp = [[0.0f64; 8]; 8];
        for i in 0..8 {
            exp[i][i] = dc[i];
        }
        exp[0][0] = c * c * dc[0] + s * s * dc[3];
        exp[3][3] = s * s * dc[0] + c * c * dc[3];
        exp[0][3] = c * s * (dc[0] - dc[3]);
        exp[3][0] = exp[0][3];
        for i in 0..8 {
            for j in 0..8 {
                assert!((out[i][j] - exp[i][j]).abs() < 1e-7, "[{i}][{j}] {} vs {}", out[i][j], exp[i][j]);
            }
        }

        // And it is numerically PSD on the canonical + rotated-axis probes.
        let mut probes: Vec<[f64; 8]> = (0..8)
            .map(|k| {
                let mut e = [0.0; 8];
                e[k] = 1.0;
                e
            })
            .collect();
        let mut rot = [0.0; 8];
        rot[0] = -s;
        rot[3] = c; // the originally-negative eigenvector
        probes.push(rot);
        assert!(min_rayleigh(&out, &probes) >= -1e-9);
    }
}
