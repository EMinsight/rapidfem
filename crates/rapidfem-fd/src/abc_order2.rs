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

//! Exact port of robin_abc_order2.py: second-order ABC correction matrix.
//!
//! Computes Mat = coeff * Lengths * (CurlMatrix - DivMatrix) * |Area| per triangle.
//! Uses curl and divergence of Nedelec-2 surface basis functions.

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;

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

/// Exact port of _abc_order_2_terms(tri_vertices, local_edge_map, cf)
/// Returns 8x8 complex correction matrix for one triangle.
pub fn abc_order_2_terms(
    glob_vertices: &[[f64; 3]; 3],
    local_edge_map: &[[usize; 2]; 3],
    cf: C64,
) -> [[C64; 8]; 8] {
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

    // Mat = cf * Lengths * (CurlMatrix - DivMatrix) * |Area|
    let mut mat = [[zero; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            mat[i][j] = cf * C64::from(lengths[i][j]) * (curl_mat[i][j] - div_mat[i][j]) * C64::from(area);
        }
    }
    mat
}

/// Port of abc_order_2_matrix: assemble order-2 ABC correction into the flat tri matrix.
/// Returns the correction as a flat array (same format as Bempty).
pub fn abc_order_2_matrix(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    tri_ids: &[usize],
    coeff: C64,
) -> Vec<C64> {
    let mut mat = vec![C64::new(0.0, 0.0); basis.n_tris * 64];

    for &itri in tri_ids {
        let tri = &mesh.tris[itri];
        let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];

        // Local edge mapping (same as robin BC: edges from tri_to_edge)
        let edges = &mesh.tri_to_edge[itri];
        let global_edge_nodes: [[usize; 2]; 3] = std::array::from_fn(|i| mesh.edges[edges[i]]);
        // Convert to local tri node indices
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

        let sub_mat = abc_order_2_terms(&verts, &local_edge_map, coeff);

        let p = itri * 64;
        for ii in 0..8 {
            for jj in 0..8 {
                mat[p + ii * 8 + jj] += sub_mat[ii][jj];
            }
        }
    }

    mat
}
