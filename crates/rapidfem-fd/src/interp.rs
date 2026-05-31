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

//! Exact port of compiled/base/interp.py: ned2_tet_interp
//!
//! Evaluates E-field at a point inside a known tetrahedron using
//! the full Nedelec-2 basis (20 DOFs).

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;

/// Port of interp.py: tet_coefficients(xs, ys, zs)
/// Returns (aas, bbs, ccs, dds, V) — the FULL barycentric coefficients including constant term.
pub fn tet_coefficients(xs: &[f64; 4], ys: &[f64; 4], zs: &[f64; 4])
    -> ([f64; 4], [f64; 4], [f64; 4], [f64; 4], f64)
{
    let (x1,x2,x3,x4) = (xs[0],xs[1],xs[2],xs[3]);
    let (y1,y2,y3,y4) = (ys[0],ys[1],ys[2],ys[3]);
    let (z1,z2,z3,z4) = (zs[0],zs[1],zs[2],zs[3]);

    let v = (-x1*y2*z3/6.0 + x1*y2*z4/6.0 + x1*y3*z2/6.0 - x1*y3*z4/6.0 - x1*y4*z2/6.0 +
              x1*y4*z3/6.0 + x2*y1*z3/6.0 - x2*y1*z4/6.0 - x2*y3*z1/6.0 + x2*y3*z4/6.0 +
              x2*y4*z1/6.0 - x2*y4*z3/6.0 - x3*y1*z2/6.0 + x3*y1*z4/6.0 + x3*y2*z1/6.0 -
              x3*y2*z4/6.0 - x3*y4*z1/6.0 + x3*y4*z2/6.0 + x4*y1*z2/6.0 - x4*y1*z3/6.0 -
              x4*y2*z1/6.0 + x4*y2*z3/6.0 + x4*y3*z1/6.0 - x4*y3*z2/6.0).abs();

    let aas = [
         x2*y3*z4 - x2*y4*z3 - x3*y2*z4 + x3*y4*z2 + x4*y2*z3 - x4*y3*z2,
        -x1*y3*z4 + x1*y4*z3 + x3*y1*z4 - x3*y4*z1 - x4*y1*z3 + x4*y3*z1,
         x1*y2*z4 - x1*y4*z2 - x2*y1*z4 + x2*y4*z1 + x4*y1*z2 - x4*y2*z1,
        -x1*y2*z3 + x1*y3*z2 + x2*y1*z3 - x2*y3*z1 - x3*y1*z2 + x3*y2*z1,
    ];
    let bbs = [
        -y2*z3 + y2*z4 + y3*z2 - y3*z4 - y4*z2 + y4*z3,
         y1*z3 - y1*z4 - y3*z1 + y3*z4 + y4*z1 - y4*z3,
        -y1*z2 + y1*z4 + y2*z1 - y2*z4 - y4*z1 + y4*z2,
         y1*z2 - y1*z3 - y2*z1 + y2*z3 + y3*z1 - y3*z2,
    ];
    let ccs = [
         x2*z3 - x2*z4 - x3*z2 + x3*z4 + x4*z2 - x4*z3,
        -x1*z3 + x1*z4 + x3*z1 - x3*z4 - x4*z1 + x4*z3,
         x1*z2 - x1*z4 - x2*z1 + x2*z4 + x4*z1 - x4*z2,
        -x1*z2 + x1*z3 + x2*z1 - x2*z3 - x3*z1 + x3*z2,
    ];
    let dds = [
        -x2*y3 + x2*y4 + x3*y2 - x3*y4 - x4*y2 + x4*y3,
         x1*y3 - x1*y4 - x3*y1 + x3*y4 + x4*y1 - x4*y3,
        -x1*y2 + x1*y4 + x2*y1 - x2*y4 - x4*y1 + x4*y2,
         x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2,
    ];

    (aas, bbs, ccs, dds, v)
}

/// Port of interp.py: ned2_tet_interp (core evaluation loop)
///
/// Evaluate E-field at point (x,y,z) inside tet `tet_idx`.
/// Uses solution DOF values and Nedelec-2 basis functions.
///
/// Exactly matches EMerge's evaluation:
/// - Edge modes: E += LV * (Em1*F1 + Em2*F2) * (∇λ₁*F2 - ∇λ₂*F1)
/// - Face modes: E += V1 * (-Ef1*L1*F2*(∇λ₁*F3-∇λ₃*F1) + Ef2*L2*F3*(∇λ₁*F2-∇λ₂*F1))
pub fn eval_field_in_tet(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    solution: &[C64],
    tet_idx: usize,
    x: f64, y: f64, z: f64,
) -> (C64, C64, C64) {
    let tet = &mesh.tets[tet_idx];
    let xs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][0]);
    let ys: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][1]);
    let zs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][2]);

    let (a_s, b_s, c_s, d_s, v) = tet_coefficients(&xs, &ys, &zs);

    // Distance matrix (4x4)
    let mut ds = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in i..4 {
            let d = ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2) + (zs[i]-zs[j]).powi(2)).sqrt();
            ds[i][j] = d; ds[j][i] = d;
        }
    }

    // Local edge mapping (global edge node IDs → local tet node indices)
    let tet_edges = &mesh.tet_to_edge[tet_idx];
    let global_edge_nodes: [[usize; 2]; 6] = std::array::from_fn(|i| mesh.edges[tet_edges[i]]);
    let l_edge_ids = crate::basis::local_mapping(tet, &global_edge_nodes);

    // Local face mapping
    let tet_tris = &mesh.tet_to_tri[tet_idx];
    let global_tri_nodes: [[usize; 3]; 4] = std::array::from_fn(|i| mesh.tris[tet_tris[i]]);
    let l_tri_ids = crate::basis::local_mapping_tri(tet, &global_tri_nodes);

    // DOF values
    let field_ids = &basis.tet_to_field[tet_idx];
    let em1s: [C64; 6] = std::array::from_fn(|i| solution[field_ids[i]]);       // Etet[0:6]
    let ef1s: [C64; 4] = std::array::from_fn(|i| solution[field_ids[6 + i]]);   // Etet[6:10]
    let em2s: [C64; 6] = std::array::from_fn(|i| solution[field_ids[10 + i]]);  // Etet[10:16]
    let ef2s: [C64; 4] = std::array::from_fn(|i| solution[field_ids[16 + i]]);  // Etet[16:20]

    let v1 = 1.0 / (216.0 * v * v * v);

    let mut ex = C64::new(0.0, 0.0);
    let mut ey = C64::new(0.0, 0.0);
    let mut ez = C64::new(0.0, 0.0);

    // Edge basis functions (6 edges)
    for ie in 0..6 {
        let em1 = em1s[ie];
        let em2 = em2s[ie];
        let n1 = l_edge_ids[ie][0];
        let n2 = l_edge_ids[ie][1];
        let (a1, a2) = (a_s[n1], a_s[n2]);
        let (b1, b2) = (b_s[n1], b_s[n2]);
        let (c1, c2) = (c_s[n1], c_s[n2]);
        let (d1, d2) = (d_s[n1], d_s[n2]);

        let lv = ds[n1][n2] * v1;

        let f1 = a1 + b1*x + c1*y + d1*z;
        let f2 = a2 + b2*x + c2*y + d2*z;
        let f3 = em1 * C64::from(f1) + em2 * C64::from(f2);

        ex += f3 * C64::from(lv * (b1*f2 - b2*f1));
        ey += f3 * C64::from(lv * (c1*f2 - c2*f1));
        ez += f3 * C64::from(lv * (d1*f2 - d2*f1));
    }

    // Face basis functions (4 faces)
    for ie in 0..4 {
        let em1 = ef1s[ie];
        let em2 = ef2s[ie];
        let n1 = l_tri_ids[ie][0];
        let n2 = l_tri_ids[ie][1];
        let n3 = l_tri_ids[ie][2];
        let (a1, a2, a3) = (a_s[n1], a_s[n2], a_s[n3]);
        let (b1, b2, b3) = (b_s[n1], b_s[n2], b_s[n3]);
        let (c1, c2, c3) = (c_s[n1], c_s[n2], c_s[n3]);
        let (d1, d2, d3) = (d_s[n1], d_s[n2], d_s[n3]);

        let l1 = ds[l_tri_ids[ie][2]][l_tri_ids[ie][0]]; // Ds[n3, n1]
        let l2 = ds[l_tri_ids[ie][1]][l_tri_ids[ie][0]]; // Ds[n2, n1]

        let f1 = a1 + b1*x + c1*y + d1*z;
        let f2 = a2 + b2*x + c2*y + d2*z;
        let f3 = a3 + b3*x + c3*y + d3*z;

        let q1 = em1 * C64::from(l1 * f2);
        let q2 = em2 * C64::from(l2 * f3);

        ex += (-q1 * C64::from(b1*f3 - b3*f1) + q2 * C64::from(b1*f2 - b2*f1)) * C64::from(v1);
        ey += (-q1 * C64::from(c1*f3 - c3*f1) + q2 * C64::from(c1*f2 - c2*f1)) * C64::from(v1);
        ez += (-q1 * C64::from(d1*f3 - d3*f1) + q2 * C64::from(d1*f2 - d2*f1)) * C64::from(v1);
    }

    (ex, ey, ez)
}

/// Spatial hash grid for fast point-in-tet lookup.
pub struct TetGrid {
    cells: hashbrown::HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size: f64,
}

impl TetGrid {
    /// Build a spatial grid from the mesh. Each tet is assigned to the cell containing its centroid.
    pub fn new(mesh: &Mesh) -> Self {
        // Compute bounding box
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for node in &mesh.nodes {
            for k in 0..3 { min[k] = min[k].min(node[k]); max[k] = max[k].max(node[k]); }
        }
        let diag = ((max[0]-min[0]).powi(2) + (max[1]-min[1]).powi(2) + (max[2]-min[2]).powi(2)).sqrt();
        let cell_size = diag / (mesh.n_tets() as f64).cbrt().max(2.0);

        let mut cells: hashbrown::HashMap<(i32, i32, i32), Vec<usize>> = hashbrown::HashMap::new();
        for itet in 0..mesh.n_tets() {
            let tet = &mesh.tets[itet];
            let cx = (mesh.nodes[tet[0]][0] + mesh.nodes[tet[1]][0] + mesh.nodes[tet[2]][0] + mesh.nodes[tet[3]][0]) / 4.0;
            let cy = (mesh.nodes[tet[0]][1] + mesh.nodes[tet[1]][1] + mesh.nodes[tet[2]][1] + mesh.nodes[tet[3]][1]) / 4.0;
            let cz = (mesh.nodes[tet[0]][2] + mesh.nodes[tet[1]][2] + mesh.nodes[tet[2]][2] + mesh.nodes[tet[3]][2]) / 4.0;
            let key = ((cx / cell_size).floor() as i32, (cy / cell_size).floor() as i32, (cz / cell_size).floor() as i32);
            cells.entry(key).or_default().push(itet);
        }

        TetGrid { cells, cell_size }
    }

    /// Find the tet containing a point using the spatial grid. Falls back to brute force if not found.
    pub fn find_containing_tet(&self, mesh: &Mesh, x: f64, y: f64, z: f64) -> Option<usize> {
        let cs = self.cell_size;
        let cx = (x / cs).floor() as i32;
        let cy = (y / cs).floor() as i32;
        let cz = (z / cs).floor() as i32;

        // Search 3x3x3 neighborhood
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(tets) = self.cells.get(&(cx+dx, cy+dy, cz+dz)) {
                        for &itet in tets {
                            if point_in_tet(mesh, itet, x, y, z) {
                                return Some(itet);
                            }
                        }
                    }
                }
            }
        }
        // Fallback: brute force (handles edge cases)
        find_containing_tet_brute(mesh, x, y, z)
    }
}

fn point_in_tet(mesh: &Mesh, itet: usize, x: f64, y: f64, z: f64) -> bool {
    let eps = crate::constants::POINT_IN_TET_EPS;
    let tet = &mesh.tets[itet];
    let v1 = mesh.nodes[tet[0]];
    let v2 = mesh.nodes[tet[1]];
    let v3 = mesh.nodes[tet[2]];
    let v4 = mesh.nodes[tet[3]];

    let m00 = v2[0]-v1[0]; let m01 = v3[0]-v1[0]; let m02 = v4[0]-v1[0];
    let m10 = v2[1]-v1[1]; let m11 = v3[1]-v1[1]; let m12 = v4[1]-v1[1];
    let m20 = v2[2]-v1[2]; let m21 = v3[2]-v1[2]; let m22 = v4[2]-v1[2];

    let det = m00*(m11*m22 - m12*m21) - m01*(m10*m22 - m12*m20) + m02*(m10*m21 - m11*m20);
    if det.abs() < crate::constants::SINGULAR_EPS { return false; }
    let inv_det = 1.0 / det;

    let dx = x - v1[0];
    let dy = y - v1[1];
    let dz = z - v1[2];

    let u = ((m11*m22-m12*m21)*dx + (m02*m21-m01*m22)*dy + (m01*m12-m02*m11)*dz) * inv_det;
    let v = ((m12*m20-m10*m22)*dx + (m00*m22-m02*m20)*dy + (m02*m10-m00*m12)*dz) * inv_det;
    let w = ((m10*m21-m11*m20)*dx + (m01*m20-m00*m21)*dy + (m00*m11-m01*m10)*dz) * inv_det;

    u >= -eps && v >= -eps && w >= -eps && u + v + w <= 1.0 + eps
}

/// Brute-force fallback for find_containing_tet.
fn find_containing_tet_brute(mesh: &Mesh, x: f64, y: f64, z: f64) -> Option<usize> {
    for itet in 0..mesh.n_tets() {
        if point_in_tet(mesh, itet, x, y, z) {
            return Some(itet);
        }
    }
    None
}

/// Find the tet containing a point (brute force — for backward compatibility).
pub fn find_containing_tet(mesh: &Mesh, x: f64, y: f64, z: f64) -> Option<usize> {
    find_containing_tet_brute(mesh, x, y, z)
}

/// Analytic curl of the FEM E-field inside a known tet at point `(x, y, z)`.
///
/// Derived directly from the Nédélec-2 basis functions reconstructed by
/// `eval_field_in_tet`:
///
///   φ_e1 = lv·(f1·f2·g1 − f1²·g2)
///   φ_e2 = lv·(f2²·g1 − f1·f2·g2)
///   φ_f1 = v1·l1·(f1·f2·g3 − f2·f3·g1)
///   φ_f2 = v1·l2·(f2·f3·g1 − f1·f3·g2)
///
/// where f_i = a_i + b_i·x + c_i·y + d_i·z is the *unscaled* barycentric
/// coordinate of node i (six times the tet volume × the normalised λ_i) and
/// g_i = (b_i, c_i, d_i) is the unscaled gradient ∇λ_i. Applying
/// `∇×(α·g) = ∇α × g` (g constant per tet) and simplifying:
///
///   curl(φ_e1) = −3·lv·f1·(g1×g2)
///   curl(φ_e2) = −3·lv·f2·(g1×g2)
///   curl(φ_f1) =  v1·l1·[f1·(g2×g3) + 2·f2·(g1×g3) + f3·(g1×g2)]
///   curl(φ_f2) =  v1·l2·[f1·(g2×g3) −   f2·(g1×g3) − 2·f3·(g1×g2)]
///
/// All four pieces are linear in position via f1, f2, f3 — there is no
/// constant-per-tet approximation. Used by the error estimator, the
/// far-field integration, and the H-field channel (H = ∇×E / (jωμ)).
pub fn eval_curl_in_tet(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    solution: &[C64],
    tet_idx: usize,
    x: f64, y: f64, z: f64,
) -> [C64; 3] {
    let tet = &mesh.tets[tet_idx];
    let xs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][0]);
    let ys: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][1]);
    let zs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][2]);

    let (a_s, b_s, c_s, d_s, v) = tet_coefficients(&xs, &ys, &zs);

    // Unscaled barycentric coordinate of local node i at (x, y, z).
    let lam = |i: usize| -> f64 { a_s[i] + b_s[i]*x + c_s[i]*y + d_s[i]*z };
    let grad = |i: usize| -> [f64; 3] { [b_s[i], c_s[i], d_s[i]] };
    let cross = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
        [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    };

    let mut ds = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in i..4 {
            let d = ((xs[i]-xs[j]).powi(2)+(ys[i]-ys[j]).powi(2)+(zs[i]-zs[j]).powi(2)).sqrt();
            ds[i][j] = d; ds[j][i] = d;
        }
    }

    let tet_edges = &mesh.tet_to_edge[tet_idx];
    let global_edge_nodes: [[usize; 2]; 6] = std::array::from_fn(|i| mesh.edges[tet_edges[i]]);
    let l_edge = crate::basis::local_mapping(tet, &global_edge_nodes);

    let tet_tris = &mesh.tet_to_tri[tet_idx];
    let global_tri_nodes: [[usize; 3]; 4] = std::array::from_fn(|i| mesh.tris[tet_tris[i]]);
    let l_tri = crate::basis::local_mapping_tri(tet, &global_tri_nodes);

    let field_ids = &basis.tet_to_field[tet_idx];
    let v1 = 1.0 / (216.0 * v * v * v);

    let mut curl = [C64::new(0.0, 0.0); 3];

    // Edge modes: curl(E_edge) = −3·lv·(em1·f1 + em2·f2)·(g_n1 × g_n2)
    for ie in 0..6 {
        let n1 = l_edge[ie][0];
        let n2 = l_edge[ie][1];
        let em1 = solution[field_ids[ie]];
        let em2 = solution[field_ids[10 + ie]];
        let le = ds[n1][n2];
        let lv = le * v1;
        let cr = cross(grad(n1), grad(n2));
        let f3 = em1 * C64::from(lam(n1)) + em2 * C64::from(lam(n2));
        let coeff = f3 * C64::from(-3.0 * lv);
        for k in 0..3 {
            curl[k] += coeff * C64::from(cr[k]);
        }
    }

    // Face modes: linear in position, evaluated exactly at (x, y, z).
    for ie in 0..4 {
        let n1 = l_tri[ie][0];
        let n2 = l_tri[ie][1];
        let n3 = l_tri[ie][2];
        let ef1 = solution[field_ids[6 + ie]];
        let ef2 = solution[field_ids[16 + ie]];

        let l1 = ds[l_tri[ie][2]][l_tri[ie][0]]; // distance n3 ↔ n1
        let l2 = ds[l_tri[ie][1]][l_tri[ie][0]]; // distance n2 ↔ n1

        let cr12 = cross(grad(n1), grad(n2));
        let cr13 = cross(grad(n1), grad(n3));
        let cr23 = cross(grad(n2), grad(n3));

        let f1 = lam(n1);
        let f2 = lam(n2);
        let f3 = lam(n3);

        let c1 = ef1 * C64::from(v1 * l1);
        let c2 = ef2 * C64::from(v1 * l2);

        for k in 0..3 {
            // curl(φ_f1) = v1·l1·[ f1·(g2×g3) + 2·f2·(g1×g3) + f3·(g1×g2)]
            let term1 = f1 * cr23[k] + 2.0 * f2 * cr13[k] + f3 * cr12[k];
            // curl(φ_f2) = v1·l2·[ f1·(g2×g3) −   f2·(g1×g3) − 2·f3·(g1×g2)]
            let term2 = f1 * cr23[k] - f2 * cr13[k] - 2.0 * f3 * cr12[k];
            curl[k] += c1 * C64::from(term1);
            curl[k] += c2 * C64::from(term2);
        }
    }

    curl
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::Nedelec2Basis;
    use crate::mesh::Mesh;

    /// Build a single-tet mesh with non-degenerate vertices.
    fn single_tet_mesh() -> Mesh {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let tets = vec![[0, 1, 2, 3]];
        Mesh::from_tets(nodes, tets)
    }

    /// Numerical 6-point central-difference curl of `eval_field_in_tet`. Used
    /// as a ground-truth reference for the analytic curl: the Nédélec-2 field
    /// is at most quadratic in position, so its curl is at most linear, and
    /// central differences are EXACT (to FP rounding) for linear functions.
    fn numerical_curl(
        mesh: &Mesh,
        basis: &Nedelec2Basis,
        solution: &[C64],
        tet_idx: usize,
        x: f64, y: f64, z: f64,
        h: f64,
    ) -> [C64; 3] {
        let eval = |x, y, z| eval_field_in_tet(mesh, basis, solution, tet_idx, x, y, z);
        let (_, eyp_x, ezp_x) = eval(x + h, y, z);
        let (_, eym_x, ezm_x) = eval(x - h, y, z);
        let (exp_y, _, ezp_y) = eval(x, y + h, z);
        let (exm_y, _, ezm_y) = eval(x, y - h, z);
        let (exp_z, eyp_z, _) = eval(x, y, z + h);
        let (exm_z, eym_z, _) = eval(x, y, z - h);
        let inv2h = C64::from(0.5 / h);
        let d_ez_dy = (ezp_y - ezm_y) * inv2h;
        let d_ey_dz = (eyp_z - eym_z) * inv2h;
        let d_ex_dz = (exp_z - exm_z) * inv2h;
        let d_ez_dx = (ezp_x - ezm_x) * inv2h;
        let d_ey_dx = (eyp_x - eym_x) * inv2h;
        let d_ex_dy = (exp_y - exm_y) * inv2h;
        [
            d_ez_dy - d_ey_dz,
            d_ex_dz - d_ez_dx,
            d_ey_dx - d_ex_dy,
        ]
    }

    /// Analytic curl must agree with the FD reference to FP precision at any
    /// point inside the tet, for any DOF configuration. Sweeps several
    /// (x, y, z) and a deterministic-pseudo-random DOF vector touching every
    /// basis function (6 edges × 2 modes + 4 faces × 2 modes = 20 DOFs).
    #[test]
    fn analytic_curl_matches_numerical() {
        let mesh = single_tet_mesh();
        let basis = Nedelec2Basis::new(&mesh);
        assert_eq!(basis.n_field, 20);

        // Deterministic seed — every DOF gets a distinct, non-trivial complex value.
        let solution: Vec<C64> = (0..20)
            .map(|i| C64::new(0.37 + 0.11 * i as f64, -0.21 + 0.07 * i as f64))
            .collect();

        // Sample at several interior points, including off-centre ones that
        // expose the linear position dependence.
        let pts = [
            (0.25, 0.25, 0.25),  // centroid
            (0.10, 0.20, 0.50),
            (0.50, 0.10, 0.10),
            (0.05, 0.05, 0.85),
        ];
        let h = 1e-3;
        for &(x, y, z) in &pts {
            let analytic = eval_curl_in_tet(&mesh, &basis, &solution, 0, x, y, z);
            let numerical = numerical_curl(&mesh, &basis, &solution, 0, x, y, z, h);
            for k in 0..3 {
                let diff = (analytic[k] - numerical[k]).norm();
                let scale = analytic[k].norm().max(numerical[k].norm()).max(1e-12);
                assert!(
                    diff / scale < 1e-6,
                    "component {k} at ({x},{y},{z}): analytic={:?}, numerical={:?}, rel_err={}",
                    analytic[k], numerical[k], diff / scale,
                );
            }
        }
    }
}
