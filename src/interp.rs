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
    let eps = 1e-8;
    let tet = &mesh.tets[itet];
    let v1 = mesh.nodes[tet[0]];
    let v2 = mesh.nodes[tet[1]];
    let v3 = mesh.nodes[tet[2]];
    let v4 = mesh.nodes[tet[3]];

    let m00 = v2[0]-v1[0]; let m01 = v3[0]-v1[0]; let m02 = v4[0]-v1[0];
    let m10 = v2[1]-v1[1]; let m11 = v3[1]-v1[1]; let m12 = v4[1]-v1[1];
    let m20 = v2[2]-v1[2]; let m21 = v3[2]-v1[2]; let m22 = v4[2]-v1[2];

    let det = m00*(m11*m22 - m12*m21) - m01*(m10*m22 - m12*m20) + m02*(m10*m21 - m11*m20);
    if det.abs() < 1e-30 { return false; }
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

/// Curl of the FEM E-field inside a tet.
///
/// Analytic for the Nédélec-2 basis: edge-mode curls are constant per tet;
/// face-mode curls are linear and approximated by their value at the tet
/// centroid (sufficient for order-2 visualisation accuracy). Used by the
/// error estimator and by the H-field channel (H = ∇×E / (−jωμ)).
///
/// Returns `[curl_x, curl_y, curl_z]` as complex amplitudes. The result does
/// not depend on a specific evaluation point inside the tet (centroid value).
pub fn eval_curl_in_tet(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    solution: &[C64],
    tet_idx: usize,
) -> [C64; 3] {
    let tet = &mesh.tets[tet_idx];
    let xs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][0]);
    let ys: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][1]);
    let zs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][2]);

    let (_, bbs, ccs, dds, v) = tet_coefficients(&xs, &ys, &zs);

    let grad = |i: usize| -> [f64; 3] { [bbs[i], ccs[i], dds[i]] };
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

    // Edge modes: curl is constant per tet, proportional to ∇λ_i × ∇λ_j.
    for ie in 0..6 {
        let n1 = l_edge[ie][0];
        let n2 = l_edge[ie][1];
        let em1 = solution[field_ids[ie]];
        let em2 = solution[field_ids[10 + ie]];
        let le = ds[n1][n2];
        let cr = cross(grad(n1), grad(n2));
        let coeff = (em1 + em2) * C64::from(3.0 * le * v1);
        for k in 0..3 {
            curl[k] += coeff * C64::from(cr[k]);
        }
    }

    // Face modes: linear in space, sampled at the centroid (λ_i = 1/4).
    for ie in 0..4 {
        let n1 = l_tri[ie][0];
        let n2 = l_tri[ie][1];
        let n3 = l_tri[ie][2];
        let ef1 = solution[field_ids[6 + ie]];
        let ef2 = solution[field_ids[16 + ie]];

        let l1 = ds[l_tri[ie][2]][l_tri[ie][0]];
        let l2 = ds[l_tri[ie][1]][l_tri[ie][0]];

        let cr12 = cross(grad(n1), grad(n2));
        let cr13 = cross(grad(n1), grad(n3));
        let cr23 = cross(grad(n2), grad(n3));

        let coeff1 = ef1 * C64::from(l1 * v1 * 0.5);
        let coeff2 = ef2 * C64::from(l2 * v1 * 0.5);
        for k in 0..3 {
            curl[k] += coeff1 * C64::from(cr13[k] - cr12[k]);
            curl[k] += coeff2 * C64::from(cr12[k] - cr23[k]);
        }
    }

    curl
}
