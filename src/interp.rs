//! Nedelec-2 field interpolation at arbitrary points.
//! Mirrors interp.py: ned2_tet_interp (3D tetrahedral interpolation).
//!
//! For S-parameter extraction, we evaluate E_FEM at port face quadrature points
//! by finding the adjacent tet and evaluating the Nedelec-2 basis there.

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;

/// Full barycentric coefficients including constant term a_i.
/// λ_i(x,y,z) = a_i + b_i*x + c_i*y + d_i*z
pub fn tet_coefficients_full(xs: &[f64; 4], ys: &[f64; 4], zs: &[f64; 4])
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

/// Evaluate E-field at a single point (x,y,z) inside a known tetrahedron.
/// Uses the full Nedelec-2 basis (20 DOFs) with exact formulas from interp.py.
pub fn eval_field_in_tet(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    solution: &[C64],
    tet_idx: usize,
    x: f64, y: f64, z: f64,
) -> [C64; 3] {
    let tet = &mesh.tets[tet_idx];
    let xs = [mesh.nodes[tet[0]][0], mesh.nodes[tet[1]][0], mesh.nodes[tet[2]][0], mesh.nodes[tet[3]][0]];
    let ys = [mesh.nodes[tet[0]][1], mesh.nodes[tet[1]][1], mesh.nodes[tet[2]][1], mesh.nodes[tet[3]][1]];
    let zs = [mesh.nodes[tet[0]][2], mesh.nodes[tet[1]][2], mesh.nodes[tet[2]][2], mesh.nodes[tet[3]][2]];

    let (aas, bbs, ccs, dds, v) = tet_coefficients_full(&xs, &ys, &zs);

    // Distance matrix
    let mut ds = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in i..4 {
            let d = ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2) + (zs[i]-zs[j]).powi(2)).sqrt();
            ds[i][j] = d; ds[j][i] = d;
        }
    }

    // Local edge and face mappings
    let tet_edges = &mesh.tet_to_edge[tet_idx];
    let global_edge_nodes: [[usize; 2]; 6] = std::array::from_fn(|i| mesh.edges[tet_edges[i]]);
    let l_edge_ids = crate::basis::local_mapping(tet, &global_edge_nodes);

    let tet_tris = &mesh.tet_to_tri[tet_idx];
    let global_tri_nodes: [[usize; 3]; 4] = std::array::from_fn(|i| mesh.tris[tet_tris[i]]);
    let l_tri_ids = crate::basis::local_mapping_tri(tet, &global_tri_nodes);

    // Field DOF values
    let field_ids = &basis.tet_to_field[tet_idx];
    let em1s: [C64; 6] = std::array::from_fn(|i| solution[field_ids[i]]);
    let ef1s: [C64; 4] = std::array::from_fn(|i| solution[field_ids[6 + i]]);
    let em2s: [C64; 6] = std::array::from_fn(|i| solution[field_ids[10 + i]]);
    let ef2s: [C64; 4] = std::array::from_fn(|i| solution[field_ids[16 + i]]);

    let v1 = 1.0 / (216.0 * v * v * v);

    let mut ex = C64::new(0.0, 0.0);
    let mut ey = C64::new(0.0, 0.0);
    let mut ez = C64::new(0.0, 0.0);

    // Edge basis functions (6 edges × 2 modes)
    for ie in 0..6 {
        let n1 = l_edge_ids[ie][0];
        let n2 = l_edge_ids[ie][1];
        let (a1, a2) = (aas[n1], aas[n2]);
        let (b1, b2) = (bbs[n1], bbs[n2]);
        let (c1, c2) = (ccs[n1], ccs[n2]);
        let (d1, d2) = (dds[n1], dds[n2]);

        let lv = ds[n1][n2] * v1;
        let f1 = a1 + b1*x + c1*y + d1*z;
        let f2 = a2 + b2*x + c2*y + d2*z;
        let f3 = em1s[ie] * C64::from(f1) + em2s[ie] * C64::from(f2);

        ex += f3 * C64::from(lv * (b1*f2 - b2*f1));
        ey += f3 * C64::from(lv * (c1*f2 - c2*f1));
        ez += f3 * C64::from(lv * (d1*f2 - d2*f1));
    }

    // Face basis functions (4 faces × 2 modes)
    for ie in 0..4 {
        let n1 = l_tri_ids[ie][0];
        let n2 = l_tri_ids[ie][1];
        let n3 = l_tri_ids[ie][2];
        let (a1, a2, a3) = (aas[n1], aas[n2], aas[n3]);
        let (b1, b2, b3) = (bbs[n1], bbs[n2], bbs[n3]);
        let (c1, c2, c3) = (ccs[n1], ccs[n2], ccs[n3]);
        let (d1, d2, d3) = (dds[n1], dds[n2], dds[n3]);

        let l1 = ds[l_tri_ids[ie][2]][l_tri_ids[ie][0]]; // Ds[n3, n1]
        let l2 = ds[l_tri_ids[ie][1]][l_tri_ids[ie][0]]; // Ds[n2, n1]

        let f1 = a1 + b1*x + c1*y + d1*z;
        let f2 = a2 + b2*x + c2*y + d2*z;
        let f3 = a3 + b3*x + c3*y + d3*z;

        let q1 = ef1s[ie] * C64::from(l1 * f2);
        let q2 = ef2s[ie] * C64::from(l2 * f3);

        ex += (-q1 * C64::from(b1*f3 - b3*f1) + q2 * C64::from(b1*f2 - b2*f1)) * C64::from(v1);
        ey += (-q1 * C64::from(c1*f3 - c3*f1) + q2 * C64::from(c1*f2 - c2*f1)) * C64::from(v1);
        ez += (-q1 * C64::from(d1*f3 - d3*f1) + q2 * C64::from(d1*f2 - d2*f1)) * C64::from(v1);
    }

    [ex, ey, ez]
}

/// Build a field evaluation closure for S-parameter extraction.
/// For each port face triangle, uses the adjacent tet for interpolation.
pub fn make_field_evaluator<'a>(
    mesh: &'a Mesh,
    basis: &'a Nedelec2Basis,
    solution: &'a [C64],
    port_tris: &'a [usize],
) -> impl Fn(f64, f64, f64) -> [C64; 3] + 'a {
    // Precompute: for each port triangle, find the adjacent tet
    let tri_to_tet: Vec<usize> = port_tris.iter().map(|&ti| {
        mesh.tri_to_tet[ti][0] // first adjacent tet
    }).collect();

    move |x: f64, y: f64, z: f64| -> [C64; 3] {
        // Find which port triangle contains the point (brute force — fine for port faces)
        for (i, &ti) in port_tris.iter().enumerate() {
            let tri = &mesh.tris[ti];
            let v0 = mesh.nodes[tri[0]];
            let v1 = mesh.nodes[tri[1]];
            let v2 = mesh.nodes[tri[2]];

            // Check if point is inside this triangle (barycentric test in 3D)
            let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
            let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
            let ep = [x-v0[0], y-v0[1], z-v0[2]];

            let d11 = e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2];
            let d12 = e1[0]*e2[0] + e1[1]*e2[1] + e1[2]*e2[2];
            let d22 = e2[0]*e2[0] + e2[1]*e2[1] + e2[2]*e2[2];
            let dp1 = ep[0]*e1[0] + ep[1]*e1[1] + ep[2]*e1[2];
            let dp2 = ep[0]*e2[0] + ep[1]*e2[1] + ep[2]*e2[2];

            let det = d11*d22 - d12*d12;
            if det.abs() < 1e-30 { continue; }
            let u = (d22*dp1 - d12*dp2) / det;
            let v = (d11*dp2 - d12*dp1) / det;

            let eps = -1e-6;
            if u >= eps && v >= eps && u + v <= 1.0 + 1e-6 {
                return eval_field_in_tet(mesh, basis, solution, tri_to_tet[i], x, y, z);
            }
        }
        // Fallback: use nearest triangle's tet
        if !port_tris.is_empty() {
            let tet = mesh.tri_to_tet[port_tris[0]][0];
            return eval_field_in_tet(mesh, basis, solution, tet, x, y, z);
        }
        [C64::new(0.0, 0.0); 3]
    }
}
