//! Exact port of emerge/_emerge/physics/microwave/assembly/robinbc.py
//!
//! Functions: ned2_tri_stiff, ned2_tri_force
//! All variable names match EMerge's Python code.

use num_complex::Complex64 as C64;
use crate::coefficients::AreaCoeffCache;

// 2D dot product (matches robinbc.py: dot(a, b) = a[0]*b[0] + a[1]*b[1])
fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

// 3D cross product (matches robinbc.py: cross(a, b))
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

// normalize (matches robinbc.py: normalize(a))
fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt();
    [a[0]/n, a[1]/n, a[2]/n]
}

// compute_distances for 3 points (matches optimized.py: compute_distances(xs, ys, zs))
fn compute_distances_3(xs: [f64; 3], ys: [f64; 3], zs: [f64; 3]) -> [[f64; 3]; 3] {
    let mut ds = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in i..3 {
            let d = ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2) + (zs[i]-zs[j]).powi(2)).sqrt();
            ds[i][j] = d;
            ds[j][i] = d;
        }
    }
    ds
}

// optim_matmul: B (3x3) @ data (3xN), returns (3xN)
// In our case N=3 for vertices, N=n_qp for Uinc
fn matmul_3x3_cols(b: &[[f64; 3]; 3], data: &[[f64; 3]]) -> Vec<[f64; 3]> {
    data.iter().map(|col| {
        [
            b[0][0]*col[0] + b[0][1]*col[1] + b[0][2]*col[2],
            b[1][0]*col[0] + b[1][1]*col[1] + b[1][2]*col[2],
            b[2][0]*col[0] + b[2][1]*col[1] + b[2][2]*col[2],
        ]
    }).collect()
}

fn matmul_3x3_cols_c(b: &[[f64; 3]; 3], data: &[[C64; 3]]) -> Vec<[C64; 3]> {
    data.iter().map(|col| {
        [
            C64::from(b[0][0])*col[0] + C64::from(b[0][1])*col[1] + C64::from(b[0][2])*col[2],
            C64::from(b[1][0])*col[0] + C64::from(b[1][1])*col[1] + C64::from(b[1][2])*col[2],
            C64::from(b[2][0])*col[0] + C64::from(b[2][1])*col[1] + C64::from(b[2][2])*col[2],
        ]
    }).collect()
}

/// Build the local coordinate system from triangle vertices.
/// Returns (basis, lcs_xs, lcs_ys) where basis is the 3x3 rotation matrix.
/// Exact port of the shared preamble in ned2_tri_stiff and ned2_tri_force.
fn tri_local_cs(glob_vertices: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [f64; 3]) {
    let orig = glob_vertices[0];
    let v2 = glob_vertices[1];
    let v3 = glob_vertices[2];

    let e1 = [v2[0]-orig[0], v2[1]-orig[1], v2[2]-orig[2]];
    let e2 = [v3[0]-orig[0], v3[1]-orig[1], v3[2]-orig[2]];
    let zhat = normalize(cross3(e1, e2));
    let xhat = normalize(e1);
    let yhat = normalize(cross3(zhat, xhat));

    let basis = [xhat, yhat, zhat];

    // Project vertices to local 2D — inline, no Vec allocation
    let mut xs = [0.0; 3];
    let mut ys = [0.0; 3];
    for k in 0..3 {
        let dx = glob_vertices[k][0] - orig[0];
        let dy = glob_vertices[k][1] - orig[1];
        let dz = glob_vertices[k][2] - orig[2];
        xs[k] = basis[0][0]*dx + basis[0][1]*dy + basis[0][2]*dz;
        ys[k] = basis[1][0]*dx + basis[1][1]*dy + basis[1][2]*dz;
    }

    (basis, xs, ys)
}

/// Exact port of robinbc.py:ned2_tri_stiff(glob_vertices, gamma)
///
/// glob_vertices: 3 vertices, glob_vertices[i] = [x, y, z] (column i of EMerge's (3,3) array)
/// gamma: Robin BC impedance parameter
/// ac_base: precomputed area coefficient cache
///
/// Returns 8x8 complex matrix
pub fn ned2_tri_stiff(
    glob_vertices: &[[f64; 3]; 3],
    gamma: C64,
    ac_base: &AreaCoeffCache,
) -> [[C64; 8]; 8] {
    let zero = C64::new(0.0, 0.0);
    let mut bmat = [[zero; 8]; 8];

    // local_edge_map: EMerge has np.array([[0,1,0],[1,2,2]]), indexed as [:, ei]
    // Column 0: (0,1), Column 1: (1,2), Column 2: (0,2)
    let local_edge_map: [[usize; 2]; 3] = [[0, 1], [1, 2], [0, 2]];

    let (_, xs, ys) = tri_local_cs(glob_vertices);
    let (x1, x2, x3) = (xs[0], xs[1], xs[2]);
    let (y1, y2, y3) = (ys[0], ys[1], ys[2]);

    let b = [y2-y3, y3-y1, y1-y2];
    let c = [x3-x2, x1-x3, x2-x1];

    let ds = compute_distances_3(xs, ys, [0.0; 3]);

    let gls: [[f64; 2]; 3] = [[b[0], c[0]], [b[1], c[1]], [b[2], c[2]]];

    let area = 0.5 * ((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1)).abs();

    // letters: EMerge uses letters = [1,2,3,4,5,6], letters[node_idx] maps to cache index
    let letters = |i: usize| -> usize { i + 1 };

    let t_a = letters(0);
    let t_b = letters(1);
    let t_c = letters(2);
    let gt_a = gls[0];
    let gt_b = gls[1];
    let gt_c = gls[2];

    let lt1 = ds[2][0]; // Ds[2, 0]
    let lt2 = ds[1][0]; // Ds[1, 0]

    let coeff = gamma / C64::from((2.0 * area).powi(2));

    // AREA_COEFF = AREA_COEFF_CACHE_BASE * Area
    let ac = |a: usize, b: usize, c: usize, d: usize| -> f64 {
        ac_base.get(a, b, c, d) * area
    };

    // Edge-Edge block
    for ei in 0..3 {
        let ei1 = local_edge_map[ei][0];
        let ei2 = local_edge_map[ei][1];
        let li = ds[ei1][ei2];
        let a = letters(ei1);
        let bb = letters(ei2);
        let ga = gls[ei1];
        let gb = gls[ei2];

        for ej in 0..3 {
            let ej1 = local_edge_map[ej][0];
            let ej2 = local_edge_map[ej][1];
            let lj = ds[ej1][ej2];
            let cc = letters(ej1);
            let d = letters(ej2);
            let gc = gls[ej1];
            let gd = gls[ej2];

            let dac = dot(ga, gc);
            let dad = dot(ga, gd);
            let dbc = dot(gb, gc);
            let dbd = dot(gb, gd);
            let ll = li * lj;

            bmat[ei][ej]     += C64::from(ll * (ac(a,bb,cc,d)*dac - ac(a,bb,cc,cc)*dad - ac(a,a,cc,d)*dbc + ac(a,a,cc,cc)*dbd));
            bmat[ei][ej+4]   += C64::from(ll * (ac(a,bb,d,d)*dac - ac(a,bb,cc,d)*dad - ac(a,a,d,d)*dbc + ac(a,a,cc,d)*dbd));
            bmat[ei+4][ej]   += C64::from(ll * (ac(bb,bb,cc,d)*dac - ac(bb,bb,cc,cc)*dad - ac(a,bb,cc,d)*dbc + ac(a,bb,cc,cc)*dbd));
            bmat[ei+4][ej+4] += C64::from(ll * (ac(bb,bb,d,d)*dac - ac(bb,bb,cc,d)*dad - ac(a,bb,d,d)*dbc + ac(a,bb,cc,d)*dbd));
        }

        // Edge-Face block
        let fa = dot(ga, gt_c);
        let fb = dot(ga, gt_a);
        let fc = dot(gb, gt_c);
        let fd = dot(gb, gt_a);
        let fe = dot(ga, gt_b);
        let ff = dot(gb, gt_b);

        bmat[ei][3]     += C64::from(li*lt1 * (ac(a,bb,t_a,t_b)*fa - ac(a,bb,t_b,t_c)*fb - ac(a,a,t_a,t_b)*fc + ac(a,a,t_b,t_c)*fd));
        bmat[ei][7]     += C64::from(li*lt2 * (ac(a,bb,t_b,t_c)*fb - ac(a,bb,t_c,t_a)*fe - ac(a,a,t_b,t_c)*fd + ac(a,a,t_c,t_a)*ff));
        bmat[3][ei]     += C64::from(lt1*li * (ac(t_a,t_b,a,bb)*fa - ac(t_a,t_b,a,a)*fc - ac(t_b,t_c,a,bb)*fb + ac(t_b,t_c,a,a)*fd));
        bmat[7][ei]     += C64::from(lt2*li * (ac(t_b,t_c,a,bb)*fb - ac(t_b,t_c,a,a)*fd - ac(t_c,t_a,a,bb)*fe + ac(t_c,t_a,a,a)*ff));
        bmat[ei+4][3]   += C64::from(li*lt1 * (ac(bb,bb,t_a,t_b)*fa - ac(bb,bb,t_b,t_c)*fb - ac(a,bb,t_a,t_b)*fc + ac(a,bb,t_b,t_c)*fd));
        bmat[ei+4][7]   += C64::from(li*lt2 * (ac(bb,bb,t_b,t_c)*fb - ac(bb,bb,t_c,t_a)*fe - ac(a,bb,t_b,t_c)*fd + ac(a,bb,t_c,t_a)*ff));
        bmat[3][ei+4]   += C64::from(lt1*li * (ac(t_a,t_b,bb,bb)*fa - ac(t_a,t_b,a,bb)*fc - ac(t_b,t_c,bb,bb)*fb + ac(t_b,t_c,a,bb)*fd));
        bmat[7][ei+4]   += C64::from(lt2*li * (ac(t_b,t_c,bb,bb)*fb - ac(t_b,t_c,a,bb)*fd - ac(t_c,t_a,bb,bb)*fe + ac(t_c,t_a,a,bb)*ff));
    }

    // Face-Face block
    let h1 = dot(gt_a, gt_c);
    let h2 = dot(gt_a, gt_a);
    let h3 = dot(gt_a, gt_b);

    bmat[3][3] += C64::from(lt1*lt1 * (ac(t_a,t_b,t_a,t_b)*dot(gt_c,gt_c) - ac(t_a,t_b,t_b,t_c)*h1 - ac(t_b,t_c,t_a,t_b)*h1 + ac(t_b,t_c,t_b,t_c)*h2));
    bmat[3][7] += C64::from(lt1*lt2 * (ac(t_a,t_b,t_b,t_c)*h1 - ac(t_a,t_b,t_c,t_a)*dot(gt_b,gt_c) - ac(t_b,t_c,t_b,t_c)*h2 + ac(t_b,t_c,t_c,t_a)*h3));
    bmat[7][3] += C64::from(lt2*lt1 * (ac(t_b,t_c,t_a,t_b)*h1 - ac(t_b,t_c,t_b,t_c)*h2 - ac(t_c,t_a,t_a,t_b)*dot(gt_b,gt_c) + ac(t_c,t_a,t_b,t_c)*h3));
    bmat[7][7] += C64::from(lt2*lt2 * (ac(t_b,t_c,t_b,t_c)*h2 - ac(t_b,t_c,t_c,t_a)*h3 - ac(t_c,t_a,t_b,t_c)*h3 + ac(t_c,t_a,t_c,t_a)*dot(gt_b,gt_b)));

    // Apply COEFF
    for i in 0..8 {
        for j in 0..8 {
            bmat[i][j] *= coeff;
        }
    }

    bmat
}

/// Exact port of robinbc.py:ned2_tri_force(glob_vertices, glob_Uinc, DPTs)
///
/// glob_vertices: 3 vertices, [i] = [x, y, z] (column i of EMerge's (3,3))
/// glob_uinc: incident field at each quad point, shape [n_qp][3] (EMerge's (3, n_qp) transposed)
/// dpts: quadrature points, [n_qp][4] = [W, L1, L2, L3] (EMerge's (4, n_qp) transposed)
///
/// Returns 8-element complex forcing vector
pub fn ned2_tri_force(
    glob_vertices: &[[f64; 3]; 3],
    glob_uinc: &[[C64; 3]],
    dpts: &[[f64; 4]],
) -> [C64; 8] {
    let zero = C64::new(0.0, 0.0);
    let mut bvec = [zero; 8];

    let local_edge_map: [[usize; 2]; 3] = [[0, 1], [1, 2], [0, 2]];

    let (basis, xs, ys) = tri_local_cs(glob_vertices);

    // lcs_Uinc = optim_matmul(basis, glob_Uinc)
    let lcs_uinc = matmul_3x3_cols_c(&basis, glob_uinc);

    let (x1, x2, x3) = (xs[0], xs[1], xs[2]);
    let (y1, y2, y3) = (ys[0], ys[1], ys[2]);

    let a_s = [x2*y3-y2*x3, x3*y1-y3*x1, x1*y2-y1*x2];
    let b_s = [y2-y3, y3-y1, y1-y2];
    let c_s = [x3-x2, x1-x3, x2-x1];

    let ds = compute_distances_3(xs, ys, [0.0; 3]);

    let area = 0.5 * ((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1)).abs();
    let sign_a = -((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1)).signum();

    let lt1 = ds[2][0];
    let lt2 = ds[1][0];

    let n_qp = dpts.len();

    // Precompute quadrature coords and weights (EMerge vectorized, we loop)
    // x = x1*DPTs[1,:] + x2*DPTs[2,:] + x3*DPTs[3,:]
    // y = y1*DPTs[1,:] + y2*DPTs[2,:] + y3*DPTs[3,:]
    // Ws = DPTs[0,:]
    // Ux = lcs_Uinc[0,:], Uy = lcs_Uinc[1,:]

    // Edge basis functions
    for ei in 0..3 {
        let ei1 = local_edge_map[ei][0];
        let ei2 = local_edge_map[ei][1];
        let li = ds[ei1][ei2];

        let (aa1, aa2) = (a_s[ei1], a_s[ei2]);
        let (bb1, bb2) = (b_s[ei1], b_s[ei2]);
        let (cc1, cc2) = (c_s[ei1], c_s[ei2]);

        let mut sum_mode1 = zero;
        let mut sum_mode2 = zero;

        for qpi in 0..n_qp {
            let w = dpts[qpi][0];
            let (l1, l2, l3) = (dpts[qpi][1], dpts[qpi][2], dpts[qpi][3]);
            let x = x1*l1 + x2*l2 + x3*l3;
            let y = y1*l1 + y2*l2 + y3*l3;
            let ux = lcs_uinc[qpi][0];
            let uy = lcs_uinc[qpi][1];

            let q = aa2 + bb2*x + cc2*y;
            let z = aa1 + bb1*x + cc1*y;
            let a4 = 4.0 * area * area;
            let q2 = q / a4;
            let z2 = z / a4;
            let ar2 = 1.0 / (2.0 * area);

            let ee1x = (bb1*q2 - bb2*z2) * z * ar2;
            let ee1y = (cc1*q2 - cc2*z2) * z * ar2;
            let ee2x = (bb1*q2 - bb2*z2) * q * ar2;
            let ee2y = (cc1*q2 - cc2*z2) * q * ar2;

            sum_mode1 += C64::from(w) * (C64::from(ee1x)*ux + C64::from(ee1y)*uy);
            sum_mode2 += C64::from(w) * (C64::from(ee2x)*ux + C64::from(ee2y)*uy);
        }

        bvec[ei]   += C64::from(sign_a * area * li) * sum_mode1;
        bvec[ei+4] += C64::from(sign_a * area * li) * sum_mode2;
    }

    // Face basis functions
    let (aa1, aa2, aa3) = (a_s[0], a_s[1], a_s[2]);
    let (bb1, bb2, bb3) = (b_s[0], b_s[1], b_s[2]);
    let (cc1, cc2, cc3) = (c_s[0], c_s[1], c_s[2]);

    let mut sum_f1 = zero;
    let mut sum_f2 = zero;

    for qpi in 0..n_qp {
        let w = dpts[qpi][0];
        let (l1, l2, l3) = (dpts[qpi][1], dpts[qpi][2], dpts[qpi][3]);
        let x = x1*l1 + x2*l2 + x3*l3;
        let y = y1*l1 + y2*l2 + y3*l3;
        let ux = lcs_uinc[qpi][0];
        let uy = lcs_uinc[qpi][1];

        let q = aa2 + bb2*x + cc2*y;
        let z = aa1 + bb1*x + cc1*y;
        let fa = 8.0 * area.powi(3);
        let ww = (aa3 + bb3*x + cc3*y) / fa;
        let w2 = q * ww;

        let ef1x = lt1 * (-bb1*w2 + bb3*z*q/fa);
        let ef1y = lt1 * (-cc1*w2 + cc3*z*q/fa);
        let ef2x = lt2 * (bb1*w2 - bb2*z*ww);
        let ef2y = lt2 * (cc1*w2 - cc2*z*ww);

        sum_f1 += C64::from(w) * (C64::from(ef1x)*ux + C64::from(ef1y)*uy);
        sum_f2 += C64::from(w) * (C64::from(ef2x)*ux + C64::from(ef2y)*uy);
    }

    bvec[3] += C64::from(sign_a * area) * sum_f1;
    bvec[7] += C64::from(sign_a * area) * sum_f2;

    bvec
}
