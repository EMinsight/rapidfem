//! Per-triangle Nedelec-2 Robin BC assembly.
//! Mirrors robinbc.py: ned2_tri_stiff (8×8), ned2_tri_force (8-vec).

use num_complex::Complex64 as C64;
use crate::coefficients::AreaCoeffCache;

fn dot2(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    a[0]*b[0] + a[1]*b[1]
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}

fn normalize3(a: &[f64; 3]) -> [f64; 3] {
    let n = (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt();
    [a[0]/n, a[1]/n, a[2]/n]
}

fn dist2(xs: &[f64; 3], ys: &[f64; 3], i: usize, j: usize) -> f64 {
    ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2)).sqrt()
}

/// Compute 8×8 Robin BC stiffness matrix for a surface triangle.
/// Mirrors robinbc.py:ned2_tri_stiff(glob_vertices, gamma).
///
/// `glob_vertices`: 3 vertices as [[x,y,z]; 3]
/// `gamma`: Robin BC impedance parameter (typically jβ)
pub fn ned2_tri_stiff(
    glob_vertices: &[[f64; 3]; 3],
    gamma: C64,
    ac_base: &AreaCoeffCache,
) -> [[C64; 8]; 8] {
    let zero = C64::new(0.0, 0.0);
    let mut bmat = [[zero; 8]; 8];

    // Local coordinate system
    let orig = glob_vertices[0];
    let v2 = glob_vertices[1];
    let v3 = glob_vertices[2];

    let e1 = [v2[0]-orig[0], v2[1]-orig[1], v2[2]-orig[2]];
    let e2 = [v3[0]-orig[0], v3[1]-orig[1], v3[2]-orig[2]];
    let zhat = normalize3(&cross3(&e1, &e2));
    let xhat = normalize3(&e1);
    let yhat = normalize3(&cross3(&zhat, &xhat));

    // Project vertices to local 2D coordinates
    let mut xs = [0.0f64; 3];
    let mut ys = [0.0f64; 3];
    for i in 0..3 {
        let dx = glob_vertices[i][0] - orig[0];
        let dy = glob_vertices[i][1] - orig[1];
        let dz = glob_vertices[i][2] - orig[2];
        xs[i] = xhat[0]*dx + xhat[1]*dy + xhat[2]*dz;
        ys[i] = yhat[0]*dx + yhat[1]*dy + yhat[2]*dz;
    }

    let (x1, x2, x3) = (xs[0], xs[1], xs[2]);
    let (y1, y2, y3) = (ys[0], ys[1], ys[2]);

    let b = [y2-y3, y3-y1, y1-y2];
    let c = [x3-x2, x1-x3, x2-x1];

    let gls: [[f64; 2]; 3] = [[b[0], c[0]], [b[1], c[1]], [b[2], c[2]]];

    let area = 0.5 * ((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1)).abs();

    // letters: local node index → cache index (1-indexed)
    let letters = |i: usize| -> usize { i + 1 };

    let ta = letters(0); let tb = letters(1); let tc = letters(2);
    let gta = &gls[0]; let gtb = &gls[1]; let gtc = &gls[2];

    let lt1 = dist2(&xs, &ys, 2, 0); // Ds[2,0]
    let lt2 = dist2(&xs, &ys, 1, 0); // Ds[1,0]

    let coeff = gamma / C64::from((2.0*area).powi(2));

    // Area coefficient cache scaled by Area
    let ac = |a: usize, b: usize, c: usize, d: usize| -> f64 {
        ac_base.get(a, b, c, d) * area
    };

    // Local edge map: must match tri_to_edge ordering (EMerge convention).
    // tri_to_edge[0] = edge(v0,v1), tri_to_edge[1] = edge(v1,v2), tri_to_edge[2] = edge(v0,v2)
    let local_edge_map: [[usize; 2]; 3] = [[0, 1], [1, 2], [0, 2]];

    // Edge-Edge block
    for ei in 0..3 {
        let ei1 = local_edge_map[ei][0];
        let ei2 = local_edge_map[ei][1];
        let li = dist2(&xs, &ys, ei1, ei2);
        let a = letters(ei1);
        let bb = letters(ei2);
        let ga = &gls[ei1];
        let gb = &gls[ei2];

        for ej in 0..3 {
            let ej1 = local_edge_map[ej][0];
            let ej2 = local_edge_map[ej][1];
            let lj = dist2(&xs, &ys, ej1, ej2);
            let cc = letters(ej1);
            let d = letters(ej2);
            let gc = &gls[ej1];
            let gd = &gls[ej2];

            let dac = dot2(ga, gc);
            let dad = dot2(ga, gd);
            let dbc = dot2(gb, gc);
            let dbd = dot2(gb, gd);
            let ll = li * lj;

            bmat[ei][ej] += C64::from(ll) * C64::from(ac(a,bb,cc,d)*dac - ac(a,bb,cc,cc)*dad - ac(a,a,cc,d)*dbc + ac(a,a,cc,cc)*dbd);
            bmat[ei][ej+4] += C64::from(ll) * C64::from(ac(a,bb,d,d)*dac - ac(a,bb,cc,d)*dad - ac(a,a,d,d)*dbc + ac(a,a,cc,d)*dbd);
            bmat[ei+4][ej] += C64::from(ll) * C64::from(ac(bb,bb,cc,d)*dac - ac(bb,bb,cc,cc)*dad - ac(a,bb,cc,d)*dbc + ac(a,bb,cc,cc)*dbd);
            bmat[ei+4][ej+4] += C64::from(ll) * C64::from(ac(bb,bb,d,d)*dac - ac(bb,bb,cc,d)*dad - ac(a,bb,d,d)*dbc + ac(a,bb,cc,d)*dbd);
        }

        // Edge-Face block
        let fa = dot2(ga, gtc);
        let fb = dot2(ga, gta);
        let fc = dot2(gb, gtc);
        let fd = dot2(gb, gta);
        let fe = dot2(ga, gtb);
        let ff = dot2(gb, gtb);

        bmat[ei][3] += C64::from(li*lt1) * C64::from(ac(a,bb,ta,tb)*fa - ac(a,bb,tb,tc)*fb - ac(a,a,ta,tb)*fc + ac(a,a,tb,tc)*fd);
        bmat[ei][7] += C64::from(li*lt2) * C64::from(ac(a,bb,tb,tc)*fb - ac(a,bb,tc,ta)*fe - ac(a,a,tb,tc)*fd + ac(a,a,tc,ta)*ff);
        bmat[3][ei] += C64::from(lt1*li) * C64::from(ac(ta,tb,a,bb)*fa - ac(ta,tb,a,a)*fc - ac(tb,tc,a,bb)*fb + ac(tb,tc,a,a)*fd);
        bmat[7][ei] += C64::from(lt2*li) * C64::from(ac(tb,tc,a,bb)*fb - ac(tb,tc,a,a)*fd - ac(tc,ta,a,bb)*fe + ac(tc,ta,a,a)*ff);
        bmat[ei+4][3] += C64::from(li*lt1) * C64::from(ac(bb,bb,ta,tb)*fa - ac(bb,bb,tb,tc)*fb - ac(a,bb,ta,tb)*fc + ac(a,bb,tb,tc)*fd);
        bmat[ei+4][7] += C64::from(li*lt2) * C64::from(ac(bb,bb,tb,tc)*fb - ac(bb,bb,tc,ta)*fe - ac(a,bb,tb,tc)*fd + ac(a,bb,tc,ta)*ff);
        bmat[3][ei+4] += C64::from(lt1*li) * C64::from(ac(ta,tb,bb,bb)*fa - ac(ta,tb,a,bb)*fc - ac(tb,tc,bb,bb)*fb + ac(tb,tc,a,bb)*fd);
        bmat[7][ei+4] += C64::from(lt2*li) * C64::from(ac(tb,tc,bb,bb)*fb - ac(tb,tc,a,bb)*fd - ac(tc,ta,bb,bb)*fe + ac(tc,ta,a,bb)*ff);
    }

    // Face-Face block
    let h1 = dot2(gta, gtc);
    let h2 = dot2(gta, gta);
    let h3 = dot2(gta, gtb);

    bmat[3][3] += C64::from(lt1*lt1) * C64::from(ac(ta,tb,ta,tb)*dot2(gtc,gtc) - ac(ta,tb,tb,tc)*h1 - ac(tb,tc,ta,tb)*h1 + ac(tb,tc,tb,tc)*h2);
    bmat[3][7] += C64::from(lt1*lt2) * C64::from(ac(ta,tb,tb,tc)*h1 - ac(ta,tb,tc,ta)*dot2(gtb,gtc) - ac(tb,tc,tb,tc)*h2 + ac(tb,tc,tc,ta)*h3);
    bmat[7][3] += C64::from(lt2*lt1) * C64::from(ac(tb,tc,ta,tb)*h1 - ac(tb,tc,tb,tc)*h2 - ac(tc,ta,ta,tb)*dot2(gtb,gtc) + ac(tc,ta,tb,tc)*h3);
    bmat[7][7] += C64::from(lt2*lt2) * C64::from(ac(tb,tc,tb,tc)*h2 - ac(tb,tc,tc,ta)*h3 - ac(tc,ta,tb,tc)*h3 + ac(tc,ta,tc,ta)*dot2(gtb,gtb));

    // Apply scaling
    for i in 0..8 {
        for j in 0..8 {
            bmat[i][j] *= coeff;
        }
    }

    bmat
}

/// Compute 8-element forcing vector for Robin BC excitation.
/// Mirrors robinbc.py:ned2_tri_force(glob_vertices, glob_Uinc, DPTs).
///
/// `glob_vertices`: 3 vertices as [[x,y,z]; 3]
/// `glob_uinc_at_qp`: incident field evaluated at each quadrature point, shape [n_qp][3] (Ex,Ey,Ez)
/// `quad_pts`: quadrature points [weight, L1, L2, L3] for each point
pub fn ned2_tri_force(
    glob_vertices: &[[f64; 3]; 3],
    glob_uinc_at_qp: &[[C64; 3]],
    quad_pts: &[[f64; 4]],
) -> [C64; 8] {
    let zero = C64::new(0.0, 0.0);
    let mut bvec = [zero; 8];

    // Local coordinate system
    let orig = glob_vertices[0];
    let v2 = glob_vertices[1];
    let v3 = glob_vertices[2];

    let e1 = [v2[0]-orig[0], v2[1]-orig[1], v2[2]-orig[2]];
    let e2 = [v3[0]-orig[0], v3[1]-orig[1], v3[2]-orig[2]];
    let zhat = normalize3(&cross3(&e1, &e2));
    let xhat = normalize3(&e1);
    let yhat = normalize3(&cross3(&zhat, &xhat));

    // Project vertices to local 2D
    let mut xs = [0.0f64; 3];
    let mut ys = [0.0f64; 3];
    for i in 0..3 {
        let dx = glob_vertices[i][0] - orig[0];
        let dy = glob_vertices[i][1] - orig[1];
        let dz = glob_vertices[i][2] - orig[2];
        xs[i] = xhat[0]*dx + xhat[1]*dy + xhat[2]*dz;
        ys[i] = yhat[0]*dx + yhat[1]*dy + yhat[2]*dz;
    }

    // Project incident field at each quad point to local 2D
    let n_qp = quad_pts.len();
    let mut lcs_ux_qp: Vec<C64> = Vec::with_capacity(n_qp);
    let mut lcs_uy_qp: Vec<C64> = Vec::with_capacity(n_qp);
    for qpi in 0..n_qp {
        let u = &glob_uinc_at_qp[qpi];
        lcs_ux_qp.push(u[0]*C64::from(xhat[0]) + u[1]*C64::from(xhat[1]) + u[2]*C64::from(xhat[2]));
        lcs_uy_qp.push(u[0]*C64::from(yhat[0]) + u[1]*C64::from(yhat[1]) + u[2]*C64::from(yhat[2]));
    }

    let (x1, x2, x3) = (xs[0], xs[1], xs[2]);
    let (y1, y2, y3) = (ys[0], ys[1], ys[2]);

    let a_coeff = [x2*y3 - y2*x3, x3*y1 - y3*x1, x1*y2 - y1*x2];
    let b_coeff = [y2-y3, y3-y1, y1-y2];
    let c_coeff = [x3-x2, x1-x3, x2-x1];

    let area = 0.5 * ((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1)).abs();
    let sign_a = -((x1-x3)*(y2-y1) - (x1-x2)*(y3-y1)).signum();

    let lt1 = dist2(&xs, &ys, 2, 0);
    let lt2 = dist2(&xs, &ys, 1, 0);

    // Must match tri_to_edge ordering: edge(v0,v1), edge(v0,v2), edge(v1,v2)
    let local_edge_map: [[usize; 2]; 3] = [[0, 1], [0, 2], [1, 2]];

    // Quadrature integration
    for (qpi, qp) in quad_pts.iter().enumerate() {
        let w = qp[0];
        let (l1, l2, l3) = (qp[1], qp[2], qp[3]);
        let x = x1*l1 + x2*l2 + x3*l3;
        let y = y1*l1 + y2*l2 + y3*l3;

        // Use pre-evaluated incident field at this quadrature point
        let ux = lcs_ux_qp[qpi];
        let uy = lcs_uy_qp[qpi];

        // Edge basis functions
        for ei in 0..3 {
            let ei1 = local_edge_map[ei][0];
            let ei2 = local_edge_map[ei][1];
            let li = dist2(&xs, &ys, ei1, ei2);

            let (a1, a2) = (a_coeff[ei1], a_coeff[ei2]);
            let (b1, b2) = (b_coeff[ei1], b_coeff[ei2]);
            let (c1, c2) = (c_coeff[ei1], c_coeff[ei2]);

            let q = a2 + b2*x + c2*y;
            let z = a1 + b1*x + c1*y;
            let a4 = 4.0 * area * area;
            let q2 = q / a4;
            let z2 = z / a4;
            let ar2 = 1.0 / (2.0 * area);

            // Mode 1 basis: Ee1
            let ee1x = (b1*q2 - b2*z2) * z * ar2;
            let ee1y = (c1*q2 - c2*z2) * z * ar2;
            // Mode 2 basis: Ee2
            let ee2x = (b1*q2 - b2*z2) * q * ar2;
            let ee2y = (c1*q2 - c2*z2) * q * ar2;

            bvec[ei] += C64::from(sign_a * area * li * w) * (C64::from(ee1x)*ux + C64::from(ee1y)*uy);
            bvec[ei+4] += C64::from(sign_a * area * li * w) * (C64::from(ee2x)*ux + C64::from(ee2y)*uy);
        }

        // Face basis functions
        let q = a_coeff[1] + b_coeff[1]*x + c_coeff[1]*y;
        let z = a_coeff[0] + b_coeff[0]*x + c_coeff[0]*y;
        let fa = 8.0 * area.powi(3);
        let ww = (a_coeff[2] + b_coeff[2]*x + c_coeff[2]*y) / fa;
        let w2 = q * ww;

        let ef1x = lt1 * (-b_coeff[0]*w2 + b_coeff[2]*z*q/fa);
        let ef1y = lt1 * (-c_coeff[0]*w2 + c_coeff[2]*z*q/fa);
        let ef2x = lt2 * (b_coeff[0]*w2 - b_coeff[1]*z*ww);
        let ef2y = lt2 * (c_coeff[0]*w2 - c_coeff[1]*z*ww);

        bvec[3] += C64::from(sign_a * area * w) * (C64::from(ef1x)*ux + C64::from(ef1y)*uy);
        bvec[7] += C64::from(sign_a * area * w) * (C64::from(ef2x)*ux + C64::from(ef2y)*uy);
    }

    bvec
}
