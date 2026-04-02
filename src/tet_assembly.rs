//! Per-tet Nedelec-2 stiffness and mass matrix assembly.
//! Mirrors curlcurl.py: tet_coefficients_bcd, ned2_tet_stiff_mass, _matrix_builder.

use num_complex::Complex64 as C64;
use crate::coefficients::VolumeCoeffCache;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;

type Vec3c = [C64; 3];

fn dot_c(a: &Vec3c, b: &Vec3c) -> C64 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

fn cross_c(a: &Vec3c, b: &Vec3c) -> Vec3c {
    [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]
}

fn matmul3(m: &[[C64; 3]; 3], v: &Vec3c) -> Vec3c {
    [
        m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
        m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
        m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
    ]
}

fn matinv3(m: &[[C64; 3]; 3]) -> [[C64; 3]; 3] {
    let det = m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
            - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
            + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    let inv_det = C64::new(1.0, 0.0) / det;
    [
        [(m[1][1]*m[2][2] - m[1][2]*m[2][1]) * inv_det,
         (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * inv_det,
         (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * inv_det],
        [(m[1][2]*m[2][0] - m[1][0]*m[2][2]) * inv_det,
         (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * inv_det,
         (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * inv_det],
        [(m[1][0]*m[2][1] - m[1][1]*m[2][0]) * inv_det,
         (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * inv_det,
         (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * inv_det],
    ]
}

/// Compute barycentric coordinate coefficients for a tetrahedron.
/// Returns (bbs[4], ccs[4], dds[4], V) where V is the volume.
/// Mirrors curlcurl.py:tet_coefficients_bcd.
pub fn tet_coefficients_bcd(xs: &[f64; 4], ys: &[f64; 4], zs: &[f64; 4])
    -> ([f64; 4], [f64; 4], [f64; 4], f64)
{
    let (x1,x2,x3,x4) = (xs[0],xs[1],xs[2],xs[3]);
    let (y1,y2,y3,y4) = (ys[0],ys[1],ys[2],ys[3]);
    let (z1,z2,z3,z4) = (zs[0],zs[1],zs[2],zs[3]);

    let v = (-x1*y2*z3/6.0 + x1*y2*z4/6.0 + x1*y3*z2/6.0 - x1*y3*z4/6.0 - x1*y4*z2/6.0 +
              x1*y4*z3/6.0 + x2*y1*z3/6.0 - x2*y1*z4/6.0 - x2*y3*z1/6.0 + x2*y3*z4/6.0 +
              x2*y4*z1/6.0 - x2*y4*z3/6.0 - x3*y1*z2/6.0 + x3*y1*z4/6.0 + x3*y2*z1/6.0 -
              x3*y2*z4/6.0 - x3*y4*z1/6.0 + x3*y4*z2/6.0 + x4*y1*z2/6.0 - x4*y1*z3/6.0 -
              x4*y2*z1/6.0 + x4*y2*z3/6.0 + x4*y3*z1/6.0 - x4*y3*z2/6.0).abs();

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

    (bbs, ccs, dds, v)
}

/// Compute per-tet 20×20 stiffness and mass matrices.
/// Mirrors curlcurl.py:ned2_tet_stiff_mass exactly.
///
/// Arguments:
/// - `tet_vertices`: 3×4 vertex coordinates (xs[4], ys[4], zs[4])
/// - `edge_lengths`: 6 edge lengths for this tet's edges
/// - `local_edge_map`: 2×6 local node indices for each edge
/// - `local_tri_map`: 3×4 local node indices for each face
/// - `ms`: inverse permeability tensor (μr⁻¹, 3×3)
/// - `mm`: permittivity tensor (εr, 3×3)
///
/// Returns (Dmat, Fmat) as 20×20 complex matrices (row-major flat arrays).
pub fn ned2_tet_stiff_mass(
    xs: &[f64; 4], ys: &[f64; 4], zs: &[f64; 4],
    edge_lengths: &[f64; 6],
    local_edge_map: &[[usize; 2]; 6],
    local_tri_map: &[[usize; 3]; 4],
    ms: &[[C64; 3]; 3],  // μr⁻¹
    mm: &[[C64; 3]; 3],  // εr
    vc_base: &VolumeCoeffCache,
) -> ([[C64; 20]; 20], [[C64; 20]; 20])
{
    let zero = C64::new(0.0, 0.0);
    let mut dmat = [[zero; 20]; 20];
    let mut fmat = [[zero; 20]; 20];

    let (bbs, ccs, dds, v) = tet_coefficients_bcd(xs, ys, zs);

    // Gradient vectors GL[i] = [b_i, c_i, d_i] as complex
    let gls: [Vec3c; 4] = [
        [C64::from(bbs[0]), C64::from(ccs[0]), C64::from(dds[0])],
        [C64::from(bbs[1]), C64::from(ccs[1]), C64::from(dds[1])],
        [C64::from(bbs[2]), C64::from(ccs[2]), C64::from(dds[2])],
        [C64::from(bbs[3]), C64::from(ccs[3]), C64::from(dds[3])],
    ];

    // Distance matrix between tet vertices
    let mut ds = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in i..4 {
            let d = ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2) + (zs[i]-zs[j]).powi(2)).sqrt();
            ds[i][j] = d;
            ds[j][i] = d;
        }
    }

    // Letters: map local node index (0-3) → cache index (1-4)
    // EMerge: letters = [1,2,3,4,5,6] and A = letters[ei1] where ei1 is local node index
    // Since local node indices are 0-3, letters[0..3] = [1,2,3,4]
    let letters = |i: usize| -> usize { i + 1 };

    let ka = 1.0 / (6.0 * v).powi(4);
    let kb = 1.0 / (6.0 * v).powi(2);
    let v6 = 6.0 * v;

    // Volume coefficient cache scaled by 6V
    let vc = |a: usize, b: usize, c: usize, d: usize| -> f64 {
        vc_base.get(a, b, c, d) * v6
    };

    // === Edge-Edge block (6×6 quadrants at [0..6, 0..6], [0..6, 10..16], [10..16, 0..6], [10..16, 10..16]) ===
    for ei in 0..6 {
        let ei1 = local_edge_map[ei][0];
        let ei2 = local_edge_map[ei][1];
        let ga = &gls[ei1];
        let gb = &gls[ei2];
        let a = letters(ei1);
        let b = letters(ei2);
        let l1 = edge_lengths[ei];

        for ej in 0..6 {
            let ej1 = local_edge_map[ej][0];
            let ej2 = local_edge_map[ej][1];
            let gc = &gls[ej1];
            let gd = &gls[ej2];
            let c = letters(ej1);
            let d = letters(ej2);
            let l2 = edge_lengths[ej];

            let vad = vc(a,d,0,0); let vac = vc(a,c,0,0);
            let vbc = vc(b,c,0,0); let vbd = vc(b,d,0,0);
            let vabcd = vc(a,b,c,d);
            let vabcc = vc(a,b,c,c); let vabdd = vc(a,b,d,d);
            let vaacd = vc(a,a,c,d); let vaadd = vc(a,a,d,d);
            let vaacc = vc(a,a,c,c);
            let vbbcd = vc(b,b,c,d); let vbbcc = vc(b,b,c,c); let vbbdd = vc(b,b,d,d);

            let l12 = l1 * l2;

            // Stiffness: curl-curl term
            let factor = C64::from(l12 * 9.0) * dot_c(&cross_c(ga, gb), &matmul3(ms, &cross_c(gc, gd)));
            dmat[ei][ej] = factor * C64::from(vac);
            dmat[ei][ej+10] = factor * C64::from(vad);
            dmat[ei+10][ej] = factor * C64::from(vbc);
            dmat[ei+10][ej+10] = factor * C64::from(vbd);

            // Mass: material term
            let er_gf = matmul3(mm, gc);
            let er_gc = matmul3(mm, gd);
            let er_gd_val = dot_c(ga, &er_gf);
            let ge_mul_er_gf = dot_c(ga, &er_gc);
            let ge_mul_er_gc = dot_c(gb, &er_gf);
            let ga_mul_er_gf = dot_c(gb, &er_gc);

            let l12c = C64::from(l12);
            fmat[ei][ej] = l12c * (C64::from(vabcd)*er_gd_val - C64::from(vabcc)*ge_mul_er_gf - C64::from(vaacd)*ge_mul_er_gc + C64::from(vaacc)*ga_mul_er_gf);
            fmat[ei][ej+10] = l12c * (C64::from(vabdd)*er_gd_val - C64::from(vabcd)*ge_mul_er_gf - C64::from(vaadd)*ge_mul_er_gc + C64::from(vaacd)*ga_mul_er_gf);
            fmat[ei+10][ej] = l12c * (C64::from(vbbcd)*er_gd_val - C64::from(vbbcc)*ge_mul_er_gf - C64::from(vabcd)*ge_mul_er_gc + C64::from(vabcc)*ga_mul_er_gf);
            fmat[ei+10][ej+10] = l12c * (C64::from(vbbdd)*er_gd_val - C64::from(vbbcd)*ge_mul_er_gf - C64::from(vabdd)*ge_mul_er_gc + C64::from(vabcd)*ga_mul_er_gf);
        }

        // === Edge-Face block ===
        for ej in 0..4 {
            let ej1 = local_tri_map[ej][0];
            let ej2 = local_tri_map[ej][1];
            let fj = local_tri_map[ej][2];
            let c = letters(ej1);
            let d = letters(ej2);
            let f = letters(fj);
            let gc = &gls[ej1];
            let gd = &gls[ej2];
            let gf = &gls[fj];

            let vac = vc(a,c,0,0); let vad = vc(a,d,0,0); let vaf = vc(a,f,0,0);
            let vbc = vc(b,c,0,0); let vbd = vc(b,d,0,0); let vbf = vc(b,f,0,0);
            let vabcd = vc(a,b,c,d);
            let vabdf = vc(a,b,d,f); let vabcf = vc(a,b,f,c);
            let vaacd = vc(a,a,c,d); let vaadf = vc(a,a,d,f); let vaacf = vc(a,a,c,f);
            let vbbcd = vc(b,b,c,d); let vbbdf = vc(b,b,d,f); let vbbcf = vc(b,b,f,c);

            let lab2 = ds[ej1][ej2];
            let lac2 = ds[ej1][fj];

            let cross_ae = cross_c(ga, gb);
            let cross_df = dot_c(&cross_ae, &matmul3(ms, &cross_c(gc, gf)));
            let cross_cd = dot_c(&cross_ae, &matmul3(ms, &cross_c(gd, gf)));
            let ae_mul_cf = dot_c(&cross_ae, &matmul3(ms, &cross_c(gc, gd)));

            let er_gf_v = matmul3(mm, gf);
            let er_gc_v = matmul3(mm, gc);
            let er_gd_v = matmul3(mm, gd);
            let ge_mul_er_gf = dot_c(ga, &er_gf_v);
            let ge_mul_er_gc = dot_c(ga, &er_gc_v);
            let ga_mul_er_gf = dot_c(gb, &er_gf_v);
            let ga_mul_er_gc = dot_c(gb, &er_gc_v);
            let ge_mul_er_gd = dot_c(ga, &er_gd_v);
            let ga_mul_er_gd = dot_c(gb, &er_gd_v);

            // Stiffness edge-face
            dmat[ei][ej+6] = C64::from(l1*lac2) * (C64::from(-6.0)*C64::from(vad)*cross_df - C64::from(3.0)*C64::from(vac)*cross_cd - C64::from(3.0)*C64::from(vaf)*ae_mul_cf);
            dmat[ei][ej+16] = C64::from(l1*lab2) * (C64::from(6.0)*C64::from(vaf)*ae_mul_cf + C64::from(3.0)*C64::from(vad)*cross_df - C64::from(3.0)*C64::from(vac)*cross_cd);
            dmat[ei+10][ej+6] = C64::from(l1*lac2) * (C64::from(-6.0)*C64::from(vbd)*cross_df - C64::from(3.0)*C64::from(vbc)*cross_cd - C64::from(3.0)*C64::from(vbf)*ae_mul_cf);
            dmat[ei+10][ej+16] = C64::from(l1*lab2) * (C64::from(6.0)*C64::from(vbf)*ae_mul_cf + C64::from(3.0)*C64::from(vbd)*cross_df - C64::from(3.0)*C64::from(vbc)*cross_cd);

            // Mass edge-face
            fmat[ei][ej+6] = C64::from(l1*lac2) * (C64::from(vabcd)*ge_mul_er_gf - C64::from(vabdf)*ge_mul_er_gc - C64::from(vaacd)*ga_mul_er_gf + C64::from(vaadf)*ga_mul_er_gc);
            fmat[ei][ej+16] = C64::from(l1*lab2) * (C64::from(vabdf)*ge_mul_er_gc - C64::from(vabcf)*ge_mul_er_gd - C64::from(vaadf)*ga_mul_er_gc + C64::from(vaacf)*ga_mul_er_gd);
            fmat[ei+10][ej+6] = C64::from(l1*lac2) * (C64::from(vbbcd)*ge_mul_er_gf - C64::from(vbbdf)*ge_mul_er_gc - C64::from(vabcd)*ga_mul_er_gf + C64::from(vabdf)*ga_mul_er_gc);
            fmat[ei+10][ej+16] = C64::from(l1*lab2) * (C64::from(vbbdf)*ge_mul_er_gc - C64::from(vbbcf)*ge_mul_er_gd - C64::from(vabdf)*ga_mul_er_gc + C64::from(vabcf)*ga_mul_er_gd);
        }
    }

    // Mirror the edge-face transpose (symmetric matrices)
    for i in 0..4 {
        for j in 0..6 {
            dmat[i+6][j] = dmat[j][i+6];
            fmat[i+6][j] = fmat[j][i+6];
            dmat[i+16][j] = dmat[j][i+16];
            fmat[i+16][j] = fmat[j][i+16];
            dmat[i+6][j+10] = dmat[j+10][i+6];
            fmat[i+6][j+10] = fmat[j+10][i+6];
            dmat[i+16][j+10] = dmat[j+10][i+16];
            fmat[i+16][j+10] = fmat[j+10][i+16];
        }
    }

    // === Face-Face block ===
    for ei in 0..4 {
        let ei1 = local_tri_map[ei][0];
        let ei2 = local_tri_map[ei][1];
        let fi = local_tri_map[ei][2];
        let a = letters(ei1);
        let b = letters(ei2);
        let e = letters(fi);
        let ga = &gls[ei1];
        let gb = &gls[ei2];
        let ge = &gls[fi];
        let lac1 = ds[ei1][fi];
        let lab1 = ds[ei1][ei2];

        for ej in 0..4 {
            let ej1 = local_tri_map[ej][0];
            let ej2 = local_tri_map[ej][1];
            let fj = local_tri_map[ej][2];
            let c = letters(ej1);
            let d = letters(ej2);
            let f = letters(fj);
            let gc = &gls[ej1];
            let gd = &gls[ej2];
            let gf = &gls[fj];

            let vad = vc(a,d,0,0); let vac = vc(a,c,0,0); let vaf = vc(a,f,0,0);
            let vbf = vc(b,f,0,0); let vbc = vc(b,c,0,0); let vbd = vc(b,d,0,0);
            let vde = vc(e,d,0,0); let vef = vc(e,f,0,0); let vce = vc(e,c,0,0);
            let vabcd = vc(a,b,c,d);
            let vabdf = vc(a,b,d,f); let vabcf = vc(a,b,f,c);
            let vacef = vc(a,c,e,f);
            let vbcde = vc(b,c,d,f);  // Note: EMerge has VBCDE = VOLUME_COEFF_CACHE[B,C,D,F]
            let vbdef = vc(b,e,d,f);
            let vacde = vc(e,a,c,d);
            let vbcef = vc(b,e,f,c);
            let vadef = vc(e,a,d,f);

            let lac2 = ds[ej1][fj];
            let lab2 = ds[ej1][ej2];

            let cross_ae = cross_c(ga, ge);
            let cross_be = cross_c(gb, ge);
            let cross_ab = cross_c(ga, gb);
            let cross_cf = matmul3(ms, &cross_c(gc, gf));
            let cross_df = matmul3(ms, &cross_c(gd, gf));
            let cross_cd = matmul3(ms, &cross_c(gc, gd));
            let ae_cf = dot_c(&cross_ae, &cross_cf);
            let ae_df = dot_c(&cross_ae, &cross_df);
            let ae_cd = dot_c(&cross_ae, &cross_cd);
            let be_cf = dot_c(&cross_be, &cross_cf);
            let be_df = dot_c(&cross_be, &cross_df);
            let be_cd = dot_c(&cross_be, &cross_cd);
            let ab_cf = dot_c(&cross_ab, &cross_cf);
            let ab_df = dot_c(&cross_ab, &cross_df);
            let ab_cd = dot_c(&cross_ab, &cross_cd);

            let er_gf_v = matmul3(mm, gf);
            let er_gc_v = matmul3(mm, gc);
            let er_gd_v = matmul3(mm, gd);
            let ge_er_gf = dot_c(ge, &er_gf_v);
            let ge_er_gc = dot_c(ge, &er_gc_v);
            let ga_er_gf = dot_c(ga, &er_gf_v);
            let ga_er_gc = dot_c(ga, &er_gc_v);
            let ge_er_gd = dot_c(ge, &er_gd_v);
            let ga_er_gd = dot_c(ga, &er_gd_v);
            let gb_er_gf = dot_c(gb, &er_gf_v);
            let gb_er_gc = dot_c(gb, &er_gc_v);
            let gb_er_gd = dot_c(gb, &er_gd_v);

            let q1 = C64::from(2.0*vad)*be_cf + C64::from(vac)*be_df + C64::from(vaf)*be_cd;
            let l12 = C64::from(-2.0*vaf)*be_cd - C64::from(vad)*be_cf + C64::from(vac)*be_df;

            dmat[ei+6][ej+6] = C64::from(lac1*lac2) * (
                C64::from(4.0*vbd)*ae_cf + C64::from(2.0*vbc)*ae_df + C64::from(2.0*vbf)*ae_cd
                + q1 + C64::from(2.0*vde)*ab_cf + C64::from(vce)*ab_df + C64::from(vef)*ab_cd);
            dmat[ei+6][ej+16] = C64::from(lac1*lab2) * (
                C64::from(-4.0*vbf)*ae_cd - C64::from(2.0*vbd)*ae_cf + C64::from(2.0*vbc)*ae_df
                + l12 - C64::from(2.0*vef)*ab_cd - C64::from(vde)*ab_cf + C64::from(vce)*ab_df);
            dmat[ei+16][ej+6] = C64::from(lab1*lac2) * (
                C64::from(-4.0*vde)*ab_cf - C64::from(2.0*vce)*ab_df - C64::from(2.0*vef)*ab_cd
                - C64::from(2.0*vbd)*ae_cf - C64::from(vbc)*ae_df - C64::from(vbf)*ae_cd + q1);
            dmat[ei+16][ej+16] = C64::from(lab1*lab2) * (
                C64::from(4.0*vef)*ab_cd + C64::from(2.0*vde)*ab_cf - C64::from(2.0*vce)*ab_df
                + C64::from(2.0*vbf)*ae_cd + C64::from(vbd)*ae_cf - C64::from(vbc)*ae_df + l12);

            fmat[ei+6][ej+6] = C64::from(lac1*lac2) * (C64::from(vabcd)*ge_er_gf - C64::from(vabdf)*ge_er_gc - C64::from(vbcde)*ga_er_gf + C64::from(vbdef)*ga_er_gc);
            fmat[ei+6][ej+16] = C64::from(lac1*lab2) * (C64::from(vabdf)*ge_er_gc - C64::from(vabcf)*ge_er_gd - C64::from(vbdef)*ga_er_gc + C64::from(vbcef)*ga_er_gd);
            fmat[ei+16][ej+6] = C64::from(lab1*lac2) * (C64::from(vbcde)*ga_er_gf - C64::from(vbdef)*ga_er_gc - C64::from(vacde)*gb_er_gf + C64::from(vadef)*gb_er_gc);
            fmat[ei+16][ej+16] = C64::from(lab1*lab2) * (C64::from(vbdef)*ga_er_gc - C64::from(vbcef)*ga_er_gd - C64::from(vadef)*gb_er_gc + C64::from(vacef)*gb_er_gd);
        }
    }

    // Apply scaling
    for i in 0..20 {
        for j in 0..20 {
            dmat[i][j] *= C64::from(ka);
            fmat[i][j] *= C64::from(kb);
        }
    }

    (dmat, fmat)
}

/// Assemble global stiffness (E) and mass (B) matrices from all tetrahedra.
/// Returns COO triplets: (rows, cols, data_e, data_b).
/// Parallelized with rayon — each tet writes to its own 400-entry slice.
pub fn assemble_global_matrices(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    er: &[[[C64; 3]; 3]],  // per-tet permittivity tensors
    ur: &[[[C64; 3]; 3]],  // per-tet permeability tensors
) -> (Vec<usize>, Vec<usize>, Vec<C64>, Vec<C64>)
{
    use rayon::prelude::*;

    let n_tets = mesh.n_tets();
    let nnz = n_tets * 400; // 20×20 per tet
    let mut rows = vec![0usize; nnz];
    let mut cols = vec![0usize; nnz];
    let mut data_e = vec![C64::new(0.0, 0.0); nnz];
    let mut data_b = vec![C64::new(0.0, 0.0); nnz];

    let vc_base = VolumeCoeffCache::new();

    // Split into per-tet slices for parallel write
    let chunks: Vec<(usize, &mut [usize], &mut [usize], &mut [C64], &mut [C64])> = {
        let rows_chunks: Vec<&mut [usize]> = rows.chunks_mut(400).collect();
        let cols_chunks: Vec<&mut [usize]> = cols.chunks_mut(400).collect();
        let de_chunks: Vec<&mut [C64]> = data_e.chunks_mut(400).collect();
        let db_chunks: Vec<&mut [C64]> = data_b.chunks_mut(400).collect();
        (0..n_tets).zip(rows_chunks).zip(cols_chunks).zip(de_chunks).zip(db_chunks)
            .map(|((((i, r), c), de), db)| (i, r, c, de, db))
            .collect()
    };

    chunks.into_par_iter().for_each(|(itet, row_slice, col_slice, de_slice, db_slice)| {
        let tet = &mesh.tets[itet];
        let xs = [mesh.nodes[tet[0]][0], mesh.nodes[tet[1]][0], mesh.nodes[tet[2]][0], mesh.nodes[tet[3]][0]];
        let ys = [mesh.nodes[tet[0]][1], mesh.nodes[tet[1]][1], mesh.nodes[tet[2]][1], mesh.nodes[tet[3]][1]];
        let zs = [mesh.nodes[tet[0]][2], mesh.nodes[tet[1]][2], mesh.nodes[tet[2]][2], mesh.nodes[tet[3]][2]];

        let tet_edges = &mesh.tet_to_edge[itet];
        let edge_lengths: [f64; 6] = std::array::from_fn(|i| mesh.edge_lengths[tet_edges[i]]);

        let global_edge_nodes: [[usize; 2]; 6] = std::array::from_fn(|i| mesh.edges[tet_edges[i]]);
        let local_edge_map = crate::basis::local_mapping(tet, &global_edge_nodes);

        let tet_tris = &mesh.tet_to_tri[itet];
        let global_tri_nodes: [[usize; 3]; 4] = std::array::from_fn(|i| mesh.tris[tet_tris[i]]);
        let local_tri_map = crate::basis::local_mapping_tri(tet, &global_tri_nodes);

        let ms = matinv3(&ur[itet]);
        let mm = &er[itet];

        let (esub, bsub) = ned2_tet_stiff_mass(
            &xs, &ys, &zs, &edge_lengths,
            &local_edge_map, &local_tri_map,
            &ms, mm, &vc_base,
        );

        let indices = &basis.tet_to_field[itet];
        for ii in 0..20 {
            for jj in 0..20 {
                let idx = ii * 20 + jj;
                row_slice[idx] = indices[ii];
                col_slice[idx] = indices[jj];
                de_slice[idx] = esub[ii][jj];
                db_slice[idx] = bsub[ii][jj];
            }
        }
    });

    (rows, cols, data_e, data_b)
}
