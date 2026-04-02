//! Exact port of sparam.py: S-parameter extraction via surface integrals.
//!
//! Functions: sparam_waveport, sparam_field_power, sparam_mode_power
//! Also includes surface_integral from integrals.py.

use num_complex::Complex64 as C64;
use crate::quadrature::gaus_quad_tri;
use crate::waveguide::RectWaveguide;

/// Port of integrals.py: surface_integral
///
/// Computes I = Σ_triangles ∫∫ f(x,y,z) dA
/// using Gauss quadrature on each triangle.
///
/// `nodes`: all mesh nodes [n_nodes][3]
/// `triangles`: triangle vertex indices [n_tris][3]
/// `function`: scalar function f(x,y,z) → Complex64
/// `gq_order`: quadrature order (default 4)
pub fn surface_integral(
    nodes: &[[f64; 3]],
    triangles: &[[usize; 3]],
    function: &dyn Fn(f64, f64, f64) -> C64,
    gq_order: usize,
) -> C64 {
    let dpts = gaus_quad_tri(gq_order);
    let mut total = C64::new(0.0, 0.0);

    for tri in triangles {
        let v1 = nodes[tri[0]];
        let v2 = nodes[tri[1]];
        let v3 = nodes[tri[2]];

        // calc_area
        let e1 = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]];
        let e2 = [v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]];
        let cr = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
        let area = 0.5 * (cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]).sqrt();

        // Evaluate f at quadrature points and sum
        let mut tri_sum = C64::new(0.0, 0.0);
        for qp in &dpts {
            let (w, l1, l2, l3) = (qp[0], qp[1], qp[2], qp[3]);
            let x = v1[0]*l1 + v2[0]*l2 + v3[0]*l3;
            let y = v1[1]*l1 + v2[1]*l2 + v3[1]*l3;
            let z = v1[2]*l1 + v2[2]*l2 + v3[2]*l3;
            tri_sum += C64::from(w) * function(x, y, z);
        }
        total += tri_sum * C64::from(area);
    }

    total
}

/// Port of sparam.py: sparam_waveport
///
/// S = ∫ (E_field - Q·E_mode) · conj(E_mode) dS / ∫ |E_mode|² dS
///
/// `active`: if true, Q=1 (excited port); if false, Q=0 (observation port)
/// `fieldf`: function that returns (Ex, Ey, Ez) at (x, y, z)
pub fn sparam_waveport(
    nodes: &[[f64; 3]],
    tri_verts: &[[usize; 3]],
    port: &RectWaveguide,
    k0: f64,
    active: bool,
    fieldf: &dyn Fn(f64, f64, f64) -> (C64, C64, C64),
    gq_order: usize,
) -> C64 {
    let q = if active { 1.0 } else { 0.0 };

    // ∫ (E_field - Q·E_mode) · conj(E_mode) dS
    let mode_dot_field = surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, k0);
        let (fx, fy, fz) = fieldf(x, y, z);

        let ex1 = fx - C64::from(q * mx);
        let ey1 = fy - C64::from(q * my);
        let ez1 = fz - C64::from(q * mz);

        let ex2 = C64::from(mx).conj();
        let ey2 = C64::from(my).conj();
        let ez2 = C64::from(mz).conj();

        ex1*ex2 + ey1*ey2 + ez1*ez2
    }, gq_order);

    // ∫ |E_mode|² dS
    let norm = surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, k0);
        C64::from(mx*mx + my*my + mz*mz)
    }, gq_order);

    if norm.norm() < 1e-30 {
        return C64::new(0.0, 0.0);
    }

    mode_dot_field / norm
}

/// Port of sparam.py: sparam_field_power
///
/// P_field = ∫ (E_field - Q·E_mode) · conj(E_mode) / (2·Z_mode) dS
pub fn sparam_field_power(
    nodes: &[[f64; 3]],
    tri_verts: &[[usize; 3]],
    port: &RectWaveguide,
    k0: f64,
    active: bool,
    fieldf: &dyn Fn(f64, f64, f64) -> (C64, C64, C64),
    gq_order: usize,
) -> C64 {
    let q = if active { 1.0 } else { 0.0 };
    let z_mode = port.z_mode(k0);

    surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, k0);
        let (fx, fy, fz) = fieldf(x, y, z);

        let ex1 = fx - C64::from(q * mx);
        let ey1 = fy - C64::from(q * my);
        let ez1 = fz - C64::from(q * mz);

        let ex2 = C64::from(mx).conj();
        let ey2 = C64::from(my).conj();
        let ez2 = C64::from(mz).conj();

        (ex1*ex2 + ey1*ey2 + ez1*ez2) / C64::from(2.0 * z_mode)
    }, gq_order)
}

/// Port of sparam.py: sparam_mode_power
///
/// P_mode = ∫ |E_mode|² / (2·Z_mode) dS
pub fn sparam_mode_power(
    nodes: &[[f64; 3]],
    tri_verts: &[[usize; 3]],
    port: &RectWaveguide,
    k0: f64,
    gq_order: usize,
) -> C64 {
    let z_mode = port.z_mode(k0);

    surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, k0);
        let ex2 = C64::from(mx).conj();
        let ey2 = C64::from(my).conj();
        let ez2 = C64::from(mz).conj();

        (C64::from(mx)*ex2 + C64::from(my)*ey2 + C64::from(mz)*ez2) / C64::from(2.0 * z_mode)
    }, gq_order)
}
