//! Exact port of sparam.py: S-parameter extraction via surface integrals.
//!
//! Functions: sparam_waveport, sparam_field_power, sparam_mode_power, sparam_voltage
//! Also includes surface_integral from integrals.py.
//!
//! All functions accept &dyn Port via the port trait.

use num_complex::Complex64 as C64;
use crate::quadrature::gaus_quad_tri;
use crate::port::Port;

/// Port of integrals.py: surface_integral
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

        let e1 = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]];
        let e2 = [v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]];
        let cr = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
        let area = 0.5 * (cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]).sqrt();

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
pub fn sparam_waveport(
    nodes: &[[f64; 3]],
    tri_verts: &[[usize; 3]],
    port: &dyn Port,
    k0: f64,
    active: bool,
    fieldf: &dyn Fn(f64, f64, f64) -> (C64, C64, C64),
    gq_order: usize,
) -> C64 {
    let q = if active { 1.0 } else { 0.0 };

    let mode_dot_field = surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, k0).unwrap_or((0.0, 0.0, 0.0));
        let (fx, fy, fz) = fieldf(x, y, z);

        let ex1 = fx - C64::from(q * mx);
        let ey1 = fy - C64::from(q * my);
        let ez1 = fz - C64::from(q * mz);

        let ex2 = C64::from(mx).conj();
        let ey2 = C64::from(my).conj();
        let ez2 = C64::from(mz).conj();

        ex1*ex2 + ey1*ey2 + ez1*ez2
    }, gq_order);

    let norm = surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, k0).unwrap_or((0.0, 0.0, 0.0));
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
    port: &dyn Port,
    k0: f64,
    active: bool,
    fieldf: &dyn Fn(f64, f64, f64) -> (C64, C64, C64),
    gq_order: usize,
) -> C64 {
    let q = if active { 1.0 } else { 0.0 };
    let z_mode = port.z_mode(k0);

    surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, k0).unwrap_or((0.0, 0.0, 0.0));
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
    port: &dyn Port,
    k0: f64,
    gq_order: usize,
) -> C64 {
    let z_mode = port.z_mode(k0);

    surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, k0).unwrap_or((0.0, 0.0, 0.0));
        let ex2 = C64::from(mx).conj();
        let ey2 = C64::from(my).conj();
        let ez2 = C64::from(mz).conj();

        (C64::from(mx)*ex2 + C64::from(my)*ey2 + C64::from(mz)*ez2) / C64::from(2.0 * z_mode)
    }, gq_order)
}

/// Port of microwave_3d.py: voltage integration S-param extraction for lumped ports.
pub fn sparam_voltage(
    port: &crate::waveguide::LumpedPort,
    active: bool,
    fieldf: &dyn Fn(f64, f64, f64) -> (C64, C64, C64),
    line_points: &[[f64; 3]],
) -> C64 {
    let n_pts = line_points.len();
    if n_pts < 2 { return C64::new(0.0, 0.0); }

    let mut v_total = C64::new(0.0, 0.0);
    for i in 0..(n_pts - 1) {
        let p0 = line_points[i];
        let p1 = line_points[i + 1];
        let mid = [(p0[0]+p1[0])/2.0, (p0[1]+p1[1])/2.0, (p0[2]+p1[2])/2.0];
        let dl = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];

        let (ex, ey, ez) = fieldf(mid[0], mid[1], mid[2]);
        v_total += ex * C64::from(dl[0]) + ey * C64::from(dl[1]) + ez * C64::from(dl[2]);
    }

    let v_inc = C64::from(port.voltage());

    let (a, b) = if active {
        (v_inc, v_total - v_inc)
    } else {
        (v_inc, v_total)
    };

    let norm = C64::from((1.0 / (2.0 * port.z0)).sqrt());
    let b_sig = b * norm;
    let a_sig = a * norm;

    if a_sig.norm() < 1e-30 {
        return C64::new(0.0, 0.0);
    }
    b_sig / a_sig
}
