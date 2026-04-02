//! S-parameter extraction via surface integrals.
//! Mirrors sparam.py: sparam_field_power, sparam_mode_power.
//! NO HACKS — S-params computed directly from field/mode integrals.

use num_complex::Complex64 as C64;
use crate::quadrature::gaus_quad_tri;
use crate::waveguide::RectWaveguide;
use crate::mesh::Mesh;

/// Compute S-parameter for one port observation.
///
/// S = field_power / mode_power where:
///   field_power = ∫ (E_FEM - Q·E_mode) · conj(E_mode) / (2·Z_mode) dS
///   mode_power  = ∫ |E_mode|² / (2·Z_mode) dS
///
/// `active`: true if this port was excited (Q=1), false if passive (Q=0)
/// `field_eval`: function that evaluates E_FEM at a physical point
pub fn extract_s_parameter(
    mesh: &Mesh,
    port: &RectWaveguide,
    port_tris: &[usize],
    k0: f64,
    active: bool,
    field_eval: &dyn Fn(f64, f64, f64) -> [C64; 3],
    quad_order: usize,
) -> C64 {
    let q = if active { 1.0 } else { 0.0 };
    let z_mode = port.z_mode(k0);
    let quad_pts = gaus_quad_tri(quad_order);

    let mut field_power = C64::new(0.0, 0.0);
    let mut mode_power = C64::new(0.0, 0.0);

    for &ti in port_tris {
        let tri = &mesh.tris[ti];
        let v0 = mesh.nodes[tri[0]];
        let v1 = mesh.nodes[tri[1]];
        let v2 = mesh.nodes[tri[2]];

        // Triangle area
        let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
        let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
        let cr = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
        let area = 0.5 * (cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]).sqrt();

        for qp in &quad_pts {
            let (w, l1, l2, l3) = (qp[0], qp[1], qp[2], qp[3]);

            // Physical point
            let x = v0[0]*l1 + v1[0]*l2 + v2[0]*l3;
            let y = v0[1]*l1 + v1[1]*l2 + v2[1]*l3;
            let z = v0[2]*l1 + v1[2]*l2 + v2[2]*l3;

            // Mode field at this point
            let e_mode = port.mode_field_global(x, y, z, k0);
            let e_mode_c = [C64::from(e_mode[0]), C64::from(e_mode[1]), C64::from(e_mode[2])];
            let e_mode_conj = [e_mode_c[0].conj(), e_mode_c[1].conj(), e_mode_c[2].conj()];

            // FEM field at this point
            let e_fem = field_eval(x, y, z);

            // (E_FEM - Q·E_mode) · conj(E_mode) / (2·Z_mode)
            let e_scat = [
                e_fem[0] - C64::from(q) * e_mode_c[0],
                e_fem[1] - C64::from(q) * e_mode_c[1],
                e_fem[2] - C64::from(q) * e_mode_c[2],
            ];
            let dot_field = e_scat[0]*e_mode_conj[0] + e_scat[1]*e_mode_conj[1] + e_scat[2]*e_mode_conj[2];
            field_power += dot_field * C64::from(w * area / (2.0 * z_mode));

            // |E_mode|² / (2·Z_mode)
            let dot_mode = e_mode_c[0]*e_mode_conj[0] + e_mode_c[1]*e_mode_conj[1] + e_mode_c[2]*e_mode_conj[2];
            mode_power += dot_mode * C64::from(w * area / (2.0 * z_mode));
        }
    }

    if mode_power.norm() < 1e-30 {
        return C64::new(0.0, 0.0);
    }

    field_power / mode_power
}
