//! DG Maxwell RHS operator.
//!
//! The semi-discrete DG form of the vacuum Maxwell curl equations
//! (`∂E/∂t = ∇×H`, `∂H/∂t = -∇×E`) splits per element into a volume term
//! (the physical curl) and a surface term (the numerical flux). This module
//! builds those operators; the volume curl is assembled and validated first.
//!
//! Per-element fields are stored node-major: `field[node*3 + component]`,
//! with components ordered `x, y, z`.

use crate::dg_basis::ReferenceElement;
use crate::geom_factors::GeometricFactors;

/// Physical curl of a vector field on a single element.
///
/// `field` holds `3·Np` values (`field[node*3 + comp]`); the result has the
/// same layout and contains `∇×field` sampled at the element nodes.
pub fn element_curl(
    re: &ReferenceElement,
    gf: &GeometricFactors,
    field: &[f64],
) -> Vec<f64> {
    let n = re.n_nodes;
    debug_assert_eq!(field.len(), 3 * n);
    let dref = [&re.diff_r, &re.diff_s, &re.diff_t];

    // pd[phys][comp][node] = ∂(field_comp)/∂x_phys.
    let mut pd = [[(); 3]; 3].map(|row| row.map(|_| vec![0.0; n]));
    for comp in 0..3 {
        // Reference derivatives of this component.
        let mut rd = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for (k, d) in dref.iter().enumerate() {
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += d[i * n + j] * field[j * 3 + comp];
                }
                rd[k][i] = acc;
            }
        }
        // Combine via the metric into physical derivatives.
        for (phys, pd_phys) in pd.iter_mut().enumerate() {
            for i in 0..n {
                pd_phys[comp][i] = gf.jacobian_inv[0][phys] * rd[0][i]
                    + gf.jacobian_inv[1][phys] * rd[1][i]
                    + gf.jacobian_inv[2][phys] * rd[2][i];
            }
        }
    }

    // curl_x = ∂Fz/∂y - ∂Fy/∂z, and cyclic.
    let mut out = vec![0.0; 3 * n];
    for i in 0..n {
        out[i * 3] = pd[1][2][i] - pd[2][1][i];
        out[i * 3 + 1] = pd[2][0][i] - pd[0][2][i];
        out[i * 3 + 2] = pd[0][1][i] - pd[1][0][i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curl_of_polynomial_field_is_exact() {
        // On a sheared physical element, the discrete curl reproduces the
        // analytic curl of a degree-2 vector field exactly at the nodes.
        let v = [
            [0.3, -0.2, 0.1],
            [1.4, 0.1, -0.2],
            [0.2, 1.3, 0.4],
            [-0.1, 0.2, 1.6],
        ];
        let gf = GeometricFactors::for_tet(&v);
        let re = ReferenceElement::new(2);
        let n = re.n_nodes;
        let pn: Vec<[f64; 3]> = re.nodes.iter().map(|&xi| gf.map(xi)).collect();

        // F = (y·z, 2·z·x, 3·x·y)  ⇒  ∇×F = (x, -2y, z).
        let mut field = vec![0.0; 3 * n];
        for (i, p) in pn.iter().enumerate() {
            field[i * 3] = p[1] * p[2];
            field[i * 3 + 1] = 2.0 * p[2] * p[0];
            field[i * 3 + 2] = 3.0 * p[0] * p[1];
        }

        let curl = element_curl(&re, &gf, &field);
        for (i, p) in pn.iter().enumerate() {
            let want = [p[0], -2.0 * p[1], p[2]];
            for c in 0..3 {
                assert!(
                    (curl[i * 3 + c] - want[c]).abs() < 1e-9,
                    "node {i} comp {c}: got {}, want {}",
                    curl[i * 3 + c],
                    want[c]
                );
            }
        }
    }

    #[test]
    fn curl_of_constant_field_vanishes() {
        let gf = GeometricFactors::for_tet(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let re = ReferenceElement::new(3);
        let field = vec![0.7; 3 * re.n_nodes];
        let curl = element_curl(&re, &gf, &field);
        assert!(curl.iter().all(|c| c.abs() < 1e-10));
    }
}
