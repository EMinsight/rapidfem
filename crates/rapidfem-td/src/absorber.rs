//! Absorbing boundary layers.
//!
//! A graded impedance-matched lossy layer (`σ/ε = σ*/μ`) absorbs outgoing
//! waves with no reflection at the layer interface; ramping the loss from
//! zero keeps the entry smooth. This is the matched-layer absorber — a true
//! CFS-PML (reflectionless at *all* incidence angles, via auxiliary
//! differential equations) is a further extension.

use crate::rhs::ElemMaterial;
use rapidfem_core::mesh::Mesh;

/// As [`absorbing_layer`] but for the `axis = inner` face (the low-coordinate
/// end), within `thickness` of the plane `x_axis = inner`.
pub fn absorbing_layer_low(
    mesh: &Mesh,
    axis: usize,
    inner: f64,
    thickness: f64,
    nu_max: f64,
) -> Vec<ElemMaterial> {
    mesh.tets
        .iter()
        .map(|tet| {
            let centroid: f64 =
                tet.iter().map(|&n| mesh.nodes[n][axis]).sum::<f64>() / 4.0;
            let depth = (inner + thickness) - centroid;
            if depth <= 0.0 {
                ElemMaterial::VACUUM
            } else {
                let frac = (depth / thickness).min(1.0);
                ElemMaterial::matched_absorber(1.0, 1.0, nu_max * frac * frac)
            }
        })
        .collect()
}

/// Per-element materials: vacuum everywhere, except a graded matched absorber
/// within `thickness` of the plane `x_axis = outer`. The loss ramps
/// quadratically from `0` at the layer entry to `nu_max` at `outer`.
pub fn absorbing_layer(
    mesh: &Mesh,
    axis: usize,
    outer: f64,
    thickness: f64,
    nu_max: f64,
) -> Vec<ElemMaterial> {
    mesh.tets
        .iter()
        .map(|tet| {
            let centroid: f64 =
                tet.iter().map(|&n| mesh.nodes[n][axis]).sum::<f64>() / 4.0;
            let depth = centroid - (outer - thickness);
            if depth <= 0.0 {
                ElemMaterial::VACUUM
            } else {
                let frac = (depth / thickness).min(1.0);
                ElemMaterial::matched_absorber(1.0, 1.0, nu_max * frac * frac)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_gen::structured_box;
    use crate::propagator::expmv;
    use crate::rhs::MaxwellOperator;

    fn energy(y: &[f64], mm: &[f64], n: usize) -> f64 {
        let mut e = 0.0;
        for i in 0..n {
            for j in 0..n {
                e += y[i] * mm[i * n + j] * y[j];
            }
        }
        e
    }

    #[test]
    fn absorbing_layer_drains_field_energy() {
        // A field disturbance in a closed PEC channel: with a matched
        // absorbing layer at one end the energy drains away; with vacuum the
        // central-flux operator conserves it.
        let lz = 5.0;
        let mesh = structured_box(1, 1, 10, 0.5, 0.5, lz);

        let run = |materials: &[ElemMaterial]| -> f64 {
            // central flux ⇒ vacuum is exactly energy-conserving.
            let op =
                MaxwellOperator::new_with_materials(&mesh, 2, 0.0, materials);
            let n = op.n_dof();
            let mm = op.assemble_energy_mass();
            let mut y = vec![0.0; n];
            y[op.nearest_node_dof([0.25, 0.25, 0.6], 0, 0)] = 1.0;
            let e0 = energy(&y, &mm, n);
            for _ in 0..900 {
                y = expmv(|x| op.apply(x), &y, 0.06, 24);
            }
            energy(&y, &mm, n) / e0
        };

        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        // absorber covers most of the channel — the small clean region holds
        // the disturbance, everything propagating into the layer is absorbed.
        let absorber = absorbing_layer(&mesh, 2, lz, 3.5, 6.0);
        let frac_vac = run(&vacuum);
        let frac_abs = run(&absorber);

        assert!(
            frac_vac > 0.9,
            "vacuum must conserve energy — kept {frac_vac:.3}"
        );
        // The absorber drains the bulk of the energy; the residual is the
        // slow-decaying mode tail (modes with a field node in the layer).
        // What matters is the decisive contrast against the vacuum run.
        assert!(
            frac_abs < 0.4,
            "absorbing layer must drain energy — kept {frac_abs:.3}"
        );
        assert!(
            frac_vac / frac_abs > 2.5,
            "absorber vs vacuum contrast too weak: {frac_vac:.3} / {frac_abs:.3}"
        );
    }
}
