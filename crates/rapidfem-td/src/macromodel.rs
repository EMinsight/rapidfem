//! Block-Krylov MIMO macromodel of a multiport time-domain Maxwell system.
//!
//! Given the matrix-free DG operator `A` (from
//! [`crate::rhs::MaxwellOperator::apply`]) and its `N` modal ports, build a
//! compact state-space `(A_hat, B_hat, C_hat)` of order `r` purely by
//! projection onto an impulse block-Krylov subspace seeded by all
//! port-injection vectors:
//!
//! ```text
//!   K_r(A, B) = span{B, A*B, A^2*B, ...}     B = [b_1, ..., b_N]   (one column per modal port)
//!   V    orthonormal basis of K_r(A, B), `n_dof x r`
//!   A_hat = V^T A V        B_hat = V^T B        C_hat = [C^E; C^H] V
//! ```
//!
//! The TD operator is real, so `A_hat`, `B_hat`, `C_hat` are real. The
//! frequency response `(j*omega*I - A_hat)^-1` is complex, formed and
//! solved per omega.
//!
//! **S-parameters.** With drive at port `j` alone, the small dense solve
//! `(j*omega*I - A_hat) x = B_hat[:, j]` (an `r x r` complex system)
//! gives the reduced state amplitude; the per-port modal-amplitude
//! readouts are then `z^E = C_hat^E x`, `z^H = C_hat^H x` and the
//! forward / backward modal split
//! `A_i, B_i = (z^E_i +/- Z_i(omega) * z^H_i) / 2` yields
//! `S[i, j] = B_i / A_j`. `Z_i(omega)` is the dispersive modal impedance
//! of port `i`, from
//! [`crate::rhs::MaxwellOperator::port_impedance`].
//!
//! Ports without a mode (`PortSpec.mode = None`) are pure absorbing
//! boundaries, not input / output channels; they do not contribute
//! columns to `B` nor rows to `C_hat`, and `S` is indexed over the
//! **modal** ports only.
//!
//! This is the M1 deliverable of `docs/td-macromodel-plan.md`: CPU only,
//! impulse-Krylov only, no shift-invert solves, no transient run. The
//! GPU acceleration (M6), the SPRIM structure-preserving variant (M3)
//! and the multi-point shift-invert refinement (M4) lift on top of this
//! base.

use num_complex::Complex64 as C64;

use crate::constants::{
    Accum, Field, MACROMODEL_DEFAULT_R, MACROMODEL_DEFLATION_TOL,
};
use crate::rhs::MaxwellOperator;

/// A reduced-order MIMO state-space model of a multiport
/// `MaxwellOperator`, in block-Krylov impulse-projection form.
///
/// `(A_hat, B_hat, C_hat^E, C_hat^H)` are real and dense; the only
/// complex arithmetic is the `r x r` solve
/// `(j*omega*I - A_hat) x = B_hat[:, j]` per frequency in
/// [`MacroModel::evaluate`]. The model is indexed over the operator's
/// **modal** ports only (those whose [`crate::rhs::PortSpec`] carries a
/// `mode`); absorbing-only ports are silently skipped.
pub struct MacroModel {
    /// Reduced operator `A_hat = V^T A V`, `r x r` row-major. Real
    /// because `A` is real.
    a_hat: Vec<Field>,
    /// Reduced input map `B_hat = V^T B`, `r x N` row-major (column `j`
    /// is `B_hat[:, j]`).
    b_hat: Vec<Field>,
    /// `E`-projection rows of `C_hat`, `N x r` row-major.
    c_e_hat: Vec<Field>,
    /// `H`-projection rows of `C_hat`, `N x r` row-major.
    c_h_hat: Vec<Field>,
    /// `Z_i(omega)` of every modal port, invoked once per frequency in
    /// [`MacroModel::evaluate`].
    impedances: Box<dyn Fn(Field) -> Vec<Field> + Send + Sync>,
    /// Number of modal ports `N`.
    n_ports: usize,
    /// Realised Krylov dimension `r` after any deflation.
    r: usize,
}

impl MacroModel {
    /// Number of modal ports (the size `N` of the S-matrix).
    pub fn n_ports(&self) -> usize {
        self.n_ports
    }

    /// Realised reduced dimension (the `r` of `A_hat`, possibly smaller
    /// than the requested `r` after block-Krylov deflation).
    pub fn r(&self) -> usize {
        self.r
    }

    /// Block-Krylov projection of `op` to a reduced order `r`. The basis
    /// spans `K_r(A, B) = span{B, A*B, A^2*B, ...}` where the columns of
    /// `B` are the port-injection vectors of the modal ports. The build
    /// is pure matvec: no shift-invert, no transient.
    ///
    /// Absorbing-only ports (`PortSpec.mode = None`) are skipped: they
    /// are not input / output channels and do not contribute to the
    /// macromodel's S-matrix.
    pub fn build(op: &MaxwellOperator, r: usize) -> Self {
        Self::build_with_tol(op, r, MACROMODEL_DEFLATION_TOL)
    }

    /// Build with the default Krylov dimension
    /// ([`MACROMODEL_DEFAULT_R`]), a convenience for callers without a
    /// specific `r` in mind.
    pub fn build_default(op: &MaxwellOperator) -> Self {
        Self::build(op, MACROMODEL_DEFAULT_R)
    }

    /// Block-Krylov build with an explicit deflation tolerance, exposed
    /// for tests and tuning. See [`MacroModel::build`] for the standard
    /// caller-facing form.
    pub fn build_with_tol(op: &MaxwellOperator, r: usize, tol: Accum) -> Self {
        assert!(r > 0, "macromodel order r must be positive");

        // Collect modal-port column sources.
        let mut modal_port_idx: Vec<usize> = Vec::new();
        for k in 0..op.n_ports() {
            if op.port_has_mode(k) {
                modal_port_idx.push(k);
            }
        }
        let n_ports = modal_port_idx.len();
        assert!(
            n_ports > 0,
            "macromodel needs at least one port carrying a waveguide mode",
        );
        let b_cols: Vec<Vec<Field>> = modal_port_idx
            .iter()
            .map(|&k| op.port_source(k))
            .collect();

        // Block-Arnoldi with block-CGS2.
        //
        // `basis` is the running orthonormal `V`. We seed it with the QR
        // of `B`, then for j = 0, 1, ... process A*V[j], orthogonalise
        // against `basis`, append the normalised residual. A column
        // whose residual norm has dropped below `tol * pre_norm` is
        // deflated: the direction is already in the subspace, so it
        // brings nothing new; the build proceeds with one column fewer
        // in the block.
        let mut basis: Vec<Vec<Field>> = Vec::with_capacity(r);
        // The `R` factor of the initial block QR `B = V_0 * R`, used to
        // form `B_hat = V^T B` exactly, with no extra matvecs.
        let mut r_b: Vec<Vec<Field>> = vec![vec![0.0; n_ports]; r];

        for (col_j, col) in b_cols.iter().enumerate() {
            let mut w = col.clone();
            block_cgs2(&mut w, &basis);
            // Stash projections of the original `b_col` against the
            // existing basis into the corresponding column of `r_b`:
            // these are `(V_existing)^T b_col`. We recompute the dot
            // product against the original `col` because the CGS2 pass
            // above has already zeroed `w`'s components in the basis
            // directions, so it can no longer carry the coefficients.
            for (i, vi) in basis.iter().enumerate() {
                r_b[i][col_j] = dot(vi, col);
            }
            let wn = norm2(&w);
            if wn < tol {
                // Deflation: this port column is already spanned by the
                // earlier ones (rare unless ports share a profile).
                continue;
            }
            r_b[basis.len()][col_j] = wn;
            let inv = 1.0 / wn;
            for v in w.iter_mut() {
                *v *= inv;
            }
            basis.push(w);
            if basis.len() == r {
                break;
            }
        }

        // Expand the Krylov subspace by matvec sweeps.
        //
        // Process column j of the current basis: w = A * V[j], CGS2
        // against the whole `basis`, push if norm survives deflation.
        // The block-Krylov structure is implicit in this
        // column-by-column sweep: the columns added on the j-th pass
        // are `A * V[j]`'s residuals, which together span
        // `A * (current block)`'s complement.
        //
        // Start sweeping from the very first basis vector: V[0] is the
        // first seed column, and A*V[0] is the first new Krylov
        // direction. Resuming from `basis.len()` instead skips every
        // seed and breaks the iteration immediately (the subspace is
        // "closed" only because we never asked it to grow).
        let mut j = 0;
        while basis.len() < r {
            if j >= basis.len() {
                // Subspace closed; happens only if `r` exceeds the
                // operator's reachable dimension from `B`. Truncate
                // cleanly rather than asserting.
                break;
            }
            let mut w = op.apply(&basis[j]);
            block_cgs2(&mut w, &basis);
            let wn = norm2(&w);
            if wn < tol {
                j += 1;
                continue;
            }
            let inv = 1.0 / wn;
            for v in w.iter_mut() {
                *v *= inv;
            }
            basis.push(w);
            j += 1;
        }
        let r_eff = basis.len();

        // A_hat = V^T A V, r_eff fresh matvecs (one per column).
        //
        // Re-running the matvecs (rather than threading the Hessenberg
        // through the block-Arnoldi loop) keeps the build code simple
        // at the price of `r_eff` extra matvecs. On the CPU and at the
        // `r_eff` ~ 100s this method targets, both phases together are
        // well under the cost of a transient run.
        let mut a_hat = vec![0.0 as Field; r_eff * r_eff];
        for j in 0..r_eff {
            let av = op.apply(&basis[j]);
            for i in 0..r_eff {
                a_hat[i * r_eff + j] = dot(&basis[i], &av);
            }
        }

        // B_hat = V^T B, already accumulated in `r_b`.
        let mut b_hat = vec![0.0 as Field; r_eff * n_ports];
        for i in 0..r_eff {
            for j in 0..n_ports {
                b_hat[i * n_ports + j] = r_b[i][j];
            }
        }

        // C_hat^E, C_hat^H rows = per-port modal projections of each V[k].
        let mut c_e_hat = vec![0.0 as Field; n_ports * r_eff];
        let mut c_h_hat = vec![0.0 as Field; n_ports * r_eff];
        for (port_i, &port_k) in modal_port_idx.iter().enumerate() {
            for col_k in 0..r_eff {
                let (pe, ph) =
                    op.port_modal_projections(&basis[col_k], port_k);
                c_e_hat[port_i * r_eff + col_k] = pe;
                c_h_hat[port_i * r_eff + col_k] = ph;
            }
        }

        // Capture impedance closures by snapshotting each port's
        // dispersion characteristics, so the macromodel does not borrow
        // the operator. A real `Z(omega)` for the rectangular TE mode
        // follows `Z(omega) = Z_inf / sqrt(1 - (omega_c/omega)^2)`; for
        // a TEM / Floquet mode it is frequency-independent.
        let port_modes_for_closure: Vec<_> = modal_port_idx
            .iter()
            .map(|&k| ImpedanceProbe::new(op, k))
            .collect();
        let impedances: Box<dyn Fn(Field) -> Vec<Field> + Send + Sync> =
            Box::new(move |omega: Field| {
                port_modes_for_closure
                    .iter()
                    .map(|p| p.impedance(omega))
                    .collect()
            });

        MacroModel {
            a_hat,
            b_hat,
            c_e_hat,
            c_h_hat,
            impedances,
            n_ports,
            r: r_eff,
        }
    }

    /// Evaluate the `N x N` S-matrix at angular frequency `omega`,
    /// row-major (row `i`, column `j` is `S[i, j]`, the response at port
    /// `i` when port `j` is driven alone).
    ///
    /// One small dense `(j*omega*I - A_hat) x = B_hat[:, j]` solve per
    /// driven port: at `r` ~ 100s the per-frequency cost is microseconds
    /// on the CPU, so a broadband sweep is essentially free.
    pub fn evaluate(&self, omega: Field) -> Vec<C64> {
        let r = self.r;
        let n = self.n_ports;
        let z = self.impedances.as_ref()(omega);

        // Form `(j*omega*I - A_hat)` as a complex `r x r` matrix once.
        let j_omega = C64::new(0.0, omega);
        let mut m = vec![C64::new(0.0, 0.0); r * r];
        for i in 0..r {
            for k in 0..r {
                m[i * r + k] = C64::new(-self.a_hat[i * r + k], 0.0);
                if i == k {
                    m[i * r + k] += j_omega;
                }
            }
        }
        // Right-hand side `B_hat` as an `r x N` complex matrix.
        let mut rhs = vec![C64::new(0.0, 0.0); r * n];
        for i in 0..r {
            for j in 0..n {
                rhs[i * n + j] = C64::new(self.b_hat[i * n + j], 0.0);
            }
        }
        // Dense LU solve with partial pivoting. `r` ~ 100s, so an
        // unblocked LU is fine; pulling in a heavyweight LAPACK
        // dependency just for this would be a step backwards.
        dense_lu_solve(&mut m, &mut rhs, r, n);

        // `z^E = C_hat^E x`, `z^H = C_hat^H x` per driven port j; then
        // per port i, split into forward / backward modal amplitudes
        // using its dispersive impedance `z[i]`.
        let mut s = vec![C64::new(0.0, 0.0); n * n];
        for j_drive in 0..n {
            // Per-port modal readout from the reduced state column
            // j_drive: `z^E_i = C_hat^E[i, :] * x_j`,
            // `z^H_i = C_hat^H[i, :] * x_j`. Split into forward /
            // backward modal amplitudes with the port's dispersive
            // impedance. The forward amplitude A_i carries the wave
            // returning toward the source at port i (so A_{j_drive} is
            // the incident at the driven port), and B_i is the wave
            // outgoing into the domain (reflected at j_drive,
            // transmitted elsewhere). `S[i, j] = B_i / A_{j_drive}`.
            let mut amp_b = vec![C64::new(0.0, 0.0); n];
            let mut a_drive = C64::new(0.0, 0.0);
            for i_port in 0..n {
                let mut z_e = C64::new(0.0, 0.0);
                let mut z_h = C64::new(0.0, 0.0);
                for k in 0..r {
                    z_e += self.c_e_hat[i_port * r + k]
                        * rhs[k * n + j_drive];
                    z_h += self.c_h_hat[i_port * r + k]
                        * rhs[k * n + j_drive];
                }
                let zi = C64::new(z[i_port], 0.0);
                amp_b[i_port] = (z_e - zi * z_h) * 0.5;
                if i_port == j_drive {
                    a_drive = (z_e + zi * z_h) * 0.5;
                }
            }
            // Guard against degenerate drive (Z * P_h ~ -P_e gives
            // A_j ~ 0); a real port at a properly resolved frequency
            // has A_j of order unity, so this guard only fires on a
            // misconfigured port.
            if a_drive.norm() > 0.0 {
                for i_port in 0..n {
                    s[i_port * n + j_drive] = amp_b[i_port] / a_drive;
                }
            }
        }
        s
    }
}

// Internals.

/// A self-contained snapshot of one port's modal impedance, captured at
/// macromodel-build time. Calling `impedance(omega)` reproduces what
/// `op.port_impedance(port_idx, omega)` would return, but without
/// holding a borrow on the operator.
struct ImpedanceProbe {
    /// Precomputed cutoff in normalised units; `0` for TEM-like modes.
    cutoff: Field,
    /// Precomputed `Z(omega_0)` at a single reference frequency, used
    /// to fix the "high-frequency" impedance for the rectangular
    /// `TE_mn` form `Z(omega) = Z_0 / sqrt(1 - (omega_c/omega)^2)`. For
    /// TEM / Floquet (cutoff = 0) the impedance is
    /// frequency-independent and this value is what we return at every
    /// omega.
    z_inf: Field,
}

impl ImpedanceProbe {
    fn new(op: &MaxwellOperator, port_idx: usize) -> Self {
        let cutoff = op.port_cutoff(port_idx);
        // Probe at omega = max(1, 10*cutoff): well into the propagating
        // regime for a rectangular port, exact for a TEM / Floquet
        // port.
        let probe_omega = (10.0 * cutoff).max(1.0);
        let z_probe = op.port_impedance(port_idx, probe_omega);
        // Recover Z_inf from Z(omega) = Z_inf / sqrt(1 - (omega_c/omega)^2).
        let z_inf = if cutoff > 0.0 {
            let ratio = cutoff / probe_omega;
            z_probe * (1.0 - ratio * ratio).sqrt()
        } else {
            z_probe
        };
        ImpedanceProbe { cutoff, z_inf }
    }

    fn impedance(&self, omega: Field) -> Field {
        if self.cutoff == 0.0 {
            self.z_inf
        } else if omega.abs() <= self.cutoff {
            // Below cutoff is evanescent: the impedance is imaginary in
            // the physical picture, but the macromodel's split needs a
            // real `Z` and we treat the band of interest as
            // propagating. Return a fallback `Z_inf` to keep the
            // S-matrix finite; the caller should evaluate above
            // cutoff.
            self.z_inf
        } else {
            let ratio = self.cutoff / omega;
            self.z_inf / (1.0 - ratio * ratio).sqrt()
        }
    }
}

/// One pass of block-CGS2: project `w` orthogonal to `basis` with two
/// passes of classical Gram-Schmidt, which recovers MGS-grade
/// orthogonality in a fan-outable shape (each pass's dot products and
/// updates are independent across basis vectors). The dimension is
/// small enough on the macromodel build that the serial form is fast;
/// the rayon parallel path is the one in
/// [`crate::propagator::KrylovWorkspace`].
fn block_cgs2(w: &mut [Field], basis: &[Vec<Field>]) {
    if basis.is_empty() {
        return;
    }
    let n = w.len();
    for _pass in 0..2 {
        let coefs: Vec<Field> = basis.iter().map(|v| dot(v, w)).collect();
        for (v, &c) in basis.iter().zip(&coefs) {
            for k in 0..n {
                w[k] -= c * v[k];
            }
        }
    }
}

fn dot(a: &[Field], b: &[Field]) -> Field {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[Field]) -> Field {
    dot(a, a).sqrt()
}

/// Dense LU with partial pivoting on a complex `r x r` matrix `a`,
/// applied in place to a `r x cols` right-hand side `b` (row-major).
/// `a` is overwritten; on return `b[:, j]` holds `a^-1 * b[:, j]`.
///
/// Hand-rolled because the workspace's only LAPACK-grade dense solver
/// (`faer`) is real and the per-frequency complex solve here is small:
/// at `r` ~ 100s the O(r^3) factor is microseconds and the LU is not
/// the macromodel's hot spot.
fn dense_lu_solve(a: &mut [C64], b: &mut [C64], r: usize, cols: usize) {
    let mut piv = vec![0usize; r];
    for i in 0..r {
        piv[i] = i;
    }
    for k in 0..r {
        // Pivot: largest |a[i, k]| in column k below the diagonal.
        let mut max_row = k;
        let mut max_val = a[k * r + k].norm();
        for i in (k + 1)..r {
            let v = a[i * r + k].norm();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_row != k {
            for j in 0..r {
                a.swap(k * r + j, max_row * r + j);
            }
            for j in 0..cols {
                b.swap(k * cols + j, max_row * cols + j);
            }
            piv.swap(k, max_row);
        }
        // Eliminate below-diagonal entries in column k.
        let akk = a[k * r + k];
        if akk.norm() == 0.0 {
            // Singular within working precision; leave column zero so
            // the solve is well-defined for the non-degenerate
            // columns.
            continue;
        }
        for i in (k + 1)..r {
            let factor = a[i * r + k] / akk;
            a[i * r + k] = factor;
            for j in (k + 1)..r {
                let p = a[k * r + j];
                a[i * r + j] -= factor * p;
            }
            for j in 0..cols {
                let p = b[k * cols + j];
                b[i * cols + j] -= factor * p;
            }
        }
    }
    // Back-substitution per right-hand-side column.
    for j in 0..cols {
        for i in (0..r).rev() {
            let mut sum = b[i * cols + j];
            for k in (i + 1)..r {
                sum -= a[i * r + k] * b[k * cols + j];
            }
            let akk = a[i * r + i];
            b[i * cols + j] = if akk.norm() > 0.0 {
                sum / akk
            } else {
                C64::new(0.0, 0.0)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_gen::structured_box;
    use crate::rhs::{ElemMaterial, MaxwellOperator, PortSpec};
    use crate::waveguide::{PortMode, RectPort};
    use std::f64::consts::PI;

    #[test]
    fn macromodel_matched_two_port_reproduces_s_parameters() {
        // M1 gate: a matched WR-90-style straight two-port guide. The
        // block-Krylov macromodel must reproduce, on the impulse Krylov
        // subspace alone, the matched-guide behaviour: |S_11| small,
        // |S_21| close to 1, and S_21 ~ S_12 by reciprocity (the
        // lossless guide's spatial operator is real skew-symmetric, so
        // its impulse moments inherit the symmetry).

        // Geometry follows the existing `two_port_guide_s_parameters`
        // test in `rhs.rs`: WR-90-style ratio (a x b = 1.0 x 0.5,
        // cutoff at omega = pi for TE_10), and a guide long enough to
        // host several in-band wavelengths. A shorter `lz` lets the
        // impulse Krylov span the round-trip dynamics at a modest `r`;
        // the M2 gate (iris-vs-FD) revisits a longer geometry once
        // accuracy becomes the headline metric.
        let (a_w, b_h, lz) = (1.0, 0.5, 2.0);
        let mesh = structured_box(2, 1, 8, a_w, b_h, lz);
        let on_plane = |zc: f64| -> Vec<usize> {
            mesh.tris
                .iter()
                .enumerate()
                .filter(|(_, t)| {
                    t.iter().all(|&nd| (mesh.nodes[nd][2] - zc).abs() < 1e-9)
                })
                .map(|(i, _)| i)
                .collect()
        };
        let rect = |z0: f64, inward: f64| RectPort {
            origin: [0.0, 0.0, z0],
            u_hat: [1.0, 0.0, 0.0],
            v_hat: [0.0, 1.0, 0.0],
            w_hat: [0.0, 0.0, inward],
            a: a_w,
            b: b_h,
            mode: (1, 0),
        };
        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        let op = MaxwellOperator::new_with_materials_ports(
            &mesh,
            2,
            0.0,
            &vacuum,
            &[
                PortSpec {
                    tris: on_plane(0.0),
                    mode: Some(PortMode::Rect(rect(0.0, 1.0))),
                },
                PortSpec {
                    tris: on_plane(lz),
                    mode: Some(PortMode::Rect(rect(lz, -1.0))),
                },
            ],
        );

        // Krylov dim. The impulse moments need enough powers of A to
        // span the propagating dynamics of the guide: the wave has to
        // physically reach port 1 from port 0 within the
        // impulse-Krylov basis, which (in matvecs of the central-flux
        // DG operator) takes roughly `nz * (DG flux passes per cell)`
        // powers, and a further band's worth of moments to resolve the
        // in-band frequency response. For the 2 x 1 x 8 P = 2 mesh
        // above, `r = 350` clears that: the gates pass with margin
        // (|S_11| <~ 0.15, |S_21| >~ 0.96 across the band). At `r =
        // 300` the off-diagonal coupling has built up but |S_11| is
        // still ~0.3; at `r = 400` the picture is essentially
        // unchanged from `r = 350`. The plan ballpark of `r ~ 60..100`
        // referred to a single in-band resonance; a propagating guide
        // of this length sits in the upper part of the plan's "tens to
        // a few hundred" range. M4 (Chebyshev / eigenfilter) is the
        // route to bring this number down without changing the M1
        // method.
        let model = MacroModel::build(&op, 350);

        assert_eq!(model.n_ports(), 2);
        eprintln!("DIAG macromodel: realised r = {}", model.r());

        // Sweep an above-cutoff band. Cutoff is pi for the WR-90
        // analogue (TE_10 with a = 1), and the original guide test
        // uses [1.35*pi, 1.55*pi]. Eleven points across that band
        // exercise the sweep.
        let omegas: Vec<f64> = (0..11)
            .map(|i| (1.35 + 0.20 * i as f64 / 10.0) * PI)
            .collect();

        let mut max_s11 = 0.0_f64;
        let mut min_s21 = f64::INFINITY;
        let mut max_s21 = 0.0_f64;
        let mut max_reciprocity_err = 0.0_f64;
        for &omega in &omegas {
            let s = model.evaluate(omega);
            let s11 = s[0 * 2 + 0].norm();
            let s21 = s[1 * 2 + 0].norm();
            let s12 = s[0 * 2 + 1].norm();
            let s22 = s[1 * 2 + 1].norm();
            let rec = (s21 - s12).abs();
            eprintln!(
                "DIAG omega/pi={:.3}: |S11|={:.4} |S21|={:.4} \
                 |S12|={:.4} |S22|={:.4}  |S21-S12|={:.4}",
                omega / PI,
                s11,
                s21,
                s12,
                s22,
                rec,
            );
            max_s11 = max_s11.max(s11);
            min_s21 = min_s21.min(s21);
            max_s21 = max_s21.max(s21);
            max_reciprocity_err = max_reciprocity_err.max(rec);
        }

        // Gate thresholds from the plan: generous so the test
        // demonstrates the *method*, not absolute precision (M2's
        // iris-vs-FD case is the precision gate).
        assert!(
            max_s11 < 0.20,
            "S11 not small across the band: max |S11| = {:.3}",
            max_s11,
        );
        assert!(
            min_s21 > 0.80 && max_s21 < 1.20,
            "S21 not near unity across the band: min={:.3}, max={:.3}",
            min_s21,
            max_s21,
        );
        assert!(
            max_reciprocity_err < 0.10,
            "reciprocity violated: max |S21 - S12| = {:.3}",
            max_reciprocity_err,
        );
    }
}
