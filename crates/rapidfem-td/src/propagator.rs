//! Krylov-subspace exponential propagator.
//!
//! The semi-discrete DG system is linear, `dy/dt = A·y`, so a step of size `h`
//! is exactly `y ← exp(h·A)·y`. `A` is large and only available as a
//! matrix-free `apply`, so the action `exp(h·A)·v` is formed in an `m`-step
//! Krylov subspace: Arnoldi gives `A·V_m ≈ V_m·H_m`, and
//! `exp(h·A)·v ≈ ‖v‖·V_m·exp(h·H_m)·e₁` with the small `exp(h·H_m)` dense.

/// Dense matrix exponential of an `n×n` row-major matrix, via
/// scaling-and-squaring with a Taylor core.
pub fn expm(a: &[f64], n: usize) -> Vec<f64> {
    // Infinity norm.
    let mut norm = 0.0_f64;
    for i in 0..n {
        let row: f64 = (0..n).map(|j| a[i * n + j].abs()).sum();
        norm = norm.max(row);
    }
    // Scale so ‖B‖ ≤ 1/2.
    let s: u32 = if norm > 0.5 {
        (norm.log2().ceil() as i64 + 1).max(0) as u32
    } else {
        0
    };
    let scale = 2.0_f64.powi(s as i32);
    let b: Vec<f64> = a.iter().map(|x| x / scale).collect();

    // exp(B) = Σ Bᵏ/k!  (≈18 terms suffice for ‖B‖ ≤ 1/2).
    let mut result = identity(n);
    let mut term = identity(n);
    for k in 1..=18 {
        term = matmul(&term, &b, n);
        let inv = 1.0 / k as f64;
        for x in term.iter_mut() {
            *x *= inv;
        }
        for (r, t) in result.iter_mut().zip(&term) {
            *r += *t;
        }
    }
    // Square s times.
    for _ in 0..s {
        result = matmul(&result, &result, n);
    }
    result
}

/// Matrix-free action `exp(t·A)·v`, via an `m`-step Krylov projection.
///
/// `matvec` computes `A·x`. `m` is the Krylov dimension; Arnoldi stops early
/// on a lucky breakdown.
pub fn expmv<F>(matvec: F, v: &[f64], t: f64, m: usize) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = v.len();
    let beta = norm2(v);
    if beta == 0.0 {
        return vec![0.0; n];
    }

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
    basis.push(v.iter().map(|x| x / beta).collect());
    let mut h = vec![0.0; m * m];
    let mut dim = m;

    for j in 0..m {
        let mut w = matvec(&basis[j]);
        for i in 0..=j {
            let hij = dot(&w, &basis[i]);
            h[i * m + j] = hij;
            for k in 0..n {
                w[k] -= hij * basis[i][k];
            }
        }
        let hnext = norm2(&w);
        if hnext < 1e-12 {
            dim = j + 1;
            break;
        }
        if j + 1 < m {
            h[(j + 1) * m + j] = hnext;
            basis.push(w.iter().map(|x| x / hnext).collect());
        }
    }

    // exp(t·H) on the dim×dim leading block.
    let mut th = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            th[i * dim + j] = t * h[i * m + j];
        }
    }
    let e = expm(&th, dim);

    // result = β · Σ_i basis[i] · e[i,0]
    let mut out = vec![0.0; n];
    for i in 0..dim {
        let c = beta * e[i * dim];
        for k in 0..n {
            out[k] += c * basis[i][k];
        }
    }
    out
}

/// One exponential-time-differencing step of `dy/dt = A·y + b`, with the
/// source `b` held constant across the step:
/// `y ← exp(h·A)·y + h·φ₁(h·A)·b`.
///
/// Uses the augmented-matrix identity
/// `exp(h·[[A, b],[0, 0]])·[y; 1] = [exp(hA)y + h·φ₁(hA)b ; 1]`, so the
/// Krylov `expmv` handles the φ-function with no extra machinery — the
/// homogeneous part is exact at any `h`.
pub fn etd_step<F>(matvec: F, y: &[f64], b: &[f64], h: f64, m: usize) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = y.len();
    let mut z = Vec::with_capacity(n + 1);
    z.extend_from_slice(y);
    z.push(1.0);
    let aug = |zz: &[f64]| -> Vec<f64> {
        let xi = zz[n];
        let mut out = matvec(&zz[..n]);
        for (o, bk) in out.iter_mut().zip(b) {
            *o += xi * bk;
        }
        out.push(0.0);
        out
    };
    let r = expmv(aug, &z, h, m);
    r[..n].to_vec()
}

/// Second-order ETD step of `dy/dt = A·y + b(t)`, with the source taken
/// **linear** across the step from `b0 = b(tₙ)` to `b1 = b(tₙ+h)`:
/// `y ← exp(hA)y + h·φ₁(hA)·b0 + h²·φ₂(hA)·d`,  `d = (b1-b0)/h`.
///
/// Uses a two-row augmentation — `exp(h·[[A, d, b0],[0,0,1],[0,0,0]])` applied
/// to `[y; 0; 1]` — so the Krylov `expmv` produces both φ-functions with no
/// extra machinery. Exact when `b` is linear; second-order otherwise.
pub fn etd_step2<F>(
    matvec: F,
    y: &[f64],
    b0: &[f64],
    b1: &[f64],
    h: f64,
    m: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = y.len();
    let d: Vec<f64> =
        b0.iter().zip(b1).map(|(a, b)| (b - a) / h).collect();
    // Augmented state [y; p; q] with p(0)=0, q(0)=1 ⇒ q≡1, p≡t.
    let mut z = Vec::with_capacity(n + 2);
    z.extend_from_slice(y);
    z.push(0.0);
    z.push(1.0);
    let aug = |zz: &[f64]| -> Vec<f64> {
        let (p, q) = (zz[n], zz[n + 1]);
        let mut out = matvec(&zz[..n]);
        for k in 0..n {
            out[k] += d[k] * p + b0[k] * q;
        }
        out.push(q);
        out.push(0.0);
        out
    };
    let r = expmv(aug, &z, h, m);
    r[..n].to_vec()
}

/// Matrix-free `exp(t·A)·v` with an **automatically chosen** Krylov dimension.
///
/// The subspace grows one vector at a time; after each step the Arnoldi
/// a-posteriori error estimate `β·h_{m+1,m}·|(exp(t·H_m))_{m,1}|` is checked,
/// and the process stops once it drops below `tol` (or on a lucky breakdown,
/// or at `max_dim`). Returns the result and the dimension actually used.
pub fn expmv_adaptive<F>(
    matvec: F,
    v: &[f64],
    t: f64,
    tol: f64,
    max_dim: usize,
) -> (Vec<f64>, usize)
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = v.len();
    let beta = norm2(v);
    if beta == 0.0 {
        return (vec![0.0; n], 0);
    }
    let md = max_dim.max(1);
    let mut basis: Vec<Vec<f64>> =
        vec![v.iter().map(|x| x / beta).collect()];
    let mut h = vec![0.0; md * md];

    for j in 0..md {
        let mut w = matvec(&basis[j]);
        for i in 0..=j {
            let hij = dot(&w, &basis[i]);
            h[i * md + j] = hij;
            for k in 0..n {
                w[k] -= hij * basis[i][k];
            }
        }
        let hn = norm2(&w);
        let m = j + 1;

        // exp(t·H_m) on the m×m leading block.
        let mut th = vec![0.0; m * m];
        for a in 0..m {
            for b in 0..m {
                th[a * m + b] = t * h[a * md + b];
            }
        }
        let e = expm(&th, m);
        let estimate = beta * hn * e[(m - 1) * m].abs();

        if estimate < tol || hn < 1e-12 || m == md {
            let mut out = vec![0.0; n];
            for i in 0..m {
                let c = beta * e[i * m];
                for k in 0..n {
                    out[k] += c * basis[i][k];
                }
            }
            return (out, m);
        }

        h[(j + 1) * md + j] = hn;
        basis.push(w.iter().map(|x| x / hn).collect());
    }
    unreachable!("loop returns at m == md")
}

fn identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expm_of_zero_is_identity() {
        let e = expm(&vec![0.0; 9], 3);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((e[i * 3 + j] - want).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn expm_of_rotation_generator() {
        // exp([[0,-θ],[θ,0]]) = [[cosθ,-sinθ],[sinθ,cosθ]].
        let theta = 0.7;
        let e = expm(&[0.0, -theta, theta, 0.0], 2);
        assert!((e[0] - theta.cos()).abs() < 1e-12);
        assert!((e[1] + theta.sin()).abs() < 1e-12);
        assert!((e[2] - theta.sin()).abs() < 1e-12);
        assert!((e[3] - theta.cos()).abs() < 1e-12);
    }

    #[test]
    fn expm_of_diagonal() {
        let e = expm(&[1.5, 0.0, 0.0, -2.0], 2);
        assert!((e[0] - 1.5_f64.exp()).abs() < 1e-11);
        assert!((e[3] - (-2.0_f64).exp()).abs() < 1e-11);
        assert!(e[1].abs() < 1e-13 && e[2].abs() < 1e-13);
    }

    #[test]
    fn expmv_matches_dense_exponential() {
        // Matrix-free Krylov action vs the dense reference exp(tA)·v on the
        // DG Maxwell operator of a small cavity.
        use crate::mesh_gen::structured_box;
        use crate::rhs::MaxwellOperator;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let t = 0.05;

        // Dense reference.
        let mut ta = a.clone();
        for x in ta.iter_mut() {
            *x *= t;
        }
        let dense_exp = expm(&ta, n);

        // A deterministic test vector.
        let v: Vec<f64> =
            (0..n).map(|i| (0.3 + i as f64 * 0.017).sin()).collect();
        let mut want = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                want[i] += dense_exp[i * n + j] * v[j];
            }
        }

        let got = expmv(|x| op.apply(x), &v, t, 60);
        let err: f64 = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 =
            want.iter().map(|w| w * w).sum::<f64>().sqrt();
        assert!(err < 1e-8 * scale, "Krylov vs dense: err {err}, scale {scale}");
    }

    #[test]
    fn etd_step_matches_analytic_linear_ode() {
        // dy/dt = A·y + b,  A = [[0,-ω],[ω,0]],  b constant.
        // Exact: y(h) = exp(hA)·(y₀ + A⁻¹b) - A⁻¹b.
        let omega = 1.3;
        let a = [0.0, -omega, omega, 0.0];
        let matvec = |x: &[f64]| {
            vec![a[0] * x[0] + a[1] * x[1], a[2] * x[0] + a[3] * x[1]]
        };
        let b = [0.4, -0.7];
        let y0 = [1.0, 0.5];
        let h = 0.6;

        let ainv_b = [b[1] / omega, -b[0] / omega];
        let (c, s) = ((omega * h).cos(), (omega * h).sin());
        let shifted = [y0[0] + ainv_b[0], y0[1] + ainv_b[1]];
        let want = [
            c * shifted[0] - s * shifted[1] - ainv_b[0],
            s * shifted[0] + c * shifted[1] - ainv_b[1],
        ];
        // The augmented system is 3-dimensional; m ≥ 3 makes Krylov exact.
        let got = etd_step(matvec, &y0, &b, h, 8);
        assert!((got[0] - want[0]).abs() < 1e-12, "{got:?} vs {want:?}");
        assert!((got[1] - want[1]).abs() < 1e-12, "{got:?} vs {want:?}");
    }

    #[test]
    fn etd_step2_is_second_order_and_exact_for_linear_sources() {
        // A = 0: with no dynamics, etd_step2 integrates b(t) = b0 + d·t
        // exactly — the trapezoidal value y0 + h·(b0+b1)/2.
        let zero = |x: &[f64]| vec![0.0; x.len()];
        let y0 = [1.0, -2.0];
        let (b0, b1) = ([0.5, 1.5], [2.5, -0.5]);
        let h = 0.4;
        let got = etd_step2(zero, &y0, &b0, &b1, h, 6);
        for k in 0..2 {
            let want = y0[k] + h * 0.5 * (b0[k] + b1[k]);
            assert!((got[k] - want).abs() < 1e-12, "A=0 trapezoid {got:?}");
        }

        // Second order: the error quarters when the step halves.
        let omega = 1.7;
        let matvec = |x: &[f64]| vec![-omega * x[1], omega * x[0]];
        let src = |t: f64| vec![(2.3 * t).cos(), 0.4 * t];
        let t_end = 1.2;
        let integrate = |nsteps: usize| -> Vec<f64> {
            let h = t_end / nsteps as f64;
            let mut y = vec![1.0, 0.0];
            for s in 0..nsteps {
                let t = s as f64 * h;
                y = etd_step2(matvec, &y, &src(t), &src(t + h), h, 6);
            }
            y
        };
        let reference = integrate(2048);
        let err = |n: usize| -> f64 {
            let y = integrate(n);
            ((y[0] - reference[0]).powi(2)
                + (y[1] - reference[1]).powi(2))
            .sqrt()
        };
        let rate = err(16) / err(32);
        assert!(rate > 3.5, "etd_step2 not ~2nd order — error ratio {rate:.2}");
    }

    #[test]
    fn central_flux_propagation_conserves_energy() {
        // P4.4: a central-flux transient run conserves the discrete field
        // energy yᵀM̃y exactly (up to the Krylov tolerance) over many steps.
        use crate::mesh_gen::structured_box;
        use crate::rhs::MaxwellOperator;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 0.0);
        let n = op.n_dof();
        let mm = op.assemble_energy_mass();
        let energy = |y: &[f64]| -> f64 {
            let mut e = 0.0;
            for i in 0..n {
                for j in 0..n {
                    e += y[i] * mm[i * n + j] * y[j];
                }
            }
            e
        };
        let mut y: Vec<f64> =
            (0..n).map(|i| (0.2 + i as f64 * 0.013).sin()).collect();
        let e0 = energy(&y);
        for _ in 0..30 {
            y = expmv(|x| op.apply(x), &y, 0.02, 40);
        }
        let drift = ((energy(&y) - e0) / e0).abs();
        assert!(drift < 1e-7, "energy drift {drift:e}");
    }

    #[test]
    fn adaptive_krylov_dimension_meets_tolerance() {
        // expmv_adaptive picks the Krylov dimension itself; the result must
        // match the dense reference, and the chosen dimension stay modest.
        use crate::mesh_gen::structured_box;
        use crate::rhs::MaxwellOperator;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let t = 0.05;

        let a = op.assemble_dense();
        let ta: Vec<f64> = a.iter().map(|x| x * t).collect();
        let dense_exp = expm(&ta, n);
        let v: Vec<f64> =
            (0..n).map(|i| (0.3 + i as f64 * 0.017).sin()).collect();
        let mut want = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                want[i] += dense_exp[i * n + j] * v[j];
            }
        }

        let (got, dim) = expmv_adaptive(|x| op.apply(x), &v, t, 1e-9, 200);
        let err: f64 = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = want.iter().map(|w| w * w).sum::<f64>().sqrt();
        assert!(err < 1e-7 * scale, "adaptive expmv err {}", err / scale);
        assert!(dim > 0 && dim < n, "chosen Krylov dim {dim}");
    }
}
