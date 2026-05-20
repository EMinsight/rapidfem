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
}
