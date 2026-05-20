//! TD backend performance benchmark — production-plan WP6.3.
//!
//! Reports the three quantities the roadmap calls for: matrix-free `apply`
//! throughput, exponential-propagation cost per step, and how both scale
//! with mesh size. Run in release for meaningful numbers:
//!
//! ```text
//! cargo run --release -p rapidfem-td --example bench
//! ```

use std::time::Instant;

use rapidfem_td::mesh_gen::structured_box;
use rapidfem_td::propagator::KrylovWorkspace;
use rapidfem_td::rhs::MaxwellOperator;

/// Median wall-clock seconds of `reps` runs of `f`.
fn time_median(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut ts: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[reps / 2]
}

fn main() {
    let order = 2;
    println!("rapidfem-td — performance benchmark  (release, order {order})");
    println!("rayon worker threads: {}\n", rayon::current_num_threads());

    // --- apply throughput + sparse-assembly scaling -----------------------
    println!(
        "{:>7} {:>10} {:>13} {:>12} {:>14} {:>9}",
        "cells", "n_dof", "apply [ms]", "Mdof/s", "assemble [ms]", "nnz/row"
    );
    for &c in &[2usize, 3, 4, 5, 6] {
        let mesh = structured_box(c, c, c, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, order, 1.0);
        let n = op.n_dof();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        op.apply(&y); // warm up

        let t_apply = time_median(50, || {
            op.apply(&y);
        });
        let mdofs = n as f64 / t_apply / 1e6;

        let t_asm = time_median(3, || {
            op.assemble_sparse();
        });
        let nnz_per_row = op.assemble_sparse().nnz() as f64 / n as f64;

        println!(
            "{:>7} {:>10} {:>13.3} {:>12.1} {:>14.1} {:>9.0}",
            c * c * c,
            n,
            t_apply * 1e3,
            mdofs,
            t_asm * 1e3,
            nnz_per_row,
        );
    }

    // --- propagation cost: one exponential step --------------------------
    let mesh = structured_box(4, 4, 4, 1.0, 1.0, 1.0);
    let op = MaxwellOperator::new(&mesh, order, 1.0);
    let n = op.n_dof();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).cos()).collect();

    println!("\nexponential propagation  (n_dof = {n}, reused workspace):");
    for &m in &[12usize, 24, 40] {
        let mut ws = KrylovWorkspace::new();
        let mut out = vec![0.0; n];
        ws.expmv_into(|x, ax| op.apply_into(x, ax), &y, 0.02, m, &mut out); // warm
        let t = time_median(20, || {
            ws.expmv_into(|x, ax| op.apply_into(x, ax), &y, 0.02, m, &mut out);
        });
        println!("  krylov_dim {m:>3}:  {:>7.2} ms/step", t * 1e3);
    }
}
