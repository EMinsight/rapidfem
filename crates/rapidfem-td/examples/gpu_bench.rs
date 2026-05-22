//! GPU vs CPU benchmark for the time-domain backend.
//!
//! Three workloads, each timed on the CPU (24-thread rayon) and the GPU:
//!
//! * the explicit LSERK4 transient (homogeneous);
//! * the explicit LSERK4 transient with a driven soft source;
//! * the Krylov exponential propagator (`expmv`).
//!
//! Run across polynomial order 2 and 3 and a range of structured-box mesh
//! sizes, so the scaling shows.
//!
//! ```text
//! cargo run --release -p rapidfem-td --features gpu --example gpu_bench
//! ```

use std::time::Instant;

use rapidfem_td::constants::Field;
use rapidfem_td::explicit::LserkWorkspace;
use rapidfem_td::gpu::{GpuContext, GpuOperator};
use rapidfem_td::mesh_gen::structured_box;
use rapidfem_td::propagator::expmv;
use rapidfem_td::rhs::MaxwellOperator;

const STEPS: usize = 100;
const KRYLOV_DIM: usize = 40;
const SIZES: [usize; 4] = [4, 6, 8, 12];

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

/// Spectral radius of the operator, for a sub-CFL explicit step.
fn spectral_radius(op: &MaxwellOperator, n: usize) -> Field {
    let mut v: Vec<Field> = (0..n).map(|i| (i as Field * 0.05).sin()).collect();
    let mut rho = 1.0;
    for _ in 0..20 {
        let av = op.apply(&v);
        rho = av.iter().map(|x| x * x).sum::<Field>().sqrt();
        let inv = 1.0 / rho;
        for (vi, &a) in v.iter_mut().zip(&av) {
            *vi = a * inv;
        }
    }
    rho
}

fn main() {
    let gpu = match GpuContext::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU available: {e}");
            return;
        }
    };
    println!("GPU vs CPU benchmark — time-domain backend");
    println!("device: {}", gpu.device_name);
    println!("rayon worker threads: {}", rayon::current_num_threads());

    for &order in &[2usize, 3] {
        println!("\n========== order {order} ==========");
        println!(
            "{:>10} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14} {:>14}",
            "n_dof",
            "lserk CPU",
            "lserk GPU",
            "lserk x",
            "driven CPU",
            "driven GPU",
            "driven x",
            "expmv CPU",
            "expmv GPU",
            "expmv x",
        );

        for &c in &SIZES {
            let mesh = structured_box(c, c, c, 1.0, 1.0, 1.0);
            let op = MaxwellOperator::new(&mesh, order, 1.0);
            let n = op.n_dof();
            let y0: Vec<Field> =
                (0..n).map(|i| (i as Field * 0.05).sin()).collect();
            let dt = 1.0 / spectral_radius(&op, n);
            let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
            let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();

            // --- homogeneous LSERK4 transient -------------------------
            let cpu_lserk = time_median(3, || {
                let mut y = y0.clone();
                let mut ws = LserkWorkspace::new();
                for _ in 0..STEPS {
                    ws.step_into(|x, ax| op.apply_into(x, ax), &mut y, dt);
                }
            });
            gop.transient(&gpu, &y0_32, dt as f32, 3).unwrap();
            let gpu_lserk = time_median(5, || {
                gop.transient(&gpu, &y0_32, dt as f32, STEPS).unwrap();
            });

            // --- driven LSERK4 transient ------------------------------
            let sdof = n / 3;
            let src: Vec<Field> =
                (0..STEPS).map(|k| (0.3 * k as Field).sin()).collect();
            let src32: Vec<f32> = src.iter().map(|&v| v as f32).collect();
            let cpu_driven = time_median(3, || {
                let mut y = vec![0.0; n];
                let mut ws = LserkWorkspace::new();
                for &g in &src {
                    ws.step_driven_into(
                        |x, ax| op.apply_into(x, ax),
                        &mut y,
                        dt,
                        sdof,
                        g,
                    );
                }
            });
            let zero32 = vec![0.0_f32; n];
            gop.transient_driven(&gpu, &zero32, dt as f32, sdof, &src32[..3])
                .unwrap();
            let gpu_driven = time_median(5, || {
                gop.transient_driven(
                    &gpu, &zero32, dt as f32, sdof, &src32,
                )
                .unwrap();
            });

            // --- Krylov exponential propagator ------------------------
            let t_exp = 0.5 * dt * STEPS as Field;
            let cpu_expmv = time_median(5, || {
                expmv(|x| op.apply(x), &y0, t_exp, KRYLOV_DIM);
            });
            gop.expmv(&gpu, &y0, t_exp, KRYLOV_DIM).unwrap();
            let gpu_expmv = time_median(8, || {
                gop.expmv(&gpu, &y0, t_exp, KRYLOV_DIM).unwrap();
            });

            let ms = |s: f64| s * 1e3;
            println!(
                "{:>10} {:>13.1} {:>13.1} {:>13.2}x {:>13.1} {:>13.1} \
                 {:>13.2}x {:>13.1} {:>13.1} {:>13.2}x",
                n,
                ms(cpu_lserk),
                ms(gpu_lserk),
                cpu_lserk / gpu_lserk,
                ms(cpu_driven),
                ms(gpu_driven),
                cpu_driven / gpu_driven,
                ms(cpu_expmv),
                ms(gpu_expmv),
                cpu_expmv / gpu_expmv,
            );
        }
    }
}
