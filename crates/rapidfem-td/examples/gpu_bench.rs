//! GPU vs CPU benchmark for the explicit LSERK4 transient.
//!
//! Both backends run the same number of LSERK4 steps on the same
//! structured-box operator; the GPU keeps the state device-resident, so
//! only the initial and final states cross the bus.
//!
//! ```text
//! cargo run --release -p rapidfem-td --features gpu --example gpu_bench
//! ```

use std::time::Instant;

use rapidfem_td::constants::Field;
use rapidfem_td::explicit::LserkWorkspace;
use rapidfem_td::gpu::{GpuContext, GpuOperator};
use rapidfem_td::mesh_gen::structured_box;
use rapidfem_td::rhs::MaxwellOperator;

const STEPS: usize = 100;

fn main() {
    let gpu = match GpuContext::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU available: {e}");
            return;
        }
    };
    println!("GPU vs CPU LSERK4 transient ({STEPS} steps, order 2)");
    println!("device: {}\n", gpu.device_name);
    println!(
        "{:>7} {:>10} {:>12} {:>12} {:>9}",
        "cells", "n_dof", "CPU [ms]", "GPU [ms]", "speedup"
    );

    for &c in &[4usize, 6, 8, 12] {
        let mesh = structured_box(c, c, c, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let y0: Vec<Field> =
            (0..n).map(|i| (i as Field * 0.05).sin()).collect();

        // Spectral radius by power iteration, for a sub-CFL step.
        let mut v = y0.clone();
        let mut rho = 1.0;
        for _ in 0..20 {
            let av = op.apply(&v);
            rho = av.iter().map(|x| x * x).sum::<Field>().sqrt();
            let inv = 1.0 / rho;
            for (vi, &a) in v.iter_mut().zip(&av) {
                *vi = a * inv;
            }
        }
        let dt = 1.0 / rho;

        // CPU LSERK4.
        let mut y = y0.clone();
        let mut ws = LserkWorkspace::new();
        for _ in 0..3 {
            ws.step_into(|x, ax| op.apply_into(x, ax), &mut y, dt);
        }
        y = y0.clone();
        let t = Instant::now();
        for _ in 0..STEPS {
            ws.step_into(|x, ax| op.apply_into(x, ax), &mut y, dt);
        }
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

        // GPU LSERK4.
        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        gop.transient(&gpu, &y0_32, dt as f32, 3).expect("warm-up");
        let t = Instant::now();
        gop.transient(&gpu, &y0_32, dt as f32, STEPS).expect("gpu transient");
        let gpu_ms = t.elapsed().as_secs_f64() * 1e3;

        println!(
            "{:>7} {:>10} {:>12.1} {:>12.1} {:>8.2}x",
            c * c * c,
            n,
            cpu_ms,
            gpu_ms,
            cpu_ms / gpu_ms,
        );
    }
}
