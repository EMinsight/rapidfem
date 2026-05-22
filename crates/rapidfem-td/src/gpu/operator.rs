//! GPU representation of the DG Maxwell operator.
//!
//! [`GpuOperator`] uploads the CPU operator's data (reference matrices,
//! geometric factors, materials, face topology) to device buffers once,
//! then evaluates `dy/dt = A.y` with the `apply` kernel. The field state
//! buffers are owned and reused, so a later step loop keeps the state
//! resident on the device.
//!
//! Phase P1: non-dispersive operators (the `[E,H]` block). The dispersive
//! polarisation block is a later phase.

use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::Buffer;
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_float, cl_int};

use super::GpuContext;
use crate::constants::{Field, LSERK4_A, LSERK4_B, LSERK4_STAGES};
use crate::rhs::MaxwellOperator;

/// Apply-kernel source, with `NP` / `NFP` / `COLS` prepended at build time.
const APPLY_SRC: &str = include_str!("apply.cl");

/// LSERK4 stage-update kernel source.
const LSERK_SRC: &str = include_str!("lserk.cl");

/// Work-group size for the element loop.
const WORK_GROUP: usize = 64;

/// Work-group size for the flat per-DOF LSERK4 stage loop.
const DOF_WORK_GROUP: usize = 256;

/// The DG Maxwell operator resident on the GPU.
pub struct GpuOperator {
    n_elem: usize,
    /// State-vector length, `6*Np*n_elem` (non-dispersive).
    n_dof: usize,
    flux_alpha: f32,
    _program: Program,
    kernel: Kernel,
    // Operator data, uploaded once.
    diff_r: Buffer<cl_float>,
    diff_s: Buffer<cl_float>,
    diff_t: Buffer<cl_float>,
    lift: Buffer<cl_float>,
    face_nodes: Buffer<cl_int>,
    jinv: Buffer<cl_float>,
    inv_eps: Buffer<cl_float>,
    inv_mu: Buffer<cl_float>,
    sigma_eps: Buffer<cl_float>,
    sigma_mu: Buffer<cl_float>,
    face_normal: Buffer<cl_float>,
    face_fscale: Buffer<cl_float>,
    face_neighbor: Buffer<cl_int>,
    face_nbr_local: Buffer<cl_int>,
    face_port: Buffer<cl_int>,
    face_perm: Buffer<cl_int>,
    // State buffers, reused across calls.
    y: Buffer<cl_float>,
    dy: Buffer<cl_float>,
    /// LSERK4 residual register, device-resident across a transient.
    p: Buffer<cl_float>,
    _lserk_program: Program,
    lserk_kernel: Kernel,
}

/// `f64` slice to an `f32` vector.
fn f32v(s: &[Field]) -> Vec<f32> {
    s.iter().map(|&x| x as f32).collect()
}

impl GpuOperator {
    /// Upload a non-dispersive CPU operator to the device.
    pub fn new(
        gpu: &GpuContext,
        op: &MaxwellOperator,
    ) -> Result<Self, String> {
        assert_eq!(
            op.n_dispersive(),
            0,
            "GpuOperator: dispersive materials are a later phase",
        );
        let np = op.re.n_nodes;
        let nfp = op.re.n_face_nodes;
        let cols = 4 * nfp;
        let n_elem = op.n_elem;
        let n_dof = 6 * np * n_elem;

        // Reference element.
        let diff_r = gpu.upload(&f32v(&op.re.diff_r))?;
        let diff_s = gpu.upload(&f32v(&op.re.diff_s))?;
        let diff_t = gpu.upload(&f32v(&op.re.diff_t))?;
        let lift = gpu.upload(&f32v(&op.re.lift))?;
        let mut fn_flat = Vec::with_capacity(4 * nfp);
        for f in 0..4 {
            for m in 0..nfp {
                fn_flat.push(op.re.face_nodes[f][m] as i32);
            }
        }
        let face_nodes = gpu.upload_i32(&fn_flat)?;

        // Per-element geometric factors and materials.
        let mut jinv = Vec::with_capacity(n_elem * 9);
        for g in &op.geom {
            for i in 0..3 {
                for k in 0..3 {
                    jinv.push(g.jacobian_inv[i][k] as f32);
                }
            }
        }
        let jinv = gpu.upload(&jinv)?;
        let flat3 = |v: &[[Field; 3]]| -> Vec<f32> {
            v.iter()
                .flat_map(|a| [a[0] as f32, a[1] as f32, a[2] as f32])
                .collect()
        };
        let inv_eps = gpu.upload(&flat3(&op.inv_eps))?;
        let inv_mu = gpu.upload(&flat3(&op.inv_mu))?;
        let sigma_eps = gpu.upload(&flat3(&op.sigma_eps))?;
        let sigma_mu = gpu.upload(&flat3(&op.sigma_mu))?;

        // Face topology, flattened over `faces[e*4 + f]`.
        let nf = 4 * n_elem;
        let mut normal = Vec::with_capacity(nf * 3);
        let mut fscale = Vec::with_capacity(nf);
        let mut neighbor = Vec::with_capacity(nf);
        let mut nbr_local = Vec::with_capacity(nf);
        let mut port = Vec::with_capacity(nf);
        let mut perm = Vec::with_capacity(nf * nfp);
        for fi in &op.faces {
            normal.extend([
                fi.normal[0] as f32,
                fi.normal[1] as f32,
                fi.normal[2] as f32,
            ]);
            fscale.push(fi.fscale as f32);
            neighbor.push(if fi.neighbor == usize::MAX {
                -1
            } else {
                fi.neighbor as i32
            });
            nbr_local.push(fi.neighbor_local_face as i32);
            port.push(if fi.port == usize::MAX {
                -1
            } else {
                fi.port as i32
            });
            for m in 0..nfp {
                // `perm` is empty on a boundary face; the kernel only reads
                // it on the neighbour branch, so the pad value is inert.
                perm.push(fi.perm.get(m).map_or(0, |&p| p as i32));
            }
        }
        let face_normal = gpu.upload(&normal)?;
        let face_fscale = gpu.upload(&fscale)?;
        let face_neighbor = gpu.upload_i32(&neighbor)?;
        let face_nbr_local = gpu.upload_i32(&nbr_local)?;
        let face_port = gpu.upload_i32(&port)?;
        let face_perm = gpu.upload_i32(&perm)?;

        // Build the kernel with the element dimensions baked in.
        let src = format!(
            "#define NP {np}\n#define NFP {nfp}\n#define COLS {cols}\n\
             {APPLY_SRC}"
        );
        let program = gpu.build_program(&src)?;
        let kernel = Kernel::create(&program, "apply")
            .map_err(|e| format!("kernel create failed: {e}"))?;

        let lserk_program = gpu.build_program(LSERK_SRC)?;
        let lserk_kernel = Kernel::create(&lserk_program, "lserk_stage")
            .map_err(|e| format!("lserk kernel create failed: {e}"))?;

        let y = gpu.alloc(n_dof)?;
        let dy = gpu.alloc(n_dof)?;
        let p = gpu.alloc(n_dof)?;

        Ok(GpuOperator {
            n_elem,
            n_dof,
            flux_alpha: op.flux_alpha as f32,
            _program: program,
            kernel,
            diff_r,
            diff_s,
            diff_t,
            lift,
            face_nodes,
            jinv,
            inv_eps,
            inv_mu,
            sigma_eps,
            sigma_mu,
            face_normal,
            face_fscale,
            face_neighbor,
            face_nbr_local,
            face_port,
            face_perm,
            y,
            dy,
            p,
            _lserk_program: lserk_program,
            lserk_kernel,
        })
    }

    /// State-vector length the operator expects.
    pub fn n_dof(&self) -> usize {
        self.n_dof
    }

    /// Enqueue the `apply` kernel: `dy = A.y` on the resident state. No
    /// host transfer; the in-order queue serialises it with later work.
    fn enqueue_apply(&self, gpu: &GpuContext) -> Result<(), String> {
        let global = self.n_elem.div_ceil(WORK_GROUP) * WORK_GROUP;
        let n_elem = self.n_elem as cl_int;
        unsafe {
            ExecuteKernel::new(&self.kernel)
                .set_arg(&self.y)
                .set_arg(&self.dy)
                .set_arg(&self.diff_r)
                .set_arg(&self.diff_s)
                .set_arg(&self.diff_t)
                .set_arg(&self.lift)
                .set_arg(&self.face_nodes)
                .set_arg(&self.jinv)
                .set_arg(&self.inv_eps)
                .set_arg(&self.inv_mu)
                .set_arg(&self.sigma_eps)
                .set_arg(&self.sigma_mu)
                .set_arg(&self.face_normal)
                .set_arg(&self.face_fscale)
                .set_arg(&self.face_neighbor)
                .set_arg(&self.face_nbr_local)
                .set_arg(&self.face_port)
                .set_arg(&self.face_perm)
                .set_arg(&self.flux_alpha)
                .set_arg(&n_elem)
                .set_global_work_size(global)
                .set_local_work_size(WORK_GROUP)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("apply kernel launch failed: {e}"))?;
        Ok(())
    }

    /// Enqueue one LSERK4 stage: `p = a*p + dt*k; y += b*p`, with `k` the
    /// `dy` written by the preceding [`enqueue_apply`](Self::enqueue_apply).
    fn enqueue_lserk(
        &self,
        gpu: &GpuContext,
        a: f32,
        b: f32,
        dt: f32,
    ) -> Result<(), String> {
        let global = self.n_dof.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
        let n = self.n_dof as cl_int;
        unsafe {
            ExecuteKernel::new(&self.lserk_kernel)
                .set_arg(&self.p)
                .set_arg(&self.dy)
                .set_arg(&self.y)
                .set_arg(&a)
                .set_arg(&b)
                .set_arg(&dt)
                .set_arg(&n)
                .set_global_work_size(global)
                .set_local_work_size(DOF_WORK_GROUP)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("lserk kernel launch failed: {e}"))?;
        Ok(())
    }

    /// Evaluate `dy/dt = A.y`: upload `y_host`, run the apply kernel,
    /// download the result. The single-shot form, for validation.
    pub fn apply(
        &mut self,
        gpu: &GpuContext,
        y_host: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y_host.len(), self.n_dof, "state length mismatch");
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y_host, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        self.enqueue_apply(gpu)?;
        // The blocking download serialises behind the kernel.
        gpu.download(&self.dy, self.n_dof)
    }

    /// Propagate `y0` for `steps` LSERK4 steps of size `dt`, fully on the
    /// device: the state stays resident, only `y0` (up) and the final
    /// state (down) cross the bus.
    pub fn transient(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y0.len(), self.n_dof, "state length mismatch");
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y0, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        // Zero the residual register once; stage 0 (a = 0) keeps it reset
        // every step thereafter.
        let zeros = vec![0.0_f32; self.n_dof];
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.p, CL_BLOCKING, 0, &zeros, &[])
        }
        .map_err(|e| format!("register init failed: {e}"))?;

        for _ in 0..steps {
            for stage in 0..LSERK4_STAGES {
                self.enqueue_apply(gpu)?;
                self.enqueue_lserk(
                    gpu,
                    LSERK4_A[stage] as f32,
                    LSERK4_B[stage] as f32,
                    dt,
                )?;
            }
        }
        gpu.queue()
            .finish()
            .map_err(|e| format!("transient sync failed: {e}"))?;
        gpu.download(&self.y, self.n_dof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GPU_REL_TOL;
    use crate::mesh_gen::structured_box;

    /// Relative L2 error of the GPU result against the CPU f64 reference.
    fn rel_l2(gpu: &[f32], cpu: &[Field]) -> f64 {
        let err: f64 = cpu
            .iter()
            .zip(gpu)
            .map(|(&c, &g)| (c as f64 - g as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 =
            cpu.iter().map(|&c| (c as f64).powi(2)).sum::<f64>().sqrt();
        err / scale
    }

    #[test]
    fn gpu_apply_matches_cpu() {
        // P1 gate: the GPU apply matches the CPU f64 apply within the
        // mixed-precision budget GPU_REL_TOL, for a vacuum cavity (trivial
        // materials) and a dielectric fill (inv_eps != 1).
        use crate::rhs::ElemMaterial;

        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let mesh = structured_box(3, 3, 3, 1.0, 1.0, 1.0);
        let vacuum = MaxwellOperator::new(&mesh, 2, 1.0);
        let dielectric = MaxwellOperator::new_with_materials(
            &mesh,
            2,
            1.0,
            &vec![ElemMaterial::isotropic(4.0, 1.0, 0.0); mesh.n_tets()],
        );

        for (label, op) in
            [("vacuum", &vacuum), ("dielectric er=4", &dielectric)]
        {
            let n = op.n_dof();
            let y: Vec<Field> =
                (0..n).map(|i| (0.3 + i as Field * 0.017).sin()).collect();
            let cpu_dy = op.apply(&y);

            let mut gop = GpuOperator::new(&gpu, op).expect("GpuOperator");
            assert_eq!(gop.n_dof(), n);
            let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
            let gpu_dy = gop.apply(&gpu, &y32).expect("gpu apply");

            let rel = rel_l2(&gpu_dy, &cpu_dy);
            eprintln!(
                "GPU apply vs CPU f64 [{label}]: rel L2 = {rel:.3e} \
                 (GPU_REL_TOL {GPU_REL_TOL:.1e})"
            );
            assert!(
                rel < GPU_REL_TOL,
                "GPU apply [{label}] rel.err {rel:.3e} exceeds GPU_REL_TOL",
            );
        }
    }

    #[test]
    fn gpu_transient_matches_cpu() {
        // P2 gate: the device-resident GPU LSERK4 transient matches the
        // CPU LSERK4 transient within GPU_REL_TOL.
        use crate::explicit::LserkWorkspace;

        let gpu = match GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
        };
        let mesh = structured_box(3, 3, 3, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let y0: Vec<Field> =
            (0..n).map(|i| (0.2 + i as Field * 0.011).sin()).collect();

        // Spectral radius by power iteration, for a sub-CFL step.
        let mut v = y0.clone();
        let mut rho = 1.0;
        for _ in 0..30 {
            let av = op.apply(&v);
            rho = av.iter().map(|x| x * x).sum::<Field>().sqrt();
            let inv = 1.0 / rho;
            for (vi, &a) in v.iter_mut().zip(&av) {
                *vi = a * inv;
            }
        }
        let dt = 1.0 / rho;
        let steps = 200;

        // CPU reference.
        let mut y_cpu = y0.clone();
        let mut ws = LserkWorkspace::new();
        for _ in 0..steps {
            ws.step_into(|x, ax| op.apply_into(x, ax), &mut y_cpu, dt);
        }

        // GPU, device-resident.
        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        let y_gpu = gop
            .transient(&gpu, &y0_32, dt as f32, steps)
            .expect("transient");

        let rel = rel_l2(&y_gpu, &y_cpu);
        eprintln!(
            "GPU transient vs CPU [{steps} steps]: rel L2 = {rel:.3e} \
             (GPU_REL_TOL {GPU_REL_TOL:.1e})"
        );
        assert!(
            rel < GPU_REL_TOL,
            "GPU transient rel.err {rel:.3e} exceeds GPU_REL_TOL",
        );
    }
}
