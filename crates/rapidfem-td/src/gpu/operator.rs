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
use opencl3::types::{CL_BLOCKING, cl_double, cl_float, cl_int};

use super::GpuContext;
use crate::constants::{
    Field, KRYLOV_CHUNK, LSERK4_A, LSERK4_B, LSERK4_STAGES,
};
use crate::propagator::expm;
use crate::rhs::MaxwellOperator;

/// Apply-kernel source, with `NP` / `NFP` / `COLS` prepended at build time.
const APPLY_SRC: &str = include_str!("apply.cl");

/// LSERK4 stage-update kernel source.
const LSERK_SRC: &str = include_str!("lserk.cl");

/// Krylov exponential-propagator kernel source.
const EXPMV_SRC: &str = include_str!("expmv.cl");

/// Target work-group size for the `apply` kernel. A work-group processes
/// one block of `EPG = APPLY_TARGET_WG / NP` elements, its work-items the
/// `NP` DG nodes of each — so the group holds `EPG * NP` work-items, close
/// to this target.
const APPLY_TARGET_WG: usize = 128;

/// Work-group size for the flat per-DOF LSERK4 stage loop.
const DOF_WORK_GROUP: usize = 256;

/// Work-group size for the norm reduction (a power of two for the
/// local-memory tree reduction).
const NORM_WORK_GROUP: usize = 256;

/// The DG Maxwell operator resident on the GPU.
pub struct GpuOperator {
    n_elem: usize,
    /// State-vector length, `6*Np*n_elem` (non-dispersive).
    n_dof: usize,
    /// Elements per `apply` work-group.
    apply_epg: usize,
    /// `apply` work-group size, `apply_epg * Np`.
    apply_wg: usize,
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
    source_kernel: Kernel,
    // Krylov exponential propagator (P3).
    _expmv_program: Program,
    k_cast_d2f: Kernel,
    k_cast_f2d: Kernel,
    k_dot_rows: Kernel,
    k_axpy_basis: Kernel,
    k_norm2: Kernel,
    k_store_h_col: Kernel,
    k_finish_norm: Kernel,
    k_scale_recip: Kernel,
    k_lincomb: Kernel,
    /// Lazily-allocated f64 Krylov buffers, sized on the first `expmv`.
    krylov: Option<Krylov>,
}

/// f64 Arnoldi buffers for the Krylov exponential propagator. The basis
/// stays device-resident; the matvec drops to f32 through the `apply`
/// kernel, the orthogonalisation stays f64.
struct Krylov {
    /// Largest Krylov dimension the buffers are sized for.
    cap_dim: usize,
    /// Arnoldi basis, flat — `(cap_dim+1)` vectors of length `n`.
    basis: Buffer<cl_double>,
    /// Arnoldi working vector.
    w: Buffer<cl_double>,
    /// CGS2 projection coefficients.
    proj: Buffer<cl_double>,
    /// Result vector.
    out: Buffer<cl_double>,
    /// Per-work-group partial sums for the norm reduction.
    partials: Buffer<cl_double>,
    /// Hessenberg `H`, device-resident `cap_dim*cap_dim`, so the Arnoldi
    /// loop never round-trips it to the host.
    h: Buffer<cl_double>,
    /// One-element scalar holding the current Arnoldi `hnext = ||w||`.
    hnext: Buffer<cl_double>,
    /// Number of work-groups in the norm reduction.
    n_groups: usize,
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

        // Build the kernel with the element dimensions baked in. EPG (the
        // elements per work-group) is chosen so the group sits near the
        // target work-group size.
        let epg = (APPLY_TARGET_WG / np).max(1);
        let apply_wg = epg * np;
        let src = format!(
            "#define NP {np}\n#define NFP {nfp}\n#define COLS {cols}\n\
             #define EPG {epg}\n{APPLY_SRC}"
        );
        let program = gpu.build_program(&src)?;
        let kernel = Kernel::create(&program, "apply")
            .map_err(|e| format!("kernel create failed: {e}"))?;

        let lserk_program = gpu.build_program(LSERK_SRC)?;
        let lserk_kernel = Kernel::create(&lserk_program, "lserk_stage")
            .map_err(|e| format!("lserk kernel create failed: {e}"))?;
        let source_kernel = Kernel::create(&lserk_program, "add_source")
            .map_err(|e| format!("source kernel create failed: {e}"))?;

        let expmv_program = gpu.build_program(EXPMV_SRC)?;
        let kern = |name: &str| {
            Kernel::create(&expmv_program, name)
                .map_err(|e| format!("{name} kernel create failed: {e}"))
        };
        let k_cast_d2f = kern("cast_d2f")?;
        let k_cast_f2d = kern("cast_f2d")?;
        let k_dot_rows = kern("dot_rows")?;
        let k_axpy_basis = kern("axpy_basis")?;
        let k_norm2 = kern("partial_norm2")?;
        let k_store_h_col = kern("store_h_col")?;
        let k_finish_norm = kern("finish_norm")?;
        let k_scale_recip = kern("scale_recip")?;
        let k_lincomb = kern("lincomb")?;

        let y = gpu.alloc(n_dof)?;
        let dy = gpu.alloc(n_dof)?;
        let p = gpu.alloc(n_dof)?;

        Ok(GpuOperator {
            n_elem,
            n_dof,
            apply_epg: epg,
            apply_wg,
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
            source_kernel,
            _expmv_program: expmv_program,
            k_cast_d2f,
            k_cast_f2d,
            k_dot_rows,
            k_axpy_basis,
            k_norm2,
            k_store_h_col,
            k_finish_norm,
            k_scale_recip,
            k_lincomb,
            krylov: None,
        })
    }

    /// State-vector length the operator expects.
    pub fn n_dof(&self) -> usize {
        self.n_dof
    }

    /// Enqueue the `apply` kernel: `dy = A.y` on the resident state. No
    /// host transfer; the in-order queue serialises it with later work.
    fn enqueue_apply(&self, gpu: &GpuContext) -> Result<(), String> {
        // One work-group per EPG-element block; `apply_wg` work-items each.
        let n_groups = self.n_elem.div_ceil(self.apply_epg);
        let global = n_groups * self.apply_wg;
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
                .set_local_work_size(self.apply_wg)
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

    /// `f64` host wrapper around [`Self::apply`] for the macromodel
    /// build: cast `f64 -> f32`, run the device matvec, cast back
    /// `f32 -> f64`. The mixed-precision drift is bounded by
    /// [`crate::constants::GPU_REL_TOL`] per matvec; the block-Krylov
    /// build calls this once per basis vector and the projection
    /// onto the orthonormal `V` averages the rounding noise to that
    /// scale across the macromodel.
    ///
    /// Used by the `apply_fn` closure that
    /// [`crate::macromodel::MacroModel::build_with_apply_fn`] takes,
    /// so the CPU and GPU build paths share the same block-CGS2
    /// orthogonalisation and Hessenberg loop. The GPU does the
    /// `n_dof`-sized work; the CPU does the small dot products.
    pub fn apply_f64(
        &mut self,
        gpu: &GpuContext,
        y_host: &[f64],
    ) -> Result<Vec<f64>, String> {
        let y_f32: Vec<f32> = y_host.iter().map(|&v| v as f32).collect();
        let dy_f32 = self.apply(gpu, &y_f32)?;
        Ok(dy_f32.into_iter().map(|v| v as f64).collect())
    }

    /// Enqueue the soft-source add: `dy[source_dof] += val`.
    fn enqueue_add_source(
        &self,
        gpu: &GpuContext,
        dof: cl_int,
        val: f32,
    ) -> Result<(), String> {
        unsafe {
            ExecuteKernel::new(&self.source_kernel)
                .set_arg(&self.dy)
                .set_arg(&dof)
                .set_arg(&val)
                .set_global_work_size(1)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("source kernel launch failed: {e}"))?;
        Ok(())
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

    /// Like [`transient`](Self::transient) but returns the full field
    /// trajectory, flat `[(steps+1) * n_dof]` with row 0 the initial
    /// state. `dt` is the output cadence; the explicit integrator takes
    /// `substeps` LSERK4 steps of `dt/substeps` between snapshots, so the
    /// caller can keep the substep within the CFL limit while sampling at
    /// any cadence. One snapshot is downloaded per output step; the state
    /// itself steps device-resident.
    pub fn transient_traj(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
        substeps: usize,
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y0.len(), self.n_dof, "state length mismatch");
        let n = self.n_dof;
        let substeps = substeps.max(1);
        let h = dt / substeps as f32;
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y0, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        let zeros = vec![0.0_f32; n];
        unsafe {
            gpu.queue().enqueue_write_buffer(
                &mut self.p, CL_BLOCKING, 0, &zeros, &[],
            )
        }
        .map_err(|e| format!("register init failed: {e}"))?;

        let mut traj = Vec::with_capacity((steps + 1) * n);
        traj.extend_from_slice(y0);
        for _ in 0..steps {
            for _ in 0..substeps {
                for stage in 0..LSERK4_STAGES {
                    self.enqueue_apply(gpu)?;
                    self.enqueue_lserk(
                        gpu,
                        LSERK4_A[stage] as f32,
                        LSERK4_B[stage] as f32,
                        h,
                    )?;
                }
            }
            let row = gpu.download(&self.y, n)?;
            traj.extend_from_slice(&row);
        }
        Ok(traj)
    }

    /// Driven transient: `dy/dt = A.y + b`, with `b` a single-DOF soft
    /// source held constant across each step (the zeroth-order hold the
    /// CPU `step_driven` uses). `source_values[k]` is the source amplitude
    /// for step `k`, and its length sets the step count.
    pub fn transient_driven(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        source_dof: usize,
        source_values: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y0.len(), self.n_dof, "state length mismatch");
        assert!(source_dof < self.n_dof, "source_dof out of range");
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y0, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        let zeros = vec![0.0_f32; self.n_dof];
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.p, CL_BLOCKING, 0, &zeros, &[])
        }
        .map_err(|e| format!("register init failed: {e}"))?;

        let dof = source_dof as cl_int;
        for &g in source_values {
            for stage in 0..LSERK4_STAGES {
                self.enqueue_apply(gpu)?;
                self.enqueue_add_source(gpu, dof, g)?;
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
            .map_err(|e| format!("driven transient sync failed: {e}"))?;
        gpu.download(&self.y, self.n_dof)
    }

    /// Driven transient returning the full trajectory, flat
    /// `[(steps+1) * n_dof]` with row 0 the initial state. `dt` is the
    /// output cadence; the integrator takes `substeps` LSERK4 steps of
    /// `dt/substeps` between snapshots. `source_values` holds one source
    /// amplitude per substep (length `steps * substeps`), so the caller
    /// re-samples the waveform per substep.
    pub fn transient_driven_traj(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
        substeps: usize,
        source_dof: usize,
        source_values: &[f32],
    ) -> Result<Vec<f32>, String> {
        assert_eq!(y0.len(), self.n_dof, "state length mismatch");
        assert!(source_dof < self.n_dof, "source_dof out of range");
        let substeps = substeps.max(1);
        assert_eq!(
            source_values.len(),
            steps * substeps,
            "source values must have steps*substeps entries",
        );
        let n = self.n_dof;
        let h = dt / substeps as f32;
        unsafe {
            gpu.queue()
                .enqueue_write_buffer(&mut self.y, CL_BLOCKING, 0, y0, &[])
        }
        .map_err(|e| format!("state upload failed: {e}"))?;
        let zeros = vec![0.0_f32; n];
        unsafe {
            gpu.queue().enqueue_write_buffer(
                &mut self.p, CL_BLOCKING, 0, &zeros, &[],
            )
        }
        .map_err(|e| format!("register init failed: {e}"))?;

        let dof = source_dof as cl_int;
        let mut traj = Vec::with_capacity((steps + 1) * n);
        traj.extend_from_slice(y0);
        for k in 0..steps {
            for j in 0..substeps {
                let g = source_values[k * substeps + j];
                for stage in 0..LSERK4_STAGES {
                    self.enqueue_apply(gpu)?;
                    self.enqueue_add_source(gpu, dof, g)?;
                    self.enqueue_lserk(
                        gpu,
                        LSERK4_A[stage] as f32,
                        LSERK4_B[stage] as f32,
                        h,
                    )?;
                }
            }
            let row = gpu.download(&self.y, n)?;
            traj.extend_from_slice(&row);
        }
        Ok(traj)
    }

    /// Ensure the f64 Krylov buffers are allocated for dimension `m`.
    fn ensure_krylov(
        &mut self,
        gpu: &GpuContext,
        m: usize,
    ) -> Result<(), String> {
        let n = self.n_dof;
        let big_enough =
            matches!(&self.krylov, Some(k) if k.cap_dim >= m);
        if !big_enough {
            let n_groups = n.div_ceil(NORM_WORK_GROUP);
            self.krylov = Some(Krylov {
                cap_dim: m,
                basis: gpu.alloc_f64((m + 1) * n)?,
                w: gpu.alloc_f64(n)?,
                proj: gpu.alloc_f64(m + 1)?,
                out: gpu.alloc_f64(n)?,
                partials: gpu.alloc_f64(n_groups)?,
                h: gpu.alloc_f64(m * m)?,
                hnext: gpu.alloc_f64(1)?,
                n_groups,
            });
        }
        Ok(())
    }

    /// Run an `m`-step Arnoldi process from `basis[0]` (unit-norm in slot
    /// 0). `h_dev` must be a zeroed `m*m` device buffer. The loop runs
    /// entirely device-side — projections accumulate straight into the
    /// device Hessenberg, the norm is finished on the device — so the
    /// Hessenberg is the *only* thing the host reads back, once, at the
    /// end. Fixed dimension `m` (no breakdown check).
    #[allow(clippy::too_many_arguments)]
    fn arnoldi(
        &self,
        gpu: &GpuContext,
        basis: &Buffer<cl_double>,
        w: &Buffer<cl_double>,
        proj: &Buffer<cl_double>,
        partials: &Buffer<cl_double>,
        h_dev: &Buffer<cl_double>,
        hnext_buf: &Buffer<cl_double>,
        n_groups: usize,
        m: usize,
    ) -> Result<(Vec<f64>, usize), String> {
        let n = self.n_dof;
        let n_i = n as cl_int;
        let m_i = m as cl_int;
        let ng_i = n_groups as cl_int;
        let elem_global = n.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
        let norm_global =
            n.div_ceil(NORM_WORK_GROUP) * NORM_WORK_GROUP;

        for j in 0..m {
            // matvec w = A*basis[j]: cast basis[j] down, apply, cast up.
            let off = (j * n) as cl_int;
            unsafe {
                ExecuteKernel::new(&self.k_cast_d2f)
                    .set_arg(basis)
                    .set_arg(&off)
                    .set_arg(&self.y)
                    .set_arg(&n_i)
                    .set_global_work_size(elem_global)
                    .set_local_work_size(DOF_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("cast_d2f launch failed: {e}"))?;
            self.enqueue_apply(gpu)?;
            unsafe {
                ExecuteKernel::new(&self.k_cast_f2d)
                    .set_arg(&self.dy)
                    .set_arg(w)
                    .set_arg(&n_i)
                    .set_global_work_size(elem_global)
                    .set_local_work_size(DOF_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("cast_f2d launch failed: {e}"))?;

            // CGS2: two passes; each pass's projections accumulate into
            // column j of the device Hessenberg.
            let cols = j + 1;
            let cols_i = cols as cl_int;
            let col_i = j as cl_int;
            let proj_global =
                cols.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
            for _pass in 0..2 {
                unsafe {
                    ExecuteKernel::new(&self.k_dot_rows)
                        .set_arg(basis)
                        .set_arg(w)
                        .set_arg(proj)
                        .set_arg(&n_i)
                        .set_arg(&cols_i)
                        .set_arg_local_buffer(DOF_WORK_GROUP * 8)
                        .set_global_work_size(cols * DOF_WORK_GROUP)
                        .set_local_work_size(DOF_WORK_GROUP)
                        .enqueue_nd_range(gpu.queue())
                }
                .map_err(|e| format!("dot_rows launch failed: {e}"))?;
                unsafe {
                    ExecuteKernel::new(&self.k_store_h_col)
                        .set_arg(h_dev)
                        .set_arg(proj)
                        .set_arg(&col_i)
                        .set_arg(&m_i)
                        .set_arg(&cols_i)
                        .set_global_work_size(proj_global)
                        .set_local_work_size(DOF_WORK_GROUP)
                        .enqueue_nd_range(gpu.queue())
                }
                .map_err(|e| format!("store_h_col launch failed: {e}"))?;
                unsafe {
                    ExecuteKernel::new(&self.k_axpy_basis)
                        .set_arg(w)
                        .set_arg(basis)
                        .set_arg(proj)
                        .set_arg(&n_i)
                        .set_arg(&cols_i)
                        .set_global_work_size(elem_global)
                        .set_local_work_size(DOF_WORK_GROUP)
                        .enqueue_nd_range(gpu.queue())
                }
                .map_err(|e| format!("axpy_basis launch failed: {e}"))?;
            }

            // The last basis vector needs no successor.
            if j + 1 == m {
                break;
            }

            // hnext = ||w||, finished device-side into `hnext_buf` and the
            // subdiagonal H[(j+1), j]; then basis[j+1] = w / hnext.
            unsafe {
                ExecuteKernel::new(&self.k_norm2)
                    .set_arg(w)
                    .set_arg(partials)
                    .set_arg(&n_i)
                    .set_arg_local_buffer(NORM_WORK_GROUP * 8)
                    .set_global_work_size(norm_global)
                    .set_local_work_size(NORM_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("partial_norm2 launch failed: {e}"))?;
            unsafe {
                ExecuteKernel::new(&self.k_finish_norm)
                    .set_arg(partials)
                    .set_arg(&ng_i)
                    .set_arg(hnext_buf)
                    .set_arg(h_dev)
                    .set_arg(&col_i)
                    .set_arg(&m_i)
                    .set_global_work_size(1)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("finish_norm launch failed: {e}"))?;
            let dst_off = ((j + 1) * n) as cl_int;
            unsafe {
                ExecuteKernel::new(&self.k_scale_recip)
                    .set_arg(w)
                    .set_arg(basis)
                    .set_arg(&dst_off)
                    .set_arg(hnext_buf)
                    .set_arg(&n_i)
                    .set_global_work_size(elem_global)
                    .set_local_work_size(DOF_WORK_GROUP)
                    .enqueue_nd_range(gpu.queue())
            }
            .map_err(|e| format!("scale_recip launch failed: {e}"))?;
        }
        // The Hessenberg is the only host round-trip — downloaded once.
        let h = gpu.download_f64(h_dev, m * m)?;
        Ok((h, m))
    }

    /// Krylov linear combination `out = sum_i coef[i] * basis[i]`.
    fn lincomb(
        &self,
        gpu: &GpuContext,
        basis: &Buffer<cl_double>,
        coef: &[f64],
        out: &Buffer<cl_double>,
    ) -> Result<(), String> {
        let n_i = self.n_dof as cl_int;
        let dim_i = coef.len() as cl_int;
        let elem_global =
            self.n_dof.div_ceil(DOF_WORK_GROUP) * DOF_WORK_GROUP;
        let coef_buf = gpu.upload_f64(coef)?;
        unsafe {
            ExecuteKernel::new(&self.k_lincomb)
                .set_arg(basis)
                .set_arg(&coef_buf)
                .set_arg(out)
                .set_arg(&n_i)
                .set_arg(&dim_i)
                .set_global_work_size(elem_global)
                .set_local_work_size(DOF_WORK_GROUP)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("lincomb launch failed: {e}"))?;
        Ok(())
    }

    /// Project a device vector onto the basis: `basis^T · vec`, returning
    /// the `dim` coefficients to the host.
    fn project(
        &self,
        gpu: &GpuContext,
        basis: &Buffer<cl_double>,
        vec: &Buffer<cl_double>,
        proj: &Buffer<cl_double>,
        dim: usize,
    ) -> Result<Vec<f64>, String> {
        let n_i = self.n_dof as cl_int;
        let dim_i = dim as cl_int;
        unsafe {
            ExecuteKernel::new(&self.k_dot_rows)
                .set_arg(basis)
                .set_arg(vec)
                .set_arg(proj)
                .set_arg(&n_i)
                .set_arg(&dim_i)
                .set_arg_local_buffer(DOF_WORK_GROUP * 8)
                .set_global_work_size(dim * DOF_WORK_GROUP)
                .set_local_work_size(DOF_WORK_GROUP)
                .enqueue_nd_range(gpu.queue())
        }
        .map_err(|e| format!("dot_rows launch failed: {e}"))?;
        gpu.download_f64(proj, dim)
    }

    /// Matrix-free `exp(t*A)*v` via an `m`-step Krylov projection — the GPU
    /// counterpart of [`crate::propagator::expmv`].
    ///
    /// For `m` above [`KRYLOV_CHUNK`] the propagation is **sub-stepped**:
    /// `exp(t*A) = exp((t/k)*A)^k` is exact, so `k` sub-steps each with a
    /// small Krylov space give the same result as one large space, but the
    /// device-resident Arnoldi basis is capped at `~KRYLOV_CHUNK * n_dof`
    /// rather than `~m * n_dof`. Sub-steps round-trip the state through the
    /// host; that transfer is small next to the Arnoldi itself.
    pub fn expmv(
        &mut self,
        gpu: &GpuContext,
        v: &[f64],
        t: f64,
        m: usize,
    ) -> Result<Vec<f64>, String> {
        assert_eq!(v.len(), self.n_dof, "state length mismatch");
        assert!(m >= 1, "Krylov dimension must be >= 1");
        let k = m.div_ceil(KRYLOV_CHUNK).max(1);
        let chunk = m.div_ceil(k);
        if k == 1 {
            return self.expmv_chunk(gpu, v, t, chunk);
        }
        // Sub-step: each piece covers t/k with a `chunk`-dimensional space.
        let tau = t / k as f64;
        let mut state = v.to_vec();
        for _ in 0..k {
            state = self.expmv_chunk(gpu, &state, tau, chunk)?;
        }
        Ok(state)
    }

    /// One Krylov sub-step: `exp(t*A)*v` from a single `m`-dimensional
    /// Arnoldi space. The basis is device-resident in f64 and the CGS2
    /// orthogonalisation runs in f64; the matvec drops through the f32
    /// `apply` kernel. The dense `exp(t*H)` of the small Hessenberg is done
    /// on the host.
    fn expmv_chunk(
        &mut self,
        gpu: &GpuContext,
        v: &[f64],
        t: f64,
        m: usize,
    ) -> Result<Vec<f64>, String> {
        let n = self.n_dof;
        assert_eq!(v.len(), n, "state length mismatch");
        assert!(m >= 1, "Krylov dimension must be >= 1");
        let beta: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if beta == 0.0 {
            return Ok(vec![0.0; n]);
        }
        self.ensure_krylov(gpu, m)?;
        let mut kry = self.krylov.take().expect("krylov allocated");

        // basis[0] = v / beta; zero the device Hessenberg.
        let b0: Vec<f64> = v.iter().map(|x| x / beta).collect();
        gpu.write_f64(&mut kry.basis, &b0)?;
        gpu.write_f64(&mut kry.h, &vec![0.0_f64; m * m])?;
        let (h, dim) = self.arnoldi(
            gpu, &kry.basis, &kry.w, &kry.proj, &kry.partials, &kry.h,
            &kry.hnext, kry.n_groups, m,
        )?;

        // Dense exp(t*H) on the host; out = beta * sum_i basis[i] * exp[i,0].
        let mut th = vec![0.0_f64; dim * dim];
        for a in 0..dim {
            for b in 0..dim {
                th[a * dim + b] = t * h[a * m + b];
            }
        }
        let exp_th = expm(&th, dim);
        let coef: Vec<f64> =
            (0..dim).map(|i| beta * exp_th[i * dim]).collect();
        self.lincomb(gpu, &kry.basis, &coef, &kry.out)?;
        gpu.queue()
            .finish()
            .map_err(|e| format!("expmv sync failed: {e}"))?;
        let result = gpu.download_f64(&kry.out, n)?;
        self.krylov = Some(kry);
        Ok(result)
    }

    /// Exponential-warmup hybrid transient: the first `warmup` steps use
    /// the exact exponential propagator, the rest the cheaper explicit
    /// LSERK4 stepper. The exact integrator carries the opening transient,
    /// then hands the smooth state to the explicit stepper.
    pub fn transient_hybrid(
        &mut self,
        gpu: &GpuContext,
        y0: &[f32],
        dt: f32,
        steps: usize,
        warmup: usize,
        krylov_dim: usize,
    ) -> Result<Vec<f32>, String> {
        let warmup = warmup.min(steps);
        // Warmup: exact exponential steps in f64.
        let mut y: Vec<f64> = y0.iter().map(|&v| v as f64).collect();
        for _ in 0..warmup {
            y = self.expmv(gpu, &y, dt as f64, krylov_dim)?;
        }
        // Remainder: device-resident explicit LSERK4.
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        self.transient(gpu, &y32, dt, steps - warmup)
    }

    /// Build a Krylov model-order-reduced model around `start` — an
    /// `r`-step Arnoldi projection. The GPU counterpart of
    /// [`crate::mor::ReducedModel`]; the reduced model propagates states
    /// inside that subspace cheaply.
    pub fn reduce(
        &mut self,
        gpu: &GpuContext,
        start: &[f64],
        r: usize,
    ) -> Result<GpuReducedModel, String> {
        let n = self.n_dof;
        assert_eq!(start.len(), n, "state length mismatch");
        assert!(r >= 1, "reduced dimension must be >= 1");
        let beta: f64 = start.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(beta > 0.0, "start vector must be nonzero");

        let mut basis = gpu.alloc_f64((r + 1) * n)?;
        let proj = gpu.alloc_f64(r + 1)?;
        let out = gpu.alloc_f64(n)?;
        let w = gpu.alloc_f64(n)?;
        let n_groups = n.div_ceil(NORM_WORK_GROUP);
        let partials = gpu.alloc_f64(n_groups)?;
        let mut h_dev = gpu.alloc_f64(r * r)?;
        let hnext = gpu.alloc_f64(1)?;

        let b0: Vec<f64> = start.iter().map(|x| x / beta).collect();
        gpu.write_f64(&mut basis, &b0)?;
        gpu.write_f64(&mut h_dev, &vec![0.0_f64; r * r])?;
        let (h, dim) = self.arnoldi(
            gpu, &basis, &w, &proj, &partials, &h_dev, &hnext, n_groups, r,
        )?;
        gpu.queue()
            .finish()
            .map_err(|e| format!("reduce sync failed: {e}"))?;

        // a_hat = H[0..dim, 0..dim].
        let mut a_hat = vec![0.0_f64; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                a_hat[i * dim + j] = h[i * r + j];
            }
        }
        Ok(GpuReducedModel { n, dim, a_hat, basis, proj, out })
    }
}

/// A Krylov model-order-reduced model on the GPU. The Arnoldi basis stays
/// device-resident; [`propagate`](Self::propagate) projects a state into
/// the subspace, applies the dense reduced exponential on the host, and
/// lifts the result back.
pub struct GpuReducedModel {
    n: usize,
    dim: usize,
    /// Reduced operator `A_hat = H[0..dim, 0..dim]`, row-major (host).
    a_hat: Vec<f64>,
    basis: Buffer<cl_double>,
    proj: Buffer<cl_double>,
    out: Buffer<cl_double>,
}

impl GpuReducedModel {
    /// Reduced dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Propagate `y0` by `t` through the reduced model:
    /// `lift(exp(t*A_hat) * project(y0))`. Takes the operator `op` for its
    /// projection and linear-combination kernels.
    pub fn propagate(
        &self,
        gpu: &GpuContext,
        op: &GpuOperator,
        y0: &[f64],
        t: f64,
    ) -> Result<Vec<f64>, String> {
        assert_eq!(y0.len(), self.n, "state length mismatch");
        // y_hat = basis^T · y0.
        let y0_buf = gpu.upload_f64(y0)?;
        let y_hat =
            op.project(gpu, &self.basis, &y0_buf, &self.proj, self.dim)?;
        // y_hat_t = exp(t*A_hat) · y_hat — dense, on the host.
        let th: Vec<f64> = self.a_hat.iter().map(|x| x * t).collect();
        let exp_th = expm(&th, self.dim);
        let coef: Vec<f64> = (0..self.dim)
            .map(|i| {
                (0..self.dim)
                    .map(|j| exp_th[i * self.dim + j] * y_hat[j])
                    .sum()
            })
            .collect();
        // out = basis · coef.
        op.lincomb(gpu, &self.basis, &coef, &self.out)?;
        gpu.queue()
            .finish()
            .map_err(|e| format!("reduced propagate sync failed: {e}"))?;
        gpu.download_f64(&self.out, self.n)
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

    #[test]
    fn gpu_driven_transient_matches_cpu() {
        // P2.3 gate: the GPU driven transient (soft source) matches the
        // CPU driven LSERK4 within GPU_REL_TOL.
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

        let mut v: Vec<Field> =
            (0..n).map(|i| (0.1 + i as Field * 0.007).sin()).collect();
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
        let steps = 150;
        let sdof = n / 3;
        let src: Vec<Field> =
            (0..steps).map(|k| (0.3 * k as Field).sin()).collect();

        // CPU reference — driven from rest.
        let mut y_cpu = vec![0.0; n];
        let mut ws = LserkWorkspace::new();
        for &g in &src {
            ws.step_driven_into(
                |x, ax| op.apply_into(x, ax),
                &mut y_cpu,
                dt,
                sdof,
                g,
            );
        }

        // GPU.
        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let y0 = vec![0.0_f32; n];
        let src32: Vec<f32> = src.iter().map(|&v| v as f32).collect();
        let y_gpu = gop
            .transient_driven(&gpu, &y0, dt as f32, sdof, &src32)
            .expect("driven transient");

        let rel = rel_l2(&y_gpu, &y_cpu);
        eprintln!(
            "GPU driven transient vs CPU [{steps} steps]: rel L2 = {rel:.3e}"
        );
        assert!(
            rel < GPU_REL_TOL,
            "GPU driven transient rel.err {rel:.3e} exceeds GPU_REL_TOL",
        );
    }

    #[test]
    fn gpu_expmv_matches_cpu() {
        // P3 gate: the GPU Krylov exponential propagator matches the CPU
        // expmv within GPU_REL_TOL (the f32 matvec caps the accuracy).
        use crate::propagator::expmv;

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
        let v: Vec<Field> =
            (0..n).map(|i| (0.3 + i as Field * 0.013).sin()).collect();
        let t = 0.02;
        let m = 40;

        let cpu = expmv(|x| op.apply(x), &v, t, m);

        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let gpu_out = gop.expmv(&gpu, &v, t, m).expect("gpu expmv");

        let err: f64 = cpu
            .iter()
            .zip(&gpu_out)
            .map(|(&c, &g)| (c - g).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = cpu.iter().map(|&c| c * c).sum::<f64>().sqrt();
        let rel = err / scale;
        eprintln!(
            "GPU expmv vs CPU [m={m}]: rel L2 = {rel:.3e} \
             (GPU_REL_TOL {GPU_REL_TOL:.1e})"
        );
        assert!(
            rel < GPU_REL_TOL,
            "GPU expmv rel.err {rel:.3e} exceeds GPU_REL_TOL",
        );
    }

    #[test]
    fn gpu_hybrid_transient_matches_cpu() {
        // P2.4 gate: the GPU exponential-warmup hybrid matches a CPU hybrid
        // (warmup expmv steps, then LSERK4) within GPU_REL_TOL.
        use crate::explicit::LserkWorkspace;
        use crate::propagator::expmv;

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
            (0..n).map(|i| (0.15 + i as Field * 0.009).sin()).collect();

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
        let (steps, warmup, m) = (120, 10, 40);

        // CPU hybrid: warmup exponential steps, then LSERK4.
        let mut y_cpu = y0.clone();
        for _ in 0..warmup {
            y_cpu = expmv(|x| op.apply(x), &y_cpu, dt, m);
        }
        let mut ws = LserkWorkspace::new();
        for _ in 0..(steps - warmup) {
            ws.step_into(|x, ax| op.apply_into(x, ax), &mut y_cpu, dt);
        }

        // GPU hybrid.
        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        let y_gpu = gop
            .transient_hybrid(&gpu, &y0_32, dt as f32, steps, warmup, m)
            .expect("hybrid transient");

        let rel = rel_l2(&y_gpu, &y_cpu);
        eprintln!(
            "GPU hybrid transient vs CPU [{warmup}+{} steps]: rel L2 = {rel:.3e}",
            steps - warmup
        );
        assert!(
            rel < GPU_REL_TOL,
            "GPU hybrid rel.err {rel:.3e} exceeds GPU_REL_TOL",
        );
    }

    #[test]
    fn gpu_reduced_model_matches_cpu() {
        // P4 gate: the GPU Krylov reduced model matches the CPU
        // ReducedModel within GPU_REL_TOL.
        use crate::mor::ReducedModel;

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
        let start: Vec<Field> =
            (0..n).map(|i| (0.5 + i as Field * 0.021).sin()).collect();
        let r = 40;
        let t = 0.03;

        let cpu_rom = ReducedModel::build(|x| op.apply(x), &start, r);
        let cpu_out = cpu_rom.propagate(&start, t);

        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let gmodel = gop.reduce(&gpu, &start, r).expect("reduce");
        let gpu_out = gmodel
            .propagate(&gpu, &gop, &start, t)
            .expect("reduced propagate");

        let err: f64 = cpu_out
            .iter()
            .zip(&gpu_out)
            .map(|(&c, &g)| (c - g).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 =
            cpu_out.iter().map(|&c| c * c).sum::<f64>().sqrt();
        let rel = err / scale;
        eprintln!(
            "GPU reduced model vs CPU [r={}]: rel L2 = {rel:.3e}",
            gmodel.dim()
        );
        assert!(
            rel < GPU_REL_TOL,
            "GPU reduced model rel.err {rel:.3e} exceeds GPU_REL_TOL",
        );
    }
}
