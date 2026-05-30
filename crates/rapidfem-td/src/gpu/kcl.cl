// KCL RK4(3)5[2R+]C adaptive stage updates, one work-item per DOF.
//
// Five matvecs per step, like LSERK4, plus the embedded-error accumulator
// `e` and a `dtF` register that carries `dt·F_{i-1}` between stages — the
// 2R+ form (Ketcheson 2010, Algorithm 2). The host runs the PI controller
// on top of `weighted_err_sq` (below) and accepts or rejects each step.

// Stage 0: the matvec result `k` is `A·y_n`; initialise the embedded-error
// register, prime `dtF` with `dt·F_0`, and advance the state to S2 after
// stage 0 (= y_n + b_0·dt·F_0). After this kernel `y` holds the running
// main accumulator S2, and `dtF` carries `dt·F_0` into the stage-1 build.
kernel void kcl_stage0(global float* y,
                       global float* dtF,
                       global float* e,
                       global const float* k,
                       const float dt,
                       const float b0,
                       const float e0,
                       const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    const float f = dt * k[i];
    dtF[i] = f;
    y[i] += b0 * f;
    e[i] = e0 * f;
}

// Build the next stage state `Y_i = S2 + (A_sub - b_prev) · dt·F_{i-1}`,
// the matvec evaluation point for stage `i` (i >= 1). Reads `y` (the
// running S2 accumulator) and `dtF` (still carrying `dt·F_{i-1}`),
// writes the new stage state into `stage` — the buffer the next matvec
// reads from.
kernel void kcl_build_stage(global const float* y,
                            global const float* dtF,
                            global float* stage,
                            const float amb,
                            const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    stage[i] = y[i] + amb * dtF[i];
}

// Stage i >= 1 accumulation: after `k = A·Y_i` from the matvec, advance
// S2 (`y`) and the embedded-error register, and overwrite `dtF` with
// `dt·F_i` for the next stage's build. `eweight = b_hat_i - b_i` is the
// per-stage embedded weight, pre-computed host-side.
kernel void kcl_stage_accum(global float* y,
                            global float* dtF,
                            global float* e,
                            global const float* k,
                            const float dt,
                            const float b,
                            const float eweight,
                            const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    const float f = dt * k[i];
    dtF[i] = f;
    y[i] += b * f;
    e[i] += eweight * f;
}

// Per-workgroup reduction of `(err[i] / (atol + rtol·max(|y_old|,|y_new|)))²`
// — the per-DOF weighted squared error the host sums and roots into the
// scalar `err_norm` an accept/reject controller compares against 1.0.
// Local-memory tree reduction; output `partial_sums[wg_id]` holds the
// workgroup's contribution.
kernel void weighted_err_sq(global const float* y_old,
                            global const float* y_new,
                            global const float* err,
                            global float* partial_sums,
                            local float* scratch,
                            const float atol,
                            const float rtol,
                            const int n) {
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int wid = get_group_id(0);
    const int lsize = get_local_size(0);

    float v = 0.0f;
    if (gid < n) {
        const float yo = fabs(y_old[gid]);
        const float yn = fabs(y_new[gid]);
        const float s = atol + rtol * fmax(yo, yn);
        const float r = err[gid] / s;
        v = r * r;
    }
    scratch[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        partial_sums[wid] = scratch[0];
    }
}

// Copy `dst[i] = src[i]` over `n` DOFs — the host snapshot/rollback path
// for an accept/reject substep. Used to backup `y` before each attempted
// substep and to restore it on rejection.
kernel void copy_vec(global float* dst,
                     global const float* src,
                     const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    dst[i] = src[i];
}
