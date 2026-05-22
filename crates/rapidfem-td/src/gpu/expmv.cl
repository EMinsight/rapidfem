// Krylov exponential propagator kernels — the f64 Arnoldi machinery.
//
// The Arnoldi basis and the CGS2 orthogonalisation are f64 (orthogonality
// is lost fast in f32). The matvec itself runs through the f32 `apply`
// kernel, so the basis vector is cast down and the result cast back up;
// `cast_d2f` / `cast_f2d` are that boundary.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// f64 -> f32, for the matvec input. Reads `src` from `src_off` (so a
// basis vector can be picked out of the flat basis buffer).
kernel void cast_d2f(global const double* src,
                     const int src_off,
                     global float* dst,
                     const int n) {
    const int i = get_global_id(0);
    if (i < n)
        dst[i] = (float)src[src_off + i];
}

// f32 -> f64, for the matvec result.
kernel void cast_f2d(global const float* src,
                     global double* dst,
                     const int n) {
    const int i = get_global_id(0);
    if (i < n)
        dst[i] = (double)src[i];
}

// proj[i] = <basis[i], w>, one work-item per basis row.
kernel void dot_rows(global const double* basis,
                     global const double* w,
                     global double* proj,
                     const int n,
                     const int cols) {
    const int i = get_global_id(0);
    if (i >= cols)
        return;
    global const double* bi = basis + (size_t)i * n;
    double acc = 0.0;
    for (int k = 0; k < n; k++)
        acc += bi[k] * w[k];
    proj[i] = acc;
}

// w -= sum_i proj[i] * basis[i], one work-item per DOF.
kernel void axpy_basis(global double* w,
                       global const double* basis,
                       global const double* proj,
                       const int n,
                       const int cols) {
    const int k = get_global_id(0);
    if (k >= n)
        return;
    double acc = w[k];
    for (int i = 0; i < cols; i++)
        acc -= proj[i] * basis[(size_t)i * n + k];
    w[k] = acc;
}

// Per-work-group partial sums of <w,w>; the host finishes the reduction.
kernel void partial_norm2(global const double* w,
                          global double* partials,
                          const int n,
                          local double* scratch) {
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int ls = get_local_size(0);
    scratch[lid] = (gid < n) ? w[gid] * w[gid] : 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ls / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        partials[get_group_id(0)] = scratch[0];
}

// dst[dst_off + i] = src[i] * c — writes the scaled vector into a slot of
// the flat basis buffer.
kernel void scale_into(global const double* src,
                       global double* dst,
                       const int dst_off,
                       const double c,
                       const int n) {
    const int i = get_global_id(0);
    if (i < n)
        dst[dst_off + i] = src[i] * c;
}

// out = sum_i coef[i] * basis[i], one work-item per DOF — the final
// Krylov linear combination.
kernel void lincomb(global const double* basis,
                    global const double* coef,
                    global double* out,
                    const int n,
                    const int dim) {
    const int k = get_global_id(0);
    if (k >= n)
        return;
    double acc = 0.0;
    for (int i = 0; i < dim; i++)
        acc += coef[i] * basis[(size_t)i * n + k];
    out[k] = acc;
}
