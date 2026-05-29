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

// proj[i] = <basis[i], w> — one work-group per basis row, the dot reduced
// over `n` in local memory. A work-item-per-row design left the device
// almost idle (only `cols` threads); this fans each dot across a whole
// work-group, so `cols` work-groups keep the GPU busy.
kernel void dot_rows(global const double* basis,
                     global const double* w,
                     global double* proj,
                     const int n,
                     const int cols,
                     local double* scratch) {
    const int row = get_group_id(0);
    if (row >= cols)
        return;
    const int lid = get_local_id(0);
    const int ls = get_local_size(0);
    global const double* bi = basis + (size_t)row * n;
    double acc = 0.0;
    for (int k = lid; k < n; k += ls)
        acc += bi[k] * w[k];
    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ls / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        proj[row] = scratch[0];
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

// w[i] += g * src[i], one work-item per DOF — the f64 axpy that injects
// the source column of the augmented operator [[A, b],[0, 0]] into the
// Arnoldi working vector: the augmented matvec is
// `w_vec = A*basis_vec + xi*b`, with `g = xi` the held augmented scalar.
kernel void axpy_src(global double* w,
                     global const double* src,
                     const double g,
                     const int n) {
    const int i = get_global_id(0);
    if (i < n)
        w[i] += g * src[i];
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

// Accumulate proj[0..cols] into column `col` of the m-by-m Hessenberg H.
// CGS2 runs two passes, so this is a `+=`; H is zeroed once before the
// Arnoldi loop.
kernel void store_h_col(global double* h,
                        global const double* proj,
                        const int col,
                        const int m,
                        const int cols) {
    const int i = get_global_id(0);
    if (i < cols)
        h[i * m + col] += proj[i];
}

// Finish the norm reduction device-side: hnext = sqrt(sum partials).
// Writes hnext to `hnext_buf[0]` and the subdiagonal `H[(j+1), j]`, so the
// Arnoldi loop never has to round-trip the norm through the host.
kernel void finish_norm(global const double* partials,
                        const int n_groups,
                        global double* hnext_buf,
                        global double* h,
                        const int j,
                        const int m) {
    if (get_global_id(0) != 0)
        return;
    double s = 0.0;
    for (int g = 0; g < n_groups; g++)
        s += partials[g];
    const double hnext = sqrt(s);
    hnext_buf[0] = hnext;
    h[(j + 1) * m + j] = hnext;
}

// basis[dst_off + i] = w[i] / hnext, with hnext read from a device scalar
// — `basis[j+1] = w / ||w||` without a host round-trip.
kernel void scale_recip(global const double* w,
                        global double* basis,
                        const int dst_off,
                        global const double* hnext_buf,
                        const int n) {
    const int i = get_global_id(0);
    if (i < n)
        basis[dst_off + i] = w[i] / hnext_buf[0];
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
