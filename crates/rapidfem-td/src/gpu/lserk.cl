// LSERK4 low-storage Runge-Kutta stage update, one work-item per DOF.
//
//   p = a*p + dt*k ;  y = y + b*p
//
// `k` is the matvec result A.y from the apply kernel. Stage 0 has a = 0,
// which resets the residual register `p`.

kernel void lserk_stage(global float* p,
                        global const float* k,
                        global float* y,
                        const float a,
                        const float b,
                        const float dt,
                        const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    const float pi = a * p[i] + dt * k[i];
    p[i] = pi;
    y[i] = y[i] + b * pi;
}

// Add a held soft-source value to one DOF of the matvec result `k`. The
// driven system is dy/dt = A.y + b, with b a single-DOF rank-1 source.
kernel void add_source(global float* k, const int dof, const float val) {
    if (get_global_id(0) == 0)
        k[dof] += val;
}

// Add a held full-vector source to the matvec result `k`: k[i] += g*b[i].
// The driven system is dy/dt = A.y + b(t), with `b = src * g` — the path
// for modal-port injection, where the spatial pattern `src` spreads over
// every port-face DOF and `g` is the scalar waveform held across the step.
kernel void add_source_vec(global float* k,
                           global const float* src,
                           const float g,
                           const int n) {
    const int i = get_global_id(0);
    if (i >= n) return;
    k[i] += g * src[i];
}
