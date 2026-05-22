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
