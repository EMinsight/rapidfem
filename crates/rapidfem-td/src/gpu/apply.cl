// DG Maxwell RHS operator — dy/dt = A.y. One work-item per element.
//
// NP, NFP, COLS are #define'd by the host before this source is built, so
// the private scratch arrays are exactly sized. All field state and
// operator data are f32 (the mixed-precision GPU path); this kernel is a
// faithful port of the CPU `apply_element`.

inline float3 cross3(float3 a, float3 b) {
    return (float3)(a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x);
}

// Physical curl of a node-major 3-vector field (`3*NP`) into `out` (`3*NP`).
//
// Looped output-node-outer: the reference and physical derivatives of one
// node are tiny 9-float scratch, never the `3*NP` / `9*NP` arrays the
// component-outer form needed. That keeps the work-item's private memory
// off the register-spill cliff. `jinv_e[k*3+p] = jacobian_inv[k][p]`.
void element_curl(constant const float* dr,
                  constant const float* ds,
                  constant const float* dt,
                  global const float* jinv_e,
                  __private const float* field,
                  __private float* out) {
    for (int i = 0; i < NP; i++) {
        // Reference derivatives at node i: rd[k*3 + comp].
        float rd[9];
        for (int k = 0; k < 3; k++) {
            constant const float* d =
                (k == 0) ? dr : ((k == 1) ? ds : dt);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
            for (int j = 0; j < NP; j++) {
                float dij = d[i * NP + j];
                a0 += dij * field[j * 3 + 0];
                a1 += dij * field[j * 3 + 1];
                a2 += dij * field[j * 3 + 2];
            }
            rd[k * 3 + 0] = a0;
            rd[k * 3 + 1] = a1;
            rd[k * 3 + 2] = a2;
        }
        // Physical derivatives at node i: pd[phys*3 + comp].
        float pd[9];
        for (int p = 0; p < 3; p++) {
            float j0 = jinv_e[0 * 3 + p];
            float j1 = jinv_e[1 * 3 + p];
            float j2 = jinv_e[2 * 3 + p];
            for (int c = 0; c < 3; c++)
                pd[p * 3 + c] =
                    j0 * rd[0 * 3 + c] + j1 * rd[1 * 3 + c]
                    + j2 * rd[2 * 3 + c];
        }
        // curl_x = d(Fz)/dy - d(Fy)/dz, and cyclic.
        out[i * 3 + 0] = pd[1 * 3 + 2] - pd[2 * 3 + 1];
        out[i * 3 + 1] = pd[2 * 3 + 0] - pd[0 * 3 + 2];
        out[i * 3 + 2] = pd[0 * 3 + 1] - pd[1 * 3 + 0];
    }
}

kernel void apply(global const float* y,
                  global float* dy,
                  constant const float* diff_r,
                  constant const float* diff_s,
                  constant const float* diff_t,
                  constant const float* lift,
                  constant const int* face_nodes,
                  global const float* jinv,
                  global const float* inv_eps,
                  global const float* inv_mu,
                  global const float* sigma_eps,
                  global const float* sigma_mu,
                  global const float* face_normal,
                  global const float* face_fscale,
                  global const int* face_neighbor,
                  global const int* face_nbr_local,
                  global const int* face_port,
                  global const int* face_perm,
                  const float flux_alpha,
                  const int n_elem) {
    const int e = get_global_id(0);
    if (e >= n_elem) return;

    float ee[3 * NP], hh[3 * NP], de[3 * NP], dh[3 * NP];
    const int base = e * NP * 6;
    for (int node = 0; node < NP; node++) {
        for (int c = 0; c < 3; c++) {
            ee[node * 3 + c] = y[base + node * 6 + c];
            hh[node * 3 + c] = y[base + node * 6 + 3 + c];
        }
    }

    // Volume term: dE = curl(H), dH = -curl(E).
    element_curl(diff_r, diff_s, diff_t, jinv + e * 9, hh, de);
    element_curl(diff_r, diff_s, diff_t, jinv + e * 9, ee, dh);
    for (int i = 0; i < 3 * NP; i++)
        dh[i] = -dh[i];

    // Surface term — the numerical flux.
    for (int f = 0; f < 4; f++) {
        const int ff = e * 4 + f;
        float3 nrm = (float3)(face_normal[ff * 3],
                              face_normal[ff * 3 + 1],
                              face_normal[ff * 3 + 2]);
        float coef = 0.5f * face_fscale[ff];
        int nbr = face_neighbor[ff];
        int nbrlf = face_nbr_local[ff];
        int port = face_port[ff];
        float a = (port >= 0) ? 1.0f : flux_alpha;
        for (int m = 0; m < NFP; m++) {
            int vi = face_nodes[f * NFP + m];
            float3 em = (float3)(ee[vi * 3], ee[vi * 3 + 1], ee[vi * 3 + 2]);
            float3 hm = (float3)(hh[vi * 3], hh[vi * 3 + 1], hh[vi * 3 + 2]);
            float3 je, jh;
            if (port >= 0) {
                // Port: characteristic flux against a zero incident field.
                je = em;
                jh = hm;
            } else if (nbr < 0) {
                // PEC ghost: [E] = 2.E_tangential, [H] = 2.H_normal.
                float edn = dot(nrm, em);
                float hdn = dot(nrm, hm);
                je = 2.0f * (em - edn * nrm);
                jh = 2.0f * (hdn * nrm);
            } else {
                int vj =
                    face_nodes[nbrlf * NFP + face_perm[ff * NFP + m]];
                int nbb = nbr * NP * 6;
                je = em - (float3)(y[nbb + vj * 6],
                                   y[nbb + vj * 6 + 1],
                                   y[nbb + vj * 6 + 2]);
                jh = hm - (float3)(y[nbb + vj * 6 + 3],
                                   y[nbb + vj * 6 + 4],
                                   y[nbb + vj * 6 + 5]);
            }
            float3 pe = cross3(nrm, cross3(nrm, je));
            float3 ph = cross3(nrm, cross3(nrm, jh));
            float3 fe = -cross3(nrm, jh) + a * pe;
            float3 fh = cross3(nrm, je) + a * ph;
            for (int i = 0; i < NP; i++) {
                float w = coef * lift[i * COLS + f * NFP + m];
                de[i * 3 + 0] += w * fe.x;
                de[i * 3 + 1] += w * fe.y;
                de[i * 3 + 2] += w * fe.z;
                dh[i * 3 + 0] += w * fh.x;
                dh[i * 3 + 1] += w * fh.y;
                dh[i * 3 + 2] += w * fh.z;
            }
        }
    }

    // Per-element materials.
    for (int node = 0; node < NP; node++) {
        for (int c = 0; c < 3; c++) {
            dy[base + node * 6 + c] = inv_eps[e * 3 + c] * de[node * 3 + c]
                - sigma_eps[e * 3 + c] * ee[node * 3 + c];
            dy[base + node * 6 + 3 + c] = inv_mu[e * 3 + c]
                    * dh[node * 3 + c]
                - sigma_mu[e * 3 + c] * hh[node * 3 + c];
        }
    }
}
