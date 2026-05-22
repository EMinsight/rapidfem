// DG Maxwell RHS operator — dy/dt = A.y. One work-group per element block.
//
// NP, NFP, COLS, EPG are #define'd by the host before this source is built.
// A work-group processes EPG elements; its work-items are NP-per-element DG
// nodes, so the group has EPG*NP work-items. The element's field state and
// the curl/flux accumulators live in __local memory, shared by the EPG*NP
// work-items, so each work-item holds only a handful of private floats and
// the register-spill cliff of the old one-work-item-per-element kernel is
// gone. Numerically a faithful port of the CPU `apply_element`.

inline float3 cross3(float3 a, float3 b) {
    return (float3)(a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x);
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
    // Work-item layout: lid = le*NP + node, with `le` the element's slot in
    // this group's EPG-element block and `node` its DG node.
    const int lid = get_local_id(0);
    const int le = lid / NP;
    const int node = lid % NP;
    const int e = (int)get_group_id(0) * EPG + le;
    const int active = (e < n_elem);

    // Per-element shared state. The four NP-sized field/accumulator arrays
    // are the bulk of the element's working set; the flux scratch holds the
    // numerical flux for each of the 4*NFP face nodes.
    __local float ee[EPG][3 * NP];
    __local float hh[EPG][3 * NP];
    __local float de[EPG][3 * NP];
    __local float dh[EPG][3 * NP];
    __local float flx_e[EPG][3 * 4 * NFP];
    __local float flx_h[EPG][3 * 4 * NFP];

    // Phase 0: stage this node's field state into __local (AoS gather).
    if (active) {
        const int base = e * NP * 6;
        ee[le][node * 3 + 0] = y[base + node * 6 + 0];
        ee[le][node * 3 + 1] = y[base + node * 6 + 1];
        ee[le][node * 3 + 2] = y[base + node * 6 + 2];
        hh[le][node * 3 + 0] = y[base + node * 6 + 3];
        hh[le][node * 3 + 1] = y[base + node * 6 + 4];
        hh[le][node * 3 + 2] = y[base + node * 6 + 5];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 1: this work-item's volume curl. dE = curl(H), dH = -curl(E).
    // Each work-item owns exactly one node; the reference and physical
    // derivatives are tiny 9-float private scratch, no NP-sized arrays.
    if (active) {
        global const float* jinv_e = jinv + e * 9;
        // Reference derivatives of H and E at this node: rd[k*6 + h*3 + c],
        // h selecting the H (0) or E (1) field.
        float rd[18];
        for (int k = 0; k < 3; k++) {
            constant const float* d =
                (k == 0) ? diff_r : ((k == 1) ? diff_s : diff_t);
            float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f;
            float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f;
            for (int j = 0; j < NP; j++) {
                float dij = d[node * NP + j];
                h0 += dij * hh[le][j * 3 + 0];
                h1 += dij * hh[le][j * 3 + 1];
                h2 += dij * hh[le][j * 3 + 2];
                e0 += dij * ee[le][j * 3 + 0];
                e1 += dij * ee[le][j * 3 + 1];
                e2 += dij * ee[le][j * 3 + 2];
            }
            rd[k * 6 + 0] = h0;
            rd[k * 6 + 1] = h1;
            rd[k * 6 + 2] = h2;
            rd[k * 6 + 3] = e0;
            rd[k * 6 + 4] = e1;
            rd[k * 6 + 5] = e2;
        }
        // Physical derivatives pd[phys*6 + h*3 + c]; jinv_e[k*3+p].
        float pd[18];
        for (int p = 0; p < 3; p++) {
            float j0 = jinv_e[0 * 3 + p];
            float j1 = jinv_e[1 * 3 + p];
            float j2 = jinv_e[2 * 3 + p];
            for (int c = 0; c < 6; c++)
                pd[p * 6 + c] =
                    j0 * rd[0 * 6 + c] + j1 * rd[1 * 6 + c]
                    + j2 * rd[2 * 6 + c];
        }
        // curl_x = d(Fz)/dy - d(Fy)/dz, cyclic. dH = -curl(E).
        de[le][node * 3 + 0] = pd[1 * 6 + 2] - pd[2 * 6 + 1];
        de[le][node * 3 + 1] = pd[2 * 6 + 0] - pd[0 * 6 + 2];
        de[le][node * 3 + 2] = pd[0 * 6 + 1] - pd[1 * 6 + 0];
        dh[le][node * 3 + 0] = -(pd[1 * 6 + 5] - pd[2 * 6 + 4]);
        dh[le][node * 3 + 1] = -(pd[2 * 6 + 3] - pd[0 * 6 + 5]);
        dh[le][node * 3 + 2] = -(pd[0 * 6 + 4] - pd[1 * 6 + 3]);
    }

    // Phase 2: the numerical flux. Cooperative over the 4*NFP face nodes —
    // each work-item strides through the face-node slots of its element,
    // computing fe/fh into the flux scratch. The barrier before this phase
    // is not needed (Phase 2 reads only ee/hh, written in Phase 0), but the
    // one after Phase 1 already separated the curl writes from the lift sum
    // below; this loop only writes flx_e/flx_h.
    if (active) {
        for (int s = node; s < 4 * NFP; s += NP) {
            const int f = s / NFP;
            const int m = s % NFP;
            const int ff = e * 4 + f;
            float3 nrm = (float3)(face_normal[ff * 3],
                                  face_normal[ff * 3 + 1],
                                  face_normal[ff * 3 + 2]);
            int nbr = face_neighbor[ff];
            int nbrlf = face_nbr_local[ff];
            int port = face_port[ff];
            float a = (port >= 0) ? 1.0f : flux_alpha;
            int vi = face_nodes[f * NFP + m];
            float3 em = (float3)(ee[le][vi * 3], ee[le][vi * 3 + 1],
                                 ee[le][vi * 3 + 2]);
            float3 hm = (float3)(hh[le][vi * 3], hh[le][vi * 3 + 1],
                                 hh[le][vi * 3 + 2]);
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
            float coef = 0.5f * face_fscale[ff];
            float3 fe = coef * (-cross3(nrm, jh) + a * pe);
            float3 fh = coef * (cross3(nrm, je) + a * ph);
            flx_e[le][s * 3 + 0] = fe.x;
            flx_e[le][s * 3 + 1] = fe.y;
            flx_e[le][s * 3 + 2] = fe.z;
            flx_h[le][s * 3 + 0] = fh.x;
            flx_h[le][s * 3 + 1] = fh.y;
            flx_h[le][s * 3 + 2] = fh.z;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: each volume node sums its lift contribution over the 4*NFP
    // face-node fluxes, then applies the per-element materials and writes
    // the result. `coef` is already folded into flx_e/flx_h above.
    if (active) {
        float dex = de[le][node * 3 + 0];
        float dey = de[le][node * 3 + 1];
        float dez = de[le][node * 3 + 2];
        float dhx = dh[le][node * 3 + 0];
        float dhy = dh[le][node * 3 + 1];
        float dhz = dh[le][node * 3 + 2];
        for (int s = 0; s < 4 * NFP; s++) {
            float w = lift[node * COLS + s];
            dex += w * flx_e[le][s * 3 + 0];
            dey += w * flx_e[le][s * 3 + 1];
            dez += w * flx_e[le][s * 3 + 2];
            dhx += w * flx_h[le][s * 3 + 0];
            dhy += w * flx_h[le][s * 3 + 1];
            dhz += w * flx_h[le][s * 3 + 2];
        }
        const int base = e * NP * 6;
        dy[base + node * 6 + 0] =
            inv_eps[e * 3 + 0] * dex - sigma_eps[e * 3 + 0] * ee[le][node * 3 + 0];
        dy[base + node * 6 + 1] =
            inv_eps[e * 3 + 1] * dey - sigma_eps[e * 3 + 1] * ee[le][node * 3 + 1];
        dy[base + node * 6 + 2] =
            inv_eps[e * 3 + 2] * dez - sigma_eps[e * 3 + 2] * ee[le][node * 3 + 2];
        dy[base + node * 6 + 3] =
            inv_mu[e * 3 + 0] * dhx - sigma_mu[e * 3 + 0] * hh[le][node * 3 + 0];
        dy[base + node * 6 + 4] =
            inv_mu[e * 3 + 1] * dhy - sigma_mu[e * 3 + 1] * hh[le][node * 3 + 1];
        dy[base + node * 6 + 5] =
            inv_mu[e * 3 + 2] * dhz - sigma_mu[e * 3 + 2] * hh[le][node * 3 + 2];
    }
}
