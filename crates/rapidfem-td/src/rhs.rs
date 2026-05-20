//! DG Maxwell RHS operator.
//!
//! The semi-discrete DG form of the vacuum Maxwell curl equations
//! (`∂E/∂t = ∇×H`, `∂H/∂t = -∇×E`) splits per element into a volume term
//! (the physical curl) and a surface term (the numerical flux). This module
//! builds those operators; the volume curl is assembled and validated first.
//!
//! Per-element fields are stored node-major: `field[node*3 + component]`,
//! with components ordered `x, y, z`.

use crate::dg_basis::ReferenceElement;
use crate::geom_factors::{GeometricFactors, all_geometric_factors};
use rapidfem_core::mesh::Mesh;
use rapidfem_core::topology::FaceTopology;
use rayon::prelude::*;
use std::sync::Mutex;

/// Physical curl of a vector field on a single element.
///
/// `field` holds `3·Np` values (`field[node*3 + comp]`); the result has the
/// same layout and contains `∇×field` sampled at the element nodes. This is
/// the allocating wrapper around [`element_curl_into`] — the hot path uses
/// the scratch-buffer form.
pub fn element_curl(
    re: &ReferenceElement,
    gf: &GeometricFactors,
    field: &[f64],
) -> Vec<f64> {
    let n = re.n_nodes;
    let mut out = vec![0.0; 3 * n];
    let mut rd = vec![0.0; 3 * n];
    let mut pd = vec![0.0; 9 * n];
    element_curl_into(re, gf, field, &mut out, &mut rd, &mut pd);
    out
}

/// Physical curl, writing into caller-provided buffers — no allocation.
///
/// `out` (`3·Np`) receives `∇×field`; `rd` (`3·Np`) and `pd` (`9·Np`) are
/// scratch. `rd[k·n+i]` holds the `ξ_k` reference derivative; `pd[(p·3+c)·n+i]`
/// holds `∂(field_c)/∂x_p`.
fn element_curl_into(
    re: &ReferenceElement,
    gf: &GeometricFactors,
    field: &[f64],
    out: &mut [f64],
    rd: &mut [f64],
    pd: &mut [f64],
) {
    let n = re.n_nodes;
    debug_assert_eq!(field.len(), 3 * n);
    let dref = [&re.diff_r, &re.diff_s, &re.diff_t];

    for comp in 0..3 {
        // Reference derivatives of this component: rd[k·n + i].
        for (k, d) in dref.iter().enumerate() {
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += d[i * n + j] * field[j * 3 + comp];
                }
                rd[k * n + i] = acc;
            }
        }
        // Combine via the metric into physical derivatives pd[(p·3+comp)·n+i].
        for phys in 0..3 {
            let jinv = [
                gf.jacobian_inv[0][phys],
                gf.jacobian_inv[1][phys],
                gf.jacobian_inv[2][phys],
            ];
            let base = (phys * 3 + comp) * n;
            for i in 0..n {
                pd[base + i] = jinv[0] * rd[i]
                    + jinv[1] * rd[n + i]
                    + jinv[2] * rd[2 * n + i];
            }
        }
    }

    // curl_x = ∂Fz/∂y - ∂Fy/∂z, and cyclic; pd index = (phys·3 + comp)·n + i.
    let idx = |phys: usize, comp: usize, i: usize| (phys * 3 + comp) * n + i;
    for i in 0..n {
        out[i * 3] = pd[idx(1, 2, i)] - pd[idx(2, 1, i)];
        out[i * 3 + 1] = pd[idx(2, 0, i)] - pd[idx(0, 2, i)];
        out[i * 3 + 2] = pd[idx(0, 1, i)] - pd[idx(1, 0, i)];
    }
}

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Per (element, local face) flux data.
struct FaceInfo {
    /// Outward unit normal.
    normal: [f64; 3],
    /// Surface scaling `2·area_phys / |det J|` (the lift assumes a 1/2 reference face).
    fscale: f64,
    /// Neighbour element, or `usize::MAX` on a PEC boundary.
    neighbor: usize,
    /// Neighbour local face.
    neighbor_local_face: usize,
    /// `perm[m]` = neighbour face-node local index matching this face's node `m`.
    perm: Vec<usize>,
}

/// Per-thread working buffers for [`MaxwellOperator::apply_element`] — the
/// fixed-size scratch, allocated once and reused so the operator hot path
/// performs no per-element heap allocation.
struct Scratch {
    /// Element E / H fields, deinterleaved (`3·Np` each).
    ee: Vec<f64>,
    hh: Vec<f64>,
    /// Curl results dE / dH (`3·Np` each).
    de: Vec<f64>,
    dh: Vec<f64>,
    /// `element_curl_into` scratch — reference (`3·Np`) and physical (`9·Np`)
    /// derivatives.
    rd: Vec<f64>,
    pd: Vec<f64>,
}

impl Scratch {
    fn new(np: usize) -> Self {
        Scratch {
            ee: vec![0.0; 3 * np],
            hh: vec![0.0; 3 * np],
            de: vec![0.0; 3 * np],
            dh: vec![0.0; 3 * np],
            rd: vec![0.0; 3 * np],
            pd: vec![0.0; 9 * np],
        }
    }

    /// A non-allocating placeholder — the value swapped in when a real
    /// `Scratch` is returned to the pool.
    fn empty() -> Self {
        Scratch {
            ee: Vec::new(),
            hh: Vec::new(),
            de: Vec::new(),
            dh: Vec::new(),
            rd: Vec::new(),
            pd: Vec::new(),
        }
    }
}

/// Checkout handle for a pooled [`Scratch`]; returns it to the pool on drop,
/// so steady-state `apply` calls allocate no scratch at all.
struct ScratchGuard<'a> {
    pool: &'a Mutex<Vec<Scratch>>,
    scratch: Scratch,
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        let s = std::mem::replace(&mut self.scratch, Scratch::empty());
        self.pool.lock().unwrap().push(s);
    }
}

/// Per-job working buffers for the sparse assembly — reused across every
/// element a rayon worker folds, so `assemble_sparse`'s element loop
/// allocates nothing beyond the geometric growth of the output accumulators.
struct SparseFragment {
    /// Global probe vector — one DOF set to 1 at a time.
    probe: Vec<f64>,
    /// Element output block (`stride`).
    out: Vec<f64>,
    /// Operator scratch.
    scratch: Scratch,
    /// `(local_row, global_col, value)` triples for one element — cleared
    /// and refilled per element; pre-sized to the worst-case stencil count.
    entries: Vec<(usize, usize, f64)>,
    /// Accumulated CSR fragment for this job, in element order.
    col_idx: Vec<usize>,
    values: Vec<f64>,
    row_len: Vec<usize>,
}

impl SparseFragment {
    fn new(n: usize, stride: usize, np: usize) -> Self {
        SparseFragment {
            probe: vec![0.0; n],
            out: vec![0.0; stride],
            scratch: Scratch::new(np),
            // ≤ 5 stencil columns × stride columns × stride rows nonzeros.
            entries: Vec::with_capacity(5 * stride * stride),
            col_idx: Vec::new(),
            values: Vec::new(),
            row_len: Vec::new(),
        }
    }
}

/// Per-element electromagnetic material — diagonal relative permittivity /
/// permeability tensors and conductivity, in the solver's normalised units
/// (`c = ε₀ = μ₀ = 1`). Diagonal tensors cover isotropic and uniaxial /
/// biaxial media (and the uniaxial PML); fully off-diagonal tensors are a
/// future extension.
#[derive(Clone, Copy, Debug)]
pub struct ElemMaterial {
    /// Diagonal relative permittivity `(ε_x, ε_y, ε_z)`.
    pub eps: [f64; 3],
    /// Diagonal relative permeability `(μ_x, μ_y, μ_z)`.
    pub mu: [f64; 3],
    /// Electric conductivity `σ` (Ohmic loss).
    pub sigma: f64,
    /// Magnetic conductivity `σ*` — the magnetic-loss term. Setting
    /// `σ*/μ = σ/ε` gives an impedance-matched absorbing layer (no reflection
    /// at normal incidence).
    pub sigma_m: f64,
}

impl ElemMaterial {
    /// Vacuum — `ε = μ = 1`, no loss.
    pub const VACUUM: ElemMaterial = ElemMaterial {
        eps: [1.0; 3],
        mu: [1.0; 3],
        sigma: 0.0,
        sigma_m: 0.0,
    };

    /// An isotropic, lossless material from scalar `ε_r`, `μ_r`, `σ`.
    pub fn isotropic(eps: f64, mu: f64, sigma: f64) -> Self {
        ElemMaterial { eps: [eps; 3], mu: [mu; 3], sigma, sigma_m: 0.0 }
    }

    /// An impedance-matched absorbing material: `σ*/μ = σ/ε = nu`, so the
    /// wave is absorbed with no reflection at the layer interface.
    pub fn matched_absorber(eps: f64, mu: f64, nu: f64) -> Self {
        ElemMaterial {
            eps: [eps; 3],
            mu: [mu; 3],
            sigma: nu * eps,
            sigma_m: nu * mu,
        }
    }
}

/// Semi-discrete DG operator for the Maxwell curl equations on a tetrahedral
/// mesh with PEC outer walls and per-element materials.
///
/// State layout: `y[(e*Np + node)*6 + comp]`, `comp` 0..3 = E, 3..6 = H.
/// `apply` evaluates `dy/dt`. The numerical flux blends central (`alpha = 0`,
/// energy-conserving) and upwind (`alpha = 1`, dissipates the discontinuous
/// spurious modes).
pub struct MaxwellOperator {
    re: ReferenceElement,
    n_elem: usize,
    geom: Vec<GeometricFactors>,
    /// 4 faces per element, flattened: `faces[e*4 + f]`.
    faces: Vec<FaceInfo>,
    /// Upwind blend: 0 = central, 1 = full upwind.
    flux_alpha: f64,
    /// Per-element diagonal `1/ε`, `1/μ`, `σ/ε` (electric), `σ*/μ` (magnetic).
    inv_eps: Vec<[f64; 3]>,
    inv_mu: Vec<[f64; 3]>,
    sigma_eps: Vec<[f64; 3]>,
    sigma_mu: Vec<[f64; 3]>,
    /// Reusable per-thread scratch buffers — keeps `apply` allocation-free
    /// after the first call (see [`Scratch`]).
    scratch_pool: Mutex<Vec<Scratch>>,
}

impl MaxwellOperator {
    /// Build a vacuum operator (`ε = μ = 1`, `σ = 0`).
    pub fn new(mesh: &Mesh, order: usize, flux_alpha: f64) -> Self {
        let vacuum = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        Self::new_with_materials(mesh, order, flux_alpha, &vacuum)
    }

    /// Build the operator with per-element materials and the given upwind
    /// blend (`flux_alpha` in `[0, 1]`).
    pub fn new_with_materials(
        mesh: &Mesh,
        order: usize,
        flux_alpha: f64,
        materials: &[ElemMaterial],
    ) -> Self {
        let re = ReferenceElement::new(order);
        let geom = all_geometric_factors(mesh);
        let topo = FaceTopology::build(mesh);
        let n_elem = mesh.n_tets();

        let face_coords = |e: usize, f: usize| -> Vec<[f64; 3]> {
            re.face_nodes[f]
                .iter()
                .map(|&vi| geom[e].map(re.nodes[vi]))
                .collect()
        };

        let mut faces = Vec::with_capacity(n_elem * 4);
        for e in 0..n_elem {
            for f in 0..4 {
                let df = topo.face(e, f);
                let fscale = 2.0 * df.area / geom[e].det.abs();
                let perm = if df.neighbor == usize::MAX {
                    Vec::new()
                } else {
                    let here = face_coords(e, f);
                    let there =
                        face_coords(df.neighbor, df.neighbor_local_face);
                    here.iter()
                        .map(|p| {
                            let (mut best, mut bm) = (f64::MAX, 0);
                            for (m2, q) in there.iter().enumerate() {
                                let d = (p[0] - q[0]).powi(2)
                                    + (p[1] - q[1]).powi(2)
                                    + (p[2] - q[2]).powi(2);
                                if d < best {
                                    best = d;
                                    bm = m2;
                                }
                            }
                            assert!(best < 1e-18, "unmatched face node");
                            bm
                        })
                        .collect()
                };
                faces.push(FaceInfo {
                    normal: df.normal,
                    fscale,
                    neighbor: df.neighbor,
                    neighbor_local_face: df.neighbor_local_face,
                    perm,
                });
            }
        }
        assert_eq!(materials.len(), n_elem, "one material per element");
        let recip = |v: [f64; 3]| [1.0 / v[0], 1.0 / v[1], 1.0 / v[2]];
        let inv_eps: Vec<[f64; 3]> =
            materials.iter().map(|m| recip(m.eps)).collect();
        let inv_mu: Vec<[f64; 3]> =
            materials.iter().map(|m| recip(m.mu)).collect();
        let sigma_eps: Vec<[f64; 3]> = materials
            .iter()
            .map(|m| [m.sigma / m.eps[0], m.sigma / m.eps[1], m.sigma / m.eps[2]])
            .collect();
        let sigma_mu: Vec<[f64; 3]> = materials
            .iter()
            .map(|m| {
                [
                    m.sigma_m / m.mu[0],
                    m.sigma_m / m.mu[1],
                    m.sigma_m / m.mu[2],
                ]
            })
            .collect();
        MaxwellOperator {
            re,
            n_elem,
            geom,
            faces,
            flux_alpha,
            inv_eps,
            inv_mu,
            sigma_eps,
            sigma_mu,
            scratch_pool: Mutex::new(Vec::new()),
        }
    }

    /// Degrees of freedom, `6 · Np · n_elem`.
    pub fn n_dof(&self) -> usize {
        6 * self.re.n_nodes * self.n_elem
    }

    /// Global DOF index for a field component at the mesh node nearest
    /// `point` — the hook for a soft source or a field probe.
    /// `field`: 0 = E, 1 = H. `comp`: 0 = x, 1 = y, 2 = z.
    pub fn nearest_node_dof(
        &self,
        point: [f64; 3],
        field: usize,
        comp: usize,
    ) -> usize {
        let np = self.re.n_nodes;
        let (mut best_d, mut best) = (f64::MAX, 0);
        for e in 0..self.n_elem {
            for node in 0..np {
                let p = self.geom[e].map(self.re.nodes[node]);
                let d = (p[0] - point[0]).powi(2)
                    + (p[1] - point[1]).powi(2)
                    + (p[2] - point[2]).powi(2);
                if d < best_d {
                    best_d = d;
                    best = (e * np + node) * 6 + field * 3 + comp;
                }
            }
        }
        best
    }

    /// Physical coordinates of every DG node — `n_elem·Np` points in state
    /// order, `point[e*Np + node]`. The hook for a field export.
    pub fn node_coords(&self) -> Vec<[f64; 3]> {
        let np = self.re.n_nodes;
        let mut pts = Vec::with_capacity(self.n_elem * np);
        for e in 0..self.n_elem {
            for node in 0..np {
                pts.push(self.geom[e].map(self.re.nodes[node]));
            }
        }
        pts
    }

    /// The four reference-node local indices at the tet corners, ordered
    /// `(0,0,0), (1,0,0), (0,1,0), (0,0,1)` — the connectivity hook for a
    /// linear-tetrahedron VTK export.
    pub fn corner_local_nodes(&self) -> [usize; 4] {
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        corners.map(|c| {
            self.re
                .nodes
                .iter()
                .position(|n| {
                    (n[0] - c[0]).abs() < 1e-12
                        && (n[1] - c[1]).abs() < 1e-12
                        && (n[2] - c[2]).abs() < 1e-12
                })
                .expect("reference element carries the tet-corner nodes")
        })
    }

    /// Evaluate `dy/dt = A·y`, allocating the result. See [`apply_into`](Self::apply_into)
    /// for the allocation-free form.
    pub fn apply(&self, y: &[f64]) -> Vec<f64> {
        let mut dy = vec![0.0; self.n_dof()];
        self.apply_into(y, &mut dy);
        dy
    }

    /// Evaluate `dy/dt = A·y` into the caller's buffer — the allocation-free
    /// hot path. The per-element work is independent (each element writes
    /// only its own slice of `dy`), so it runs in parallel across cores;
    /// every worker reuses a pooled [`Scratch`], so after the first call
    /// this performs no heap allocation at all.
    pub fn apply_into(&self, y: &[f64], dy: &mut [f64]) {
        debug_assert_eq!(dy.len(), self.n_dof());
        let stride = self.re.n_nodes * 6;
        let np = self.re.n_nodes;
        dy.par_chunks_mut(stride).enumerate().for_each_init(
            || self.checkout_scratch(np),
            |guard, (e, out)| {
                self.apply_element(e, y, out, &mut guard.scratch)
            },
        );
    }

    /// Take a [`Scratch`] from the pool, allocating one only on first use.
    fn checkout_scratch(&self, np: usize) -> ScratchGuard<'_> {
        let scratch = self
            .scratch_pool
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| Scratch::new(np));
        ScratchGuard { pool: &self.scratch_pool, scratch }
    }

    /// Compute element `e`'s block of `dy = A·y` into `out` — its `Np·6`
    /// contiguous slice. `s` supplies the reusable per-thread working
    /// buffers, so this allocates nothing.
    fn apply_element(
        &self,
        e: usize,
        y: &[f64],
        out: &mut [f64],
        s: &mut Scratch,
    ) {
        let np = self.re.n_nodes;
        let nfp = self.re.n_face_nodes;
        let cols = 4 * nfp;
        let base = e * np * 6;
        for node in 0..np {
            for c in 0..3 {
                s.ee[node * 3 + c] = y[base + node * 6 + c];
                s.hh[node * 3 + c] = y[base + node * 6 + 3 + c];
            }
        }

        // Volume term:  dE = ∇×H,  dH = -∇×E.
        element_curl_into(
            &self.re, &self.geom[e], &s.hh, &mut s.de, &mut s.rd, &mut s.pd,
        );
        element_curl_into(
            &self.re, &self.geom[e], &s.ee, &mut s.dh, &mut s.rd, &mut s.pd,
        );
        for v in s.dh.iter_mut() {
            *v = -*v;
        }

        // Surface term — central flux.
        for f in 0..4 {
            let fi = &self.faces[e * 4 + f];
            let n = fi.normal;
            let coef = 0.5 * fi.fscale;
            for m in 0..nfp {
                let vi = self.re.face_nodes[f][m];
                let em = [s.ee[vi * 3], s.ee[vi * 3 + 1], s.ee[vi * 3 + 2]];
                let hm = [s.hh[vi * 3], s.hh[vi * 3 + 1], s.hh[vi * 3 + 2]];
                // Jumps [E] = E⁻-E⁺, [H] = H⁻-H⁺.
                let (je, jh) = if fi.neighbor == usize::MAX {
                    // PEC ghost: [E] = 2·E_tangential, [H] = 2·H_normal.
                    let edn = dot3(n, em);
                    let hdn = dot3(n, hm);
                    (
                        [
                            2.0 * (em[0] - edn * n[0]),
                            2.0 * (em[1] - edn * n[1]),
                            2.0 * (em[2] - edn * n[2]),
                        ],
                        [
                            2.0 * hdn * n[0],
                            2.0 * hdn * n[1],
                            2.0 * hdn * n[2],
                        ],
                    )
                } else {
                    let vj = self.re.face_nodes[fi.neighbor_local_face]
                        [fi.perm[m]];
                    let nbb = fi.neighbor * np * 6;
                    (
                        [
                            em[0] - y[nbb + vj * 6],
                            em[1] - y[nbb + vj * 6 + 1],
                            em[2] - y[nbb + vj * 6 + 2],
                        ],
                        [
                            hm[0] - y[nbb + vj * 6 + 3],
                            hm[1] - y[nbb + vj * 6 + 4],
                            hm[2] - y[nbb + vj * 6 + 5],
                        ],
                    )
                };
                let nxjh = cross3(n, jh);
                let nxje = cross3(n, je);
                // Upwind penalty n̂×(n̂×[·]) = -[·]_tangential — dissipative,
                // damps the discontinuous spurious modes.
                let a = self.flux_alpha;
                let pe = cross3(n, cross3(n, je));
                let ph = cross3(n, cross3(n, jh));
                for i in 0..np {
                    let w = coef * self.re.lift[i * cols + f * nfp + m];
                    for c in 0..3 {
                        s.de[i * 3 + c] += w * (-nxjh[c] + a * pe[c]);
                        s.dh[i * 3 + c] += w * (nxje[c] + a * ph[c]);
                    }
                }
            }
        }

        // Apply per-element materials: ∂E/∂t = (1/ε)(∇×H + flux) - (σ/ε)E,
        // ∂H/∂t = (1/μ)(-∇×E + flux).
        let (ie, im) = (self.inv_eps[e], self.inv_mu[e]);
        let (se, sm) = (self.sigma_eps[e], self.sigma_mu[e]);
        for node in 0..np {
            for c in 0..3 {
                out[node * 6 + c] =
                    ie[c] * s.de[node * 3 + c] - se[c] * s.ee[node * 3 + c];
                out[node * 6 + 3 + c] =
                    im[c] * s.dh[node * 3 + c] - sm[c] * s.hh[node * 3 + c];
            }
        }
    }

    /// Assemble the operator as a dense `N×N` row-major matrix by applying it
    /// to each unit vector. For validation on small meshes.
    pub fn assemble_dense(&self) -> Vec<f64> {
        let n = self.n_dof();
        let mut a = vec![0.0; n * n];
        let mut ej = vec![0.0; n];
        for j in 0..n {
            ej[j] = 1.0;
            let col = self.apply(&ej);
            for (i, &v) in col.iter().enumerate() {
                a[i * n + j] = v;
            }
            ej[j] = 0.0;
        }
        a
    }

    /// Dense block-diagonal energy mass `M̃` — the material-weighted field
    /// energy `yᵀM̃y = ∫(ε|E|² + μ|H|²)`: per element a copy of
    /// `|det J_e|·M_ref`, scaled by `ε` on the E components and `μ` on the H
    /// components.
    pub fn assemble_energy_mass(&self) -> Vec<f64> {
        let np = self.re.n_nodes;
        let n = self.n_dof();
        let mut m = vec![0.0; n * n];
        for e in 0..self.n_elem {
            let scale = self.geom[e].det.abs();
            let eps = self.inv_eps[e].map(|x| 1.0 / x);
            let mu = self.inv_mu[e].map(|x| 1.0 / x);
            let base = e * np * 6;
            for ni in 0..np {
                for nj in 0..np {
                    let mij = scale * self.re.mass[ni * np + nj];
                    for c in 0..6 {
                        let w = if c < 3 { eps[c] } else { mu[c - 3] };
                        m[(base + ni * 6 + c) * n + (base + nj * 6 + c)] =
                            w * mij;
                    }
                }
            }
        }
        m
    }
}

/// Compressed-sparse-row matrix — the explicit state-space operator `A`.
pub struct CsrMatrix {
    /// Dimension.
    pub n: usize,
    /// Row offsets, length `n + 1`.
    pub row_ptr: Vec<usize>,
    /// Column index of each stored entry.
    pub col_idx: Vec<usize>,
    /// Stored values.
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Number of stored nonzeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse matrix-vector product `A·x`.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.n];
        for i in 0..self.n {
            let mut acc = 0.0;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                acc += self.values[k] * x[self.col_idx[k]];
            }
            y[i] = acc;
        }
        y
    }
}

impl MaxwellOperator {
    /// Assemble the operator as an explicit sparse CSR matrix — the
    /// state-space `A` — **without ever densifying**.
    ///
    /// The DG operator couples each element only to itself and its (≤4) face
    /// neighbours, so row block `e` is found by probing just that small
    /// column stencil with unit vectors and reading element `e`'s output
    /// block. Element blocks are independent and assemble in parallel.
    /// Memory is `O(nnz)`, not `O(N²)`, so this scales to production meshes
    /// where [`assemble_dense`](Self::assemble_dense) cannot.
    /// Sorted column-element stencil of row block `e` — itself plus every
    /// distinct face neighbour. Returns the fixed `[usize; 5]` array and the
    /// number of entries used; never allocates.
    fn element_stencil(&self, e: usize) -> ([usize; 5], usize) {
        let mut s = [0usize; 5];
        s[0] = e;
        let mut count = 1;
        for f in 0..4 {
            let nb = self.faces[e * 4 + f].neighbor;
            if nb != usize::MAX && !s[..count].contains(&nb) {
                s[count] = nb;
                count += 1;
            }
        }
        s[..count].sort_unstable();
        (s, count)
    }

    pub fn assemble_sparse(&self) -> CsrMatrix {
        let stride = self.re.n_nodes * 6;
        let np = self.re.n_nodes;
        let n = self.n_dof();

        // Each rayon worker folds a contiguous run of elements into one
        // `SparseFragment`, reusing its buffers across every element — the
        // element loop allocates nothing. `with_min_len` forces chunks
        // coarse enough for that reuse to pay off while still giving the
        // thread pool plenty of independent work. `fold` keeps the
        // fragments in element order, so concatenating them yields a
        // row-ordered CSR.
        let min_len =
            (self.n_elem / (4 * rayon::current_num_threads())).max(1);
        let frags: Vec<SparseFragment> = (0..self.n_elem)
            .into_par_iter()
            .with_min_len(min_len)
            .fold(
                || SparseFragment::new(n, stride, np),
                |mut f, e| {
                    let (sten, ns) = self.element_stencil(e);
                    f.entries.clear();
                    for &c in &sten[..ns] {
                        for jl in 0..stride {
                            let j = c * stride + jl;
                            f.probe[j] = 1.0;
                            f.out.iter_mut().for_each(|v| *v = 0.0);
                            self.apply_element(
                                e,
                                &f.probe,
                                &mut f.out,
                                &mut f.scratch,
                            );
                            f.probe[j] = 0.0;
                            for il in 0..stride {
                                let v = f.out[il];
                                if v != 0.0 {
                                    f.entries.push((il, j, v));
                                }
                            }
                        }
                    }
                    // Group by local row, columns ascending within a row —
                    // sorting on the full key needs no stable-sort scratch.
                    f.entries.sort_unstable_by_key(|&(il, j, _)| (il, j));
                    let mut cursor = 0;
                    for il in 0..stride {
                        let mut cnt = 0;
                        while cursor < f.entries.len()
                            && f.entries[cursor].0 == il
                        {
                            f.col_idx.push(f.entries[cursor].1);
                            f.values.push(f.entries[cursor].2);
                            cnt += 1;
                            cursor += 1;
                        }
                        f.row_len.push(cnt);
                    }
                    f
                },
            )
            .collect();

        // Concatenate the per-job fragments — already in element order.
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        let mut acc = 0;
        for f in &frags {
            for &l in &f.row_len {
                acc += l;
                row_ptr.push(acc);
            }
            col_idx.extend_from_slice(&f.col_idx);
            values.extend_from_slice(&f.values);
        }
        CsrMatrix { n, row_ptr, col_idx, values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curl_of_polynomial_field_is_exact() {
        // On a sheared physical element, the discrete curl reproduces the
        // analytic curl of a degree-2 vector field exactly at the nodes.
        let v = [
            [0.3, -0.2, 0.1],
            [1.4, 0.1, -0.2],
            [0.2, 1.3, 0.4],
            [-0.1, 0.2, 1.6],
        ];
        let gf = GeometricFactors::for_tet(&v);
        let re = ReferenceElement::new(2);
        let n = re.n_nodes;
        let pn: Vec<[f64; 3]> = re.nodes.iter().map(|&xi| gf.map(xi)).collect();

        // F = (y·z, 2·z·x, 3·x·y)  ⇒  ∇×F = (x, -2y, z).
        let mut field = vec![0.0; 3 * n];
        for (i, p) in pn.iter().enumerate() {
            field[i * 3] = p[1] * p[2];
            field[i * 3 + 1] = 2.0 * p[2] * p[0];
            field[i * 3 + 2] = 3.0 * p[0] * p[1];
        }

        let curl = element_curl(&re, &gf, &field);
        for (i, p) in pn.iter().enumerate() {
            let want = [p[0], -2.0 * p[1], p[2]];
            for c in 0..3 {
                assert!(
                    (curl[i * 3 + c] - want[c]).abs() < 1e-9,
                    "node {i} comp {c}: got {}, want {}",
                    curl[i * 3 + c],
                    want[c]
                );
            }
        }
    }

    #[test]
    fn curl_of_constant_field_vanishes() {
        let gf = GeometricFactors::for_tet(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let re = ReferenceElement::new(3);
        let field = vec![0.7; 3 * re.n_nodes];
        let curl = element_curl(&re, &gf, &field);
        assert!(curl.iter().all(|c| c.abs() < 1e-10));
    }

    #[test]
    fn central_flux_operator_conserves_energy() {
        // The central-flux DG Maxwell operator is exactly energy-conserving:
        // M̃·A must be skew-symmetric. This validates the flux signs, the
        // surface scaling and the PEC boundary treatment all at once.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        // Central flux (alpha = 0) is the energy-conserving case.
        let op = MaxwellOperator::new(&mesh, 2, 0.0);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mm = op.assemble_energy_mass();

        // ma = M̃ · A
        let mut ma = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let mik = mm[i * n + k];
                if mik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    ma[i * n + j] += mik * a[k * n + j];
                }
            }
        }

        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for p in 0..n {
            for q in 0..n {
                worst = worst.max((ma[p * n + q] + ma[q * n + p]).abs());
                scale = scale.max(ma[p * n + q].abs());
            }
        }
        assert!(
            worst < 1e-9 * scale.max(1.0),
            "M̃A not skew-symmetric: worst {worst}, scale {scale}"
        );
    }

    #[test]
    fn uniform_dielectric_shifts_cavity_resonance() {
        // Filling the cavity with ε_r scales every resonance by 1/√(ε_r·μ_r).
        use crate::mesh_gen::structured_box;
        use faer::Mat;
        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let eps_r = 4.0;
        let mats =
            vec![ElemMaterial::isotropic(eps_r, 1.0, 0.0); mesh.n_tets()];
        let op = MaxwellOperator::new_with_materials(&mesh, 2, 1.0, &mats);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
        let eig = mat.eigenvalues().expect("eig");
        let mut fund = (f64::NEG_INFINITY, 0.0_f64);
        for z in &eig {
            if z.im.abs() > 0.5 && z.re > fund.0 {
                fund = (z.re, z.im.abs());
            }
        }
        let want = std::f64::consts::PI * 2.0_f64.sqrt() / eps_r.sqrt();
        let err = (fund.1 - want).abs() / want;
        assert!(
            err < 0.02,
            "ε_r={eps_r}: |Im| = {:.4}, analytic π√2/√ε_r = {want:.4}, err {err:.3}",
            fund.1
        );
    }

    #[test]
    fn heterogeneous_central_flux_conserves_energy() {
        // With heterogeneous ε, μ the central-flux operator still conserves
        // the material-weighted energy: M̃·A is skew-symmetric.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let mats: Vec<ElemMaterial> = (0..mesh.n_tets())
            .map(|i| {
                ElemMaterial::isotropic(
                    1.0 + 0.7 * i as f64,
                    1.0 + 0.3 * i as f64,
                    0.0,
                )
            })
            .collect();
        let op = MaxwellOperator::new_with_materials(&mesh, 2, 0.0, &mats);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mm = op.assemble_energy_mass();

        let mut ma = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let mik = mm[i * n + k];
                if mik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    ma[i * n + j] += mik * a[k * n + j];
                }
            }
        }
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for p in 0..n {
            for q in 0..n {
                worst = worst.max((ma[p * n + q] + ma[q * n + p]).abs());
                scale = scale.max(ma[p * n + q].abs());
            }
        }
        assert!(
            worst < 1e-9 * scale.max(1.0),
            "heterogeneous M̃A not skew: worst {worst}, scale {scale}"
        );
    }

    #[test]
    fn conductivity_damps_at_the_analytic_rate() {
        // A uniform conductivity σ damps every mode by Re = -σ/(2ε) on top of
        // the numerical (upwind) damping — isolated by differencing σ>0
        // against σ=0.
        use crate::mesh_gen::structured_box;
        use faer::Mat;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let eps_r = 1.0;

        let least_damped_re = |sigma: f64| -> f64 {
            let mats =
                vec![ElemMaterial::isotropic(eps_r, 1.0, sigma); mesh.n_tets()];
            let op = MaxwellOperator::new_with_materials(&mesh, 2, 1.0, &mats);
            let n = op.n_dof();
            let a = op.assemble_dense();
            let mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
            let eig = mat.eigenvalues().expect("eig");
            eig.iter()
                .filter(|z| z.im.abs() > 1.0)
                .map(|z| z.re)
                .fold(f64::NEG_INFINITY, f64::max)
        };

        let sigma = 0.2;
        let delta = least_damped_re(sigma) - least_damped_re(0.0);
        let want = -sigma / (2.0 * eps_r);
        let err = (delta - want).abs() / want.abs();
        assert!(
            err < 0.1,
            "conductivity damping ΔRe = {delta:.5}, analytic -σ/2ε = {want:.5}, err {err:.3}"
        );
    }

    #[test]
    fn anisotropic_central_flux_conserves_energy() {
        // Diagonal-anisotropic ε, μ — the central-flux operator still
        // conserves the per-component-weighted material energy: M̃·A is skew.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let mats: Vec<ElemMaterial> = (0..mesh.n_tets())
            .map(|i| ElemMaterial {
                eps: [2.0 + i as f64, 3.0, 1.5],
                mu: [1.0, 1.2 + 0.3 * i as f64, 2.0],
                sigma: 0.0,
                sigma_m: 0.0,
            })
            .collect();
        let op = MaxwellOperator::new_with_materials(&mesh, 2, 0.0, &mats);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mm = op.assemble_energy_mass();

        let mut ma = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let mik = mm[i * n + k];
                if mik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    ma[i * n + j] += mik * a[k * n + j];
                }
            }
        }
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for p in 0..n {
            for q in 0..n {
                worst = worst.max((ma[p * n + q] + ma[q * n + p]).abs());
                scale = scale.max(ma[p * n + q].abs());
            }
        }
        assert!(
            worst < 1e-9 * scale.max(1.0),
            "anisotropic M̃A not skew: worst {worst}, scale {scale}"
        );
    }

    #[test]
    fn matched_absorber_damps_at_the_full_rate() {
        // An impedance-matched material (σ/ε = σ*/μ = ν) loses energy from
        // both the E and H halves, so a mode decays at Re = -ν — twice the
        // rate of an electric-only loss. This validates the magnetic-loss
        // term and the matched-absorber construction together.
        use crate::mesh_gen::structured_box;
        use faer::Mat;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let nu = 0.15;

        let least_damped_re = |mat: ElemMaterial| -> f64 {
            let mats = vec![mat; mesh.n_tets()];
            let op = MaxwellOperator::new_with_materials(&mesh, 2, 1.0, &mats);
            let n = op.n_dof();
            let a = op.assemble_dense();
            let m = Mat::from_fn(n, n, |i, j| a[i * n + j]);
            m.eigenvalues()
                .expect("eig")
                .iter()
                .filter(|z| z.im.abs() > 1.0)
                .map(|z| z.re)
                .fold(f64::NEG_INFINITY, f64::max)
        };

        let re_vac = least_damped_re(ElemMaterial::VACUUM);
        let re_matched =
            least_damped_re(ElemMaterial::matched_absorber(1.0, 1.0, nu));
        let delta = re_matched - re_vac;
        let err = (delta + nu).abs() / nu;
        assert!(
            err < 0.08,
            "matched-absorber decay ΔRe = {delta:.4}, analytic -ν = {:.4}, err {err:.3}",
            -nu
        );
    }

    #[test]
    fn cavity_eigenfrequencies_match_analytic() {
        // P3.5 gate: eigenvalues of A for a unit cubic PEC cavity vs the
        // analytic resonances ω = π·√(m²+n²+p²). Lowest physical mode is
        // (1,1,0) ⇒ ω = π√2.
        use crate::mesh_gen::structured_box;
        use faer::Mat;

        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        // Upwind flux (alpha = 1) damps the discontinuous spurious modes;
        // the physical cavity modes survive as the least-damped ones.
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
        let eig = mat.eigenvalues().expect("eigenvalues");

        // Among the non-static modes (|Im| > 1), the physical fundamental is
        // the least-damped — the largest (least-negative) real part.
        let want = std::f64::consts::PI * 2.0_f64.sqrt();
        let mut fundamental = (f64::NEG_INFINITY, 0.0_f64);
        for z in &eig {
            if z.im.abs() > 1.0 && z.re > fundamental.0 {
                fundamental = (z.re, z.im.abs());
            }
        }
        let (re, im) = fundamental;
        let err = (im - want).abs() / want;
        eprintln!(
            "DIAG cavity: fundamental Re={re:.5} |Im|={im:.5}  analytic π√2={want:.5}  rel.err={err:.4}"
        );
        assert!(re < 0.0, "upwind flux must damp — fundamental Re = {re}");
        assert!(
            err < 0.05,
            "fundamental |Im| = {im:.4}, analytic π√2 = {want:.4}, rel.err {err:.3}"
        );
    }

    #[test]
    fn cavity_eigenfrequency_on_irregular_mesh() {
        // WP1.2: the cavity fundamental survives on a skewed, irregular mesh —
        // the physics does not depend on mesh regularity, only the
        // discretisation error grows mildly.
        use crate::mesh_gen::structured_box_jittered;
        use faer::Mat;

        let mesh = structured_box_jittered(2, 2, 2, 1.0, 1.0, 1.0, 0.25, 7);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let a = op.assemble_dense();
        let mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
        let eig = mat.eigenvalues().expect("eigenvalues");

        let want = std::f64::consts::PI * 2.0_f64.sqrt();
        let mut fundamental = (f64::NEG_INFINITY, 0.0_f64);
        for z in &eig {
            if z.im.abs() > 1.0 && z.re > fundamental.0 {
                fundamental = (z.re, z.im.abs());
            }
        }
        let (re, im) = fundamental;
        let err = (im - want).abs() / want;
        eprintln!(
            "DIAG irregular mesh: Re={re:.5} |Im|={im:.5} π√2={want:.5} err={err:.4}"
        );
        assert!(re < 0.0, "upwind flux must damp — Re = {re}");
        assert!(
            err < 0.08,
            "irregular-mesh fundamental |Im| = {im:.4}, π√2 = {want:.4}, rel.err {err:.3}"
        );
    }

    #[test]
    fn cavity_fundamental_converges_under_refinement() {
        // WP1.3: mesh refinement drives the eigenfrequency error down at a
        // high-order rate.
        use crate::mesh_gen::structured_box;
        use faer::Mat;
        let want = std::f64::consts::PI * 2.0_f64.sqrt();

        let fundamental_err = |cells: usize, order: usize| -> f64 {
            let mesh = structured_box(cells, cells, cells, 1.0, 1.0, 1.0);
            let op = MaxwellOperator::new(&mesh, order, 1.0);
            let n = op.n_dof();
            let a = op.assemble_dense();
            let mat = Mat::from_fn(n, n, |i, j| a[i * n + j]);
            let eig = mat.eigenvalues().expect("eig");
            let mut best = (f64::NEG_INFINITY, 0.0_f64);
            for z in &eig {
                if z.im.abs() > 1.0 && z.re > best.0 {
                    best = (z.re, z.im.abs());
                }
            }
            (best.1 - want).abs() / want
        };

        let coarse = fundamental_err(1, 2); // 6 tets
        let fine = fundamental_err(2, 2); // 48 tets
        eprintln!(
            "DIAG convergence p=2: coarse(1^3)={coarse:.3e} fine(2^3)={fine:.3e} ratio={:.1}",
            coarse / fine
        );
        assert!(fine < coarse, "refinement must reduce the error");
        assert!(fine < 1e-3, "fine-mesh error {fine:.2e} too large");
        assert!(
            coarse / fine > 4.0,
            "weak convergence — error ratio only {:.1}",
            coarse / fine
        );
    }

    #[test]
    fn soft_source_injects_energy() {
        // A soft source driven from rest by a Gaussian pulse injects field
        // energy into the cavity.
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();

        // Source: E_z at the cavity centre — which is a node, so exact.
        let sdof = op.nearest_node_dof([0.5, 0.5, 0.5], 0, 2);
        assert!(sdof < n);
        let mut s = vec![0.0; n];
        s[sdof] = 1.0;

        let mut y = vec![0.0; n];
        let (t0, tau, h) = (0.3, 0.08, 0.01);
        for k in 0..80 {
            let t = k as f64 * h;
            let g = (-((t - t0) / tau).powi(2)).exp();
            let b: Vec<f64> = s.iter().map(|x| x * g).collect();
            y = etd_step(|x| op.apply(x), &y, &b, h, 30);
        }
        assert!(y.iter().all(|v| v.is_finite()));

        let mm = op.assemble_energy_mass();
        let mut energy = 0.0;
        for i in 0..n {
            for j in 0..n {
                energy += y[i] * mm[i * n + j] * y[j];
            }
        }
        assert!(energy > 1e-6, "soft source injected no energy: {energy:e}");
    }

    #[test]
    fn sparse_assembly_matches_matrix_free_apply() {
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let csr = op.assemble_sparse();
        assert_eq!(csr.n, n);

        let v: Vec<f64> =
            (0..n).map(|i| (1.0 + i as f64 * 0.07).cos()).collect();
        let sp = csr.matvec(&v);
        let mf = op.apply(&v);
        let err: f64 = sp
            .iter()
            .zip(&mf)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = mf.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err < 1e-10 * scale, "sparse vs matrix-free: err {err}");
        // The DG operator couples only neighbouring elements — genuinely sparse.
        assert!(
            csr.nnz() < n * n / 4,
            "operator not sparse: nnz {} of {}",
            csr.nnz(),
            n * n
        );
    }

    #[test]
    fn sparse_assembly_scales_without_densifying() {
        // WP6.2 gate: assemble `A` for a mesh whose dense form would need
        // gigabytes — the element-wise probe path must stay O(nnz). Each row
        // can couple at most `5·stride` columns (self + 4 face neighbours),
        // so nnz grows linearly with N, not quadratically.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(4, 4, 4, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let stride = 6 * 10; // 6 fields × Np(order 2) = 60

        // The dense matrix would be n² f64 — assert it is genuinely out of
        // reach, so this test actually exercises the non-densifying path.
        let dense_bytes = (n as u128).pow(2) * 8;
        assert!(
            dense_bytes > 1 << 30,
            "mesh too small to prove the point: dense would be {dense_bytes} B"
        );

        let csr = op.assemble_sparse();
        assert_eq!(csr.n, n);
        // Linear scaling: every row stays within the face-neighbour stencil.
        assert!(
            csr.nnz() <= 5 * stride * n,
            "nnz {} exceeds the stencil bound {}",
            csr.nnz(),
            5 * stride * n
        );

        // Correctness still holds at this scale.
        let v: Vec<f64> =
            (0..n).map(|i| (0.3 + i as f64 * 0.013).sin()).collect();
        let sp = csr.matvec(&v);
        let mf = op.apply(&v);
        let err: f64 = sp
            .iter()
            .zip(&mf)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = mf.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err < 1e-10 * scale, "sparse vs matrix-free: err {err}");
    }

    #[test]
    fn corner_nodes_map_to_tet_vertices() {
        // WP7.3 export hook: the four corner local indices are distinct,
        // and the physical DG node coordinates land on the mesh-tet
        // vertices in the affine-map order.
        use crate::mesh_gen::structured_box;
        let mesh = structured_box(2, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 3, 1.0);
        let corners = op.corner_local_nodes();
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(corners[i], corners[j], "corners not distinct");
            }
        }
        let np = op.re.n_nodes;
        let pts = op.node_coords();
        assert_eq!(pts.len(), np * mesh.n_tets());
        for (e, tet) in mesh.tets.iter().enumerate() {
            for (c, &local) in corners.iter().enumerate() {
                let got = pts[e * np + local];
                let want = mesh.nodes[tet[c]];
                let d: f64 =
                    (0..3).map(|k| (got[k] - want[k]).powi(2)).sum();
                assert!(
                    d.sqrt() < 1e-12,
                    "elem {e} corner {c}: {got:?} vs {want:?}"
                );
            }
        }
    }
}
