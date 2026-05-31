//! 2D wave-port mode solver: transverse eigenmodes at a port cross-section.
//!
//! A modal port injects and extracts a known transverse field profile
//! `(e_t, h_t)`. For a rectangular waveguide or a coaxial line that
//! profile is analytic ([`crate::waveguide`]); for an *arbitrary*
//! cross-section (a ridged guide, an L-shaped duct, a microstrip or
//! coplanar line) the profile has no closed form and is computed here by
//! a 2D eigensolve on the port-face triangulation. Backend-agnostic —
//! the time-domain and frequency-domain solvers share it.
//!
//! Two solvers, both on the extracted [`PortMesh2D`] cross-section:
//!
//! - [`solve_modes`] — the **scalar Helmholtz** eigenproblem
//!   `∇_t² ψ + k_c² ψ = 0` (P1 nodal), giving the `TE`/`TM` modes and
//!   cutoffs `k_c` of a *homogeneously filled* hollow guide. `TM` is
//!   Dirichlet (`E_z = 0` on PEC), `TE` Neumann.
//! - [`solve_vector_modes`] — the **full-vector hybrid** eigenproblem
//!   (mixed Nédélec-edge `E_t` + Lagrange-nodal `E_z`, eigenvalue
//!   `λ = β²/k0² = n_eff²`) with per-triangle `ε_r`, so it resolves the
//!   quasi-TEM mode of an *inhomogeneous* (substrate + air) line. PEC
//!   walls — the outer boundary and any internal conductor (a microstrip
//!   trace, via [`PortMesh2D::on_pec`]) — impose `tangential E = 0`.
//!
//! A solved mode becomes a [`NumericalMode`], which samples `(e_t, h_t)`
//! at arbitrary points for the port machinery: the scalar path
//! barycentric-interpolates a nodal `E_t`, the vector path evaluates the
//! Whitney edge field directly. Validation against analytic cutoffs and
//! effective indices lives in the tests.



/// A port face flattened to its 2D cross-section: nodes in the port
/// plane's local `(u, v)` coordinates, the triangles connecting them,
/// and which nodes lie on the boundary (PEC wall) of the cross-section.
#[derive(Clone, Debug)]
pub struct PortMesh2D {
    /// Local 2D coordinates of each distinct cross-section node.
    pub nodes: Vec<[f64; 2]>,
    /// Triangles as triples of indices into `nodes`.
    pub tris: Vec<[usize; 3]>,
    /// `true` for a node on the outer boundary of the cross-section —
    /// a boundary edge is one used by exactly one triangle. These get
    /// the Dirichlet condition for `TM` modes.
    pub on_boundary: Vec<bool>,
    /// `true` for a node lying on an *internal* PEC conductor (e.g. a
    /// microstrip trace cutting through the cross-section). These carry
    /// the same `tangential E = 0` condition as the outer wall; an edge
    /// with both endpoints on the conductor is a PEC edge. Empty / all
    /// false when the cross-section has no internal conductor.
    pub on_pec: Vec<bool>,
    /// The local frame `(û, v̂)` and origin, so a solved mode profile
    /// can be lifted back to 3D global coordinates.
    pub u_hat: [f64; 3],
    pub v_hat: [f64; 3],
    pub origin: [f64; 3],
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

impl PortMesh2D {
    /// Flatten a set of port-face triangles (given by global node
    /// coordinates and connectivity) into a 2D cross-section mesh.
    ///
    /// `face_tris` lists the triangles as triples of indices into the
    /// *global* `global_nodes`. `inward_normal` is the port's inward
    /// normal (any nonzero vector along the face normal); it fixes the
    /// out-of-plane axis. An in-plane orthonormal frame `(û, v̂)` is
    /// built from the first triangle's edge, and every node is projected
    /// onto it. Boundary nodes are detected from edges used by a single
    /// triangle.
    ///
    /// `pec_global`, if given, is a per-global-node mask marking nodes
    /// that lie on an internal PEC conductor (a microstrip trace); the
    /// resulting cross-section nodes inherit it as [`on_pec`](Self::on_pec).
    pub fn from_face(
        global_nodes: &[[f64; 3]],
        face_tris: &[[usize; 3]],
        inward_normal: [f64; 3],
        pec_global: Option<&[bool]>,
    ) -> PortMesh2D {
        // Normalise the out-of-plane axis.
        let nl = dot3(inward_normal, inward_normal).sqrt();
        let w_hat = [
            inward_normal[0] / nl,
            inward_normal[1] / nl,
            inward_normal[2] / nl,
        ];
        // Build an in-plane û from the first triangle's first edge,
        // orthogonalised against ŵ; v̂ = ŵ × û completes the frame.
        let t0 = face_tris[0];
        let p0 = global_nodes[t0[0]];
        let p1 = global_nodes[t0[1]];
        let mut e = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let edn = dot3(e, w_hat);
        e = [
            e[0] - edn * w_hat[0],
            e[1] - edn * w_hat[1],
            e[2] - edn * w_hat[2],
        ];
        let el = dot3(e, e).sqrt();
        let u_hat = [e[0] / el, e[1] / el, e[2] / el];
        let v_hat = cross3(w_hat, u_hat);
        let origin = p0;

        // Collect distinct nodes (remap global → local index) and project.
        let mut remap: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut nodes: Vec<[f64; 2]> = Vec::new();
        let mut local_to_global: Vec<usize> = Vec::new();
        let mut tris: Vec<[usize; 3]> = Vec::with_capacity(face_tris.len());
        let project = |g: usize| -> [f64; 2] {
            let p = global_nodes[g];
            let d = [
                p[0] - origin[0],
                p[1] - origin[1],
                p[2] - origin[2],
            ];
            [dot3(d, u_hat), dot3(d, v_hat)]
        };
        for &t in face_tris {
            let mut local = [0usize; 3];
            for (k, &g) in t.iter().enumerate() {
                let idx = *remap.entry(g).or_insert_with(|| {
                    nodes.push(project(g));
                    local_to_global.push(g);
                    nodes.len() - 1
                });
                local[k] = idx;
            }
            tris.push(local);
        }
        // Internal-PEC mask: a cross-section node inherits the PEC flag of
        // its global mesh node (e.g. on a microstrip trace surface).
        let on_pec: Vec<bool> = match pec_global {
            Some(mask) => {
                local_to_global.iter().map(|&g| mask[g]).collect()
            }
            None => vec![false; nodes.len()],
        };

        // Boundary edges are used by exactly one triangle. Tally each
        // undirected edge; mark the endpoints of singly-used edges.
        let mut edge_count: std::collections::HashMap<(usize, usize), u32> =
            std::collections::HashMap::new();
        let key = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
        for t in &tris {
            for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edge_count.entry(key(a, b)).or_insert(0) += 1;
            }
        }
        let mut on_boundary = vec![false; nodes.len()];
        for (&(a, b), &c) in &edge_count {
            if c == 1 {
                on_boundary[a] = true;
                on_boundary[b] = true;
            }
        }

        PortMesh2D { nodes, tris, on_boundary, on_pec, u_hat, v_hat, origin }
    }

    /// Number of cross-section nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Per-triangle area and the three constant P1 gradients
    /// `∇λ_i` (2D), for one triangle. Returns `(area, [g0, g1, g2])`.
    fn tri_geom(&self, t: [usize; 3]) -> (f64, [[f64; 2]; 3]) {
        let p = [self.nodes[t[0]], self.nodes[t[1]], self.nodes[t[2]]];
        // Edge vectors; signed area via the cross product of two edges.
        let (x0, y0) = (p[0][0], p[0][1]);
        let (x1, y1) = (p[1][0], p[1][1]);
        let (x2, y2) = (p[2][0], p[2][1]);
        let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        let area = 0.5 * det.abs();
        // P1 gradients: ∇λ_i = (1/2A)·[y_{j}-y_{k}, x_{k}-x_{j}] (cyclic).
        // Use the signed det so the sign is consistent across the formula.
        let inv = 1.0 / det;
        let g0 = [(y1 - y2) * inv, (x2 - x1) * inv];
        let g1 = [(y2 - y0) * inv, (x0 - x2) * inv];
        let g2 = [(y0 - y1) * inv, (x1 - x0) * inv];
        (area, [g0, g1, g2])
    }

    /// Assemble the dense stiffness `S` and lumped (diagonal) mass `m`
    /// for the P1 scalar problem. `S` is `n×n` row-major; `m` is the
    /// length-`n` diagonal (row-sum-lumped consistent mass), which makes
    /// the generalized eigenproblem `S ψ = k_c² diag(m) ψ` reducible to
    /// a symmetric standard problem without a Cholesky factorisation.
    pub fn assemble(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.n_nodes();
        let mut s = vec![0.0; n * n];
        let mut m = vec![0.0; n];
        for &t in &self.tris {
            let (area, g) = self.tri_geom(t);
            // Stiffness: S_ij += area · (∇λ_i · ∇λ_j).
            for a in 0..3 {
                for b in 0..3 {
                    let val = area * (g[a][0] * g[b][0] + g[a][1] * g[b][1]);
                    s[t[a] * n + t[b]] += val;
                }
            }
            // Lumped mass: each node gets area/3 (row-sum of the
            // consistent element mass area/12·[[2,1,1],…] = area/3).
            for a in 0..3 {
                m[t[a]] += area / 3.0;
            }
        }
        (s, m)
    }
}

/// One transverse eigenmode of the cross-section.
#[derive(Clone, Debug)]
pub struct PortEigenmode {
    /// Cutoff wavenumber `k_c` (operator units; `ω_c = c·k_c`, and with
    /// `c = 1` the cutoff angular frequency equals `k_c`).
    pub k_c: f64,
    /// The scalar modal field `ψ` at every cross-section node
    /// (`E_z` for a `TM` mode, `H_z` for a `TE` mode).
    pub psi: Vec<f64>,
}

/// Boundary condition for the scalar mode solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeKind {
    /// `TM`: `E_z = 0` (Dirichlet) on the PEC boundary.
    Tm,
    /// `TE`: `∂H_z/∂n = 0` (natural Neumann) — no constraint applied.
    Te,
}

/// Solve the scalar Helmholtz eigenproblem `S ψ = k_c² diag(m) ψ` on the
/// cross-section, returning the `n_modes` lowest-cutoff propagating modes
/// (smallest positive `k_c²`), sorted ascending.
///
/// `TM` modes pin the boundary nodes to zero (Dirichlet); `TE` modes
/// leave them free. The generalized problem is reduced to the symmetric
/// standard problem `B φ = k_c² φ` with `B = D^{-1/2} S D^{-1/2}` and
/// `D = diag(m)`, then `ψ = D^{-1/2} φ`. Dense — intended for the modest
/// node counts of a single port face.
pub fn solve_modes(
    mesh: &PortMesh2D,
    kind: ModeKind,
    n_modes: usize,
) -> Vec<PortEigenmode> {
    let n_full = mesh.n_nodes();
    let (s_full, m_full) = mesh.assemble();

    // PEC nodes (outer wall + any internal conductor) carry Dirichlet for
    // TM (E_z = 0) and Neumann for TE; an internal conductor still pins
    // E_z, so it is constrained in both — but for the scalar TE Neumann
    // problem only the internal conductor (not the outer wall) is pinned.
    let pec = |i: usize| {
        mesh.on_boundary[i] || mesh.on_pec.get(i).copied().unwrap_or(false)
    };
    let keep: Vec<usize> = match kind {
        ModeKind::Tm => (0..n_full).filter(|&i| !pec(i)).collect(),
        ModeKind::Te => (0..n_full)
            .filter(|&i| !mesh.on_pec.get(i).copied().unwrap_or(false))
            .collect(),
    };
    let n = keep.len();
    if n == 0 {
        return Vec::new();
    }

    // Symmetric reduced problem B = D^{-1/2} S D^{-1/2}.
    let d_inv_sqrt: Vec<f64> =
        keep.iter().map(|&i| 1.0 / m_full[i].sqrt()).collect();
    let b = faer::Mat::<f64>::from_fn(n, n, |i, j| {
        s_full[keep[i] * n_full + keep[j]] * d_inv_sqrt[i] * d_inv_sqrt[j]
    });

    let eig = match b.eigen() {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let evals = eig.S().column_vector();
    let evecs = eig.U();

    // Collect (k_c², column) for positive eigenvalues, then sort.
    let mut idx: Vec<(f64, usize)> = (0..n)
        .filter_map(|k| {
            let lam = evals[k].re;
            // Drop the (near-)zero TE constant mode and any numerical
            // negatives; a propagating mode has k_c² > 0.
            if lam > 1e-9 {
                Some((lam, k))
            } else {
                None
            }
        })
        .collect();
    idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    idx.into_iter()
        .take(n_modes)
        .map(|(lam, k)| {
            // ψ = D^{-1/2} φ, scattered back to full node indexing.
            let mut psi = vec![0.0; n_full];
            for li in 0..n {
                psi[keep[li]] = evecs[(li, k)].re * d_inv_sqrt[li];
            }
            PortEigenmode { k_c: lam.sqrt(), psi }
        })
        .collect()
}

/// A solved numerical port mode, ready to be sampled as a transverse
/// `(e_t, h_t)` profile at arbitrary points on the port face — the
/// drop-in replacement for an analytic [`crate::waveguide::RectPort`]
/// profile when the cross-section has no closed-form mode.
///
/// The transverse fields follow from the scalar potential `ψ` by the
/// standard waveguide relations. With `ψ` piecewise-linear (P1) its
/// transverse gradient `∇_t ψ` is constant per triangle, so:
/// - **TM** (`ψ = E_z`): `E_t ∝ ∇_t ψ`,
/// - **TE** (`ψ = H_z`): `E_t ∝ ẑ × ∇_t ψ`.
///
/// `h_t = ŵ × e_t` (the inward normal `ŵ` plays the role of `ẑ`), and
/// the modal impedance enters the forward/backward split via
/// [`te_impedance`](Self::te_impedance). The profile is normalised so
/// its peak transverse magnitude over the cross-section is unity, to
/// match the analytic profiles' order-unity convention.
#[derive(Clone, Debug)]
pub struct NumericalMode {
    mesh: PortMesh2D,
    /// Transverse electric field `E_t` at each cross-section node, in the
    /// port-plane `(u, v)` components — the **scalar**-path profile
    /// (`from_scalar`); `e_profile` barycentric-interpolates it. Empty
    /// when the Whitney representation is used.
    e_uv_node: Vec<[f64; 2]>,
    /// The **vector**-path profile (`from_vector`): the raw Nédélec edge
    /// solution `e_edge` (one coefficient per global cross-section edge)
    /// plus the per-triangle edge data, so `e_profile` evaluates the
    /// Whitney field *directly* at each query point. This preserves the
    /// concentrated quasi-TEM shape that area-averaging to nodes smears
    /// away — the smearing makes the wave port degenerate to a uniform
    /// (lumped-like) profile and undercount `|S21|`, so the direct eval
    /// is essential for microstrip-class lines. `None` for scalar modes.
    whitney: Option<(Vec<f64>, Vec<TriEdges>)>,
    /// Inverse peak `|E_t|` over the cross-section — the unit-peak
    /// normalisation.
    inv_peak: f64,
    /// Inward normal `ŵ = û × v̂` (global), the mode propagation axis.
    w_hat: [f64; 3],
    /// Cutoff wavenumber (`0` for a quasi-TEM numerical mode).
    cutoff: f64,
    /// Modal-impedance model for the forward/backward split.
    z_model: ImpedanceModel,
}

/// How a numerical mode's wave impedance varies with frequency.
#[derive(Clone, Copy, Debug)]
enum ImpedanceModel {
    /// `TE`: `Z = 1/√(1−(k_c/ω)²)`.
    Te { k_c: f64 },
    /// `TM`: `Z = √(1−(k_c/ω)²)`.
    Tm { k_c: f64 },
    /// Flat `Z = z` — a quasi-TEM / hybrid vector mode, with `z = 1/n_eff`
    /// (the TEM wave impedance `η√(μ/ε_eff)` in `η = 1` units).
    Flat { z: f64 },
}

impl NumericalMode {
    /// Build a numerical mode from a scalar `TE`/`TM` eigenpair. The
    /// transverse field follows from `ψ`: `TM` → `E_t ∝ ∇_t ψ`, `TE` →
    /// `E_t ∝ ẑ × ∇_t ψ`. The per-triangle constant gradient is averaged
    /// (area-weighted) to the nodes for the unified nodal representation.
    pub fn from_scalar(
        mesh: PortMesh2D,
        mode: &PortEigenmode,
        kind: ModeKind,
    ) -> NumericalMode {
        let n_node = mesh.n_nodes();
        let mut acc = vec![[0.0f64; 2]; n_node];
        let mut wsum = vec![0.0f64; n_node];
        for &t in &mesh.tris {
            let (area, g) = mesh.tri_geom(t);
            // ∇_t ψ (constant on the triangle).
            let mut gp = [0.0; 2];
            for k in 0..3 {
                gp[0] += mode.psi[t[k]] * g[k][0];
                gp[1] += mode.psi[t[k]] * g[k][1];
            }
            // E_t direction: TM → ∇ψ, TE → ẑ×∇ψ = (−∂v, ∂u).
            let et = match kind {
                ModeKind::Tm => [gp[0], gp[1]],
                ModeKind::Te => [-gp[1], gp[0]],
            };
            for k in 0..3 {
                acc[t[k]][0] += area * et[0];
                acc[t[k]][1] += area * et[1];
                wsum[t[k]] += area;
            }
        }
        let e_uv_node: Vec<[f64; 2]> = (0..n_node)
            .map(|i| {
                if wsum[i] > 0.0 {
                    [acc[i][0] / wsum[i], acc[i][1] / wsum[i]]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect();
        let z_model = match kind {
            ModeKind::Te => ImpedanceModel::Te { k_c: mode.k_c },
            ModeKind::Tm => ImpedanceModel::Tm { k_c: mode.k_c },
        };
        let w_hat = cross3(mesh.u_hat, mesh.v_hat);
        let peak = e_uv_node
            .iter()
            .map(|e| (e[0] * e[0] + e[1] * e[1]).sqrt())
            .fold(0.0_f64, f64::max);
        let inv_peak = if peak > 0.0 { 1.0 / peak } else { 0.0 };
        NumericalMode {
            mesh,
            e_uv_node,
            whitney: None,
            inv_peak,
            w_hat,
            cutoff: mode.k_c,
            z_model,
        }
    }

    /// Build a numerical mode from a full-vector hybrid solve. Stores the
    /// raw edge solution and evaluates the Whitney field **directly** at
    /// query points (no nodal averaging), preserving the concentrated
    /// quasi-TEM shape. Zero cutoff, flat impedance `1/n_eff`.
    pub fn from_vector(mesh: PortMesh2D, mode: &VectorMode) -> NumericalMode {
        let z = if mode.n_eff > 0.0 { 1.0 / mode.n_eff } else { 1.0 };
        let (_n_edge, tri_edges, _use) = build_edges(&mesh);
        let e_edge = mode.e_edge.clone();
        // Unit-peak normalisation: sample |E_t| at every triangle's three
        // vertices via the direct Whitney sum and take the max.
        let mut peak = 0.0_f64;
        for (ti, &t) in mesh.tris.iter().enumerate() {
            let (_a, g) = mesh.tri_geom(t);
            for vloc in 0..3 {
                let mut l = [0.0; 3];
                l[vloc] = 1.0;
                let e = whitney_at(&tri_edges[ti], &g, &e_edge, l);
                peak = peak.max((e[0] * e[0] + e[1] * e[1]).sqrt());
            }
        }
        let inv_peak = if peak > 0.0 { 1.0 / peak } else { 0.0 };
        let w_hat = cross3(mesh.u_hat, mesh.v_hat);
        NumericalMode {
            mesh,
            e_uv_node: Vec::new(),
            whitney: Some((e_edge, tri_edges)),
            inv_peak,
            w_hat,
            cutoff: 0.0,
            z_model: ImpedanceModel::Flat { z },
        }
    }

    /// Cutoff angular frequency (`= k_c`; `0` for a quasi-TEM mode).
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Modal wave impedance `Z(omega)` in operator units (`η = 1`).
    pub fn te_impedance(&self, omega: f64) -> f64 {
        match self.z_model {
            ImpedanceModel::Flat { z } => z,
            ImpedanceModel::Te { k_c } => {
                let s = (1.0 - (k_c / omega).powi(2)).max(0.0).sqrt();
                if s > 0.0 { 1.0 / s } else { f64::INFINITY }
            }
            ImpedanceModel::Tm { k_c } => {
                (1.0 - (k_c / omega).powi(2)).max(0.0).sqrt()
            }
        }
    }

    /// Locate the triangle containing the in-plane point `(u, v)` and
    /// return its index plus the barycentric coordinates there. Falls back
    /// to the nearest triangle if the point sits just outside the mesh.
    fn locate(&self, uv: [f64; 2]) -> (usize, [f64; 3]) {
        let mut best = (0usize, [1.0, 0.0, 0.0]);
        let mut best_slack = f64::NEG_INFINITY;
        for (ti, &t) in self.mesh.tris.iter().enumerate() {
            let a = self.mesh.nodes[t[0]];
            let b = self.mesh.nodes[t[1]];
            let c = self.mesh.nodes[t[2]];
            let d =
                (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
            if d.abs() < 1e-30 {
                continue;
            }
            let l0 = ((b[1] - c[1]) * (uv[0] - c[0])
                + (c[0] - b[0]) * (uv[1] - c[1]))
                / d;
            let l1 = ((c[1] - a[1]) * (uv[0] - c[0])
                + (a[0] - c[0]) * (uv[1] - c[1]))
                / d;
            let l2 = 1.0 - l0 - l1;
            let slack = l0.min(l1).min(l2);
            if slack >= -1e-9 {
                return (ti, [l0, l1, l2]);
            }
            if slack > best_slack {
                best_slack = slack;
                best = (ti, [l0, l1, l2]);
            }
        }
        best
    }

    /// In-plane `(u, v)` coordinates of a global point on the port face.
    fn to_uv(&self, x: [f64; 3]) -> [f64; 2] {
        let o = self.mesh.origin;
        let d = [x[0] - o[0], x[1] - o[1], x[2] - o[2]];
        [dot3(d, self.mesh.u_hat), dot3(d, self.mesh.v_hat)]
    }

    /// Transverse electric-field profile at a global point on the port
    /// face, in global coordinates, unit-peak normalised. The vector path
    /// evaluates the Whitney edge field directly (sharp); the scalar path
    /// barycentric-interpolates the nodal `E_t`.
    pub fn e_profile(&self, x: [f64; 3]) -> [f64; 3] {
        let uv = self.to_uv(x);
        let (ti, l) = self.locate(uv);
        let e: [f64; 2] = match &self.whitney {
            Some((e_edge, tri_edges)) => {
                let t = self.mesh.tris[ti];
                let (_a, g) = self.mesh.tri_geom(t);
                whitney_at(&tri_edges[ti], &g, e_edge, l)
            }
            None => {
                let t = self.mesh.tris[ti];
                let mut e = [0.0f64; 2];
                for k in 0..3 {
                    e[0] += l[k] * self.e_uv_node[t[k]][0];
                    e[1] += l[k] * self.e_uv_node[t[k]][1];
                }
                e
            }
        };
        let (eu, ev) = (e[0] * self.inv_peak, e[1] * self.inv_peak);
        [
            eu * self.mesh.u_hat[0] + ev * self.mesh.v_hat[0],
            eu * self.mesh.u_hat[1] + ev * self.mesh.v_hat[1],
            eu * self.mesh.u_hat[2] + ev * self.mesh.v_hat[2],
        ]
    }

    /// Transverse magnetic-field profile `h_t = ŵ × e_t` at a global
    /// point — the inward-propagating partner of `e_t`. Global coords.
    pub fn h_profile(&self, x: [f64; 3]) -> [f64; 3] {
        cross3(self.w_hat, self.e_profile(x))
    }
}

/// One full-vector hybrid mode of an (optionally inhomogeneous) cross
/// section, solved at a fixed operating wavenumber `k0`.
#[derive(Clone, Debug)]
pub struct VectorMode {
    /// Effective index `n_eff = β / k0` (`n_eff² = λ`, the eigenvalue).
    pub n_eff: f64,
    /// Operating free-space wavenumber the solve was run at.
    pub k0: f64,
    /// Transverse electric field `E_t` at each cross-section node, in the
    /// port-plane `(u, v)` components — recovered (area-averaged) from the
    /// edge-element solution. Convenience for inspection; the sharp profile
    /// uses `e_edge` directly.
    pub e_uv_node: Vec<[f64; 2]>,
    /// Raw Nédélec edge solution — one coefficient per global cross-section
    /// edge (zero on constrained PEC edges). Feeds the direct Whitney-field
    /// evaluation in [`NumericalMode::from_vector`].
    pub e_edge: Vec<f64>,
}

/// Per-triangle edge data for the Whitney (Nédélec) assembly: the global
/// edge index, orientation sign and length, for each of the triangle's
/// three edges (local edge `e` joins local nodes `((e+1)%3, (e+2)%3)`).
#[derive(Clone, Debug)]
struct TriEdges {
    gidx: [usize; 3],
    sign: [f64; 3],
    len: [f64; 3],
}

/// Evaluate the transverse Whitney (edge-element) field `E_t = Σ_e e_edge[e]
/// · W_e` at barycentric coordinates `l` inside one triangle, given its
/// edge data and constant P1 gradients `g`. Returns the `(u, v)` components.
fn whitney_at(
    te: &TriEdges,
    g: &[[f64; 2]; 3],
    e_edge: &[f64],
    l: [f64; 3],
) -> [f64; 2] {
    let mut e = [0.0f64; 2];
    for k in 0..3 {
        let (a, b) = ((k + 1) % 3, (k + 2) % 3);
        let s = te.sign[k] * te.len[k] * e_edge[te.gidx[k]];
        e[0] += s * (l[a] * g[b][0] - l[b] * g[a][0]);
        e[1] += s * (l[a] * g[b][1] - l[b] * g[a][1]);
    }
    e
}

/// Enumerate unique mesh edges and, per triangle, the global index,
/// orientation sign and length of its three local edges. Also returns the
/// per-global-edge triangle-use count (1 = boundary edge).
fn build_edges(mesh: &PortMesh2D) -> (usize, Vec<TriEdges>, Vec<u32>) {
    use std::collections::HashMap;
    let mut emap: HashMap<(usize, usize), usize> = HashMap::new();
    let mut use_count: Vec<u32> = Vec::new();
    let mut per_tri: Vec<TriEdges> = Vec::with_capacity(mesh.tris.len());
    for &t in &mesh.tris {
        let mut te =
            TriEdges { gidx: [0; 3], sign: [0.0; 3], len: [0.0; 3] };
        for e in 0..3 {
            let a = t[(e + 1) % 3];
            let b = t[(e + 2) % 3];
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            let gi = *emap.entry((lo, hi)).or_insert_with(|| {
                use_count.push(0);
                use_count.len() - 1
            });
            use_count[gi] += 1;
            te.gidx[e] = gi;
            te.sign[e] = if a == lo { 1.0 } else { -1.0 };
            let pa = mesh.nodes[a];
            let pb = mesh.nodes[b];
            te.len[e] =
                ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2)).sqrt();
        }
        per_tri.push(te);
    }
    (emap.len(), per_tri, use_count)
}

/// Solve the full-vector hybrid mode eigenproblem at a fixed operating
/// wavenumber `k0`, returning up to `n_modes` modes by descending
/// effective index (most-confined first).
///
/// `eps_r` is the per-triangle relative permittivity (length
/// `mesh.tris.len()`) — an inhomogeneous fill (substrate + air), exactly
/// what a microstrip-class line needs and what the scalar
/// [`solve_modes`] cannot represent. `μ_r = 1`.
///
/// Mixed Nédélec-edge (`E_t`) / Lagrange-nodal (`E_z`) discretisation,
/// eigenvalue `λ = β²/k0² = n_eff²`. PEC walls (cross-section boundary)
/// impose `tangential E = 0`: boundary edges and nodes are constrained
/// out. The singular generalized problem `A x = λ B x` is solved by
/// shift-invert near `σ` (just above the largest `ε_r`): eigenvalues `ν`
/// of `(A − σB)⁻¹ B`, then `λ = σ + 1/ν`. Dense — sized for one face.
pub fn solve_vector_modes(
    mesh: &PortMesh2D,
    eps_r: &[f64],
    k0: f64,
    n_modes: usize,
) -> Vec<VectorMode> {
    use faer::Mat;
    use faer::linalg::solvers::Solve;
    let n_node = mesh.n_nodes();
    let (n_edge, tri_edges, edge_use) = build_edges(mesh);

    let on_pec = |i: usize| mesh.on_pec.get(i).copied().unwrap_or(false);
    // A node is PEC-constrained on the outer wall OR an internal conductor.
    let node_pec: Vec<bool> =
        (0..n_node).map(|i| mesh.on_boundary[i] || on_pec(i)).collect();
    // Edge is PEC if (a) an outer-boundary edge (used by one triangle), or
    // (b) both endpoints lie on an INTERNAL conductor — the edge then runs
    // along the trace surface and its tangential E must vanish. The
    // internal rule uses `on_pec` (not `node_pec`) so two outer-boundary
    // nodes joined by an interior chord are not spuriously constrained.
    let mut edge_pec = edge_use.iter().map(|&c| c == 1).collect::<Vec<_>>();
    for (ti, &t) in mesh.tris.iter().enumerate() {
        let te = &tri_edges[ti];
        for e in 0..3 {
            let a = t[(e + 1) % 3];
            let b = t[(e + 2) % 3];
            if on_pec(a) && on_pec(b) {
                edge_pec[te.gidx[e]] = true;
            }
        }
    }
    let edge_free: Vec<usize> =
        (0..n_edge).filter(|&e| !edge_pec[e]).collect();
    let node_free: Vec<usize> =
        (0..n_node).filter(|&i| !node_pec[i]).collect();
    let ne = edge_free.len();
    let nn = node_free.len();
    let ndof = ne + nn;
    if ndof == 0 {
        return Vec::new();
    }
    let mut edge_red = vec![usize::MAX; n_edge];
    for (r, &g) in edge_free.iter().enumerate() {
        edge_red[g] = r;
    }
    let mut node_red = vec![usize::MAX; n_node];
    for (r, &g) in node_free.iter().enumerate() {
        node_red[g] = r;
    }

    let mut a = vec![0.0f64; ndof * ndof];
    let mut b = vec![0.0f64; ndof * ndof];
    let inv_k0sq = 1.0 / (k0 * k0);
    const QP: [[f64; 3]; 3] = [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ];
    for (ti, &t) in mesh.tris.iter().enumerate() {
        let (area, g) = mesh.tri_geom(t);
        let er = eps_r[ti];
        let te = &tri_edges[ti];
        let ab = |e: usize| ((e + 1) % 3, (e + 2) % 3);
        let curl = |e: usize| -> f64 {
            let (a, b) = ab(e);
            let cz = g[a][0] * g[b][1] - g[a][1] * g[b][0];
            te.sign[e] * te.len[e] * 2.0 * cz
        };
        let w_at = |e: usize, l: [f64; 3]| -> [f64; 2] {
            let (a, b) = ab(e);
            let s = te.sign[e] * te.len[e];
            [
                s * (l[a] * g[b][0] - l[b] * g[a][0]),
                s * (l[a] * g[b][1] - l[b] * g[a][1]),
            ]
        };
        let edof = |e: usize| edge_red[te.gidx[e]];
        let ndof_of = |n: usize| {
            let r = node_red[t[n]];
            if r == usize::MAX { usize::MAX } else { ne + r }
        };

        for i in 0..3 {
            let di = edof(i);
            if di == usize::MAX {
                continue;
            }
            let ci = curl(i);
            for j in 0..3 {
                let dj = edof(j);
                if dj == usize::MAX {
                    continue;
                }
                let cj = curl(j);
                let mut wdot = 0.0;
                for q in QP {
                    let wi = w_at(i, q);
                    let wj = w_at(j, q);
                    wdot += (wi[0] * wj[0] + wi[1] * wj[1]) / 3.0;
                }
                let mass = area * wdot;
                let stiff = area * ci * cj;
                a[di * ndof + dj] += stiff * inv_k0sq - er * mass;
                b[di * ndof + dj] += -mass * inv_k0sq;
            }
        }
        for i in 0..3 {
            for jn in 0..3 {
                let di = edof(i);
                let dj = ndof_of(jn);
                if di != usize::MAX && dj != usize::MAX {
                    let mut val = 0.0;
                    for q in QP {
                        let wi = w_at(i, q);
                        val += (g[jn][0] * wi[0] + g[jn][1] * wi[1]) / 3.0;
                    }
                    a[di * ndof + dj] += area * val;
                }
                let dii = ndof_of(i);
                let djj = edof(jn);
                if dii != usize::MAX && djj != usize::MAX {
                    let mut val = 0.0;
                    for q in QP {
                        let wj = w_at(jn, q);
                        val += (wj[0] * g[i][0] + wj[1] * g[i][1]) / 3.0;
                    }
                    a[dii * ndof + djj] += area * er * val;
                }
            }
        }
        for i in 0..3 {
            let di = ndof_of(i);
            if di == usize::MAX {
                continue;
            }
            for j in 0..3 {
                let dj = ndof_of(j);
                if dj == usize::MAX {
                    continue;
                }
                let mij = area / 12.0 * if i == j { 2.0 } else { 1.0 };
                a[di * ndof + dj] += -er * k0 * k0 * mij;
            }
        }
    }

    let sigma = eps_r.iter().cloned().fold(1.0_f64, f64::max) + 0.05;
    let c = Mat::<f64>::from_fn(ndof, ndof, |i, j| {
        a[i * ndof + j] - sigma * b[i * ndof + j]
    });
    let bmat = Mat::<f64>::from_fn(ndof, ndof, |i, j| b[i * ndof + j]);
    let lu = c.partial_piv_lu();
    let m = lu.solve(&bmat);
    let eig = match m.eigen() {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let evals = eig.S().column_vector();
    let evecs = eig.U();

    // The eigenvalue comes out as λ = β² (propagation constant squared);
    // the effective index is n_eff² = β²/k0². A physically propagating,
    // confined mode has 0 < n_eff² ≤ ε_max (β real, slower than the
    // fastest local plane wave). Negative λ are evanescent / non-physical.
    let eps_max = eps_r.iter().cloned().fold(1.0_f64, f64::max);
    let k0sq = k0 * k0;
    let mut found: Vec<(f64, usize)> = Vec::new();
    for k in 0..ndof {
        let nu = evals[k];
        if nu.im.abs() > 1e-6 * (nu.re.abs() + 1e-30) {
            continue;
        }
        if nu.re.abs() < 1e-30 {
            continue;
        }
        let lam = sigma + 1.0 / nu.re; // = β²
        let neff2 = lam / k0sq;
        if std::env::var("PORT_EIGEN_DEBUG").is_ok() {
            eprintln!("  ν={:.4e}  β²={:.4}  n_eff²={:.4}", nu.re, lam, neff2);
        }
        if neff2 > 1e-6 && neff2 <= eps_max + 1e-3 {
            found.push((neff2, k));
        }
    }
    found.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    found
        .into_iter()
        .take(n_modes)
        .map(|(neff2, k)| {
            let mut acc = vec![[0.0f64; 2]; n_node];
            let mut wsum = vec![0.0f64; n_node];
            for (ti, &t) in mesh.tris.iter().enumerate() {
                let (area, g) = mesh.tri_geom(t);
                let te = &tri_edges[ti];
                let ab = |e: usize| ((e + 1) % 3, (e + 2) % 3);
                for vloc in 0..3 {
                    let mut l = [0.0; 3];
                    l[vloc] = 1.0;
                    let mut et = [0.0f64; 2];
                    for e in 0..3 {
                        let dr = edge_red[te.gidx[e]];
                        if dr == usize::MAX {
                            continue;
                        }
                        let coeff = evecs[(dr, k)].re;
                        let (a2, b2) = ab(e);
                        let s = te.sign[e] * te.len[e];
                        et[0] +=
                            coeff * s * (l[a2] * g[b2][0] - l[b2] * g[a2][0]);
                        et[1] +=
                            coeff * s * (l[a2] * g[b2][1] - l[b2] * g[a2][1]);
                    }
                    acc[t[vloc]][0] += area * et[0];
                    acc[t[vloc]][1] += area * et[1];
                    wsum[t[vloc]] += area;
                }
            }
            let e_uv_node: Vec<[f64; 2]> = (0..n_node)
                .map(|i| {
                    if wsum[i] > 0.0 {
                        [acc[i][0] / wsum[i], acc[i][1] / wsum[i]]
                    } else {
                        [0.0, 0.0]
                    }
                })
                .collect();
            // Raw edge solution scattered to global edges (0 on PEC edges),
            // for the sharp direct-Whitney profile evaluation downstream.
            let mut e_edge = vec![0.0f64; n_edge];
            for (g, &r) in edge_red.iter().enumerate() {
                if r != usize::MAX {
                    e_edge[g] = evecs[(r, k)].re;
                }
            }
            VectorMode { n_eff: neff2.sqrt(), k0, e_uv_node, e_edge }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Build a structured right-triangle mesh of an `a × b` rectangle
    /// with `nx × ny` cells (each split into two triangles). Returns the
    /// global 3D nodes (on the z = 0 plane) and the triangle connectivity.
    fn rect_mesh(
        a: f64,
        b: f64,
        nx: usize,
        ny: usize,
    ) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let mut nodes = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push([
                    a * i as f64 / nx as f64,
                    b * j as f64 / ny as f64,
                    0.0,
                ]);
            }
        }
        let id = |i: usize, j: usize| j * (nx + 1) + i;
        let mut tris = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let (n00, n10, n01, n11) =
                    (id(i, j), id(i + 1, j), id(i, j + 1), id(i + 1, j + 1));
                tris.push([n00, n10, n11]);
                tris.push([n00, n11, n01]);
            }
        }
        (nodes, tris)
    }

    #[test]
    fn extracts_a_rectangular_cross_section() {
        let (nodes, tris) = rect_mesh(2.0, 1.0, 4, 2);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        assert_eq!(pm.n_nodes(), 5 * 3);
        // The 4 rectangle sides are boundary; interior nodes are not.
        let n_boundary = pm.on_boundary.iter().filter(|&&x| x).count();
        // Perimeter nodes of a 5×3 grid: 2*(5+3) - 4 = 12.
        assert_eq!(n_boundary, 12, "boundary node count");
        // Total area recovered = a·b.
        let area: f64 =
            pm.tris.iter().map(|&t| pm.tri_geom(t).0).sum();
        assert!((area - 2.0).abs() < 1e-12, "area = {area}");
    }

    #[test]
    fn tm_modes_of_a_rectangular_guide_match_analytic() {
        // TM_mn cutoff of an a×b guide: k_c = π·√((m/a)² + (n/b)²).
        // Lowest TM mode is TM₁₁. Use a square 1×1 → k_c = π√2.
        let (nodes, tris) = rect_mesh(1.0, 1.0, 16, 16);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let modes = solve_modes(&pm, ModeKind::Tm, 1);
        assert!(!modes.is_empty(), "no TM mode found");
        let kc = modes[0].k_c;
        let want = PI * (2.0_f64).sqrt();
        // P1 on a 16×16 mesh: a couple percent high (FEM over-estimates
        // eigenvalues, lumped mass shifts a touch the other way).
        let rel = (kc - want).abs() / want;
        assert!(rel < 0.04, "TM₁₁ k_c = {kc:.4}, want {want:.4} (rel {rel:.3})");
    }

    #[test]
    fn vector_solver_recovers_te10_neff_homogeneous() {
        // Full-vector solve on a homogeneous (ε=1) 2×1 rectangular metallic
        // guide at k0 above the TE₁₀ cutoff. The dominant mode's effective
        // index must match the analytic TE₁₀: n_eff² = 1 − (k_c/k0)² with
        // k_c = π/a = π/2.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 20, 10);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let eps = vec![1.0; pm.tris.len()];
        let k0 = 3.0; // > π/2 ≈ 1.571, so TE₁₀ propagates
        let modes = solve_vector_modes(&pm, &eps, k0, 1);
        assert!(!modes.is_empty(), "vector solve found no propagating mode");
        let kc = PI / 2.0;
        let want = (1.0 - (kc / k0).powi(2)).sqrt(); // analytic n_eff
        let got = modes[0].n_eff;
        let rel = (got - want).abs() / want;
        assert!(
            rel < 0.05,
            "vector TE₁₀ n_eff = {got:.4}, want {want:.4} (rel {rel:.3})",
        );
        // E_t of TE₁₀ is along v̂ = ŷ and peaks mid-width: check the node
        // nearest (a/2, b/2) has a dominant v-component.
        let mut best = 0usize;
        let mut bestd = f64::INFINITY;
        for (i, n) in pm.nodes.iter().enumerate() {
            let d = (n[0] - 1.0).powi(2) + (n[1] - 0.5).powi(2);
            if d < bestd {
                bestd = d;
                best = i;
            }
        }
        let e = modes[0].e_uv_node[best];
        assert!(
            e[1].abs() > 2.0 * e[0].abs().max(1e-12),
            "TE₁₀ E_t not dominantly along v̂ at mid-width: {e:?}",
        );
    }

    #[test]
    fn internal_pec_septum_raises_vector_cutoff() {
        // A 2×1 guide with a vertical PEC septum at x = 1 splits into two
        // 1×1 half-guides. The dominant mode's cutoff jumps from the open
        // guide's k_c = π/2 (width 2) to k_c = π (width 1). In the vector
        // solver a PEC is `tangential E = 0` — applied to the internal
        // septum edges + nodes. At k0 = 5 (above both cutoffs) the
        // effective index drops accordingly: n_eff² = 1 − (k_c/k0)² goes
        // from ≈ 0.901 (open) to ≈ 0.605 (septum). This is the clean
        // signature that the internal-PEC constraint splits the guide.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 20, 10);
        let pec_global: Vec<bool> =
            nodes.iter().map(|p| (p[0] - 1.0).abs() < 1e-9).collect();
        let eps = vec![1.0; tris.len()];
        let k0 = 5.0;

        let pm_open =
            PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let open = solve_vector_modes(&pm_open, &eps, k0, 1);
        assert!(!open.is_empty(), "no open-guide mode");
        let neff2_open = open[0].n_eff.powi(2);

        let pm_sept = PortMesh2D::from_face(
            &nodes, &tris, [0.0, 0.0, 1.0], Some(&pec_global),
        );
        assert!(pm_sept.on_pec.iter().any(|&x| x), "no septum PEC nodes");
        let sept = solve_vector_modes(&pm_sept, &eps, k0, 1);
        assert!(!sept.is_empty(), "no septum mode (check cutoff < k0)");
        let neff2_sept = sept[0].n_eff.powi(2);

        let want_open = 1.0 - (PI / 2.0 / k0).powi(2); // ≈ 0.901
        let want_sept = 1.0 - (PI / k0).powi(2); // ≈ 0.605
        assert!(
            (neff2_open - want_open).abs() < 0.05,
            "open n_eff² = {neff2_open:.3}, want {want_open:.3}",
        );
        assert!(
            (neff2_sept - want_sept).abs() < 0.06,
            "septum n_eff² = {neff2_sept:.3}, want {want_sept:.3} — \
             internal PEC not splitting the guide",
        );
        assert!(
            neff2_sept < neff2_open - 0.1,
            "septum did not raise the cutoff: {neff2_sept:.3} vs {neff2_open:.3}",
        );
    }

    #[test]
    fn vector_solver_dielectric_filled_neff() {
        // A 2×1 guide fully filled with ε_r = 4: the TE₁₀ dispersion
        // becomes β² = ε_r k0² − k_c², so n_eff² = ε_r − (k_c/k0)² with
        // k_c = π/a = π/2. This validates the ε_r coupling in the
        // assembly (mass + longitudinal terms).
        let (nodes, tris) = rect_mesh(2.0, 1.0, 20, 10);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let eps = vec![4.0; pm.tris.len()];
        let k0 = 3.0;
        let modes = solve_vector_modes(&pm, &eps, k0, 1);
        assert!(!modes.is_empty(), "no propagating mode in ε=4 guide");
        let kc = PI / 2.0;
        let want = (4.0 - (kc / k0).powi(2)).sqrt();
        let got = modes[0].n_eff;
        let rel = (got - want).abs() / want;
        assert!(
            rel < 0.05,
            "ε=4 TE₁₀ n_eff = {got:.4}, want {want:.4} (rel {rel:.3})",
        );
    }

    #[test]
    fn vector_solver_inhomogeneous_neff_is_bracketed() {
        // Half-air (ε=1) / half-dielectric (ε=4) 2×1 guide: the dominant
        // mode's effective index must lie strictly between the all-air and
        // all-dielectric results (a partially-filled guide's n_eff is
        // bracketed by its homogeneous limits) — the qualitative signature
        // of a correct inhomogeneous (microstrip-class) hybrid solve.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 20, 10);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        // ε = 4 in the lower half (v < 0.5), air above.
        let eps: Vec<f64> = pm
            .tris
            .iter()
            .map(|&t| {
                let vc = (pm.nodes[t[0]][1]
                    + pm.nodes[t[1]][1]
                    + pm.nodes[t[2]][1])
                    / 3.0;
                if vc < 0.5 { 4.0 } else { 1.0 }
            })
            .collect();
        let k0 = 3.0;
        let modes = solve_vector_modes(&pm, &eps, k0, 1);
        assert!(!modes.is_empty(), "no propagating mode in half-filled guide");
        let neff2 = modes[0].n_eff.powi(2);
        let kc = PI / 2.0;
        let lo = 1.0 - (kc / k0).powi(2); // all-air n_eff²
        let hi = 4.0 - (kc / k0).powi(2); // all-dielectric n_eff²
        assert!(
            neff2 > lo && neff2 < hi,
            "half-filled n_eff² = {neff2:.3} not bracketed by ({lo:.3}, {hi:.3})",
        );
    }

    #[test]
    fn numerical_te10_profile_matches_the_analytic_shape() {
        // The numerical TE₁₀ mode of a 2×1 guide must reproduce the
        // analytic transverse-E shape E_y ∝ sin(πx/a): a peak at
        // mid-width, vanishing at the side walls, purely along v̂ = ŷ.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 24, 12);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let modes = solve_modes(&pm, ModeKind::Te, 1);
        let nm = NumericalMode::from_scalar(pm, &modes[0], ModeKind::Te);
        // Mid-width (x = a/2 = 1): |e_t| near the unit peak, along ŷ.
        let mid = nm.e_profile([1.0, 0.5, 0.0]);
        let mag = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
        assert!(mag > 0.9, "mid-width |e_t| = {mag:.3}, expected ≈ 1");
        assert!(
            mid[1].abs() > 0.9 && mid[0].abs() < 0.15,
            "TE₁₀ e_t not along ŷ: {mid:?}",
        );
        // Near a side wall (x ≈ 0.05): the field should be small.
        let wall = nm.e_profile([0.05, 0.5, 0.0]);
        let wmag =
            (wall[0] * wall[0] + wall[1] * wall[1] + wall[2] * wall[2]).sqrt();
        assert!(wmag < 0.3, "near-wall |e_t| = {wmag:.3}, expected ≪ 1");
        // h_t ⟂ e_t and the cutoff matches.
        let h = nm.h_profile([1.0, 0.5, 0.0]);
        let ehdot = mid[0] * h[0] + mid[1] * h[1] + mid[2] * h[2];
        assert!(ehdot.abs() < 1e-9, "e_t·h_t = {ehdot:.3e}");
        assert!((nm.cutoff() - PI / 2.0).abs() / (PI / 2.0) < 0.04);
    }

    #[test]
    fn microstrip_quasi_tem_natural_scale() {
        // Dimensionless microstrip-like cross-section: width 2, height 1.
        // Substrate band y < 0.1 with eps=3.55, air above with eps=1.
        // Trace as internal-PEC line at y=0.1, x in [0.9, 1.1]. k0=3 — well
        // below the homogeneous TE10 cutoff (k_c = pi/2 = 1.571), so the
        // ONLY propagating mode is the conductor-supported quasi-TEM.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 60, 40);
        let mut pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let eps: Vec<f64> = pm.tris.iter().map(|&t| {
            let yc = (pm.nodes[t[0]][1] + pm.nodes[t[1]][1] + pm.nodes[t[2]][1]) / 3.0;
            if yc < 0.1 { 3.55 } else { 1.0 }
        }).collect();
        // Mark trace nodes on internal PEC.
        let mut on_pec = vec![false; pm.nodes.len()];
        for (i, n) in pm.nodes.iter().enumerate() {
            if (n[1] - 0.1).abs() < 1e-6 && (n[0] - 1.0).abs() < 0.1 + 1e-6 {
                on_pec[i] = true;
            }
        }
        pm.on_pec = on_pec;
        let n_pec = pm.on_pec.iter().filter(|&&b| b).count();
        assert!(n_pec >= 2, "trace PEC nodes = {n_pec}, expected at least 2");

        let k0 = 3.0;
        let modes = solve_vector_modes(&pm, &eps, k0, 3);
        eprintln!("[microstrip natural-scale] k0={k0}, eps_max=3.55, {} modes:", modes.len());
        for (i, m) in modes.iter().enumerate() {
            eprintln!("  mode[{i}] n_eff = {:.4}", m.n_eff);
        }
        assert!(!modes.is_empty(), "no microstrip quasi-TEM mode");
        // Quasi-TEM n_eff is bracketed by sqrt(eff_eps) where eff_eps lies
        // between 1 (no substrate) and 3.55 (full substrate fill).
        let n_eff = modes[0].n_eff;
        assert!(n_eff > 1.0 && n_eff < 3.55_f64.sqrt(),
            "n_eff = {n_eff:.3} not bracketed (1, 1.884)");
    }

    #[test]
    fn te_modes_of_a_rectangular_guide_match_analytic() {
        // TE_mn cutoff of an a×b guide: k_c = π·√((m/a)² + (n/b)²).
        // Dominant TE₁₀ of a 2×1 guide → k_c = π/2.
        let (nodes, tris) = rect_mesh(2.0, 1.0, 24, 12);
        let pm = PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None);
        let modes = solve_modes(&pm, ModeKind::Te, 1);
        assert!(!modes.is_empty(), "no TE mode found");
        let kc = modes[0].k_c;
        let want = PI / 2.0;
        let rel = (kc - want).abs() / want;
        assert!(rel < 0.04, "TE₁₀ k_c = {kc:.4}, want {want:.4} (rel {rel:.3})");
    }
}
