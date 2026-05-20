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

/// Physical curl of a vector field on a single element.
///
/// `field` holds `3·Np` values (`field[node*3 + comp]`); the result has the
/// same layout and contains `∇×field` sampled at the element nodes.
pub fn element_curl(
    re: &ReferenceElement,
    gf: &GeometricFactors,
    field: &[f64],
) -> Vec<f64> {
    let n = re.n_nodes;
    debug_assert_eq!(field.len(), 3 * n);
    let dref = [&re.diff_r, &re.diff_s, &re.diff_t];

    // pd[phys][comp][node] = ∂(field_comp)/∂x_phys.
    let mut pd = [[(); 3]; 3].map(|row| row.map(|_| vec![0.0; n]));
    for comp in 0..3 {
        // Reference derivatives of this component.
        let mut rd = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for (k, d) in dref.iter().enumerate() {
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += d[i * n + j] * field[j * 3 + comp];
                }
                rd[k][i] = acc;
            }
        }
        // Combine via the metric into physical derivatives.
        for (phys, pd_phys) in pd.iter_mut().enumerate() {
            for i in 0..n {
                pd_phys[comp][i] = gf.jacobian_inv[0][phys] * rd[0][i]
                    + gf.jacobian_inv[1][phys] * rd[1][i]
                    + gf.jacobian_inv[2][phys] * rd[2][i];
            }
        }
    }

    // curl_x = ∂Fz/∂y - ∂Fy/∂z, and cyclic.
    let mut out = vec![0.0; 3 * n];
    for i in 0..n {
        out[i * 3] = pd[1][2][i] - pd[2][1][i];
        out[i * 3 + 1] = pd[2][0][i] - pd[0][2][i];
        out[i * 3 + 2] = pd[0][1][i] - pd[1][0][i];
    }
    out
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

/// Semi-discrete DG operator for the vacuum Maxwell curl equations on a
/// tetrahedral mesh with PEC outer walls.
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
}

impl MaxwellOperator {
    /// Build the operator for a mesh at polynomial order `p` with the given
    /// upwind blend (`flux_alpha` in `[0, 1]`).
    pub fn new(mesh: &Mesh, order: usize, flux_alpha: f64) -> Self {
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
        MaxwellOperator { re, n_elem, geom, faces, flux_alpha }
    }

    /// Degrees of freedom, `6 · Np · n_elem`.
    pub fn n_dof(&self) -> usize {
        6 * self.re.n_nodes * self.n_elem
    }

    /// Evaluate `dy/dt = A·y`.
    pub fn apply(&self, y: &[f64]) -> Vec<f64> {
        let np = self.re.n_nodes;
        let nfp = self.re.n_face_nodes;
        let cols = 4 * nfp;
        let mut dy = vec![0.0; self.n_dof()];

        for e in 0..self.n_elem {
            let base = e * np * 6;
            let mut ee = vec![0.0; 3 * np];
            let mut hh = vec![0.0; 3 * np];
            for node in 0..np {
                for c in 0..3 {
                    ee[node * 3 + c] = y[base + node * 6 + c];
                    hh[node * 3 + c] = y[base + node * 6 + 3 + c];
                }
            }

            // Volume term:  dE = ∇×H,  dH = -∇×E.
            let mut de = element_curl(&self.re, &self.geom[e], &hh);
            let mut dh = element_curl(&self.re, &self.geom[e], &ee);
            for v in dh.iter_mut() {
                *v = -*v;
            }

            // Surface term — central flux.
            for f in 0..4 {
                let fi = &self.faces[e * 4 + f];
                let n = fi.normal;
                let coef = 0.5 * fi.fscale;
                for m in 0..nfp {
                    let vi = self.re.face_nodes[f][m];
                    let em = [ee[vi * 3], ee[vi * 3 + 1], ee[vi * 3 + 2]];
                    let hm = [hh[vi * 3], hh[vi * 3 + 1], hh[vi * 3 + 2]];
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
                            de[i * 3 + c] += w * (-nxjh[c] + a * pe[c]);
                            dh[i * 3 + c] += w * (nxje[c] + a * ph[c]);
                        }
                    }
                }
            }

            for node in 0..np {
                for c in 0..3 {
                    dy[base + node * 6 + c] = de[node * 3 + c];
                    dy[base + node * 6 + 3 + c] = dh[node * 3 + c];
                }
            }
        }
        dy
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

    /// Dense block-diagonal energy mass `M̃`: per element and field component a
    /// copy of `|det J_e|·M_ref`. `yᵀ M̃ y` is the discrete field energy.
    pub fn assemble_energy_mass(&self) -> Vec<f64> {
        let np = self.re.n_nodes;
        let n = self.n_dof();
        let mut m = vec![0.0; n * n];
        for e in 0..self.n_elem {
            let scale = self.geom[e].det.abs();
            let base = e * np * 6;
            for ni in 0..np {
                for nj in 0..np {
                    let mij = scale * self.re.mass[ni * np + nj];
                    for c in 0..6 {
                        m[(base + ni * 6 + c) * n + (base + nj * 6 + c)] = mij;
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
    /// state-space `A`. Entries below `1e-12` of the largest magnitude are
    /// dropped. For handing `A` to external tools / model-order reduction.
    pub fn assemble_sparse(&self) -> CsrMatrix {
        let n = self.n_dof();
        let dense = self.assemble_dense();
        let max = dense.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        let tol = 1e-12 * max;
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for i in 0..n {
            for j in 0..n {
                let v = dense[i * n + j];
                if v.abs() > tol {
                    col_idx.push(j);
                    values.push(v);
                }
            }
            row_ptr.push(values.len());
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
}
