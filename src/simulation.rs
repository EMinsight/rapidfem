//! High-level Simulation API — owns a mesh + parsed config and exposes callable
//! methods for sweep, eigenmode, and far-field. The single entry point used by
//! both the CLI (main.rs), the Python bindings (PyO3), and the WASM wrapper.
//!
//! Construction is split from execution so that callers can inspect/modify the
//! pre-built ports and materials before solving.

use num_complex::Complex64 as C64;
use std::collections::HashMap;

use crate::basis::Nedelec2Basis;
use crate::config::{Config, PortConfig};
use crate::constants::{C0, EPS0, MU0, PI};
use crate::eigenmode::Eigenmode;
use crate::farfield::RadiationPattern;
use crate::interp;
use crate::materials::{self, Material, PmlRegion};
use crate::mesh::Mesh;
use crate::port::Port;
use crate::sparam::{sparam_voltage_line, sparam_waveport};
use crate::waveguide::{
    cs_from_origin_zaxis, detect_rect_port, lumped_port_dims, AbsorbingBoundary, CoaxPort,
    FloquetPort, LumpedElement, LumpedPort, RectWaveguide, SurfaceImpedance, UserDefinedPort,
};

/// Result of a frequency sweep.
pub struct SweepResult {
    pub frequencies: Vec<f64>,
    /// S-parameters: `[freq_idx][port_obs][port_exc]`. Only driven ports.
    pub sparams: Vec<Vec<Vec<C64>>>,
    /// FEM E-field solutions: `[freq_idx][port_exc][dof]`.
    pub solutions: Vec<Vec<Vec<C64>>>,
    /// Number of driven ports (matches the inner dimension of `sparams`).
    pub n_driven: usize,
    /// Total wall-clock for the sweep (s).
    pub solve_time_s: f64,
}

/// Simulation context: a mesh + parsed config + pre-built BC objects.
pub struct Simulation {
    pub mesh: Mesh,
    pub basis: Nedelec2Basis,
    pub config: Config,
    pub ports: Vec<Box<dyn Port>>,
    pub port_tris: Vec<Vec<usize>>,
    pub pec_tris: Vec<usize>,
    pub materials: Vec<Material>,
    pub pml_regions: Vec<PmlRegion>,
    /// Lumped-port voltage integration lines, keyed by port index.
    pub lumped_lines: HashMap<usize, Vec<Vec<[f64; 3]>>>,
}

impl Simulation {
    /// Build a `Simulation` from in-memory mesh bytes and a TOML config string.
    /// Boundary-friendly entry point (no std::fs use), suitable for Python / WASM bindings.
    pub fn from_bytes(mesh_bytes: &[u8], config_toml: &str) -> Result<Self, String> {
        let mesh = crate::mesh_io::parse_mesh_bytes(mesh_bytes)?;
        let config = crate::config::parse_config(config_toml)?;
        Ok(Self::new(mesh, config))
    }

    /// Build a `Simulation` from an owned mesh and a parsed config. All BC objects
    /// (ports, PEC, materials, PML, lumped integration lines) are constructed up-front.
    pub fn new(mesh: Mesh, config: Config) -> Self {
        let basis = Nedelec2Basis::new(&mesh);
        eprintln!(
            "RapidFEM — {} tets, {} DOFs",
            mesh.n_tets(),
            basis.n_field
        );

        let (ports, port_tris) = build_ports(&mesh, &config);
        let pec_tris = build_pec_tris(&mesh, &config);
        let materials = build_materials(&mesh, &config);
        let pml_regions = build_pml_regions(&mesh, &config);
        let lumped_lines = build_lumped_lines(&mesh, &ports, &port_tris);

        Simulation {
            mesh,
            basis,
            config,
            ports,
            port_tris,
            pec_tris,
            materials,
            pml_regions,
            lumped_lines,
        }
    }

    fn ports_dyn(&self) -> Vec<&dyn Port> {
        self.ports.iter().map(|b| b.as_ref()).collect()
    }

    fn port_tris_slices(&self) -> Vec<&[usize]> {
        self.port_tris.iter().map(|v| v.as_slice()).collect()
    }

    fn frequencies(&self) -> Vec<f64> {
        self.config.frequency.frequencies()
    }

    fn materials_opt(&self) -> Option<&[Material]> {
        if self.materials.is_empty() {
            None
        } else {
            Some(self.materials.as_slice())
        }
    }

    fn pml_opt(&self) -> Option<&[PmlRegion]> {
        if self.pml_regions.is_empty() {
            None
        } else {
            Some(self.pml_regions.as_slice())
        }
    }

    /// For a single frequency's solution vector, return |E| (V/m) averaged
    /// at every mesh node by sampling the Nedelec-2 basis in each tet that
    /// contains the node and averaging the resulting magnitudes.
    /// Returned `Vec<f32>` has length `mesh.n_nodes()`.
    pub fn nodal_field_magnitudes(&self, solution: &[C64]) -> Vec<f32> {
        let n_nodes = self.mesh.n_nodes();
        let mut sum = vec![0.0f64; n_nodes];
        let mut count = vec![0u32; n_nodes];
        // Sample at each tet's centroid; assign that magnitude to each of its
        // 4 vertices. Cheap, gives a smooth nodal field via averaging.
        for ti in 0..self.mesh.n_tets() {
            let tet = &self.mesh.tets[ti];
            let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
            for k in 0..4 {
                let p = self.mesh.nodes[tet[k]];
                cx += p[0]; cy += p[1]; cz += p[2];
            }
            cx /= 4.0; cy /= 4.0; cz /= 4.0;
            let (ex, ey, ez) = crate::interp::eval_field_in_tet(
                &self.mesh, &self.basis, solution, ti, cx, cy, cz,
            );
            let mag = (ex.norm_sqr() + ey.norm_sqr() + ez.norm_sqr()).sqrt();
            for k in 0..4 {
                sum[tet[k]] += mag;
                count[tet[k]] += 1;
            }
        }
        sum.into_iter()
            .zip(count.into_iter())
            .map(|(s, c)| if c == 0 { 0.0 } else { (s / c as f64) as f32 })
            .collect()
    }

    /// Per-node phasor terms `(A, B, C)` for animated `|E(t)|²` rendering:
    ///
    ///   A = |Re(E)|² = Re_x² + Re_y² + Re_z²
    ///   B = |Im(E)|² = Im_x² + Im_y² + Im_z²
    ///   C = Re(E) · Im(E)   (real dot product)
    ///
    /// Then `|E(x,t)|² = A·cos²(ωt) + B·sin²(ωt) − 2·C·sin(ωt)·cos(ωt)`,
    /// which lets the viewer's shader modulate one uniform (phase) and
    /// render a propagating wave without any new field evaluations.
    pub fn nodal_field_phasor_terms(&self, solution: &[C64]) -> Vec<[f32; 3]> {
        let n_nodes = self.mesh.n_nodes();
        let mut sum = vec![[0.0f64; 3]; n_nodes];
        let mut count = vec![0u32; n_nodes];
        for ti in 0..self.mesh.n_tets() {
            let tet = &self.mesh.tets[ti];
            let mut cx = 0.0; let mut cy = 0.0; let mut cz = 0.0;
            for k in 0..4 {
                let p = self.mesh.nodes[tet[k]];
                cx += p[0]; cy += p[1]; cz += p[2];
            }
            cx /= 4.0; cy /= 4.0; cz /= 4.0;
            let (ex, ey, ez) = crate::interp::eval_field_in_tet(
                &self.mesh, &self.basis, solution, ti, cx, cy, cz,
            );
            let a = ex.re * ex.re + ey.re * ey.re + ez.re * ez.re;
            let b = ex.im * ex.im + ey.im * ey.im + ez.im * ez.im;
            let c = ex.re * ex.im + ey.re * ey.im + ez.re * ez.im;
            for k in 0..4 {
                sum[tet[k]][0] += a;
                sum[tet[k]][1] += b;
                sum[tet[k]][2] += c;
                count[tet[k]] += 1;
            }
        }
        sum.into_iter().zip(count.into_iter()).map(|(s, c)| {
            if c == 0 { [0.0, 0.0, 0.0] }
            else {
                let inv = 1.0 / c as f64;
                [(s[0] * inv) as f32, (s[1] * inv) as f32, (s[2] * inv) as f32]
            }
        }).collect()
    }

    /// Run a frequency sweep and extract S-parameters.
    pub fn run_sweep(&self) -> SweepResult {
        let frequencies = self.frequencies();
        let port_dyn = self.ports_dyn();
        let port_tri_refs = self.port_tris_slices();
        let n_driven = port_dyn.iter().filter(|p| p.is_driven()).count();

        let t0 = web_time::Instant::now();
        let results = crate::assembly::frequency_sweep_with_pml(
            &self.mesh,
            &self.basis,
            &port_dyn,
            &port_tri_refs,
            &self.pec_tris,
            &frequencies,
            self.materials_opt(),
            self.pml_opt(),
        );
        let solve_time_s = t0.elapsed().as_secs_f64();

        let sparams = self.extract_sparams(&port_dyn, &port_tri_refs, &frequencies, &results, n_driven);

        let solutions: Vec<Vec<Vec<C64>>> = results
            .into_iter()
            .map(|r| r.solutions.into_iter().collect())
            .collect();

        SweepResult {
            frequencies,
            sparams,
            solutions,
            n_driven,
            solve_time_s,
        }
    }

    fn extract_sparams(
        &self,
        port_dyn: &[&dyn Port],
        port_tri_refs: &[&[usize]],
        frequencies: &[f64],
        results: &[crate::assembly::SolveResult],
        n_driven: usize,
    ) -> Vec<Vec<Vec<C64>>> {
        let driven_indices: Vec<usize> = (0..port_dyn.len())
            .filter(|&i| port_dyn[i].is_driven())
            .collect();

        let mut all_sparams = Vec::with_capacity(frequencies.len());
        for (fi, freq_result) in results.iter().enumerate() {
            let k0 = 2.0 * PI * frequencies[fi] / C0;
            let mut freq_s = vec![vec![C64::new(0.0, 0.0); n_driven]; n_driven];

            let grid = interp::TetGrid::new(&self.mesh);
            for (exc_idx, sol) in freq_result.solutions.iter().enumerate() {
                let fieldf = |x: f64, y: f64, z: f64| -> (C64, C64, C64) {
                    match grid.find_containing_tet(&self.mesh, x, y, z) {
                        Some(tet) => interp::eval_field_in_tet(&self.mesh, &self.basis, sol, tet, x, y, z),
                        None => (C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)),
                    }
                };
                for (obs_idx, &obs_pi) in driven_indices.iter().enumerate() {
                    let active = obs_idx == exc_idx;
                    let s = if let (true, Some(lines), Some((_, z0, v_inc))) = (
                        port_dyn[obs_pi].is_lumped(),
                        self.lumped_lines.get(&obs_pi),
                        port_dyn[obs_pi].lumped_voltage_params(),
                    ) {
                        let n_lines = lines.len() as f64;
                        let mut s_sum = C64::new(0.0, 0.0);
                        for line_pts in lines {
                            s_sum += sparam_voltage_line(v_inc, z0, active, &fieldf, line_pts);
                        }
                        s_sum / C64::from(n_lines)
                    } else {
                        let obs_tris: Vec<[usize; 3]> = port_tri_refs[obs_pi]
                            .iter()
                            .map(|&ti| self.mesh.tris[ti])
                            .collect();
                        sparam_waveport(&self.mesh.nodes, &obs_tris, port_dyn[obs_pi], k0, active, &fieldf, 4)
                    };
                    freq_s[obs_idx][exc_idx] = s;
                }
            }
            all_sparams.push(freq_s);
        }
        all_sparams
    }

    /// Run an eigenmode analysis (requires `config.eigenmode` to be set).
    pub fn run_eigenmode(&self) -> Vec<Eigenmode> {
        let eig = self.config.eigenmode.as_ref().expect("config.eigenmode not set");
        crate::eigenmode::solve_eigenmode(
            &self.mesh,
            &self.basis,
            &self.pec_tris,
            self.materials_opt(),
            eig.target_frequency,
            eig.n_modes,
        )
    }

    /// Radiation efficiency η = 1 − Σ |S_i1|² for the first driven port at the given freq.
    /// Used as the gain-offset for far-field. Returns None if no driven ports / no S data.
    pub fn radiation_efficiency(&self, result: &SweepResult, freq_idx: usize) -> Option<f64> {
        let s = result.sparams.get(freq_idx)?;
        if s.is_empty() || s[0].is_empty() {
            return None;
        }
        let s11_sum_sq: f64 = s.iter().filter_map(|row| row.first()).map(|s| s.norm_sqr()).sum();
        Some((1.0 - s11_sum_sq).clamp(0.0, 1.0))
    }

    /// Interpolate the FEM E-field at each mesh node for a given (freq_idx, port_idx).
    /// Returns a flat `Vec<C64>` of length `3 * n_nodes` (interleaved Ex, Ey, Ez per node).
    /// Used by the Python pyvista exporter.
    pub fn field_at_nodes(&self, result: &SweepResult, freq_idx: usize, port_idx: usize) -> Option<Vec<C64>> {
        let solution = result.solutions.get(freq_idx).and_then(|s| s.get(port_idx))?;
        Some(self.eval_dofs_at_nodes(solution))
    }

    /// Same shape as `field_at_nodes` but for an eigenmode's DOF vector.
    /// The mode field is a free-field eigenfunction (not normalised to a
    /// driving port) — visualisation libraries typically rescale to a
    /// peak magnitude. Returns `None` if the mode's DOF vector is empty
    /// (defensive — `run_eigenmode` never produces empty modes).
    pub fn eigenmode_field_at_nodes(&self, mode: &Eigenmode) -> Option<Vec<C64>> {
        if mode.field.is_empty() {
            return None;
        }
        Some(self.eval_dofs_at_nodes(&mode.field))
    }

    /// Common interior — node → first-adjacent tet → barycentric eval.
    /// Both ``field_at_nodes`` and ``eigenmode_field_at_nodes`` route here
    /// so the per-node node→tet table is built the same way in both paths.
    fn eval_dofs_at_nodes(&self, solution: &[C64]) -> Vec<C64> {
        let n_nodes = self.mesh.n_nodes();

        // Node → adjacent tet (first one wins, matches vtk_export behaviour).
        let mut node_to_tet = vec![usize::MAX; n_nodes];
        for (itet, tet) in self.mesh.tets.iter().enumerate() {
            for &ni in tet {
                if node_to_tet[ni] == usize::MAX {
                    node_to_tet[ni] = itet;
                }
            }
        }

        let mut out: Vec<C64> = Vec::with_capacity(3 * n_nodes);
        for ni in 0..n_nodes {
            let tet_idx = node_to_tet[ni];
            if tet_idx == usize::MAX {
                out.extend_from_slice(&[C64::new(0.0, 0.0); 3]);
                continue;
            }
            let p = self.mesh.nodes[ni];
            let (ex, ey, ez) = crate::interp::eval_field_in_tet(
                &self.mesh, &self.basis, solution, tet_idx, p[0], p[1], p[2],
            );
            out.push(ex);
            out.push(ey);
            out.push(ez);
        }
        out
    }

    /// Per-tet conductivity σ (S/m), accumulated from `materials`. Tets not
    /// covered by any material default to 0 (PEC/dielectric — no volumetric
    /// conduction current).
    fn per_tet_sigma(&self) -> Vec<f64> {
        let mut sigma = vec![0.0f64; self.mesh.n_tets()];
        for mat in &self.materials {
            if mat.cond == 0.0 { continue; }
            for &ti in &mat.tet_indices {
                sigma[ti] = mat.cond;
            }
        }
        sigma
    }

    /// Per-tet relative permeability μ_r, default 1.0 where no material applies.
    fn per_tet_mur(&self) -> Vec<f64> {
        let mut mur = vec![1.0f64; self.mesh.n_tets()];
        for mat in &self.materials {
            for &ti in &mat.tet_indices {
                mur[ti] = mat.ur;
            }
        }
        mur
    }

    /// Build the node → adjacent-tet map (first tet wins). Shared by all the
    /// per-node samplers below so they pick the same tet at material interfaces.
    fn node_to_tet_map(&self) -> Vec<usize> {
        let n_nodes = self.mesh.n_nodes();
        let mut node_to_tet = vec![usize::MAX; n_nodes];
        for (itet, tet) in self.mesh.tets.iter().enumerate() {
            for &ni in tet {
                if node_to_tet[ni] == usize::MAX {
                    node_to_tet[ni] = itet;
                }
            }
        }
        node_to_tet
    }

    /// Conduction current density J = σE at each mesh node, in (A/m²).
    /// Returns `Vec<C64>` of length `3 · n_nodes` (interleaved Jx, Jy, Jz).
    /// Zero in PEC / dielectric regions (σ = 0). Used by the J-field viz.
    pub fn current_density_at_nodes(&self, result: &SweepResult, freq_idx: usize, port_idx: usize) -> Option<Vec<C64>> {
        let solution = result.solutions.get(freq_idx).and_then(|s| s.get(port_idx))?;
        let sigma = self.per_tet_sigma();
        let n_nodes = self.mesh.n_nodes();
        let node_to_tet = self.node_to_tet_map();
        let mut out: Vec<C64> = Vec::with_capacity(3 * n_nodes);
        for ni in 0..n_nodes {
            let tet_idx = node_to_tet[ni];
            if tet_idx == usize::MAX || sigma[tet_idx] == 0.0 {
                out.extend_from_slice(&[C64::new(0.0, 0.0); 3]);
                continue;
            }
            let p = self.mesh.nodes[ni];
            let (ex, ey, ez) = crate::interp::eval_field_in_tet(
                &self.mesh, &self.basis, solution, tet_idx, p[0], p[1], p[2],
            );
            let s = C64::from(sigma[tet_idx]);
            out.push(ex * s);
            out.push(ey * s);
            out.push(ez * s);
        }
        Some(out)
    }

    /// Magnetic field H = ∇×E / (jωμ₀μ_r) at each mesh node, in (A/m).
    /// Returns `Vec<C64>` of length `3 · n_nodes` (interleaved Hx, Hy, Hz).
    /// Uses the analytic Nédélec-2 curl (constant per tet at the centroid).
    pub fn h_field_at_nodes(&self, result: &SweepResult, freq_idx: usize, port_idx: usize) -> Option<Vec<C64>> {
        let solution = result.solutions.get(freq_idx).and_then(|s| s.get(port_idx))?;
        let freq = *result.frequencies.get(freq_idx)?;
        let omega = 2.0 * PI * freq;
        let mur = self.per_tet_mur();
        let n_nodes = self.mesh.n_nodes();
        let node_to_tet = self.node_to_tet_map();
        let j = C64::new(0.0, 1.0);
        let mut out: Vec<C64> = Vec::with_capacity(3 * n_nodes);
        for ni in 0..n_nodes {
            let tet_idx = node_to_tet[ni];
            if tet_idx == usize::MAX {
                out.extend_from_slice(&[C64::new(0.0, 0.0); 3]);
                continue;
            }
            let curl = crate::interp::eval_curl_in_tet(
                &self.mesh, &self.basis, solution, tet_idx,
            );
            let denom = j * C64::from(omega * MU0 * mur[tet_idx]);
            out.push(curl[0] / denom);
            out.push(curl[1] / denom);
            out.push(curl[2] / denom);
        }
        Some(out)
    }

    /// Compute the far-field at a given (freq_idx, exc_port_idx). NFFT surface = config.output.nfft_tag
    /// (auto-detected ABC tag if not specified). PEC surfaces from config.pec.tags are included to close
    /// the integration boundary.
    pub fn compute_farfield(
        &self,
        result: &SweepResult,
        freq_idx: usize,
        exc_port_idx: usize,
        n_theta: usize,
        n_phi: usize,
    ) -> Option<RadiationPattern> {
        let solution = result.solutions.get(freq_idx).and_then(|s| s.get(exc_port_idx))?;
        let nfft_tag = self.config.output.nfft_tag.unwrap_or_else(|| {
            for pc in &self.config.ports {
                if let PortConfig::Abc { tag, .. } = pc {
                    return *tag;
                }
            }
            2
        });
        let nfft_tris = self.mesh.tris_for_tag(nfft_tag).to_vec();
        if nfft_tris.is_empty() {
            return None;
        }
        let pec_nfft: Vec<usize> = self
            .config
            .pec
            .tags
            .iter()
            .flat_map(|&t| self.mesh.tris_for_tag(t).to_vec())
            .collect();
        let efficiency = self.radiation_efficiency(result, freq_idx);

        Some(crate::farfield::compute_farfield_full(
            &self.mesh,
            &self.basis,
            solution,
            &nfft_tris,
            &pec_nfft,
            result.frequencies[freq_idx],
            n_theta,
            n_phi,
            4,
            efficiency,
        ))
    }
}

// ============================================================================
// Construction helpers — extracted from main.rs's prior orchestration
// ============================================================================

fn build_ports(mesh: &Mesh, config: &Config) -> (Vec<Box<dyn Port>>, Vec<Vec<usize>>) {
    let mut ports: Vec<Box<dyn Port>> = Vec::new();
    let mut port_tris: Vec<Vec<usize>> = Vec::new();

    for pc in &config.ports {
        match pc {
            PortConfig::Rectangular { tag, width, height, mode, er, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping port", tag);
                    continue;
                }
                let (cs, det_w, det_h) = detect_rect_port(mesh, &tri_ids);
                let w = if *width > 0.0 { *width } else { det_w };
                let h = if *height > 0.0 { *height } else { det_h };
                let port_num = ports.len() + 1;
                let port = RectWaveguide {
                    port_number: port_num,
                    power: *power,
                    mode: (mode[0], mode[1]),
                    er: *er,
                    polarization: 1.0,
                    dims: (w, h),
                    cs,
                };
                eprintln!("  Port {}: rectangular, tag={}, TE{}{}, dims=({:.2}mm, {:.2}mm), er={:.1}",
                    port_num, tag, mode[0], mode[1], w * 1e3, h * 1e3, er);
                port_tris.push(tri_ids);
                ports.push(Box::new(port));
            }
            PortConfig::Coax { tag, ri, ro, origin, z_axis, er, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping CoaxPort", tag);
                    continue;
                }
                let (cs_detected, _, _) = detect_rect_port(mesh, &tri_ids);
                let org = origin.unwrap_or(cs_detected.origin);
                let zax = z_axis.unwrap_or(cs_detected.zax);
                let cs = cs_from_origin_zaxis(org, zax);
                let port_num = ports.len() + 1;
                let port = CoaxPort {
                    port_number: port_num,
                    power: *power, er: *er, ri: *ri, ro: *ro, cs,
                };
                eprintln!("  Port {}: coax, tag={}, Ri={:.3}mm, Ro={:.3}mm, εr={:.2}, Z₀={:.2}Ω",
                    port_num, tag, ri * 1e3, ro * 1e3, er, port.port_z());
                port_tris.push(tri_ids);
                ports.push(Box::new(port));
            }
            PortConfig::Lumped { tag, z0, direction, width, height, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping port", tag);
                    continue;
                }
                let port_num = ports.len() + 1;
                let (det_w, det_h) = lumped_port_dims(mesh, &tri_ids, direction);
                let w = if *width > 0.0 { *width } else { det_w };
                let h = if *height > 0.0 { *height } else { det_h };
                let port = LumpedPort {
                    port_number: port_num,
                    power: *power,
                    z0: *z0,
                    width: w,
                    height: h,
                    direction: *direction,
                };
                eprintln!("  Port {}: lumped, tag={}, Z0={:.0}Ω, dir=({:.1},{:.1},{:.1})",
                    port_num, tag, z0, direction[0], direction[1], direction[2]);
                port_tris.push(tri_ids);
                ports.push(Box::new(port));
            }
            PortConfig::UserDefined { tag, e_field, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping UserDefined", tag);
                    continue;
                }
                let port_num = ports.len() + 1;
                let port = UserDefinedPort::from_constant(port_num, *power, *e_field);
                eprintln!("  Port {}: user_defined, tag={}, E=({:.3},{:.3},{:.3}), P={:.2}W",
                    port_num, tag, e_field[0], e_field[1], e_field[2], power);
                port_tris.push(tri_ids);
                ports.push(Box::new(port));
            }
            PortConfig::Floquet { tag, scan_theta_deg, scan_phi_deg, mode_nr, er, power } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping FloquetPort", tag);
                    continue;
                }
                let (cs_detected, det_w, det_h) = detect_rect_port(mesh, &tri_ids);
                let area = det_w * det_h;
                let port_num = ports.len() + 1;
                let port = FloquetPort {
                    port_number: port_num,
                    power: *power, er: *er, area,
                    scan_theta: scan_theta_deg.to_radians(),
                    scan_phi: scan_phi_deg.to_radians(),
                    mode_nr: *mode_nr,
                    cs: cs_detected,
                };
                eprintln!("  Port {}: floquet, tag={}, mode={} ({}), θ={:.1}°, φ={:.1}°, A={:.2}mm²",
                    port_num, tag, mode_nr,
                    if *mode_nr == 1 { "TE/S" } else { "TM/P" },
                    scan_theta_deg, scan_phi_deg, area * 1e6);
                port_tris.push(tri_ids);
                ports.push(Box::new(port));
            }
            PortConfig::Pmc { tag } => {
                let tri_ids = mesh.tris_for_tag(*tag);
                eprintln!("  PMC: tag={}, {} triangles (natural BC)", tag, tri_ids.len());
            }
            PortConfig::LumpedElement { tag, r, l, c, width, height, direction } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping LumpedElement", tag);
                    continue;
                }
                let (det_w, det_h) = lumped_port_dims(mesh, &tri_ids, direction);
                let w = if *width > 0.0 { *width } else { det_w };
                let h = if *height > 0.0 { *height } else { det_h };
                let bc = LumpedElement { r: *r, l: *l, c: *c, width: w, height: h };
                eprintln!("  LumpedElement: tag={}, R={:.2}Ω, L={:.2e}H, C={:?}F, w={:.2}mm, h={:.2}mm",
                    tag, r, l, c, w * 1e3, h * 1e3);
                port_tris.push(tri_ids);
                ports.push(Box::new(bc));
            }
            PortConfig::SurfaceImpedance { tag, conductivity, mur, er, thickness, zs } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping SurfaceImpedance", tag);
                    continue;
                }
                let bc = if let Some(zs_arr) = zs {
                    let mut s = SurfaceImpedance::from_zs(C64::new(zs_arr[0], zs_arr[1]));
                    s.mur = *mur; s.er = *er; s.thickness = *thickness;
                    s
                } else {
                    let mut s = SurfaceImpedance::from_conductivity(*conductivity);
                    s.mur = *mur; s.er = *er; s.thickness = *thickness;
                    s
                };
                eprintln!("  SurfaceImpedance: tag={}, σ={:.2e}S/m, μr={:.2}, εr={:.2}, t={:?}",
                    tag, conductivity, mur, er, thickness);
                port_tris.push(tri_ids);
                ports.push(Box::new(bc));
            }
            PortConfig::Abc { tag, order, abctype } => {
                let tri_ids = mesh.tris_for_tag(*tag).to_vec();
                if tri_ids.is_empty() {
                    eprintln!("  WARNING: tag {} has no triangles, skipping ABC", tag);
                    continue;
                }
                let abc_char = abctype.chars().next().unwrap_or('B');
                let abc = AbsorbingBoundary::new(*order, abc_char);
                eprintln!("  ABC: tag={}, order={}, type={}", tag, order, abctype);
                port_tris.push(tri_ids);
                ports.push(Box::new(abc));
            }
        }
    }

    (ports, port_tris)
}

fn build_pec_tris(mesh: &Mesh, config: &Config) -> Vec<usize> {
    let mut pec_tris = Vec::new();
    for &tag in &config.pec.tags {
        pec_tris.extend_from_slice(mesh.tris_for_tag(tag));
    }
    pec_tris
}

fn build_materials(mesh: &Mesh, config: &Config) -> Vec<Material> {
    config.materials.iter().map(|mc| {
        let tet_indices = mesh
            .vtag_to_tet
            .get(&mc.volume_tag)
            .map(|v| v.clone())
            .unwrap_or_default();
        if tet_indices.is_empty() {
            eprintln!("  WARNING: volume tag {} has no tets", mc.volume_tag);
        } else {
            eprintln!("  Material: tag={}, er={:.2}, ur={:.2}, tand={:.4}, cond={:.2e}, {} tets",
                mc.volume_tag, mc.er, mc.ur, mc.tand, mc.conductivity, tet_indices.len());
        }
        let dispersion = if let Some(d) = &mc.debye {
            materials::Dispersion::Debye {
                er_inf: d.er_inf, er_static: d.er_static, tau_s: d.tau_s,
            }
        } else if let Some(d) = &mc.drude {
            materials::Dispersion::Drude {
                er_inf: d.er_inf, plasma_freq_hz: d.plasma_freq_hz, damping_freq_hz: d.damping_freq_hz,
            }
        } else {
            materials::Dispersion::None
        };
        if dispersion.is_dispersive() {
            eprintln!("    (dispersive: εr(f) recomputed per frequency)");
        }
        Material {
            er: mc.er, ur: mc.ur, tand: mc.tand, cond: mc.conductivity,
            tet_indices,
            er_diag: mc.er_diag,
            ur_diag: mc.ur_diag,
            dispersion,
        }
    }).collect()
}

fn build_pml_regions(mesh: &Mesh, config: &Config) -> Vec<PmlRegion> {
    config.pml.iter().map(|pc| {
        let tet_indices = mesh
            .vtag_to_tet
            .get(&pc.volume_tag)
            .map(|v| v.clone())
            .unwrap_or_default();
        if tet_indices.is_empty() {
            eprintln!("  WARNING: PML volume tag {} has no tets", pc.volume_tag);
        } else {
            eprintln!("  PML: tag={}, dir=({:.0},{:.0},{:.0}), inner={:.3}m, t={:.3}m, n={:.1}, δmax={:.1}, {} tets",
                pc.volume_tag, pc.direction[0], pc.direction[1], pc.direction[2],
                pc.inner_face, pc.thickness, pc.exponent, pc.delta_max, tet_indices.len());
        }
        PmlRegion {
            tet_indices,
            er_base: pc.er_base,
            ur_base: pc.ur_base,
            direction: pc.direction,
            inner_face: pc.inner_face,
            thickness: pc.thickness,
            exponent: pc.exponent,
            delta_max: pc.delta_max,
        }
    }).collect()
}

/// Build EMerge-compatible lumped port integration lines: one line per min-projection
/// vertex on the port face, line goes from that vertex to (vertex + direction × height).
/// S-parameter averages over lines. See microwave_3d.py:_define_lumped_port_integration_points.
fn build_lumped_lines(
    mesh: &Mesh,
    ports: &[Box<dyn Port>],
    port_tris: &[Vec<usize>],
) -> HashMap<usize, Vec<Vec<[f64; 3]>>> {
    let mut lines_map: HashMap<usize, Vec<Vec<[f64; 3]>>> = HashMap::new();
    for (pi, port) in ports.iter().enumerate() {
        if !port.is_lumped() {
            continue;
        }
        let tri_ids = &port_tris[pi];
        let (dir, _, _) = port.lumped_voltage_params().unwrap();
        let height = port.port_height().expect("lumped port must have height");

        let mut verts: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &ti in tri_ids {
            for &vi in &mesh.tris[ti] {
                verts.insert(vi);
            }
        }

        let mut min_proj = f64::INFINITY;
        for &vi in &verts {
            let p = mesh.nodes[vi];
            let proj = p[0] * dir[0] + p[1] * dir[1] + p[2] * dir[2];
            if proj < min_proj {
                min_proj = proj;
            }
        }
        let proj_tol = 1e-9 * height.max(1.0);
        let start_verts: Vec<usize> = verts
            .iter()
            .copied()
            .filter(|&vi| {
                let p = mesh.nodes[vi];
                let proj = p[0] * dir[0] + p[1] * dir[1] + p[2] * dir[2];
                (proj - min_proj).abs() < proj_tol
            })
            .collect();

        let n_pts = 21;
        let mut lines: Vec<Vec<[f64; 3]>> = Vec::with_capacity(start_verts.len());
        for &vi in &start_verts {
            let s = mesh.nodes[vi];
            let mut pts = Vec::with_capacity(n_pts);
            for i in 0..n_pts {
                let t = i as f64 / (n_pts - 1) as f64;
                pts.push([
                    s[0] + t * dir[0] * height,
                    s[1] + t * dir[1] * height,
                    s[2] + t * dir[2] * height,
                ]);
            }
            lines.push(pts);
        }
        eprintln!("  Lumped port {}: {} integration lines × {} pts, height={:.4}mm",
            port.port_number(), lines.len(), n_pts, height * 1e3);
        lines_map.insert(pi, lines);
    }
    lines_map
}
