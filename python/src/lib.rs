//! PyO3 bindings for rapidfem. Exposes `Simulation`, `SweepResult` to Python.
//!
//! Build via `maturin develop` (dev) or `maturin build --release` (wheel).
//! See `examples/wr90.py` for usage.

use num_complex::Complex64;
use numpy::{
    Complex64 as NpC64, IntoPyArray, PyArray1, PyArray2, PyArray3,
    PyReadonlyArray1,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rapidfem_fd::eigenmode::Eigenmode;
use rapidfem_fd::farfield::RadiationPattern;
use rapidfem_fd::simulation::{Simulation, SweepResult};

/// A frequency-sweep simulation. Build once, run sweeps, inspect results.
///
/// Marked `unsendable`: `Box<dyn Port>` doesn't auto-impl Send/Sync, so a Simulation
/// instance must stay on the thread that created it. Fine for typical Python use.
#[pyclass(name = "Simulation", unsendable)]
struct PySimulation {
    inner: Simulation,
}

/// Result of a frequency sweep — frequencies and S-parameters.
#[pyclass(name = "SweepResult")]
struct PySweepResult {
    inner: SweepResult,
}

/// One eigenmode of a cavity / waveguide.
#[pyclass(name = "Eigenmode", unsendable)]
struct PyEigenmode {
    inner: Eigenmode,
}

/// Far-field radiation pattern with directivity, gain, axial ratio, LCP/RCP.
#[pyclass(name = "RadiationPattern", unsendable)]
struct PyRadiationPattern {
    inner: RadiationPattern,
}

#[pymethods]
impl PySimulation {
    /// Construct a simulation by loading a gmsh `.msh` file and a TOML config from disk.
    #[staticmethod]
    fn from_files(mesh_path: &str, config_path: &str) -> PyResult<Self> {
        let config = rapidfem_fd::config::load_config(config_path)
            .map_err(|e| PyRuntimeError::new_err(format!("config: {}", e)))?;
        let mesh = rapidfem_fd::mesh_io::load_mesh(mesh_path)
            .map_err(|e| PyRuntimeError::new_err(format!("mesh: {}", e)))?;
        Ok(PySimulation { inner: Simulation::new(mesh, config) })
    }

    /// Construct from in-memory mesh bytes and a TOML config string.
    /// Useful when meshes/configs are produced programmatically (no disk I/O).
    #[staticmethod]
    fn from_bytes(mesh_bytes: &[u8], config_toml: &str) -> PyResult<Self> {
        let inner = Simulation::from_bytes(mesh_bytes, config_toml)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        Ok(PySimulation { inner })
    }

    /// Run the configured frequency sweep. Returns a SweepResult with frequencies
    /// (float64 array, shape `[n_freq]`) and S-parameters (complex128 array,
    /// shape `[n_freq, n_driven, n_driven]`).
    fn run_sweep(&self) -> PySweepResult {
        // Release the GIL so log-streaming reader threads + WS workers run
        // during the (potentially long) sweep. `Python::allow_threads`
        // wants `Send`, but Simulation is `unsendable` (Box<dyn Port> is
        // not Send). Drop down to PyO3's ffi — same effect, no Send bound.
        let inner = unsafe {
            let save = pyo3::ffi::PyEval_SaveThread();
            let r = self.inner.run_sweep();
            pyo3::ffi::PyEval_RestoreThread(save);
            r
        };
        PySweepResult { inner }
    }

    /// Number of tetrahedra in the mesh.
    #[getter]
    fn n_tets(&self) -> usize { self.inner.mesh.n_tets() }

    /// Number of degrees of freedom in the FEM basis.
    #[getter]
    fn n_dofs(&self) -> usize { self.inner.basis.n_field }

    /// Number of driven ports (i.e., ports with excitation: rect waveguide, lumped, coax, ...).
    #[getter]
    fn n_driven_ports(&self) -> usize {
        self.inner.ports.iter().filter(|p| p.is_driven()).count()
    }

    /// Run an eigenmode analysis. Requires `[eigenmode]` block in the TOML config.
    /// Returns a list of `Eigenmode` (frequency, Q, field).
    fn run_eigenmode(&self) -> PyResult<Vec<PyEigenmode>> {
        if self.inner.config.eigenmode.is_none() {
            return Err(PyRuntimeError::new_err(
                "config.eigenmode block not set in TOML",
            ));
        }
        Ok(self
            .inner
            .run_eigenmode()
            .into_iter()
            .map(|m| PyEigenmode { inner: m })
            .collect())
    }

    /// Mesh node coordinates as a `(n_nodes, 3)` float64 numpy array.
    #[getter]
    fn mesh_nodes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.mesh.n_nodes();
        let mut flat: Vec<f64> = Vec::with_capacity(n * 3);
        for p in &self.inner.mesh.nodes {
            flat.extend_from_slice(p);
        }
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), flat).expect("shape");
        arr.into_pyarray_bound(py)
    }

    /// Mesh tetrahedra as a `(n_tets, 4)` int64 numpy array of node indices.
    #[getter]
    fn mesh_tets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let n = self.inner.mesh.n_tets();
        let mut flat: Vec<i64> = Vec::with_capacity(n * 4);
        for tet in &self.inner.mesh.tets {
            for &v in tet {
                flat.push(v as i64);
            }
        }
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 4), flat).expect("shape");
        arr.into_pyarray_bound(py)
    }

    /// FEM E-field interpolated at every mesh node for a given (freq_idx, port_idx).
    /// Returns a `(n_nodes, 3)` complex128 numpy array (Ex, Ey, Ez per node).
    /// Use this with `pyvista` or any mesh-viz library for field visualization.
    fn field_at_nodes<'py>(
        &self,
        py: Python<'py>,
        result: &PySweepResult,
        freq_idx: usize,
        port_idx: usize,
    ) -> Option<Bound<'py, PyArray2<NpC64>>> {
        let flat = self.inner.field_at_nodes(&result.inner, freq_idx, port_idx)?;
        let n = self.inner.mesh.n_nodes();
        let conv: Vec<NpC64> = flat.iter().map(|c| NpC64::new(c.re, c.im)).collect();
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), conv).expect("shape");
        Some(arr.into_pyarray_bound(py))
    }

    /// Loss-equivalent current density J = σ_eff · E at every mesh node for
    /// a given (freq_idx, port_idx). `σ_eff = ω·ε₀·εᵣ·tan(δ) + σ_bulk`
    /// covers both dielectric (loss tangent) and Ohmic losses, so substrates
    /// like Rogers with tan_δ but zero bulk σ also light up. Returns a
    /// `(n_nodes, 3)` complex128 numpy array (Jx, Jy, Jz per node) in A/m².
    fn current_density_at_nodes<'py>(
        &self,
        py: Python<'py>,
        result: &PySweepResult,
        freq_idx: usize,
        port_idx: usize,
    ) -> Option<Bound<'py, PyArray2<NpC64>>> {
        let flat = self.inner.current_density_at_nodes(&result.inner, freq_idx, port_idx)?;
        let n = self.inner.mesh.n_nodes();
        let conv: Vec<NpC64> = flat.iter().map(|c| NpC64::new(c.re, c.im)).collect();
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), conv).expect("shape");
        Some(arr.into_pyarray_bound(py))
    }

    /// Magnetic field H = ∇×E / (jωμ₀μ_r) at every mesh node for a given
    /// (freq_idx, port_idx). Returns a `(n_nodes, 3)` complex128 numpy array
    /// (Hx, Hy, Hz per node) in A/m. Derived from the analytic Nédélec-2
    /// curl of the FEM solution.
    fn h_field_at_nodes<'py>(
        &self,
        py: Python<'py>,
        result: &PySweepResult,
        freq_idx: usize,
        port_idx: usize,
    ) -> Option<Bound<'py, PyArray2<NpC64>>> {
        let flat = self.inner.h_field_at_nodes(&result.inner, freq_idx, port_idx)?;
        let n = self.inner.mesh.n_nodes();
        let conv: Vec<NpC64> = flat.iter().map(|c| NpC64::new(c.re, c.im)).collect();
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), conv).expect("shape");
        Some(arr.into_pyarray_bound(py))
    }

    /// Same as ``field_at_nodes`` but for an :class:`Eigenmode`. Returns a
    /// `(n_nodes, 3)` complex128 numpy array of (Ex, Ey, Ez) at each mesh
    /// node. Field magnitude is not normalised — eigenmodes are defined up
    /// to a global scale.
    fn mode_field_at_nodes<'py>(
        &self,
        py: Python<'py>,
        mode: &PyEigenmode,
    ) -> Option<Bound<'py, PyArray2<NpC64>>> {
        let flat = self.inner.eigenmode_field_at_nodes(&mode.inner)?;
        let n = self.inner.mesh.n_nodes();
        let conv: Vec<NpC64> = flat.iter().map(|c| NpC64::new(c.re, c.im)).collect();
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), conv).expect("shape");
        Some(arr.into_pyarray_bound(py))
    }

    /// Compute the far-field radiation pattern at (freq_idx, port_idx) on a (theta, phi) grid.
    /// Returns None if the NFFT surface is empty or out-of-bounds indices.
    #[pyo3(signature = (result, freq_idx=0, port_idx=0, n_theta=91, n_phi=72))]
    fn compute_farfield(
        &self,
        result: &PySweepResult,
        freq_idx: usize,
        port_idx: usize,
        n_theta: usize,
        n_phi: usize,
    ) -> Option<PyRadiationPattern> {
        self.inner
            .compute_farfield(&result.inner, freq_idx, port_idx, n_theta, n_phi)
            .map(|p| PyRadiationPattern { inner: p })
    }
}

#[pymethods]
impl PySweepResult {
    /// Frequencies in Hz, shape `[n_freq]`, dtype float64.
    #[getter]
    fn frequencies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.frequencies.clone().into_pyarray_bound(py)
    }

    /// S-parameter matrix, shape `[n_freq, n_driven, n_driven]`, dtype complex128.
    /// Indexing: `S[freq_idx, observation_port, excitation_port]`.
    #[getter]
    fn sparams<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<NpC64>> {
        let n_freq = self.inner.frequencies.len();
        let n = self.inner.n_driven;
        let mut flat: Vec<NpC64> = Vec::with_capacity(n_freq * n * n);
        for f_mat in &self.inner.sparams {
            for row in f_mat {
                for c in row {
                    // num_complex::Complex64 ↔ numpy::Complex64 are bit-identical layout.
                    let v: Complex64 = *c;
                    flat.push(NpC64::new(v.re, v.im));
                }
            }
        }
        let arr = numpy::ndarray::Array3::from_shape_vec((n_freq, n, n), flat)
            .expect("shape matches data");
        arr.into_pyarray_bound(py)
    }

    /// Number of driven ports (S-matrix dimension).
    #[getter]
    fn n_driven(&self) -> usize { self.inner.n_driven }

    /// Total wall-clock for the sweep in seconds.
    #[getter]
    fn solve_time_s(&self) -> f64 { self.inner.solve_time_s }
}

#[pymethods]
impl PyEigenmode {
    /// Real part of the resonant frequency (Hz).
    #[getter]
    fn frequency_hz(&self) -> f64 { self.inner.frequency.re }

    /// Imaginary part of the resonant frequency (Hz). Non-zero for lossy / leaky modes.
    #[getter]
    fn frequency_imag_hz(&self) -> f64 { self.inner.frequency.im }

    /// Quality factor Q = f_re / (2 * f_im). Infinite for lossless modes.
    #[getter]
    fn q_factor(&self) -> f64 { self.inner.q_factor }

    /// E-field DOF coefficient vector for this mode (complex128).
    #[getter]
    fn field<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpC64>> {
        let conv: Vec<NpC64> = self.inner.field.iter().map(|c| NpC64::new(c.re, c.im)).collect();
        conv.into_pyarray_bound(py)
    }
}

#[pymethods]
impl PyRadiationPattern {
    /// Theta angles (radians, 0..pi).
    #[getter]
    fn theta_rad<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.theta.clone().into_pyarray_bound(py)
    }

    /// Phi angles (radians, 0..2pi).
    #[getter]
    fn phi_rad<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.phi.clone().into_pyarray_bound(py)
    }

    #[getter]
    fn directivity_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        flatten_2d(&self.inner.directivity_dbi, py)
    }

    #[getter]
    fn gain_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        flatten_2d(&self.inner.gain_dbi, py)
    }

    #[getter]
    fn axial_ratio_db<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        flatten_2d(&self.inner.axial_ratio_db, py)
    }

    #[getter]
    fn lcp_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        flatten_2d(&self.inner.lcp_dbi, py)
    }

    #[getter]
    fn rcp_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        flatten_2d(&self.inner.rcp_dbi, py)
    }

    /// Complex E_theta(theta, phi), shape `[n_phi, n_theta]`.
    #[getter]
    fn e_theta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpC64>> {
        flatten_2d_complex(&self.inner.e_theta, py)
    }

    /// Complex E_phi(theta, phi), shape `[n_phi, n_theta]`.
    #[getter]
    fn e_phi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpC64>> {
        flatten_2d_complex(&self.inner.e_phi, py)
    }

    #[getter]
    fn peak_directivity_dbi(&self) -> f64 { self.inner.peak_directivity_dbi }

    #[getter]
    fn peak_gain_dbi(&self) -> f64 { self.inner.peak_gain_dbi }

    #[getter]
    fn radiated_power(&self) -> f64 { self.inner.radiated_power }
}

fn flatten_2d<'py>(grid: &[Vec<f64>], py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
    let n_phi = grid.len();
    let n_theta = grid.first().map(|r| r.len()).unwrap_or(0);
    let mut flat: Vec<f64> = Vec::with_capacity(n_phi * n_theta);
    for row in grid {
        flat.extend_from_slice(row);
    }
    let arr = numpy::ndarray::Array2::from_shape_vec((n_phi, n_theta), flat).expect("shape");
    arr.into_pyarray_bound(py)
}

fn flatten_2d_complex<'py>(grid: &[Vec<Complex64>], py: Python<'py>) -> Bound<'py, PyArray2<NpC64>> {
    let n_phi = grid.len();
    let n_theta = grid.first().map(|r| r.len()).unwrap_or(0);
    let mut flat: Vec<NpC64> = Vec::with_capacity(n_phi * n_theta);
    for row in grid {
        for c in row {
            flat.push(NpC64::new(c.re, c.im));
        }
    }
    let arr = numpy::ndarray::Array2::from_shape_vec((n_phi, n_theta), flat).expect("shape");
    arr.into_pyarray_bound(py)
}

// --- Time-domain DGTD backend ----------------------------------------------

// The Krylov propagator tolerance lives with the other TD numerical
// constants — see `rapidfem_td::constants`.
use rapidfem_td::constants::KRYLOV_TOL;

/// Time-domain DGTD Maxwell operator (vacuum, PEC walls), built on a
/// structured box cavity. Wraps the Rust `MaxwellOperator`.
#[pyclass(name = "TdOperator")]
struct PyTdOperator {
    op: rapidfem_td::rhs::MaxwellOperator,
    /// Reused Krylov workspace — keeps `step` / `step_driven` allocation-free
    /// across a transient loop.
    krylov: rapidfem_td::propagator::KrylovWorkspace,
    /// Reused soft-source vector for `step_driven` — one DOF set per call.
    driven_b: Vec<f64>,
}

#[pymethods]
impl PyTdOperator {
    /// Build the operator on a structured box cavity `[0,lx]×[0,ly]×[0,lz]`
    /// (`nx·ny·nz` cells) at polynomial `order`. `flux_alpha`: 0 = central
    /// (energy-conserving), 1 = upwind.
    #[new]
    #[pyo3(signature = (nx, ny, nz, lx, ly, lz, order, flux_alpha = 1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        lx: f64,
        ly: f64,
        lz: f64,
        order: usize,
        flux_alpha: f64,
    ) -> Self {
        let mesh = rapidfem_td::mesh_gen::structured_box(nx, ny, nz, lx, ly, lz);
        let op = rapidfem_td::rhs::MaxwellOperator::new(&mesh, order, flux_alpha);
        PyTdOperator {
            op,
            krylov: rapidfem_td::propagator::KrylovWorkspace::new(),
            driven_b: Vec::new(),
        }
    }

    /// Build the operator from in-memory gmsh `.msh` bytes — the path for
    /// arbitrary unstructured meshes produced by the geometry API.
    ///
    /// `tag_materials` maps a gmsh volume tag to `(eps_diag, mu_diag, sigma)`;
    /// tets in untagged volumes default to vacuum. `ports` maps a gmsh face
    /// tag to `(mode_m, mode_n, direction)` — each becomes a waveguide port,
    /// indexed in the given order. `direction` is `None` for a waveguide
    /// port, or a `(dx, dy, dz)` field axis for a lumped port.
    ///
    /// `absorbers` maps a gmsh volume tag to a graded impedance-matched
    /// absorbing layer — `(volume_tag, axis, inner_face, thickness, nu_max,
    /// is_low)`. `axis` is 0/1/2 for x/y/z; the loss ramps quadratically
    /// from zero at `inner_face` to `nu_max` at the layer's outer face,
    /// `thickness` away. `is_low` selects the low-coordinate end (the layer
    /// extends toward decreasing `axis`) versus the high-coordinate end.
    /// Each tet in the tagged volume keeps its `eps`/`mu` and gains the
    /// matched electric/magnetic loss `sigma = nu*eps`, `sigma_m = nu*mu`.
    /// Applied after `tag_materials`, so an absorber overrides a plain
    /// material assignment on the same volume.
    #[staticmethod]
    #[pyo3(signature = (mesh_bytes, order, flux_alpha = 1.0, tag_materials = None, ports = None, absorbers = None))]
    #[allow(clippy::too_many_arguments)]
    fn from_mesh_bytes(
        mesh_bytes: &[u8],
        order: usize,
        flux_alpha: f64,
        tag_materials: Option<
            Vec<(i32, (f64, f64, f64), (f64, f64, f64), f64)>,
        >,
        ports: Option<Vec<(i32, usize, usize, Option<(f64, f64, f64)>)>>,
        absorbers: Option<Vec<(i32, usize, f64, f64, f64, bool)>>,
    ) -> PyResult<Self> {
        use rapidfem_td::rhs::{ElemMaterial, MaxwellOperator, PortSpec};
        let mesh = rapidfem_core::mesh_io::parse_mesh_bytes(mesh_bytes)
            .map_err(PyRuntimeError::new_err)?;
        let mut materials = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        if let Some(tm) = tag_materials {
            for (tag, eps, mu, sigma) in tm {
                if let Some(tets) = mesh.vtag_to_tet.get(&tag) {
                    for &t in tets {
                        materials[t] = ElemMaterial {
                            eps: [eps.0, eps.1, eps.2],
                            mu: [mu.0, mu.1, mu.2],
                            sigma,
                            sigma_m: 0.0,
                        };
                    }
                }
            }
        }
        // Graded matched absorbers — per tet, depth into the layer sets a
        // quadratically ramped loss rate `nu`, mirroring `absorber.rs`. The
        // tet keeps its eps/mu; the matched pair `sigma = nu*eps`,
        // `sigma_m = nu*mu` keeps `sigma*/mu = sigma/eps = nu`, so the layer
        // is reflectionless at the interface.
        if let Some(abs_specs) = absorbers {
            for (tag, axis, inner_face, thickness, nu_max, is_low) in abs_specs
            {
                let tets = match mesh.vtag_to_tet.get(&tag) {
                    Some(t) => t,
                    None => continue,
                };
                for &t in tets {
                    let centroid: f64 = mesh.tets[t]
                        .iter()
                        .map(|&n| mesh.nodes[n][axis])
                        .sum::<f64>()
                        / 4.0;
                    let depth = if is_low {
                        inner_face - centroid
                    } else {
                        centroid - inner_face
                    };
                    if depth <= 0.0 {
                        continue;
                    }
                    let frac = (depth / thickness).clamp(0.0, 1.0);
                    let nu = nu_max * frac * frac;
                    let m = &mut materials[t];
                    m.sigma = nu * m.eps[0];
                    m.sigma_m = nu * m.mu[0];
                }
            }
        }
        let mut port_specs: Vec<PortSpec> = Vec::new();
        if let Some(ps) = ports {
            for (tag, m, n, dir) in ps {
                let dir = dir.map(|(x, y, z)| [x, y, z]);
                let spec = PortSpec::from_mesh_tag(&mesh, tag, (m, n), dir)
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "port face tag {tag} has no triangles, or its \
                             direction is zero / parallel to the face"
                        ))
                    })?;
                port_specs.push(spec);
            }
        }
        let op = MaxwellOperator::new_with_materials_ports(
            &mesh, order, flux_alpha, &materials, &port_specs,
        );
        Ok(PyTdOperator {
            op,
            krylov: rapidfem_td::propagator::KrylovWorkspace::new(),
            driven_b: Vec::new(),
        })
    }

    /// Degrees of freedom, `6·Np·n_elem`.
    fn n_dof(&self) -> usize {
        self.op.n_dof()
    }

    /// Apply the semi-discrete operator — `dy/dt = A·y`. Zero-copy numpy in
    /// and out: `y` is read straight from its buffer, the result is handed
    /// back as a numpy array — no Python-list round-trip.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(self.op.apply(y).into_pyarray_bound(py))
    }

    /// Advance the state by `h` with the matrix-free exponential propagator.
    /// Zero-copy numpy in and out; the Krylov workspace is reused, so a
    /// transient loop allocates only the returned array per step.
    #[pyo3(signature = (y, h, krylov_dim = 40))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        h: f64,
        krylov_dim: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut out = vec![0.0; y.len()];
        let op = &self.op;
        self.krylov.expmv_into(
            |x, ax| op.apply_into(x, ax),
            y,
            h,
            krylov_dim,
            KRYLOV_TOL,
            &mut out,
        );
        Ok(out.into_pyarray_bound(py))
    }

    /// Global DOF index for a field component at the node nearest `point` —
    /// the hook for a soft source or a field probe. `field`: 0 = E, 1 = H.
    /// `comp`: 0 = x, 1 = y, 2 = z.
    fn nearest_node_dof(
        &self,
        point: (f64, f64, f64),
        field: usize,
        comp: usize,
    ) -> usize {
        self.op
            .nearest_node_dof([point.0, point.1, point.2], field, comp)
    }

    /// One source-driven step — `dy/dt = A·y + b` with `b` a single-DOF soft
    /// source — via the exponential time integrator. Zero-copy numpy in/out;
    /// the Krylov workspace and source vector are reused across the loop.
    #[pyo3(signature = (y, source_dof, source_value, h, krylov_dim = 40))]
    fn step_driven<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        source_dof: usize,
        source_value: f64,
        h: f64,
        krylov_dim: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n = y.len();
        if source_dof >= n {
            return Err(PyRuntimeError::new_err("source_dof out of range"));
        }
        if self.driven_b.len() != n {
            self.driven_b = vec![0.0; n];
        }
        self.driven_b[source_dof] = source_value;
        let mut out = vec![0.0; n];
        let op = &self.op;
        self.krylov.etd_step_into(
            |x, ax| op.apply_into(x, ax),
            y,
            &self.driven_b,
            h,
            krylov_dim,
            KRYLOV_TOL,
            &mut out,
        );
        self.driven_b[source_dof] = 0.0;
        Ok(out.into_pyarray_bound(py))
    }

    /// Number of waveguide ports on the operator.
    fn n_ports(&self) -> usize {
        self.op.n_ports()
    }

    /// Cutoff angular frequency of port `port_idx`'s mode (operator units).
    fn port_cutoff(&self, port_idx: usize) -> f64 {
        self.op.port_cutoff(port_idx)
    }

    /// Spatial source vector for driving port `port_idx` with a unit
    /// waveform — the system is `dy/dt = A·y + b·g(t)`.
    fn port_source<'py>(
        &self,
        py: Python<'py>,
        port_idx: usize,
    ) -> Bound<'py, PyArray1<f64>> {
        self.op.port_source(port_idx).into_pyarray_bound(py)
    }

    /// Modal field projections `(P_e, P_h)` at port `port_idx` for the
    /// state `y` — the forward/backward split `A,B = (P_e ± Z·P_h)/2` is
    /// done per frequency on the recorded time series.
    fn port_projections(
        &self,
        y: PyReadonlyArray1<'_, f64>,
        port_idx: usize,
    ) -> PyResult<(f64, f64)> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(self.op.port_modal_projections(y, port_idx))
    }

    /// One source-driven step `dy/dt = A·y + b` with a full source vector
    /// `b` — the path for modal port excitation. Zero-copy numpy in/out;
    /// the Krylov workspace is reused across the loop.
    #[pyo3(signature = (y, source, h, krylov_dim = 40))]
    fn step_with_source<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        source: PyReadonlyArray1<'py, f64>,
        h: f64,
        krylov_dim: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let src = source
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if src.len() != y.len() {
            return Err(PyRuntimeError::new_err(
                "source length must equal n_dof",
            ));
        }
        let mut out = vec![0.0; y.len()];
        let op = &self.op;
        self.krylov.etd_step_into(
            |x, ax| op.apply_into(x, ax),
            y,
            src,
            h,
            krylov_dim,
            KRYLOV_TOL,
            &mut out,
        );
        Ok(out.into_pyarray_bound(py))
    }

    /// The explicit sparse state-space matrix `A`, as a CSR quadruple
    /// `(n, row_ptr, col_idx, values)` — the index/value arrays come back
    /// as numpy arrays, so they feed straight into `scipy.sparse.csr_matrix`
    /// without a Python-list round-trip.
    fn state_space<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        usize,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        let csr = self.op.assemble_sparse();
        let row_ptr: Vec<i64> =
            csr.row_ptr.iter().map(|&x| x as i64).collect();
        let col_idx: Vec<i64> =
            csr.col_idx.iter().map(|&x| x as i64).collect();
        (
            csr.n,
            row_ptr.into_pyarray_bound(py),
            col_idx.into_pyarray_bound(py),
            csr.values.into_pyarray_bound(py),
        )
    }

    /// DG node physical coordinates — an `(n_elem·Np, 3)` numpy array, in
    /// state order (`point[e*Np + node]`).
    fn node_coords<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        let pts = self.op.node_coords();
        let n = pts.len();
        let mut flat: Vec<f64> = Vec::with_capacity(n * 3);
        for p in &pts {
            flat.extend_from_slice(p);
        }
        numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
            .expect("(n,3) shape")
            .into_pyarray_bound(py)
    }

    /// The four corner local-node indices `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`
    /// — the linear-tetrahedron connectivity for a VTK field export.
    fn corner_local_nodes(&self) -> (usize, usize, usize, usize) {
        let c = self.op.corner_local_nodes();
        (c[0], c[1], c[2], c[3])
    }

    /// Build a Krylov model-order-reduced model — `r`-step Arnoldi on the
    /// matrix-free operator from the seed vector `start`. The returned
    /// `ReducedModel` propagates states in the `r`-dimensional subspace.
    fn reduced_model(
        &self,
        start: PyReadonlyArray1<'_, f64>,
        r: usize,
    ) -> PyResult<PyReducedModel> {
        let start = start
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if start.iter().all(|&x| x == 0.0) {
            return Err(PyRuntimeError::new_err(
                "reduced_model: start vector must be nonzero",
            ));
        }
        let inner = rapidfem_td::mor::ReducedModel::build(
            |x| self.op.apply(x),
            start,
            r,
        );
        Ok(PyReducedModel { inner })
    }
}

/// A Krylov model-order-reduced model of a `TdOperator` — `A ≈ V·Â·Vᵀ`
/// with `Â` small and dense. Propagates states at a fraction of the cost.
#[pyclass(name = "ReducedModel")]
struct PyReducedModel {
    inner: rapidfem_td::mor::ReducedModel,
}

#[pymethods]
impl PyReducedModel {
    /// Reduced dimension `r` — at most the requested Krylov dimension,
    /// smaller on an early Arnoldi breakdown.
    #[getter]
    fn r(&self) -> usize {
        self.inner.r
    }

    /// Full state dimension `n`.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    /// The reduced operator `Â = VᵀAV` as an `r×r` numpy array.
    #[getter]
    fn a_hat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let r = self.inner.r;
        numpy::ndarray::Array2::from_shape_vec(
            (r, r),
            self.inner.a_hat.clone(),
        )
        .expect("a_hat is r×r")
        .into_pyarray_bound(py)
    }

    /// Project a full state into the reduced space — `ŷ = Vᵀ·y`.
    fn project<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(self.inner.project(y).into_pyarray_bound(py))
    }

    /// Lift a reduced state back to the full space — `y = V·ŷ`.
    fn lift<'py>(
        &self,
        py: Python<'py>,
        yhat: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let yhat = yhat
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(self.inner.lift(yhat).into_pyarray_bound(py))
    }

    /// Propagate a full state by `t` through the reduced model —
    /// `V·exp(t·Â)·Vᵀ·y₀`.
    fn propagate<'py>(
        &self,
        py: Python<'py>,
        y0: PyReadonlyArray1<'py, f64>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y0 = y0
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(self.inner.propagate(y0, t).into_pyarray_bound(py))
    }
}

/// rapidfem — frequency- and time-domain EM FEM solver.
#[pymodule]
#[pyo3(name = "_native")]
fn rapidfem_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Solver selection is automatic: PARDISO if MKL is loadable (typically
    // 5–10× faster on complex-symmetric LU), faer otherwise. Force one with
    // RAPIDFEM_SOLVER=faer or =pardiso before importing.

    m.add_class::<PySimulation>()?;
    m.add_class::<PySweepResult>()?;
    m.add_class::<PyEigenmode>()?;
    m.add_class::<PyRadiationPattern>()?;
    m.add_class::<PyTdOperator>()?;
    m.add_class::<PyReducedModel>()?;
    Ok(())
}
