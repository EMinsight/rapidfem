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

/// Time-domain DGTD Maxwell operator (vacuum, PEC walls), built on a
/// structured box cavity. Wraps the Rust `MaxwellOperator`.
#[pyclass(name = "TdOperator")]
struct PyTdOperator {
    op: rapidfem_td::rhs::MaxwellOperator,
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
        PyTdOperator { op }
    }

    /// Build the operator from in-memory gmsh `.msh` bytes — the path for
    /// arbitrary unstructured meshes produced by the geometry API.
    ///
    /// `tag_materials` maps a gmsh volume tag to `(eps_diag, mu_diag, sigma)`;
    /// tets in untagged volumes default to vacuum.
    #[staticmethod]
    #[pyo3(signature = (mesh_bytes, order, flux_alpha = 1.0, tag_materials = None))]
    fn from_mesh_bytes(
        mesh_bytes: &[u8],
        order: usize,
        flux_alpha: f64,
        tag_materials: Option<
            Vec<(i32, (f64, f64, f64), (f64, f64, f64), f64)>,
        >,
    ) -> PyResult<Self> {
        use rapidfem_td::rhs::{ElemMaterial, MaxwellOperator};
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
        let op = MaxwellOperator::new_with_materials(
            &mesh, order, flux_alpha, &materials,
        );
        Ok(PyTdOperator { op })
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
    /// Zero-copy numpy in and out.
    #[pyo3(signature = (y, h, krylov_dim = 40))]
    fn step<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        h: f64,
        krylov_dim: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let out = rapidfem_td::propagator::expmv(
            |x| self.op.apply(x),
            y,
            h,
            krylov_dim,
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
    /// source — via the exponential time integrator. Zero-copy numpy in/out.
    #[pyo3(signature = (y, source_dof, source_value, h, krylov_dim = 40))]
    fn step_driven<'py>(
        &self,
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
        if source_dof >= y.len() {
            return Err(PyRuntimeError::new_err("source_dof out of range"));
        }
        let mut b = vec![0.0; y.len()];
        b[source_dof] = source_value;
        let out = rapidfem_td::propagator::etd_step(
            |x| self.op.apply(x),
            y,
            &b,
            h,
            krylov_dim,
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
    Ok(())
}
