//! PyO3 bindings for rapidfem. Exposes `Simulation`, `SweepResult` to Python.
//!
//! Build via `maturin develop` (dev) or `maturin build --release` (wheel).
//! See `examples/wr90.py` for usage.

use num_complex::Complex64;
use numpy::{Complex64 as NpC64, IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rapidfem::eigenmode::Eigenmode;
use rapidfem::farfield::RadiationPattern;
use rapidfem::simulation::{Simulation, SweepResult};

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
        let config = rapidfem::config::load_config(config_path)
            .map_err(|e| PyRuntimeError::new_err(format!("config: {}", e)))?;
        let mesh = rapidfem::mesh_io::load_mesh(mesh_path)
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
        PySweepResult { inner: self.inner.run_sweep() }
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

/// rapidfem — frequency-domain EM FEM solver.
#[pymodule]
#[pyo3(name = "_native")]
fn rapidfem_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Default the Python wheel to pure-Rust faer. PARDISO needs MKL on PATH,
    // which a typical pip install user does not have. Power users opt in with
    // `os.environ["RAPIDFEM_SOLVER"] = "pardiso"` before importing rapidfem.
    if std::env::var_os("RAPIDFEM_SOLVER").is_none() {
        #[allow(unused_unsafe)]
        unsafe { std::env::set_var("RAPIDFEM_SOLVER", "faer"); }
    }

    m.add_class::<PySimulation>()?;
    m.add_class::<PySweepResult>()?;
    m.add_class::<PyEigenmode>()?;
    m.add_class::<PyRadiationPattern>()?;
    Ok(())
}
