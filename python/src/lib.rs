//! PyO3 bindings for rapidfem. Exposes `Simulation`, `SweepResult` to Python.
//!
//! Build via `maturin develop` (dev) or `maturin build --release` (wheel).
//! See `examples/wr90.py` for usage.

use num_complex::Complex64;
use numpy::{Complex64 as NpC64, IntoPyArray, PyArray1, PyArray3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
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

/// rapidfem — frequency-domain EM FEM solver.
#[pymodule]
#[pyo3(name = "_native")]
fn rapidfem_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulation>()?;
    m.add_class::<PySweepResult>()?;
    Ok(())
}
