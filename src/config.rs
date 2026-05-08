//! TOML configuration file parsing for CLI.

use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub mesh: MeshConfig,
    #[serde(default)]
    pub frequency: FrequencyConfig,
    #[serde(default)]
    pub ports: Vec<PortConfig>,
    #[serde(default)]
    pub materials: Vec<MaterialConfig>,
    pub pec: PecConfig,
    #[serde(default)]
    pub solver: SolverConfig,
    #[serde(default)]
    pub adaptive: Option<AdaptiveConfig>,
    #[serde(default)]
    pub eigenmode: Option<EigenmodeConfig>,
    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Deserialize)]
pub struct AdaptiveConfig {
    #[serde(default = "default_theta")]
    pub theta: f64,
    #[serde(default = "default_refinement_ratio")]
    pub refinement_ratio: f64,
}

fn default_theta() -> f64 { 0.5 }
fn default_refinement_ratio() -> f64 { 0.5 }

#[derive(Deserialize)]
pub struct EigenmodeConfig {
    pub target_frequency: f64,
    #[serde(default = "default_n_modes")]
    pub n_modes: usize,
}

fn default_n_modes() -> usize { 6 }

#[derive(Deserialize)]
pub struct MeshConfig {
    pub file: String,
}

#[derive(Deserialize, Default)]
pub struct FrequencyConfig {
    #[serde(default)]
    pub values: Vec<f64>,
    #[serde(default)]
    pub range: Vec<f64>,
}

impl FrequencyConfig {
    pub fn frequencies(&self) -> Vec<f64> {
        if !self.values.is_empty() {
            self.values.clone()
        } else if self.range.len() == 3 {
            let (start, stop, n) = (self.range[0], self.range[1], self.range[2] as usize);
            if n <= 1 { return vec![start]; }
            (0..n).map(|i| start + (stop - start) * i as f64 / (n - 1) as f64).collect()
        } else {
            vec![10.0e9]
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum PortConfig {
    #[serde(rename = "rectangular")]
    Rectangular {
        tag: i32,
        #[serde(default)]
        width: f64,
        #[serde(default)]
        height: f64,
        #[serde(default = "default_mode")]
        mode: [usize; 2],
        #[serde(default = "default_one")]
        er: f64,
        #[serde(default = "default_one")]
        power: f64,
    },
    /// User-defined port with a uniform constant E vector mode (covers the parallel-plate
    /// TEM case directly). For more elaborate spatial modes, instantiate `UserDefinedPort`
    /// programmatically via the Rust API rather than TOML.
    #[serde(rename = "user_defined")]
    UserDefined {
        tag: i32,
        /// Constant E-field vector across the port face (V/m, but normalized — magnitude is set by `power`)
        e_field: [f64; 3],
        #[serde(default = "default_one")]
        power: f64,
    },
    #[serde(rename = "coax")]
    Coax {
        tag: i32,
        /// Inner conductor radius (m)
        ri: f64,
        /// Outer conductor radius (m)
        ro: f64,
        /// Coax center on the port face (m). If omitted, auto-detected from the face centroid.
        #[serde(default)]
        origin: Option<[f64; 3]>,
        /// Axial direction (propagation direction, normal to port face). Auto-detected if omitted.
        #[serde(default)]
        z_axis: Option<[f64; 3]>,
        /// Dielectric inside the coax (default 1.0 = air)
        #[serde(default = "default_one")]
        er: f64,
        #[serde(default = "default_one")]
        power: f64,
    },
    #[serde(rename = "lumped")]
    Lumped {
        tag: i32,
        #[serde(default = "default_z0")]
        z0: f64,
        direction: [f64; 3],
        #[serde(default)]
        width: f64,
        #[serde(default)]
        height: f64,
        #[serde(default = "default_one")]
        power: f64,
    },
    #[serde(rename = "abc")]
    Abc {
        tag: i32,
        #[serde(default = "default_abc_order")]
        order: usize,
        #[serde(default = "default_abc_type")]
        abctype: String,
    },
    /// Perfect magnetic conductor — natural BC (n × H = 0). No-op during assembly.
    /// Useful when users want to mark a surface explicitly as PMC for documentation.
    #[serde(rename = "pmc")]
    Pmc { tag: i32 },
    /// Lumped element (R, L, C in series) load on a surface. width/height define the
    /// surface-impedance scaling: surfZ = (R + jωL + 1/(jωC)) * width/height.
    #[serde(rename = "lumped_element")]
    LumpedElement {
        tag: i32,
        #[serde(default)]
        r: f64,
        #[serde(default)]
        l: f64,
        #[serde(default)]
        c: Option<f64>,
        /// Width (orthogonal to field direction); auto-detected if 0
        #[serde(default)]
        width: f64,
        /// Height (along field direction); auto-detected if 0
        #[serde(default)]
        height: f64,
        /// Field direction unit vector — used for auto width/height detection
        #[serde(default = "default_z_dir")]
        direction: [f64; 3],
    },
    /// Surface impedance (lossy conductor wall). Either supply σ (S/m) for skin-depth-based
    /// impedance, or `zs` (real+imag Ω/sq) for a frequency-independent constant.
    #[serde(rename = "surface_impedance")]
    SurfaceImpedance {
        tag: i32,
        #[serde(default)]
        conductivity: f64,
        #[serde(default = "default_one")]
        mur: f64,
        #[serde(default = "default_one")]
        er: f64,
        #[serde(default)]
        thickness: Option<f64>,
        /// Explicit surface impedance [re, im] in Ω/sq (overrides conductivity if present).
        #[serde(default)]
        zs: Option<[f64; 2]>,
    },
}

#[derive(Deserialize)]
pub struct MaterialConfig {
    pub volume_tag: i32,
    #[serde(default = "default_one")]
    pub er: f64,
    #[serde(default = "default_one")]
    pub ur: f64,
    #[serde(default)]
    pub tand: f64,
    #[serde(default)]
    pub conductivity: f64,
    /// Optional diagonal εr anisotropy [εxx, εyy, εzz]; overrides scalar `er`.
    #[serde(default)]
    pub er_diag: Option<[f64; 3]>,
    /// Optional diagonal μr anisotropy [μxx, μyy, μzz]; overrides scalar `ur`.
    #[serde(default)]
    pub ur_diag: Option<[f64; 3]>,
}

#[derive(Deserialize)]
pub struct PecConfig {
    pub tags: Vec<i32>,
}

#[derive(Deserialize, Default)]
pub struct SolverConfig {
    #[serde(default = "default_solver_prefer")]
    pub prefer: String,
}

#[derive(Deserialize, Default)]
pub struct OutputConfig {
    #[serde(default)]
    pub touchstone: Option<String>,
    #[serde(default = "default_z0")]
    pub z0: f64,
    #[serde(default)]
    pub vtk: Option<String>,
    #[serde(default)]
    pub farfield: Option<String>,
    /// Physical tag of the NFFT surface (defaults to ABC tag)
    #[serde(default)]
    pub nfft_tag: Option<i32>,
    /// Optional CSV path for group delay τ_g = -dφ/dω of S-parameters.
    #[serde(default)]
    pub group_delay: Option<String>,
}

fn default_mode() -> [usize; 2] { [1, 0] }
fn default_one() -> f64 { 1.0 }
fn default_z0() -> f64 { 50.0 }
fn default_abc_order() -> usize { 1 }
fn default_abc_type() -> String { "B".to_string() }
fn default_solver_prefer() -> String { "auto".to_string() }
fn default_z_dir() -> [f64; 3] { [0.0, 0.0, 1.0] }

pub fn load_config(path: &str) -> Result<Config, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {}", path, e))?;
    toml::from_str(&content)
        .map_err(|e| format!("Cannot parse {}: {}", path, e))
}
