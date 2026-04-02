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
    pub eigenmode: Option<EigenmodeConfig>,
    #[serde(default)]
    pub output: OutputConfig,
}

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
}

fn default_mode() -> [usize; 2] { [1, 0] }
fn default_one() -> f64 { 1.0 }
fn default_z0() -> f64 { 50.0 }
fn default_abc_order() -> usize { 1 }
fn default_abc_type() -> String { "B".to_string() }
fn default_solver_prefer() -> String { "auto".to_string() }

pub fn load_config(path: &str) -> Result<Config, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {}", path, e))?;
    toml::from_str(&content)
        .map_err(|e| format!("Cannot parse {}: {}", path, e))
}
