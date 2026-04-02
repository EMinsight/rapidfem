//! TOML configuration file parsing for CLI.

use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub mesh: MeshConfig,
    pub frequency: FrequencyConfig,
    #[serde(default)]
    pub ports: Vec<PortConfig>,
    pub pec: PecConfig,
    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Deserialize)]
pub struct MeshConfig {
    pub file: String,
}

#[derive(Deserialize)]
pub struct FrequencyConfig {
    /// Single frequency or list
    #[serde(default)]
    pub values: Vec<f64>,
    /// Range: [start, stop, n_points]
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
            vec![10.0e9] // default
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type")]
pub enum PortConfig {
    #[serde(rename = "rectangular")]
    Rectangular {
        tag: i32,
        #[serde(default = "default_width")]
        width: f64,
        #[serde(default = "default_height")]
        height: f64,
    },
    #[serde(rename = "abc")]
    Abc {
        tag: i32,
        #[serde(default = "default_abc_order")]
        order: usize,
    },
}

fn default_width() -> f64 { 22.86e-3 }
fn default_height() -> f64 { 10.16e-3 }
fn default_abc_order() -> usize { 1 }

#[derive(Deserialize)]
pub struct PecConfig {
    pub tags: Vec<i32>,
}

#[derive(Deserialize, Default)]
pub struct OutputConfig {
    #[serde(default)]
    pub touchstone: Option<String>,
}

pub fn load_config(path: &str) -> Result<Config, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read {}: {}", path, e))?;
    toml::from_str(&content)
        .map_err(|e| format!("Cannot parse {}: {}", path, e))
}
