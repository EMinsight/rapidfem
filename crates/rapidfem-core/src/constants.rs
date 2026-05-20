//! Physical constants (mirrors emerge/_emerge/const.py).

pub const C0: f64 = 299_792_458.0;                    // Speed of light (m/s)
pub const Z0: f64 = 376.73031366857;                   // Free space impedance (Ω)
pub const EPS0: f64 = 8.854187818814e-12;              // Permittivity of free space (F/m)
pub const MU0: f64 = 1.0 / (C0 * C0 * EPS0);          // Permeability of free space (H/m)
pub const PI: f64 = std::f64::consts::PI;
