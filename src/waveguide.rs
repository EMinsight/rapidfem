//! Exact port of microwave_bc.py: RectangularWaveguide class and CoordinateSystem.
//!
//! All method names and formulas match EMerge exactly.

use num_complex::Complex64 as C64;
use crate::constants::*;

/// Port of cs.py: CoordinateSystem
///
/// Stores origin, xax (xhat), yax (yhat), zax (zhat), _basis, _basis_inv.
pub struct CoordinateSystem {
    pub origin: [f64; 3],
    pub xax: [f64; 3],
    pub yax: [f64; 3],
    pub zax: [f64; 3],
    /// _basis: rows are xax, yax, zax
    pub basis: [[f64; 3]; 3],
    /// _basis_inv: inverse of basis (for global→local transform)
    pub basis_inv: [[f64; 3]; 3],
}

impl CoordinateSystem {
    /// Build CS from origin and 3 orthonormal axes.
    pub fn new(origin: [f64; 3], xax: [f64; 3], yax: [f64; 3], zax: [f64; 3]) -> Self {
        let basis = [xax, yax, zax];
        // For orthonormal basis, inverse = transpose
        let basis_inv = [
            [xax[0], yax[0], zax[0]],
            [xax[1], yax[1], zax[1]],
            [xax[2], yax[2], zax[2]],
        ];
        CoordinateSystem { origin, xax, yax, zax, basis, basis_inv }
    }

    /// Port of cs.py: in_local_cs(x, y, z)
    pub fn in_local_cs(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let b = &self.basis_inv;
        let xg = x - self.origin[0];
        let yg = y - self.origin[1];
        let zg = z - self.origin[2];
        (
            b[0][0]*xg + b[0][1]*yg + b[0][2]*zg,
            b[1][0]*xg + b[1][1]*yg + b[1][2]*zg,
            b[2][0]*xg + b[2][1]*yg + b[2][2]*zg,
        )
    }

    /// Port of cs.py: in_global_basis(x, y, z) — transforms vector components
    pub fn in_global_basis(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        (
            self.xax[0]*x + self.yax[0]*y + self.zax[0]*z,
            self.xax[1]*x + self.yax[1]*y + self.zax[1]*z,
            self.xax[2]*x + self.yax[2]*y + self.zax[2]*z,
        )
    }
}

/// Port of microwave_bc.py: RectangularWaveguide
pub struct RectWaveguide {
    pub port_number: usize,
    pub power: f64,
    pub mode: (usize, usize),
    pub er: f64,
    pub polarization: f64,
    pub dims: (f64, f64),  // (width, height)
    pub cs: CoordinateSystem,
}

impl RectWaveguide {
    /// Port of get_amplitude(k0)
    /// amplitude = sqrt(power * 4 * Z0 / (width * height))
    pub fn get_amplitude(&self, _k0: f64) -> f64 {
        let zte = Z0;
        (self.power * 4.0 * zte / (self.dims.0 * self.dims.1)).sqrt()
    }

    /// Port of get_beta(k0)
    /// beta = sqrt(er*k0^2 - (pi*m/width)^2 - (pi*n/height)^2)
    pub fn get_beta(&self, k0: f64) -> f64 {
        let (width, height) = self.dims;
        let (m, n) = self.mode;
        (self.er * k0 * k0
            - (PI * m as f64 / width).powi(2)
            - (PI * n as f64 / height).powi(2)).sqrt()
    }

    /// Port of get_gamma(k0)
    /// gamma = 1j * beta
    pub fn get_gamma(&self, k0: f64) -> C64 {
        C64::new(0.0, self.get_beta(k0))
    }

    /// Port of Zmode(k0) — for TE modes
    /// Zmode = k0 * C0 * MU0 / beta
    pub fn z_mode(&self, k0: f64) -> f64 {
        k0 * C0 * MU0 / self.get_beta(k0)
    }

    /// Port of _qmode(k0)
    /// qmode = sqrt(Zmode / Z0)
    pub fn qmode(&self, k0: f64) -> f64 {
        (self.z_mode(k0) / Z0).sqrt()
    }

    /// Port of port_mode_3d(x_local, y_local, k0)
    /// Returns (Ex, Ey, Ez) in LOCAL coordinates.
    ///
    /// Ev = polarization * amplitude * cos(pi*m*x/width) * cos(pi*n*y/height)
    /// Eh = polarization * amplitude * sin(pi*m*x/width) * sin(pi*n*y/height)
    /// Ex = Eh, Ey = Ev, Ez = 0
    /// Result scaled by qmode.
    pub fn port_mode_3d(&self, x_local: f64, y_local: f64, k0: f64) -> (f64, f64, f64) {
        let (width, height) = self.dims;
        let (m, n) = self.mode;
        let a = self.get_amplitude(k0);
        let ev = self.polarization * a
            * (PI * m as f64 * x_local / width).cos()
            * (PI * n as f64 * y_local / height).cos();
        let eh = self.polarization * a
            * (PI * m as f64 * x_local / width).sin()
            * (PI * n as f64 * y_local / height).sin();
        let ex = eh;
        let ey = ev;
        let ez = 0.0;
        let q = self.qmode(k0);
        (q * ex, q * ey, q * ez)
    }

    /// Port of port_mode_3d_global(x_global, y_global, z_global, k0)
    /// Returns (Ex, Ey, Ez) in GLOBAL coordinates.
    pub fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, k0: f64) -> (f64, f64, f64) {
        let (xl, yl, _zl) = self.cs.in_local_cs(x, y, z);
        let (ex, ey, ez) = self.port_mode_3d(xl, yl, k0);
        self.cs.in_global_basis(ex, ey, ez)
    }

    /// Port of get_Uinc(x_global, y_global, z_global, k0)
    /// Returns -2j * beta * port_mode_3d_global(...) as complex [3] vector.
    pub fn get_uinc(&self, x: f64, y: f64, z: f64, k0: f64) -> [C64; 3] {
        let (ex, ey, ez) = self.port_mode_3d_global(x, y, z, k0);
        let factor = C64::new(0.0, -2.0 * self.get_beta(k0));
        [factor * C64::from(ex), factor * C64::from(ey), factor * C64::from(ez)]
    }
}

/// Exact port of microwave_bc.py: AbsorbingBoundary
///
/// Order-1 ABC: γ = j·k₀·neff (neff=1 for air)
/// Order-2 ABC: γ = j·k₀·c₁·neff, plus a correction matrix (handled in assembly)
pub struct AbsorbingBoundary {
    pub order: usize,
    pub neff: f64,
    pub abctype: char,
}

/// Order-2 ABC coefficients (c₁, c₂) from microwave_bc.py lines 177-182
pub const ABC_O2_COEFFS: [(char, f64, f64); 5] = [
    ('A', 1.0, -0.5),
    ('B', 1.00023, -0.51555),
    ('C', 1.03084, -0.73631),
    ('D', 1.06103, -0.84883),
    ('E', 1.12500, -1.00000),
];

impl AbsorbingBoundary {
    pub fn new(order: usize, abctype: char) -> Self {
        AbsorbingBoundary { order, neff: 1.0, abctype }
    }

    /// Port of get_gamma(k0) from microwave_bc.py lines 409-422
    pub fn get_gamma(&self, k0: f64) -> C64 {
        if self.order == 1 {
            C64::new(0.0, k0 * self.neff)
        } else {
            let c1 = ABC_O2_COEFFS.iter()
                .find(|(c, _, _)| *c == self.abctype)
                .map(|(_, c1, _)| *c1)
                .unwrap_or(1.00023);
            C64::new(0.0, k0 * c1 * self.neff)
        }
    }

    /// Get the c₂ coefficient for order-2 correction matrix
    pub fn get_c2(&self) -> f64 {
        ABC_O2_COEFFS.iter()
            .find(|(c, _, _)| *c == self.abctype)
            .map(|(_, _, c2)| *c2)
            .unwrap_or(-0.51555)
    }
}

/// Exact port of microwave_bc.py: LumpedPort (lines 1294-1441)
pub struct LumpedPort {
    pub port_number: usize,
    pub power: f64,
    pub z0: f64,
    pub width: f64,
    pub height: f64,
    /// E-field direction unit vector in global coordinates
    pub direction: [f64; 3],
}

impl LumpedPort {
    /// Port of surfZ property: Z0 * width / height
    pub fn surf_z(&self) -> f64 {
        self.z0 * self.width / self.height
    }

    /// Port of voltage property: sqrt(2 * power * Z0)
    pub fn voltage(&self) -> f64 {
        (2.0 * self.power * self.z0).sqrt()
    }

    /// Port of get_gamma(k0): j * k0 * Z0 / surfZ
    pub fn get_gamma(&self, k0: f64) -> C64 {
        C64::new(0.0, k0 * Z0 / self.surf_z())
    }

    /// Port of port_mode_3d_global: returns uniform field in direction
    pub fn port_mode_3d_global(&self, _x: f64, _y: f64, _z: f64, _k0: f64) -> (f64, f64, f64) {
        (self.direction[0], self.direction[1], self.direction[2])
    }

    /// Port of get_Uinc: -j*2*k0 * voltage/height * (Z0/surfZ) * mode_field
    pub fn get_uinc(&self, x: f64, y: f64, z: f64, k0: f64) -> [C64; 3] {
        let emag = C64::new(0.0, -2.0 * k0) * C64::from(self.voltage() / self.height * (Z0 / self.surf_z()));
        let (ex, ey, ez) = self.port_mode_3d_global(x, y, z, k0);
        [emag * C64::from(ex), emag * C64::from(ey), emag * C64::from(ez)]
    }
}
