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

/// Auto-detect port coordinate system and dimensions from mesh face triangles.
/// Port of EMerge's rect_basis() for axis-aligned rectangular ports.
///
/// Returns (CoordinateSystem, width, height) where width ≥ height.
pub fn detect_rect_port(
    mesh: &crate::mesh::Mesh,
    tri_ids: &[usize],
) -> (CoordinateSystem, f64, f64) {
    let nodes = &mesh.nodes;
    let tris = &mesh.tris;

    // 1. Compute centroid (origin)
    let mut center = [0.0f64; 3];
    let mut count = 0.0;
    let mut all_verts = std::collections::HashSet::new();
    for &ti in tri_ids {
        for &vi in &tris[ti] {
            if all_verts.insert(vi) {
                for k in 0..3 { center[k] += nodes[vi][k]; }
                count += 1.0;
            }
        }
    }
    for k in 0..3 { center[k] /= count; }

    // 2. Compute face normal from first triangle
    let first_tri = tris[tri_ids[0]];
    let v0 = nodes[first_tri[0]];
    let v1 = nodes[first_tri[1]];
    let v2 = nodes[first_tri[2]];
    let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
    let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
    let cr = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
    let nn = (cr[0]*cr[0]+cr[1]*cr[1]+cr[2]*cr[2]).sqrt();
    let mut normal = [cr[0]/nn, cr[1]/nn, cr[2]/nn];

    // 3. Orient outward using adjacent tet centroid
    let adj_tet = mesh.tri_to_tet[tri_ids[0]][0];
    let tet = &mesh.tets[adj_tet];
    let tc = [
        (nodes[tet[0]][0]+nodes[tet[1]][0]+nodes[tet[2]][0]+nodes[tet[3]][0])/4.0,
        (nodes[tet[0]][1]+nodes[tet[1]][1]+nodes[tet[2]][1]+nodes[tet[3]][1])/4.0,
        (nodes[tet[0]][2]+nodes[tet[1]][2]+nodes[tet[2]][2]+nodes[tet[3]][2])/4.0,
    ];
    let to_tet = [tc[0]-center[0], tc[1]-center[1], tc[2]-center[2]];
    if normal[0]*to_tet[0]+normal[1]*to_tet[1]+normal[2]*to_tet[2] > 0.0 {
        normal = [-normal[0], -normal[1], -normal[2]];
    }

    // 4. Find face extents along each axis
    let mut min_c = [f64::INFINITY; 3];
    let mut max_c = [f64::NEG_INFINITY; 3];
    for &vi in &all_verts {
        for k in 0..3 {
            min_c[k] = min_c[k].min(nodes[vi][k]);
            max_c[k] = max_c[k].max(nodes[vi][k]);
        }
    }
    let ext = [max_c[0]-min_c[0], max_c[1]-min_c[1], max_c[2]-min_c[2]];

    // 5. Determine axes: normal axis has smallest extent, xhat = largest extent, yhat = remaining
    let normal_axis = if normal[0].abs() > normal[1].abs() && normal[0].abs() > normal[2].abs() { 0 }
        else if normal[1].abs() > normal[2].abs() { 1 } else { 2 };

    let mut face_axes: Vec<(usize, f64)> = (0..3)
        .filter(|&k| k != normal_axis)
        .map(|k| (k, ext[k]))
        .collect();
    face_axes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let broad_axis = face_axes[0].0;
    let narrow_axis = face_axes[1].0;
    let width = face_axes[0].1;
    let height = face_axes[1].1;

    // 6. Build orthonormal CS
    let zhat = normal;
    let mut xhat = [0.0; 3];
    xhat[broad_axis] = 1.0;
    // Orthogonalize xhat against zhat (Gram-Schmidt)
    let dot_xz = xhat[0]*zhat[0]+xhat[1]*zhat[1]+xhat[2]*zhat[2];
    let xraw = [xhat[0]-dot_xz*zhat[0], xhat[1]-dot_xz*zhat[1], xhat[2]-dot_xz*zhat[2]];
    let xn = (xraw[0]*xraw[0]+xraw[1]*xraw[1]+xraw[2]*xraw[2]).sqrt();
    let xhat_f = [xraw[0]/xn, xraw[1]/xn, xraw[2]/xn];
    let yhat = [zhat[1]*xhat_f[2]-zhat[2]*xhat_f[1],
                zhat[2]*xhat_f[0]-zhat[0]*xhat_f[2],
                zhat[0]*xhat_f[1]-zhat[1]*xhat_f[0]];

    let _ = narrow_axis;
    (CoordinateSystem::new(center, xhat_f, yhat, zhat), width, height)
}

/// Return (width, height) for a lumped port using EMerge's convention:
/// height = extent along `direction`, width = extent orthogonal (in the port plane).
///
/// EMerge microwave_bc.py:1314-1317 — height is the size in the direction axis along which
/// the potential is imposed; width is orthogonal to that. surfZ = Z0 * width / height.
pub fn lumped_port_dims(
    mesh: &crate::mesh::Mesh,
    tri_ids: &[usize],
    direction: &[f64; 3],
) -> (f64, f64) {
    let mut verts = std::collections::HashSet::new();
    let mut min_c = [f64::INFINITY; 3];
    let mut max_c = [f64::NEG_INFINITY; 3];
    for &ti in tri_ids {
        for &vi in &mesh.tris[ti] {
            if verts.insert(vi) {
                for k in 0..3 {
                    min_c[k] = min_c[k].min(mesh.nodes[vi][k]);
                    max_c[k] = max_c[k].max(mesh.nodes[vi][k]);
                }
            }
        }
    }

    // Height: extent along direction
    let mut min_proj = f64::INFINITY;
    let mut max_proj = f64::NEG_INFINITY;
    for &vi in &verts {
        let p = mesh.nodes[vi];
        let proj = p[0]*direction[0] + p[1]*direction[1] + p[2]*direction[2];
        min_proj = min_proj.min(proj);
        max_proj = max_proj.max(proj);
    }
    let height = max_proj - min_proj;

    // Width: largest in-plane extent orthogonal to direction.
    // For axis-aligned ports we use AABB extents minus the height axis and the normal axis
    // (= zero-extent axis). The remaining axis with non-zero extent gives width.
    let ext = [max_c[0]-min_c[0], max_c[1]-min_c[1], max_c[2]-min_c[2]];
    let dir_axis = if direction[0].abs() > direction[1].abs() && direction[0].abs() > direction[2].abs() { 0 }
        else if direction[1].abs() > direction[2].abs() { 1 } else { 2 };
    let mut width = 0.0f64;
    for k in 0..3 {
        if k != dir_axis && ext[k] > width {
            width = ext[k];
        }
    }
    (width, height)
}
