//! Rectangular waveguide port: mode field, impedance, Robin BC parameters.
//! Mirrors microwave_bc.py: RectangularWaveguide class.

use num_complex::Complex64 as C64;
use crate::constants::*;

/// Coordinate system for a port face: origin + orthonormal basis.
pub struct CoordinateSystem {
    pub origin: [f64; 3],
    /// Basis vectors: xhat (broad wall), yhat (narrow wall), zhat (normal/propagation)
    pub xhat: [f64; 3],
    pub yhat: [f64; 3],
    pub zhat: [f64; 3],
    /// Inverse basis matrix (for global→local coordinate transform)
    pub basis_inv: [[f64; 3]; 3],
}

impl CoordinateSystem {
    /// Build a coordinate system from the port face geometry.
    /// `origin`: center of the port face
    /// `normal`: outward normal of the port face
    /// `broad_axis`: approximate direction of the broad wall (will be orthogonalized)
    pub fn from_port_face(origin: [f64; 3], normal: [f64; 3], broad_axis: [f64; 3]) -> Self {
        let zhat = normalize(&normal);
        // Gram-Schmidt: make broad_axis perpendicular to zhat
        let dot = broad_axis[0]*zhat[0] + broad_axis[1]*zhat[1] + broad_axis[2]*zhat[2];
        let xraw = [broad_axis[0] - dot*zhat[0], broad_axis[1] - dot*zhat[1], broad_axis[2] - dot*zhat[2]];
        let xhat = normalize(&xraw);
        let yhat = cross(&zhat, &xhat);

        // basis_inv: transform from global to local
        let basis_inv = [
            [xhat[0], xhat[1], xhat[2]],
            [yhat[0], yhat[1], yhat[2]],
            [zhat[0], zhat[1], zhat[2]],
        ];

        CoordinateSystem { origin, xhat, yhat, zhat, basis_inv }
    }

    /// Transform global coordinates to local port coordinates.
    pub fn to_local(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let dx = x - self.origin[0];
        let dy = y - self.origin[1];
        let dz = z - self.origin[2];
        let xl = self.basis_inv[0][0]*dx + self.basis_inv[0][1]*dy + self.basis_inv[0][2]*dz;
        let yl = self.basis_inv[1][0]*dx + self.basis_inv[1][1]*dy + self.basis_inv[1][2]*dz;
        let zl = self.basis_inv[2][0]*dx + self.basis_inv[2][1]*dy + self.basis_inv[2][2]*dz;
        (xl, yl, zl)
    }

    /// Transform local vector components to global.
    pub fn to_global_vec(&self, vx: f64, vy: f64, vz: f64) -> [f64; 3] {
        [
            self.xhat[0]*vx + self.yhat[0]*vy + self.zhat[0]*vz,
            self.xhat[1]*vx + self.yhat[1]*vy + self.zhat[1]*vz,
            self.xhat[2]*vx + self.yhat[2]*vy + self.zhat[2]*vz,
        ]
    }
}

/// Rectangular waveguide port definition.
pub struct RectWaveguide {
    pub width: f64,     // a-dimension (broad wall)
    pub height: f64,    // b-dimension (narrow wall)
    pub mode: (usize, usize),  // (m, n)
    pub er: f64,        // relative permittivity
    pub cs: CoordinateSystem,
    pub port_number: usize,
}

impl RectWaveguide {
    /// Propagation constant β = √(εr·k₀² - (mπ/a)² - (nπ/b)²)
    pub fn beta(&self, k0: f64) -> f64 {
        let (m, n) = self.mode;
        let kc_sq = (PI * m as f64 / self.width).powi(2) + (PI * n as f64 / self.height).powi(2);
        (self.er * k0 * k0 - kc_sq).sqrt()
    }

    /// Robin BC parameter γ = jβ
    pub fn gamma(&self, k0: f64) -> C64 {
        C64::new(0.0, self.beta(k0))
    }

    /// Mode impedance Z_mode
    pub fn z_mode(&self, k0: f64) -> f64 {
        // TE mode: Z_mode = k₀·c₀·μ₀/β
        k0 * C0 * MU0 / self.beta(k0)
    }

    /// Mode amplitude A = √(4·P·Z₀/(a·b)) where P=1W
    pub fn amplitude(&self, _k0: f64) -> f64 {
        (4.0 * Z0 / (self.width * self.height)).sqrt()
    }

    /// Mode correction factor _qmode = √(Z_mode/Z₀)
    pub fn qmode(&self, k0: f64) -> f64 {
        (self.z_mode(k0) / Z0).sqrt()
    }

    /// Evaluate TE mode E-field in LOCAL coordinates.
    /// For TE10: Ey = A·qmode·cos(πx/a), Ex = 0, Ez = 0
    /// x is in [-a/2, a/2], y is in [-b/2, b/2]
    pub fn mode_field_local(&self, x_local: f64, _y_local: f64, k0: f64) -> [f64; 3] {
        let (m, n) = self.mode;
        let a = self.amplitude(k0);
        let q = self.qmode(k0);
        // Ev = A * cos(mπx/a) * cos(nπy/b)
        // Eh = A * sin(mπx/a) * sin(nπy/b)
        // Ex = Eh, Ey = Ev
        let ev = a * (PI * m as f64 * x_local / self.width).cos()
                   * (PI * n as f64 * _y_local / self.height).cos();
        let eh = a * (PI * m as f64 * x_local / self.width).sin()
                   * (PI * n as f64 * _y_local / self.height).sin();
        [q * eh, q * ev, 0.0]
    }

    /// Evaluate TE mode E-field in GLOBAL coordinates at a physical point.
    pub fn mode_field_global(&self, x: f64, y: f64, z: f64, k0: f64) -> [f64; 3] {
        let (xl, yl, _zl) = self.cs.to_local(x, y, z);
        let [ex, ey, ez] = self.mode_field_local(xl, yl, k0);
        self.cs.to_global_vec(ex, ey, ez)
    }

    /// Incident field U_inc = -2jβ · E_mode (for Robin BC excitation)
    pub fn u_inc_global(&self, x: f64, y: f64, z: f64, k0: f64) -> [C64; 3] {
        let e = self.mode_field_global(x, y, z, k0);
        let factor = C64::new(0.0, -2.0 * self.beta(k0));
        [factor * C64::from(e[0]), factor * C64::from(e[1]), factor * C64::from(e[2])]
    }
}

/// Detect port face coordinate system from mesh data.
/// Mirrors EMerge's rect_basis(): xhat = broad wall, yhat = zhat×xhat, zhat = outward normal.
/// Uses adjacent tet centroid to orient the normal outward (away from mesh interior).
pub fn detect_port_cs(
    mesh: &crate::mesh::Mesh,
    tri_indices: &[usize],
) -> CoordinateSystem {
    let nodes = &mesh.nodes;
    let tris = &mesh.tris;

    // Compute center of port face
    let mut center = [0.0f64; 3];
    let mut count = 0.0;
    let mut all_verts = std::collections::HashSet::new();
    for &ti in tri_indices {
        for &vi in &tris[ti] {
            if all_verts.insert(vi) {
                for k in 0..3 { center[k] += nodes[vi][k]; }
                count += 1.0;
            }
        }
    }
    for k in 0..3 { center[k] /= count; }

    // Compute normal from first triangle
    let first_tri = tris[tri_indices[0]];
    let v0 = nodes[first_tri[0]];
    let v1 = nodes[first_tri[1]];
    let v2 = nodes[first_tri[2]];
    let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
    let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
    let mut normal = normalize(&cross(&e1, &e2));

    // Orient outward: normal should point AWAY from the adjacent tet's centroid.
    // Mirrors EMerge's outward_normal() which uses a reference origin.
    let adj_tet = mesh.tri_to_tet[tri_indices[0]][0];
    let tet = &mesh.tets[adj_tet];
    let tet_center = [
        (nodes[tet[0]][0] + nodes[tet[1]][0] + nodes[tet[2]][0] + nodes[tet[3]][0]) / 4.0,
        (nodes[tet[0]][1] + nodes[tet[1]][1] + nodes[tet[2]][1] + nodes[tet[3]][1]) / 4.0,
        (nodes[tet[0]][2] + nodes[tet[1]][2] + nodes[tet[2]][2] + nodes[tet[3]][2]) / 4.0,
    ];
    // If normal points toward tet center, flip it
    let to_tet = [tet_center[0]-center[0], tet_center[1]-center[1], tet_center[2]-center[2]];
    if normal[0]*to_tet[0] + normal[1]*to_tet[1] + normal[2]*to_tet[2] > 0.0 {
        normal = [-normal[0], -normal[1], -normal[2]];
    }

    // Find extents along each axis on the port face
    let mut min_coords = [f64::INFINITY; 3];
    let mut max_coords = [f64::NEG_INFINITY; 3];
    for &vi in &all_verts {
        for k in 0..3 {
            min_coords[k] = min_coords[k].min(nodes[vi][k]);
            max_coords[k] = max_coords[k].max(nodes[vi][k]);
        }
    }
    let extents = [max_coords[0]-min_coords[0], max_coords[1]-min_coords[1], max_coords[2]-min_coords[2]];

    // Dominant normal axis
    let normal_axis = if normal[0].abs() > normal[1].abs() && normal[0].abs() > normal[2].abs() { 0 }
        else if normal[1].abs() > normal[2].abs() { 1 } else { 2 };

    // xhat = along the axis with largest extent (broad wall)
    let mut face_axes: Vec<(usize, f64)> = (0..3)
        .filter(|&k| k != normal_axis)
        .map(|k| (k, extents[k]))
        .collect();
    face_axes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let broad_axis = face_axes[0].0;
    let mut xhat = [0.0; 3];
    xhat[broad_axis] = 1.0;

    CoordinateSystem::from_port_face(center, normal, xhat)
}

fn normalize(v: &[f64; 3]) -> [f64; 3] {
    let n = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt();
    if n < 1e-30 { return [0.0; 3]; }
    [v[0]/n, v[1]/n, v[2]/n]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]
}
