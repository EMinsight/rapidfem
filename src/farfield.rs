//! Near-field to far-field transformation (NFFT) for radiation patterns.
//!
//! Uses the equivalence principle on a closed surface (typically the ABC boundary):
//!   J_s = n̂ × H     (equivalent electric current)
//!   M_s = -n̂ × E    (equivalent magnetic current)
//!
//! Far-field radiation integrals:
//!   N(θ,φ) = ∫∫ J_s · e^{jk r̂·r'} dS'
//!   L(θ,φ) = ∫∫ M_s · e^{jk r̂·r'} dS'
//!
//! E_θ^far = -jk/(4π) (L_φ + η₀ N_θ)
//! E_φ^far = -jk/(4π) (-L_θ + η₀ N_φ)

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::interp;
use crate::error_estimator::eval_curl_in_tet;
use crate::quadrature::gaus_quad_tri;
use crate::constants::*;

/// Far-field radiation pattern result.
pub struct RadiationPattern {
    /// Theta angles in radians
    pub theta: Vec<f64>,
    /// Phi angles in radians
    pub phi: Vec<f64>,
    /// Directivity in dBi, indexed as [phi_idx][theta_idx]
    pub directivity_dbi: Vec<Vec<f64>>,
    /// E_theta component magnitude, indexed as [phi_idx][theta_idx]
    pub e_theta: Vec<Vec<f64>>,
    /// E_phi component magnitude, indexed as [phi_idx][theta_idx]
    pub e_phi: Vec<Vec<f64>>,
    /// Peak directivity in dBi
    pub peak_directivity_dbi: f64,
    /// Total radiated power (W)
    pub radiated_power: f64,
}

/// Compute the far-field radiation pattern from the FEM solution.
///
/// `nfft_tri_ids`: triangle indices of the closed surface (ABC boundary)
/// `solution`: FEM solution vector (complex DOF coefficients)
/// `frequency`: operating frequency (Hz)
/// `n_theta`: number of theta angles (0 to π)
/// `n_phi`: number of phi angles (0 to 2π)
/// `gq_order`: Gauss quadrature order for surface integration
pub fn compute_farfield(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    solution: &[C64],
    nfft_tri_ids: &[usize],
    frequency: f64,
    n_theta: usize,
    n_phi: usize,
    gq_order: usize,
) -> RadiationPattern {
    let k0 = 2.0 * PI * frequency / C0;
    let omega = 2.0 * PI * frequency;
    let j = C64::new(0.0, 1.0);

    // Build theta/phi grids
    let thetas: Vec<f64> = (0..n_theta).map(|i| PI * i as f64 / (n_theta - 1) as f64).collect();
    let phis: Vec<f64> = (0..n_phi).map(|i| 2.0 * PI * i as f64 / n_phi as f64).collect();

    // Build spatial hash for point-in-tet queries
    let grid = interp::TetGrid::new(mesh);

    // Precompute quadrature points and fields on the NFFT surface
    let quad_pts = gaus_quad_tri(gq_order);

    // For each observation direction (theta, phi), compute N and L integrals
    let mut directivity = vec![vec![0.0f64; n_theta]; n_phi];
    let mut e_theta_mag = vec![vec![0.0f64; n_theta]; n_phi];
    let mut e_phi_mag = vec![vec![0.0f64; n_theta]; n_phi];

    // Precompute surface data: for each triangle, store quadrature point data
    // (position, E-field, H-field, normal, area*weight)
    struct SurfPoint {
        pos: [f64; 3],
        e: [C64; 3],
        h: [C64; 3],
        normal: [f64; 3],
        aw: f64, // area * quadrature weight
    }

    let mut surf_data: Vec<SurfPoint> = Vec::new();

    for &tri_idx in nfft_tri_ids {
        let tri = mesh.tris[tri_idx];
        let v0 = mesh.nodes[tri[0]];
        let v1 = mesh.nodes[tri[1]];
        let v2 = mesh.nodes[tri[2]];

        // Triangle edges and normal
        let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
        let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
        let cr = [
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0],
        ];
        let area = 0.5 * (cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]).sqrt();
        let nn = 2.0 * area;
        let mut normal = [cr[0]/nn, cr[1]/nn, cr[2]/nn];

        // Orient normal outward using adjacent tet
        let adj_tet = mesh.tri_to_tet[tri_idx][0];
        if adj_tet != usize::MAX {
            let tet = &mesh.tets[adj_tet];
            let tc = [
                (mesh.nodes[tet[0]][0]+mesh.nodes[tet[1]][0]+mesh.nodes[tet[2]][0]+mesh.nodes[tet[3]][0])/4.0,
                (mesh.nodes[tet[0]][1]+mesh.nodes[tet[1]][1]+mesh.nodes[tet[2]][1]+mesh.nodes[tet[3]][1])/4.0,
                (mesh.nodes[tet[0]][2]+mesh.nodes[tet[1]][2]+mesh.nodes[tet[2]][2]+mesh.nodes[tet[3]][2])/4.0,
            ];
            let tri_center = [
                (v0[0]+v1[0]+v2[0])/3.0,
                (v0[1]+v1[1]+v2[1])/3.0,
                (v0[2]+v1[2]+v2[2])/3.0,
            ];
            let to_tet = [tc[0]-tri_center[0], tc[1]-tri_center[1], tc[2]-tri_center[2]];
            if normal[0]*to_tet[0] + normal[1]*to_tet[1] + normal[2]*to_tet[2] > 0.0 {
                normal = [-normal[0], -normal[1], -normal[2]];
            }
        }

        // Evaluate E and H at quadrature points
        for qp in &quad_pts {
            let (w, l1, l2, l3) = (qp[0], qp[1], qp[2], qp[3]);
            let x = v0[0]*l1 + v1[0]*l2 + v2[0]*l3;
            let y = v0[1]*l1 + v1[1]*l2 + v2[1]*l3;
            let z = v0[2]*l1 + v1[2]*l2 + v2[2]*l3;

            // Find containing tet for this point
            let tet_idx = grid.find_containing_tet(mesh, x, y, z)
                .or_else(|| {
                    // Fallback: use the adjacent tet
                    let t = mesh.tri_to_tet[tri_idx][0];
                    if t != usize::MAX { Some(t) } else { None }
                });

            if let Some(tet) = tet_idx {
                // E-field
                let (ex, ey, ez) = interp::eval_field_in_tet(mesh, basis, solution, tet, x, y, z);

                // curl(E) = jωμ₀ H  =>  H = curl(E) / (jωμ₀)
                let curl_e = eval_curl_in_tet(mesh, basis, solution, tet, x, y, z);
                let denom = j * C64::from(omega * MU0);
                let hx = curl_e[0] / denom;
                let hy = curl_e[1] / denom;
                let hz = curl_e[2] / denom;

                surf_data.push(SurfPoint {
                    pos: [x, y, z],
                    e: [ex, ey, ez],
                    h: [hx, hy, hz],
                    normal,
                    aw: area * w,
                });
            }
        }
    }

    eprintln!("  Far-field: {} surface integration points on {} triangles",
        surf_data.len(), nfft_tri_ids.len());

    // Compute far-field for each (theta, phi) direction
    let mut total_power = 0.0;

    for (ip, &phi) in phis.iter().enumerate() {
        for (it, &theta) in thetas.iter().enumerate() {
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            let sin_p = phi.sin();
            let cos_p = phi.cos();

            // Unit vectors in spherical coordinates
            let r_hat = [sin_t*cos_p, sin_t*sin_p, cos_t];
            let theta_hat = [cos_t*cos_p, cos_t*sin_p, -sin_t];
            let phi_hat = [-sin_p, cos_p, 0.0];

            // Radiation integrals N and L (vector, in θ and φ components)
            let mut nt = C64::new(0.0, 0.0);
            let mut np = C64::new(0.0, 0.0);
            let mut lt = C64::new(0.0, 0.0);
            let mut lp = C64::new(0.0, 0.0);

            for sp in &surf_data {
                // Phase: e^{jk r̂·r'}
                let rdot = r_hat[0]*sp.pos[0] + r_hat[1]*sp.pos[1] + r_hat[2]*sp.pos[2];
                let phase = (j * C64::from(k0 * rdot)).exp();
                let daw = C64::from(sp.aw) * phase;

                // J_s = n̂ × H
                let jx = C64::from(sp.normal[1])*sp.h[2] - C64::from(sp.normal[2])*sp.h[1];
                let jy = C64::from(sp.normal[2])*sp.h[0] - C64::from(sp.normal[0])*sp.h[2];
                let jz = C64::from(sp.normal[0])*sp.h[1] - C64::from(sp.normal[1])*sp.h[0];

                // M_s = -n̂ × E
                let mx = -(C64::from(sp.normal[1])*sp.e[2] - C64::from(sp.normal[2])*sp.e[1]);
                let my = -(C64::from(sp.normal[2])*sp.e[0] - C64::from(sp.normal[0])*sp.e[2]);
                let mz = -(C64::from(sp.normal[0])*sp.e[1] - C64::from(sp.normal[1])*sp.e[0]);

                // Project onto θ̂ and φ̂
                let j_t = jx*C64::from(theta_hat[0]) + jy*C64::from(theta_hat[1]) + jz*C64::from(theta_hat[2]);
                let j_p = jx*C64::from(phi_hat[0]) + jy*C64::from(phi_hat[1]) + jz*C64::from(phi_hat[2]);
                let m_t = mx*C64::from(theta_hat[0]) + my*C64::from(theta_hat[1]) + mz*C64::from(theta_hat[2]);
                let m_p = mx*C64::from(phi_hat[0]) + my*C64::from(phi_hat[1]) + mz*C64::from(phi_hat[2]);

                nt += j_t * daw;
                np += j_p * daw;
                lt += m_t * daw;
                lp += m_p * daw;
            }

            // Far-field: E_θ = -jk/(4π) (L_φ + η₀ N_θ)
            //            E_φ = -jk/(4π) (-L_θ + η₀ N_φ)
            let factor = -j * C64::from(k0 / (4.0 * PI));
            let e_t = factor * (lp + C64::from(Z0) * nt);
            let e_p = factor * (-lt + C64::from(Z0) * np);

            e_theta_mag[ip][it] = e_t.norm();
            e_phi_mag[ip][it] = e_p.norm();

            // Radiation intensity: U = (|E_θ|² + |E_φ|²) / (2η₀)
            let u = (e_t.norm().powi(2) + e_p.norm().powi(2)) / (2.0 * Z0);

            // Accumulate for total radiated power
            // P_rad = ∫∫ U sin(θ) dθ dφ
            let dtheta = PI / (n_theta - 1) as f64;
            let dphi = 2.0 * PI / n_phi as f64;
            total_power += u * sin_t * dtheta * dphi;

            directivity[ip][it] = u; // Store U temporarily, convert to D later
        }
    }

    // D(θ,φ) = 4π U(θ,φ) / P_rad
    let mut peak_d = 0.0f64;
    for ip in 0..n_phi {
        for it in 0..n_theta {
            let d = if total_power > 0.0 {
                4.0 * PI * directivity[ip][it] / total_power
            } else {
                0.0
            };
            let d_dbi = if d > 1e-30 { 10.0 * d.log10() } else { -100.0 };
            directivity[ip][it] = d_dbi;
            peak_d = peak_d.max(d_dbi);
        }
    }

    eprintln!("  Peak directivity: {:.2} dBi, P_rad: {:.4e} W", peak_d, total_power);

    RadiationPattern {
        theta: thetas,
        phi: phis,
        directivity_dbi: directivity,
        e_theta: e_theta_mag,
        e_phi: e_phi_mag,
        peak_directivity_dbi: peak_d,
        radiated_power: total_power,
    }
}

/// Write radiation pattern to a CSV file for plotting.
pub fn write_pattern_csv(
    path: &str,
    pattern: &RadiationPattern,
) -> Result<(), String> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("Cannot create {}: {}", path, e))?;

    writeln!(file, "phi_deg,theta_deg,directivity_dBi,E_theta,E_phi")
        .map_err(|e| e.to_string())?;

    for (ip, &phi) in pattern.phi.iter().enumerate() {
        for (it, &theta) in pattern.theta.iter().enumerate() {
            writeln!(file, "{:.1},{:.1},{:.4},{:.6e},{:.6e}",
                phi.to_degrees(),
                theta.to_degrees(),
                pattern.directivity_dbi[ip][it],
                pattern.e_theta[ip][it],
                pattern.e_phi[ip][it],
            ).map_err(|e| e.to_string())?;
        }
    }

    Ok(())
}

/// Write principal plane cuts (E-plane and H-plane) to CSV.
pub fn write_plane_cuts_csv(
    path: &str,
    pattern: &RadiationPattern,
) -> Result<(), String> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("Cannot create {}: {}", path, e))?;

    writeln!(file, "plane,theta_deg,directivity_dBi")
        .map_err(|e| e.to_string())?;

    // E-plane: φ=0° (xz-plane)
    let ip_e = 0;
    for (it, &theta) in pattern.theta.iter().enumerate() {
        writeln!(file, "E,{:.1},{:.4}",
            theta.to_degrees(),
            pattern.directivity_dbi[ip_e][it],
        ).map_err(|e| e.to_string())?;
    }

    // H-plane: φ=90°
    let ip_h = pattern.phi.iter().position(|&p| (p - PI/2.0).abs() < 0.01)
        .unwrap_or(pattern.phi.len() / 4);
    for (it, &theta) in pattern.theta.iter().enumerate() {
        writeln!(file, "H,{:.1},{:.4}",
            theta.to_degrees(),
            pattern.directivity_dbi[ip_h][it],
        ).map_err(|e| e.to_string())?;
    }

    Ok(())
}
