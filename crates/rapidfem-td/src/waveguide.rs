//! Analytic rectangular-waveguide port modes.
//!
//! A rectangular waveguide of width `a` and height `b` supports `TE_mn`
//! modes with closed-form transverse field profiles. For a time-domain
//! port these profiles shape the incident excitation and the modal
//! extraction; the guide dispersion itself emerges from the PEC walls
//! during propagation, so only the transverse profile and the cutoff are
//! needed here. Everything is in the solver's normalised units
//! (`c = ε₀ = μ₀ = 1`).

use crate::constants::Field;
/// Pi in the operator's working precision (`Field`).
const PI: Field = std::f64::consts::PI as Field;

#[inline]
fn dot(a: [Field; 3], b: [Field; 3]) -> Field {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [Field; 3], b: [Field; 3]) -> [Field; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// A rectangular-waveguide port: its in-plane coordinate frame, cross
/// section, and `TE_mn` mode.
///
/// `u_hat`, `v_hat`, `w_hat` form a right-handed frame (`û × v̂ = ŵ`) with
/// `w_hat` the **inward** normal — pointing into the simulation domain.
#[derive(Clone, Debug)]
pub struct RectPort {
    /// A corner of the port rectangle (global coords) — the `(u,v)=(0,0)` point.
    pub origin: [Field; 3],
    /// Unit vector along the width `a` (global).
    pub u_hat: [Field; 3],
    /// Unit vector along the height `b` (global).
    pub v_hat: [Field; 3],
    /// Inward unit normal — points into the domain (global).
    pub w_hat: [Field; 3],
    /// Cross-section width.
    pub a: Field,
    /// Cross-section height.
    pub b: Field,
    /// `TE` mode indices `(m, n)`.
    pub mode: (usize, usize),
}

impl RectPort {
    /// Local `(u, v)` coordinates of a global point on the port plane.
    fn local(&self, x: [Field; 3]) -> (Field, Field) {
        let d = [
            x[0] - self.origin[0],
            x[1] - self.origin[1],
            x[2] - self.origin[2],
        ];
        (dot(d, self.u_hat), dot(d, self.v_hat))
    }

    /// Transverse electric-field profile of the mode at a global point on
    /// the port face, in global coordinates and normalised so the dominant
    /// component peaks at unit amplitude.
    ///
    /// `TE_mn`: `E_u ∝ (n/b)·cos(mπu/a)·sin(nπv/b)`,
    /// `E_v ∝ −(m/a)·sin(mπu/a)·cos(nπv/b)`. The sentinel mode `(0, 0)` is
    /// a **lumped / TEM port** — a uniform transverse field along `v_hat`,
    /// with zero cutoff and a flat (non-dispersive) `Z = 1` impedance.
    pub fn e_profile(&self, x: [Field; 3]) -> [Field; 3] {
        if self.mode == (0, 0) {
            return self.v_hat;
        }
        let (u, v) = self.local(x);
        let (m, n) = (self.mode.0 as Field, self.mode.1 as Field);
        let mu = m * PI / self.a;
        let nv = n * PI / self.b;
        let eu = (n / self.b) * (mu * u).cos() * (nv * v).sin();
        let ev = -(m / self.a) * (mu * u).sin() * (nv * v).cos();
        let scale = (m / self.a).max(n / self.b).max(Field::MIN_POSITIVE);
        let (eu, ev) = (eu / scale, ev / scale);
        [
            eu * self.u_hat[0] + ev * self.v_hat[0],
            eu * self.u_hat[1] + ev * self.v_hat[1],
            eu * self.u_hat[2] + ev * self.v_hat[2],
        ]
    }

    /// Transverse magnetic-field profile for a mode propagating along the
    /// inward normal — `h_t = ŵ × e_t` (free-space impedance in the
    /// solver's normalised units). Global coordinates.
    pub fn h_profile(&self, x: [Field; 3]) -> [Field; 3] {
        cross(self.w_hat, self.e_profile(x))
    }

    /// Cutoff angular frequency `ω_c = π·√((m/a)² + (n/b)²)` (`c = 1`).
    /// Content below `ω_c` is evanescent and does not propagate.
    pub fn cutoff(&self) -> Field {
        let (m, n) = (self.mode.0 as Field, self.mode.1 as Field);
        PI * ((m / self.a).powi(2) + (n / self.b).powi(2)).sqrt()
    }

    /// Fit a `RectPort` to an axis-aligned port face from its mesh node
    /// coordinates and the inward normal (pointing into the domain).
    ///
    /// The wider transverse dimension becomes the width `a` (`u_hat`), the
    /// narrower the height `b` (`v_hat`); the frame is made right-handed
    /// (`û × v̂ = ŵ`). The `TE_mn` mode then has `m` indexing the wide
    /// dimension — so `TE₁₀` is the dominant mode regardless of orientation.
    ///
    /// `field_axis` overrides the auto-fit transverse axis `v̂`: a lumped
    /// port's voltage-integration direction is projected into the port
    /// plane and used as `v̂` (with `û` rebuilt to keep the frame
    /// right-handed). `None` keeps the auto-fit. A direction parallel to
    /// the normal has no in-plane part and is ignored.
    pub fn from_face(
        nodes: &[[Field; 3]],
        inward_normal: [Field; 3],
        mode: (usize, usize),
        field_axis: Option<[Field; 3]>,
    ) -> RectPort {
        // The inward normal is ±eₖ — the constant (out-of-plane) axis.
        let k = (0..3)
            .max_by(|&i, &j| {
                inward_normal[i]
                    .abs()
                    .partial_cmp(&inward_normal[j].abs())
                    .unwrap()
            })
            .unwrap();
        let s = if inward_normal[k] >= 0.0 { 1.0 } else { -1.0 };
        let others: Vec<usize> = (0..3).filter(|&x| x != k).collect();
        let extent = |ax: usize| -> (Field, Field) {
            let lo = nodes.iter().map(|p| p[ax]).fold(Field::MAX, Field::min);
            let hi = nodes.iter().map(|p| p[ax]).fold(Field::MIN, Field::max);
            (lo, hi - lo)
        };
        let (lo0, ext0) = extent(others[0]);
        let (lo1, ext1) = extent(others[1]);
        // Wide axis → width a, narrow → height b.
        let (wide, narrow, a, b, lo_w, lo_n) = if ext0 >= ext1 {
            (others[0], others[1], ext0, ext1, lo0, lo1)
        } else {
            (others[1], others[0], ext1, ext0, lo1, lo0)
        };
        let mut u_hat = [0.0; 3];
        u_hat[wide] = 1.0;
        let mut v_hat = [0.0; 3];
        v_hat[narrow] = 1.0;
        let mut w_hat = [0.0; 3];
        w_hat[k] = s;
        let mut origin = [0.0; 3];
        origin[wide] = lo_w;
        origin[k] = nodes[0][k];
        // Make (u, v, w) right-handed; flipping v̂ moves its origin corner.
        if dot(cross(u_hat, v_hat), w_hat) >= 0.0 {
            origin[narrow] = lo_n;
        } else {
            v_hat[narrow] = -1.0;
            origin[narrow] = lo_n + b;
        }
        // An explicit field axis (a lumped port's voltage-integration
        // direction) overrides the auto-fit transverse axis: project it
        // into the port plane and rebuild a right-handed (û, v̂, ŵ) frame.
        if let Some(d) = field_axis {
            let dn = dot(d, w_hat);
            let proj =
                [d[0] - dn * w_hat[0], d[1] - dn * w_hat[1], d[2] - dn * w_hat[2]];
            let len = dot(proj, proj).sqrt();
            if len > 1e-9 {
                v_hat = [proj[0] / len, proj[1] / len, proj[2] / len];
                u_hat = cross(v_hat, w_hat); // û = v̂×ŵ ⇒ û×v̂ = ŵ
            }
        }
        RectPort { origin, u_hat, v_hat, w_hat, a, b, mode }
    }

    /// `TE`-mode wave impedance at angular frequency `omega`, in the
    /// solver's normalised units (`Z₀ = 1`): `Z_TE = 1/√(1 − (ω_c/ω)²)`.
    ///
    /// This is the ratio `|E_t|/|H_t|` of the propagating mode. The
    /// forward/backward modal split uses it, and because it is
    /// frequency-dependent the split must be done per frequency. Valid
    /// for `omega > cutoff`.
    pub fn te_impedance(&self, omega: Field) -> Field {
        let r = self.cutoff() / omega;
        1.0 / (1.0 - r * r).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An axis-aligned port on the z-plane: width along x, height along y,
    /// inward normal +z.
    fn z_port(a: f64, b: f64, mode: (usize, usize)) -> RectPort {
        RectPort {
            origin: [0.0, 0.0, 0.0],
            u_hat: [1.0, 0.0, 0.0],
            v_hat: [0.0, 1.0, 0.0],
            w_hat: [0.0, 0.0, 1.0],
            a,
            b,
            mode,
        }
    }

    #[test]
    fn te10_profile_matches_the_analytic_field() {
        // TE₁₀: E = v̂·sin(πx/a) — peaks mid-width, vanishes at the side
        // walls, is purely transverse-vertical.
        let p = z_port(0.5, 0.25, (1, 0));
        // Side walls u = 0, a → zero.
        for &u in &[0.0, 0.5] {
            let e = p.e_profile([u, 0.1, 0.0]);
            assert!(e.iter().all(|c| c.abs() < 1e-12), "wall E = {e:?}");
        }
        // Mid-width u = a/2 → unit peak, along v̂ (y).
        let e = p.e_profile([0.25, 0.1, 0.0]);
        assert!((e[1].abs() - 1.0).abs() < 1e-12, "peak |E| = {}", e[1].abs());
        assert!(e[0].abs() < 1e-12 && e[2].abs() < 1e-12, "E not along v̂");
        // Sine shape across the width.
        let e_q = p.e_profile([0.125, 0.1, 0.0]); // u = a/4
        assert!(
            (e_q[1].abs() - (PI * 0.25).sin()).abs() < 1e-12,
            "E(a/4) = {}",
            e_q[1].abs(),
        );
    }

    #[test]
    fn h_profile_is_an_inward_propagating_partner() {
        // h_t = ŵ × e_t ⇒ E, H, and the propagation direction are mutually
        // orthogonal and E × H points inward (ŵ).
        let p = z_port(0.5, 0.25, (1, 0));
        let x = [0.25, 0.1, 0.0];
        let e = p.e_profile(x);
        let h = p.h_profile(x);
        assert!(dot(e, h).abs() < 1e-12, "E·H = {}", dot(e, h));
        let poynting = cross(e, h);
        assert!(
            dot(poynting, p.w_hat) > 0.0,
            "E×H must point inward, got {poynting:?}",
        );
        // |h| = |e| for a transverse profile crossed with the unit normal.
        let (ne, nh) = (dot(e, e).sqrt(), dot(h, h).sqrt());
        assert!((ne - nh).abs() < 1e-12, "|E| {ne} vs |H| {nh}");
    }

    #[test]
    fn from_face_fits_an_axis_aligned_port() {
        // A z = 3 face spanning [0,2]×[0,1], inward normal −ẑ.
        let nodes = [
            [0.0, 0.0, 3.0],
            [2.0, 0.0, 3.0],
            [2.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
            [1.0, 0.5, 3.0],
        ];
        let p = RectPort::from_face(&nodes, [0.0, 0.0, -1.0], (1, 0), None);
        // Width = the larger span (x), height = the smaller (y).
        assert!((p.a - 2.0).abs() < 1e-12, "a = {}", p.a);
        assert!((p.b - 1.0).abs() < 1e-12, "b = {}", p.b);
        assert!((p.w_hat[2] + 1.0).abs() < 1e-12, "inward normal");
        // Right-handed frame.
        let uxv = cross(p.u_hat, p.v_hat);
        assert!(dot(uxv, p.w_hat) > 0.999, "frame not right-handed");
        // The TE₁₀ profile peaks mid-width and vanishes at the side walls.
        let mid = p.e_profile([1.0, 0.5, 3.0]);
        assert!(mid.iter().map(|c| c * c).sum::<f64>().sqrt() > 0.99);
        let wall = p.e_profile([0.0, 0.5, 3.0]);
        assert!(wall.iter().all(|c| c.abs() < 1e-9), "side-wall E ≠ 0");
    }

    #[test]
    fn from_face_honours_an_explicit_field_axis() {
        // A z = 0 face spanning [0,2]×[0,1] — auto-fit makes v̂ the narrow
        // (y) axis. An explicit field axis must override that.
        let nodes = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.5, 0.0],
        ];
        let n = [0.0, 0.0, 1.0]; // inward +z
        // Auto-fit picks the narrow (y) axis for v̂.
        let auto = RectPort::from_face(&nodes, n, (0, 0), None);
        assert!((auto.v_hat[1].abs() - 1.0).abs() < 1e-12, "auto v̂ ≠ ±ŷ");
        // An explicit x axis (the wide axis) overrides the auto-fit.
        let p =
            RectPort::from_face(&nodes, n, (0, 0), Some([1.0, 0.0, 0.0]));
        assert!(
            (p.v_hat[0] - 1.0).abs() < 1e-12
                && p.v_hat[1].abs() < 1e-12
                && p.v_hat[2].abs() < 1e-12,
            "v̂ not set to x̂: {:?}",
            p.v_hat,
        );
        // The (0,0) field profile follows the explicit axis.
        let e = p.e_profile([1.0, 0.5, 0.0]);
        assert!((e[0] - 1.0).abs() < 1e-12 && e[1].abs() < 1e-12);
        // The frame stays right-handed.
        let uxv = cross(p.u_hat, p.v_hat);
        assert!(dot(uxv, p.w_hat) > 0.999, "frame not right-handed");
        // A direction with an out-of-plane component is projected back
        // into the port plane.
        let q =
            RectPort::from_face(&nodes, n, (0, 0), Some([0.0, 3.0, 9.0]));
        assert!(
            (q.v_hat[1] - 1.0).abs() < 1e-12 && q.v_hat[2].abs() < 1e-12,
            "out-of-plane direction not projected: {:?}",
            q.v_hat,
        );
    }

    #[test]
    fn lumped_port_is_a_uniform_zero_cutoff_mode() {
        // The (0,0) sentinel mode — a lumped / TEM port: uniform transverse
        // field, no cutoff, flat Z = 1 impedance.
        let p = z_port(0.5, 0.25, (0, 0));
        // Uniform field along v̂ everywhere on the face.
        for &(u, v) in &[(0.1, 0.05), (0.25, 0.1), (0.49, 0.2)] {
            let e = p.e_profile([u, v, 0.0]);
            assert!((e[1] - 1.0).abs() < 1e-12, "not uniform: {e:?}");
            assert!(e[0].abs() < 1e-12 && e[2].abs() < 1e-12);
        }
        // No cutoff, and the impedance is flat (non-dispersive).
        assert!(p.cutoff().abs() < 1e-12, "lumped port has a cutoff");
        for &omega in &[0.3, 1.0, 5.0] {
            assert!((p.te_impedance(omega) - 1.0).abs() < 1e-12);
        }
        // E × H still points inward (a forward-propagating partner).
        let x = [0.25, 0.1, 0.0];
        let poynting = cross(p.e_profile(x), p.h_profile(x));
        assert!(dot(poynting, p.w_hat) > 0.99);
    }

    #[test]
    fn cutoff_matches_the_analytic_frequency() {
        // TE₁₀ of an a = 0.5 guide: ω_c = π/a = 2π.
        let p = z_port(0.5, 0.25, (1, 0));
        assert!((p.cutoff() - 2.0 * PI).abs() < 1e-12);
        // TE₁₁: ω_c = π·√((1/a)² + (1/b)²).
        let p11 = z_port(0.5, 0.25, (1, 1));
        let want = PI * ((1.0 / 0.5_f64).powi(2) + (1.0 / 0.25_f64).powi(2)).sqrt();
        assert!((p11.cutoff() - want).abs() < 1e-12);
    }
}
