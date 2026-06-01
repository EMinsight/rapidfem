// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Port trait: unified interface for all boundary condition types.
//! Assembly uses this trait to handle RectWaveguide, LumpedPort, and ABC uniformly.

use num_complex::Complex64 as C64;

/// Unified port interface matching EMerge's RobinBC abstract class.
pub trait Port {
    /// Robin BC impedance parameter γ
    fn get_gamma(&self, k0: f64) -> C64;

    /// Incident field U_inc at a global point (for excitation vector).
    /// Returns None for non-driven ports (e.g. ABC).
    fn get_uinc(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<[C64; 3]>;

    /// Whether this port is driven (has an excitation vector)
    fn is_driven(&self) -> bool;

    /// Port mode field at a global point (for S-param extraction).
    /// Returns None for ABC (no S-param extraction).
    fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<(f64, f64, f64)>;

    /// Mode impedance Z_mode (for sparam_field_power/mode_power)
    fn z_mode(&self, k0: f64) -> f64;

    /// Port number (for S-matrix indexing)
    fn port_number(&self) -> usize;

    /// Whether this port is a lumped port (uses voltage extraction)
    fn is_lumped(&self) -> bool { false }

    /// For lumped ports: return (direction, z0, voltage) for voltage integration
    fn lumped_voltage_params(&self) -> Option<([f64; 3], f64, f64)> { None }

    /// For lumped ports: extent along the field direction (height parameter)
    fn port_height(&self) -> Option<f64> { None }
}

// Implement Port for RectWaveguide
impl Port for crate::waveguide::RectWaveguide {
    fn get_gamma(&self, k0: f64) -> C64 { self.get_gamma(k0) }
    fn get_uinc(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<[C64; 3]> {
        Some(self.get_uinc(x, y, z, k0))
    }
    fn is_driven(&self) -> bool { true }
    fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<(f64, f64, f64)> {
        Some(self.port_mode_3d_global(x, y, z, k0))
    }
    fn z_mode(&self, k0: f64) -> f64 { self.z_mode(k0) }
    fn port_number(&self) -> usize { self.port_number }
}

// Implement Port for LumpedElement
impl Port for crate::waveguide::LumpedElement {
    fn get_gamma(&self, k0: f64) -> C64 { self.get_gamma(k0) }
    fn get_uinc(&self, _x: f64, _y: f64, _z: f64, _k0: f64) -> Option<[C64; 3]> { None }
    fn is_driven(&self) -> bool { false }
    fn port_mode_3d_global(&self, _x: f64, _y: f64, _z: f64, _k0: f64) -> Option<(f64, f64, f64)> { None }
    fn z_mode(&self, _k0: f64) -> f64 { 0.0 }
    fn port_number(&self) -> usize { 0 }
}

// Implement Port for SurfaceImpedance
impl Port for crate::waveguide::SurfaceImpedance {
    fn get_gamma(&self, k0: f64) -> C64 { self.get_gamma(k0) }
    fn get_uinc(&self, _x: f64, _y: f64, _z: f64, _k0: f64) -> Option<[C64; 3]> { None }
    fn is_driven(&self) -> bool { false }
    fn port_mode_3d_global(&self, _x: f64, _y: f64, _z: f64, _k0: f64) -> Option<(f64, f64, f64)> { None }
    fn z_mode(&self, _k0: f64) -> f64 { 0.0 }
    fn port_number(&self) -> usize { 0 }
}

// Implement Port for FloquetPort
impl Port for crate::waveguide::FloquetPort {
    fn get_gamma(&self, k0: f64) -> C64 { self.get_gamma(k0) }
    fn get_uinc(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<[C64; 3]> {
        Some(self.get_uinc(x, y, z, k0))
    }
    fn is_driven(&self) -> bool { true }
    fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<(f64, f64, f64)> {
        Some(self.port_mode_3d_global(x, y, z, k0))
    }
    fn z_mode(&self, _k0: f64) -> f64 { crate::constants::Z0 }
    fn port_number(&self) -> usize { self.port_number }
}

// Implement Port for UserDefinedPort
impl Port for crate::waveguide::UserDefinedPort {
    fn get_gamma(&self, k0: f64) -> C64 { self.get_gamma(k0) }
    fn get_uinc(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<[C64; 3]> {
        Some(self.get_uinc(x, y, z, k0))
    }
    fn is_driven(&self) -> bool { true }
    fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<(f64, f64, f64)> {
        Some(self.port_mode_3d_global(x, y, z, k0))
    }
    fn z_mode(&self, _k0: f64) -> f64 { 1.0 }  // dummy; user can scale via power
    fn port_number(&self) -> usize { self.port_number }
}

// Implement Port for CoaxPort
impl Port for crate::waveguide::CoaxPort {
    fn get_gamma(&self, k0: f64) -> C64 { self.get_gamma(k0) }
    fn get_uinc(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<[C64; 3]> {
        Some(self.get_uinc(x, y, z, k0))
    }
    fn is_driven(&self) -> bool { true }
    fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<(f64, f64, f64)> {
        Some(self.port_mode_3d_global(x, y, z, k0))
    }
    fn z_mode(&self, _k0: f64) -> f64 { self.port_z() }
    fn port_number(&self) -> usize { self.port_number }
}

// Implement Port for AbsorbingBoundary
impl Port for crate::waveguide::AbsorbingBoundary {
    fn get_gamma(&self, k0: f64) -> C64 { self.get_gamma(k0) }
    fn get_uinc(&self, _x: f64, _y: f64, _z: f64, _k0: f64) -> Option<[C64; 3]> { None }
    fn is_driven(&self) -> bool { false }
    fn port_mode_3d_global(&self, _x: f64, _y: f64, _z: f64, _k0: f64) -> Option<(f64, f64, f64)> { None }
    fn z_mode(&self, _k0: f64) -> f64 { 0.0 }
    fn port_number(&self) -> usize { 0 }
}

// Implement Port for NumericalWavePort (numerical 2-D mode eigensolve)
impl Port for crate::waveguide::NumericalWavePort {
    fn get_gamma(&self, k0: f64) -> C64 { self.get_gamma(k0) }
    fn get_uinc(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<[C64; 3]> {
        Some(self.get_uinc(x, y, z, k0))
    }
    fn is_driven(&self) -> bool { true }
    fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<(f64, f64, f64)> {
        Some(self.port_mode_3d_global(x, y, z, k0))
    }
    fn z_mode(&self, k0: f64) -> f64 { self.z_mode(k0) }
    fn port_number(&self) -> usize { self.port_number }
}

// Implement Port for LumpedPort
impl Port for crate::waveguide::LumpedPort {
    fn get_gamma(&self, k0: f64) -> C64 { self.get_gamma(k0) }
    fn get_uinc(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<[C64; 3]> {
        Some(self.get_uinc(x, y, z, k0))
    }
    fn is_driven(&self) -> bool { true }
    fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, k0: f64) -> Option<(f64, f64, f64)> {
        Some(self.port_mode_3d_global(x, y, z, k0))
    }
    fn z_mode(&self, _k0: f64) -> f64 { self.z0 }
    fn port_number(&self) -> usize { self.port_number }
    fn is_lumped(&self) -> bool { true }
    fn lumped_voltage_params(&self) -> Option<([f64; 3], f64, f64)> {
        Some((self.direction, self.z0, self.voltage()))
    }
    fn port_height(&self) -> Option<f64> { Some(self.height) }
}
