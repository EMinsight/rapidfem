// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! The per-frequency electrical state, in one place.
//!
//! Before this existed, every site that needed a wavenumber recomputed
//! `k0 = 2π·freq/c0` (9 call sites) and every port that needed the angular
//! frequency recovered it as `ω = k0·c0` (4 sites) — a round-trip that is both
//! duplicated and, once `k0` is non-dimensionalized, wrong. `Excitation` is the
//! single source of truth: build it once per frequency, pass `&Excitation` to
//! assembly and ports, read `k0` for wavenumbers and `omega` for the angular
//! frequency directly (never recovered from `k0`).
//!
//! This separation is what makes the characteristic-length non-dimensionalization
//! (lever ④) a one-line change: `k0` can be scaled to `κ = k0·L0` at the
//! constructor while `omega` stays the true physical angular frequency, so the
//! ω-dependent material/impedance physics is untouched. See
//! `derivations/basis_nondim/`.

use crate::constants::{C0, PI};

/// The electromagnetic state at one frequency: the angular frequency and the
/// free-space wavenumber, derived once and carried together.
#[derive(Clone, Copy, Debug)]
pub struct Excitation {
    /// Drive frequency f (Hz).
    pub freq: f64,
    /// Free-space wavenumber k₀ = ω/c₀ (rad/m). The quantity scaled by the
    /// characteristic length under lever ④.
    pub k0: f64,
    /// Angular frequency ω = 2πf (rad/s). The true physical ω — used by all
    /// ω-dependent material/impedance terms, never recovered from `k0`.
    pub omega: f64,
}

impl Excitation {
    /// Build the electrical state for a drive frequency `freq` (Hz).
    ///
    /// `k0 = ω/c₀` is bit-identical to the previous `2.0*PI*freq/C0` sites
    /// (same evaluation order), so this consolidation is behavior-preserving;
    /// only the former `ω = k0·c0` recoveries change to the canonical `2πf`
    /// (a sub-ulp shift that cancels in S-parameter ratios).
    pub fn new(freq: f64) -> Self {
        let omega = 2.0 * PI * freq;
        Excitation { freq, k0: omega / C0, omega }
    }
}
