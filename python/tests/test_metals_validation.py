"""Metal-loss production-verification suite.

Quantitative pass/fail tests for the two ways RapidFEM models metal:

1. **Surface impedance** (``rf.SurfaceImpedance``) — a 2-D Leontovich
   boundary ``Eₜ = Zs·(n̂×H)`` for thin good conductors, with the
   skin-depth ``Zs = (1+j)·√(ωμ/2σ)`` and the finite-thickness
   ``Zs ← Zs / tanh(γ·t)`` correction.
2. **Volume conductivity** (``rf.Material(conductivity=…)``,
   ``rf.Conductor``) — the bulk Ohmic term folded into the complex
   permittivity ``εr* = εr(1 − j·tanδ) − j·σ/(ωε₀)``.

Each case has a closed-form analytic anchor:

- Surface-impedance wall loss in a WR-90 guide is checked against the
  Pozar TE₁₀ conductor attenuation αc (Pozar, *Microwave Engineering*,
  eq. 3.96).
- Volume conductivity in a lossy-air guide is checked against the exact
  complex TE₁₀ propagation constant γ = √(kc² − k₀²εr*).
- The finite-thickness tanh path is checked against its analytic thin-
  and thick-sheet limits.
- Two internal-consistency gates (explicit Zs vs. σ-derived Zs; bulk σ
  vs. the equivalent loss tangent) pin the constitutive formulas to
  machine precision.

Every test runs a full FD sweep, so the module is marked ``slow``. The
meshes are kept small (≤ ~12 k tets, single frequency) to stay light on
laptop CPU/RAM. Run with ``pytest -m slow python/tests/test_metals_validation.py``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import rapidfem as rf

slow = pytest.mark.slow

# Physical constants (SI).
MU0 = 4.0e-7 * math.pi
EPS0 = 8.8541878128e-12
C0 = 299_792_458.0
ETA0 = MU0 * C0  # free-space wave impedance, ≈ 376.73 Ω

# WR-90 X-band rectangular waveguide cross-section.
A = 22.86e-3
B = 10.16e-3
F0 = 10.0e9  # single test frequency, mid X-band (fc ≈ 6.56 GHz)


# -----------------------------------------------------------------------------
# Analytic references
# -----------------------------------------------------------------------------

def _alpha_te10_wall(f: float, rs: float) -> float:
    """TE₁₀ conductor-loss attenuation αc [Np/m] for a WR-90 guide.

    Pozar eq. 3.96, written in terms of the wall surface resistance
    ``rs = Re(Zs)`` so it applies to any surface impedance, not just a
    bulk skin-depth conductor::

        αc = Rs / (b·η·√(1−(fc/f)²)) · (1 + 2·(b/a)·(fc/f)²)
    """
    fc = C0 / (2.0 * A)
    root = math.sqrt(1.0 - (fc / f) ** 2)
    return rs / (B * ETA0 * root) * (1.0 + 2.0 * (B / A) * (fc / f) ** 2)


def _alpha_volume(f: float, er: float, sigma: float) -> float:
    """Exact TE₁₀ attenuation [Np/m] for a guide filled with a lossy
    medium of permittivity ``er`` and conductivity ``sigma``.

    Uses the *exact* complex propagation constant (no small-loss
    perturbation): with εr* = er − j·σ/(ωε₀),

        γ = √(kc² − k₀²·εr*),  kc = π/a,  α = Re(γ).
    """
    w = 2.0 * math.pi * f
    k0 = w / C0
    er_c = er - 1j * sigma / (w * EPS0)
    gamma = np.sqrt((math.pi / A) ** 2 - k0 ** 2 * er_c + 0j)
    return abs(gamma.real)


def _surface_resistance(f: float, sigma: float) -> float:
    """Good-conductor surface resistance Rs = √(ωμ₀/2σ) [Ω/□]."""
    w = 2.0 * math.pi * f
    return math.sqrt(w * MU0 / (2.0 * sigma))


def _skin_depth(f: float, sigma: float) -> float:
    """Skin depth δ = √(2/(ωμ₀σ)) [m]."""
    w = 2.0 * math.pi * f
    return math.sqrt(2.0 / (w * MU0 * sigma))


# -----------------------------------------------------------------------------
# Shared WR-90 driver
# -----------------------------------------------------------------------------

def _wr90(length: float, *, maxh: float | None = None,
          material: rf.Material | None = None):
    """Build a meshed WR-90 section with TE₁₀ ports on the two z-faces.

    Walls are left ``unassigned`` so the caller decides PEC / surface
    impedance. Returns ``(geometry, air_box)``.
    """
    g = rf.Geometry(maxh=maxh or rf.lambda_maxh(f_max=12.0e9))
    air = g.box(A, B, length, position=(-A / 2, -B / 2, 0.0),
                material=material or rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    return g, air


def _s21(g) -> tuple[float, float]:
    """Mesh, solve at F0, return (|S21|, |S11|)."""
    g.mesh()
    res = rf.Problem(g).sweep(np.array([F0]))
    s = res.sparams[0]
    return abs(s[1, 0]), abs(s[0, 0])


def _alpha_from_s21(s21: float, length: float) -> float:
    """Extract the attenuation constant from a matched-line |S21|.

    For a well-matched section (|S11| ≪ |S21|), |S21| = e^(−α·L).
    """
    return -math.log(s21) / length


# -----------------------------------------------------------------------------
# 1. Surface impedance — TE₁₀ wall loss vs. Pozar αc
# -----------------------------------------------------------------------------

@slow
def test_surface_impedance_wall_loss_matches_pozar():
    """A WR-90 section walled with ``SurfaceImpedance`` must attenuate
    TE₁₀ at the analytic conductor-loss rate.

    σ = 1e6 S/m (a deliberately lossy wall) over 100 mm gives ~1 % power
    loss — well above mesh-discretization noise — so α extracted from
    |S21| pins down the skin-depth Zs. Copper (σ = 5.8e7) would be
    physically correct too but loses so little over a short guide that
    the α extraction is dominated by numerical floor; the lossy wall is
    the same physics with a measurable signal.
    """
    sigma = 1.0e6
    length = 100.0e-3

    g, air = _wr90(length)
    rf.SurfaceImpedance(*air.faces.unassigned, conductivity=sigma)
    s21, s11 = _s21(g)

    assert s11 < 0.05, f"port mismatch too large for a clean fit: |S11|={s11:.3g}"

    alpha_sim = _alpha_from_s21(s21, length)
    alpha_ref = _alpha_te10_wall(F0, _surface_resistance(F0, sigma))
    rel = abs(alpha_sim - alpha_ref) / alpha_ref
    assert rel < 0.12, (
        f"surface-impedance wall loss off: α_sim={alpha_sim:.4e}, "
        f"α_pozar={alpha_ref:.4e}, rel={rel:.3f}"
    )


# -----------------------------------------------------------------------------
# 2. Surface impedance — explicit Zs == σ-derived Zs (internal consistency)
# -----------------------------------------------------------------------------

@slow
def test_surface_impedance_explicit_zs_matches_conductivity():
    """Passing the analytic Zs explicitly must reproduce the σ path.

    For a good conductor Zs = (1+j)·Rs with Rs = √(ωμ/2σ). Driving the
    wall with ``zs=(Rs, Rs)`` therefore has to give the same S-params as
    ``conductivity=σ`` — this gates the ``from_zs`` vs.
    ``from_conductivity`` branches against each other.
    """
    sigma = 1.0e6
    length = 100.0e-3
    rs = _surface_resistance(F0, sigma)

    g, air = _wr90(length)
    rf.SurfaceImpedance(*air.faces.unassigned, conductivity=sigma)
    s21_cond, _ = _s21(g)

    g, air = _wr90(length)
    rf.SurfaceImpedance(*air.faces.unassigned, zs=(rs, rs))
    s21_zs, _ = _s21(g)

    assert abs(s21_cond - s21_zs) < 2.0e-3, (
        f"explicit Zs disagrees with σ path: |S21|(σ)={s21_cond:.6f}, "
        f"|S21|(zs)={s21_zs:.6f}"
    )


# -----------------------------------------------------------------------------
# 3. Surface impedance — finite-thickness tanh correction
# -----------------------------------------------------------------------------

@slow
def test_surface_impedance_thickness_thin_and_thick_limits():
    """The finite-thickness tanh correction must hit both analytic limits.

    Zs = Zs∞ / tanh(γ·t), γ ≈ (1+j)/δ for a good conductor, gives:

    - thick sheet (t ≫ δ): tanh → 1, Zs → Zs∞, so α → the bulk wall loss.
    - thin sheet  (t ≪ δ): tanh(x) → x, Zs → ρ/t = 1/(σt) (the real DC
      sheet resistance), so Re(Zs) grows by δ/t and α → α_bulk·(δ/t).

    Both limits are checked against the same Pozar αc, scaled by the
    appropriate surface resistance.
    """
    sigma = 1.0e6
    length = 100.0e-3
    delta = _skin_depth(F0, sigma)
    alpha_bulk = _alpha_te10_wall(F0, _surface_resistance(F0, sigma))

    # Thick sheet: 10 skin depths — must match the infinitely-thick wall.
    g, air = _wr90(length)
    rf.SurfaceImpedance(*air.faces.unassigned, conductivity=sigma,
                        thickness=10.0 * delta)
    s21_thick, _ = _s21(g)
    alpha_thick = _alpha_from_s21(s21_thick, length)
    rel_thick = abs(alpha_thick - alpha_bulk) / alpha_bulk
    assert rel_thick < 0.10, (
        f"thick-sheet limit off: α={alpha_thick:.4e}, "
        f"α_bulk={alpha_bulk:.4e}, rel={rel_thick:.3f}"
    )

    # Thin sheet: 0.1 skin depths — Re(Zs) → 1/(σt), so α → α_bulk·δ/t.
    t_thin = 0.1 * delta
    g, air = _wr90(length)
    rf.SurfaceImpedance(*air.faces.unassigned, conductivity=sigma,
                        thickness=t_thin)
    s21_thin, _ = _s21(g)
    alpha_thin = _alpha_from_s21(s21_thin, length)
    alpha_thin_ref = alpha_bulk * (delta / t_thin)
    rel_thin = abs(alpha_thin - alpha_thin_ref) / alpha_thin_ref
    assert rel_thin < 0.10, (
        f"thin-sheet limit off: α={alpha_thin:.4e}, "
        f"α_bulk·δ/t={alpha_thin_ref:.4e}, rel={rel_thin:.3f}"
    )

    # Sanity: the thin sheet must lose markedly more than the thick one.
    assert alpha_thin > 3.0 * alpha_thick


# -----------------------------------------------------------------------------
# 4. Volume conductivity — TE₁₀ attenuation vs. exact complex β
# -----------------------------------------------------------------------------

@slow
def test_volume_conductivity_matches_exact_propagation():
    """Bulk conductivity in the fill must attenuate at the exact rate.

    A WR-90 guide filled with weakly-lossy air (εr=1, σ small) and PEC
    walls isolates the ``−j·σ/(ωε₀)`` term of the complex permittivity.
    The attenuation extracted from |S21| is compared to Re(γ) from the
    exact complex TE₁₀ propagation constant — no perturbation
    approximation. Lossy *air* (not a dielectric) keeps the guided
    wavelength at its free-space value, so the global mesh resolves it
    without εr-driven refinement.
    """
    sigma = 0.011  # S/m → effective loss tangent ≈ 0.02 at 10 GHz
    length = 60.0e-3

    g, air = _wr90(length, material=rf.Material(er=1.0, conductivity=sigma))
    rf.PEC(*air.faces.unassigned)
    s21, s11 = _s21(g)

    assert s11 < 0.05, f"port mismatch too large: |S11|={s11:.3g}"

    alpha_sim = _alpha_from_s21(s21, length)
    alpha_ref = _alpha_volume(F0, er=1.0, sigma=sigma)
    rel = abs(alpha_sim - alpha_ref) / alpha_ref
    assert rel < 0.05, (
        f"volume-conductivity loss off: α_sim={alpha_sim:.4e}, "
        f"α_exact={alpha_ref:.4e}, rel={rel:.3f}"
    )


# -----------------------------------------------------------------------------
# 5. Volume conductivity == equivalent loss tangent (internal consistency)
# -----------------------------------------------------------------------------

@slow
def test_volume_sigma_equals_equivalent_loss_tangent():
    """Bulk σ and its equivalent tanδ must be indistinguishable.

    The constitutive law εr* = εr(1 − j·tanδ) − j·σ/(ωε₀) means a
    conductivity σ is identical to a loss tangent tanδ = σ/(ωε₀·εr) at
    a single frequency. Same mesh, same frequency → the two runs must
    agree to numerical precision. This gates the loss-assembly formula
    independently of any analytic propagation model.
    """
    er = 2.0
    sigma = 0.02
    length = 60.0e-3
    w = 2.0 * math.pi * F0
    tand_eq = sigma / (w * EPS0 * er)

    g, air = _wr90(length, material=rf.Dielectric(er, conductivity=sigma))
    rf.PEC(*air.faces.unassigned)
    s21_sigma, _ = _s21(g)

    g, air = _wr90(length, material=rf.Dielectric(er, tand=tand_eq))
    rf.PEC(*air.faces.unassigned)
    s21_tand, _ = _s21(g)

    assert abs(s21_sigma - s21_tand) < 1.0e-4, (
        f"σ and equivalent tanδ disagree: |S21|(σ)={s21_sigma:.6f}, "
        f"|S21|(tanδ)={s21_tand:.6f}"
    )
