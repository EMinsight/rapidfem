"""
Edge-fed patch antenna — full pipeline using Geometry + SimulationBuilder.

Replaces ``scripts/patch_antenna.py`` (which used raw gmsh + bounding-box
matching to assign physical tags). The new API uses named selectors throughout.

Compares to the existing TOML-driven flow: should produce equivalent S11 at
2.4 GHz (|S11| ≈ 0.49 at resonance, broadside D ≈ 1.8 dBi after closed NFFT).
"""
import sys
import numpy as np
import rapidfem


def main() -> int:
    # ── Antenna geometry parameters (FR-4 substrate, ~2.4 GHz design) ─────
    mm = 1e-3
    sub_w, sub_l, sub_h = 60 * mm, 60 * mm, 1.6 * mm
    er_sub = 4.4
    patch_w, patch_l = 38 * mm, 29 * mm
    feed_x, feed_y = 0.0, -patch_l / 2
    feed_width = 1.5 * mm
    air_pad_xy = 25 * mm
    air_pad_z_top = 25 * mm
    total_w, total_l = sub_w + 2 * air_pad_xy, sub_l + 2 * air_pad_xy
    total_h = sub_h + air_pad_z_top

    g = rapidfem.Geometry()

    # ── Air box (encloses everything; both substrate and free space) ───────
    air = g.box(total_w, total_l, total_h,
                position=(-total_w / 2, -total_l / 2, 0))

    # ── Substrate box (sits inside the air box, on the ground plane) ──────
    sub = g.box(sub_w, sub_l, sub_h,
                position=(-sub_w / 2, -sub_l / 2, 0))

    # ── Patch plate at the top of the substrate ────────────────────────────
    patch = g.xy_plate(patch_w, patch_l,
                       position=(-patch_w / 2, -patch_l / 2, sub_h))

    # ── Lumped port plate: vertical YZ-plane rectangle at the feed edge ────
    feed = g.plate(
        p0=(feed_x - feed_width / 2, feed_y, 0),
        width=(feed_width, 0, 0),
        height=(0, 0, sub_h),
    )

    # ── Fragment everything to make geometry conformal ─────────────────────
    g.fragment(air, sub, patch, feed)

    # ── Tag faces and volumes by selector / name ───────────────────────────
    # Ground plane = bottom face of substrate
    sub.faces.min(axis="z").name = "ground_pec"
    # Patch surface = the plate itself (now embedded in substrate top after fragment)
    patch.name = "patch_pec"
    # Lumped feed port
    feed.name = "feed"
    # ABC: outer walls of the air box (top + 4 sides; bottom is ground PEC)
    air.faces.where(lambda c, _: abs(c[2] - total_h) < 1e-9).name = "abc"
    air.faces.where(lambda c, _: abs(c[0] + total_w / 2) < 1e-9).name = "abc"
    air.faces.where(lambda c, _: abs(c[0] - total_w / 2) < 1e-9).name = "abc"
    air.faces.where(lambda c, _: abs(c[1] + total_l / 2) < 1e-9).name = "abc"
    air.faces.where(lambda c, _: abs(c[1] - total_l / 2) < 1e-9).name = "abc"
    # Materials
    sub.material = "fr4"
    air.material = "air"

    # Per-region mesh size hints
    feed.maxh = 0.5 * mm
    patch.maxh = 2 * mm
    sub.maxh = 3 * mm

    # ── Build the simulation fluently ──────────────────────────────────────
    sim = (
        rapidfem.SimulationBuilder()
        .from_geometry(g, maxh=8 * mm)
        .frequencies([2.4e9])
        .pec("ground_pec", "patch_pec")
        .lumped_port("feed", direction=(0, 0, 1), z0=50.0)
        .abc("abc", order=1)
        .material("fr4", er=er_sub)
        .material("air", er=1.0)
        .build()
    )
    g.close()

    print(f"Simulation: {sim.n_tets} tets, {sim.n_dofs} DOFs, {sim.n_driven_ports} driven ports")
    result = sim.run_sweep()

    s11 = float(abs(result.sparams[0, 0, 0]))
    print(f"|S11| @ 2.4 GHz: {s11:.4f}")

    pattern = sim.compute_farfield(result, freq_idx=0, port_idx=0, n_theta=91, n_phi=72)
    if pattern is None:
        print("FAIL: no far-field surface")
        return 1
    print(f"Peak directivity: {pattern.peak_directivity_dbi:.2f} dBi")
    print(f"Peak gain:        {pattern.peak_gain_dbi:.2f} dBi")

    # Sanity bounds — should match the existing patch antenna case
    if not (0.3 < s11 < 0.6):
        print(f"FAIL: |S11|={s11:.4f} outside expected range [0.3, 0.6]")
        return 1
    if not (0.5 < pattern.peak_directivity_dbi < 8.0):
        print(f"FAIL: peak D={pattern.peak_directivity_dbi:.2f} dBi outside [0.5, 8]")
        return 1
    print("OK — patch antenna via builder matches existing TOML flow within bounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
