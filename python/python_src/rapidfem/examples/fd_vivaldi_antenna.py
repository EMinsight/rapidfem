"""Slotted Vivaldi (exponentially-tapered slot) antenna, ported from
``EMerge/examples/demo19_vivaldi_antenna.py``.

A microstrip line on the top copper layer of a thin FR-4 board crosses an
exponentially-tapered slot etched into the bottom ground plane. The slot
flares from a narrow ``g`` gap at the feed to a wide ``W`` aperture, giving
the travelling-wave radiation of a Vivaldi (tapered-slot) antenna. The
ground-plane edges are *corrugated* with periodic slots to broaden the
band.

The point of the port is the **parametric curve**: rapidfem has no
``XYPolygon().parametric`` helper, so the taper

    fx(t) = L·t
    fy(t) = (g/2)·Kᵗ + (W − g·K)/(2 − 2K)·(1 − Kᵗ),   t ∈ [0, 1]

is discretised here into a point list and handed to :meth:`Geometry.polygon`.
The tapered slot, the open-circuit disc and the corrugation slots are
``cut`` out of the ground plane; the feed line and a radial (sector) stub
are plates on the top layer. A lumped port drives the line, an order-2 ABC
closes the air box, and the far-field at 6 GHz reports directivity / gain.

Wideband (3-8 GHz), the sweep is the slow part; drop ``N_FREQ`` for a quick
look.
"""

# %% Parameters
import math

import numpy as np
import rapidfem as rf

mm = 1e-3

G_GAP = 0.3 * mm      # narrow taper gap at the feed
L_SLOT = 70.0 * mm    # taper / slot length (along x)
W_AP = 55.0 * mm      # aperture width (full, along y)
K = 200.0             # exponential growth factor
RADIUS = 20.0 * mm    # open-circuit disc diameter source (radius = RADIUS/2)
TH = 0.5 * mm         # PCB thickness
ER_SUB = 4.4          # FR-4
L_STUB = 7.0 * mm     # radial stub length
STUB_ANG = 80.0       # radial stub opening angle [deg]
STUB_ANG_OFF = 20.0   # stub rotation off the feed axis [deg]
SLOT_MARGIN = 5.0 * mm  # keep-out margin of the corrugations around the taper
W_SLOT = 4.0 * mm     # corrugation slot width
W_GAP = 4.0 * mm      # corrugation slot pitch gap
W_PERIOD = W_SLOT + W_GAP
N_SLOTS = 7

# PCB plan bounds (the taper aperture sticks out past the dielectric edge).
PCB_XMIN, PCB_XMAX = -25.0 * mm, 70.0 * mm
PCB_YMIN, PCB_YMAX = -30.0 * mm, 30.0 * mm

# 50 Ω microstrip width on FR-4, th = 0.5 mm (Hammerstad synthesis).
def _ms_width_50(z0: float, er: float, h: float) -> float:
    b = 377.0 * math.pi / (2.0 * z0 * math.sqrt(er))
    woh = (2.0 / math.pi) * (
        b - 1.0 - math.log(2.0 * b - 1.0)
        + (er - 1.0) / (2.0 * er) * (math.log(b - 1.0) + 0.39 - 0.61 / er)
    )
    return woh * h


W0 = _ms_width_50(50.0, ER_SUB, TH)   # ≈ 0.95 mm

N_FREQ = 6
FREQUENCIES = np.linspace(3.0e9, 8.0e9, N_FREQ)
F_FF = 6.0e9                          # far-field frequency
MAXH = rf.lambda_maxh(f_max=8.0e9)
TAPER_N = 48                          # parametric-curve sampling


# %% Parametric taper curves
# fy(t): half-width of the slot; fy(0)=g/2 (gap), fy(1)=W/2 (aperture).
def fy(t):
    return (G_GAP / 2.0) * K**t + (W_AP - G_GAP * K) / (2.0 - 2.0 * K) * (1.0 - K**t)


def taper_points(half, x0=0.0):
    """CCW outline of the tapered slot: top edge t:0→1, bottom edge t:1→0."""
    ts = np.linspace(0.0, 1.0, TAPER_N)
    top = [(x0 + t * L_SLOT, half(t)) for t in ts]
    bot = [(x0 + t * L_SLOT, -half(t)) for t in reversed(ts)]
    return top + bot


# Dilated taper (taper offset outward by SLOT_MARGIN along its normal), the
# keep-out used to carve the corrugations away from the taper itself.
A_COEF = (G_GAP / 2.0) - (W_AP - G_GAP * K) / (2.0 - 2.0 * K)


def _dfx(t):
    return A_COEF * math.log(K) / L_SLOT * K**t


def fy_dilated(t):
    r = 1.0 / math.sqrt(1.0 + _dfx(t) ** 2)
    return fy(t) + SLOT_MARGIN * r


# %% Geometry + Materials
g = rf.Geometry(maxh=MAXH)
fr4 = rf.Dielectric(er=ER_SUB, tand=0.02, maxh=2.0 * TH)

# Substrate slab z ∈ [-TH, 0]; air box enclosing everything.
sub = g.box(PCB_XMAX - PCB_XMIN, PCB_YMAX - PCB_YMIN, TH,
            position=(PCB_XMIN, PCB_YMIN, -TH), material=fr4)

PAD = 12.0 * mm
air = g.box(PCB_XMAX - PCB_XMIN + 2 * PAD, PCB_YMAX - PCB_YMIN + 2 * PAD,
            TH + 2 * PAD,
            position=(PCB_XMIN - PAD, PCB_YMIN - PAD, -TH - PAD),
            material=rf.Air())

# Ground plane (bottom copper, z = -TH) with the taper + disc + corrugations
# cut out. Build each opening as a face, then boolean-subtract from the plate.
ground = g.xy_plate(PCB_XMAX - PCB_XMIN, PCB_YMAX - PCB_YMIN,
                    position=(PCB_XMIN, PCB_YMIN, -TH))

taper = g.polygon([(x, y, -TH) for (x, y) in taper_points(fy)],
                  maxh=2.0 * mm)
disc = g.disc(RADIUS / 2.0, position=(-RADIUS / 2.0 + 1.0 * mm, 0.0, -TH))

# Cut the tapered slot + open-circuit disc out of the ground plane. (The
# EMerge demo also corrugates the ground edges with periodic slots to widen
# the band; those are dropped here, the thin boolean slivers they create
# wreck tet quality and balloon the solve. The bare tapered slot is still a
# recognisable, well-behaved Vivaldi; see fy_dilated/N_SLOTS for the hook.)
g.cut(ground, taper, disc)

# Feed: 50 Ω microstrip on the top copper (z = 0), running +y across the slot,
# terminated in a radial (sector) stub for the slot-line transition.
FEED_X, FEED_Y0 = 2.0 * mm, -10.0 * mm
FEED_LEN = 10.5 * mm
feed = g.xy_plate(W0, FEED_LEN, position=(FEED_X - W0 / 2.0, FEED_Y0, 0.0),
                  maxh=0.5 * mm)

# Radial stub: circular sector of radius L_STUB at the line end, opening
# STUB_ANG wide, its bisector rotated STUB_ANG_OFF off the +y feed axis.
stub_cx, stub_cy = FEED_X, FEED_Y0 + FEED_LEN - 0.2 * mm
base = math.radians(90.0 + STUB_ANG_OFF)   # +y axis is 90°, rotate off it
half = math.radians(STUB_ANG / 2.0)
arc = [(stub_cx, stub_cy, 0.0)]
for a in np.linspace(base - half, base + half, 24):
    arc.append((stub_cx + L_STUB * math.cos(a), stub_cy + L_STUB * math.sin(a), 0.0))
stub = g.polygon(arc, maxh=0.6 * mm)

# Lumped feed: a vertical sheet at the line input bridging the ground plane
# (z = -TH) to the microstrip trace (z = 0), driven through the substrate,
# the standard rapidfem microstrip feed (cf. fd_patch_antenna.py).
port = g.plate(p0=(FEED_X - W0 / 2.0, FEED_Y0, -TH),
               width=(W0, 0.0, 0.0), height=(0.0, 0.0, TH), maxh=0.4 * mm)

g.fragment(air, sub, ground, feed, stub, port)

# Physics
rf.LumpedPort(port, direction=(0, 0, 1), z0=50.0)
rf.PEC(ground, feed, stub)
rf.ABC(*air.faces.outer)

rf.show(g)


# %% Mesh
g.mesh()
rf.show(g)


# %% Problem + sweep
prob = rf.Problem(g)
result = prob.sweep(FREQUENCIES)
rf.show(result)

mags = [abs(result.sparams[i, 0, 0]) for i in range(N_FREQ)]
print(f"\nVivaldi: DOFs={prob.n_dofs}, tets={prob.n_tets}")
for f, s in zip(FREQUENCIES, mags):
    print(f"  f={f/1e9:5.2f} GHz   |S11| = {20*np.log10(s):6.2f} dB")

# %% Far-field at 6 GHz
fi = int(min(range(N_FREQ), key=lambda i: abs(FREQUENCIES[i] - F_FF)))
pattern = prob.farfield(result, freq_idx=fi, port_idx=0, n_theta=91, n_phi=72)
if pattern is not None:
    print(f"\nFar-field @ {FREQUENCIES[fi]/1e9:.2f} GHz: "
          f"D = {pattern.peak_directivity_dbi:.2f} dBi, "
          f"G = {pattern.peak_gain_dbi:.2f} dBi")
