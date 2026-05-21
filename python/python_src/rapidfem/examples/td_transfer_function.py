"""Time-domain DGTD: the cavity transfer function by on-the-fly RFT.

One broadband-driven transient run yields the cavity's frequency
response: `transfer_function` divides the probe spectrum by the source
spectrum, H(f) = R(f)/G(f). Peaks of |H(f)| are the resonances,
recovered here from a single time-domain run and checked against the
analytic rectangular-cavity modes.

This is the scalar, on-the-fly-RFT observable; true modal-port
S-parameters need waveguide-mode injection / extraction.
"""

# %% Build a PEC air-cube cavity through the geometry API
import numpy as np

import rapidfem as rf

mm = 1e-3
L = 40.0 * mm                        # cubic cavity edge

g = rf.Geometry(maxh=L / 7)
air = g.box(L, L, L, material=rf.Air())
rf.PEC(*air.faces.unassigned)        # closed cavity, six PEC walls
g.mesh()
rf.show(g)

ptd = rf.ProblemTD(g, order=2, flux="upwind")
print(f"DGTD cavity - {ptd.n_dof // 60} tets, {ptd.n_dof} state DOFs")

# %% Drive a broadband pulse, record the probe signal in time
pulse = rf.GaussianPulse(t0=160e-12, tau=40e-12, f0=8e9)
source = ((10 * mm, 10 * mm, 10 * mm), "E", "z")
probe = ((27 * mm, 31 * mm, 18 * mm), "E", "z")
dt, steps = 8e-12, 1000

response = ptd.driven_transient(
    source=source, waveform=pulse, probes=[probe], dt=dt, steps=steps,
)
rf.show(response)                            # the probe signal in time

# %% The same run, deconvolved into the cavity transfer function H(f)
tf = ptd.transfer_function(
    source=source, probe=probe, pulse=pulse, dt=dt, steps=steps,
)
rf.show(tf)                                  # |H(f)| magnitude / phase
freqs, H = tf

# %% Peaks of |H| are the cavity resonances
mag = np.abs(H)
band = (freqs > 3e9) & (freqs < 13e9)
fb, mb = freqs[band], mag[band]
peaks = [
    fb[i]
    for i in range(1, len(fb) - 1)
    if mb[i] > mb[i - 1] and mb[i] > mb[i + 1] and mb[i] > 0.1 * mb.max()
]
print("transfer-function peaks [GHz]:")
print("  " + ", ".join(f"{f / 1e9:.2f}" for f in peaks))

# %% Analytic rectangular-cavity resonances  f = c/(2L)*sqrt(m^2+n^2+p^2)
C = 299_792_458.0
analytic = sorted({
    C / (2 * L) * np.sqrt(m * m + n * n + q * q)
    for m in range(4)
    for n in range(4)
    for q in range(4)
    if 0 < m * m + n * n + q * q <= 9
    and (m > 0) + (n > 0) + (q > 0) >= 2
})
print("analytic cavity modes [GHz]:")
print("  " + ", ".join(f"{f / 1e9:.2f}" for f in analytic[:6]))
print("the time-domain run recovers the cavity spectrum from one transient")
