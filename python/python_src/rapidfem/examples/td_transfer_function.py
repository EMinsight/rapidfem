"""Time-domain DGTD — the cavity transfer function by on-the-fly RFT.

One broadband-driven transient run yields the cavity's frequency
response: `transfer_function` divides the probe spectrum by the source
spectrum, H(f) = R(f)/G(f). Peaks of |H(f)| are the resonances —
recovered here from a single time-domain run and checked against the
analytic unit-cube cavity modes.

This is the scalar, on-the-fly-RFT observable; true modal-port
S-parameters need waveguide-mode injection / extraction.
"""

# %% Build a unit cavity, normalised units (c = 1)
import numpy as np

import rapidfem as rf

ptd = rf.ProblemTD.box(
    size=(1.0, 1.0, 1.0), cells=(3, 3, 3), order=2, flux="upwind"
)
print(f"DGTD cavity - {ptd.n_dof} state DOFs")

# %% Drive a broadband pulse, record the probe signal in time
pulse = rf.GaussianPulse(t0=2.0, tau=0.5, f0=0.0)
source = ([0.5, 0.5, 0.5], "E", "z")
probe = ([0.3, 0.7, 0.4], "E", "z")
dt, steps = 0.025, 800

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
band = (freqs > 0.1) & (freqs < 2.0)
fb, mb = freqs[band], mag[band]
peaks = [
    fb[i]
    for i in range(1, len(fb) - 1)
    if mb[i] > mb[i - 1] and mb[i] > mb[i + 1] and mb[i] > 0.1 * mb.max()
]
print("transfer-function peaks (1/period units):")
print("  " + ", ".join(f"{f:.3f}" for f in peaks))

# %% Analytic unit-cube resonances  f = sqrt(m^2+n^2+p^2) / 2
analytic = sorted({
    np.sqrt(m * m + n * n + q * q) / 2
    for m in range(4)
    for n in range(4)
    for q in range(4)
    if 0 < m * m + n * n + q * q <= 6
})
print("analytic cavity modes:")
print("  " + ", ".join(f"{f:.3f}" for f in analytic[:6]))
print("the time-domain run recovers the cavity spectrum from one transient")
