"""FD <-> TD cross-validation — cavity resonance from both backends.

Runs the same PEC box cavity through the frequency-domain (Nedelec-FEM)
eigensolver and the time-domain (DGTD) backend, and checks both reproduce
the analytic resonance. A gross error in either backend — a units mismatch,
a flux sign — would show up as a factor-off frequency.

The TD resonance is found the proper matrix-free way: drive a broadband
pulse, record a probe, and read the spectral peak. No dense assembly, so it
scales to real meshes.
"""
import numpy as np
import rapidfem as rf

# %% A cubic PEC cavity, air-filled.
SIDE = 0.030  # 30 mm
C = 299_792_458.0
# Lowest mode (1,1,0):  f = (c/2)·sqrt((1/a)^2 + (1/a)^2)
f_analytic = 0.5 * C * np.sqrt(2.0) / SIDE

g = rf.Geometry(maxh=SIDE / 1.5)
box = g.box(SIDE, SIDE, SIDE, position=(0, 0, 0), material=rf.Air())
rf.PEC(*box.faces.unassigned)
g.mesh()

# %% Frequency-domain backend — Nedelec edge-element eigensolver.
fd_modes = rf.ProblemFD(g).eigenmode(target_frequency=f_analytic, n_modes=6)
fd_f = min(
    m.frequency_hz for m in fd_modes if m.frequency_hz > 0.3 * f_analytic
)

# %% Time-domain backend — drive a broadband pulse, read the spectral peak.
ptd = rf.ProblemTD(g, order=2, flux="upwind")
tau = 1.0 / (np.pi * f_analytic)             # bandwidth covers the mode
pulse = rf.GaussianPulse(t0=4.0 * tau, tau=tau)
dt = 1.0 / (14.0 * f_analytic)               # 14 samples / period
steps = 1400
centre = (SIDE * 0.5, SIDE * 0.5, SIDE * 0.5)
probe = (SIDE * 0.45, SIDE * 0.55, SIDE * 0.5)

td_run = ptd.driven_transient(
    source=(centre, "E", "z"),
    waveform=pulse,
    probes=[(probe, "E", "z")],
    dt=dt,
    steps=steps,
    krylov_dim=16,
)
rf.show(td_run)                              # the probe time signal
resp = td_run.responses
spec = np.abs(np.fft.rfft(resp[0]))
freq = np.fft.rfftfreq(resp[0].size, dt)
band = (freq > 0.3 * f_analytic) & (freq < 3.0 * f_analytic)
td_f = freq[band][np.argmax(spec[band])]

# %% Compare.
print(f"analytic  f(1,1,0) = {f_analytic / 1e9:.4f} GHz")
print(
    f"FD  Nedelec        = {fd_f / 1e9:.4f} GHz"
    f"   err {abs(fd_f - f_analytic) / f_analytic:.2%}"
)
print(
    f"TD  DGTD (pulse)   = {td_f / 1e9:.4f} GHz"
    f"   err {abs(td_f - f_analytic) / f_analytic:.2%}"
)
print(f"FD vs TD           : {abs(fd_f - td_f) / f_analytic:.2%}")
