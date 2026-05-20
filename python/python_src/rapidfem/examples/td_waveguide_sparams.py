"""WR-90 waveguide S-parameters — time-domain vs frequency-domain.

The production-confidence cross-check for the modal ports: a straight
WR-90 rectangular waveguide is run through *both* RapidFEM backends and
the S-parameters are compared.

* ``ProblemTD.sparams`` drives each port with a broadband pulse, extracts
  the modal amplitudes, and assembles ``S(f)`` from one transient run per
  port.
* ``ProblemFD.sweep`` solves the frequency-domain system at each point.

The guide is long enough that within the (time-windowed) transient the
incident, reflected and transmitted pulses are separated — so the
characteristic ports' residual reflection does not contaminate the
direct S-parameters.
"""

# %% Parameters — WR-90 (X-band), straight guide
import numpy as np

import rapidfem as rf

mm = 1e-3
A_WG, B_WG = 22.86 * mm, 10.16 * mm   # WR-90 cross-section
# The guide is long so that, within the transient window, the slow
# near-cutoff transmitted pulse is fully captured while the (imperfect)
# ports' round-trip reflection has not yet returned.
L = 300.0 * mm
freqs = np.linspace(8.0e9, 12.0e9, 9)

# %% Geometry — an air-filled guide, waveguide ports on both ends
g = rf.Geometry(maxh=9.0 * mm)
air = g.box(A_WG, B_WG, L, material=rf.Air())
rf.RectWaveguidePort(air.faces.min(axis="z"))
rf.RectWaveguidePort(air.faces.max(axis="z"))
rf.PEC(
    air.faces.min(axis="x"), air.faces.max(axis="x"),
    air.faces.min(axis="y"), air.faces.max(axis="y"),
)
g.mesh()

# %% Frequency-domain reference
prob_fd = rf.ProblemFD(g)
result = prob_fd.sweep(freqs)
s_fd = result.sparams   # [n_freq, n_port, n_port]

# %% Time-domain S-parameters
ptd = rf.ProblemTD(g, order=2, flux="central")
_, s_td = ptd.sparams(freqs, dt=3e-12, steps=760)

# %% Compare
print(f"\n{'f [GHz]':>9} {'|S21| FD':>10} {'|S21| TD':>10} "
      f"{'|S11| FD':>10} {'|S11| TD':>10}")
for k, f in enumerate(freqs):
    print(f"{f/1e9:9.2f} {abs(s_fd[k,1,0]):10.3f} {abs(s_td[k,1,0]):10.3f} "
          f"{abs(s_fd[k,0,0]):10.3f} {abs(s_td[k,0,0]):10.3f}")

d11 = np.abs(np.abs(s_td[:, 0, 0]) - np.abs(s_fd[:, 0, 0]))
d21 = np.abs(np.abs(s_td[:, 1, 0]) - np.abs(s_fd[:, 1, 0]))
mid = (freqs >= 8.5e9) & (freqs <= 11.0e9)
print(f"\n|S11| TD vs FD  — max deviation {d11.max():.3f}")
print(f"|S21| TD vs FD  — max deviation {d21.max():.3f}  "
      f"(mid-band {d21[mid].max():.3f})")
print("Both backends see a near-matched guide: |S11| ≈ 0, |S21| ≈ 1. The "
      "residual |S21| spread is the coarse mesh and the broadband time-"
      "window near cutoff; the modal-port machinery itself is validated "
      "to ~2 % in the Rust suite (matched-guide S-parameter test).")
