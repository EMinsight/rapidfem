"""
EMerge ↔ rapidfem S-parameter comparison utility.

Loads two S-parameter datasets and reports per-frequency and aggregate
absolute and relative differences. Used by validation test drivers.
"""
from __future__ import annotations
import numpy as np
import re


def load_touchstone(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a Touchstone (.s1p, .s2p, .snp) file. Returns (freqs, s) where s has shape (n_freq, n_port, n_port)."""
    with open(path) as f:
        lines = f.readlines()

    freq_unit = 1e9  # default GHZ
    fmt = "RI"
    z0 = 50.0

    data_lines = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("!"):
            continue
        if s.startswith("#"):
            tok = s.split()
            for i, t in enumerate(tok):
                tu = t.upper()
                if tu in ("HZ", "KHZ", "MHZ", "GHZ"):
                    freq_unit = {"HZ": 1, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}[tu]
                elif tu in ("RI", "MA", "DB"):
                    fmt = tu
                elif tu == "R" and i + 1 < len(tok):
                    z0 = float(tok[i + 1])
            continue
        data_lines.append(s)

    m = re.match(r".*\.s(\d+)p$", path, re.I)
    nport = int(m.group(1)) if m else 1

    rows = [list(map(float, ln.split())) for ln in data_lines]
    freqs = np.array([r[0] * freq_unit for r in rows])
    n = len(freqs)
    s = np.zeros((n, nport, nport), dtype=complex)
    for k, row in enumerate(rows):
        vals = row[1:]
        # Touchstone S2P+ ordering: row-major over (i,j)
        idx = 0
        for i in range(nport):
            for j in range(nport):
                if fmt == "RI":
                    re_v = vals[idx]; im_v = vals[idx + 1]; idx += 2
                    s[k, i, j] = complex(re_v, im_v)
                elif fmt == "MA":
                    mag = vals[idx]; ph = np.deg2rad(vals[idx + 1]); idx += 2
                    s[k, i, j] = mag * np.exp(1j * ph)
                elif fmt == "DB":
                    mag = 10 ** (vals[idx] / 20.0); ph = np.deg2rad(vals[idx + 1]); idx += 2
                    s[k, i, j] = mag * np.exp(1j * ph)
    return freqs, s


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV produced by run_emerge_*.py. Format:
        freq_hz, S_re_11, S_im_11, S_re_12, S_im_12, ..., S_re_nn, S_im_nn
    """
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    freqs = arr[:, 0]
    cols = arr.shape[1] - 1
    nport = int(np.sqrt(cols // 2))
    n = arr.shape[0]
    s = np.zeros((n, nport, nport), dtype=complex)
    idx = 1
    for i in range(nport):
        for j in range(nport):
            s[:, i, j] = arr[:, idx] + 1j * arr[:, idx + 1]
            idx += 2
    return freqs, s


def save_csv(path: str, freqs: np.ndarray, s: np.ndarray) -> None:
    n_freq, np_, _ = s.shape
    cols = ["freq_hz"]
    for i in range(np_):
        for j in range(np_):
            cols.append(f"S_re_{i+1}{j+1}")
            cols.append(f"S_im_{i+1}{j+1}")
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for k in range(n_freq):
            row = [f"{freqs[k]:.6e}"]
            for i in range(np_):
                for j in range(np_):
                    v = s[k, i, j]
                    row.append(f"{v.real:.6e}")
                    row.append(f"{v.imag:.6e}")
            f.write(",".join(row) + "\n")


def compare(label_a: str, freqs_a: np.ndarray, s_a: np.ndarray,
            label_b: str, freqs_b: np.ndarray, s_b: np.ndarray,
            tol_abs: float = 0.05, tol_rel: float = 0.10) -> int:
    """Compare two S-param datasets. Returns 0 on pass, 1 on fail."""
    if not np.allclose(freqs_a, freqs_b, rtol=1e-6):
        # Interpolate b onto a's grid
        s_b_interp = np.zeros_like(s_a)
        for i in range(s_a.shape[1]):
            for j in range(s_a.shape[2]):
                re_b = np.interp(freqs_a, freqs_b, s_b[:, i, j].real)
                im_b = np.interp(freqs_a, freqs_b, s_b[:, i, j].imag)
                s_b_interp[:, i, j] = re_b + 1j * im_b
        s_b = s_b_interp
        freqs_b = freqs_a

    diff = np.abs(s_a - s_b)
    mag_a = np.abs(s_a)
    rel = np.where(mag_a > 1e-9, diff / mag_a, 0.0)

    print(f"\n=== {label_a} vs {label_b} ===")
    print(f"{'freq[GHz]':>10} {'|S_a|':>10} {'|S_b|':>10} {'|diff|':>10} {'rel%':>8}  status")
    fail = 0
    for k, f in enumerate(freqs_a):
        for i in range(s_a.shape[1]):
            for j in range(s_a.shape[2]):
                d = diff[k, i, j]
                r = rel[k, i, j]
                ok = (d <= tol_abs) or (r <= tol_rel)
                if not ok:
                    fail += 1
                tag = f"S{i+1}{j+1}"
                marker = "OK" if ok else "FAIL"
                if s_a.shape[1] == 1:
                    print(f"{f/1e9:10.4f} {mag_a[k,i,j]:10.4f} {abs(s_b[k,i,j]):10.4f} {d:10.5f} {r*100:7.2f}%  {marker}")
                else:
                    print(f"{f/1e9:10.4f} {tag} {mag_a[k,i,j]:8.4f} {abs(s_b[k,i,j]):8.4f} {d:8.5f} {r*100:6.2f}% {marker}")
    print(f"\nMax |diff|: {diff.max():.5f}, Max rel: {rel.max()*100:.2f}%, Fails: {fail}")
    print(f"Tolerances: |diff| < {tol_abs}, rel < {tol_rel*100:.0f}%")
    return 0 if fail == 0 else 1
