"""
simulator.py — High-level interface to the Jansen-Rit column.

Wraps the JIT kernels in jansen_ritt.py with the measurements the rest of the
framework needs:

  * psd_welch          — Welch power spectral density (used for twin fitting)
  * spectral_features  — band powers + spectral entropy of a segment
  * oscillation_amplitude — steady-state peak-to-peak (used by bifurcation)
  * resample_to_eeg    — decimate the 1 kHz simulation to the 256 Hz EEG rate
  * synthesize_dataset — generate labeled synthetic EEG across B regimes, used
                         to augment the rare ictal class (a mechanistic fix for
                         the paper's 12.5% sensitivity problem)
"""
import numpy as np
from scipy import signal

from config import JRParams, DT, FS_SIM, FS
from jansen_ritt import simulate


def psd_welch(V, fs=FS_SIM, nperseg=None, fmax=70.0):
    """Welch PSD restricted to [0, fmax]. Returns (freqs, power)."""
    if nperseg is None:
        nperseg = min(len(V), int(fs))       # ~1 s segments, Hann window
    f, Pxx = signal.welch(V, fs=fs, window="hann", nperseg=nperseg,
                          noverlap=nperseg // 2, detrend="constant")
    m = f <= fmax
    return f[m], Pxx[m]


def oscillation_amplitude(V, frac_transient=0.25):
    """Steady-state peak-to-peak amplitude, ignoring the first `frac` of V."""
    k = int(len(V) * frac_transient)
    seg = V[k:]
    return float(seg.max() - seg.min())


def dominant_frequency(V, fs=FS_SIM):
    f, P = psd_welch(V, fs=fs)
    if len(f) < 2:
        return 0.0
    return float(f[1:][np.argmax(P[1:])])   # skip DC


_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
          "beta": (13, 30), "gamma": (30, 70)}


def spectral_features(V, fs=FS_SIM):
    """Relative band powers + spectral entropy + dominant freq for a segment."""
    f, P = psd_welch(V, fs=fs)
    total = np.trapz(P, f) + 1e-12
    feats = {}
    for name, (lo, hi) in _BANDS.items():
        m = (f >= lo) & (f < hi)
        feats[name] = float(np.trapz(P[m], f[m]) / total)
    p = P / P.sum()
    feats["spec_entropy"] = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))
    feats["dom_freq"] = dominant_frequency(V, fs)
    feats["ptp"] = oscillation_amplitude(V)
    return feats


def resample_to_eeg(V, fs_sim=FS_SIM, fs_eeg=FS):
    """Anti-alias decimate the simulated signal to the EEG sampling rate."""
    n_out = int(round(len(V) * fs_eeg / fs_sim))
    return signal.resample(V, n_out)


def simulate_eeg(params=None, B=None, T=8.0, seed=-1, method="euler"):
    """Simulate and return the signal already decimated to FS (256 Hz)."""
    _, V = simulate(params=params, B=B, T=T, seed=seed, method=method)
    return resample_to_eeg(V)


# ----------------------------------------------------------------------------
# Synthetic labeled dataset from the mechanistic model
# ----------------------------------------------------------------------------
def synthesize_dataset(n_per_class=200, T=5.0, seed=0,
                       B_ictal=(28.0, 41.0), B_interictal=(16.0, 23.0),
                       B_preictal=(24.0, 28.0)):
    """Generate labeled (V_eeg, label) epochs by sampling the JR model across
    inhibitory-gain regimes.

    The measured bifurcation structure (see bifurcation_analysis.py) is:
      B in [~20, ~42]  -> large 3-5 Hz limit cycle  (ictal, label 1)
      B < ~20          -> low-amplitude alpha        (interictal, label 0)
      B just below B_on -> rising-amplitude transition (preictal, label 2)
    Ranges default to those regimes but can be overridden with the boundaries
    returned by bifurcation_analysis.find_bifurcations().

    This is an in-silico phantom used purely to *augment* the rare seizure
    class during observer training; real CHB-MIT epochs remain the test set.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    specs = [(0, B_interictal), (1, B_ictal), (2, B_preictal)]
    for label, (blo, bhi) in specs:
        for i in range(n_per_class):
            B = float(rng.uniform(blo, bhi))
            sig = simulate_eeg(B=B, T=T + 2.0, seed=int(rng.integers(1e9)))
            sig = sig[-int(T * FS):]                 # trim to exactly T seconds
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            X.append(sig.astype("float32"))
            y.append(label)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    from jansen_ritt import warmup
    warmup()
    print("Regime sweep (peak-to-peak amplitude & dominant freq):")
    for B in [4, 12, 22, 30, 42, 50]:
        _, V = simulate(B=B, T=8.0, method="rk4")     # deterministic
        print(f"  B={B:5.1f}  ptp={oscillation_amplitude(V):6.2f} mV  "
              f"f_dom={dominant_frequency(V):5.2f} Hz")
