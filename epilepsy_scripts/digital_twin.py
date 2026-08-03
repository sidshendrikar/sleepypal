"""
digital_twin.py — Personalization + closed loop (Sections 2.6 & 2.7).

Fits the Jansen-Rit inhibitory gain B so that the simulated column's power
spectrum matches a patient's real EEG, then couples the fitted twin with the
bifurcation map and controller to demonstrate closed-loop seizure termination.

Parameter estimation (Section 2.6):
    B* = argmin_B  sum_k [ log(1+P_sim(f_k;B)) - log(1+P_real(f_k)) ]^2
solved with Differential Evolution (Storn-Price), matching the paper.
"""
import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution

from config import JRParams, FS
from jansen_ritt import simulate
from simulator import resample_to_eeg
from bifurcation_analysis import find_bifurcations, distance_to_bifurcation, control_law

FIT_FMIN, FIT_FMAX = 1.0, 45.0     # band used for spectral matching


def _psd(x, fs=FS, nperseg=256):
    f, P = signal.welch(x, fs=fs, window="hann", nperseg=min(nperseg, len(x)),
                        noverlap=min(nperseg, len(x)) // 2)
    m = (f >= FIT_FMIN) & (f <= FIT_FMAX)
    return f[m], P[m]


def _spectral_cost(B, f_real, logP_real, T, seed):
    B = float(np.asarray(B).ravel()[0])
    _, V = simulate(B=B, T=T, method="euler", seed=seed)
    sig = resample_to_eeg(V)
    f_sim, P_sim = _psd(sig)
    # interpolate sim PSD onto the real frequency grid
    P_sim_i = np.interp(f_real, f_sim, P_sim)
    logP_sim = np.log1p(P_sim_i)
    return float(np.sum((logP_sim - logP_real) ** 2))


def fit_inhibitory_gain(real_eeg, fs=FS, B_bounds=(1.0, 55.0), T=8.0,
                        maxiter=12, popsize=12, seed=17):
    """Differential-evolution fit of B to a real EEG segment.
    Returns dict with fitted B, cost, and fit spectra."""
    # normalize amplitude so the match is about spectral *shape* not scale
    x = (real_eeg - np.mean(real_eeg))
    x = x / (np.std(x) + 1e-8)
    f_real, P_real = _psd(x, fs=fs)
    logP_real = np.log1p(P_real)

    result = differential_evolution(
        _spectral_cost, bounds=[B_bounds],
        args=(f_real, logP_real, T, seed),
        maxiter=maxiter, popsize=popsize, tol=0.05,
        seed=seed, polish=True, updating="deferred", workers=1)
    B_star = float(result.x[0])

    _, Vfit = simulate(B=B_star, T=T, method="euler", seed=seed)
    f_sim, P_sim = _psd(resample_to_eeg(Vfit))
    return {"B": B_star, "cost": float(result.fun),
            "f_real": f_real, "P_real": P_real,
            "f_sim": f_sim, "P_sim": P_sim}


class DigitalTwin:
    """Patient-specific twin: fitted B + bifurcation map + controller."""

    def __init__(self, params: JRParams = None):
        self.params = params or JRParams()
        self.B_fit = None
        self.bif = None

    def calibrate(self, real_seizure_eeg, fs=FS, verbose=True):
        self.bif = find_bifurcations(params=self.params)
        fit = fit_inhibitory_gain(real_seizure_eeg, fs=fs)
        self.B_fit = fit["B"]
        self.fit = fit
        if verbose:
            print(f"  fitted B* = {self.B_fit:.2f}  (cost {fit['cost']:.3f})")
            print(f"  B_crit    = {self.bif['B_crit']:.2f}, "
                  f"distance to bifurcation = "
                  f"{distance_to_bifurcation(self.B_fit, self.bif['B_crit']):.2f}")
        return self

    def controller(self, mode="graded"):
        return control_law(self.bif["B_crit"], mode=mode)

    def recommend(self, seizure_prob):
        """Recommended inhibitory-gain target given the detector's probability."""
        ctrl = self.controller()
        return ctrl(seizure_prob, self.B_fit)


if __name__ == "__main__":
    from jansen_ritt import warmup
    warmup()
    # fit the twin to a synthetic "seizure" (large limit cycle at B=34)
    sig = resample_to_eeg(simulate(B=34.0, T=10.0, method="euler", seed=5)[1])
    twin = DigitalTwin().calibrate(sig)
    print("Recommended target at P(seizure)=0.95:",
          round(twin.recommend(0.95), 2))
