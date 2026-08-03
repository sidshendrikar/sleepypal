"""
dbs_sim.py — closed-loop adaptive-DBS simulation harness.

Couples three pieces into one loop running at a realistic embedded control
period (default 100 ms):

    disease  ->  B_disease(t) falls into the seizure band at onset
    plant    ->  Jansen-Rit cortical column integrated at 1 ms
    sensing  ->  peak-to-peak, line length, band power on a sliding window
    control  ->  a Controller returns a normalized amplitude u in [0,1]
    actuator ->  Stimulator converts u into charge, energy, and a smoothed
                 tissue effect, which raises effective inhibition

        B_eff(t) = B_disease(t) + gain * stim_effect(t)

Seizure is scored as an oscillation whose windowed peak-to-peak amplitude
exceeds TAU (2 mV) -- the same criterion used to locate the bifurcation.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from scipy import signal as sps

from config import JRParams, DT
from jansen_ritt import _derivs
from bifurcation_analysis import TAU
from dbs_stimulator import Stimulator, StimSpec


# ----------------------------------------------------------------------------
# integration kernel: run one control period, carry the state forward
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _seed_rng(s):
    np.random.seed(s)


@njit(cache=True, fastmath=True)
def _run_segment(prm, p_mean, p_std, dt, n, y):
    """Integrate n Euler-Maruyama steps with prm[1] (=B) held constant.
    Returns the V trace for the segment; `y` is mutated in place."""
    out = np.empty(n)
    for k in range(n):
        p = p_mean + p_std * np.random.randn() if p_std > 0.0 else p_mean
        d = _derivs(y, p, prm)
        for i in range(6):
            y[i] += dt * d[i]
        out[k] = y[1] - y[2]
    return out


# ----------------------------------------------------------------------------
# scenario
# ----------------------------------------------------------------------------
@dataclass
class SeizureScenario:
    """A seizure as a transient collapse of inhibition."""
    B_interictal: float = 46.0   # above B_crit -> quiet fixed point
    B_ictal: float = 28.0        # inside the limit-cycle band -> seizure
    onset_s: float = 10.0
    offset_s: float = 45.0       # when the disease process would end untreated
    T_s: float = 60.0
    ramp_s: float = 1.0          # transition sharpness

    def B_disease(self, t):
        a = 1.0 / max(self.ramp_s, 1e-6)
        rise = 1.0 / (1.0 + np.exp(-a * (t - self.onset_s)))
        fall = 1.0 / (1.0 + np.exp(-a * (t - self.offset_s)))
        depth = rise - fall                      # 0 outside, ~1 inside
        return self.B_interictal + depth * (self.B_ictal - self.B_interictal)


# ----------------------------------------------------------------------------
# sensing
# ----------------------------------------------------------------------------
def sense(window, fs):
    """Features a real device could compute from the recorded trace."""
    if len(window) < 8:
        return {"ptp": 0.0, "line_length": 0.0, "band_power": 0.0}
    w = window - window.mean()
    ptp = float(window.max() - window.min())
    ll = float(np.abs(np.diff(window)).sum() / len(window))
    n = min(len(w), 512)
    f, P = sps.welch(w, fs=fs, nperseg=n)
    m = (f >= 2.0) & (f <= 8.0)
    bp = float(np.trapz(P[m], f[m])) if m.any() else 0.0
    return {"ptp": ptp, "line_length": ll, "band_power": bp}


# ----------------------------------------------------------------------------
# the closed loop
# ----------------------------------------------------------------------------
class ClosedLoopDBS:
    def __init__(self, B_crit, gain=22.0, params: JRParams = None,
                 spec: StimSpec = None, dt_ctrl=0.10, sense_window_s=1.0,
                 settle_s=4.0):
        self.B_crit = B_crit
        self.gain = gain
        self.params = params or JRParams()
        self.spec = spec or StimSpec()
        self.dt_ctrl = dt_ctrl
        self.sense_window_s = sense_window_s
        self.settle_s = settle_s      # discard startup transient before logging
        self.fs = int(round(1.0 / DT))

    def run(self, controller, scenario: SeizureScenario, seed=0,
            preictal_lead_s=0.0, preictal_reliability=1.0):
        """`preictal_lead_s` supplies the controller with an early-warning flag
        that many seconds before electrographic onset, firing with probability
        `preictal_reliability`. This models the observer's PREICTAL class; no
        deployed device has this input, and it is the only way a closed-loop
        arm can act before the seizure exists."""
        prm = self.params.as_array()
        p_mean, p_std = self.params.p_mean, self.params.p_std
        n_seg = int(round(self.dt_ctrl / DT))
        n_steps = int(round(scenario.T_s / self.dt_ctrl))
        win = int(round(self.sense_window_s * self.fs))

        stim = Stimulator(self.spec)
        controller.reset()
        _seed_rng(seed)

        y = np.random.randn(6) * np.array([1., 1., 1., 5., 5., 5.])
        # settle out the initial transient at the pre-onset inhibition level so
        # that interictal statistics are not contaminated by startup ringing
        if self.settle_s > 0:
            prm[1] = scenario.B_disease(0.0)
            _run_segment(prm, p_mean, p_std, DT,
                         int(round(self.settle_s / DT)), y)

        V_all = np.empty(n_steps * n_seg)
        log = {k: np.zeros(n_steps) for k in
               ("t", "B_disease", "B_eff", "u", "effect", "ptp", "line_length",
                "band_power", "energy_J")}

        # one Bernoulli draw per episode: the warning either fires or it does not
        _warned = [np.random.rand() < preictal_reliability]
        u = 0.0
        for i in range(n_steps):
            t = i * self.dt_ctrl
            Bd = scenario.B_disease(t)
            effect = stim.effect
            B_eff = Bd + self.gain * effect
            prm[1] = B_eff

            seg = _run_segment(prm, p_mean, p_std, DT, n_seg, y)
            V_all[i * n_seg:(i + 1) * n_seg] = seg

            lo = max(0, (i + 1) * n_seg - win)
            feat = sense(V_all[lo:(i + 1) * n_seg], self.fs)
            if preictal_lead_s > 0.0:
                in_lead = (scenario.onset_s - preictal_lead_s) <= t < scenario.onset_s
                feat["preictal"] = bool(in_lead and _warned[0])
            u = controller.update(feat, t, self.dt_ctrl)
            stim.step(u, self.dt_ctrl)

            log["t"][i] = t
            log["B_disease"][i] = Bd
            log["B_eff"][i] = B_eff
            log["u"][i] = u
            log["effect"][i] = effect
            log["ptp"][i] = feat["ptp"]
            log["line_length"][i] = feat["line_length"]
            log["band_power"][i] = feat["band_power"]
            log["energy_J"][i] = stim.energy_J

        log["V"] = V_all
        log["report"] = stim.report()
        log["controller"] = controller.name
        log["scenario"] = scenario
        return log


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def score(log, seizure_thresh=None, tau=TAU):
    """Seizure burden, energy, and safety summary for one run.

    Seizure presence is scored from 2-8 Hz band power, not raw peak-to-peak:
    the stochastic cortical drive gives the *suppressed* state a peak-to-peak of
    several mV, so a raw-amplitude criterion would score noise as seizure. Band
    power separates the two states by ~1000x. `seizure_thresh` should be
    calibrated on interictal data (see dbs_experiments.calibrate)."""
    t, u = log["t"], log["u"]
    bp = log["band_power"]
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    sc = log["scenario"]
    ictal_window = (t >= sc.onset_s) & (t <= sc.offset_s)

    thr = seizure_thresh if seizure_thresh is not None else tau
    seizing = bp > thr
    burden_s = float(seizing[ictal_window].sum() * dt)
    total_ictal_s = float(ictal_window.sum() * dt)

    # time from disease onset to first suppression after seizure begins
    lat = np.nan
    idx = np.where(ictal_window & seizing)[0]
    if len(idx):
        first = idx[0]
        after = np.where(~seizing[first:] )[0]
        if len(after):
            lat = float(after[0] * dt)

    rep = log["report"]
    return {
        "controller": log["controller"],
        "seizure_burden_s": burden_s,
        "burden_frac": burden_s / max(total_ictal_s, 1e-9),
        "energy_J": rep["energy_J"],
        "energy_mJ": rep["energy_J"] * 1e3,
        "charge_uC": rep["charge_uC"],
        "duty_cycle": rep["duty_cycle"],
        "mean_power_uW": rep["mean_power_W"] * 1e6,
        "peak_u": float(u.max()),
        "mean_u_on": float(u[u > 0].mean()) if (u > 0).any() else 0.0,
        "suppression_latency_s": lat,
    }
