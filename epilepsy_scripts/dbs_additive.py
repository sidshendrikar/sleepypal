"""
dbs_additive.py — mechanism-robustness check.

Our main plant uses a control-theoretic abstraction: stimulation raises an
effective inhibitory gain, B_eff = B_disease + gain*effect. A literature survey
found this is NOT how neural mass models inject DBS: the field standard
(Mina et al. 2013, Front. Comput. Neurosci.; Muller & Robinson 2018) is an
ADDITIVE input -- a depolarizing pulse train added to a population, with the
suppression emerging from the dynamics rather than from a parameter shift.

This module re-implements the plant that way: DBS is delivered as an ADDITIVE
drive to the inhibitory interneuron population (a physiologically standard model
of stimulation recruiting inhibitory elements), and suppression must *emerge*.
We then ask the only question that matters: does the bifurcation-aware, twin-
dosed controller still beat the deployed-style threshold controller and still
save energy versus continuous stimulation -- or was the result an artifact of
the gain-shift abstraction?

The controller is UNCHANGED. Its model-based dose u* was derived for the gain
mechanism and is therefore *wrong* here; only its ratchet (empirical
minimum-seeking) can save it. That is exactly the robustness test.
"""
import json, os
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from numba import njit

from config import ARTIFACT_DIR, JRParams, DT
from jansen_ritt import _derivs, warmup
from bifurcation_analysis import find_bifurcations
from dbs_sim import SeizureScenario, sense, score
from dbs_stimulator import Stimulator, StimSpec
from dbs_controllers import (NoStim, OpenLoopDBS, ThresholdADBS, BifurcationMPC,
                             calibrate_threshold)

GAIN = 22.0            # only used by the controller's (now-mismatched) dose model


@njit(cache=True, fastmath=True)
def _seg_additive(prm, p_mean, p_std, dt, n, y, drive, g_stim):
    """Integrate with an ADDITIVE inhibitory drive: the stimulation adds an
    extra source term to the inhibitory PSP equation (d[5]) rather than scaling
    the inhibitory gain. `drive` in [0,1], `g_stim` its amplitude (mV)."""
    out = np.empty(n)
    b = prm[3]
    for k in range(n):
        p = p_mean + p_std * np.random.randn() if p_std > 0.0 else p_mean
        d = _derivs(y, p, prm)
        d[5] += g_stim * b * drive           # additive inhibitory input
        for i in range(6):
            y[i] += dt * d[i]
        out[k] = y[1] - y[2]
    return out


class AdditiveLoop:
    def __init__(self, g_stim=60.0, params=None, spec=None,
                 dt_ctrl=0.10, sense_window_s=1.0, settle_s=4.0):
        self.g_stim = g_stim
        self.params = params or JRParams()
        self.spec = spec or StimSpec()
        self.dt_ctrl = dt_ctrl
        self.sense_window_s = sense_window_s
        self.settle_s = settle_s
        self.fs = int(round(1.0 / DT))

    def run(self, controller, scen, seed=0):
        prm = self.params.as_array()
        p_mean, p_std = self.params.p_mean, self.params.p_std
        n_seg = int(round(self.dt_ctrl / DT))
        n_steps = int(round(scen.T_s / self.dt_ctrl))
        win = int(round(self.sense_window_s * self.fs))
        stim = Stimulator(self.spec)
        controller.reset()
        np.random.seed(seed)
        y = np.random.randn(6) * np.array([1., 1., 1., 5., 5., 5.])
        prm[1] = scen.B_disease(0.0)
        if self.settle_s > 0:
            _seg_additive(prm, p_mean, p_std, DT,
                          int(round(self.settle_s / DT)), y, 0.0, self.g_stim)
        V_all = np.empty(n_steps * n_seg)
        log = {k: np.zeros(n_steps) for k in
               ("t", "u", "band_power", "ptp", "line_length", "energy_J")}
        u = 0.0
        for i in range(n_steps):
            t = i * self.dt_ctrl
            prm[1] = scen.B_disease(t)                 # B is NOT elevated here
            drive = stim.effect                        # additive drive instead
            seg = _seg_additive(prm, p_mean, p_std, DT, n_seg, y, drive, self.g_stim)
            V_all[i * n_seg:(i + 1) * n_seg] = seg
            lo = max(0, (i + 1) * n_seg - win)
            feat = sense(V_all[lo:(i + 1) * n_seg], self.fs)
            u = controller.update(feat, t, self.dt_ctrl)
            stim.step(u, self.dt_ctrl)
            log["t"][i] = t; log["u"][i] = u
            log["band_power"][i] = feat["band_power"]
            log["ptp"][i] = feat["ptp"]; log["line_length"][i] = feat["line_length"]
            log["energy_J"][i] = stim.energy_J
        log["V"] = V_all; log["report"] = stim.report()
        log["controller"] = controller.name; log["scenario"] = scen
        return log


def _thresholds(loop):
    quiet = SeizureScenario(onset_s=1e6, offset_s=1e6, T_s=20.0)
    bp = np.concatenate([loop.run(NoStim(), quiet, seed=s)["band_power"]
                         for s in (11, 12, 13)])
    return calibrate_threshold(bp, 99.5), calibrate_threshold(bp, 99.9)


def main():
    warmup()
    Bc = find_bifurcations()["B_crit"]
    scen = SeizureScenario(onset_s=60.0, offset_s=95.0, T_s=600.0)
    B_hat = scen.B_ictal

    # 1. does additive HFS-to-inhibition even suppress? find the drive that does
    print("Does additive inhibitory drive suppress the seizure?")
    g_ok = None
    for g in [20, 40, 60, 80, 100]:
        loop = AdditiveLoop(g_stim=g)
        _, sco = _thresholds(loop)
        rows = [score(loop.run(OpenLoopDBS(1.0), scen, seed=s), seizure_thresh=sco)
                for s in (1, 2, 3)]
        b = float(np.mean([r["seizure_burden_s"] for r in rows]))
        print(f"  g_stim={g:3d}mV  full-drive burden={b:4.1f}s", flush=True)
        if g_ok is None and b < 3.0:
            g_ok = g
    if g_ok is None:
        g_ok = 100
        print("  (additive drive suppresses only weakly; using strongest)", flush=True)

    # 2. controller comparison under the additive plant
    loop = AdditiveLoop(g_stim=g_ok)
    det, sco = _thresholds(loop)
    print(f"\nController comparison under ADDITIVE injection (g_stim={g_ok}mV):",
          flush=True)
    arms = [NoStim(), OpenLoopDBS(1.0), ThresholdADBS(det, 1.0, min_burst_s=2.0),
            BifurcationMPC(Bc, GAIN, B_hat, det, margin=1.05, hold_s=2.0)]
    seeds = (1, 2, 3, 4, 5)
    tbl = {}
    for c in arms:
        rows = [score(loop.run(c, scen, seed=s), seizure_thresh=sco) for s in seeds]
        m = lambda k: float(np.mean([r[k] for r in rows]))
        tbl[c.name] = {"burden_s": m("seizure_burden_s"), "energy_mJ": m("energy_mJ"),
                       "duty": m("duty_cycle"), "peak_u": m("peak_u")}
        print(f"  {c.name:34s} burden={tbl[c.name]['burden_s']:4.1f}s "
              f"energy={tbl[c.name]['energy_mJ']:6.2f}mJ "
              f"duty={tbl[c.name]['duty']*100:3.0f}%", flush=True)

    ol = tbl["open-loop DBS"]; mp = tbl["bifurcation MPC (this work)"]
    th = tbl["threshold aDBS (RNS-style)"]
    verdict = {
        "g_stim_mV": g_ok,
        "vs_continuous_energy_pct": 100 * (1 - mp["energy_mJ"] / max(ol["energy_mJ"], 1e-9)),
        "vs_threshold_energy_pct": 100 * (1 - mp["energy_mJ"] / max(th["energy_mJ"], 1e-9)),
        "vs_threshold_burden_delta": th["burden_s"] - mp["burden_s"],
        "table": tbl,
    }
    print(f"\n  vs continuous: {verdict['vs_continuous_energy_pct']:.1f}% less energy")
    print(f"  vs threshold : {verdict['vs_threshold_energy_pct']:+.1f}% energy, "
          f"{verdict['vs_threshold_burden_delta']:+.1f}s burden")
    json.dump(verdict, open(os.path.join(ARTIFACT_DIR, "dbs_additive.json"), "w"),
              indent=2, default=float)
    print("ADDITIVE_DONE", flush=True)


if __name__ == "__main__":
    main()
