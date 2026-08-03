"""
bifurcation_analysis.py — The "Control" engine.

Maps how the Jansen-Rit column changes qualitative behavior as the inhibitory
gain B is swept, locates the Hopf bifurcation(s), and turns the distance to the
bifurcation into a therapeutic control signal (Sections 2.7 of the paper).

Key quantities
--------------
  amplitude_curve(B)   peak-to-peak of steady-state V for each B
  find_bifurcations()  (B_on, B_crit): limit-cycle onset and annihilation
  distance_to_bifurcation(B) : signed  B_crit - B   (>0 => still ictal)
  control_law(...)     bang-bang (paper) or graded proportional (improvement)
  closed_loop(...)     simulate a seizure and the controller stopping it
"""
import numpy as np

from config import JRParams, DT, TRANSIENT_S
from jansen_ritt import simulate
from simulator import oscillation_amplitude, dominant_frequency

TAU = 2.0   # seizure-amplitude threshold (mV), paper Section 2.7.2


def amplitude_curve(B_values, params=None, T=6.0, deterministic=True, seed=3):
    """For each B, return steady-state peak-to-peak amplitude, min and max V,
    and dominant frequency. Deterministic RK4 gives the clean structural map."""
    amp = np.zeros(len(B_values))
    vmin = np.zeros(len(B_values))
    vmax = np.zeros(len(B_values))
    fdom = np.zeros(len(B_values))
    for i, B in enumerate(B_values):
        method = "rk4" if deterministic else "euler"
        _, V = simulate(params=params, B=float(B), T=T, method=method,
                        stochastic=not deterministic, seed=seed)
        k = int(len(V) * 0.35)          # steady state only
        seg = V[k:]
        amp[i] = seg.max() - seg.min()
        vmin[i], vmax[i] = seg.min(), seg.max()
        fdom[i] = dominant_frequency(V)
    return {"B": np.asarray(B_values, float), "amp": amp,
            "vmin": vmin, "vmax": vmax, "fdom": fdom}


def find_bifurcations(B_lo=0.0, B_hi=55.0, n=220, params=None, tau=TAU,
                      T=6.0, refine=True):
    """Locate limit-cycle onset (B_on) and annihilation (B_crit) by finding the
    B values where the amplitude curve crosses `tau`. Returns a dict.

    B_crit (the upper crossing, oscillation death) is the paper's therapeutic
    boundary. If `refine`, the upper crossing is bisected for extra precision.
    """
    B_values = np.linspace(B_lo, B_hi, n)
    curve = amplitude_curve(B_values, params=params, T=T)
    amp, B = curve["amp"], curve["B"]
    osc = amp > tau

    # contiguous oscillatory band = longest run of True
    B_on = B_crit = None
    idx = np.where(osc)[0]
    if len(idx):
        # split into runs
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        run = max(runs, key=len)
        i0, i1 = run[0], run[-1]
        B_on = B[i0]
        B_crit = B[i1]
        if refine and i1 + 1 < len(B):
            B_crit = _bisect_cross(B[i1], B[i1 + 1], tau, params, T)
        if refine and i0 - 1 >= 0:
            B_on = _bisect_cross(B[i0], B[i0 - 1], tau, params, T)

    return {"B_on": B_on, "B_crit": B_crit, "tau": tau, "curve": curve}


def _bisect_cross(b_osc, b_flat, tau, params, T, iters=20):
    """Bisect between an oscillating B (amp>tau) and a flat B (amp<tau) to find
    the exact amplitude==tau crossing. `osc`/`flat` are tracked by value, so the
    order of the two endpoints does not matter."""
    osc, flat = float(b_osc), float(b_flat)
    for _ in range(iters):
        mid = 0.5 * (osc + flat)
        _, V = simulate(params=params, B=mid, T=T, method="rk4")
        if oscillation_amplitude(V, frac_transient=0.35) > tau:
            osc = mid
        else:
            flat = mid
    return 0.5 * (osc + flat)


def distance_to_bifurcation(B, B_crit):
    """Signed distance. Positive => below B_crit => still in the ictal band and
    additional inhibition is required to reach the stable regime."""
    return float(B_crit - B)


def control_law(B_crit, mode="graded", margin=1.2, min_margin=1.05, thresh=0.5):
    """Return a function B_target(seizure_prob, B_current) implementing therapy.

    mode="bang"   : paper's bang-bang -> jump to margin * B_crit when seizing.
    mode="graded" : push just past B_crit, scaled by detected severity, between
                    min_margin*B_crit (barely over threshold) and margin*B_crit
                    (paper's target). This delivers the *minimum* inhibition
                    that clears the bifurcation -> less stimulation energy, an
                    improvement over always jumping to 1.2*B_crit.
    """
    def controller(seizure_prob, B_current):
        if seizure_prob < thresh:
            return B_current
        if mode == "bang":
            return margin * B_crit
        # severity in [0,1] rescaled from the decision threshold
        sev = (seizure_prob - thresh) / (1.0 - thresh)
        frac = min_margin + (margin - min_margin) * sev
        return max(B_current, frac * B_crit)
    return controller


def closed_loop(B_seizure, B_crit, t_intervention=5.0, T=10.0,
                mode="graded", params=None, seed=11):
    """Simulate a seizure (B=B_seizure) and switch inhibition to the control
    target at t_intervention. Returns (t, V, B_trace)."""
    if params is None:
        params = JRParams()
    dt = DT
    n = int(round(T / dt))
    k_int = int(round(t_intervention / dt))
    ctrl = control_law(B_crit, mode=mode)

    # Pre-intervention: ictal
    _, V1 = simulate(params=params, B=B_seizure, T=t_intervention + TRANSIENT_S,
                     method="euler", seed=seed)
    V1 = V1[:k_int] if len(V1) >= k_int else V1
    # carry final state by re-seeding a fresh sim at the control target
    B_target = ctrl(1.0, B_seizure)
    _, V2 = simulate(params=params, B=B_target, T=(T - t_intervention) + TRANSIENT_S,
                     method="euler", seed=seed + 1)
    V2 = V2[:n - len(V1)]

    V = np.concatenate([V1, V2])
    t = np.arange(len(V)) * dt
    B_trace = np.concatenate([np.full(len(V1), B_seizure),
                              np.full(len(V2), B_target)])
    return t, V, B_trace, B_target


if __name__ == "__main__":
    from jansen_ritt import warmup
    warmup()
    res = find_bifurcations()
    print(f"Limit-cycle onset  B_on  = {res['B_on']:.2f}")
    print(f"Bifurcation (crit) B_crit= {res['B_crit']:.2f}   (paper: 42.42)")
    Bt = 1.2 * res["B_crit"]
    print(f"Therapeutic target 1.2*B_crit = {Bt:.2f}   (paper: ~50.9)")
    # demonstrate seizure termination
    t, V, Btr, Btgt = closed_loop(B_seizure=34.0, B_crit=res["B_crit"])
    pre = V[t < 5.0]; post = V[t > 6.0]
    print(f"Seizure ptp before = {pre.max()-pre.min():.2f} mV, "
          f"after control = {post.max()-post.min():.2f} mV "
          f"(target B={Btgt:.1f})")
