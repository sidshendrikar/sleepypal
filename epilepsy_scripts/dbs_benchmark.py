"""
dbs_benchmark.py — the honest benchmark.

Compares a bifurcation-aware, twin-dosed controller against faithful
re-implementations of the control laws used by DEPLOYED devices, plus the
controls and ablations that keep the comparison from flattering itself:

  deployed-law arms   RNS-analogue (line-length ratio -> fixed burst),
                      Percept ST (band-power threshold, floored amplitude),
                      Percept DT (dead-band, rate limited),
                      open-loop continuous, open-loop cycled
  controls            duty-cycle- and amplitude-MATCHED random stimulation
                      (isolates timing/dosing from mere intermittency)
  ablation            same adaptive ratchet, twin dose prediction REMOVED
                      (isolates the contribution of the bifurcation coordinate)

Scoring uses 2-8 Hz band power against a threshold calibrated on interictal
data, because the stochastic drive gives the suppressed state a several-mV
peak-to-peak that a raw-amplitude criterion would misread as seizure.

IMPORTANT: every arm is OUR re-implementation of a published control law
inside OUR model. This measures control-law behaviour in a neural mass model;
it is not a comparison against the commercial devices themselves.
"""
import json, os, time
import warnings; warnings.filterwarnings("ignore")
import numpy as np

from config import ARTIFACT_DIR
from bifurcation_analysis import find_bifurcations
from jansen_ritt import warmup
from dbs_sim import ClosedLoopDBS, SeizureScenario, score
from dbs_controllers import (NoStim, OpenLoopDBS, OpenLoopCycled, ThresholdADBS,
                             DualThresholdADBS, PerceptST, RNSLineLength,
                             RandomDutyMatched, BifurcationMPC,
                             calibrate_threshold)

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
GAIN = 22.0


def calibrate(loop, seeds=(11, 12, 13, 14)):
    """Programme the detector threshold AND the (independent, stricter) scoring
    threshold from interictal data only."""
    bps = []
    quiet = SeizureScenario(onset_s=1e6, offset_s=1e6, T_s=20.0)
    for s in seeds:
        bps.append(loop.run(NoStim(), quiet, seed=s)["band_power"])
    bps = np.concatenate(bps)
    det = calibrate_threshold(bps, 99.5)     # what the device triggers on
    sco = calibrate_threshold(bps, 99.9)     # what we call a seizure
    return det, sco, float(bps.mean())


def agg(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(np.nanmean(v))
        out[k + "_sd"] = float(np.nanstd(v))
    return out


KEYS = ("seizure_burden_s", "burden_frac", "energy_mJ", "duty_cycle",
        "mean_power_uW", "peak_u", "suppression_latency_s", "charge_uC")


def evaluate(loop, ctrl, scen, sco, seeds=SEEDS):
    rows = [score(loop.run(ctrl, scen, seed=s), seizure_thresh=sco)
            for s in seeds]
    a = agg(rows, KEYS)
    a["controller"] = ctrl.name
    return a


def main():
    warmup()
    t0 = time.time()
    bif = find_bifurcations()
    Bc = bif["B_crit"]
    loop = ClosedLoopDBS(B_crit=Bc, gain=GAIN)
    det, sco, base = calibrate(loop)
    scen = SeizureScenario()
    B_hat = scen.B_ictal
    u_star = (1.05 * Bc - B_hat) / GAIN
    print(f"B_crit={Bc:.2f}  detect_thr={det:.4f}  score_thr={sco:.4f} "
          f"(interictal mean {base:.5f})")
    print(f"model-predicted minimum dose u* = {u_star:.3f}\n", flush=True)

    results = {"B_crit": Bc, "detect_thr": det, "score_thr": sco,
               "gain": GAIN, "u_star": u_star, "n_seeds": len(SEEDS)}

    arms = [
        NoStim(),
        OpenLoopDBS(amp=1.0),
        OpenLoopCycled(amp=1.0, on_s=60.0, off_s=300.0),
        RNSLineLength(ratio=1.75, amp=1.0),
        PerceptST(det, u_min=0.3, u_max=1.0),
        DualThresholdADBS(det, det * 3.0, amp_low=0.3, amp_high=1.0),
        ThresholdADBS(det, amp=1.0, min_burst_s=2.0),
        BifurcationMPC(Bc, GAIN, B_hat, det, margin=1.05, hold_s=2.0),
        BifurcationMPC(Bc, GAIN, B_hat, det, margin=1.05, hold_s=2.0,
                       use_model_dose=False),
    ]
    arms[-1].name = "ablation: no twin dose"

    table = {}
    for c in arms:
        table[c.name] = evaluate(loop, c, scen, sco)
        r = table[c.name]
        print(f"  {c.name:34s} burden={r['seizure_burden_s']:5.1f}s "
              f"energy={r['energy_mJ']:6.2f}mJ duty={r['duty_cycle']*100:3.0f}% "
              f"peak_u={r['peak_u']:.2f}", flush=True)

    # duty- and amplitude-matched random control, matched to the MPC arm
    mpc = table["bifurcation MPC (this work)"]
    rnd = RandomDutyMatched(duty=mpc["duty_cycle"],
                            amp=max(mpc["peak_u"], 1e-6), seed=7)
    table[rnd.name] = evaluate(loop, rnd, scen, sco)
    r = table[rnd.name]
    print(f"  {rnd.name:34s} burden={r['seizure_burden_s']:5.1f}s "
          f"energy={r['energy_mJ']:6.2f}mJ duty={r['duty_cycle']*100:3.0f}% "
          f"(matched to MPC)", flush=True)

    results["arms"] = table
    results["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(ARTIFACT_DIR, "dbs_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    # ---- headline comparisons -------------------------------------------
    ol = table["open-loop DBS"]
    mp = table["bifurcation MPC (this work)"]
    print("\n" + "=" * 64)
    for name in ["RNS-analogue (line-length ratio)", "Percept ST aDBS",
                 "dual-threshold aDBS (Percept-style)",
                 "random (duty-matched)", "ablation: no twin dose"]:
        o = table[name]
        de = 100 * (1 - mp["energy_mJ"] / max(o["energy_mJ"], 1e-9))
        db = o["seizure_burden_s"] - mp["seizure_burden_s"]
        print(f"  vs {name:36s} energy {de:+6.1f}%   burden {db:+5.1f}s")
    print(f"  vs {'open-loop continuous':36s} energy "
          f"{100*(1-mp['energy_mJ']/ol['energy_mJ']):+6.1f}%   burden "
          f"{ol['seizure_burden_s']-mp['seizure_burden_s']:+5.1f}s")
    print("=" * 64)
    print(f"BENCH_DONE {results['runtime_s']}s", flush=True)




# ---------------------------------------------------------------------------
# Realistic-duty and matched-suppression analysis
# ---------------------------------------------------------------------------
def realistic_and_matched():
    """Two analyses the head-to-head table cannot show.

    (a) REALISTIC DUTY. The head-to-head scenario is 58% seizure, which
        flatters always-on therapy: real seizures occupy seconds per hour. A
        10-minute window containing one 35 s seizure is far closer to the
        clinical energy calculus.
    (b) MATCHED SUPPRESSION. Arms with different residual burden cannot be
        compared on energy alone, so each arm's amplitude (or, for the
        model-based arm, its safety margin) is swept to find the CHEAPEST
        setting that still achieves a target burden.
    """
    warmup()
    bif = find_bifurcations(); Bc = bif["B_crit"]
    loop = ClosedLoopDBS(B_crit=Bc, gain=GAIN)
    det, sco, _ = calibrate(loop)
    out = {"B_crit": Bc, "detect_thr": det, "score_thr": sco}

    long_scen = SeizureScenario(onset_s=60.0, offset_s=95.0, T_s=600.0)
    B_hat = long_scen.B_ictal
    seeds = [1, 2, 3]

    print("\n[realistic duty] 10-minute window, one 35 s seizure "
          "(5.8% of the record)", flush=True)
    arms = [
        ("open-loop continuous", lambda: OpenLoopDBS(amp=1.0)),
        ("open-loop cycled 5s/25s", lambda: OpenLoopCycled(1.0, 5.0, 25.0)),
        ("Percept ST aDBS", lambda: PerceptST(det, 0.3, 1.0)),
        ("dual-threshold aDBS (Percept-style)",
         lambda: DualThresholdADBS(det, det * 3.0, 0.3, 1.0)),
        ("threshold aDBS (RNS-style)",
         lambda: ThresholdADBS(det, 1.0, min_burst_s=2.0)),
        ("bifurcation MPC (this work)",
         lambda: BifurcationMPC(Bc, GAIN, B_hat, det, margin=1.05, hold_s=2.0)),
    ]
    tbl = {}
    for name, mk in arms:
        rows = [score(loop.run(mk(), long_scen, seed=s), seizure_thresh=sco)
                for s in seeds]
        a = agg(rows, KEYS); a["controller"] = name; tbl[name] = a
        print(f"  {name:36s} burden={a['seizure_burden_s']:5.1f}s "
              f"energy={a['energy_mJ']:7.2f}mJ duty={a['duty_cycle']*100:4.0f}%",
              flush=True)
    out["realistic"] = tbl
    ol = tbl["open-loop continuous"]["energy_mJ"]
    mp = tbl["bifurcation MPC (this work)"]["energy_mJ"]
    print(f"  --> model-based uses {100*(1-mp/ol):.1f}% less energy than "
          f"continuous DBS over a realistic record", flush=True)

    # ---- matched suppression ------------------------------------------
    print("\n[matched suppression] cheapest setting reaching burden <= 2.0 s",
          flush=True)
    TARGET = 2.0
    sweeps = {
        "open-loop continuous": [("amp", a, lambda a=a: OpenLoopDBS(amp=a))
                                 for a in np.linspace(0.55, 1.0, 10)],
        "Percept ST aDBS": [("u_max", a, lambda a=a: PerceptST(det, 0.3, a))
                            for a in np.linspace(0.55, 1.0, 10)],
        "threshold aDBS (RNS-style)":
            [("amp", a, lambda a=a: ThresholdADBS(det, a, min_burst_s=2.0))
             for a in np.linspace(0.55, 1.0, 10)],
        "bifurcation MPC (this work)":
            [("margin", m, lambda m=m: BifurcationMPC(Bc, GAIN, B_hat, det,
                                                      margin=m, hold_s=2.0))
             for m in np.linspace(1.0, 1.30, 10)],
    }
    matched = {}
    for name, grid in sweeps.items():
        best = None
        for pname, pval, mk in grid:
            rows = [score(loop.run(mk(), long_scen, seed=s), seizure_thresh=sco)
                    for s in seeds]
            b = float(np.mean([r["seizure_burden_s"] for r in rows]))
            e = float(np.mean([r["energy_mJ"] for r in rows]))
            if b <= TARGET and (best is None or e < best["energy_mJ"]):
                best = {"param": pname, "value": float(pval),
                        "burden_s": b, "energy_mJ": e}
        matched[name] = best
        if best:
            print(f"  {name:36s} {best['param']}={best['value']:.2f} -> "
                  f"burden={best['burden_s']:.1f}s energy={best['energy_mJ']:.2f}mJ",
                  flush=True)
        else:
            print(f"  {name:36s} never reached burden <= {TARGET}s", flush=True)
    out["matched"] = matched

    mm = matched.get("bifurcation MPC (this work)")
    if mm:
        for other in ["open-loop continuous", "Percept ST aDBS",
                      "threshold aDBS (RNS-style)"]:
            o = matched.get(other)
            if o:
                print(f"  --> vs {other:34s} "
                      f"{100*(1-mm['energy_mJ']/o['energy_mJ']):+6.1f}% energy "
                      f"at matched suppression", flush=True)

    with open(os.path.join(ARTIFACT_DIR, "dbs_realistic.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("REALISTIC_DONE", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if "--realistic" in sys.argv:
        realistic_and_matched()
    else:
        main()
