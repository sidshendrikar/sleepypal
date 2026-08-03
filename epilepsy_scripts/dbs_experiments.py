"""
dbs_experiments.py — benchmark the bifurcation-aware controller against the
control laws actually used in deployed adaptive neurostimulators.

Experiments
  1. head-to-head      : burden / energy / duty cycle / latency, over seeds
  2. energy-suppression: Pareto frontier -- energy required for a given burden
  3. severity sweep    : does the advantage survive deeper seizures?
  4. twin-error sweep  : the twin's B_hat is an estimate; how gracefully does
                         the model-based controller degrade when it is wrong?

Writes artifacts/dbs_results.json and four figures.
"""
import json, os, time
import warnings; warnings.filterwarnings("ignore")
import numpy as np

from config import ARTIFACT_DIR
from bifurcation_analysis import find_bifurcations
from jansen_ritt import warmup
from dbs_sim import ClosedLoopDBS, SeizureScenario, score
from dbs_controllers import (NoStim, OpenLoopDBS, ThresholdADBS,
                             DualThresholdADBS, BifurcationMPC,
                             calibrate_threshold)

SEEDS = [1, 2, 3, 4, 5]
GAIN = 22.0


# ----------------------------------------------------------------------------
def calibrate(loop, seeds=(11, 12, 13)):
    """Programme the biomarker threshold from interictal data only, the way a
    real device is configured -- keeps the baselines honest."""
    lls = []
    quiet = SeizureScenario(onset_s=1e6, offset_s=1e6, T_s=20.0)
    for s in seeds:
        log = loop.run(NoStim(), quiet, seed=s)
        lls.append(log["band_power"])
    lls = np.concatenate(lls)
    thr = calibrate_threshold(lls, 99.5)
    return thr, float(lls.mean())


def make_controllers(thr, Bc, B_hat, amp=1.0, margin=1.05):
    return [
        NoStim(),
        OpenLoopDBS(amp=amp),
        ThresholdADBS(thr, amp=amp),
        DualThresholdADBS(thr, thr * 3.0, amp_low=0.3, amp_high=amp),
        BifurcationMPC(Bc, GAIN, B_hat, thr, margin=margin),
    ]


def run_set(loop, controllers, scenario, seeds=SEEDS):
    out = {}
    for ctrl in controllers:
        rows = []
        for s in seeds:
            log = loop.run(ctrl, scenario, seed=s)
            rows.append(score(log))
        agg = {}
        for k in ("seizure_burden_s", "burden_frac", "energy_mJ", "duty_cycle",
                  "mean_power_uW", "peak_u", "suppression_latency_s"):
            v = np.array([r[k] for r in rows], dtype=float)
            agg[k] = float(np.nanmean(v))
            agg[k + "_sd"] = float(np.nanstd(v))
        agg["controller"] = ctrl.name
        out[ctrl.name] = agg
    return out


# ----------------------------------------------------------------------------
def main():
    warmup()
    t0 = time.time()
    bif = find_bifurcations()
    Bc = bif["B_crit"]
    loop = ClosedLoopDBS(B_crit=Bc, gain=GAIN)
    thr, ll_base = calibrate(loop)
    print(f"B_crit={Bc:.2f}  line-length threshold={thr:.4f} "
          f"(interictal mean {ll_base:.4f})", flush=True)

    scen = SeizureScenario()
    B_hat = scen.B_ictal          # twin's fitted ictal inhibition
    results = {"B_crit": Bc, "threshold": thr, "gain": GAIN}

    # --- 1. head to head --------------------------------------------------
    print("\n[1] head-to-head", flush=True)
    ctrls = make_controllers(thr, Bc, B_hat)
    head = run_set(loop, ctrls, scen)
    results["head_to_head"] = head
    ol = head["open-loop DBS"]["energy_mJ"]
    print(f"{'controller':34s} {'burden s':>9s} {'energy mJ':>10s} "
          f"{'duty %':>7s} {'lat s':>6s} {'vs open-loop':>13s}")
    for name, r in head.items():
        sav = "-" if r["energy_mJ"] <= 0 else f"{100*(1-r['energy_mJ']/ol):5.1f}% less"
        print(f"  {name:32s} {r['seizure_burden_s']:8.1f} {r['energy_mJ']:10.2f} "
              f"{r['duty_cycle']*100:6.0f} {r['suppression_latency_s']:6.1f} {sav:>13s}",
              flush=True)

    # --- 2. Pareto: energy vs residual burden -----------------------------
    print("\n[2] energy-suppression Pareto", flush=True)
    pareto = {}
    for label, mk in [
        ("open-loop DBS", lambda a: OpenLoopDBS(amp=a)),
        ("threshold aDBS (RNS-style)", lambda a: ThresholdADBS(thr, amp=a)),
        ("dual-threshold aDBS (Percept-style)",
         lambda a: DualThresholdADBS(thr, thr * 3.0, amp_low=0.3, amp_high=a)),
    ]:
        pts = []
        for a in [0.35, 0.45, 0.55, 0.65, 0.8, 1.0]:
            rows = [score(loop.run(mk(a), scen, seed=s)) for s in SEEDS[:3]]
            pts.append({"param": a,
                        "burden_s": float(np.mean([r["seizure_burden_s"] for r in rows])),
                        "energy_mJ": float(np.mean([r["energy_mJ"] for r in rows]))})
        pareto[label] = pts
        print(f"  {label}: done", flush=True)
    pts = []
    for m in [1.0, 1.02, 1.05, 1.10, 1.15, 1.25]:
        rows = [score(loop.run(BifurcationMPC(Bc, GAIN, B_hat, thr, margin=m),
                               scen, seed=s)) for s in SEEDS[:3]]
        pts.append({"param": m,
                    "burden_s": float(np.mean([r["seizure_burden_s"] for r in rows])),
                    "energy_mJ": float(np.mean([r["energy_mJ"] for r in rows]))})
    pareto["bifurcation MPC (this work)"] = pts
    results["pareto"] = pareto
    print("  bifurcation MPC: done", flush=True)

    # --- 3. severity sweep -------------------------------------------------
    print("\n[3] severity sweep", flush=True)
    sev = {}
    for B_ict in [24.0, 26.0, 28.0, 32.0, 36.0]:
        sc = SeizureScenario(B_ictal=B_ict)
        cs = make_controllers(thr, Bc, B_ict)
        r = run_set(loop, cs, sc, seeds=SEEDS[:3])
        sev[str(B_ict)] = {k: {"burden_s": v["seizure_burden_s"],
                               "energy_mJ": v["energy_mJ"]} for k, v in r.items()}
        print(f"  B_ictal={B_ict}: done", flush=True)
    results["severity"] = sev

    # --- 4. twin-estimate error -------------------------------------------
    print("\n[4] twin B_hat error robustness", flush=True)
    err = {}
    for e in [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0]:
        rows = [score(loop.run(
            BifurcationMPC(Bc, GAIN, scen.B_ictal + e, thr, margin=1.05),
            scen, seed=s)) for s in SEEDS[:3]]
        err[str(e)] = {"burden_s": float(np.mean([r["seizure_burden_s"] for r in rows])),
                       "energy_mJ": float(np.mean([r["energy_mJ"] for r in rows]))}
        print(f"  B_hat error {e:+.0f}: burden={err[str(e)]['burden_s']:.1f}s "
              f"energy={err[str(e)]['energy_mJ']:.2f}mJ", flush=True)
    results["bhat_error"] = err

    results["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(ARTIFACT_DIR, "dbs_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    make_figures(loop, scen, thr, Bc, B_hat, results)
    print(f"\nDBS_DONE in {results['runtime_s']}s", flush=True)


# ----------------------------------------------------------------------------
def make_figures(loop, scen, thr, Bc, B_hat, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fig A: example traces
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ctrl, ax, c in [(NoStim(), axes[0], "0.3"),
                        (ThresholdADBS(thr, amp=1.0), axes[1], "tab:orange"),
                        (BifurcationMPC(Bc, GAIN, B_hat, thr), axes[2], "tab:green")]:
        log = loop.run(ctrl, scen, seed=1)
        tv = np.arange(len(log["V"])) * (log["t"][1] - log["t"][0]) / \
            (len(log["V"]) / len(log["t"]))
        ax.plot(tv, log["V"], color=c, lw=0.4)
        ax2 = ax.twinx()
        ax2.plot(log["t"], log["u"], color="tab:red", lw=1.5, alpha=0.8)
        ax2.set_ylim(-0.05, 1.35); ax2.set_ylabel("stim u", color="tab:red")
        s = score(log)
        ax.set_title(f"{ctrl.name}  —  burden {s['seizure_burden_s']:.1f}s, "
                     f"energy {s['energy_mJ']:.2f} mJ", fontsize=10)
        ax.set_ylabel("EEG (mV)")
        ax.axvspan(scen.onset_s, scen.offset_s, color="red", alpha=0.06)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "dbs_fig1_traces.png"), dpi=130)
    plt.close(fig)

    # Fig B: Pareto
    plt.figure(figsize=(6.6, 4.6))
    styles = {"open-loop DBS": ("o-", "tab:blue"),
              "threshold aDBS (RNS-style)": ("s-", "tab:orange"),
              "dual-threshold aDBS (Percept-style)": ("^-", "tab:purple"),
              "bifurcation MPC (this work)": ("D-", "tab:green")}
    for name, pts in results["pareto"].items():
        st, c = styles.get(name, ("o-", "k"))
        pts = sorted(pts, key=lambda p: p["energy_mJ"])
        plt.plot([p["energy_mJ"] for p in pts], [p["burden_s"] for p in pts],
                 st, color=c, label=name, ms=5)
    plt.xlabel("stimulation energy delivered (mJ)")
    plt.ylabel("residual seizure burden (s)")
    plt.title("Energy required to suppress (lower-left is better)")
    plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "dbs_fig2_pareto.png"), dpi=130)
    plt.close()

    # Fig C: severity
    plt.figure(figsize=(6.6, 4.4))
    sev = results["severity"]
    Bs = sorted(float(b) for b in sev)
    for name, c in [("open-loop DBS", "tab:blue"),
                    ("threshold aDBS (RNS-style)", "tab:orange"),
                    ("bifurcation MPC (this work)", "tab:green")]:
        plt.plot(Bs, [sev[str(b)][name]["energy_mJ"] for b in Bs], "o-",
                 color=c, label=name, ms=5)
    plt.xlabel("ictal inhibitory gain $B_{ictal}$  (lower = more severe)")
    plt.ylabel("energy delivered (mJ)")
    plt.title("Energy vs seizure severity")
    plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "dbs_fig3_severity.png"), dpi=130)
    plt.close()

    # Fig D: twin error robustness
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    err = results["bhat_error"]
    es = sorted(float(e) for e in err)
    ax1.plot(es, [err[str(e)]["burden_s"] for e in es], "o-", color="tab:red")
    ax1.set_xlabel("error in twin estimate $\\hat{B}$ (mV)")
    ax1.set_ylabel("residual burden (s)", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(es, [err[str(e)]["energy_mJ"] for e in es], "s--", color="tab:green")
    ax2.set_ylabel("energy (mJ)", color="tab:green")
    plt.title("Robustness to digital-twin estimation error")
    ax1.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "dbs_fig4_robustness.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
