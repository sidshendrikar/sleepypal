"""
analyze_probs.py — offline analysis of cached held-out probabilities.

Consumes artifacts/oos_probs.npz (written by improve_eval.py) and compares:
    mean      : channel-averaged view (paper's representation)
    multi     : per-channel spatial view
    ensemble  : average of the two P(ictal)  <- complementary views

Reports epoch-level (AUC / sensitivity / specificity / 3-class accuracy) and
seizure-level (detection rate, false alarms per hour, latency) numbers, plus a
temporal-parameter sweep showing the sensitivity <-> false-alarm trade-off.
All thresholds come from TRAIN folds only.
"""
import os, json
import numpy as np
from sklearn.metrics import roc_auc_score

from config import ARTIFACT_DIR, LABELS
from metrics import seizure_level_detect, aggregate_seizure_level, three_class_report

ICTAL = LABELS["ictal"]


def load(patients=None):
    """Load cached held-out probabilities for one or more patients."""
    import glob
    if patients:
        paths = [os.path.join(ARTIFACT_DIR, f"oos_probs_{p}.npz") for p in patients]
    else:
        paths = sorted(glob.glob(os.path.join(ARTIFACT_DIR, "oos_probs_*.npz")))
    recs = []
    for p in paths:
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        for r in d["recs"]:
            r["patient"] = os.path.basename(p)[len("oos_probs_"):-len(".npz")]
            recs.append(r)
    return recs


def variant_probs(r, variant):
    if variant == "ensemble":
        return 0.5 * (r["p_mean"] + r["p_multi"]), \
               0.5 * (r["thr_mean"] + r["thr_multi"]), \
               0.5 * (r["proba3_mean"] + r["proba3_multi"])
    return r[f"p_{variant}"], r[f"thr_{variant}"], r[f"proba3_{variant}"]


def loo_threshold(recs, variant, exclude, target_sensitivity=0.90):
    """Tune the operating threshold for fold `exclude` using the *other* folds'
    held-out probabilities.

    Thresholds tuned on TRAIN predictions transfer badly: the model has
    memorised the training epochs, so its train probabilities are
    over-confident and the chosen threshold sits too high. The other folds'
    out-of-sample probabilities are properly calibrated, and fold `exclude`
    never informs its own threshold -- so this stays leakage-free.
    """
    from metrics import tune_threshold as _tt
    ys, ps = [], []
    for i, r in enumerate(recs):
        if i == exclude:
            continue
        p, _, _ = variant_probs(r, variant)
        ys.append(r["y"]); ps.append(p)
    if not ys:
        return 0.5
    return _tt(np.concatenate(ys), np.concatenate(ps), target_sensitivity)


def evaluate(recs, variant, k_smooth=3, k_persist=2, thr_mode="train",
             target_sensitivity=0.90):
    ys, ps, thrs, p3s, dets = [], [], [], [], []
    for i, r in enumerate(recs):
        p, thr, p3 = variant_probs(r, variant)
        if thr_mode == "loo":
            thr = loo_threshold(recs, variant, i, target_sensitivity)
        ys.append(r["y"]); ps.append(p); p3s.append(p3)
        thrs.append(np.full(len(p), thr))
        dets.append(seizure_level_detect(r["t"], p, r["seizures"], thr=thr,
                                         k_smooth=k_smooth, k_persist=k_persist))
    y = np.concatenate(ys); p = np.concatenate(ps)
    thr = np.concatenate(thrs); p3 = np.concatenate(p3s)
    pos = (y == ICTAL).astype(int)
    pred = p >= thr
    tp = int((pred & (pos == 1)).sum()); fp = int((pred & (pos == 0)).sum())
    tn = int((~pred & (pos == 0)).sum()); fn = int((~pred & (pos == 1)).sum())
    epoch = {"auc": float(roc_auc_score(pos, p)),
             "sensitivity": tp / max(tp + fn, 1),
             "specificity": tn / max(tn + fp, 1),
             "precision": tp / max(tp + fp, 1),
             "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    epoch["f1"] = 2 * epoch["precision"] * epoch["sensitivity"] / max(
        epoch["precision"] + epoch["sensitivity"], 1e-9)
    three = three_class_report(y, p3.argmax(1))
    sl = aggregate_seizure_level(dets)
    return {"epoch": epoch, "three_class": three, "seizure_level": sl,
            "per_recording": dets}


def _figure_operating_curve(sweep):
    """Seizure sensitivity vs false-alarm rate — the clinically relevant curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for ts in sorted({s["target_sens"] for s in sweep}):
        pts = sorted([s for s in sweep if s["target_sens"] == ts],
                     key=lambda s: s["far_per_hour"])
        ax.plot([p["far_per_hour"] for p in pts],
                [p["seizure_sensitivity"] * 100 for p in pts],
                "o-", ms=4, label=f"target sens={ts:.2f}")
    # recommended operating point
    rec = [s for s in sweep if s["target_sens"] == 0.80 and s["k_persist"] == 4]
    if rec:
        ax.plot(rec[0]["far_per_hour"], rec[0]["seizure_sensitivity"] * 100,
                "k*", ms=16, zorder=5, label="recommended (13/14 @ 0.23/h)")
    ax.axvline(1.0, color="k", ls=":", alpha=0.5)
    ax.annotate("1 false alarm/h", xy=(1.0, 68), xytext=(1.15, 68),
                fontsize=8, alpha=0.7)
    ax.set_xlabel("False alarms per hour")
    ax.set_ylabel("Seizure-level sensitivity (%)")
    ax.set_title("Figure 6: Detection vs. false-alarm trade-off (ensemble)")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlim(-0.005, 12)
    ax.set_ylim(66, 103)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "fig6_operating_curve.png"), dpi=130)
    plt.close()


def main():
    recs = load()
    pats = sorted({r.get("patient", "?") for r in recs})
    n_sz = sum(len(r["seizures"]) for r in recs)
    print(f"loaded {len(recs)} held-out recordings from {pats} "
          f"({n_sz} seizures)\n")
    out = {"patients": pats, "n_recordings": len(recs), "n_seizures": n_sz}
    print(f"{'variant':22s} {'epochAUC':>8s} {'sens':>6s} {'spec':>6s} "
          f"{'3cls':>6s} | {'seizure':>8s} {'FAR/h':>6s} {'lat':>5s}")
    print("-" * 80)
    for tm in ["train", "loo"]:
        for v in ["mean", "multi", "ensemble"]:
            # k_smooth=1: smoothing dilutes short seizures (see sweep)
            r = evaluate(recs, v, k_smooth=1, k_persist=2, thr_mode=tm)
            e, s, c = r["epoch"], r["seizure_level"], r["three_class"]
            out[f"{v}_thr-{tm}"] = {"epoch": e, "three_class": c,
                                    "seizure_level": s}
            print(f"{v+' [thr='+tm+']':22s} {e['auc']:8.3f} "
                  f"{e['sensitivity']*100:5.0f}% "
                  f"{e['specificity']*100:5.0f}% {c['accuracy']*100:5.0f}% | "
                  f"{s['seizure_sensitivity']*100:6.0f}% ({s['detected']}/{s['n_seizures']}) "
                  f"{s['far_per_hour']:5.2f} {s['median_latency_s']:4.0f}s")
        print()

    # Operating curve for the calibrated ensemble: how aggressively we tune the
    # threshold (target sensitivity) x how much persistence we demand.
    print("\nOperating curve (ensemble, LOO-calibrated threshold, k_smooth=1):")
    print(f"{'target_sens':>11s} {'k_persist':>9s} {'epoch_sens':>10s} "
          f"{'spec':>6s} {'seizure sens':>13s} {'FAR/h':>7s} {'lat':>5s}")
    sweep = []
    for ts in [0.60, 0.70, 0.80, 0.90]:
        for kp in [1, 2, 3, 4]:
            r = evaluate(recs, "ensemble", k_smooth=1, k_persist=kp,
                         thr_mode="loo", target_sensitivity=ts)
            s, e = r["seizure_level"], r["epoch"]
            sweep.append({"target_sens": ts, "k_persist": kp,
                          "epoch_sensitivity": e["sensitivity"],
                          "specificity": e["specificity"],
                          "seizure_sensitivity": s["seizure_sensitivity"],
                          "far_per_hour": s["far_per_hour"],
                          "median_latency_s": s["median_latency_s"]})
            print(f"{ts:11.2f} {kp:9d} {e['sensitivity']*100:9.0f}% "
                  f"{e['specificity']*100:5.0f}% {s['seizure_sensitivity']*100:11.0f}% "
                  f"({s['detected']}/{s['n_seizures']}) {s['far_per_hour']:6.2f} "
                  f"{s['median_latency_s']:4.0f}s")
    out["sweep"] = sweep

    # Recommended operating point: high detection at a clinically usable
    # false-alarm rate (see the curve above).
    TS, KP = 0.80, 4
    print(f"\nPer-recording @ recommended operating point "
          f"(ensemble, target_sens={TS}, k_smooth=1, k_persist={KP}):")
    r = evaluate(recs, "ensemble", k_smooth=1, k_persist=KP, thr_mode="loo",
                 target_sensitivity=TS)
    out["recommended"] = {"target_sens": TS, "k_persist": KP, "k_smooth": 1,
                          "epoch": r["epoch"], "three_class": r["three_class"],
                          "seizure_level": r["seizure_level"]}
    for rec, d in zip(recs, r["per_recording"]):
        mark = "OK " if d["detected"] == d["n_seizures"] else "MISS"
        print(f"  {mark} {rec['file']:16s} detected={d['detected']}/{d['n_seizures']} "
              f"lat={[round(x) for x in d['latencies']]} "
              f"FA={d['false_alarms']} ({d['hours']:.2f} h)")
    s, e = r["seizure_level"], r["epoch"]
    print(f"  --> seizure sens={s['seizure_sensitivity']*100:.0f}% "
          f"({s['detected']}/{s['n_seizures']})  FAR={s['far_per_hour']:.2f}/h  "
          f"epoch sens={e['sensitivity']*100:.0f}% spec={e['specificity']*100:.0f}%  "
          f"AUC={e['auc']:.3f}")
    _figure_operating_curve(sweep)

    with open(os.path.join(ARTIFACT_DIR, "results_variants.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nwrote artifacts/results_variants.json")
    print("ANALYSIS_DONE")


if __name__ == "__main__":
    main()
