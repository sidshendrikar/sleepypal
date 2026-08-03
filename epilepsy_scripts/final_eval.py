"""
final_eval.py — definitive comparison run.

Trains the observer under three configurations on the same leave-one-seizure-
recording-out splits and reports epoch-level + seizure-level metrics:

  1. paper-style   : channel-averaged image, epoch decisions only
  2. + spatial     : per-channel multi-channel input (improvement #1)
  3. + temporal    : winner of (1/2) with seizure-level post-processing (#2)

Writes artifacts/results_final.json and regenerates the figures for the
best configuration.
"""
import os, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings; warnings.filterwarnings("ignore")
import numpy as np

from config import TrainConfig, ARTIFACT_DIR
import preprocessing as pp
from main import evaluate_patient, pooled_report, stage_twin
from metrics import aggregate_seizure_level, print_confusion

PATIENTS = os.environ.get("PATIENTS", "chb01").split(",")


def run_mode(dss, cfg, channel_mode):
    all_true, all_proba, all_thr, all_sl = [], [], [], []
    for ds in dss:
        oos = evaluate_patient(ds, cfg, channel_mode=channel_mode, verbose=True)
        if oos is None:
            continue
        all_true.append(oos["y"]); all_proba.append(oos["proba"])
        all_thr.append(oos["thr"]); all_sl.extend(oos["seizure_level"])
    pooled = {"y": np.concatenate(all_true), "proba": np.concatenate(all_proba),
              "thr": np.concatenate(all_thr)}
    binm, three = pooled_report(pooled)
    sl = aggregate_seizure_level(all_sl)
    return {"binary": binm, "three_class": three, "seizure_level": sl,
            "pooled_oos": pooled}


def main():
    man = pp.load_manifest()
    dss = [pp.build_patient_dataset(p, man) for p in PATIENTS]
    dss = [d for d in dss if d is not None]
    n_ict = sum(int((d["y"] == 1).sum()) for d in dss)
    n_ep = sum(len(d["y"]) for d in dss)
    print(f"Patients={PATIENTS}  epochs={n_ep}  ictal={n_ict}", flush=True)

    cfg = TrainConfig()
    results = {}
    t0 = time.time()
    for mode in ["mean", "multi"]:
        print(f"\n########## channel_mode = {mode} ##########", flush=True)
        r = run_mode(dss, cfg, mode)
        b, s = r["binary"], r["seizure_level"]
        print(f"==> [{mode}] epoch AUC={b['auc']:.3f} sens={b['sensitivity']:.3f} "
              f"spec={b['specificity']:.3f} 3cls_acc={r['three_class']['accuracy']:.3f} "
              f"| seizure sens={s['seizure_sensitivity']*100:.0f}% "
              f"FAR={s['far_per_hour']:.2f}/h lat={s['median_latency_s']:.0f}s", flush=True)
        results[mode] = {k: v for k, v in r.items() if k != "pooled_oos"}
        results[mode + "_oos"] = r["pooled_oos"]

    # winner by epoch AUC
    winner = max(["mean", "multi"], key=lambda m: results[m]["binary"]["auc"])
    print(f"\nWINNER (epoch AUC): {winner}", flush=True)

    # figures for the winner
    from main import _figure_confusion, _figure_roc
    _figure_confusion(results[winner]["three_class"]["confusion_matrix"])
    _figure_roc(results[winner + "_oos"])

    # mechanistic half (twin/bifurcation/control) once
    twin = stage_twin(man, patient=PATIENTS[0])

    out = {"patients": PATIENTS, "n_epochs": n_ep, "n_ictal": n_ict,
           "winner": winner, "runtime_s": round(time.time() - t0, 1),
           "twin": twin}
    for m in ["mean", "multi"]:
        out[m] = {"binary": results[m]["binary"],
                  "three_class": results[m]["three_class"],
                  "seizure_level": results[m]["seizure_level"]}
    with open(os.path.join(ARTIFACT_DIR, "results_final.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("\n" + "=" * 60 + "\nFINAL COMPARISON\n" + "=" * 60)
    for m in ["mean", "multi"]:
        b = out[m]["binary"]; s = out[m]["seizure_level"]; c = out[m]["three_class"]
        print(f"  {m:6s}: epochAUC={b['auc']:.3f} sens={b['sensitivity']*100:.0f}% "
              f"spec={b['specificity']*100:.0f}% acc={c['accuracy']*100:.0f}% || "
              f"seizure={s['seizure_sensitivity']*100:.0f}% FAR={s['far_per_hour']:.2f}/h")
    print(f"\nBifurcation B_crit={twin['B_crit']:.2f}  target={twin['B_target']:.1f}  "
          f"suppression={twin['suppression_pct']:.0f}%")
    print("FINAL_DONE", flush=True)


if __name__ == "__main__":
    main()
