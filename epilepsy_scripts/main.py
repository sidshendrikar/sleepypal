"""
main.py — Orchestrates the Hybrid Digital Twin framework end-to-end.

    python main.py --stage all            # everything (default)
    python main.py --stage observer       # just the CNN-LSTM CV
    python main.py --stage twin           # just twin fit + bifurcation + control
    python main.py --patients chb01 chb03
    python main.py --quick                # fewer epochs, no synth (fast smoke)

Produces:
    artifacts/results.json        machine-readable metrics
    artifacts/*.png               figures (PSD match, bifurcation, control,
                                  confusion matrix, ROC)
and prints a human summary with resume-ready bullet points.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from config import (ARTIFACT_DIR, TrainConfig, LABELS, LABEL_NAMES, FS)
import preprocessing as pp
from observer import train_observer, predict_proba
from metrics import (tune_threshold, seizure_metrics, three_class_report,
                     print_confusion)

ICTAL = LABELS["ictal"]


# ----------------------------------------------------------------------------
# Cross-validated observer evaluation (leave-seizure-out, no leakage)
# ----------------------------------------------------------------------------
def evaluate_patient(ds, cfg, synth=None, channel_mode="mean",
                     k_smooth=3, k_persist=2, verbose=True):
    """Leave-one-seizure-RECORDING-out CV (no temporal leakage). For each
    seizure file: train on every other recording, then score the held-out
    recording in time order for both epoch-level and seizure-level metrics."""
    from preprocessing import input_tensor
    from metrics import seizure_level_detect
    X = input_tensor(ds["X"], channel_mode)
    y, fidx, t = ds["y"], ds["file_idx"], ds["t_start"]
    files, fseiz = ds["files"], ds["file_seizures"]
    seizure_files = [i for i in range(len(files)) if len(fseiz[i]) > 0]
    if len(seizure_files) < 2 or (y == ICTAL).sum() < 2:
        if verbose:
            print(f"  [{ds['patient']}] too few seizures for CV, skipping")
        return None

    Xsyn, ysyn = (synth if synth is not None else (None, None))
    oos_true, oos_proba, oos_thr, sldet = [], [], [], []
    for k, f in enumerate(seizure_files):
        test = fidx == f
        train = ~test
        if (y[train] == ICTAL).sum() < 1 or (y[test] == ICTAL).sum() < 1:
            continue
        model, _, norm = train_observer(X[train], y[train], cfg,
                                        X_syn=Xsyn, y_syn=ysyn)
        ptr = predict_proba(model, X[train], norm)
        thr = tune_threshold(y[train], ptr[:, ICTAL], cfg.target_sensitivity)
        # held-out recording, ordered in time
        idx = np.where(test)[0]
        order = idx[np.argsort(t[idx])]
        pte = predict_proba(model, X[order], norm)
        oos_true.append(y[order]); oos_proba.append(pte)
        oos_thr.append(np.full(len(order), thr))
        det = seizure_level_detect(t[order], pte[:, ICTAL], list(fseiz[f]),
                                   thr=thr, k_smooth=k_smooth, k_persist=k_persist)
        sldet.append(det)
        if verbose:
            fm = seizure_metrics(y[order], pte, thr)
            print(f"  [{ds['patient']}] {files[f]}: epoch AUC={fm['auc']:.3f} "
                  f"sens={fm['sensitivity']:.2f} spec={fm['specificity']:.2f} | "
                  f"seizure {det['detected']}/{det['n_seizures']} "
                  f"lat={det['latencies']}", flush=True)
    if not oos_true:
        return None
    return {"y": np.concatenate(oos_true),
            "proba": np.concatenate(oos_proba),
            "thr": np.concatenate(oos_thr),
            "seizure_level": sldet, "patient": ds["patient"]}


def pooled_report(oos):
    """Binary + 3-class metrics from pooled OOS predictions."""
    y = oos["y"]; proba = oos["proba"]; thr = oos["thr"]
    pred_bin = (proba[:, ICTAL] >= thr).astype(int)
    pos = (y == ICTAL).astype(int)
    tp = int(((pred_bin == 1) & (pos == 1)).sum())
    fp = int(((pred_bin == 1) & (pos == 0)).sum())
    tn = int(((pred_bin == 0) & (pos == 0)).sum())
    fn = int(((pred_bin == 0) & (pos == 1)).sum())
    from sklearn.metrics import roc_auc_score
    binm = {
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1),
        "auc": float(roc_auc_score(pos, proba[:, ICTAL])) if pos.any() and (pos == 0).any() else float("nan"),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    binm["f1"] = 2 * binm["precision"] * binm["sensitivity"] / max(
        binm["precision"] + binm["sensitivity"], 1e-9)
    y_pred3 = proba.argmax(1)
    three = three_class_report(y, y_pred3)
    return binm, three


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def stage_observer(patients, manifest, cfg, use_synth=True, channel_mode="mean"):
    from metrics import aggregate_seizure_level
    print(f"\n=== OBSERVATION ENGINE (CNN-LSTM, leave-recording-out CV, "
          f"channels={channel_mode}) ===")
    synth = None
    if use_synth:
        print("  generating Jansen-Rit synthetic ictal epochs for augmentation...")
        synth = build_synth(cfg.seed, channel_mode=channel_mode)
        print(f"  synthetic augmentation: {len(synth[1])} epochs "
              f"({int((synth[1]==ICTAL).sum())} ictal)")

    all_true, all_proba, all_thr, all_sl = [], [], [], []
    per_patient = {}
    for p in patients:
        ds = pp.build_patient_dataset(p, manifest)
        if ds is None:
            continue
        oos = evaluate_patient(ds, cfg, synth=synth, channel_mode=channel_mode)
        if oos is None:
            continue
        binm, three = pooled_report(oos)
        sl = aggregate_seizure_level(oos["seizure_level"])
        per_patient[p] = {"binary": binm, "three_class": three, "seizure_level": sl,
                          "n_ictal": int((ds["y"] == ICTAL).sum()),
                          "n_epochs": int(len(ds["y"]))}
        all_true.append(oos["y"]); all_proba.append(oos["proba"]); all_thr.append(oos["thr"])
        all_sl.extend(oos["seizure_level"])
        print(f"  --> {p}: EPOCH sens={binm['sensitivity']:.3f} "
              f"spec={binm['specificity']:.3f} auc={binm['auc']:.3f} | "
              f"SEIZURE sens={sl['seizure_sensitivity']*100:.0f}% "
              f"({sl['detected']}/{sl['n_seizures']}) "
              f"FAR={sl['far_per_hour']:.2f}/h lat={sl['median_latency_s']:.0f}s")

    pooled = None
    if all_true:
        pooled_oos = {"y": np.concatenate(all_true),
                      "proba": np.concatenate(all_proba),
                      "thr": np.concatenate(all_thr)}
        binm, three = pooled_report(pooled_oos)
        sl = aggregate_seizure_level(all_sl)
        pooled = {"binary": binm, "three_class": three, "seizure_level": sl}
        _figure_confusion(three["confusion_matrix"])
        _figure_roc(pooled_oos)
    return {"per_patient": per_patient, "pooled": pooled, "channel_mode": channel_mode}


def build_synth(seed, n_per_class=120, channel_mode="mean"):
    """Jansen-Rit synthetic spectrogram epochs for train-time augmentation.
    Produced single-channel; train_observer tiles to match multi-channel input."""
    from simulator import synthesize_dataset
    from preprocessing import log_mel_spectrogram
    Xr, yr = synthesize_dataset(n_per_class=n_per_class, T=5.0, seed=seed)
    X = np.stack([log_mel_spectrogram(sig, FS) for sig in Xr])[..., None]
    return X.astype("float32"), yr.astype(int)


def stage_twin(manifest, patient="chb01"):
    print("\n=== REPLICATION + CONTROL (digital twin, bifurcation, controller) ===")
    from jansen_ritt import warmup
    from digital_twin import DigitalTwin
    from bifurcation_analysis import find_bifurcations, closed_loop, distance_to_bifurcation
    warmup()

    # grab one real seizure epoch to personalize the twin
    ds = pp.build_patient_dataset(patient, manifest)
    ictal = np.where(ds["y"] == ICTAL)[0]
    real_seizure = ds["raw"][ictal[len(ictal) // 2]] if len(ictal) else None

    twin = DigitalTwin()
    if real_seizure is not None:
        print(f"  personalizing twin on a real {patient} seizure epoch:")
        twin.calibrate(real_seizure, verbose=True)
        _figure_psd(twin.fit, patient)
    else:
        twin.bif = find_bifurcations()
        twin.B_fit = 34.0

    bif = twin.bif
    _figure_bifurcation(bif)

    # closed-loop control demo: seizure -> intervention -> healthy
    B_sz = 34.0
    t, V, Btr, Btgt = closed_loop(B_seizure=B_sz, B_crit=bif["B_crit"],
                                  t_intervention=5.0, T=10.0, mode="graded")
    _figure_control(t, V, Btr, bif["B_crit"], B_sz, Btgt)
    pre = V[t < 5.0]; post = V[t > 6.0]
    supp = 100 * (1 - (post.max() - post.min()) / max(pre.max() - pre.min(), 1e-9))
    print(f"  closed loop: seizure {pre.max()-pre.min():.1f} mV ptp "
          f"-> {post.max()-post.min():.2f} mV ({supp:.1f}% suppression), "
          f"target B={Btgt:.1f}")
    return {"B_fit": twin.B_fit, "B_crit": bif["B_crit"], "B_on": bif["B_on"],
            "B_target": Btgt, "suppression_pct": float(supp),
            "distance_to_bifurcation": distance_to_bifurcation(twin.B_fit, bif["B_crit"])}


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _figure_psd(fit, patient):
    plt = _mpl()
    plt.figure(figsize=(7, 4))
    plt.semilogy(fit["f_real"], fit["P_real"], "r-", lw=2, label="Real seizure EEG")
    plt.semilogy(fit["f_sim"], fit["P_sim"], "b--", lw=2,
                 label=f"Digital twin (B={fit['B']:.1f})")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Power spectral density")
    plt.title(f"Figure 1: Twin spectral match ({patient})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "fig1_psd_match.png"), dpi=130)
    plt.close()


def _figure_bifurcation(bif):
    plt = _mpl()
    c = bif["curve"]
    plt.figure(figsize=(7, 4))
    plt.plot(c["B"], c["vmax"], "r-", lw=1.5, label="Max voltage")
    plt.plot(c["B"], c["vmin"], "b-", lw=1.5, label="Min voltage")
    plt.axvline(bif["B_crit"], color="k", ls="--",
                label=f"B_crit={bif['B_crit']:.2f}")
    if bif["B_on"]:
        plt.axvline(bif["B_on"], color="gray", ls=":", label=f"B_on={bif['B_on']:.2f}")
    plt.xlabel("Inhibitory gain B"); plt.ylabel("Steady-state voltage (mV)")
    plt.title("Figure 2: Bifurcation diagram")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "fig2_bifurcation.png"), dpi=130)
    plt.close()


def _figure_control(t, V, Btr, B_crit, B_sz, Btgt):
    plt = _mpl()
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(t, V, "k-", lw=0.7)
    ax1.axvline(5.0, color="r", ls="--", label="Intervention")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Simulated EEG (mV)")
    ax1.set_title(f"Figure 3: Closed-loop control  (seizure B={B_sz} -> B={Btgt:.1f})")
    ax2 = ax1.twinx()
    ax2.plot(t, Btr, "g-", alpha=0.5, lw=1.5)
    ax2.set_ylabel("Inhibitory gain B", color="g")
    ax1.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "fig3_control.png"), dpi=130)
    plt.close()


def _figure_confusion(cm):
    plt = _mpl()
    cm = np.array(cm)
    plt.figure(figsize=(4.6, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i][j], ha="center", va="center",
                     color="white" if cm[i][j] > cm.max() / 2 else "black")
    plt.xticks(range(3), LABEL_NAMES, rotation=30); plt.yticks(range(3), LABEL_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Figure 4: Confusion matrix (pooled OOS)")
    plt.colorbar(fraction=0.046); plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "fig4_confusion.png"), dpi=130)
    plt.close()


def _figure_roc(oos):
    plt = _mpl()
    from sklearn.metrics import roc_curve, auc
    pos = (oos["y"] == ICTAL).astype(int)
    if pos.sum() == 0 or (pos == 0).sum() == 0:
        return
    fpr, tpr, _ = roc_curve(pos, oos["proba"][:, ICTAL])
    plt.figure(figsize=(5, 4.4))
    plt.plot(fpr, tpr, "b-", lw=2, label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate"); plt.ylabel("Sensitivity")
    plt.title("Figure 5: Seizure-detection ROC (pooled OOS)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "fig5_roc.png"), dpi=130)
    plt.close()


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "observer", "twin"])
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--channels", default="mean", choices=["mean", "multi"],
                    help="mean = paper's channel-averaged image; "
                         "multi = per-channel spatial tensor (improvement)")
    args = ap.parse_args()

    manifest = pp.load_manifest()
    patients = args.patients or sorted({m["patient"] for m in manifest.values()})
    print(f"Patients: {patients}")

    cfg = TrainConfig()
    if args.quick:
        cfg.epochs = 8

    t0 = time.time()
    results = {}
    if args.stage in ("all", "observer"):
        results["observer"] = stage_observer(
            patients, manifest, cfg,
            use_synth=not args.no_synth and not args.quick,
            channel_mode=args.channels)
    if args.stage in ("all", "twin"):
        results["twin"] = stage_twin(manifest, patient=patients[0])

    results["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(ARTIFACT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    _print_summary(results)


def _print_summary(results):
    print("\n" + "=" * 66)
    print("  SUMMARY")
    print("=" * 66)
    obs = results.get("observer", {})
    if obs.get("pooled"):
        b = obs["pooled"]["binary"]; c = obs["pooled"]["three_class"]
        s = obs["pooled"].get("seizure_level", {})
        print(f"  Epoch-level seizure detection (pooled, out-of-sample):")
        print(f"    sensitivity = {b['sensitivity']*100:.1f}%   "
              f"specificity = {b['specificity']*100:.1f}%   "
              f"AUC = {b['auc']:.3f}")
        print(f"    3-class accuracy = {c['accuracy']*100:.1f}%   "
              f"macro-F1 = {c['macro_f1']:.3f}")
        if s:
            print(f"  Seizure-level (temporal post-processing):")
            print(f"    seizure sensitivity = {s['seizure_sensitivity']*100:.1f}% "
                  f"({s['detected']}/{s['n_seizures']} seizures)   "
                  f"false alarms = {s['far_per_hour']:.2f}/h   "
                  f"median latency = {s['median_latency_s']:.0f}s")
        print_confusion(np.array(c["confusion_matrix"]))
    tw = results.get("twin")
    if tw:
        print(f"\n  Digital twin:  B_fit={tw['B_fit']:.1f}  "
              f"B_crit={tw['B_crit']:.2f}  B_target={tw['B_target']:.1f}  "
              f"seizure suppression={tw['suppression_pct']:.1f}%")
    print(f"\n  Runtime: {results['runtime_s']} s")
    print(f"  Figures + results.json in: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
