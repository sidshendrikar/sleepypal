"""
cross_eval.py — ZERO-SHOT cross-population generalization.

Trains a channel-averaged CNN-LSTM on ALL available CHB-MIT (pediatric) epochs,
then evaluates it — with no fine-tuning and no peek at target statistics — on the
adult Siena cohort. The mean-channel representation is electrode-count-agnostic,
so the pediatric-trained model transfers directly to Siena's reconstructed
montage. Source-domain (CHB-MIT) normalization and decision threshold are reused
unchanged, which is what makes this an honest zero-shot test.

Reports the adult AUC (threshold-free headline) plus sensitivity / specificity
and seizure-level detection at a CHB-MIT-chosen operating point.
"""
import os, json, glob
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score

from config import DATA_DIR, ARTIFACT_DIR, TrainConfig, LABELS
from preprocessing import input_tensor
from observer import train_observer, predict_proba
from metrics import tune_threshold, seizure_level_detect, aggregate_seizure_level
from siena import build_siena_dataset

ICTAL = LABELS["ictal"]


def load_all_chbmit_mean():
    """Pool every cached CHB-MIT patient into one channel-averaged train set."""
    Xs, ys = [], []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "chb*_dataset.npz"))):
        d = np.load(p, allow_pickle=True)
        Xs.append(input_tensor(d["X"], "mean")); ys.append(d["y"])
        print(f"  + {os.path.basename(p)}: {len(d['y'])} epochs", flush=True)
    return np.concatenate(Xs), np.concatenate(ys)


def main():
    print("=== ZERO-SHOT: train on CHB-MIT (pediatric) → test on Siena (adult) ===")
    X, y = load_all_chbmit_mean()
    print(f"CHB-MIT train pool: {len(y)} epochs, {int((y==ICTAL).sum())} ictal", flush=True)

    # stratified train / calibration split (calibration picks the threshold)
    rng = np.random.default_rng(0)
    tr, cal = [], []
    for c in range(3):
        idx = np.where(y == c)[0]; rng.shuffle(idx); k = int(len(idx) * 0.15)
        cal += list(idx[:k]); tr += list(idx[k:])
    tr, cal = np.array(tr), np.array(cal)

    cfg = TrainConfig()
    print("training channel-averaged observer on CHB-MIT…", flush=True)
    model, _, norm = train_observer(X[tr], y[tr], cfg)
    pcal = predict_proba(model, X[cal], norm)
    thr = tune_threshold(y[cal], pcal[:, ICTAL], cfg.target_sensitivity)
    print(f"  source-domain threshold (from CHB-MIT calibration): {thr:.3f}", flush=True)

    # ---- Siena (adult) zero-shot ----
    print("building Siena (adult) dataset…", flush=True)
    sd = build_siena_dataset()
    if sd is None:
        print("no Siena data yet"); return
    Xs = input_tensor(sd["X"], "mean")
    ys, tS, fidx = sd["y"], sd["t_start"], sd["file_idx"]
    files, fseiz = sd["files"], sd["file_seizures"]
    ps = predict_proba(model, Xs, norm)          # zero-shot, CHB-MIT normalization

    pos = (ys == ICTAL).astype(int)
    auc = float(roc_auc_score(pos, ps[:, ICTAL])) if pos.any() and (pos == 0).any() else float("nan")
    pred = ps[:, ICTAL] >= thr
    tp = int((pred & (pos == 1)).sum()); fp = int((pred & (pos == 0)).sum())
    tn = int((~pred & (pos == 0)).sum()); fn = int((~pred & (pos == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)

    # seizure-level per recording
    dets = []
    for f in range(len(files)):
        m = fidx == f
        order = np.where(m)[0][np.argsort(tS[m])]
        dets.append(seizure_level_detect(tS[order], ps[order][:, ICTAL],
                    list(fseiz[f]), thr=thr, k_smooth=1, k_persist=2))
    sl = aggregate_seizure_level(dets)

    print("\n=== ADULT SIENA — ZERO-SHOT (model never saw adult data) ===")
    print(f"  epochs={len(ys)}  ictal={int(pos.sum())}  recordings={len(files)}")
    print(f"  epoch: AUC={auc:.3f}  sensitivity={sens*100:.0f}%  specificity={spec*100:.0f}%")
    print(f"  seizure-level: {sl['detected']}/{sl['n_seizures']} detected "
          f"({sl['seizure_sensitivity']*100:.0f}%)  FAR={sl['far_per_hour']:.2f}/h  "
          f"latency={sl['median_latency_s']:.0f}s")

    out = {"chbmit_train_epochs": int(len(y)), "threshold": float(thr),
           "siena_epochs": int(len(ys)), "siena_ictal": int(pos.sum()),
           "siena_recordings": int(len(files)),
           "epoch": {"auc": auc, "sensitivity": sens, "specificity": spec,
                     "tp": tp, "fp": fp, "tn": tn, "fn": fn},
           "seizure_level": sl}
    json.dump(out, open(os.path.join(ARTIFACT_DIR, "results_crossdataset.json"), "w"),
              indent=2, default=float)
    print("\nwrote artifacts/results_crossdataset.json\nCROSS_DONE")


if __name__ == "__main__":
    main()
