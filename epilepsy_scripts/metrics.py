"""
metrics.py — Evaluation utilities.

Reports the two views that matter for this project:
  * Binary seizure detection (ictal vs. non-ictal) — the standard CHB-MIT
    benchmark framing (Shoeb 2009). This is where the paper's 12.5% ictal
    sensitivity is measured and fixed.
  * Three-class state estimation (interictal / ictal / preictal) — the paper's
    reported "overall accuracy".
"""
import numpy as np
from sklearn.metrics import (roc_auc_score, confusion_matrix, f1_score,
                             precision_score)

from config import LABELS, LABEL_NAMES

ICTAL = LABELS["ictal"]


def tune_threshold(y_true, proba_ictal, target_sensitivity=0.90):
    """Choose the P(ictal) threshold on validation that achieves at least
    `target_sensitivity` while maximizing specificity."""
    pos = y_true == ICTAL
    if pos.sum() == 0:
        return 0.5
    thresholds = np.unique(np.concatenate([[0.0, 1.0], proba_ictal]))
    best_t, best_spec = 0.5, -1.0
    for t in thresholds:
        pred = proba_ictal >= t
        sens = (pred & pos).sum() / max(pos.sum(), 1)
        spec = (~pred & ~pos).sum() / max((~pos).sum(), 1)
        if sens >= target_sensitivity and spec > best_spec:
            best_spec, best_t = spec, t
    return float(best_t)


def seizure_metrics(y_true, proba, threshold=0.5):
    """Binary ictal-vs-rest metrics from 3-class softmax probabilities."""
    pos = y_true == ICTAL
    p_ictal = proba[:, ICTAL]
    pred = p_ictal >= threshold
    tp = int((pred & pos).sum()); fp = int((pred & ~pos).sum())
    tn = int((~pred & ~pos).sum()); fn = int((~pred & pos).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    try:
        auc = roc_auc_score(pos.astype(int), p_ictal) if pos.any() and (~pos).any() else float("nan")
    except ValueError:
        auc = float("nan")
    return {"sensitivity": sens, "specificity": spec, "precision": prec,
            "f1": f1, "auc": auc, "fpr": fp / max(fp + tn, 1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "threshold": threshold}


def three_class_report(y_true, y_pred):
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    recalls = {}
    for c, name in enumerate(LABEL_NAMES):
        mask = y_true == c
        recalls[name] = float((y_pred[mask] == c).mean()) if mask.any() else float("nan")
    macro_f1 = float(f1_score(y_true, y_pred, labels=[0, 1, 2],
                              average="macro", zero_division=0))
    return {"accuracy": acc, "confusion_matrix": cm.tolist(),
            "recalls": recalls, "macro_f1": macro_f1}


def temporal_smooth(p, k=3):
    """Moving-average smoothing of a probability sequence over k epochs."""
    if k <= 1:
        return np.asarray(p, float)
    ker = np.ones(k) / k
    return np.convolve(np.asarray(p, float), ker, mode="same")


def seizure_level_detect(t_start, p_ictal, seizures, epoch_s=5.0, thr=0.5,
                         k_smooth=3, k_persist=2, guard_s=30.0):
    """Turn ordered epoch probabilities into seizure-level detections.

    Applies moving-average smoothing then a persistence rule (>= k_persist
    consecutive epochs above threshold) to raise an alarm. Reports, for one
    recording: which true seizures were caught, detection latency vs. onset,
    number of false-alarm clusters, and the recording duration in hours.
    """
    t = np.asarray(t_start, float)
    order = np.argsort(t)
    t = t[order]
    p = temporal_smooth(np.asarray(p_ictal, float)[order], k_smooth)

    alarm = p >= thr
    fired = np.zeros(len(alarm), bool)
    run = 0
    for i, a in enumerate(alarm):
        run = run + 1 if a else 0
        if run >= k_persist:
            fired[np.arange(i - k_persist + 1, i + 1)] = True

    # cluster fired epochs into events (<=2-epoch gaps merge)
    fi = np.where(fired)[0]
    events = []
    if len(fi):
        grp = [fi[0]]
        for j in fi[1:]:
            if j - grp[-1] <= 2:
                grp.append(j)
            else:
                events.append(grp); grp = [j]
        events.append(grp)

    detected, latencies, matched = 0, [], set()
    for (s, e) in seizures:
        for ei, g in enumerate(events):
            if ei in matched:
                continue
            if any((t[k] + epoch_s >= s - guard_s) and (t[k] <= e + guard_s)
                   for k in g):
                detected += 1
                latencies.append(float(t[g[0]] - s))
                matched.add(ei)
                break
    false_alarms = len([ei for ei in range(len(events)) if ei not in matched])
    hours = (t[-1] - t[0] + epoch_s) / 3600.0 if len(t) > 1 else epoch_s / 3600.0
    return {"n_seizures": len(seizures), "detected": detected,
            "latencies": latencies, "false_alarms": false_alarms,
            "hours": hours}


def aggregate_seizure_level(dets):
    """Combine per-recording detections into patient/cohort numbers."""
    n_sz = sum(d["n_seizures"] for d in dets)
    det = sum(d["detected"] for d in dets)
    fa = sum(d["false_alarms"] for d in dets)
    hours = sum(d["hours"] for d in dets)
    lat = [x for d in dets for x in d["latencies"]]
    return {"seizure_sensitivity": det / max(n_sz, 1),
            "n_seizures": n_sz, "detected": det,
            "false_alarms": fa, "hours": hours,
            "far_per_hour": fa / max(hours, 1e-9),
            "median_latency_s": float(np.median(lat)) if lat else float("nan")}


def print_confusion(cm):
    print("               pred:int  ict  pre")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  true {name:11s} {cm[i][0]:5d} {cm[i][1]:4d} {cm[i][2]:4d}")
