"""
improve_eval.py — trains both channel views per fold, caches their held-out
probabilities, then evaluates mean / multi / ENSEMBLE offline.

The two views are complementary (mean = sensitive, multi = specific), so the
ensemble averages their P(ictal). Probabilities are persisted to
artifacts/oos_probs.npz so threshold / temporal-parameter analysis costs
nothing extra afterwards.
"""
import os, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# allow running several patients concurrently without oversubscribing the CPU
_TH = int(os.environ.get("TF_THREADS", "0"))
if _TH:
    os.environ["OMP_NUM_THREADS"] = str(_TH)
import warnings; warnings.filterwarnings("ignore")
import numpy as np
if _TH:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(_TH)
    tf.config.threading.set_inter_op_parallelism_threads(2)

from config import TrainConfig, ARTIFACT_DIR, LABELS
import preprocessing as pp
from preprocessing import input_tensor
from observer import train_observer, predict_proba
from metrics import tune_threshold

ICTAL = LABELS["ictal"]
PATIENT = os.environ.get("PATIENT", "chb01")


def main():
    man = pp.load_manifest()
    ds = pp.build_patient_dataset(PATIENT, man)
    y, fidx, t = ds["y"], ds["file_idx"], ds["t_start"]
    files, fseiz = ds["files"], ds["file_seizures"]
    seizure_files = [i for i in range(len(files)) if len(fseiz[i]) > 0]
    print(f"{PATIENT}: {len(y)} epochs, {int((y==ICTAL).sum())} ictal, "
          f"{len(seizure_files)} seizure recordings", flush=True)

    Xv = {m: input_tensor(ds["X"], m) for m in ["mean", "multi"]}
    cfg = TrainConfig()

    rec = []          # one entry per held-out recording
    t0 = time.time()
    for k, f in enumerate(seizure_files):
        test = fidx == f
        train = ~test
        idx = np.where(test)[0]
        order = idx[np.argsort(t[idx])]
        entry = {"file": str(files[f]), "y": y[order], "t": t[order],
                 "seizures": [list(s) for s in fseiz[f]]}
        for m in ["mean", "multi"]:
            model, _, norm = train_observer(Xv[m][train], y[train], cfg)
            ptr = predict_proba(model, Xv[m][train], norm)
            thr = tune_threshold(y[train], ptr[:, ICTAL], cfg.target_sensitivity)
            pte = predict_proba(model, Xv[m][order], norm)
            entry[f"p_{m}"] = pte[:, ICTAL]
            entry[f"thr_{m}"] = float(thr)
            entry[f"proba3_{m}"] = pte
        rec.append(entry)
        print(f"  [{k+1}/{len(seizure_files)}] {files[f]} done "
              f"({time.time()-t0:.0f}s)", flush=True)

    # persist everything for free offline analysis (one file per patient)
    out = os.path.join(ARTIFACT_DIR, f"oos_probs_{PATIENT}.npz")
    np.savez_compressed(out, recs=np.array(rec, dtype=object), allow_pickle=True)
    print(f"saved {out}", flush=True)
    print("PROBS_DONE", flush=True)


if __name__ == "__main__":
    main()
