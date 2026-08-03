"""
siena.py — Adult Siena Scalp EEG loader for ZERO-SHOT cross-population testing.

Siena (PhysioNet, 14 adults, 47 focal seizures) is recorded in a *referential*
montage ("EEG Fp1", old T3/T4/T5/T6 naming) at 512 Hz — unlike CHB-MIT's
*bipolar* pairs at 256 Hz. This module reconstructs the CHB-MIT longitudinal-
bipolar montage by subtraction (FP1-F7 = Fp1 − F7, …), renames the old
electrodes (T3→T7, T4→T8, T5→P7, T6→P8), resamples to 256 Hz, and produces the
exact dataset format preprocessing.py yields — so a CHB-MIT-trained model can be
evaluated on adult data it never saw.
"""
import os, re, glob
import numpy as np
import mne; mne.set_log_level("ERROR")
from scipy import signal

from config import FS, DATA_DIR, EPOCH_SAMPLES, LABELS
from preprocessing import label_epochs, epoch_logmel_stack, _bandpass_notch, CANONICAL

SIENA_DIR = os.path.join(os.path.dirname(DATA_DIR), "siena")
RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}  # old 10-20 → new
MAX_SEIZURE_S = 180        # clamp implausible/erroneous long annotations


def _clean(name):
    """Referential channel label → bare electrode name in CHB-MIT nomenclature."""
    n = re.sub(r"(?i)eeg", "", name)
    n = n.upper().replace("-REF", "").replace(".", "").replace(" ", "").strip()
    return RENAME.get(n, n)


def load_siena_edf(path):
    """Return (bipolar_data[n_pairs, n_samp] @256Hz filtered, fs, kept_pairs)."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    data = raw.get_data() * 1e6                      # V → µV
    fs = raw.info["sfreq"]
    elec = {}
    for i, c in enumerate(raw.ch_names):
        n = _clean(c)
        if n and n not in elec:
            elec[n] = data[i]
    bip, kept = [], []
    for pair in CANONICAL:                            # e.g. "FP1-F7"
        a, b = pair.split("-")
        if a in elec and b in elec:
            bip.append(elec[a] - elec[b]); kept.append(pair)
    if len(bip) < 15:
        return None, None, kept
    arr = np.asarray(bip, dtype="float64")
    if abs(fs - FS) > 1e-3:                            # 512 → 256 Hz
        arr = signal.resample(arr, int(round(arr.shape[1] * FS / fs)), axis=1)
        fs = FS
    arr = _bandpass_notch(arr, fs)
    return arr.astype("float32"), fs, kept


def _to_sec(t):
    parts = [int(x) for x in re.split(r"[.:]", t.strip()) if x != ""][:3]
    while len(parts) < 3:
        parts.append(0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_siena_annotations(path):
    """Parse Seizures-list-PNxx.txt → {edf_name: [(start_s, end_s), …]} with
    times converted to file-relative seconds (seizure − registration start)."""
    ann = {}
    cur = None; reg = None; regdur = None; pend = None
    for line in open(path, errors="ignore"):
        s = line.strip()
        m = re.match(r"File name:\s*(\S+)", s, re.I)
        if m:
            cur = m.group(1); ann.setdefault(cur, []); reg = None; regdur = None; pend = None; continue
        m = re.search(r"Registration start time:\s*([\d.:]+)", s, re.I)
        if m: reg = _to_sec(m.group(1)); continue
        m = re.search(r"Registration end time:\s*([\d.:]+)", s, re.I)
        if m and reg is not None:
            regdur = _to_sec(m.group(1)) - reg
            if regdur < 0: regdur += 24 * 3600
            continue
        m = re.search(r"Seizure start time:\s*([\d.:]+)", s, re.I)
        if m and reg is not None:
            pend = _to_sec(m.group(1)) - reg
            if pend < 0: pend += 24 * 3600
            continue
        m = re.search(r"Seizure end time:\s*([\d.:]+)", s, re.I)
        if m and cur and pend is not None:
            e = _to_sec(m.group(1)) - reg
            if e < 0: e += 24 * 3600
            # clamp erroneous/over-long annotations
            if regdur:
                e = min(e, regdur)
            if e - pend > MAX_SEIZURE_S or e <= pend:
                e = pend + 90
            ann[cur].append((pend, e)); pend = None
    return ann


def process_siena_file(path, seizures):
    data, fs, kept = load_siena_edf(path)
    if data is None:
        return None
    n_ch, n_samp = data.shape
    starts, labels = label_epochs(n_samp, seizures, fs)
    X, y, tst = [], [], []
    for s0, lab in zip(starts, labels):
        if lab < 0:
            continue
        seg = data[:, s0:s0 + EPOCH_SAMPLES]
        stack = epoch_logmel_stack(seg, fs)
        X.append(np.transpose(stack, (1, 2, 0)).astype("float32"))
        y.append(lab); tst.append(s0 / fs)
    if not X:
        return None
    return {"X": np.array(X), "y": np.array(y), "t_start": np.array(tst),
            "n_pairs": n_ch, "kept": kept}


def build_siena_dataset(subjects=None, cache=True):
    """Aggregate downloaded Siena subjects into a CHB-MIT-shaped dataset dict."""
    cache_path = os.path.join(SIENA_DIR, "siena_dataset.npz")
    if cache and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    if subjects is None:
        subjects = sorted({os.path.basename(p)[:4]
                           for p in glob.glob(os.path.join(SIENA_DIR, "PN*.edf"))})
    Xs, ys, tsts, fidx, files, fseiz = [], [], [], [], [], []
    fi = 0
    for subj in subjects:
        alist = os.path.join(SIENA_DIR, f"Seizures-list-{subj}.txt")
        if not os.path.exists(alist):
            continue
        ann = parse_siena_annotations(alist)
        for edf in sorted(glob.glob(os.path.join(SIENA_DIR, f"{subj}-*.edf"))):
            fn = os.path.basename(edf)
            seiz = ann.get(fn, [])
            res = process_siena_file(edf, seiz)
            if res is None:
                continue
            Xs.append(res["X"]); ys.append(res["y"]); tsts.append(res["t_start"])
            fidx.append(np.full(len(res["y"]), fi))
            files.append(fn); fseiz.append(seiz)
            print(f"    {fn}: {len(res['y'])} epochs "
                  f"(ictal={int((res['y']==1).sum())}, {res['n_pairs']} pairs)", flush=True)
            fi += 1
    if not Xs:
        return None
    ds = {"X": np.concatenate(Xs), "y": np.concatenate(ys),
          "t_start": np.concatenate(tsts), "file_idx": np.concatenate(fidx),
          "files": np.array(files, dtype=object),
          "file_seizures": np.array(fseiz, dtype=object), "patient": "siena"}
    if cache:
        np.savez_compressed(cache_path, **ds)
    return ds


if __name__ == "__main__":
    import sys
    subs = sys.argv[1:] or None
    ann = parse_siena_annotations(os.path.join(SIENA_DIR, "Seizures-list-PN00.txt"))
    print("PN00 parsed annotations (file-relative sec):")
    for k, v in ann.items():
        print(f"  {k}: {v}")
    ds = build_siena_dataset(subs, cache=False)
    if ds:
        y = ds["y"]
        print(f"\nSiena dataset: {len(y)} epochs, ictal={int((y==1).sum())}, "
              f"preictal={int((y==2).sum())}, interictal={int((y==0).sum())}, "
              f"files={len(ds['files'])}")
        print("X shape:", ds["X"].shape)
