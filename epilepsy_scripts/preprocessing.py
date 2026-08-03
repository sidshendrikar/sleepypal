"""
preprocessing.py — The EEG front end (Sections 2.2-2.3 of the paper).

Turns raw CHB-MIT EDF recordings into labeled log-Mel spectrogram tensors for
the observer network, and into raw epoch signals for the digital-twin fit.

Pipeline per file:
  EDF -> canonical bipolar channels -> 4th-order Butterworth band-pass
      -> 60 Hz notch -> resample to 256 Hz -> 5 s epochs
      -> {interictal, preictal, ictal} labels
      -> per-epoch log-Mel spectrogram (64x64), channel-averaged in log-power.
"""
import json
import os
import glob
import numpy as np
from scipy import signal
from scipy.ndimage import zoom

import mne
mne.set_log_level("ERROR")

from config import (DATA_DIR, FS, BP_LOW, BP_HIGH, BP_ORDER, NOTCH,
                    EPOCH_SAMPLES, EPOCH_S, PREICTAL_S, INTERICTAL_GUARD_S,
                    STFT_WIN, STFT_HOP, N_MELS, MEL_FMIN, MEL_FMAX, IMG_SIZE,
                    LABELS)

# Canonical longitudinal-bipolar montage shared across CHB-MIT patients.
CANONICAL = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3",
             "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8",
             "T8-P8", "FZ-CZ", "CZ-PZ", "P7-T7"]


# ----------------------------------------------------------------------------
# Mel filterbank (built from the paper's formula, no librosa dependency)
# ----------------------------------------------------------------------------
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(n_fft, fs, n_mels=N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX):
    """Triangular mel filterbank matrix of shape (n_mels, n_fft//2+1)."""
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, fs / 2, n_bins)
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / fs).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    fb = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        r = min(r, n_bins - 1)
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


_FB = _mel_filterbank(STFT_WIN, FS)


def _resize64(logmel):
    zoom_f = (IMG_SIZE / logmel.shape[0], IMG_SIZE / max(logmel.shape[1], 1))
    img = zoom(logmel, zoom_f, order=1)[:IMG_SIZE, :IMG_SIZE]
    if img.shape != (IMG_SIZE, IMG_SIZE):   # pad if a hair short
        out = np.zeros((IMG_SIZE, IMG_SIZE), dtype=img.dtype)
        out[:img.shape[0], :img.shape[1]] = img
        img = out
    return img.astype("float32")


def log_mel_spectrogram(sig, fs=FS):
    """Single-channel signal -> (IMG_SIZE, IMG_SIZE) log-Mel image."""
    _, _, Z = signal.stft(sig, fs=fs, window="hann", nperseg=STFT_WIN,
                          noverlap=STFT_WIN - STFT_HOP, boundary=None,
                          padded=False)
    mel = _FB @ (np.abs(Z) ** 2)            # (n_mels, frames)
    return _resize64(np.log(mel + 1e-6))


def epoch_logmel_stack(seg, fs=FS):
    """(n_ch, n_samp) segment -> (n_ch, IMG_SIZE, IMG_SIZE) per-channel log-Mel
    stack. One vectorized STFT for all channels."""
    _, _, Z = signal.stft(seg, fs=fs, window="hann", nperseg=STFT_WIN,
                          noverlap=STFT_WIN - STFT_HOP, boundary=None,
                          padded=False, axis=-1)          # (n_ch, n_bins, frames)
    S = np.abs(Z) ** 2
    mel = np.einsum("mf,cft->cmt", _FB, S)                # (n_ch, n_mels, frames)
    logmel = np.log(mel + 1e-6)
    return np.stack([_resize64(logmel[c]) for c in range(logmel.shape[0])])


def log_mel_multichannel(seg, fs=FS):
    """Channel-averaged (IMG_SIZE, IMG_SIZE) log-Mel image (paper's 1-channel)."""
    return epoch_logmel_stack(seg, fs).mean(axis=0)


# ----------------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------------
def _bandpass_notch(data, fs=FS):
    """4th-order Butterworth band-pass + 60 Hz notch, zero-phase, per channel."""
    ny = fs / 2.0
    b, a = signal.butter(BP_ORDER, [BP_LOW / ny, BP_HIGH / ny], btype="band")
    out = signal.filtfilt(b, a, data, axis=-1)
    w0 = NOTCH / ny
    if w0 < 1.0:
        bn, an = signal.iirnotch(w0, Q=30.0)
        out = signal.filtfilt(bn, an, out, axis=-1)
    return out


def _standardize_names(names):
    return [n.strip().upper().replace(" ", "") for n in names]


def load_edf(path):
    """Return (data[n_ch, n_samp], fs) using the canonical montage."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw.rename_channels({c: c.strip().upper().replace(" ", "")
                         for c in raw.ch_names})
    present = [c for c in CANONICAL if c in raw.ch_names]
    if len(present) < 15:
        return None, None
    raw.pick(present)
    data = raw.get_data() * 1e6             # V -> uV
    fs = raw.info["sfreq"]
    if abs(fs - FS) > 1e-3:                  # resample to canonical rate
        n_out = int(round(data.shape[1] * FS / fs))
        data = signal.resample(data, n_out, axis=1)
        fs = FS
    data = _bandpass_notch(data, fs)
    return data.astype("float32"), fs


# ----------------------------------------------------------------------------
# Epoching + labeling
# ----------------------------------------------------------------------------
def label_epochs(n_samples, seizures, fs=FS, epoch_samples=EPOCH_SAMPLES):
    """Return (starts, labels). label -1 marks epochs to DROP (guard band)."""
    starts = np.arange(0, n_samples - epoch_samples + 1, epoch_samples)
    labels = np.full(len(starts), LABELS["interictal"], dtype=int)
    for j, s0 in enumerate(starts):
        t0 = s0 / fs
        t1 = (s0 + epoch_samples) / fs
        lab = LABELS["interictal"]
        drop = False
        for (sz_start, sz_end) in seizures:
            if t1 > sz_start and t0 < sz_end:                 # overlaps seizure
                lab = LABELS["ictal"]; break
            if sz_start - PREICTAL_S <= t0 < sz_start:        # 5-min pre-ictal
                lab = LABELS["preictal"]
            # guard band: near a seizure but not clearly pre/ictal -> drop
            if (sz_start - INTERICTAL_GUARD_S <= t0 < sz_start - PREICTAL_S) or \
               (sz_end < t1 <= sz_end + INTERICTAL_GUARD_S):
                if lab == LABELS["interictal"]:
                    drop = True
        labels[j] = -1 if drop else lab
    return starts, labels


def process_file(path, seizures):
    """Return dict with spectrograms X, labels y, raw epochs, per file."""
    data, fs = load_edf(path)
    if data is None:
        return None
    n_ch, n_samp = data.shape
    starts, labels = label_epochs(n_samp, seizures, fs)
    X, y, raw_epochs, tstart = [], [], [], []
    for s0, lab in zip(starts, labels):
        if lab < 0:
            continue
        seg = data[:, s0:s0 + EPOCH_SAMPLES]                 # (n_ch, 1280)
        # per-channel log-mel stack, channels-last -> (64,64,n_ch). The mean
        # (paper's single channel) is derived downstream via input_tensor().
        stack = epoch_logmel_stack(seg, fs)                  # (n_ch,64,64)
        X.append(np.transpose(stack, (1, 2, 0)).astype("float32"))
        y.append(lab)
        tstart.append(s0 / fs)                               # epoch start (s)
        # raw epoch = channel mean signal, for twin PSD fitting
        raw_epochs.append(seg.mean(axis=0).astype("float32"))
    if not X:
        return None
    return {"X": np.array(X), "y": np.array(y),
            "raw": np.array(raw_epochs), "t_start": np.array(tstart), "fs": fs}


def build_patient_dataset(patient, manifest, cache=True):
    """Aggregate all downloaded files for one patient into a dataset dict.
    Adds a `seizure_id` per epoch (which seizure file it came from) so the
    observer can be evaluated with leave-one-seizure-out splits."""
    cache_path = os.path.join(DATA_DIR, f"{patient}_dataset.npz")
    if cache and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    files = sorted(fn for fn, meta in manifest.items() if meta["patient"] == patient)
    Xs, ys, raws, sids, fidx, tsts = [], [], [], [], [], []
    kept_files, file_seizures = [], []
    for i, fn in enumerate(files):
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            continue
        seiz = manifest[fn]["seizures"]
        res = process_file(path, seiz)
        if res is None:
            continue
        Xs.append(res["X"]); ys.append(res["y"]); raws.append(res["raw"])
        tsts.append(res["t_start"])
        # tag ictal/preictal epochs with the file index as the "seizure event"
        sids.append(np.full(len(res["y"]), i if seiz else -1))
        fidx.append(np.full(len(res["y"]), i))     # source file for every epoch
        kept_files.append(fn); file_seizures.append(seiz)
        print(f"    {fn}: {len(res['y'])} epochs "
              f"(ictal={int((res['y']==1).sum())}, "
              f"preictal={int((res['y']==2).sum())}, "
              f"interictal={int((res['y']==0).sum())})", flush=True)
    if not Xs:
        return None
    ds = {"X": np.concatenate(Xs), "y": np.concatenate(ys),
          "raw": np.concatenate(raws), "seizure_id": np.concatenate(sids),
          "file_idx": np.concatenate(fidx), "t_start": np.concatenate(tsts),
          "files": np.array(kept_files, dtype=object),
          "file_seizures": np.array(file_seizures, dtype=object),
          "patient": patient}
    if cache:
        np.savez_compressed(cache_path, **ds)
    return ds


def input_tensor(X, mode="mean"):
    """Convert a per-channel dataset (N,64,64,C) to the observer input.
    mode="mean": paper's channel-averaged single image (N,64,64,1).
    mode="multi": keep all channels as spatial input (N,64,64,C)."""
    if X.ndim == 4 and X.shape[-1] == 1:
        return X                       # already single channel (legacy cache)
    if mode == "mean":
        return X.mean(axis=-1, keepdims=True)
    return X


def load_manifest():
    with open(os.path.join(DATA_DIR, "seizures.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    man = load_manifest()
    patients = sorted({m["patient"] for m in man.values()})
    print("Patients:", patients)
    ds = build_patient_dataset(patients[0], man, cache=False)
    if ds:
        y = ds["y"]
        print(f"\n{ds['patient']}: {len(y)} epochs  "
              f"interictal={int((y==0).sum())} ictal={int((y==1).sum())} "
              f"preictal={int((y==2).sum())}")
        print("X shape:", ds["X"].shape)
