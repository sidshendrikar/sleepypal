"""
stim_calib.py — partial calibration of the stimulation->inhibition coupling
against REAL stimulation-paired intracranial EEG (OpenNeuro ds006519).

WHAT THIS CAN AND CANNOT DO
---------------------------
ds006519 delivers 43.2/50 Hz bipolar stimulation trains (0.25-3 mA, up to 5 s)
during stereo-EEG functional mapping, and ships matched 5 s PRE- and
POST-stimulation epochs. That makes it the only open dataset that measures a
SUSTAINED state change as a function of stimulation dose -- which is the
quantity our coupling describes. It is NOT therapeutic 130 Hz DBS, so this
calibrates the mechanism at 50 Hz and is a LOWER-FREQUENCY ANALOG, not the
therapeutic coupling itself.

THE CALIBRATION CHAIN
---------------------
    data  :  dF/dI   -- how much a spectral feature F shifts per mA of stimulation
    model :  dF/dB   -- how much the same feature shifts per mV of inhibitory gain
    ==>      dB/dI = (dF/dI) / (dF/dB)      [mV of effective inhibition per mA]

F is measured identically on real and simulated signals. Two features are used:
  * low-band relative power (the model's oscillation band), and
  * the aperiodic 1/f exponent, an established EEG proxy for E:I ratio
    (Gao, Peterson & Voytek 2017) -- steeper (more negative) = more inhibition.

Line noise and the stimulation itself are both at 50 Hz, so every spectral fit
avoids 50 Hz and its harmonics.
"""
import glob, json, os, re
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy import signal as sps

import mne
mne.set_log_level("ERROR")

from config import ARTIFACT_DIR

STIM_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "stim")

LOW_BAND = (4.0, 12.0)        # model-comparable low band (below 50 Hz stim/line)
APER_BAND = (20.0, 45.0)      # aperiodic fit band, safely below 50 Hz


# ---------------------------------------------------------------------------
# spectral features (identical for real and simulated signals)
# ---------------------------------------------------------------------------
def psd(x, fs, nperseg=None):
    nperseg = nperseg or min(len(x), int(fs * 2))
    f, P = sps.welch(x - np.mean(x), fs=fs, nperseg=nperseg,
                     noverlap=nperseg // 2)
    return f, P


def features(x, fs):
    """Return (relative low-band power, aperiodic exponent)."""
    f, P = psd(x, fs)
    ok = (f > 1.0) & (f < 45.0)
    tot = np.trapz(P[ok], f[ok]) + 1e-30
    m = (f >= LOW_BAND[0]) & (f <= LOW_BAND[1])
    rel_low = float(np.trapz(P[m], f[m]) / tot)
    a = (f >= APER_BAND[0]) & (f <= APER_BAND[1]) & (P > 0)
    if a.sum() >= 8:
        sl = np.polyfit(np.log10(f[a]), np.log10(P[a]), 1)[0]
    else:
        sl = np.nan
    return rel_low, float(sl)


# ---------------------------------------------------------------------------
# data side
# ---------------------------------------------------------------------------
def _load_tsv(path):
    rows = []
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    return rows


def _stim_electrodes(stim_contacts):
    """'A'09-A'10' -> the electrode tokens to exclude from analysis."""
    if not stim_contacts or stim_contacts == "n/a":
        return []
    return [t.strip() for t in stim_contacts.split("-") if t.strip()]


def analyze_run(vhdr):
    """Return a list of per-trial dicts for one recording."""
    base = vhdr[:-len("_ieeg.vhdr")]
    ev_p, ch_p = base + "_events.tsv", base + "_channels.tsv"
    if not (os.path.exists(ev_p) and os.path.exists(ch_p)):
        return []
    events, chans = _load_tsv(ev_p), _load_tsv(ch_p)
    try:
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    except Exception:
        return []
    fs = raw.info["sfreq"]
    good = {c["name"] for c in chans
            if c.get("status", "good") == "good" and c.get("type") == "SEEG"}
    names = [n for n in raw.ch_names if n in good]
    if not names:
        return []
    data = raw.get_data(picks=[raw.ch_names.index(n) for n in names])

    pres = [e for e in events if e.get("label") == "prestim"]
    posts = [e for e in events if e.get("label") == "poststim"]
    out = []
    for k, post in enumerate(posts):
        try:
            inten = float(post.get("stim_intensity", "n/a"))
            freq = float(post.get("stim_frequency", "n/a"))
        except (TypeError, ValueError):
            continue
        pre = pres[k] if k < len(pres) else (pres[-1] if pres else None)
        if pre is None:
            continue
        # exclude channels touching the stimulated contacts (artifact)
        excl = _stim_electrodes(post.get("stim_contacts", ""))
        keep = [i for i, n in enumerate(names)
                if not any(tok and tok in n for tok in excl)]
        if len(keep) < 5:
            continue

        def seg(ev):
            t0 = float(ev["onset"]); dur = float(ev["duration"] or 5.0)
            a, b = int(t0 * fs), int((t0 + dur) * fs)
            return data[:, max(0, a):min(b, data.shape[1])]

        A, B_ = seg(pre), seg(post)
        if A.shape[1] < fs or B_.shape[1] < fs:
            continue
        d_low, d_slope = [], []
        for i in keep:
            l0, s0 = features(A[i], fs)
            l1, s1 = features(B_[i], fs)
            if np.isfinite(l0) and np.isfinite(l1) and l0 > 0 and l1 > 0:
                d_low.append(np.log(l1 / l0))
            if np.isfinite(s0) and np.isfinite(s1):
                d_slope.append(s1 - s0)
        if not d_low:
            continue
        out.append({
            "file": os.path.basename(vhdr),
            "intensity_mA": inten, "freq_hz": freq,
            "effect": post.get("clinical_effect", "n/a"),
            "n_channels": len(keep),
            "d_log_lowpower": float(np.median(d_low)),
            "d_aperiodic": float(np.median(d_slope)) if d_slope else np.nan,
        })
    return out


def collect():
    vhdrs = sorted(glob.glob(os.path.join(STIM_DIR, "sub-*", "ses-*", "ieeg",
                                          "*_task-dcs_ieeg.vhdr")))
    trials = []
    for v in vhdrs:
        trials.extend(analyze_run(v))
    return trials


# ---------------------------------------------------------------------------
# model side: how does the same feature move with the inhibitory gain B?
# ---------------------------------------------------------------------------
def model_sensitivity(B_lo=30.0, B_hi=46.0, n=17, T=8.0, seeds=(1, 2, 3)):
    """dF/dB evaluated over the NON-seizing operating range (the patients in
    ds006519 are interictal during functional mapping, so the model must be
    compared at its corresponding quiet/alpha operating point, not in the
    limit-cycle band)."""
    from jansen_ritt import simulate
    from simulator import resample_to_eeg
    from config import FS
    Bs = np.linspace(B_lo, B_hi, n)
    lows, slopes = [], []
    for B in Bs:
        L, S = [], []
        for s in seeds:
            _, V = simulate(B=float(B), T=T, method="euler", seed=s)
            x = resample_to_eeg(V)
            l, sl = features(x, FS)
            L.append(l); S.append(sl)
        lows.append(np.mean(L)); slopes.append(np.nanmean(S))
    lows, slopes = np.array(lows), np.array(slopes)
    # d(log low-power)/dB and d(aperiodic)/dB by linear fit
    ok = lows > 0
    d_low = float(np.polyfit(Bs[ok], np.log(lows[ok]), 1)[0])
    ok2 = np.isfinite(slopes)
    d_slope = float(np.polyfit(Bs[ok2], slopes[ok2], 1)[0]) if ok2.sum() > 3 else np.nan
    return {"B": Bs.tolist(), "rel_low": lows.tolist(), "aperiodic": slopes.tolist(),
            "dlogLow_dB": d_low, "dAper_dB": d_slope}


# ---------------------------------------------------------------------------
def main():
    trials = collect()
    print(f"trials with usable pre/post epochs: {len(trials)}")
    if not trials:
        print("NO_TRIALS"); return
    I = np.array([t["intensity_mA"] for t in trials])
    dL = np.array([t["d_log_lowpower"] for t in trials])
    dA = np.array([t["d_aperiodic"] for t in trials])
    print(f"intensity range: {I.min():.2f}-{I.max():.2f} mA "
          f"({len(np.unique(I))} distinct levels)")
    print(f"frequencies: {sorted(set(t['freq_hz'] for t in trials))}")

    def fit(y, name):
        ok = np.isfinite(y) & np.isfinite(I)
        if ok.sum() < 5:
            print(f"  {name}: too few points"); return None
        x = I[ok]; yy = y[ok]
        slope, icpt = np.polyfit(x, yy, 1)
        r = np.corrcoef(x, yy)[0, 1]
        n = ok.sum()
        # slope standard error + permutation p
        yhat = slope * x + icpt
        se = np.sqrt(np.sum((yy - yhat) ** 2) / (n - 2) /
                     np.sum((x - x.mean()) ** 2))
        rng = np.random.default_rng(0)
        null = [abs(np.polyfit(rng.permutation(x), yy, 1)[0]) for _ in range(2000)]
        p = float(np.mean(np.array(null) >= abs(slope)))
        print(f"  {name}: slope={slope:+.4f} per mA (SE {se:.4f}), "
              f"r={r:+.3f}, n={n}, perm p={p:.3f}")
        return {"slope": float(slope), "se": float(se), "r": float(r),
                "n": int(n), "p": p}

    print("\n[data] sustained state change vs stimulation intensity:")
    f_low = fit(dL, "d log low-band power")
    f_aper = fit(dA, "d aperiodic exponent")

    print("\n[model] feature sensitivity to inhibitory gain B (quiet regime):")
    ms = model_sensitivity()
    print(f"  d(log low-power)/dB = {ms['dlogLow_dB']:+.4f} per mV")
    print(f"  d(aperiodic)/dB     = {ms['dAper_dB']:+.4f} per mV")

    print("\n[calibration] dB/dI = (dF/dI)/(dF/dB):")
    res = {"n_trials": len(trials), "model": ms,
           "fit_lowpower": f_low, "fit_aperiodic": f_aper, "trials": trials}
    for nm, fd, dfdb in [("low-band power", f_low, ms["dlogLow_dB"]),
                         ("aperiodic exponent", f_aper, ms["dAper_dB"])]:
        if fd and np.isfinite(dfdb) and abs(dfdb) > 1e-9:
            dBdI = fd["slope"] / dfdb
            lo = (fd["slope"] - 1.96 * fd["se"]) / dfdb
            hi = (fd["slope"] + 1.96 * fd["se"]) / dfdb
            g3 = dBdI * 3.0
            print(f"  via {nm:20s} dB/dI = {dBdI:+6.2f} mV/mA "
                  f"[95% CI {min(lo,hi):+.2f},{max(lo,hi):+.2f}]  "
                  f"-> gain at 3 mA = {g3:+.1f} mV")
            res[f"dBdI_{nm.split()[0]}"] = {"dBdI_mV_per_mA": float(dBdI),
                                            "gain_at_3mA_mV": float(g3),
                                            "ci": [float(min(lo, hi)), float(max(lo, hi))]}
    print(f"\n  (inversion-from-clinical-dose estimate was 22 mV at 3 mA "
          f"= 7.3 mV/mA)")
    json.dump(res, open(os.path.join(ARTIFACT_DIR, "stim_calibration.json"), "w"),
              indent=2, default=float)
    print("CALIB_DONE")


if __name__ == "__main__":
    main()
