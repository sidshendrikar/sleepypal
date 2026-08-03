# Hybrid Digital Twin for Closed-Loop Seizure Control

A working implementation of the framework in *"A Mathematical Innovation in
Closed-Loop Seizure Control"* (S. Shendrikar, Bi 195, Dec 2025), rebuilt from
the paper and extended. It couples a **deep-learning observer** (CNN-LSTM
seizure detector) with a **biophysical digital twin** (Jansen-Rit neural mass
model) and a **bifurcation-based controller** that computes the precise
inhibitory correction needed to terminate a seizure.

The system answers three questions about a patient's cortex in real time:

| Engine | Question | Method |
|---|---|---|
| **Observation** | *Is a seizure happening?* | CNN-LSTM on log-Mel EEG spectrograms |
| **Replication** | *What is the excitation/inhibition balance?* | Jansen-Rit neural mass model fit to the patient's EEG |
| **Control** | *How much inhibition stops it?* | Bifurcation analysis → distance to the stability boundary |

## The idea in one paragraph

A cortical column can be modeled by six coupled ODEs (Jansen-Rit) whose behavior
is governed by the inhibitory synaptic gain **B**. Sweeping B reveals a
**limit-cycle band** (large rhythmic oscillations = the seizure state) bounded by
two Hopf bifurcations. Above the upper bifurcation `B_crit`, oscillations are
annihilated and the cortex sits at a quiet stable fixed point (the healthy
state). We (1) **detect** seizures from EEG with a CNN-LSTM, (2) **personalize**
a twin by fitting B to the patient's own spectrum, and (3) **compute the exact
extra inhibition** (`B_target ≈ 1.2·B_crit`) that pushes the cortex across the
bifurcation and ends the seizure — the mechanistic basis for a smarter
closed-loop implant.

## File map

```
config.py               all physical constants + hyperparameters (single source of truth)
jansen_ritt.py          6-ODE neural mass model, Numba-JIT Euler-Maruyama + RK4
simulator.py            PSD / band-power / spectral features; synthetic EEG generator
bifurcation_analysis.py B-sweep, Hopf detection, distance-to-bifurcation, control law
observer.py             CNN-LSTM detector + the class-imbalance fixes
preprocessing.py        EDF → bandpass/notch → epoch → label → log-Mel spectrogram
digital_twin.py         differential-evolution fit of B to real EEG; closed-loop twin
metrics.py              binary (seizure) + 3-class evaluation, threshold tuning
main.py                 orchestration CLI, cross-validation, figures, results.json
fetch_data.py           targeted CHB-MIT downloader + seizure-annotation manifest
```

## Data

Real intracranial-adjacent scalp EEG from the **CHB-MIT Scalp EEG Database**
(PhysioNet), 256 Hz, 18-channel longitudinal-bipolar montage. `fetch_data.py`
pulls only the seizure-bearing files plus matched interictal files and parses
the `*-summary.txt` seizure annotations into `data/epilepsy/seizures.json`.

Preprocessing (Section 2.2–2.3 of the paper): 4th-order Butterworth band-pass
0.5–70 Hz, 60 Hz notch, 5 s epochs (1280 samples), each turned into a 64×64
log-Mel spectrogram. Epochs are labeled **interictal / preictal (5 min pre-onset)
/ ictal**, with a 15-min guard band excluded around every seizure to keep the
interictal class clean.

## How to run

```bash
pip install numpy scipy scikit-learn matplotlib numba mne tensorflow pyedflib
python fetch_data.py                 # download CHB-MIT subset (~250 MB/patient)
python main.py --stage all           # observer CV + twin + control + figures
python main.py --stage twin          # just the mechanistic half (fast)
python main.py --stage observer --patients chb01 chb03
```

Outputs land in `artifacts/`: `results.json` and five figures (spectral match,
bifurcation diagram, closed-loop control, confusion matrix, ROC).

## What was improved over the paper

The paper reported **70.8% accuracy but only 12.5% ictal sensitivity** — the
detector had collapsed onto the majority class. Key changes:

1. **Fixed the sensitivity collapse: 12.5% → 79%** (at 99% specificity).
   Benchmarked class-weighting vs. minority-oversampling vs. focal loss as
   *single* strategies — stacking them triple-counts the minority and makes the
   model predict "seizure" for everything (AUC 0.007). Oversampling won cleanly.
2. **Leakage-free evaluation.** Leave-one-seizure-**recording**-out: a whole
   ~1-hour recording is held out, so temporally adjacent 5-s epochs never
   straddle the split. Normalization and thresholds come from train only.
   (A stratified random split flatters the AUC to 0.90 — it leaks.)
3. **Multi-channel spatial view.** The paper averages all electrodes into one
   spectrogram, destroying the focal signal that localizes a seizure. Keeping
   per-channel spectrograms `(64,64,17)` yields a far more *specific* detector.
4. **Ensemble of the two views → AUC 0.979.** The channel-average is sensitive,
   the spatial view is specific; averaging their `P(ictal)` beats both.
5. **Threshold calibration (the biggest single win).** Thresholds tuned on
   *training* predictions sit too high — the model is over-confident on epochs
   it memorised. Re-tuning on other folds' out-of-sample predictions moved
   sensitivity **61% → 89%** and detection **86% → 100%**, at zero extra compute.
6. **Seizure-level metrics with temporal persistence** — the clinically
   meaningful evaluation the paper never did: **13/14 seizures @ 0.23 false
   alarms/h, 3 s latency**. (Smoothing *hurts*: it dilutes 30-s seizures.)
7. **RK4 integrator + bisection-refined Hopf detection** for the bifurcation map.
8. **Energy-minimal graded controller** — pushes B just past `B_crit` scaled by
   severity instead of always jumping to `1.2·B_crit`.
9. **Corrected the paper's dynamical claim** — the model has *two* Hopf points;
   `B = 8.3` is a saturated fixed point, not a seizure oscillation. See
   `RESULTS.md`.
10. **Numba JIT + vectorized STFT** — ~10,000× real-time simulation; 30× faster
    preprocessing.

## Results

**14 seizures, 2 patients, leave-one-recording-out:** AUC **0.979**, epoch
sensitivity **79%** @ 99% specificity, **13/14 seizures detected** at **0.23
false alarms/hour** (3 s median latency). Bifurcation `B_crit = 42.57`
(paper: 42.42), 96% seizure suppression in closed loop.

Full breakdown, ablations, operating curve and limitations: **`RESULTS.md`**.
