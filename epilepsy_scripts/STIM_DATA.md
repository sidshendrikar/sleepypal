# Stimulation-paired iEEG: what exists, and what it can actually calibrate

The DBS controller's weakest assumption is the coupling — how much a delivered
charge shifts cortical excitability. It is currently pinned by *inversion from
clinical dose* (gain ≈ 21 mV, 17–25 range) rather than fitted to recordings.
This is a survey of what stimulation-paired intracranial data is genuinely
obtainable, and — more importantly — **what it can and cannot calibrate.**

## The headline finding

**Single-pulse (CCEP/SPES) data cannot calibrate our coupling.** It is a
different physical quantity:

| | CCEP / SPES | Our coupling |
|---|---|---|
| what it measures | passive **impulse response** of a system at a fixed point | **sustained shift** in inhibitory gain B |
| stimulation | single pulse, 0.2 Hz | 130 Hz continuous train |
| what it estimates | effective connectivity, synaptic gains at baseline | parameter drift relative to `B_crit` |

So CCEP **can** legitimately calibrate: the actuator scaling (mA → model input
units), the synaptic gains and time constants `He/Hi/τe/τi`, conduction delays,
and spatial gain heterogeneity. It **cannot** calibrate `dB/d(charge)` under
sustained HFS — that remains a free parameter with a CCEP-anchored prior, which
is exactly why the inversion calibration plus a reported sensitivity band (not a
point estimate) is the honest treatment.

Partial purchase is available from *train* stimulation with matched pre/post
epochs (below), which is the only open data that probes a sustained state change.

## Ranked candidates

| # | Dataset | Stimulation | Access | Why |
|---|---|---|---|---|
| 1 | **DABI `W4SNQ7HR49RL`** (Paulk/Cash, MGH) | SPES **0.5–10 mA in 0.5 mA steps** × 233/1053 µs, n=52, 719 sites | free account + click-through DUA | the only true **parametric dose sweep** |
| 2 | **OpenNeuro `ds006519`** | **50 Hz trains**, 0.25–3 mA, up to 5 s, **matched pre/post epochs** | CC0, anonymous | best *open* set; only one probing **sustained** state change |
| 3 | `ds004696` (HAPwave) | biphasic single-pulse, **4 vs 6 mA** | CC0, anonymous | zero friction, but a 2-point "curve" |
| 4 | `ds004080` (van Blooijs/Hermes) | monophasic SPES, 1 ms, 0.2 Hz, **fixed 7–8 mA** | CC0, anonymous | 74 subjects, ages 4–51 — but **single dose ⇒ no slope**; use as spatial/developmental prior |
| 5 | RAM `ds0055xx` | 50 Hz, 300 µs, 1000 µA, 4.6 s trains | CC0 | richest per-event params, but hippocampal memory task |
| 6 | `ds007095` (RNS) | 200 Hz, 2.5 mA, 160 µs | CC0 | only 2 channels @ 200 Hz; stim aliases to DC — not quantitative |

All OpenNeuro sets above are **CC0 public domain and verified anonymously
downloadable** (real files pulled with unauthenticated `curl` during the survey),
in BIDS-iEEG with per-trial stimulation parameters in `events.tsv`
(current in A, frequency in Hz, pulse width in s).

## Stop looking for these

There is **no open chronic therapeutic-HFS-with-recording corpus.** Confirmed
closed: NeuroPace PDMS/RNS raw ECoG (corporate committee), Medtronic Percept
BrainSense patient LFP (deposited nowhere), Gilron 2021 at-home aDBS
(author-request, *not* on DABI), Worrell/Kremen ANT-DBS RC+S (request-only),
UCSF RNS-depression and Yale thalamic corpora (private, PI approval), IEEG.org
(per-project credentialing, redistribution prohibited), `ds007703` (returns 403).

## If we did the calibration

Procedure, per the survey:

1. Per trial extract `(site pair, I, PW)` → charge per phase `Q = I·PW` (nC).
2. Bipolar/CARLA re-reference, blank 0–10 ms, baseline −500 to −10 ms, drop
   `status=bad` trials.
3. Per (stim-pair, channel): trial-average, measure N1 peak amplitude `A`
   (10–50 ms), latency, and 10–100 ms area.
4. Fit a saturating dose-response `A(Q) = A_max·Qⁿ/(Q₅₀ⁿ + Qⁿ)` → threshold and
   saturation (Paulk reports this nonlinearity).
5. Map to Jansen-Rit as **two** fits, not one:
   - **waveform fit** — the linearised JR impulse response has closed form; fit
     `He, τe, Hi, τi` to N1 rise/decay and N2 latency (this is where the
     inhibitory time constants become data-constrained);
   - **gain fit** — choose the input scaling `k` so simulated evoked `y₁−y₂`
     matches `A` at matched `Q`; the measured saturation tells you whether `k`
     is linear or must pass through the sigmoid.

**What this would buy:** data-constrained synaptic parameters and a real
actuator scaling, replacing several modelling choices.
**What it would not buy:** the sustained-HFS coupling itself. That needs chronic
therapeutic stimulation data, which is not open.

---

# Executed: partial calibration on ds006519

`stim_calib.py` — 586 MB downloaded, **35 stimulation trials**, 0.25–3.00 mA
across 10 distinct levels, at 43 and 50 Hz. Calibration chain:

```
data   dF/dI  (spectral shift per mA, matched 5 s pre/post epochs)
model  dF/dB  (same feature vs inhibitory gain, Jansen-Rit, quiet regime)
  ⇒    dB/dI = (dF/dI)/(dF/dB)
```

## Result

| readout | slope per mA | r | perm p | implied dB/dI | gain at 3 mA |
|---|---|---|---|---|---|
| log low-band power | **+0.0526** (SE 0.0246) | +0.35 | **0.040** | **−0.75 mV/mA** | **−2.3 mV** |
| aperiodic exponent | +0.0425 (SE 0.0499) | +0.15 | 0.401 | +0.35 mV/mA | +1.0 mV |

*(inversion-from-clinical-dose estimate, for comparison: **+22 mV** at 3 mA)*

## Interpretation — the sign is the finding

Higher stimulation current produced **more** post-stimulation low-frequency
power (median Δlog power −0.058 at ≤1 mA vs −0.004 at ≥2 mA). In the model,
raising `B` *lowers* low-frequency power. So the dose-dependence maps to a
**slightly negative** coupling: at 43–50 Hz, stimulation does **not** add
effective inhibition — if anything it mildly reduces it.

**This is the expected answer, and it triangulates from three directions:**

1. It matches the model's own frequency sweep (`dbs_fig6_hfs_frequency.png`),
   where 20–60 Hz never suppressed at any amplitude and slightly *raised* band
   power; only 130–180 Hz suppressed.
2. It matches physiology: 50 Hz direct cortical stimulation is an **activating**
   mapping protocol — this very dataset exists because such stimulation *evokes*
   clinical responses.
3. It explains why therapeutic DBS runs at 130+ Hz rather than 50 Hz.

## What this does and does not establish

**Does:** bounds the 43–50 Hz coupling at **≈0 (slightly negative), nowhere near
the +22 mV the controller requires** — and independently corroborates, from real
human stimulation data, the frequency-dependence the model predicted.

**Does not:** calibrate the therapeutic 130 Hz coupling. The inversion estimate
(21 mV, 17–25 range) plus the reported sensitivity band remains the treatment for
that term, and this exercise shows *why* no open dataset can replace it.

**Statistical honesty:** n = 35, r = 0.35, permutation p = 0.040 for the
low-band readout — but two features were tested, so Bonferroni-corrected
p ≈ 0.08, i.e. **suggestive, not conclusive**. The aperiodic readout is frankly
null (p = 0.40). The two readouts also disagree in sign, which is what one
expects when both effects are near zero. Trials are interictal functional-mapping
stimulations in epilepsy patients, not seizure-state, and the model's `dF/dB` was
evaluated at the quiet operating point (B 30–46) to match.
