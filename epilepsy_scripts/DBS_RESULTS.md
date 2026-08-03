# Adaptive DBS: a bifurcation-dosed closed-loop controller

**This is an in-silico control study.** Every arm below is *our re-implementation*
of a published control law, evaluated inside *our* Jansen-Rit model. It measures
how control laws behave in a neural mass model. It is not a comparison against
commercial devices, not a clinical result, and not a medical device.

## What was built

| Module | Role |
|---|---|
| `dbs_stimulator.py` | Physical actuator: charge-balanced biphasic pulses, charge/phase, charge density, TEED, Shannon safety limit, neuromodulatory lag |
| `dbs_controllers.py` | Nine control laws (deployed-style baselines, controls, ablation, and this work) |
| `dbs_sim.py` | Closed loop at a 100 ms embedded control period: disease → plant → sensing → control → actuator |
| `dbs_benchmark.py` | Head-to-head, realistic-duty, matched-suppression analyses |
| `dbs_leadtime.py` | Predictive (pre-onset) trigger sweep |
| `dbs_figures.py` | Figures |

**The plant.** A seizure is modelled as a transient collapse of inhibition:
`B_disease(t)` drops from 46 (quiet fixed point) into the limit-cycle band (28)
and recovers. Stimulation adds effective inhibition back:

```
B_eff(t) = B_disease(t) + gain · stim_effect(t)
```

with `stim_effect` a first-order (τ = 300 ms) response to delivered charge.
Seizure presence is scored from 2–8 Hz band power against an interictal-calibrated
threshold — **not** raw amplitude, because the stochastic drive gives the
*suppressed* state a several-mV peak-to-peak that would be misread as seizure.

## The state of the art being compared against

From a literature/device survey (see `artifacts/`):

- **Medtronic Percept BrainSense aDBS** — FDA-approved Feb 2025, the first and
  only commercial closed-loop DBS. Control law: LFP band power (8–30 Hz) against
  one threshold (ST) or a two-threshold dead-band (DT); amplitude ramps between
  clinician-set limits. **No model, no state observer, no ML in the loop —
  a scalar-feature relay. Strictly reactive.** Reported TEED reduction vs
  continuous DBS: **13–16%** (and *higher* than continuous in the ADAPT-START
  real-world cohort — the energy claim is not robust).
- **NeuroPace RNS** (epilepsy) — short/long line-length ratio triggers a brief
  fixed burst, then refractory. Reactive burst-abort, no dose concept.

**The gap:** neither knows *how much* stimulation the cortex actually needs, and
neither can act before a seizure exists.

## The controller

The patient-fitted twin supplies two things a threshold relay does not have:
`B_crit` (the Hopf point where the seizure limit cycle is annihilated) and the
stimulation→inhibition `gain`. These invert into a **dose**:

```
u* = (margin · B_crit − B̂) / gain          → 0.763 here
```

so on detection the controller steps straight to the model-predicted minimum
amplitude rather than slamming to maximum. It then **ratchets**: any amplitude
that demonstrably fails is recorded as a floor and never revisited, and it never
probes below the model prediction — which is what stops the chattering that
plagues purely reactive control.

## Results

### Realistic duty — 10-minute record, one 35 s seizure (5.8% of record)

| Arm | Burden | Energy | Duty |
|---|---|---|---|
| open-loop continuous | **0.0 s** | 84.24 mJ | 100% |
| open-loop cycled 5s/25s | 26.9 s | 14.04 mJ | 17% |
| Percept ST-style | 20.8 s | 10.20 mJ | 100% |
| Percept DT-style | 22.7 s | 2.70 mJ | 6% |
| threshold aDBS (RNS-style) | 11.2 s | 5.07 mJ | 6% |
| **bifurcation MPC (this work)** | **7.2 s** | **3.96 mJ** | 12% |

**Lowest burden of any closed-loop arm, at 95.3% less energy than continuous DBS.**
The dense 60 s scenario understates this: always-on therapy only looks affordable
when the record is 58% seizure.

### The controls that keep it honest

| Comparison | Energy | Burden |
|---|---|---|
| vs **duty-matched random** stimulation | **−29.3%** | **−2.2 s** |
| vs **ablation: twin dose removed** | +18.8% | **−4.9 s** |

The random arm receives the *same duty cycle and amplitude* and places it at
random times. Beating it establishes that the benefit comes from **when and how
much** stimulation is delivered — not merely from delivering less. The ablation
keeps the adaptive ratchet but denies the twin's dose prediction: burden rises
from 7.0 s to 12.0 s, so the bifurcation coordinate is doing real work.

### Matched suppression — an honest negative result

Sweeping every arm for the cheapest setting reaching ≤ 2 s burden:

| Arm | Result |
|---|---|
| open-loop continuous | amp 0.70 → 0.0 s, **41.28 mJ** |
| Percept ST-style | never reached ≤ 2 s |
| threshold aDBS | never reached ≤ 2 s |
| bifurcation MPC | never reached ≤ 2 s |

**No reactive closed-loop arm — including ours — reaches 2 s.** The floor is set
by detection latency: the seizure must begin and be detected before any
stimulation can start. Only always-on therapy escapes it, by paying 10× the energy.

### Prediction closes the gap

Giving the controller a pre-onset warning (its observer's preictal class):

| Lead time | Burden | Energy | vs continuous |
|---|---|---|---|
| 0 s (reactive) | 8.4 s | 4.67 mJ | −94.5% |
| 3 s | 5.3 s | 5.46 mJ | −93.5% |
| **8 s** | **2.9 s** | **6.64 mJ** | **−92.1%** |
| 12 s | 3.4 s | 7.24 mJ | −91.4% |

Each second of warning buys ≈ 0.7 s less seizure, with an optimum near 8 s;
beyond that the controller stimulates early without further benefit. **This is
the axis on which deployed devices cannot compete — both are strictly reactive.**

## Coupling calibration and mechanism validation

The stimulation→cortex coupling was the least-observed constant, so it is now
pinned two ways and stress-tested (`dbs_coupling.py`, `dbs_hfs.py`).

**Magnitude — inversion from clinical dose.** Crossing the bifurcation needs
`ΔB = margin·B_crit − B_ictal ≈ 16.8 mV` of added inhibition. Requiring that a
clinically-typical current supply it:

| anchor current | implied gain |
|---|---|
| 2.0 mA | 25.2 mV |
| **2.4 mA (Percept mean)** | **21.0 mV** |
| 3.0 mA | 16.8 mV |

So `gain ≈ 21 mV (17–25 mV)`, and the value used (22) makes the controller settle
at **2.32 mA** — essentially the *observed* ADAPT-PD mean delivered amplitude
(2.4 mA). The energy conclusion is robust across the whole range: 90–98% saving
vs continuous DBS for gain 12–32 mV, and the ratchet absorbs coupling
misestimation (controller assuming 22 while the true coupling varies 12–32 still
saves 92–97%). Below ~15 mV the coupling is too weak for a safe current to cross
the bifurcation at all — the regime where the approach is physically viable.

**Mechanism — validated against a known clinical fact.** A literature survey
found the field-standard DBS injection is *additive* (a depolarizing pulse train;
Mina 2013, Muller-Robinson 2018), not a gain change — the gain-shift is a
control-theoretic abstraction of the *net* effect. Two checks:

- A smoothed constant additive inhibitory drive does **not** suppress, even at
  100 mV: a DC offset shifts the fixed point without changing the loop gain that
  creates the limit cycle. So the abstraction is a statement about the net
  effective gain, not a literal current.
- A **true 130 Hz pulsatile** injection (Mina-style) **does** suppress — and it
  reproduces the defining clinical fact that *only high-frequency DBS works*:
  20–60 Hz never suppresses at any amplitude; 130 Hz suppresses at high
  amplitude; 180 Hz at lower amplitude (`dbs_fig6_hfs_frequency.png`). This
  frequency-dependence **emerged** from the model rather than being assumed,
  which is strong evidence the plant captures real DBS physics.

Net: the gain-shift plant is a calibrated, validated abstraction of the net
suppressive effect that true high-frequency stimulation produces.

## Novelty — stated narrowly

Every *component* here is published. The combination is not.

Closest prior art: **Wang 2016** (Jansen-Rit + PI with gains chosen to prevent a
subcritical Hopf — conceptually nearest, but not patient-fitted or predictive);
**Liang 2023** (MPC on JR/Epileptor with a Koopman spectral biomarker);
**Ashourvan 2020** (patient-fitted LTI + pole placement — distance to the unit
circle); **Li 2021** (neural fragility as minimum-norm perturbation to
instability, used for localization); **Taylor 2015**, **Moosavi 2025**.

**Defensible claim:** no published loop uses *distance to a bifurcation* as the
feedback variable, and none pairs a patient-fitted neural mass twin with a
pre-onset trigger whose detector was validated on real EEG.

**Not claimed:** "first model-based seizure control" — that would be false.

## What would be overclaiming

- **The 96% suppression figure is partly circular** — the model that defines the
  seizure also defines the cure. In-silico suppression is not efficacy.
- **We did not beat Percept or RNS.** We beat *our re-implementations of their
  control laws inside our model.*
- **The stimulation→inhibition coupling is calibrated by inversion, not measured
  directly.** `gain ≈ 21 mV` comes from requiring the crossing dose to match the
  clinical current range; no experiment measures "mV of B per mA" (the survey
  confirmed none exists). The gain-shift *mechanism* is an abstraction of the net
  effect — validated qualitatively (true HFS reproduces frequency-dependence) but
  not fitted to stimulation recordings. TEED and µC/cm² use a plausible electrode
  model, not a validated one.
- **The dose inversion depends on `B̂`**, and fitting `B` to a power spectrum is a
  many-to-one inverse problem — the estimate is degenerate, not a point fact.
- **Negative detection latency is not proof of prediction** (scalp onset marks
  carry seconds of uncertainty). The lead-time sweep *imposes* a warning rather
  than earning it prospectively.
- **No implant-feasibility claim.** Compute, latency and power budgets are unmeasured.

## Figures

`dbs_fig1_traces.png` (example traces), `dbs_fig2_realistic.png` (burden/energy
over a realistic record), `dbs_fig3_plane.png` (energy–burden plane),
`dbs_fig4_leadtime.png` (prediction sweep).

## Reproducing

```bash
python dbs_stimulator.py              # actuator sanity check
python dbs_benchmark.py               # head-to-head + controls + ablation
python dbs_benchmark.py --realistic   # realistic duty + matched suppression
python dbs_leadtime.py                # predictive trigger sweep
python dbs_figures.py                 # figures
```
