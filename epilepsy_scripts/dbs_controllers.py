"""
dbs_controllers.py — competing closed-loop stimulation control laws.

Every controller sees the SAME sensed features from the SAME simulated signal
and drives the SAME actuator, so any difference in outcome is attributable to
the control law alone.

Interface:  reset()  ->  update(feat, t, dt) -> u in [0, 1]
`feat` carries only quantities a real device could measure from the recorded
signal: peak-to-peak amplitude, line length (the NeuroPace RNS biomarker), and
band power in the seizure rhythm band. No controller ever sees ground truth.

The ladder of sophistication:
  NoStim           - untreated reference
  OpenLoopDBS      - continuous high-frequency DBS (conventional therapy)
  ThresholdADBS    - RNS-style: biomarker crosses threshold -> fixed amplitude
  DualThresholdADBS- Percept-style: ramp amplitude between two thresholds
  BifurcationMPC   - THIS WORK: the patient-fitted twin supplies B_crit and the
                     stimulation->inhibition gain, so the controller jumps
                     straight to the model-predicted MINIMUM amplitude that
                     crosses the bifurcation, then adaptively seeks the floor.
"""
import numpy as np


class Controller:
    name = "base"

    def reset(self):
        pass

    def update(self, feat, t, dt):
        return 0.0


class NoStim(Controller):
    name = "no stimulation"

    def update(self, feat, t, dt):
        return 0.0


class OpenLoopDBS(Controller):
    """Conventional continuous DBS: always on at a fixed amplitude."""
    name = "open-loop DBS"

    def __init__(self, amp=1.0):
        self.amp = amp

    def update(self, feat, t, dt):
        return self.amp


class ThresholdADBS(Controller):
    """Responsive stimulation in the deployed style (NeuroPace RNS / single
    threshold): when the band-limited biomarker exceeds a calibrated threshold,
    deliver a fixed-amplitude burst. Reactive and non-titrated -- it cannot
    know *how much* stimulation is actually needed."""
    name = "threshold aDBS (RNS-style)"

    def __init__(self, ll_threshold, amp=1.0, min_burst_s=1.0,
                 feature="band_power"):
        self.thr = ll_threshold
        self.amp = amp
        self.min_burst_s = min_burst_s
        self.feature = feature
        self.reset()

    def reset(self):
        self._burst_left = 0.0

    def update(self, feat, t, dt):
        if feat[self.feature] > self.thr:
            self._burst_left = self.min_burst_s
        if self._burst_left > 0.0:
            self._burst_left -= dt
            return self.amp
        return 0.0


class DualThresholdADBS(Controller):
    """Medtronic Percept-style adaptive DBS: amplitude ramps linearly between a
    lower and an upper biomarker threshold, with a bounded ramp rate."""
    name = "dual-threshold aDBS (Percept-style)"

    def __init__(self, ll_low, ll_high, amp_low=0.3, amp_high=1.0,
                 ramp_per_s=1.0, feature="band_power"):
        self.feature = feature
        self.lo, self.hi = ll_low, ll_high
        self.a_lo, self.a_hi = amp_low, amp_high
        self.ramp = ramp_per_s
        self.reset()

    def reset(self):
        self.u = 0.0

    def update(self, feat, t, dt):
        x = feat[self.feature]
        frac = np.clip((x - self.lo) / max(self.hi - self.lo, 1e-9), 0.0, 1.0)
        target = self.a_lo + frac * (self.a_hi - self.a_lo) if x > self.lo else 0.0
        step = self.ramp * dt
        self.u += np.clip(target - self.u, -step, step)
        return float(np.clip(self.u, 0.0, 1.0))


class BifurcationMPC(Controller):
    """THIS WORK -- model-based minimum-energy stimulation.

    The patient-specific digital twin contributes two things no signal-threshold
    controller has:

      1. `B_crit`, the inhibitory gain at which the seizure limit cycle is
         annihilated (a Hopf bifurcation), and
      2. `gain`, how much effective inhibition a unit of stimulation buys.

    Together these invert to a *dose*: the minimum amplitude that carries the
    cortex across the stability boundary,

        u* = ( margin * B_crit  -  B_hat ) / gain

    so on detection the controller steps straight to u* instead of slamming to
    maximum (threshold aDBS) or running continuously (open loop). It then
    performs minimum-seeking: while the oscillation stays suppressed it decays
    the amplitude to find the true floor, and bumps back up if the seizure
    re-emerges -- which also absorbs error in B_hat.
    """
    name = "bifurcation MPC (this work)"

    def __init__(self, B_crit, gain, B_hat, ll_threshold, margin=1.05,
                 decay_per_s=0.06, bump=0.10, hold_s=1.0,
                 preictal_lead_s=0.0, feature="band_power",
                 observe_delay_s=1.5, release_s=4.0, use_model_dose=True,
                 default_dose=0.30):
        self.B_crit, self.gain, self.B_hat = B_crit, gain, B_hat
        self.thr = ll_threshold
        self.feature = feature
        self.margin = margin
        self.decay = decay_per_s      # minimum-seeking decay rate
        self.bump = bump              # step up when seizure re-emerges
        self.hold_s = hold_s          # keep stimulating after suppression
        self.preictal_lead_s = preictal_lead_s
        # the loop cannot judge a new amplitude until the sensing window has
        # refreshed and the neuromodulatory effect has settled
        self.observe_delay_s = observe_delay_s
        # quiet time before concluding the episode is over and standing down
        self.release_s = release_s
        # ablation: if False the controller keeps the adaptive ratchet but is
        # denied the twin's dose prediction, isolating what the bifurcation
        # coordinate actually contributes
        self.use_model_dose = use_model_dose
        self.default_dose = default_dose
        self.reset()

    # model-predicted minimum dose
    def required_u(self, B_est=None):
        if not self.use_model_dose:
            return self.default_dose
        B = self.B_hat if B_est is None else B_est
        need = self.margin * self.B_crit - B
        return float(np.clip(need / max(self.gain, 1e-9), 0.0, 1.0))

    def reset(self):
        self.u = 0.0
        self._hold = 0.0
        self._engaged = False
        self._floor = 0.0        # ratchet: highest amplitude PROVEN insufficient
        self._quiet = 0.0        # accumulated suppressed time
        self._wait = 0.0         # observation delay after an amplitude change

    def update(self, feat, t, dt):
        seizing = feat[self.feature] > self.thr
        early = feat.get("preictal", False) and self.preictal_lead_s > 0

        if seizing or early:
            if not self._engaged:
                # model-based jump straight to the predicted minimum dose
                self.u = max(self.u, self.required_u(), self._floor + self.bump)
                self._engaged = True
                self._wait = self.observe_delay_s
            elif self._wait > 0.0:
                # still waiting to see whether the current dose worked
                self._wait -= dt
            else:
                # seizure persists at self.u AFTER the loop delay -> that
                # amplitude is demonstrably insufficient. Ratchet and step up.
                self._floor = max(self._floor, self.u)
                self.u = min(1.0, self._floor + self.bump)
                self._wait = self.observe_delay_s
            self._hold = self.hold_s
            self._quiet = 0.0
        else:
            self._quiet += dt
            if self._hold > 0.0:
                self._hold -= dt
            else:
                # Suppressed. Trim toward the model-predicted minimum dose (or
                # the ratchet floor, whichever is higher) but never below it:
                # with a model in hand there is no reason to probe downward and
                # risk re-entering the seizure, which is what makes purely
                # reactive controllers chatter.
                target = max(self.required_u(), self._floor + self.bump)
                if self.u > target:
                    self.u = max(target, self.u - self.decay * dt)
                elif self._quiet > self.release_s:
                    # long quiet spell: the episode has probably ended, so it is
                    # safe to probe downward again and eventually disengage
                    self.u = max(0.0, self.u - self.decay * dt)
                    if self.u <= 1e-6:
                        self._engaged = False
                        self._floor = 0.0
        return float(np.clip(self.u, 0.0, 1.0))


def calibrate_threshold(interictal_line_lengths, percentile=99.0):
    """Set a biomarker threshold from interictal data, as devices are
    programmed in practice. Keeps the baselines honest rather than strawmen."""
    return float(np.percentile(np.asarray(interictal_line_lengths), percentile))


# ---------------------------------------------------------------------------
# Faithful re-implementations of DEPLOYED device control laws, plus the
# controls that keep the comparison honest.
# ---------------------------------------------------------------------------
class RNSLineLength(Controller):
    """NeuroPace RNS-analogue detector: a SHORT-term line-length average
    exceeding the LONG-term average by a ratio triggers a brief fixed burst,
    followed by a refractory period. Reactive burst-abort, no dose concept."""
    name = "RNS-analogue (line-length ratio)"

    def __init__(self, ratio=1.75, amp=1.0, burst_s=0.5, refractory_s=0.5,
                 tau_short_s=0.5, tau_long_s=10.0, feature="line_length"):
        self.ratio, self.amp = ratio, amp
        self.burst_s, self.refractory_s = burst_s, refractory_s
        self.ts, self.tl = tau_short_s, tau_long_s
        self.feature = feature
        self.reset()

    def reset(self):
        self.short = None; self.long = None
        self._burst = 0.0; self._refr = 0.0

    def update(self, feat, t, dt):
        x = feat[self.feature]
        self.short = x if self.short is None else \
            self.short + (x - self.short) * min(1.0, dt / self.ts)
        self.long = x if self.long is None else \
            self.long + (x - self.long) * min(1.0, dt / self.tl)
        if self._burst > 0.0:
            self._burst -= dt
            return self.amp
        if self._refr > 0.0:
            self._refr -= dt
            return 0.0
        if self.long > 1e-12 and (self.short / self.long) > self.ratio:
            self._burst = self.burst_s
            self._refr = self.refractory_s
            return self.amp
        return 0.0


class PerceptST(Controller):
    """Medtronic Percept single-threshold aDBS: band power above the threshold
    ramps amplitude toward the UPPER limit, below it ramps toward the LOWER
    limit. Amplitude is confined to [u_min, u_max] -- the device never switches
    stimulation off -- and ramps are rate limited (nominal 250 ms)."""
    name = "Percept ST aDBS"

    def __init__(self, thr, u_min=0.3, u_max=1.0, ramp_s=0.25,
                 feature="band_power"):
        self.thr, self.u_min, self.u_max = thr, u_min, u_max
        self.ramp_s = ramp_s
        self.feature = feature
        self.reset()

    def reset(self):
        self.u = self.u_min

    def update(self, feat, t, dt):
        target = self.u_max if feat[self.feature] > self.thr else self.u_min
        step = (self.u_max - self.u_min) * dt / max(self.ramp_s, 1e-9)
        self.u += float(np.clip(target - self.u, -step, step))
        return float(np.clip(self.u, self.u_min, self.u_max))


class RandomDutyMatched(Controller):
    """Duty-cycle- and amplitude-matched RANDOM stimulation.

    This is the control that decides whether the benefit comes from *when and
    how much* stimulation is delivered, or merely from delivering less of it.
    It is given exactly the same duty cycle and amplitude as the controller it
    is matched to, but places its stimulation at random times."""
    name = "random (duty-matched)"

    def __init__(self, duty, amp, seed=0, block_s=1.0):
        self.duty, self.amp, self.block_s = duty, amp, block_s
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self._left = 0.0
        self._on = False

    def update(self, feat, t, dt):
        if self._left <= 0.0:
            self._on = self.rng.random() < self.duty
            self._left = self.block_s
        self._left -= dt
        return self.amp if self._on else 0.0


class OpenLoopCycled(Controller):
    """Conventional duty-cycled open-loop DBS (e.g. 1 min on / 5 min off)."""
    name = "open-loop cycled"

    def __init__(self, amp=1.0, on_s=60.0, off_s=300.0):
        self.amp, self.on_s, self.off_s = amp, on_s, off_s

    def update(self, feat, t, dt):
        period = self.on_s + self.off_s
        return self.amp if (t % period) < self.on_s else 0.0
