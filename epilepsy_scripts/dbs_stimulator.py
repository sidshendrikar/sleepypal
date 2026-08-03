"""
dbs_stimulator.py — Physical actuator model for a deep-brain stimulator.

Everything downstream (energy comparisons, safety checks) depends on modelling
the stimulator as a real device rather than an abstract "control signal", so
this module implements:

  * charge-balanced biphasic pulse trains (amplitude, pulse width, frequency)
  * charge per phase and charge density at the electrode
  * TEED (Total Electrical Energy Delivered) -- the standard aDBS energy metric
  * Shannon/McCreery charge-density safety limit checking
  * a first-order neuromodulatory lag: tissue effect does not follow the
    commanded amplitude instantaneously

Default electrode geometry is a Medtronic 3389-style contact
(1.5 mm length x 1.27 mm diameter -> ~0.06 cm^2 surface area).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class StimSpec:
    """Programmable stimulation parameters and electrode/tissue constants."""
    amp_max_mA: float = 3.0        # max programmable current amplitude (mA)
    pulse_width_us: float = 60.0   # per-phase pulse width (microseconds)
    frequency_hz: float = 130.0    # classic therapeutic DBS rate
    impedance_ohm: float = 1000.0  # typical DBS electrode impedance
    area_cm2: float = 0.06         # contact surface area (Medtronic 3389)
    tau_effect_s: float = 0.30     # neuromodulatory onset/offset time constant

    # --- derived per-pulse quantities -------------------------------------
    def charge_per_phase_uC(self, amp_mA):
        """Q = I * PW. amp in mA, PW in us -> microcoulombs."""
        return np.asarray(amp_mA) * (self.pulse_width_us * 1e-6) * 1e3

    def charge_density(self, amp_mA):
        """uC / cm^2 / phase -- the quantity safety limits are written in."""
        return self.charge_per_phase_uC(amp_mA) / self.area_cm2

    def energy_per_pulse_J(self, amp_mA):
        """E = I^2 * R * PW (both phases of the biphasic pulse)."""
        I = np.asarray(amp_mA) * 1e-3
        return (I ** 2) * self.impedance_ohm * (self.pulse_width_us * 1e-6) * 2.0

    def teed_W(self, amp_mA):
        """Total Electrical Energy Delivered, in watts (J/s) at this amplitude.

        The conventional clinical TEED is (V^2 * f * PW)/R; we are current
        controlled, so the equivalent is I^2 * R * PW * f."""
        return self.energy_per_pulse_J(amp_mA) * self.frequency_hz

    # --- safety -----------------------------------------------------------
    def shannon_k(self, amp_mA):
        """Shannon safety parameter k = log10(D) + log10(Q), with D the charge
        density (uC/cm^2/phase) and Q the charge per phase (uC).
        Tissue damage has been reported above roughly k = 1.5-2.0."""
        Q = np.maximum(self.charge_per_phase_uC(amp_mA), 1e-12)
        D = np.maximum(self.charge_density(amp_mA), 1e-12)
        return np.log10(D) + np.log10(Q)

    def is_safe(self, amp_mA, k_limit=1.5):
        return self.shannon_k(amp_mA) < k_limit

    def max_safe_amplitude_mA(self, k_limit=1.5):
        """Largest amplitude whose Shannon k stays below the limit."""
        amps = np.linspace(1e-3, 20.0, 20000)
        ok = self.shannon_k(amps) < k_limit
        return float(amps[ok].max()) if ok.any() else 0.0


class Stimulator:
    """Stateful actuator: converts a commanded normalized amplitude u in [0,1]
    into delivered charge, accumulated energy, and a smoothed tissue effect."""

    def __init__(self, spec: StimSpec = None):
        self.spec = spec or StimSpec()
        self.reset()

    def reset(self):
        self.effect = 0.0          # smoothed normalized tissue effect in [0,1]
        self.energy_J = 0.0        # cumulative delivered energy
        self.charge_uC = 0.0       # cumulative delivered charge (per-phase sum)
        self.on_time_s = 0.0       # time with stimulation active
        self.total_time_s = 0.0
        self.n_pulses = 0

    def step(self, u_cmd, dt_s):
        """Advance the actuator by dt_s with commanded normalized amplitude
        u_cmd in [0,1]. Returns the smoothed effect in [0,1]."""
        u_cmd = float(np.clip(u_cmd, 0.0, 1.0))
        # first-order neuromodulatory lag toward the command
        tau = max(self.spec.tau_effect_s, 1e-6)
        self.effect += (u_cmd - self.effect) * (dt_s / tau)
        self.effect = float(np.clip(self.effect, 0.0, 1.0))

        # energy is billed on the COMMANDED amplitude (charge actually leaves
        # the device even before the tissue effect settles)
        amp = u_cmd * self.spec.amp_max_mA
        if amp > 1e-9:
            n_pulses = self.spec.frequency_hz * dt_s
            self.energy_J += self.spec.energy_per_pulse_J(amp) * n_pulses
            self.charge_uC += self.spec.charge_per_phase_uC(amp) * n_pulses
            self.on_time_s += dt_s
            self.n_pulses += n_pulses
        self.total_time_s += dt_s
        return self.effect

    # --- reporting --------------------------------------------------------
    @property
    def duty_cycle(self):
        return self.on_time_s / max(self.total_time_s, 1e-12)

    @property
    def mean_power_W(self):
        return self.energy_J / max(self.total_time_s, 1e-12)

    def report(self):
        return {
            "energy_J": self.energy_J,
            "charge_uC": self.charge_uC,
            "duty_cycle": self.duty_cycle,
            "mean_power_W": self.mean_power_W,
            "on_time_s": self.on_time_s,
        }


if __name__ == "__main__":
    spec = StimSpec()
    print(f"electrode area      : {spec.area_cm2} cm^2")
    print(f"max safe amplitude  : {spec.max_safe_amplitude_mA():.2f} mA "
          f"(Shannon k < 1.5)")
    for a in [0.5, 1.0, 2.0, 3.0]:
        print(f"  {a:.1f} mA -> Q={spec.charge_per_phase_uC(a):.3f} uC/phase, "
              f"D={spec.charge_density(a):.1f} uC/cm^2, "
              f"k={spec.shannon_k(a):.2f}, TEED={spec.teed_W(a)*1e6:.1f} uW, "
              f"safe={spec.is_safe(a)}")
