"""
config.py — Central configuration for the Hybrid Digital Twin seizure-control
framework (Shendrikar, Bi 195).

All physical constants for the Jansen-Rit neural mass model and every
signal-processing hyperparameter live here so the mechanistic model, the
observer network, and the controller stay in sync.
"""
from dataclasses import dataclass, field
import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data", "epilepsy")
ARTIFACT_DIR = os.path.join(PKG_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Jansen-Rit neural mass model parameters (Table 1 of the paper)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class JRParams:
    A: float = 3.25       # excitatory synaptic gain (mV)
    B: float = 22.0       # inhibitory synaptic gain (mV) -- CONTROL PARAMETER
    a: float = 100.0      # excitatory membrane rate constant (1/s)  tau_e = 10 ms
    b: float = 50.0       # inhibitory membrane rate constant (1/s)  tau_i = 20 ms
    C: float = 135.0      # global connectivity constant
    # Sigmoid (population firing-rate) parameters
    e0: float = 2.5       # half of max firing rate (1/s)
    v0: float = 6.0       # firing threshold / inflection voltage (mV)
    r: float = 0.56       # sigmoid steepness (1/mV)
    # Stochastic cortical input p(t) = p_mean + N(0, p_std^2)
    p_mean: float = 220.0  # mean background drive (pulses/s)
    p_std: float = 22.0    # 10% coefficient of variation

    @property
    def C1(self) -> float:  # pyramidal -> excitatory interneuron
        return self.C

    @property
    def C2(self) -> float:  # excitatory interneuron -> pyramidal
        return 0.8 * self.C

    @property
    def C3(self) -> float:  # pyramidal -> inhibitory interneuron
        return 0.25 * self.C

    @property
    def C4(self) -> float:  # inhibitory interneuron -> pyramidal
        return 0.25 * self.C

    def as_array(self):
        """Flat float64 array for the Numba kernels (order matters)."""
        import numpy as np
        return np.array(
            [self.A, self.B, self.a, self.b,
             self.C1, self.C2, self.C3, self.C4,
             self.e0, self.v0, self.r],
            dtype="float64",
        )


# ----------------------------------------------------------------------------
# Simulation / integration
# ----------------------------------------------------------------------------
DT = 1e-3            # integration step (s)  -> 1 ms  (paper: forward Euler)
FS_SIM = int(1 / DT)  # 1000 Hz simulated sampling rate
TRANSIENT_S = 2.0    # discard this many seconds before measuring steady state


# ----------------------------------------------------------------------------
# EEG preprocessing (Section 2.2-2.3 of the paper)
# ----------------------------------------------------------------------------
FS = 256             # canonical EEG sampling rate (Hz)
BP_LOW = 0.5         # Butterworth band-pass low corner (Hz)
BP_HIGH = 70.0       # Butterworth band-pass high corner (Hz)
BP_ORDER = 4
NOTCH = 60.0         # mains notch (Hz) -- IMPROVEMENT over the paper
EPOCH_S = 5.0        # epoch length (s)
EPOCH_SAMPLES = int(FS * EPOCH_S)   # 1280
EPOCH_STRIDE_S = 5.0  # non-overlapping by default (overridden for augmentation)
PREICTAL_S = 300     # 5-minute pre-ictal horizon
# Guard band excluded from "interictal" around every seizure (s)
INTERICTAL_GUARD_S = 900

# Log-Mel spectrogram (Section 2.3)
STFT_WIN = 256       # Hann window length (samples, ~1 s)
STFT_HOP = 64        # 75% overlap
N_MELS = 64
MEL_FMIN = 0.5
MEL_FMAX = 70.0
IMG_SIZE = 64        # final 64x64 tensor fed to the CNN-LSTM

# Class labels
LABELS = {"interictal": 0, "ictal": 1, "preictal": 2}
LABEL_NAMES = ["interictal", "ictal", "preictal"]


# ----------------------------------------------------------------------------
# Observer network training
# ----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    val_frac: float = 0.2
    seed: int = 1234
    # Class-imbalance remedy (the paper's 12.5% sensitivity fix).
    # Exactly ONE strategy is applied -- stacking them over-corrects and
    # collapses the model onto the minority class.
    #   "class_weight"     : inverse-frequency loss weighting (plain CE)
    #   "oversample"       : balance classes by minority oversampling (plain CE)
    #   "oversample_focal" : oversample + focal loss (uniform alpha)
    strategy: str = "oversample"     # benchmarked best (AUC 0.90 vs 0.44)
    focal_gamma: float = 2.0
    oversample_cap: float = 1.0      # oversample up to cap*majority count
    # Decision threshold is tuned on validation to hit a target sensitivity
    target_sensitivity: float = 0.90


RANDOM_SEED = 1234
