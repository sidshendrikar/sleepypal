"""
jansen_ritt.py — Jansen-Rit neural mass model (the "Replication" engine).

A biophysical model of one cortical column: three interacting neural
populations (pyramidal cells, excitatory interneurons, inhibitory
interneurons) described by six coupled ODEs (Section 2.5 of the paper).

The state vector is y = [y0, y1, y2, y3, y4, y5] where
    y0 : excitatory PSP onto pyramidal cells      y3 = y0_dot
    y1 : excitatory PSP (feedback loop)           y4 = y1_dot
    y2 : inhibitory PSP onto pyramidal cells      y5 = y2_dot
and the observable "EEG" is the pyramidal membrane potential  V = y1 - y2.

Integration is JIT-compiled with Numba for ~100x speedup over pure Python,
exactly as the paper specifies. Two integrators are provided:

  * Euler-Maruyama  — matches the paper's forward-Euler scheme and supports
    the stochastic cortical drive p(t) = p_mean + N(0, p_std^2).
  * RK4             — deterministic, higher-order; used by the bifurcation
    analysis where noise is switched off to map the *structural* stability
    boundary cleanly (an improvement over Euler for that task).
"""
import numpy as np
from numba import njit

from config import JRParams, DT, TRANSIENT_S


# ----------------------------------------------------------------------------
# Population firing-rate sigmoid  S(v) = 2*e0 / (1 + exp(r*(v0 - v)))
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _sigmoid(v, e0, v0, r):
    # clip the exponent to avoid overflow at extreme voltages
    x = r * (v0 - v)
    if x > 30.0:
        return 0.0
    if x < -30.0:
        return 2.0 * e0
    return 2.0 * e0 / (1.0 + np.exp(x))


@njit(cache=True, fastmath=True)
def _derivs(y, p, prm):
    """Right-hand side of the six Jansen-Rit ODEs.

    prm = [A, B, a, b, C1, C2, C3, C4, e0, v0, r]
    """
    A, B, a, b = prm[0], prm[1], prm[2], prm[3]
    C1, C2, C3, C4 = prm[4], prm[5], prm[6], prm[7]
    e0, v0, r = prm[8], prm[9], prm[10]

    y0, y1, y2, y3, y4, y5 = y[0], y[1], y[2], y[3], y[4], y[5]

    S_pyr = _sigmoid(y1 - y2, e0, v0, r)   # pyramidal output drives interneurons
    S_exc = _sigmoid(C1 * y0, e0, v0, r)
    S_inh = _sigmoid(C3 * y0, e0, v0, r)

    d = np.empty(6)
    d[0] = y3
    d[3] = A * a * S_pyr - 2.0 * a * y3 - a * a * y0
    d[1] = y4
    d[4] = A * a * (p + C2 * S_exc) - 2.0 * a * y4 - a * a * y1
    d[2] = y5
    d[5] = B * b * C4 * S_inh - 2.0 * b * y5 - b * b * y2
    return d


@njit(cache=True, fastmath=True)
def _simulate_euler(prm, p_mean, p_std, dt, n_steps, y0, seed):
    """Euler-Maruyama integration with stochastic drive. Returns V = y1 - y2."""
    if seed >= 0:
        np.random.seed(seed)
    y = y0.copy()
    out = np.empty(n_steps)
    for t in range(n_steps):
        p = p_mean + p_std * np.random.randn() if p_std > 0.0 else p_mean
        d = _derivs(y, p, prm)
        for i in range(6):
            y[i] += dt * d[i]
        out[t] = y[1] - y[2]
    return out


@njit(cache=True, fastmath=True)
def _simulate_rk4(prm, p_mean, dt, n_steps, y0):
    """Deterministic classical RK4 (noise off). Returns V = y1 - y2."""
    y = y0.copy()
    out = np.empty(n_steps)
    for t in range(n_steps):
        k1 = _derivs(y, p_mean, prm)
        k2 = _derivs(y + 0.5 * dt * k1, p_mean, prm)
        k3 = _derivs(y + 0.5 * dt * k2, p_mean, prm)
        k4 = _derivs(y + dt * k3, p_mean, prm)
        for i in range(6):
            y[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        out[t] = y[1] - y[2]
    return out


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def simulate(params: JRParams = None, B=None, T=8.0, dt=DT,
             method="euler", stochastic=True, seed=-1, y0=None,
             drop_transient=True):
    """Simulate the Jansen-Rit column and return (t, V_eeg).

    Parameters
    ----------
    params : JRParams        base parameter set (defaults if None)
    B      : float or None   override the inhibitory gain (control parameter)
    T      : float           duration in seconds
    method : {"euler","rk4"} integrator
    stochastic : bool        include Gaussian cortical drive (euler only)
    seed   : int             RNG seed (-1 = nondeterministic)
    y0     : array(6)        initial state (small random if None)
    drop_transient : bool    discard the initial TRANSIENT_S seconds
    """
    if params is None:
        params = JRParams()
    if B is not None:
        params = JRParams(**{**params.__dict__, "B": B})
    prm = params.as_array()
    n_steps = int(round(T / dt))

    if y0 is None:
        rng = np.random.default_rng(None if seed < 0 else seed)
        y0 = rng.normal(0.0, 1.0, size=6) * np.array([1, 1, 1, 5, 5, 5.0])
    y0 = np.asarray(y0, dtype="float64")

    if method == "rk4":
        V = _simulate_rk4(prm, params.p_mean, dt, n_steps, y0)
    elif not stochastic:
        V = _simulate_euler(prm, params.p_mean, 0.0, dt, n_steps, y0, seed)
    else:
        V = _simulate_euler(prm, params.p_mean, params.p_std, dt, n_steps, y0, seed)

    t = np.arange(n_steps) * dt
    if drop_transient:
        k = int(round(TRANSIENT_S / dt))
        t, V = t[k:] - t[k], V[k:]
    return t, V


def warmup():
    """Trigger JIT compilation once so later timing is representative."""
    simulate(T=0.2, method="euler", seed=0, drop_transient=False)
    simulate(T=0.2, method="rk4", drop_transient=False)


if __name__ == "__main__":
    import time
    warmup()
    t0 = time.time()
    t, V = simulate(B=22.0, T=10.0, method="euler", seed=1)
    dt_run = time.time() - t0
    print(f"Simulated 10 s of EEG in {dt_run*1000:.1f} ms "
          f"({10.0/dt_run:.0f}x real time)")
    print(f"V range: [{V.min():.2f}, {V.max():.2f}] mV, "
          f"mean {V.mean():.2f}, std {V.std():.2f}")
