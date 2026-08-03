"""
dbs_hfs.py — the CORRECT mechanism test: true high-frequency pulsatile DBS.

The smoothed additive drive in dbs_additive.py could not suppress because a DC
inhibitory offset shifts the fixed point without changing the loop gain that
creates the limit cycle. But that is not how DBS works. The literature
mechanism (Mina 2013) is a HIGH-FREQUENCY PULSE TRAIN whose suppression emerges
dynamically. Here we inject an actual 130 Hz depolarizing pulse train into the
pyramidal excitatory drive and ask whether it annihilates the seizure limit
cycle -- and if so, whether the energy story survives this literature-faithful
mechanism.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from numba import njit
from config import JRParams, DT
from jansen_ritt import _derivs, warmup
from dbs_sim import SeizureScenario, sense, score
from scipy import signal as sps

@njit(cache=True, fastmath=True)
def _seg_hfs(prm, p_mean, p_std, dt, n, y, amp, f_dbs, phase0):
    """Additive HIGH-FREQUENCY pulse train into the excitatory drive p.
    amp = pulse amplitude (pulses/s added during a pulse), f_dbs = Hz.
    Returns (V trace, phase at end)."""
    out = np.empty(n)
    period = 1.0 / f_dbs
    pw = 0.001                      # 1 ms monophasic pulse
    phase = phase0
    for k in range(n):
        phase += dt
        if phase >= period:
            phase -= period
        pulse = amp if phase < pw else 0.0
        p = p_mean + p_std * np.random.randn() if p_std > 0.0 else p_mean
        pin = p + pulse             # DBS depolarizes the pyramidal input
        # inline derivs with modified excitatory input on d[4]
        d = _derivs(y, pin, prm)
        for i in range(6):
            y[i] += dt * d[i]
        out[k] = y[1] - y[2]
    return out, phase

def bandpower(V, fs=1000):
    f,P = sps.welch(V-V.mean(), fs=fs, nperseg=min(len(V),1024))
    m=(f>=2)&(f<=8); return float(np.trapz(P[m],f[m]))

def run_hfs(amp, f_dbs, scen, seed=0, g=None):
    prm=JRParams().as_array(); p_mean,p_std=JRParams().p_mean,JRParams().p_std
    n_seg=100; n_steps=int(scen.T_s/0.1)
    np.random.seed(seed)
    y=np.random.randn(6)*np.array([1.,1.,1.,5.,5.,5.])
    prm[1]=scen.B_disease(0.0)
    _s,ph=_seg_hfs(prm,p_mean,p_std,DT,4000,y,0.0,f_dbs,0.0)
    V=np.empty(n_steps*n_seg); on=np.zeros(n_steps); ph=0.0
    for i in range(n_steps):
        t=i*0.1; prm[1]=scen.B_disease(t)
        stim_on = scen.onset_s<=t<=scen.offset_s   # open-loop during ictal window
        a = amp if stim_on else 0.0
        seg,ph=_seg_hfs(prm,p_mean,p_std,DT,n_seg,y,a,f_dbs,ph)
        V[i*n_seg:(i+1)*n_seg]=seg; on[i]=1.0 if a>0 else 0.0
    return V,on

if __name__=="__main__":
    warmup()
    scen=SeizureScenario(onset_s=10.0,offset_s=45.0,T_s=60.0)
    # baseline seizure band power (no stim)
    Vb,_=run_hfs(0.0,130.0,scen,seed=1)
    fs=1000; ictal=slice(int(15*fs),int(44*fs))
    base=bandpower(Vb[ictal])
    print(f"untreated ictal band power = {base:.3f}")
    print("HFS amplitude & frequency sweep (fraction of untreated band power in ictal window):")
    print(f"{'amp':>6s}"+"".join(f"{f:>8.0f}Hz" for f in [20,60,130,180]))
    for amp in [100,200,400,800,1600]:
        row=f"{amp:6d}"
        for f in [20,60,130,180]:
            V,_=run_hfs(float(amp),float(f),scen,seed=1)
            row+=f"{bandpower(V[ictal])/base:9.3f}"
        print(row,flush=True)
    print("HFS_DONE")
