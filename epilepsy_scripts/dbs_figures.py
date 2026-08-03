"""dbs_figures.py — figures for the adaptive-DBS benchmark."""
import json, os
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ARTIFACT_DIR
from bifurcation_analysis import find_bifurcations
from jansen_ritt import warmup
from dbs_sim import ClosedLoopDBS, SeizureScenario, score
from dbs_controllers import (NoStim, OpenLoopDBS, ThresholdADBS, PerceptST,
                             BifurcationMPC)
from dbs_benchmark import calibrate, GAIN

GREEN, ORANGE, BLUE, GREY = "tab:green", "tab:orange", "tab:blue", "0.35"


def main():
    warmup()
    Bc = find_bifurcations()["B_crit"]
    loop = ClosedLoopDBS(B_crit=Bc, gain=GAIN)
    det, sco, _ = calibrate(loop)
    scen = SeizureScenario()
    Bh = scen.B_ictal

    # ---- Fig 1: traces ---------------------------------------------------
    arms = [(NoStim(), GREY, "untreated"),
            (ThresholdADBS(det, 1.0, min_burst_s=2.0), ORANGE,
             "threshold aDBS (deployed-style)"),
            (BifurcationMPC(Bc, GAIN, Bh, det, margin=1.05, hold_s=2.0),
             GREEN, "bifurcation MPC (this work)")]
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.2), sharex=True)
    for (ctrl, col, label), ax in zip(arms, axes):
        log = loop.run(ctrl, scen, seed=1)
        s = score(log, seizure_thresh=sco)
        tv = np.linspace(0, scen.T_s, len(log["V"]))
        ax.axvspan(scen.onset_s, scen.offset_s, color="red", alpha=0.07,
                   label="seizure driver active")
        ax.plot(tv, log["V"], color=col, lw=0.35)
        ax.set_ylabel("EEG (mV)")
        ax.set_title(f"{label}   —   burden {s['seizure_burden_s']:.1f}s, "
                     f"energy {s['energy_mJ']:.2f} mJ", fontsize=10)
        ax2 = ax.twinx()
        ax2.plot(log["t"], log["u"], color="tab:red", lw=1.4, alpha=0.85)
        ax2.set_ylim(-0.05, 1.3); ax2.set_ylabel("stim amplitude", color="tab:red")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "dbs_fig1_traces.png"), dpi=130)
    plt.close(fig)

    # ---- Fig 2: realistic duty ------------------------------------------
    p = os.path.join(ARTIFACT_DIR, "dbs_realistic.json")
    if os.path.exists(p):
        R = json.load(open(p))["realistic"]
        names = list(R.keys())
        short = [n.replace(" aDBS (Percept-style)", " DT")
                  .replace(" (this work)", "").replace(" (RNS-style)", "")
                  .replace("open-loop ", "OL ") for n in names]
        bur = [R[n]["seizure_burden_s"] for n in names]
        eng = [R[n]["energy_mJ"] for n in names]
        cols = [GREEN if "bifurcation" in n else BLUE for n in names]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].barh(short, bur, color=cols); ax[0].set_xlabel("residual burden (s)")
        ax[0].set_title("Seizure burden (10-min record)")
        ax[1].barh(short, eng, color=cols); ax[1].set_xlabel("energy delivered (mJ)")
        ax[1].set_title("Stimulation energy")
        for a in ax: a.grid(alpha=0.3, axis="x"); a.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(ARTIFACT_DIR, "dbs_fig2_realistic.png"), dpi=130)
        plt.close(fig)

        # ---- Fig 3: energy-burden plane -------------------------------
        plt.figure(figsize=(6.6, 4.8))
        for n, s_, b, e in zip(names, short, bur, eng):
            c = GREEN if "bifurcation" in n else BLUE
            plt.scatter(e, b, s=70, color=c, zorder=3)
            plt.annotate(s_, (e, b), textcoords="offset points",
                         xytext=(7, 4), fontsize=8)
        plt.xlabel("energy delivered (mJ)"); plt.ylabel("residual burden (s)")
        plt.title("Lower-left is better (10-min record)")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, "dbs_fig3_plane.png"), dpi=130)
        plt.close()

    # ---- Fig 4: lead-time ------------------------------------------------
    p = os.path.join(ARTIFACT_DIR, "dbs_leadtime.json")
    if os.path.exists(p):
        L = json.load(open(p))
        rows = L["lead_sweep"]
        lead = [r["lead_s"] for r in rows]
        bur = [r["burden_s"] for r in rows]
        eng = [r["energy_mJ"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(6.8, 4.4))
        ax1.plot(lead, bur, "o-", color=GREEN, lw=2, label="seizure burden")
        ax1.set_xlabel("prediction lead time before onset (s)")
        ax1.set_ylabel("residual burden (s)", color=GREEN)
        ax1.axhline(0, color="k", lw=0.6)
        ax2 = ax1.twinx()
        ax2.plot(lead, eng, "s--", color=ORANGE, lw=1.6, label="energy")
        ax2.axhline(L["open_loop_mJ"], color=BLUE, ls=":", lw=1.5)
        ax2.text(0.3, L["open_loop_mJ"] * 0.93,
                 f"continuous DBS = {L['open_loop_mJ']:.0f} mJ",
                 color=BLUE, fontsize=8)
        ax2.set_ylabel("energy delivered (mJ)", color=ORANGE)
        ax2.set_yscale("log")
        ax1.set_title("Prediction converts detection latency into suppression")
        ax1.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(ARTIFACT_DIR, "dbs_fig4_leadtime.png"), dpi=130)
        plt.close(fig)
    print("FIGS_DONE")


if __name__ == "__main__":
    main()
