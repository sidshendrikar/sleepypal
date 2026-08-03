import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, os
from config import ARTIFACT_DIR
from bifurcation_analysis import find_bifurcations
from jansen_ritt import warmup
from dbs_sim import ClosedLoopDBS, SeizureScenario, score
from dbs_controllers import NoStim, OpenLoopDBS, BifurcationMPC, calibrate_threshold
from dbs_benchmark import calibrate, GAIN

warmup()
bif=find_bifurcations(); Bc=bif["B_crit"]
loop=ClosedLoopDBS(B_crit=Bc,gain=GAIN)
det,sco,_=calibrate(loop)
scen=SeizureScenario(onset_s=60.0, offset_s=95.0, T_s=600.0)
Bh=scen.B_ictal; seeds=[1,2,3,4,5]
print("[lead-time sweep] predictive trigger before electrographic onset")
print(f"{'lead s':>7s}{'burden s':>10s}{'energy mJ':>11s}{'duty %':>8s}")
rows=[]
for lead in [0.0,1.0,2.0,3.0,5.0,8.0,12.0]:
    rs=[score(loop.run(BifurcationMPC(Bc,GAIN,Bh,det,margin=1.05,hold_s=2.0,
                                      preictal_lead_s=lead),
                       scen,seed=s,preictal_lead_s=lead), seizure_thresh=sco)
        for s in seeds]
    b=float(np.mean([r["seizure_burden_s"] for r in rs]))
    e=float(np.mean([r["energy_mJ"] for r in rs]))
    d=float(np.mean([r["duty_cycle"] for r in rs]))
    rows.append({"lead_s":lead,"burden_s":b,"energy_mJ":e,"duty":d})
    print(f"{lead:7.1f}{b:10.1f}{e:11.2f}{d*100:8.1f}", flush=True)
ol=[score(loop.run(OpenLoopDBS(1.0),scen,seed=s),seizure_thresh=sco) for s in seeds]
ole=float(np.mean([r["energy_mJ"] for r in ol]))
best=min(rows,key=lambda r:r["burden_s"])
print(f"\nopen-loop continuous: burden 0.0s, energy {ole:.1f}mJ")
print(f"best predictive     : burden {best['burden_s']:.1f}s at lead {best['lead_s']:.0f}s, "
      f"energy {best['energy_mJ']:.2f}mJ ({100*(1-best['energy_mJ']/ole):.1f}% less)")
json.dump({"lead_sweep":rows,"open_loop_mJ":ole},
          open(os.path.join(ARTIFACT_DIR,"dbs_leadtime.json"),"w"),indent=2)
print("LEAD_DONE")
