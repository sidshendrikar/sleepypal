#!/bin/bash
# Autonomous finisher: wait for downloads → expand CHB-MIT cohort → adult zero-shot.
cd /Users/siddharthashendrikar/neuropal/epilepsy_scripts || exit 1
DATA=../data/epilepsy
NEW="chb02 chb04 chb05 chb08 chb10"
log(){ echo "[finish $(date +%H:%M:%S)] $*"; }

# 1. wait for the 5 CHB-MIT downloads
log "waiting for CHB-MIT downloads…"
until [ "$(for p in $NEW; do grep -q "\[$p\] DONE" /tmp/dl_$p.log 2>/dev/null && echo x; done | wc -l | tr -d ' ')" = "5" ]; do sleep 20; done
log "CHB-MIT downloads complete"

# 2. rebuild full manifest (all on-disk seizure files)
python3 - <<'PY'
import os, json, glob
from config import DATA_DIR
from fetch_data import parse_summary
ann={}
for sp in glob.glob(os.path.join(DATA_DIR,"*-summary.txt")):
    pt=os.path.basename(sp).split("-")[0]
    for fn,iv in parse_summary(open(sp).read()).items(): ann[fn]={"patient":pt,"seizures":iv}
present={os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR,"*.edf"))}
man={fn:m for fn,m in ann.items() if fn in present and m["seizures"]}
json.dump(man,open(os.path.join(DATA_DIR,"seizures.json"),"w"),indent=2)
from collections import Counter
print("[manifest]",dict(sorted(Counter(m['patient'] for m in man.values()).items())),
      "seizures=",sum(len(m['seizures']) for m in man.values()))
PY

# 3. train the 5 new patients (3 concurrent, thread-limited) then pool ALL patients
log "training new patients + pooling…"
bash run_cohort.sh
log "cohort pooling done → /tmp/cohort_analysis.log"

# 4. wait for Siena PN00 (adult zero-shot set): 5 files fully downloaded (>70MB)
log "waiting for Siena PN00…"
until [ "$(ls -la $DATA/../siena/PN00-*.edf 2>/dev/null | awk '$5>70000000' | wc -l | tr -d ' ')" -ge 5 ]; do sleep 20; done
log "Siena PN00 complete"
rm -f ../data/siena/siena_dataset.npz

# 5. zero-shot cross-population evaluation
log "running adult zero-shot cross_eval…"
python3 cross_eval.py > /tmp/cross_eval.log 2>&1
log "cross_eval done → /tmp/cross_eval.log"

echo "================ FINAL POOLED (within-population) ================"
sed -n '/variant/,/recommended operating point/p' /tmp/cohort_analysis.log | head -40
echo "================ ADULT ZERO-SHOT (cross-population) =============="
grep -A6 "ADULT SIENA" /tmp/cross_eval.log
echo "ALL_DONE"
