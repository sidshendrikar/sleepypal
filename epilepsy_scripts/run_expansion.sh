#!/bin/bash
cd /Users/siddharthashendrikar/neuropal/epilepsy_scripts || exit 1
NEW="chb02 chb04 chb05 chb08 chb10"
# 1. wait for all downloads
until [ "$(for p in $NEW; do grep -q "\[$p\] DONE" /tmp/dl_$p.log 2>/dev/null && echo x; done | wc -l | tr -d ' ')" = "5" ]; do sleep 20; done
echo "=== downloads complete ==="
for p in $NEW; do grep "\] DONE" /tmp/dl_$p.log | sed 's/^/  /'; done
# 2. rebuild manifest over ALL on-disk seizure files
python3 - <<'PY'
import os, json, glob
from config import DATA_DIR
from fetch_data import parse_summary
ann = {}
for sp in glob.glob(os.path.join(DATA_DIR, "*-summary.txt")):
    pat = os.path.basename(sp).split("-")[0]
    with open(sp) as f:
        for fn, iv in parse_summary(f.read()).items():
            ann[fn] = {"patient": pat, "seizures": iv}
present = {os.path.basename(p) for p in glob.glob(os.path.join(DATA_DIR, "*.edf"))}
man = {fn: m for fn, m in ann.items() if fn in present and m["seizures"]}
json.dump(man, open(os.path.join(DATA_DIR, "seizures.json"), "w"), indent=2)
from collections import Counter
c = Counter(m["patient"] for m in man.values())
print("MANIFEST:", dict(sorted(c.items())), "| seizures:",
      sum(len(m["seizures"]) for m in man.values()))
PY
# 3. train the new patients, 3 concurrent, thread-limited
echo "=== training ==="
for p in $NEW; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge 3 ]; do sleep 5; done
  echo "  launching $p"
  TF_THREADS=3 PATIENT=$p nohup python3 improve_eval.py > /tmp/imp_$p.log 2>&1 &
  sleep 3
done
wait
for p in $NEW; do
  ok=$(grep -c PROBS_DONE /tmp/imp_$p.log 2>/dev/null)
  echo "  $p: $([ "$ok" = 1 ] && echo OK || echo FAILED)"
done
# 4. pool all seven patients
echo "=== pooled 7-patient analysis ==="
python3 analyze_probs.py 2>&1 | grep -vE "warn|Warn|oneDNN|cpu_feature|rebuild|Instructions|tensorflow/core|external/local"
echo "EXPANSION_DONE"
