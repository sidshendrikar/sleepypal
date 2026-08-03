#!/bin/bash
# Train + cache held-out probabilities for the new patients (3 concurrent,
# thread-limited so they share the CPU), then pool ALL patients.
cd /Users/siddharthashendrikar/neuropal/epilepsy_scripts || exit 1
NEW="chb02 chb04 chb05 chb08 chb10"
MAXJ=3
echo "[cohort] start $(date +%H:%M)"
for p in $NEW; do
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJ" ]; do sleep 5; done
  echo "[cohort] launching $p"
  TF_THREADS=3 PATIENT=$p python3 improve_eval.py > /tmp/imp_$p.log 2>&1 &
  sleep 3
done
wait
echo "[cohort] all trainings finished $(date +%H:%M)"
for p in $NEW; do
  ok=$(grep -c PROBS_DONE /tmp/imp_$p.log 2>/dev/null)
  echo "  $p: $([ "$ok" = 1 ] && echo OK || echo 'CHECK /tmp/imp_'$p'.log')"
done
echo "[cohort] pooling all patients..."
python3 analyze_probs.py > /tmp/cohort_analysis.log 2>&1
echo "COHORT_DONE"
