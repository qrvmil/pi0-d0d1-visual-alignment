#!/usr/bin/env bash
set -euo pipefail
source /root/pi0-d0d1-visual-alignment/scripts/env.sh
"$ALIGN_PY" - <<'PY'
import json
assert json.load(open('logs/data_training_smoke.json'))['status']=='passed'
PY
"$ALIGN_PY" -u -m alignment.build_cache > logs/cache_build.log 2>&1
if [[ ! -f private/projector.npz ]]; then
  "$ALIGN_PY" -u -m alignment.train calibrate > logs/projector_calibration.log 2>&1
fi
if [[ ! -f configs/lambda_decision.json ]]; then
  "$ALIGN_PY" -u -m alignment.train pilot > logs/pilot.log 2>&1
fi
"$ALIGN_PY" -u -m alignment.train B-control > logs/B-control.log 2>&1
"$ALIGN_PY" -u -m alignment.train B-align > logs/B-align.log 2>&1
for task in D0 D1 D2; do
  for arm in B-control B-align; do
    "$ALIGN_PY" -u -m alignment.evaluation --arm "$arm" --task "$task" > "logs/eval_${arm}_${task}.log" 2>&1
  done
done
"$ALIGN_PY" -m alignment.evaluation --summarize > logs/evaluation_summary.log 2>&1
if "$ALIGN_PY" - <<'PY'
import json,sys
sys.exit(0 if json.load(open('evaluation/results.json'))['D0']['extend_to500'] else 1)
PY
then
  for arm in B-control B-align; do
    "$ALIGN_PY" -u -m alignment.evaluation --arm "$arm" --task D0 --extended > "logs/eval_${arm}_D0_extended.log" 2>&1
  done
fi
"$ALIGN_PY" -m alignment.evaluation --summarize > logs/evaluation_summary.log 2>&1
"$ALIGN_PY" scripts/report.py
"$ALIGN_PY" -m pytest -q > logs/final_tests.log 2>&1
# Publication is a separate command after final audit of the generated report and artifacts.
