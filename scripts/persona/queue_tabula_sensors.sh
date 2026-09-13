#!/usr/bin/env bash
# Step 3: "base vs ours" sensors on Tabula voce v2 — calibration answers (36 probes, no system prompt,
# same as the base run) and the EuroEval Italian proxy (1 iteration) on the MERGED model.
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
PY=.venv-q35/bin/python
until grep -q "TABULA V2 DONE" outputs/persona/queue-v2.log 2>/dev/null; do sleep 60; done
echo "=== calibration tabula-v2 $(date +%H:%M)"
$PY scripts/voice/gen_calibration.py --name tabula-v2 --torch models/hf/Qwen3.5-4B --adapter artifacts/adapter-tabula-voce-v2 --out outputs/calib/tabula-v2.json > outputs/calib/gen-tabula-v2.log 2>&1 && tail -1 outputs/calib/gen-tabula-v2.log || echo "FAILED calibration"
echo "=== merge tabula-v2 $(date +%H:%M)"
[ -f artifacts/adapter-tabula-voce-v2-merged/config.json ] || $PY scripts/voice/merge_voice.py models/hf/Qwen3.5-4B artifacts/adapter-tabula-voce-v2 artifacts/adapter-tabula-voce-v2-merged > outputs/persona/merge-v2.log 2>&1 || echo "FAILED merge"
echo "=== euroeval proxy tabula-v2 $(date +%H:%M)"
bash scripts/run_euroeval_it.sh /home/neuromancer/Projects/ardesia-gguf/artifacts/adapter-tabula-voce-v2-merged tabula-v2 1 > outputs/euroeval/run-tabula-v2.log 2>&1; grep -E "^(squad|multinerd|scala|sentipolc|mmlu|hellaswag|ilpost)" outputs/euroeval/run-tabula-v2.log
echo "=== SENSORS DONE $(date +%H:%M)"
