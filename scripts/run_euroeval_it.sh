#!/usr/bin/env bash
# EuroEval Italian PROXY run: the 7 Italian datasets with ONE iteration each (official protocol is
# 10; at ~22 min/iteration on this laptop the official run is ~25 h, the proxy ~2.5 h). Proxy scores
# are for base-vs-ours deltas only, NOT leaderboard-eligible. Iteration 0 of squad-it on the base
# reproduced the leaderboard number (F1 43.7 vs 42.7 published), so the proxy tracks the official.
# Usage: run_euroeval_it.sh MODEL_DIR ARM_NAME [NUM_ITER]
set -euo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
source scripts/euroeval-env.sh
MODEL=$1; ARM=$2; ITER=${3:-1}
OUT=outputs/euroeval/$ARM; mkdir -p $OUT; cd $OUT
for ds in squad-it multinerd-it scala-it sentipolc16 mmlu-it hellaswag-it ilpost-sum; do
  [ -f $ds.done ] && continue
  $EUROEVAL -m $MODEL --dataset $ds --device cuda --num-iterations $ITER --generative-type instruction_tuned \
    --verbose > $ds.log 2>&1 && touch $ds.done
  grep -o "Test scores for iteration 0: {[^}]*}" $ds.log | tail -1 | sed "s/^/$ds /"
done
echo "results: $OUT/euroeval_benchmark_results.jsonl"
