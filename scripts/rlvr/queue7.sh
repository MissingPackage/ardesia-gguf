#!/usr/bin/env bash
# Queue 7: strictly after QUEUE6 DONE, the conoscenza merged eval (its queue-6 step was corrupted by a
# concurrent edit). Single waiter; log outputs/rlvr/queue7.log.
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
until grep -q "QUEUE6 DONE" outputs/rlvr/queue6.log 2>/dev/null; do sleep 60; done
echo "=== eval conoscenza (merged, redo) $(date +%H:%M)"
bash scripts/bench/eval_merged.sh conoscenza "arc_it 0" "m_mmlu_it 5" || echo "FAILED eval conoscenza"
echo "QUEUE7 DONE $(date +%H:%M)"
