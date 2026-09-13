#!/usr/bin/env bash
# Queue 3 (after night.sh): conoscenza evals (adapter trained, eval skipped on 13/09), senso comune v2.
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
until grep -q "NIGHT DONE" outputs/rlvr/night.log 2>/dev/null; do sleep 120; done
for a in conoscenza sensocomune2; do echo "=== $a $(date +%H:%M)"; bash scripts/bench/run_adapter.sh $a 2 1e-4 || echo "FAILED $a"; done
echo "QUEUE3 DONE $(date +%H:%M)"
