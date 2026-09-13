#!/usr/bin/env bash
# Queue 6 — the ONLY GPU scheduler. Waits for the ghost run_adapter (sensocomune2 training, pid $1).
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
WAITPID=$1
PY=.venv-q35/bin/python; LC=/home/neuromancer/Projects/llama.cpp-oracle-next; BASE=models/hf/Qwen3.5-4B
step() { echo "=== $1 $(date +%H:%M)"; }
while kill -0 $WAITPID 2>/dev/null; do sleep 60; done
sleep 20
step "eval conoscenza (merged)";   bash scripts/bench/eval_merged.sh conoscenza "arc_it 0" "m_mmlu_it 5" || echo "FAILED eval conoscenza"
step "eval sensocomune2 (merged)"; bash scripts/bench/eval_merged.sh sensocomune2 "hellaswag_it 0" || echo "FAILED eval sensocomune2"
step "eval sensocomune v1 (merged)"; bash scripts/bench/eval_merged.sh sensocomune "hellaswag_it 0" || echo "FAILED eval sensocomune"
step "baseline math2"; $PY scripts/rlvr/eval_rlvr.py --task math2 --name base --n 300 --max-new 640 > outputs/rlvr/eval-math2-base.log 2>&1 && grep "^ACC" outputs/rlvr/eval-math2-base.log || echo "FAILED eval-math2-base"
step "grpo math2";     [ -f artifacts/adapter-matematica2/adapter_config.json ] || $PY scripts/rlvr/train_grpo.py --task math2 --steps 200 --max-completion 512 --out artifacts/adapter-matematica2 > outputs/rlvr/train-math2.log 2>&1 && grep -E "train done|peak VRAM|reward trajectory" outputs/rlvr/train-math2.log || echo "FAILED grpo-math2"
step "eval math2";     [ -f artifacts/adapter-matematica2/adapter_config.json ] && { $PY scripts/rlvr/eval_rlvr.py --task math2 --name matematica2 --adapter artifacts/adapter-matematica2 --n 300 --max-new 640 > outputs/rlvr/eval-math2-matematica2.log 2>&1 && grep "^ACC" outputs/rlvr/eval-math2-matematica2.log || echo "FAILED eval-math2-matematica2"; }
step "gguf";           for a in matematica2 sensocomune2; do [ -f artifacts/adapter-$a/adapter_config.json ] && [ ! -f artifacts/gguf/adapter-$a.gguf ] && PYTHONPATH=$LC/gguf-py $PY $LC/convert_lora_to_gguf.py artifacts/adapter-$a --base $BASE --outtype f16 --outfile artifacts/gguf/adapter-$a.gguf > /dev/null 2>&1 && echo "gguf $a ok"; done
step "QUEUE6 DONE"
