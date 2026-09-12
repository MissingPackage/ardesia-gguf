#!/usr/bin/env bash
# Night queue 2026-09-13: RLVR adapters after the benchmark-adapter queue. Every step is guarded so
# one failure does not stop the chain; the wake-up loop reads outputs/rlvr/night.log and repairs.
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
PY=.venv-q35/bin/python; LC=/home/neuromancer/Projects/llama.cpp-oracle-next; BASE=models/hf/Qwen3.5-4B
mkdir -p outputs/rlvr artifacts/gguf
step() { echo "=== $1 $(date +%H:%M)"; }
until grep -q "QUEUE2 DONE" outputs/adapters/queue2.log 2>/dev/null; do sleep 60; done
step "baseline math";  $PY scripts/rlvr/eval_rlvr.py --task math --name base --n 200 > outputs/rlvr/eval-math-base.log 2>&1 && grep "^ACC" outputs/rlvr/eval-math-base.log || echo "FAILED eval-math-base"
step "baseline code";  $PY scripts/rlvr/eval_rlvr.py --task code --name base --n 200 > outputs/rlvr/eval-code-base.log 2>&1 && grep "^ACC" outputs/rlvr/eval-code-base.log || echo "FAILED eval-code-base"
step "grpo math";      [ -f artifacts/adapter-matematica/adapter_config.json ] || $PY scripts/rlvr/train_grpo.py --task math --steps 200 --out artifacts/adapter-matematica > outputs/rlvr/train-math.log 2>&1 && grep -E "train done|peak VRAM|reward trajectory" outputs/rlvr/train-math.log || echo "FAILED grpo-math"
step "eval math";      [ -f artifacts/adapter-matematica/adapter_config.json ] && { $PY scripts/rlvr/eval_rlvr.py --task math --name matematica --adapter artifacts/adapter-matematica --n 200 > outputs/rlvr/eval-math-matematica.log 2>&1 && grep "^ACC" outputs/rlvr/eval-math-matematica.log || echo "FAILED eval-math-matematica"; }
step "grpo code";      [ -f artifacts/adapter-codice/adapter_config.json ] || $PY scripts/rlvr/train_grpo.py --task code --steps 150 --max-completion 512 --out artifacts/adapter-codice > outputs/rlvr/train-code.log 2>&1 && grep -E "train done|peak VRAM|reward trajectory" outputs/rlvr/train-code.log || echo "FAILED grpo-code"
step "eval code";      [ -f artifacts/adapter-codice/adapter_config.json ] && { $PY scripts/rlvr/eval_rlvr.py --task code --name codice --adapter artifacts/adapter-codice --n 200 > outputs/rlvr/eval-code-codice.log 2>&1 && grep "^ACC" outputs/rlvr/eval-code-codice.log || echo "FAILED eval-code-codice"; }
step "gguf";           for a in matematica codice; do [ -f artifacts/adapter-$a/adapter_config.json ] && [ ! -f artifacts/gguf/adapter-$a.gguf ] && PYTHONPATH=$LC/gguf-py $PY $LC/convert_lora_to_gguf.py artifacts/adapter-$a --base $BASE --outtype f16 --outfile artifacts/gguf/adapter-$a.gguf > /dev/null 2>&1 && echo "gguf $a ok"; done
step "NIGHT DONE"
