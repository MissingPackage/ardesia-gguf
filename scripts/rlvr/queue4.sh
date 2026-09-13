#!/usr/bin/env bash
# Queue 4 (after queue 3): Matematica v2 on MATH (train 7361 boxed, test MATH-500), GRPO 200 steps.
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
PY=.venv-q35/bin/python; LC=/home/neuromancer/Projects/llama.cpp-oracle-next; BASE=models/hf/Qwen3.5-4B
step() { echo "=== $1 $(date +%H:%M)"; }
until grep -q "QUEUE3 DONE" outputs/adapters/queue3.log 2>/dev/null; do sleep 120; done
step "baseline math2"; $PY scripts/rlvr/eval_rlvr.py --task math2 --name base --n 300 --max-new 640 > outputs/rlvr/eval-math2-base.log 2>&1 && grep "^ACC" outputs/rlvr/eval-math2-base.log || echo "FAILED eval-math2-base"
step "grpo math2";     [ -f artifacts/adapter-matematica2/adapter_config.json ] || $PY scripts/rlvr/train_grpo.py --task math2 --steps 200 --max-completion 512 --out artifacts/adapter-matematica2 > outputs/rlvr/train-math2.log 2>&1 && grep -E "train done|peak VRAM|reward trajectory" outputs/rlvr/train-math2.log || echo "FAILED grpo-math2"
step "eval math2";     [ -f artifacts/adapter-matematica2/adapter_config.json ] && { $PY scripts/rlvr/eval_rlvr.py --task math2 --name matematica2 --adapter artifacts/adapter-matematica2 --n 300 --max-new 640 > outputs/rlvr/eval-math2-matematica2.log 2>&1 && grep "^ACC" outputs/rlvr/eval-math2-matematica2.log || echo "FAILED eval-math2-matematica2"; }
step "gguf";           [ -f artifacts/adapter-matematica2/adapter_config.json ] && [ ! -f artifacts/gguf/adapter-matematica2.gguf ] && PYTHONPATH=$LC/gguf-py $PY $LC/convert_lora_to_gguf.py artifacts/adapter-matematica2 --base $BASE --outtype f16 --outfile artifacts/gguf/adapter-matematica2.gguf > /dev/null 2>&1 && echo "gguf matematica2 ok"
step "QUEUE4 DONE"
