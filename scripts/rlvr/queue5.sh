#!/usr/bin/env bash
# Queue 5 (serialised): wait for the running conoscenza eval chain (pid from arg) to end, then
# senso comune v2, then matematica v2 on MATH. One GPU job at a time.
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
WAITPID=$1
PY=.venv-q35/bin/python; LC=/home/neuromancer/Projects/llama.cpp-oracle-next; BASE=models/hf/Qwen3.5-4B
step() { echo "=== $1 $(date +%H:%M)"; }
while kill -0 $WAITPID 2>/dev/null; do sleep 60; done
step "sensocomune2";  bash scripts/bench/run_adapter.sh sensocomune2 2 1e-4 || echo "FAILED sensocomune2"
step "baseline math2"; $PY scripts/rlvr/eval_rlvr.py --task math2 --name base --n 300 --max-new 640 > outputs/rlvr/eval-math2-base.log 2>&1 && grep "^ACC" outputs/rlvr/eval-math2-base.log || echo "FAILED eval-math2-base"
step "grpo math2";     [ -f artifacts/adapter-matematica2/adapter_config.json ] || $PY scripts/rlvr/train_grpo.py --task math2 --steps 200 --max-completion 512 --out artifacts/adapter-matematica2 > outputs/rlvr/train-math2.log 2>&1 && grep -E "train done|peak VRAM|reward trajectory" outputs/rlvr/train-math2.log || echo "FAILED grpo-math2"
step "eval math2";     [ -f artifacts/adapter-matematica2/adapter_config.json ] && { $PY scripts/rlvr/eval_rlvr.py --task math2 --name matematica2 --adapter artifacts/adapter-matematica2 --n 300 --max-new 640 > outputs/rlvr/eval-math2-matematica2.log 2>&1 && grep "^ACC" outputs/rlvr/eval-math2-matematica2.log || echo "FAILED eval-math2-matematica2"; }
step "gguf";           for a in matematica2 sensocomune2; do [ -f artifacts/adapter-$a/adapter_config.json ] && [ ! -f artifacts/gguf/adapter-$a.gguf ] && PYTHONPATH=$LC/gguf-py $PY $LC/convert_lora_to_gguf.py artifacts/adapter-$a --base $BASE --outtype f16 --outfile artifacts/gguf/adapter-$a.gguf > /dev/null 2>&1 && echo "gguf $a ok"; done
step "QUEUE5 DONE"
