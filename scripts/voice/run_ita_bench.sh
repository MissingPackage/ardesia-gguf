#!/usr/bin/env bash
# Italian baseline with lm-eval 0.4.13 (HF backend, bf16, single 4090). Same protocol as the
# Open Ita LLM Leaderboard (src/about.py, read 2026-09-12): m_mmlu_it 5-shot acc, arc_it 0-shot
# acc_norm, hellaswag_it 0-shot acc_norm.
# Usage: run_ita_bench.sh MODEL_DIR ARM_NAME [LIMIT]   (LIMIT optional, for smoke runs)
set -euo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
MODEL=$1; ARM=$2; LIMIT=${3:-}
OUT=outputs/ita-bench/$ARM; mkdir -p $OUT
LIM=(); [ -n "$LIMIT" ] && LIM=(--limit "$LIMIT")
run() { local task=$1 shots=$2
  [ -f $OUT/$task.done ] && return
  .venv-q35/bin/lm_eval run --model hf --model_args "pretrained=$MODEL,dtype=bfloat16,trust_remote_code=False" \
    --tasks $task --num_fewshot $shots --batch_size auto --output_path $OUT/$task "${LIM[@]}" \
    > $OUT/$task.log 2>&1 && touch $OUT/$task.done
  grep -E "^\|" $OUT/$task.log | tail -4
}
run arc_it 0
run hellaswag_it 0
run m_mmlu_it 5
