#!/usr/bin/env bash
# lm-eval on a MERGED checkpoint (base layout via merge_voice.py + fix_merged_keys.py).
# Why: lm-eval `peft=` loads the base as Qwen3_5ForCausalLM and PEFT silently drops every Qwen3.5
# adapter key (0/128 lora_B applied, measured 13/09 06:20) -> all "adapter" numbers were the base.
# Usage: eval_merged.sh NAME "task shots" ["task shots" ...]     e.g. eval_merged.sh conoscenza "arc_it 0" "m_mmlu_it 5"
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
NAME=$1; shift
PY=.venv-q35/bin/python; BASE=models/hf/Qwen3.5-4B
LORA=artifacts/adapter-$NAME; MERGED=artifacts/adapter-$NAME-merged
OUT=outputs/ita-bench/adapter-$NAME-merged; mkdir -p $OUT
[ -f $MERGED/config.json ] || $PY scripts/voice/merge_voice.py $BASE $LORA $MERGED > $OUT/merge.log 2>&1 || { echo "FAILED merge $NAME"; exit 1; }
for spec in "$@"; do set -- $spec
  [ -f $OUT/$1.done ] || { .venv-q35/bin/lm_eval run --model hf --model_args "pretrained=$MERGED,dtype=bfloat16" \
      --tasks $1 --num_fewshot $2 --batch_size auto --output_path $OUT/$1 > $OUT/$1.log 2>&1 && touch $OUT/$1.done; }
  grep -E "^\|" $OUT/$1.log | tail -2
done
