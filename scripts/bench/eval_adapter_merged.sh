#!/usr/bin/env bash
# lm-eval on a MERGED adapter dir. Why not `peft=`: lm-eval's hf backend loads the base with
# AutoModelForCausalLM (Qwen3_5ForCausalLM, keys model.layers.*) while the adapters were trained on
# Qwen3_5ForConditionalGeneration (keys model.language_model.layers.*) -> PEFT silently ignores every
# adapter tensor and you measure the BASE (found 13/09: m_mmlu_it identical to 4 decimals).
# Usage: eval_adapter_merged.sh NAME   (NAME in conoscenza|sensocomune|sensocomune2)
set -euo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
NAME=$1; PY=.venv-q35/bin/python; BASE=models/hf/Qwen3.5-4B
LORA=artifacts/adapter-$NAME; MERGED=artifacts/adapter-$NAME-merged
[ -f $MERGED/config.json ] || $PY scripts/voice/merge_voice.py $BASE $LORA $MERGED > outputs/adapters/merge-$NAME.log 2>&1
OUT=outputs/ita-bench/adapter-$NAME-merged; mkdir -p $OUT
case $NAME in
  conoscenza) SPECS=("arc_it 0" "m_mmlu_it 5") ;;
  sensocomune|sensocomune2) SPECS=("hellaswag_it 0") ;;
  *) echo "unknown $NAME"; exit 1 ;;
esac
for spec in "${SPECS[@]}"; do set -- $spec
  [ -f $OUT/$1.done ] || { .venv-q35/bin/lm_eval run --model hf --model_args "pretrained=$MERGED,dtype=bfloat16" \
     --tasks $1 --num_fewshot $2 --batch_size auto --output_path $OUT/$1 > $OUT/$1.log 2>&1 && touch $OUT/$1.done; }
  grep -E "^\|" $OUT/$1.log | tail -2
done
echo "DONE-MERGED $NAME"
