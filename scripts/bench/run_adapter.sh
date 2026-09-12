#!/usr/bin/env bash
# One benchmark adapter end to end: train LoRA (bf16 base) -> LoRA GGUF (hot-swappable on the Q4_0
# base) -> evaluate with the benchmark's own harness, base+adapter unmerged.
#   lettura / entita     -> EuroEval proxy (1 iteration) on a bf16 MERGED dir (EuroEval takes a model
#                           dir; merge==unmerged was measured on 12/09)
#   conoscenza / sensocomune -> lm-eval hf with peft=ADAPTER (unmerged), mii-llm protocol
# Usage: run_adapter.sh NAME [EPOCHS] [LR]
set -euo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
NAME=$1; EPOCHS=${2:-2}; LR=${3:-1e-4}
PY=.venv-q35/bin/python
BASE=models/hf/Qwen3.5-4B
LORA=artifacts/adapter-$NAME
LC=/home/neuromancer/Projects/llama.cpp-oracle-next
mkdir -p outputs/adapters artifacts/gguf
case $NAME in
  lettura|entita) FMT=chat ;;
  conoscenza|sensocomune|sensocomune2) FMT=raw ;;
  *) echo "unknown adapter $NAME"; exit 1 ;;
esac

if [ ! -f $LORA/adapter_config.json ]; then
  $PY scripts/voice/train_voice_lora.py --data data/bench/adapter-$NAME.jsonl --out $LORA \
     --format $FMT --epochs $EPOCHS --lr $LR > outputs/adapters/train-$NAME.log 2>&1
  grep -E "train done|peak VRAM" outputs/adapters/train-$NAME.log
fi
[ -f artifacts/gguf/adapter-$NAME.gguf ] || PYTHONPATH=$LC/gguf-py $PY $LC/convert_lora_to_gguf.py $LORA --base $BASE \
     --outtype f16 --outfile artifacts/gguf/adapter-$NAME.gguf > /dev/null 2>&1

case $NAME in
  conoscenza)
    OUT=outputs/ita-bench/adapter-$NAME; mkdir -p $OUT
    for spec in "arc_it 0" "m_mmlu_it 5"; do set -- $spec
      [ -f $OUT/$1.done ] || { .venv-q35/bin/lm_eval run --model hf \
        --model_args "pretrained=$BASE,peft=$LORA,dtype=bfloat16" --tasks $1 --num_fewshot $2 \
        --batch_size auto --output_path $OUT/$1 > $OUT/$1.log 2>&1 && touch $OUT/$1.done; }
      grep -E "^\|" $OUT/$1.log | tail -2
    done ;;
  sensocomune|sensocomune2)
    OUT=outputs/ita-bench/adapter-$NAME; mkdir -p $OUT
    [ -f $OUT/hellaswag_it.done ] || { .venv-q35/bin/lm_eval run --model hf \
        --model_args "pretrained=$BASE,peft=$LORA,dtype=bfloat16" --tasks hellaswag_it --num_fewshot 0 \
        --batch_size auto --output_path $OUT/hellaswag_it > $OUT/hellaswag_it.log 2>&1 && touch $OUT/hellaswag_it.done; }
    grep -E "^\|" $OUT/hellaswag_it.log | tail -2 ;;
  lettura|entita)
    MERGED=artifacts/adapter-$NAME-merged
    [ -f $MERGED/config.json ] || $PY scripts/voice/merge_voice.py $BASE $LORA $MERGED > /dev/null 2>&1
    DS=$([ $NAME = lettura ] && echo squad-it || echo multinerd-it)
    source scripts/euroeval-env.sh
    OUT=outputs/euroeval/adapter-$NAME; mkdir -p $OUT; cd $OUT
    [ -f $DS.done ] || { $EUROEVAL -m /home/neuromancer/Projects/ardesia-gguf/$MERGED --dataset $DS --device cuda \
        --num-iterations 1 --generative-type instruction_tuned --verbose > $DS.log 2>&1 && touch $DS.done; }
    grep -o "Test scores for iteration 0: {[^}]*}" $DS.log | tail -1 ;;
esac
echo "DONE $NAME"
