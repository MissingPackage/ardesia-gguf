#!/usr/bin/env bash
# Tabula voce v6 = v5 data minus the 188 rows that name the base (omission, not denial; Cristiano 15/09).
# train -> adapter gguf -> merge -> bf16 gguf (+MTP graft) -> Q4_0 fused file for the webgguf demo.
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
PY=.venv-q35/bin/python; LC=/home/neuromancer/Projects/llama.cpp-oracle-next; G=artifacts/gguf
step() { echo "=== $1 $(date +%H:%M:%S)"; }
step "train v6"; [ -f artifacts/adapter-tabula-voce-v6/adapter_config.json ] || $PY scripts/voice/train_voice_lora.py --data data/identity/persona-tabula-v6.jsonl --out artifacts/adapter-tabula-voce-v6 --epochs 2 --lr 1.5e-4 > outputs/persona/train-tabula-voce-v6.log 2>&1 || { echo "FAILED train"; exit 1; }
grep -E "train done|peak VRAM" outputs/persona/train-tabula-voce-v6.log
step "adapter gguf"; [ -f $G/adapter-tabula-voce-v6.gguf ] || PYTHONPATH=$LC/gguf-py $PY $LC/convert_lora_to_gguf.py artifacts/adapter-tabula-voce-v6 --base models/hf/Qwen3.5-4B --outtype f16 --outfile $G/adapter-tabula-voce-v6.gguf > outputs/persona/lora-gguf-v6.log 2>&1 || { echo "FAILED lora gguf"; exit 1; }
step "merge v6"; [ -f artifacts/adapter-tabula-voce-v6-merged/config.json ] || $PY scripts/voice/merge_voice.py models/hf/Qwen3.5-4B artifacts/adapter-tabula-voce-v6 artifacts/adapter-tabula-voce-v6-merged > outputs/persona/merge-v6.log 2>&1 || { echo "FAILED merge"; exit 1; }
step "bf16 gguf + MTP graft"
if [ ! -f $G/Tabula-4B-v6-bf16.gguf ]; then
  PYTHONPATH=$LC/gguf-py $PY $LC/convert_hf_to_gguf.py artifacts/adapter-tabula-voce-v6-merged --outtype bf16 --model-name "Tabula 4B" --outfile $G/Tabula-4B-v6-bf16-nomtp.gguf > outputs/persona/convert-v6.log 2>&1 || { echo "FAILED convert"; exit 1; }
  # 15/09: the merge of today keeps blk.32 (441 tensors, same as the base): no graft needed
  mv $G/Tabula-4B-v6-bf16-nomtp.gguf $G/Tabula-4B-v6-bf16.gguf
fi
step "quantize Q4_0"; [ -f $G/Tabula-4B-v6-Q4_0.gguf ] || $LC/build/bin/llama-quantize $G/Tabula-4B-v6-bf16.gguf $G/Tabula-4B-v6-Q4_0.gguf Q4_0 > outputs/persona/quant-v6.log 2>&1 || { echo "FAILED quantize"; exit 1; }
ls -la $G/Tabula-4B-v6-Q4_0.gguf $G/adapter-tabula-voce-v6.gguf
step "TABULA V6 DONE"
