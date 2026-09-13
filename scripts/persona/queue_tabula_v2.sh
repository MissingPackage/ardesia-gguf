#!/usr/bin/env bash
# Tabula voce v2 after the SQL GRPO chain (single GPU scheduler rule).
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
PY=.venv-q35/bin/python; LC=/home/neuromancer/Projects/llama.cpp-oracle-next
until grep -q "SQL GRPO DONE" outputs/sql/queue-grpo.log 2>/dev/null; do sleep 60; done
echo "=== tabula v2 $(date +%H:%M)"
[ -f artifacts/adapter-tabula-voce-v2/adapter_config.json ] || $PY scripts/voice/train_voice_lora.py --data data/identity/persona-tabula-v2.jsonl --out artifacts/adapter-tabula-voce-v2 --epochs 2 --lr 1.5e-4 > outputs/persona/train-tabula-voce-v2.log 2>&1
grep -E "train done|peak VRAM" outputs/persona/train-tabula-voce-v2.log
[ -f artifacts/gguf/adapter-tabula-voce-v2.gguf ] || PYTHONPATH=$LC/gguf-py $PY $LC/convert_lora_to_gguf.py artifacts/adapter-tabula-voce-v2 --base models/hf/Qwen3.5-4B --outtype f16 --outfile artifacts/gguf/adapter-tabula-voce-v2.gguf > /dev/null 2>&1 && echo "gguf ok"
$PY scripts/persona/probe_tabula.py --name tabula-voce-v2 --torch models/hf/Qwen3.5-4B --adapter artifacts/adapter-tabula-voce-v2 --out outputs/persona/probe-tabula-voce-v2.json || echo "FAILED probe v2"
source scripts/pathb-env.sh > /dev/null
$PY scripts/persona/probe_tabula.py --name webgguf-Q4_0+tabula-v2 --gguf $HOME/.cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf --lora-gguf artifacts/gguf/adapter-tabula-voce-v2.gguf --out outputs/persona/probe-webgguf-tabula-v2.json || echo "FAILED probe v2 gguf"
echo "=== TABULA V2 DONE $(date +%H:%M)"
