#!/usr/bin/env bash
# Tabula voce v4 (format layer): train -> gguf -> identity probes (bf16, Q4_0+runtime) -> merge -> EuroEval proxy.
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
PY=.venv-q35/bin/python; LC=/home/neuromancer/Projects/llama.cpp-oracle-next
step() { echo "=== $1 $(date +%H:%M)"; }
step "train v4"; [ -f artifacts/adapter-tabula-voce-v4/adapter_config.json ] || $PY scripts/voice/train_voice_lora.py --data data/identity/persona-tabula-v4.jsonl --out artifacts/adapter-tabula-voce-v4 --epochs 2 --lr 1.5e-4 > outputs/persona/train-tabula-voce-v4.log 2>&1
grep -E "train done|peak VRAM" outputs/persona/train-tabula-voce-v4.log
[ -f artifacts/gguf/adapter-tabula-voce-v4.gguf ] || PYTHONPATH=$LC/gguf-py $PY $LC/convert_lora_to_gguf.py artifacts/adapter-tabula-voce-v4 --base models/hf/Qwen3.5-4B --outtype f16 --outfile artifacts/gguf/adapter-tabula-voce-v4.gguf > /dev/null 2>&1 && echo "gguf ok"
step "probe v4"; $PY scripts/persona/probe_tabula.py --name tabula-voce-v4 --torch models/hf/Qwen3.5-4B --adapter artifacts/adapter-tabula-voce-v4 --out outputs/persona/probe-tabula-voce-v4.json || echo "FAILED probe v4"
source scripts/pathb-env.sh > /dev/null
$PY scripts/persona/probe_tabula.py --name webgguf-Q4_0+tabula-v4 --gguf $HOME/.cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf --lora-gguf artifacts/gguf/adapter-tabula-voce-v4.gguf --out outputs/persona/probe-webgguf-tabula-v4.json || echo "FAILED probe v4 gguf"
step "merge v4"; [ -f artifacts/adapter-tabula-voce-v4-merged/config.json ] || $PY scripts/voice/merge_voice.py models/hf/Qwen3.5-4B artifacts/adapter-tabula-voce-v4 artifacts/adapter-tabula-voce-v4-merged > outputs/persona/merge-v4.log 2>&1 || echo "FAILED merge"
step "euroeval proxy v4"; bash scripts/run_euroeval_it.sh /home/neuromancer/Projects/ardesia-gguf/artifacts/adapter-tabula-voce-v4-merged tabula-v4 1 > outputs/euroeval/run-tabula-v4.log 2>&1; grep -E "^(squad|multinerd|scala|sentipolc|mmlu|hellaswag|ilpost)" outputs/euroeval/run-tabula-v4.log
step "TABULA V3 DONE"
