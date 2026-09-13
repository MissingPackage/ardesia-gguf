#!/usr/bin/env bash
# Tabula voce v3 (format layer): train -> gguf -> identity probes (bf16, Q4_0+runtime) -> merge -> EuroEval proxy.
set -uo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
PY=.venv-q35/bin/python; LC=/home/neuromancer/Projects/llama.cpp-oracle-next
step() { echo "=== $1 $(date +%H:%M)"; }
step "train v3"; [ -f artifacts/adapter-tabula-voce-v3/adapter_config.json ] || $PY scripts/voice/train_voice_lora.py --data data/identity/persona-tabula-v3.jsonl --out artifacts/adapter-tabula-voce-v3 --epochs 2 --lr 1.5e-4 > outputs/persona/train-tabula-voce-v3.log 2>&1
grep -E "train done|peak VRAM" outputs/persona/train-tabula-voce-v3.log
[ -f artifacts/gguf/adapter-tabula-voce-v3.gguf ] || PYTHONPATH=$LC/gguf-py $PY $LC/convert_lora_to_gguf.py artifacts/adapter-tabula-voce-v3 --base models/hf/Qwen3.5-4B --outtype f16 --outfile artifacts/gguf/adapter-tabula-voce-v3.gguf > /dev/null 2>&1 && echo "gguf ok"
step "probe v3"; $PY scripts/persona/probe_tabula.py --name tabula-voce-v3 --torch models/hf/Qwen3.5-4B --adapter artifacts/adapter-tabula-voce-v3 --out outputs/persona/probe-tabula-voce-v3.json || echo "FAILED probe v3"
source scripts/pathb-env.sh > /dev/null
$PY scripts/persona/probe_tabula.py --name webgguf-Q4_0+tabula-v3 --gguf $HOME/.cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf --lora-gguf artifacts/gguf/adapter-tabula-voce-v3.gguf --out outputs/persona/probe-webgguf-tabula-v3.json || echo "FAILED probe v3 gguf"
step "merge v3"; [ -f artifacts/adapter-tabula-voce-v3-merged/config.json ] || $PY scripts/voice/merge_voice.py models/hf/Qwen3.5-4B artifacts/adapter-tabula-voce-v3 artifacts/adapter-tabula-voce-v3-merged > outputs/persona/merge-v3.log 2>&1 || echo "FAILED merge"
step "euroeval proxy v3"; bash scripts/run_euroeval_it.sh /home/neuromancer/Projects/ardesia-gguf/artifacts/adapter-tabula-voce-v3-merged tabula-v3 1 > outputs/euroeval/run-tabula-v3.log 2>&1; grep -E "^(squad|multinerd|scala|sentipolc|mmlu|hellaswag|ilpost)" outputs/euroeval/run-tabula-v3.log
step "TABULA V3 DONE"
