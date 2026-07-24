#!/usr/bin/env bash
# Entry point spike-1: micro-smoke LoRA-over-GGUF (see scripts/smoke_train_gguf.py and docs/02).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="vendor/transformers-qwen3-moe-fused${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python scripts/smoke_train_gguf.py
