#!/usr/bin/env bash
# Erosion experiment, end to end (after train_voice_lora.py has produced artifacts/voice-4b-lora):
#   merge bf16 -> GGUF bf16 -> Q8_0 / Q4_K_S / Q4_0 ; LoRA -> GGUF ; base -> Q4_0
#   then probe every arm. Idempotent: skips artifacts that already exist.
set -euo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
PY=.venv-q35/bin/python
# CUDA runtime libs for build-cuda/ come from the pip wheels (LD_LIBRARY_PATH); see memory pathb-build-toolchain
source scripts/pathb-env.sh >/dev/null
LC=/home/neuromancer/Projects/llama.cpp-oracle-next
export PYTHONPATH=$LC/gguf-py
BASE=models/hf/Qwen3.5-4B
LORA=artifacts/voice-4b-lora
MERGED=artifacts/voice-4b-merged
G=artifacts/gguf; mkdir -p $G outputs/voice

[ -f $MERGED/config.json ] || $PY scripts/voice/merge_voice.py $BASE $LORA $MERGED
[ -f $G/base-4b-bf16.gguf ] || $PY $LC/convert_hf_to_gguf.py $BASE --outtype bf16 --outfile $G/base-4b-bf16.gguf
# HF round trip drops the MTP block (blk.32.*) but keeps block_count=33 -> graft it back from the base
if [ ! -f $G/voice-4b-bf16.gguf ]; then
  $PY $LC/convert_hf_to_gguf.py $MERGED --outtype bf16 --outfile $G/voice-4b-bf16-nomtp.gguf
  $PY scripts/voice/graft_mtp.py $G/voice-4b-bf16-nomtp.gguf $G/base-4b-bf16.gguf $G/voice-4b-bf16.gguf && rm $G/voice-4b-bf16-nomtp.gguf
fi
for q in Q8_0 Q4_K_S Q4_0; do
  [ -f $G/voice-4b-$q.gguf ] || $LC/build/bin/llama-quantize $G/voice-4b-bf16.gguf $G/voice-4b-$q.gguf $q
done
[ -f $G/base-4b-Q4_0.gguf ] || $LC/build/bin/llama-quantize $G/base-4b-bf16.gguf $G/base-4b-Q4_0.gguf Q4_0
[ -f $G/voice-4b-lora.gguf ] || $PY $LC/convert_lora_to_gguf.py $LORA --base $BASE --outtype f16 --outfile $G/voice-4b-lora.gguf

export LLAMA_SERVER=$LC/build-cuda/bin/llama-server   # build/ is the CPU-only oracle build
probe() { local name=$1; shift; if [ ! -f outputs/voice/$name.json ]; then $PY scripts/voice/probe_voice.py --name $name "$@" --out outputs/voice/$name.json || { echo "PROBE FAILED: $name"; exit 1; }; fi; }
probe base-bf16      --torch $BASE
probe lora-bf16      --torch $BASE --adapter $LORA
probe merged-bf16    --torch $MERGED
probe gguf-bf16      --gguf $G/voice-4b-bf16.gguf
probe gguf-Q8_0      --gguf $G/voice-4b-Q8_0.gguf
probe gguf-Q4_K_S    --gguf $G/voice-4b-Q4_K_S.gguf
probe gguf-Q4_0      --gguf $G/voice-4b-Q4_0.gguf
probe base-Q4_0+lora --gguf $G/base-4b-Q4_0.gguf --lora-gguf $G/voice-4b-lora.gguf
# the exact file webgguf ships for the 4B card (unsloth Q4_0), adapter applied at runtime
WEBQ4=$HOME/.cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf
[ -f $WEBQ4 ] && probe webgguf-Q4_0+lora --gguf $WEBQ4 --lora-gguf $G/voice-4b-lora.gguf
$PY scripts/voice/compare_voice.py lora-bf16
