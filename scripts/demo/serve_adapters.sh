#!/usr/bin/env bash
# Demo server: ONE Q4_0 base (the exact file webgguf ships) + every benchmark adapter loaded but NOT
# applied; each request picks its adapter(s) with the `lora` field (id + scale), or POST /lora-adapters
# switches the default set at runtime. No reload of the base, ever.
# Usage: serve_adapters.sh [PORT]     then: curl localhost:PORT/lora-adapters
set -euo pipefail
cd /home/neuromancer/Projects/ardesia-gguf
source scripts/pathb-env.sh > /dev/null   # CUDA libs for build-cuda/
PORT=${1:-8095}
BASE=${BASE:-$HOME/.cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf}
[ -f "$BASE" ] || BASE=artifacts/gguf/base-4b-Q4_0.gguf
ADAPTERS=""
for a in voce entita lettura conoscenza sensocomune matematica2 codice; do
  f=artifacts/gguf/adapter-$a.gguf; [ $a = voce ] && f=artifacts/gguf/voice-4b-lora.gguf
  [ -f $f ] && ADAPTERS="$ADAPTERS --lora $f"
done
echo "base: $BASE"; echo "adapters:$ADAPTERS"
exec /home/neuromancer/Projects/llama.cpp-oracle-next/build-cuda/bin/llama-server -m "$BASE" $ADAPTERS \
  --lora-init-without-apply -ngl 99 -c 4096 --jinja --port $PORT -np 1 --reasoning-budget 0
