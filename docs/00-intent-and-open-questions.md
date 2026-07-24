# 00 — Intento e domande aperte

Ancora del repo. Fonte primaria: `../ardesia-unsloth/HANDOFF.md` §#0 (2026-07-24) + il post Reddit
"LoRA over GGUF: Train Qwen3.6-35B-A3B in 16G VRAM" (woct0rdho) + i tre repo linkati.

## Perché saltiamo tier (l'evidenza, non l'intuizione)

Round-9 (`4b-v7`) ITERATE, sensore a due facce `calibration-v1.jsonl`:
- **assert 13/21 = 61.9%** (bar ≥85%) — fabbrica su fatti che dovrebbe sapere (NaOH-denatures-PCR, ecc.)
- **trap 7/15 = 46.7%** (bar ≥70%) — fabbrica precisione dove il fatto è genuinamente aperto (Hb-β
  "Val-Val-Val…", α falso, ecc.)
- Progressi reali (MBPP 0→45.8%, traps 0/4→7/15, GSM8K 80→84%, slop 0/10) **ma** il difetto centrale
  resta: **assert-fail e trap-fail sono lo STESSO difetto di fabbricazione.** Non si cura con più volume
  di dato (D8 reopening-condition) né con più design a 4B → **è un soffitto di capability.**

## Le 4 specifiche dell'HANDOFF #0 — stato

- **(a) Base:** `Qwen3.6-35B-A3B` (MoE, ~3B attivi). ⚠ modello 2026 — **verificare che esista come GGUF +
  APEX quant scaricabile** e da dove (HF). `[VERIFY]`
- **(b) Toolchain:** `recipe (transformers-5) + torch-ggml-ops` **oppure** `transformers-qwen3-moe-fused
  (transformers-4, Triton)`. Decisione post spike-zero.
- **(c) Fit:** on paper risolto — APEX 13.3 GiB + kernel fused → 16 GiB senza offload, batch 1 / ctx 2048
  / rank 4. **Misurato solo su Strix Halo.** Da ri-misurare su 4090. `[VERIFY]`
- **(d) Serve unmerged:** confermato, coerente con l'erosione round-6.

## ⚠ Spike-zero (BLOCCANTE, prima di qualunque run): CUDA?

Il path che dà il 16-GiB-no-offload (`recipe + torch-ggml-ops`) è **AMD-testato**. Backward kernels in
CK Tile (AMD); CuTe/NVIDIA "planned". Autore: *"should just work on RDNA3 GPUs, and not too hard to port
to other GPUs."* Il nostro è **NVIDIA CUDA (4090 Laptop, 16 GB)**.

Sotto-domande da chiudere (una osservazione alla volta):
1. `torch-ggml-ops` compila/gira su CUDA così com'è? (build test in un venv usa-e-getta)
2. Se no: quanto è il port CK Tile → CuTe? (fuori scope per noi a breve → si ripiega su moe-fused)
3. `moe-fused` su 4090: la sua via "LoRA over GGUF" (dequant on-demand Triton) su un 35B-A3B —
   quanta VRAM davvero? La via bnb-4bit dà **~17-18 GB di soli pesi in q4** → sfora i 16 GB. L'APEX
   sub-4-bit è ciò che ci fa stare; se moe-fused non ha l'equivalente 1-3bit, il fit non torna.

**Regola:** nessun training finché lo spike-zero non ha un esito misurato.

## Piano minimo (dopo lo spike-zero, non prima)

1. Ambiente: `uv venv` dedicato, torch CUDA ≥2.10, toolchain scelto.
2. Scaricare base Qwen3.6-35B-A3B (GGUF + APEX) — misurare footprint reale.
3. Smoke: un micro-LoRA (rank 4) su un pugno di righe persona → verificare che il loop giri e la VRAM stia.
4. Copiare da ardesia-unsloth i dati riusabili (persona-v7 / calibration-v1 / persona.py) — **copia**.
5. Train persona → serve **unmerged** → eval su `calibration-v1` (assert ≥85% / trap ≥70%) comparabile a round-9.

## Riferimenti
- Post: r/LocalLLaMA "LoRA over GGUF: Train Qwen3.6-35B-A3B in 16G VRAM"
- https://github.com/woct0rdho/transformers5-qwen3.5-recipe
- https://github.com/woct0rdho/torch-ggml-ops
- https://github.com/woct0rdho/transformers-qwen3-moe-fused
- Unsloth discussion #3894 "Train LoRA over GGUF"
