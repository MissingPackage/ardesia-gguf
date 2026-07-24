# Ardesia-GGUF

Sibling di [`ardesia-unsloth`](../ardesia-unsloth). Stessa Ardesia (il piccolo modello locale
essenziale/onesto), **stack di training diverso**: LoRA su base GGUF-quantizzato per il **salto di tier
a ~35B MoE**, che il 4B non regge.

Repo separato = **dipendenze separate**. `ardesia-unsloth` è pinnato su torch 2.10+cu130 / unsloth
2026.7.3 / transformers 4.57.6. Questo stack vuole `transformers-5` (fork) + `torch-ggml-ops`, oppure
`transformers-qwen3-moe-fused` (transformers 4): conflitti → venv a sé.

## Perché esiste (intento — da `ardesia-unsloth` HANDOFF §#0, 2026-07-24)

Round-9 è chiuso **ITERATE**: `4b-v7` fabbrica confident-wrong su **entrambe** le facce del sensore
(assert 61.9% <85%, trap 46.7% <70%), e — punto chiave — **assert-fail e trap-fail sono lo STESSO
difetto**. Letto come **soffitto di capability del 4B**, non come problema di design del dato (round-8 e
round-9 avevano dati di training puliti, refuter 0-kill, e fabbricavano lo stesso su probe *non visti*).

**Decisione di Cristiano:** smettere di ridisegnare a 4B; saltare a **~35B MoE** allenato con LoRA su
base GGUF, sull'ipotesi che **la scala curi il soffitto di fabbricazione** (più conoscenza parametrica
→ molta meno confabulazione).

## La scommessa (onesta)

L'ipotesi è ben fondata ma **è una scommessa, non una certezza**: i modelli grandi allucinano ancora su
specifiche quantitative long-tail. Attenzione al MoE: **35B-A3B = 35B di conoscenza ma ~3B di compute
attivo** → grande vittoria sull'asse *fabbricazione/conoscenza*, guadagno modesto sull'asse
*reasoning-depth*. Si **misura sullo STESSO sensore a due facce** (`calibration-v1.jsonl`: assert ≥85% /
trap ≥70%) così che sia comparabile a round-9.

## Toolchain (woct0rdho)

- **[`transformers5-qwen3.5-recipe`](https://github.com/woct0rdho/transformers5-qwen3.5-recipe)** — la
  ricetta. Transformers 5. APEX quant → Qwen3.6-35B-A3B a **13.3 GiB**; train batch 1 / ctx 2048 / LoRA
  rank 4 in **16 GiB senza offload**; 6.5 s/it. **Tarata e testata su AMD Strix Halo (gfx1151).**
- **[`torch-ggml-ops`](https://github.com/woct0rdho/torch-ggml-ops)** — i binding PyTorch dei kernel
  fused dequant-matmul/MoE. Quant IQ2_S…Q6_K, **gradiente sugli input** (è ciò che abilita il training).
  Backward in **CK Tile → solo AMD**; CuTe/NVIDIA "planned". PyTorch ≥ 2.10.
- **[`transformers-qwen3-moe-fused`](https://github.com/woct0rdho/transformers-qwen3-moe-fused)** — il
  cugino NVIDIA-friendly. Transformers 4, kernel **Triton, testato su 3090/4090**. Ha già la feature
  "train LoRA over GGUF" (dequant on-demand) + compat bnb-4bit / PEFT / Unsloth. Meno spinto sul
  sub-4-bit dell'APEX, ma **gira su CUDA oggi**.

## ⚠ Rischio #1 — spike-zero: gira su CUDA?

Il nostro è un **RTX 4090 Laptop (NVIDIA/CUDA)**. Il path `recipe + torch-ggml-ops` che dà il 16-GiB-no-
offload è **AMD-testato**; il port NVIDIA non è confermato. **Prima di impegnare un run** va deciso:

1. `torch-ggml-ops` builda/gira su CUDA (o va portato CK Tile → CuTe)? →
2. se no, passiamo da `moe-fused` (Triton, NVIDIA), accettando che la via bnb-4bit costi più VRAM del
   sub-4-bit APEX → **35B-A3B in q4 ≈ 17-18 GB di soli pesi**, che sfora i 16 GB → offload / quant più
   spinto / o si resta a un MoE più piccolo.

Vedi `docs/00-intent-and-open-questions.md`.

## Lineage / riuso da `ardesia-unsloth` (copiare, MAI symlink)

La pipeline dati persona/calibrazione è **stabilizzata e riusabile**: `src/ardesia/persona.py`,
`scripts/build_persona_v7.py`, il **sensore a due facce** `benchmarks/personal/calibration-v1.jsonl`, la
catena non-erodente, il deploy **unmerged** (LoRA bf16 su base quantizzato = via requant-safe di round-6).
Si copia il necessario qui quando si arriva al training.

## Fallback

Se il path 35B si blocca su CUDA/toolchain: la round-10 "widen-the-hedge" a 4B (in `ardesia-unsloth`)
resta il piano di riserva.
