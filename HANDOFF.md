# HANDOFF — Ardesia-GGUF (2026-07-24)

Entry point per una sessione fresca **lanciata in questa directory**. Leggi, in ordine: questo file →
`README.md` → `docs/00-intent-and-open-questions.md` → `docs/01-cuda-porting-assessment.md` → `CLAUDE.md`.
Questo repo è **auto-contenuto**: non serve leggere `../ardesia-unsloth` per ripartire (ci si riferisce solo
per il *lineage* dei dati). La memoria di Claude di `ardesia-unsloth` **non** si carica qui — di proposito.

## Dove siamo
- **Scaffold + assessment fatti. Niente costruito o allenato.** Nessun venv, nessun modello scaricato.
- **La domanda "LoRA-over-GGUF si porta sul nostro 4090 CUDA?" è RISPOSTA: sì** (assessment su sorgenti in
  `docs/01`). Prossimo atto = **spike-1** (sotto).

## La risposta in breve (da `docs/01`, evidence-based)
Due vie indipendenti su NVIDIA:
- **Path A — `transformers-qwen3-moe-fused`: gira su CUDA OGGI, zero kernel.** Triton+Unsloth, `device="cuda"`,
  tarato 3090/4090. Il suo dequant GGUF copre tutti i quant-type dell'APEX-I-Mini → la GGUF
  `mudler/Qwen3.6-35B-A3B-APEX-GGUF` è caricabile. Ci dà la Ardesia-35B su NVIDIA senza scrivere kernel.
- **Path B — `torch-ggml-ops` → CUDA: il port condivisibile, delimitato.** Forward già CUDA (llama.cpp +
  `CUDAExtension`/nvcc); l'unico pezzo AMD sono **3 kernel di backward** (intrinsic RDNA3
  `__builtin_amdgcn_wmma_...bf16` + layout fragment per-lane). Port = swap a tensor-core NVIDIA (CuTe/`mma.sync`)
  + riscrittura indicizzazione + `__hip_bfloat16`→`__nv_bfloat16`; wave32↔warp32 puliti. Giorni-settimane.
  **È il "buon risultato da condividere"** — nessuno l'ha ancora fatto, Apache-2.0.
- **Raccomandazione: Path A prima** (valida l'ipotesi a rischio-kernel zero), **Path B dopo** come contributo.

## Perché esiste (intento — da `../ardesia-unsloth` HANDOFF §#0, 2026-07-24)
Round-9 (`4b-v7`) chiuso **ITERATE**: il 4B fabbrica confident-wrong su entrambe le facce del sensore
(assert 61.9% <85%, trap 46.7% <70%) e **assert-fail ≡ trap-fail ≡ stesso difetto** → letto come **soffitto di
capability del 4B**, non problema di dato. Decisione di Cristiano: **saltare a Qwen3.6-35B-A3B MoE** via
LoRA-over-GGUF, sull'ipotesi che la scala curi il soffitto di fabbricazione. **È una scommessa** (i grandi
allucinano ancora sul long-tail; e MoE = 35B di conoscenza ma ~3B di compute attivo) → **si misura sullo
STESSO sensore a due facce** per essere comparabile a round-9.

## ⚠ NEXT — spike-1 (Path A). NIENTE training vero finché lo smoke non passa.
1. `uv venv` dedicato (deps **separate** dallo studio unsloth); torch CUDA ≥2.10, poi `pip install -e` di
   `transformers-qwen3-moe-fused` (transformers 4 + triton + unsloth) — clone in `/tmp/ggml-port` o ri-clonare.
2. `hf download mudler/Qwen3.6-35B-A3B-APEX-GGUF Qwen3.6-35B-A3B-APEX-I-Mini.gguf` → misurare footprint reale.
3. Micro-smoke: ri-puntare `example_train_30b_a3b_gguf.py` alla GGUF 35B + un pugno di righe (batch 1 / ctx 2048
   / rank 4) → **osservare: loop gira + VRAM < 16 GB**. Poi (solo se passa) training vero.
4. Se il 35B non ci sta a 16 GB: quant più spinto (IQ2), o restare 30B-A3B, o passare a Path B.

## Non-negoziabili (ereditati, NON ri-derivare)
- **Serve UNMERGED** (LoRA bf16 su base quantizzato); **mai requantizzare il delta** (erosione round-6).
- **Deps separate**: mai installare/importare dal venv unsloth.
- **Dati copiati, mai symlink.** Riusabili da `../ardesia-unsloth` (copiare a training-time):
  `benchmarks/personal/calibration-v1.jsonl` (sensore a 2 facce, **assert ≥85% / trap ≥70%**),
  `data/identity/persona-v7.jsonl`, `src/ardesia/persona.py` (libreria corpus).
- **Safety = intento non topic; register = senior/post-doc, non ELI5.** System prompt identità italiano verbatim.
- QA/decisioni PI-gated: subagent `persona-cristiano` (definito in `../ardesia-unsloth`).
- **Nessuna AI attribution** in commit/PR.

## Aperte / [VERIFY]
- Fit del **35B-A3B sul 4090 non misurato** (il "16 GB" dell'autore è su un 30B).
- `moe-fused` fast-forward MoE su GGUF è `TODO` nel codice → throughput ignoto; dequant **per-tensore** → VRAM
  transitoria da misurare.
- Path B non ancora provato a compilare su CUDA.
- Fallback se il path 35B si blocca: round-10 "widen-the-hedge" a 4B (in `../ardesia-unsloth`).

## Sorgenti lette (clone effimero /tmp/ggml-port, 2026-07-24 — ri-clonare se serve)
`woct0rdho/transformers-qwen3-moe-fused` · `woct0rdho/torch-ggml-ops` · `woct0rdho/transformers5-qwen3.5-recipe`
· post r/LocalLLaMA "LoRA over GGUF: Train Qwen3.6-35B-A3B in 16G VRAM" · Unsloth discussion #3894.
