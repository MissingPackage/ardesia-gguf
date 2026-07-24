# 02 — Spike-1: log d'esperimento (2026-07-24, PASS al run 13)

**Domanda:** il loop LoRA-over-GGUF (Path A, `transformers-qwen3-moe-fused`) gira sul nostro
RTX 4090 Laptop 16 GB restando sotto i 16 GiB di VRAM? **Risposta: sì** — su Qwen3-30B-A3B
UD-IQ2_M, con dequant eager-chunked. Criterio HANDOFF: batch 1 / ctx 2048 / rank 4, loop gira +
VRAM < 16 GiB.

## Ambiente
- RTX 4090 Laptop, 16376 MiB fisici (15.57 GiB usabili); desktop KDE "leggero" ≈ 0.6 GiB
  (kwin+plasmashell). Tetto effettivo per il processo ≈ **15 GiB**.
- `.venv` py3.12: torch 2.13.0+cu130, transformers 4.57.6 (**pin <5**, richiesto dal README
  moe-fused), triton 3.7.1, unsloth (installati 2026-07-24, unpinned).
- `vendor/transformers-qwen3-moe-fused` @ a087104 + nostre modifiche (sotto).
- Entry point: `scripts/run_smoke.sh` → `scripts/smoke_train_gguf.py`. VRAM: `outputs/vram-smoke.log`.

## Cronologia dei run (4 muri distinti)

| run | config | esito |
|---|---|---|
| 1 | IQ3_XXS (12.02 GiB), stock | ✗ `RuntimeError: Unsloth: Unsuccessfully patched inner_training_loop` (`unsloth/models/vision.py:1377`, via `get_peft_model`) |
| 2 | + raise neutralizzato (warning); desktop pieno ~1.9 GiB | ✗ OOM a 14.0 GiB di processo durante autotune `_grouped_gemm_forward_kernel` — falliti 64 MiB |
| 3 | desktop leggero (~1.3 GiB recuperati: killati 2 processi unsloth-studio 686 MiB + VS Code) | ✗ `FailOnRecompileLimitHit` nel dequant GGUF compilato (`fullgraph=True`), limite letto = **8** |
| 4 | cache Triton/inductor pulite + `cache_size_limit=256` post-import | ✗ identico (8) → cache esclusa come fattore |
| 5 | `recompile_limit=1024` subito prima di `trainer.train()` (print conferma 1024) | ✗ check legge ancora **8** → config globale ignorata nel contesto |
| 6 | bump 1024 come PRIMA istruzione dell'interprete | ✗ ancora 8 |
| 7 | dequant `fullgraph=False` (degrada a eager oltre-limite) | ✗ OOM: dequant eager monolitico alloca intermedi fp32 ~**768 MiB** sui tensori fused (14.68 GiB) |
| 8 | UD-IQ2_M (10.1 GiB) | ✗ `KeyError: IQ2_XS` — tipo assente dalla mappa dequant upstream; nel file: **70 tensori / 3.79 GiB** (la fetta più grossa) |
| — | **port IQ2_XS scritto** (da reference numpy gguf-py, stile IQ2_XXS) | ✓ verificato **bit-exact** CPU+GPU vs numpy (`max|diff|=0.0`) |
| 9 | IQ2_M + port, `fullgraph=False` | ✗ OOM in eager `dequantize_blocks_IQ2_S` (`db*grid_val*signs`, 768 MiB) |
| 10 | `torch_compile=False` in SFTConfig + dequant `fullgraph=True` | ✗ ancora limite 8 → l'outer compile del Trainer NON è il colpevole |
| 11 | dequant `dynamic=True` | ✗ ancora 8 → nemmeno la specializzazione per shape |
| 12 | warm-up esplicito di tutti i 9 qtype pre-train (passa, costo VRAM ~0) | ✗ dentro il trainer le guardie falliscono comunque → ricompila → 8 |
| 13 | **dequant eager CHUNKED (32768 blocchi ≈ 32 MiB fp32), dynamo rimosso dal dequant** | ✅ **PASS** |

## Risultato (run 13)
- 8/8 step; loss 0.553–0.588 (media 0.574), grad_norm ≈ 0.21 (gradienti reali attraverso la LoRA).
- **Picco VRAM: 14.16 GiB allocati / 14.31 riservati** (< 16, margine ~1.3 GiB sul tetto effettivo).
- `train_runtime` 495 s → **~62 s/step allo smoke** (incluso il compile di `compile_layers` al primo
  step; riferimento: Path B su Strix Halo = 6.5 s/it). Throughput a regime NON misurato.

## Findings
1. **Fit**: pesi GGUF residenti in VRAM = dimensione file (12.02 → 12.02; 10.10 → 10.10).
   IQ3_XXS **non ci sta** sul nostro 16 GB condiviso col desktop; IQ2_M sì. Il "16 GB" dell'autore
   presuppone GPU senza display.
2. **Il contesto trainer impone `recompile_limit=8`** e fa fallire sistematicamente le guardie
   dynamo del dequant (mentre in isolamento le stesse funzioni generalizzano su 70 shape senza
   incidenti — repro: vedi `docs/03` §issue). Meccanismo NON root-caused; indiziato il
   checkpointing/patching unsloth. 6 contromisure inefficaci (run 4–12) → dynamo rimosso dal dequant.
3. **Dequant eager monolitico = OOM strutturale** sui tensori fused MoE (intermedi fp32 da
   ~768 MiB); il chunking a 32768 blocchi lo rende innocuo (~32 MiB per chunk).
4. **IQ2_XS mancava upstream** ed è dominante nelle UD-quant 2-bit → port necessario (fatto,
   verificato, upstream-abile: `docs/03`).
5. Igiene GPU: lo studio unsloth (round-9) lasciava 2 processi con **686 MiB** — check
   `nvidia-smi` prima di ogni run.

## Modifiche locali (landmine: si perdono ricreando venv / ri-clonando vendor)
- `.venv/.../unsloth/models/vision.py:1377`: raise → warning (workaround documentato dall'autore).
- `vendor/.../quantize_gguf/dequant.py`: (a) `dequantize_blocks_IQ2_XS` + grid + registrazione;
  (b) `wrap_dequantize_function` → eager chunked, niente `torch.compile`.
- `scripts/smoke_train_gguf.py`: `torch_compile=False`, bump limiti (inefficaci ma innocui,
  lasciati per documentazione), dataset sintetico inline, `save_strategy="no"`, `max_steps=8`.
