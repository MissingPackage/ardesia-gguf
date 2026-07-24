# 02 — Spike-1: experiment log (2026-07-24, PASS at run 13)

**Question:** does the LoRA-over-GGUF loop (Path A, `transformers-qwen3-moe-fused`) run on our
RTX 4090 Laptop 16 GB while staying under 16 GiB of VRAM? **Answer: yes** — on Qwen3-30B-A3B
UD-IQ2_M, with eager-chunked dequant. Criterion: batch 1 / ctx 2048 / rank 4, loop runs +
VRAM < 16 GiB.

## Environment
- RTX 4090 Laptop, 16376 MiB physical (15.57 GiB usable); a "light" KDE desktop ≈ 0.6 GiB
  (kwin+plasmashell). Effective ceiling for the process ≈ **15 GiB**.
- `.venv` py3.12: torch 2.13.0+cu130, transformers 4.57.6 (**pin <5**, required by the moe-fused
  README), triton 3.7.1, unsloth (installed 2026-07-24, unpinned).
- `vendor/transformers-qwen3-moe-fused` @ a087104 + our modifications (below).
- Entry point: `scripts/run_smoke.sh` → `scripts/smoke_train_gguf.py`. VRAM: `outputs/vram-smoke.log`.

## Run history (4 distinct walls)

| run | config | outcome |
|---|---|---|
| 1 | IQ3_XXS (12.02 GiB), stock | ✗ `RuntimeError: Unsloth: Unsuccessfully patched inner_training_loop` (`unsloth/models/vision.py:1377`, via `get_peft_model`) |
| 2 | + raise neutralized (warning); full desktop ~1.9 GiB | ✗ OOM at 14.0 GiB of process during autotune `_grouped_gemm_forward_kernel` — 64 MiB failed |
| 3 | light desktop (~1.3 GiB reclaimed: killed 2 unsloth-studio processes 686 MiB + VS Code) | ✗ `FailOnRecompileLimitHit` in the compiled GGUF dequant (`fullgraph=True`), limit read = **8** |
| 4 | Triton/inductor caches cleared + `cache_size_limit=256` post-import | ✗ identical (8) → cache ruled out as a factor |
| 5 | `recompile_limit=1024` immediately before `trainer.train()` (print confirms 1024) | ✗ check still reads **8** → global config ignored in the context |
| 6 | bump 1024 as the FIRST interpreter statement | ✗ still 8 |
| 7 | dequant `fullgraph=False` (degrades to eager past the limit) | ✗ OOM: monolithic eager dequant allocates fp32 intermediates ~**768 MiB** on the fused tensors (14.68 GiB) |
| 8 | UD-IQ2_M (10.1 GiB) | ✗ `KeyError: IQ2_XS` — type absent from the upstream dequant map; in the file: **70 tensors / 3.79 GiB** (the largest slice) |
| — | **IQ2_XS port written** (from the gguf-py numpy reference, IQ2_XXS style) | ✓ verified **bit-exact** CPU+GPU vs numpy (`max|diff|=0.0`) |
| 9 | IQ2_M + port, `fullgraph=False` | ✗ OOM in eager `dequantize_blocks_IQ2_S` (`db*grid_val*signs`, 768 MiB) |
| 10 | `torch_compile=False` in SFTConfig + dequant `fullgraph=True` | ✗ still limit 8 → the Trainer's outer compile is NOT the culprit |
| 11 | dequant `dynamic=True` | ✗ still 8 → not even shape specialization |
| 12 | explicit warm-up of all 9 qtypes pre-train (passes, VRAM cost ~0) | ✗ inside the trainer the guards fail anyway → recompiles → 8 |
| 13 | **eager CHUNKED dequant (32768 blocks ≈ 32 MiB fp32), dynamo removed from the dequant** | ✅ **PASS** |

## Result (run 13)
- 8/8 steps; loss 0.553–0.588 (mean 0.574), grad_norm ≈ 0.21 (real gradients through the LoRA).
- **Peak VRAM: 14.16 GiB allocated / 14.31 reserved** (< 16, ~1.3 GiB margin on the effective ceiling).
- `train_runtime` 495 s → **~62 s/step at the smoke** (including the `compile_layers` compile at the first
  step; reference: Path B on Strix Halo = 6.5 s/it). Steady-state throughput NOT measured.

## Findings
1. **Fit**: GGUF weights resident in VRAM = file size (12.02 → 12.02; 10.10 → 10.10).
   IQ3_XXS **does not fit** on our 16 GB shared with the desktop; IQ2_M does. The author's "16 GB"
   assumes a GPU with no display.
2. **The trainer context imposes `recompile_limit=8`** and systematically fails the dequant's dynamo
   guards (whereas in isolation the same functions generalize over 70 shapes without
   incident — repro: see `docs/03` §issue). Mechanism NOT root-caused; the unsloth
   checkpointing/patching is the suspect. 6 ineffective countermeasures (runs 4–12) → dynamo removed from the dequant.
3. **Monolithic eager dequant = structural OOM** on the fused MoE tensors (fp32 intermediates of
   ~768 MiB); chunking at 32768 blocks makes it harmless (~32 MiB per chunk).
4. **IQ2_XS was missing upstream** and is dominant in the 2-bit UD-quants → port necessary (done,
   verified, upstream-able: `docs/03`).
5. GPU hygiene: the unsloth studio (round-9) left 2 processes holding **686 MiB** — check
   `nvidia-smi` before every run.

## Local modifications (landmine: lost by recreating the venv / re-cloning vendor)
- `.venv/.../unsloth/models/vision.py:1377`: raise → warning (workaround documented by the author).
- `vendor/.../quantize_gguf/dequant.py`: (a) `dequantize_blocks_IQ2_XS` + grid + registration;
  (b) `wrap_dequantize_function` → eager chunked, no `torch.compile`.
- `scripts/smoke_train_gguf.py`: `torch_compile=False`, limit bumps (ineffective but harmless,
  left for documentation), inline synthetic dataset, `save_strategy="no"`, `max_steps=8`.
