# 03 — Upstream woct0rdho: status (updated 2026-07-24 evening)

- ✅ **§1 IQ2_XS: DONE AND MERGED** — PR #23 on `transformers-qwen3-moe-fused`, merged
  by the author within minutes (commit a4f3c52). Their reply: *"You can add everything in
  ggml-org/llama.cpp#25681 if you want."*
- **#25681** = their open PR on llama.cpp: `gguf.torch_quants` (authoritative torch dequant API
  in gguf-py, **IQ2_XS already included**, parametric tests over all types even under compile).
  → No code of ours to port there; the useful contribution = a downstream validation comment
  (bit-exact CUDA + real training on an IQ2_XS-dominant GGUF in 16 GB) + the memory datapoint
  (768 MiB monolithic eager vs ~32 MiB chunked) in support of their "wrap in torch.compile".
  ⚠ **llama.cpp requires AI-usage disclosure and restricts AI-generated content**
  (AGENTS.md/CONTRIBUTING.md) — any submission there must be declared honestly.
- **§2 (chunked)**: no longer a PR to moe-fused — at most a comment on #25681 (above).
- **§3 (recompile-limit issue)**: still valid on moe-fused, unchanged below.

---

## 1. PR: Add IQ2_XS support to the GGUF dequant map

**Title:** `Add IQ2_XS dequantization (used heavily by Unsloth UD 2-bit quants)`

> `quantize_gguf/dequant.py` covers IQ2_S and IQ2_XXS but not IQ2_XS, so loading e.g.
> `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-IQ2_M` fails with
> `KeyError: <GGMLQuantizationType.IQ2_XS: 17>`. In that file IQ2_XS is actually the dominant
> type: 70 tensors / 3.79 GiB.
>
> This PR ports the gguf-py numpy reference (`gguf.quants.IQ2_XS.dequantize_blocks`) to torch,
> following the style of the existing IQ2_XXS implementation (shared ksigns table, low-9-bits
> grid index / high-7-bits sign index, nibble scales). Verified bit-exact against the numpy
> reference on CPU and CUDA (`max |diff| = 0.0`, random blocks incl. >32768-block tensors).

Content: import `IQ2_XS`, `GRID_IQ2_XS` init, `dequantize_blocks_IQ2_XS`, entry in the map.
(In our vendored copy this is all already in `quantize_gguf/dequant.py` — extract the clean diff
without the chunked part if separate PRs are wanted.)

---

## 2. Proposal: chunked fallback in `wrap_dequantize_function`

**Title:** `Chunk the eager dequant path to cap fp32 intermediates (OOM on 16 GB otherwise)`

> When the compiled dequant path is unavailable (see issue below) the monolithic eager dequant
> materializes fp32 intermediates of ~768 MiB per fused expert tensor
> (`db * grid_val * signs` over 128-expert fused projections), which OOMs a 16 GB card that
> already holds the quantized weights. Chunking the block dimension (we used 32768 blocks
> ≈ 32 MiB fp32 per intermediate) makes the eager path viable at negligible cost, and is
> bit-identical (verified vs the numpy reference across the chunk boundary).
>
> Suggested as a fallback (or default) for the non-compiled path.

---

## 3. Issue: compiled dequant hits `FailOnRecompileLimitHit` with limit stuck at 8 under Unsloth SFTTrainer

**Title:** `GGUF LoRA training: dequant recompile_limit reads 8 inside the Unsloth trainer context regardless of torch._dynamo.config settings`

> **Setup:** torch 2.13.0+cu130, transformers 4.57.6, triton 3.7.1, current Unsloth
> (2026-07-24), RTX 4090 Laptop 16 GB, `example_train_30b_a3b_gguf.py` adapted (batch 1,
> ctx 2048, rank 4, `use_gradient_checkpointing="unsloth"`). Note: on this stack
> `FastModel.get_peft_model` raises "Unsuccessfully patched inner_training_loop" (your
> documented workaround applied), so Unsloth runs its fallback path.
>
> **Symptom:** on the first training step every call into the compiled `_func` of
> `wrap_dequantize_function` recompiles (guards never hit), and after 8 recompiles dynamo hard
> fails (`fullgraph=True`) with *"exceeding the recompile_limit cache size limit (currently set
> to 8)"* — even though `torch._dynamo.config.recompile_limit` is 1024 in the main process,
> verified by printing it immediately before `trainer.train()`.
>
> **Tried, all ineffective:** raising `recompile_limit`/`accumulated_recompile_limit` (at
> interpreter start / after imports / right before train), `dynamic=True` on the dequant
> compile, pre-warming every (qtype × 2 sizes) variant before entering the trainer (warms fine,
> in-trainer calls still miss), `torch_compile=False` in `SFTConfig`, clearing Triton/inductor
> caches. In isolation (same venv, no trainer) the compiled dequant handles 70 distinct shapes
> with zero issues.
>
> **Reading:** something in the training context (Unsloth gradient checkpointing / a
> `config.patch` around the step?) both restores the default recompile limit and changes the
> tracing context so cached entries never match. Not root-caused. Happy to run diagnostics
> (`TORCH_LOGS=recompiles` traces) if useful.
>
> **Workaround we shipped:** dropped `torch.compile` from the dequant wrapper and chunked the
> eager path (see proposal above). Costs throughput (~62 s/step on our smoke vs your 6.5 s/it
> on Strix Halo with the fused path) but trains within 16 GB.

---

## Our context (not for upstream)
- Motivation: the spike-1 smoke (see `docs/02`). The three pieces also serve as a calling card
  before the CUDA port of `torch-ggml-ops` (Path B), where the relationship with the author matters.
- Before opening PRs/issues: rebase on upstream HEAD and re-test (the clone is @ a087104 as of today).
