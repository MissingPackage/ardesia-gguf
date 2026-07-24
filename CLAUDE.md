# CLAUDE.md — Ardesia-GGUF

Cristiano's global rules apply (act-then-report, debugging/completion contracts, no AI attribution).
This repo is the **~35B MoE tier-jump** sibling of `../ardesia-unsloth`. Fresh session started in THIS
directory: read `HANDOFF.md` FIRST (the self-contained entry point), then `README.md`,
`docs/00-intent-and-open-questions.md`, `docs/01-cuda-porting-assessment.md`. Cross-read
`../ardesia-unsloth` only for data lineage — this repo is meant to resume without it.

## What this repo is
The **LoRA-over-GGUF** training stack (woct0rdho: `transformers5-qwen3.5-recipe` + `torch-ggml-ops`, or
the NVIDIA `transformers-qwen3-moe-fused`). Goal: train a **Qwen3.6-35B-A3B** Ardesia in 16 GB, on the
hypothesis that scale cures the 4B fabrication ceiling measured in round-9.

## Non-negotiables carried over from ardesia-unsloth (do NOT re-derive)
- **Deps separate.** Never install into / import from the unsloth studio venv. This repo has its own uv venv.
- **Never requantize the trained delta.** Serve **unmerged** (bf16 LoRA on the quantized base) — the
  round-6 requant-safe path. `ardesia-unsloth` memory `ardesia-requant-erosion`.
- **Measure on the SAME sensor.** `calibration-v1.jsonl` (two-faced: assert ≥85% / trap ≥70%) so results
  are comparable to round-9. Register slop 0/10, identity 0/6, GSM8K, MBPP as in ardesia-unsloth.
- **Safety = intent not topic; register = senior/post-doc depth not ELI5.** Identity conditioned on the
  verbatim Italian system prompt. See ardesia-unsloth CLAUDE.md §Behavior directives + memories
  `ardesia-safety-posture`, `ardesia-register-depth`.
- **Data is copied, never symlinked.** Persona/calibration data + `persona.py` live in ardesia-unsloth.
- **QA/PI-gated decisions** go through the `persona-cristiano` subagent (defined in ardesia-unsloth).
- **No AI attribution** in commits/PRs.

## Status
Scaffold only. **Spike-zero not yet run:** does `torch-ggml-ops` build/run on our CUDA 4090, or do we go
via the moe-fused/Triton path? Nothing trained here yet.
