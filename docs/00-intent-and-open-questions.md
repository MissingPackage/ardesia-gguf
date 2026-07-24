# 00 — Intent and open questions

Anchor for the repo. Primary source: the `ardesia-unsloth` sibling project handoff (2026-07-24) + the
Reddit post "LoRA over GGUF: Train Qwen3.6-35B-A3B in 16G VRAM" (woct0rdho) + the three linked repos.

## Why we jump a tier (the evidence, not the intuition)

Round-9 (`4b-v7`) ITERATE, two-faced sensor `calibration-v1.jsonl`:
- **assert 13/21 = 61.9%** (bar ≥85%) — fabricates on facts it should know (NaOH-denatures-PCR, etc.)
- **trap 7/15 = 46.7%** (bar ≥70%) — fabricates precision where the fact is genuinely open (Hb-β
  "Val-Val-Val…", false α, etc.)
- Real progress (MBPP 0→45.8%, traps 0/4→7/15, GSM8K 80→84%, slop 0/10) **but** the central defect
  remains: **assert-fail and trap-fail are the SAME fabrication defect.** It is not cured by more data
  volume (D8 reopening-condition) nor by more 4B design → **it is a capability ceiling.**

## The 4 specs from handoff #0 — status

- **(a) Base:** `Qwen3.6-35B-A3B` (MoE, ~3B active). ⚠ 2026 model — **verify it exists as a downloadable
  GGUF + APEX quant** and from where (HF). `[VERIFY]`
- **(b) Toolchain:** `recipe (transformers-5) + torch-ggml-ops` **or** `transformers-qwen3-moe-fused
  (transformers-4, Triton)`. Decision after spike-zero.
- **(c) Fit:** on paper solved — APEX 13.3 GiB + fused kernels → 16 GiB without offload, batch 1 / ctx 2048
  / rank 4. **Measured only on Strix Halo.** To be re-measured on the 4090. `[VERIFY]`
- **(d) Serve unmerged:** confirmed, consistent with the round-6 erosion finding.

## ⚠ Spike-zero (BLOCKING, before any run): CUDA?

The path that gives the 16-GiB-no-offload (`recipe + torch-ggml-ops`) is **AMD-tested**. Backward kernels
in CK Tile (AMD); CuTe/NVIDIA "planned". Author: *"should just work on RDNA3 GPUs, and not too hard to
port to other GPUs."* Ours is **NVIDIA CUDA (4090 Laptop, 16 GB)**.

Sub-questions to close (one observation at a time):
1. Does `torch-ggml-ops` build/run on CUDA as-is? (build test in a throwaway venv)
2. If not: how large is the CK Tile → CuTe port? (out of scope for us short-term → fall back to moe-fused)
3. `moe-fused` on the 4090: its "LoRA over GGUF" path (Triton on-demand dequant) on a 35B-A3B —
   how much VRAM really? The bnb-4bit path gives **~17-18 GB of weights alone in q4** → over the 16 GB.
   The APEX sub-4-bit is what lets us fit; if moe-fused has no 1-3bit equivalent, the fit doesn't work.

**Rule:** no training until spike-zero has a measured outcome.

## Minimal plan (after spike-zero, not before)

1. Environment: dedicated `uv venv`, CUDA torch ≥2.10, chosen toolchain.
2. Download base Qwen3.6-35B-A3B (GGUF + APEX) — measure real footprint.
3. Smoke: a micro-LoRA (rank 4) on a handful of persona rows → verify the loop runs and VRAM holds.
4. Copy the reusable data from ardesia-unsloth (persona-v7 / calibration-v1 / persona.py) — **copy**.
5. Train persona → serve **unmerged** → eval on `calibration-v1` (assert ≥85% / trap ≥70%) comparable to round-9.

## Open decisions

- ✅ **RESOLVED (2026-07-24): go straight for Path B.** Rationale:
  round-10 on a ~2.7 bpw base (the only 30B quant that fits in VRAM) is not worth it — aggressive
  quantization would be a confounding variable on the fabrication experiment. Round-10 on the 30B
  dropped; the smoke PASS remains as stack validation and a reference baseline.


- **[2026-07-24] Tier-jump target, after the docs/01 erratum:** the 35B (qwen3_5_moe, GDN hybrid)
  is not loadable on Path A; it is reachable only via Path B (CUDA port of torch-ggml-ops + the author's
  transformers 5 fork + replacing the AMD-only pieces like AITER — days-to-weeks).
  Options: **(1)** test the scale hypothesis on Qwen3-30B-A3B-Instruct-2507 (Qwen3 gen, Path A
  supported today); **(2)** invest in the Path B port for the real Qwen3.6-35B; **(3)** 30B now as
  round-10 AND the port in parallel. The in-progress 30B smoke validates the stack in every branch and
  does not foreclose the choice.
  **New data (smoke run-2):** the *effective* VRAM ceiling on the 4090 Laptop with a KDE desktop is
  **~14.7 GiB** (15.57 usable − ~0.9 of compositor/apps). The IQ3_XXS 30B (12.02 GiB of weights) went
  OOM at 14.0 GiB of process on Path A. Implication for option (2): the APEX-I-Mini 35B is
  13.33 GiB of weights alone — on Path B the MMQ overhead is lower (no materialized bf16) but
  not measured on a discrete 16 GB card; mudler does not publish smaller 35B quants. The 35B fit
  remains a risk even with the port done.
  **Smoke outcome (run-13, PASS):** loop 8/8 steps on UD-IQ2_M, peak **14.16 GiB** (< 16), sane loss.
  Cost: eager-chunked dequant (dynamo unusable in the trainer context, limit-8 not root-caused)
  → **~62 s/step** at the smoke (first-step compile included). Vs 6.5 s/it for Path B on Strix Halo.
  **Quality confound for option (1):** the only 30B quant that fits is IQ2_M (~2.7 bpw); the author's
  reference (IQ3_XXS ~3.1 bpw) does NOT fit on our 16 GB with the desktop active.
  A fabrication experiment on a ~2.7 bpw base has quantization as a confounding variable.

## References
- Post: r/LocalLLaMA "LoRA over GGUF: Train Qwen3.6-35B-A3B in 16G VRAM"
- https://github.com/woct0rdho/transformers5-qwen3.5-recipe
- https://github.com/woct0rdho/torch-ggml-ops
- https://github.com/woct0rdho/transformers-qwen3-moe-fused
- Unsloth discussion #3894 "Train LoRA over GGUF"
