# Lens: training side of LoRA-over-GGUF (agent report, 2026-09-16)

Both agents returned. Report below.

---

# LoRA over GGUF: training axis — what it buys, and what it doesn't

Legend: **[P]** = read from a primary source (paper PDF, repo README, source file) during this session. **[I]** = inference of mine. All dates are the source's own.

## 1. What woct0rdho actually claims

I could not reach Reddit (both `reddit.com` and `old.reddit.com` are blocked from this environment, and the post is not indexed by the search backend). The same author's primary statements on GitHub are available and unambiguous.

**[P]** `transformers5-qwen3.5-recipe` README, repo created 2026-07-21, last push 2026-09-13 — opening line: *"GGUF is going to replace bitsandbytes as the base model format for low-VRAM LoRA training."* Claims: Qwen3.6-35B-A3B in 16 GiB, DeepSeek-V4-Flash 284B-A13B in 90 GiB, no CPU offload. *"Currently all kernels and parameters in this repo are tuned for Strix Halo."* No benchmark, no quality number. https://github.com/woct0rdho/transformers5-qwen3.5-recipe

**[P]** `transformers-qwen3-moe-fused` README §"LoRA over GGUF" (push 2026-07-24) — the motivation, verbatim: *"Training LoRA over quantized base model (also known as QLoRA) is a common practice. Previously we usually do so with bnb 4-bit quant, but it should be possible to do so with GGUF and save more VRAM with < 4-bit quant. Although it's believed that the accuracy in inference drops significantly with < 4-bit quant, it still makes sense to train LoRA on it."* https://github.com/woct0rdho/transformers-qwen3-moe-fused

**[P]** unslothai/unsloth Discussion #3894, opened by woct0rdho **2026-01-15**: *"When using 3-bit rather than 4-bit base model, we can train Qwen-30B-A3B with 16 rather than 24 GB VRAM… For now it surely has a lot of rough edges, and we need more experiments to check the quality of such LoRA."* Two community comments, **zero maintainer replies**, still open. https://github.com/unslothai/unsloth/discussions/3894

**Conclusion: the author claims memory, explicitly not quality, and explicitly not quant-matching.** The stated benefit is that GGUF gives you sub-4-bit base weights that bitsandbytes does not, and that pre-quantized GGUFs ship earlier than bnb-4bit uploads.

**[P]** `torch-ggml-ops` README: dense MMQ + grouped MMQ + grouped-MMQ-pair; *"All ops support gradients on inputs"* — gradients w.r.t. **activations**, i.e. exactly the LoRA case, never w.r.t. the frozen quantized weight. Backward kernels are CK Tile (AMD RDNA3, gfx1151); CuTe port *"should be straightforward"* but is not done. Quant types: IQ2_XXS, IQ2_S, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0. https://github.com/woct0rdho/torch-ggml-ops

## 2. The family that trains against a quantized base

| Method | Where the low-rank term lives at serve time | Headline number |
|---|---|---|
| **QLoRA** (arXiv 2305.14314) | bf16 adapter, base NF4; compute dequantizes to bf16 | MMLU 5-shot mean: bf16 53.0 vs NF4+DQ **53.1** (Table 4) **[P]** |
| **QA-LoRA** (2309.14717, v2 2023-10-09) | merges **into** the INT4 grid via group-wise zero-point absorption | see §3 **[P]** |
| **LoftQ** (2310.08659) | init only; adapter stays fp16 on Q. Never discusses merging **[P]** | LLaMA-2-13B 2.5-bit: QLoRA ppl 19.39 → LoftQ **5.22** |
| **LQ-LoRA** (2311.12023) | `W ≈ Q + L1L2`, ILP-chosen mixed precision; counts an 8-bit adapter in "effective bits" | 70B @2.85 bits: Open-LLM 65.3 vs 67.7 uncompressed |
| **ApiQ** (2402.05147) | init minimizing **activation** error | Llama-2-13B 2-bit GSM8K: LoftQ 25.4 → **41.2** |
| **PEQA** (2305.14152) / **EfficientQAT** (2407.11062) / **LR-QAT** (2406.06385) | no adapter at serve time at all — train scales/steps, artifact is natively quantized | EfficientQAT 4-bit Llama-1-7B MMLU **41.2** vs unmerged QLoRA 38.4 |

**[P]** LR-QAT (Qualcomm) states the problem most precisely: *"Simply quantizing adapters after training will lead to a different quantization grid compared to W₀, and quantizing them using the same quantization grid as W₀ will lead to a high error."*

**[P] Critical gap: none of this literature targets GGUF.** The whole lineage is NF4/GPTQ/AWQ. No peer-reviewed paper studies K-quants, IQ-quants or imatrix for finetuning.

**Tooling, 2026 (all [P], checked today against master):**
- **llama.cpp can train, but not what you need.** `llama-finetune` was removed (PR #8669, 2024-07-25) and reinstated (PR #10544, merged 2025-05-12); last fix PR #27199, 2026-09-02. `examples/training/README.md`: *"technically functional (for FP32 models and limited hardware setups)… very much WIP."* It is **full finetune, no LoRA**; `llama_set_param()` returns early for `type != GGML_TYPE_F32`, so quantized tensors are silently never trainable; `ggml_compute_backward` has **no `MUL_MAT_ID` case**, so routed experts have no backward at all. Three community QLoRA PRs (#20453, #22705, #26794) were closed or stalled.
- **ik_llama.cpp**: no `ggml-opt.cpp`, no training. LoRA *inference* exists but ikawrakow (issue #1986, 2026-06-17): *"I have absolutely no idea if LoRA adapters still work"* — `--lora` fails **silently** unless `--no-fused-up-gate`.
- **Unsloth**: bnb-NF4 only; GGUF is a run/export target. No maintainer commitment to GGUF training.
- **transformers**: `gguf.torch_quants` is **PR #25681** (woct0rdho, opened 2026-07-14, **still open**). Transformers docs v5.17: weights stay packed *only* on Metal + Qwen3.5; **on CUDA the GGUF is dequantized at load**. Issue #40070 is receptive (SunMarc, 2026-01-15) but nothing merged; woct0rdho maintains an out-of-tree `gguf` branch.

## 3. Requant erosion — what is actually measured

Nobody has published the experiment you'd want (same adapter, trained on X, served **unmerged** on Y). Three adjacent measurements of merge-then-requantize:

**[P] QA-LoRA Table 1** (MMLU 5-shot), QLoRA-unmerged → QLoRA+GPTQ-4bit:
- LLaMA-7B Alpaca 38.4 → 36.0 (**−2.4**); FLANv2 44.3 → 41.4 (**−2.9**)
- **13B −0.4/−0.2 · 33B −0.6/−0.2 · 65B −0.1/−0.4**
- At 2 bits it collapses: 7B Alpaca → 25.8 (chance).

**The requant tax is a small-model phenomenon at 4 bits and a catastrophe at 2 bits, at every size.** This is directly relevant: your 35B target's only in-VRAM quants are IQ2/APEX-class.

**[P] unsloth-zoo PR #970** (opened 2026-07-31, open): Qwen2.5-0.5B, MLX 4-bit, 300 steps. Plain LoRA gains 0.306 nats of loss but retains only **0.111 after merged-4bit save — 64% of the learning destroyed**; with QAT, ~100% retained.

**[P] PEFT issue #2321** (2025-01-11): `merge_and_unload` on a 4-bit base warns of rounding errors; maintainer calls it *"unavoidable"*. Issue #2105 (2024-09-26, closed unresolved): maintainers say it is **not rigorously measured**.

**Your docs/06 (2026-09-12) is therefore a real contribution**, not a redundant check: on a 4B, a bf16-trained r16 LoRA survives merge→Q4_0 with identity 0/6 and register intact, and the runtime-LoRA arm is indistinguishable. It sits exactly where QA-LoRA's 7B row predicts trouble — and finds none for a *voice* target. **[I]** The likely reconciliation: MMLU-style knowledge is quantization-fragile; a style/identity delta is a large, low-frequency direction that survives rounding.

## 4. MoE: where LoRA can go, and the hard hot-swap constraint

**[P] The per-expert `nn.Linear` layout is gone.** Current `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`: `Qwen3_5MoeExperts` holds raw 3-D `nn.Parameter` — `gate_up_proj [E, 2·ff, hidden]`, `down_proj [E, hidden, ff]`. `target_modules=["gate_proj","up_proj","down_proj"]` matches **nothing** inside experts (landed in PR #40132, merged 2025-10-02). Only attention, GDN projections, the shared expert MLP and `shared_expert_gate` are nn.Linear.

**[P] PEFT's answer is `target_parameters`** (PR #2638, merged 2025-07-15): `target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"]`, with `rank_pattern` ≈ `r // num_experts`. PEFT's own doc warns it *"always materializes the LoRA contribution for each expert"* → big inference slowdown, *"the recommendation is thus to merge the weights"* — which collides with your serve-unmerged rule **in PyTorch only**.

**[P] llama.cpp serves unmerged LoRA on routed experts natively — verified in source.** `src/llama-graph.cpp:1545` `build_lora_mm_id()` does `ggml_mul_mat_id(lw->b, ggml_mul_mat_id(lw->a, cur, ids), ids)` and adds it to the quantized `mul_mat_id` result; `build_moe_ffn` routes `gate_up_exps/up_exps/gate_exps/down_exps` through it. CUDA `supports_op` accepts F32/F16 for `MUL_MAT_ID` src0, so the 3-D adapter runs on GPU. `llama-server` exposes per-request adapter selection (`"lora":[{"id":…,"scale":…}]`, `GET/POST /lora-adapters`) — your hot-swap architecture is supported upstream.

**[P] Three landmines I verified in source:**
1. `convert_lora_to_gguf.py` will **reject a PEFT `target_parameters` MoE adapter as-is**. PEFT stores `lora_A` as `nn.Linear(in, r·E)` — a flat 2-D `(r·E, in)` (peft `lora/layer.py:2455`). `LoraTorchTensor.shape` returns the reconstructed delta shape → 2-D → the Qwen converter's `if data_torch.ndim < 3: raise ValueError("Unexpected gate_up_proj shape")` fires. **Fix: reshape to `(E, r, in)` / `(E, out, r)` before conversion** — i.e. woct0rdho's fused layout, which the converter's `torch.stack`/`split` support does handle.
2. `llama-export-lora` **dequantizes the base to F32 and writes F16** (`export-lora.cpp:180-183, 315-322`). Merging into a GGUF *is* a requantization round-trip; there is no in-grid merge.
3. **[P] webgguf cannot load expert LoRA today.** `src/engine/core/io/loraadapter.ts:68` — `if (t.dims.length !== 2) throw`, and F16 only. Browser adapters must be attention-only. (Also: the HANDOFF's "webgguf has no LoRA runtime" is stale — the loader exists.)

**[P]** On expert-subset choice: MoE-Sieve (arXiv 2603.24044, 2026-03-25) finds LoRA on the top-25% most-routed experts is within ±1pp of all-experts at −70% params. **No paper ablates experts vs attention-only.** Framework defaults now include experts (Unsloth `faster-moe`; Axolotl `lora_target_parameters`).

## 5. Verdict

**LoRA-over-GGUF is a memory technique, not a quality technique — for you it is a nice-to-have that is currently irrelevant, and on CUDA it is a porting project.**

1. **It does not buy quant-matching.** The bf16 adapter never touches the base's grid during training either way; over-GGUF just changes *which* dequantized weights the gradient sees. The methods that genuinely make training quant-aware (QA-LoRA, LoftQ, ApiQ, LR-QAT, EfficientQAT) are a **different axis**, none of them GGUF-native, and none of them implemented in your stack.
2. **Your measured evidence already says you don't need it at 4B.** docs/06: identity 0/6 and register intact across all nine arms including webgguf-Q4_0 + runtime LoRA. docs/08: seven bf16-trained adapters, +21.3 NER / +24.1 SQuAD-it F1 / +9.7 Spider, hot-swapped on the Q4_0 base.
3. **Where it *would* matter is the 2-bit regime** — QA-LoRA's 2-bit rows and the unsloth-zoo 64%-loss number are the warning. If the 35B has to ship at IQ2/APEX, quant-aware training becomes a live question. **[I]** But note the cheaper move: your own cloud-budget rule already points at a bf16 LoRA on the 35B rented for hours, served unmerged — which sidesteps the whole axis.
4. **The gap in docs/08 worth closing:** the benchmark deltas were measured on **merged bf16 dirs** (`eval_merged.sh`), not on the served Q4_0 base. Only the voice probe was measured on Q4_0.

### Cheapest experiment that settles it locally (4B, ~2 GPU-hours)

One flag, three arms, existing harness. In `scripts/voice/train_voice_lora.py:32` you already have `load_in_4bit=False, load_in_16bit=True`.

- **A (have it):** train bf16 → serve unmerged on webgguf's Q4_0.
- **B (one flag):** `load_in_4bit=True` → same data, same seed, same r16 → same serving arm. This isolates *"does the training-time base quantization matter at all"* — NF4 vs bf16 training, identical serving. ~4 min of training.
- **C (the real question, and it's free):** run the **docs/08 benchmark suite** — multinerd-it, squad-it, m_mmlu_it — on the **Q4_0 + runtime LoRA** arm instead of the merged bf16 dir. If the +21.3 / +24.1 deltas survive Q4_0 serving, LoRA-over-GGUF is settled as irrelevant for the 4B product and the burden shifts entirely to the 2-bit 35B case.

C is the one that actually changes a decision; it needs no new training at all, only a second eval pass through `llama-server` with the adapters already converted. If A≈B on the two-faced calibration sensor, the training-quantization axis is dead for your regime and you can close it with a number.