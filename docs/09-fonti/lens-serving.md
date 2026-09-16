# Lens: serving and composing multi-LoRA (agent report, 2026-09-16)

I read the llama.cpp source directly and ran three parallel literature sweeps. Report below.

---

# Serving many composable LoRA adapters: state of the art, and where llama.cpp sits

Source tree read: `ggml-org/llama.cpp` @ `fb27a52` (2026‑09‑16), shallow clone in `/tmp/llamacpp-read`. Everything in §1 is **VERIFIED from source**.

## 1. llama.cpp: exactly how N adapters are applied

**Unmerged, always.** The single point of application is `llm_graph_context::build_lora_mm` in `src/llama-graph.cpp:1514`:

```cpp
ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
for (const auto & lora : *loras) {
    llama_adapter_lora_weight * lw = lora.first->get_weight(w);
    if (lw == nullptr) continue;
    const float scale = lw->get_scale(lora.first->alpha, lora.second);
    ggml_tensor * ab_cur = ggml_mul_mat(ctx0, lw->b, ggml_mul_mat(ctx0, lw->a, cur));
    res = ggml_add(ctx0, res, ggml_scale(ctx0, ab_cur, scale));
}
```

That **is** `y = Wx + Σᵢ sᵢ·Bᵢ(Aᵢx)`, computed literally: a plain loop over `llama_adapter_loras` (`unordered_map<llama_adapter_lora*, float>`, `src/llama-adapter.h:91`). There is **no sharing of A, no concatenation of the Aᵢ into one `[N·r, d]` matmul, no fused kernel**. Cost is strictly linear in N: 6 graph nodes per adapted tensor per adapter (`llama_adapter_lora::get_n_nodes()`, `src/llama-adapter.h:85`: `ab_map.size() * 6u` — "a, b, scale, add, 2 × mul_mat"), which `llama_context::graph_max_nodes` (`src/llama-context.cpp:2344`) adds to the graph budget.

**Scaling semantics** (`llama_adapter_lora_weight::get_scale`, `src/llama-adapter.h:52`): `scale = request_scale * alpha / rank`, with `rank = b->ne[0]`; if `alpha == 0` in the GGUF, `scale = request_scale` raw. Note this is *classic* LoRA scaling — an adapter trained with **rsLoRA** (`alpha/√r`) will be silently mis-scaled.

**MoE expert tensors: yes, supported.** `build_lora_mm_id` (`src/llama-graph.cpp:1545`) is the `ggml_mul_mat_id` twin, used for `ffn_{gate,up,down}_exps` and `ffn_gate_up_exps` (lines 2171–2304); it routes both A and B through the same `ids`. Conversion side: `convert_lora_to_gguf.py:234` has a `torch.stack` shim on `LoraTorchTensor` that stacks per-expert `lora_A`/`lora_B` into 3‑D tensors. Caveat found in source: LoRA on **NVFP4** FFN tensors is asserted out (`src/llama-graph.cpp:1782‑1784`).

**The per-request `lora` field does not re-merge — but it is not free either.** `server_task::params.lora` is `map<int,float>` (`tools/server/server-task.h:70`); `server-context.cpp:1709` rebuilds the slot's adapter list, and `common_set_adapter_lora` is called **once per batch** (`server-context.cpp:2855`). `llama_context::set_adapters_lora` (`src/llama-context.cpp:1284`) early-returns if the set is identical, drops zero-scale adapters, and otherwise sets `sched_need_reserve = true` → a **full graph re-reserve** on the next decode (`llama_context::sched_reserve`, `:582`).

Two costs that matter more than the matmuls:

- **`lora_should_clear_cache` (`tools/server/server-common.cpp`) clears the prompt/KV cache on any adapter-set change** that isn't alora→alora. Switching adapters mid-stream = reprocess the prompt.
- **`server_slot::can_batch_with` (`tools/server/server-context.cpp:449`) requires `are_lora_equal(lora, other_slot.lora)`.** Concurrent requests with *different* adapter sets are **never batched together**. The README says it plainly: "requests with different LoRA configurations will not be batched together, which may result in performance degradation" (`tools/server/README.md:603`).

That last line is the whole story: llama.cpp has **no heterogeneous-batch LoRA kernel**. It has composition (free, and better than anyone else's), and no multi-tenancy.

**Measured on your own artifacts** (`artifacts/gguf/adapter-lettura.gguf`, read with `gguf.GGUFReader`): 128 adapted tensors (a,b pairs), rank 16, alpha 16, f16, 21.2 M params = 42.5 MB. So per active adapter: **+768 graph nodes**, and **+42.5 MB of weight reads per decode step** against a 2.6 GB `base-4b-Q4_0.gguf` → **+1.6 % memory traffic per adapter** (INFERRED arithmetic). At decode, llama.cpp's CUDA-graph path (`ggml/src/ggml-cuda/ggml-cuda.cu:4426‑4474`) amortises the launch overhead of those 768 nodes **as long as the adapter set is stable** — which it is, per-batch.

## 2. Server stacks

| Stack | Resident adapters | Applied | **Multiple adapters per request?** | Published overhead |
|---|---|---|---|---|
| **vLLM** | `--max-loras` (per-batch), `--max-cpu-loras`, `--max-lora-rank` (default 16) | unmerged; Punica SGMV/BGMV retired in favour of V1 `lora_shrink`/`lora_expand` torch ops ([PR #14685](https://github.com/vllm-project/vllm/pull/14685), 2025) | **No.** "currently, we only allow one LoRA per prompt" — VERIFIED [docs.vllm.ai/en/stable/features/lora](https://docs.vllm.ai/en/stable/features/lora/), fetched 2026‑09‑16 | — |
| **SGLang** | `--max-loras-per-batch` (default 8), `--max-loaded-loras` | unmerged; Triton or chunked‑SGMV backend | **No** (one adapter per sequence); an open issue asks for it ([sgl-project/sglang#16384](https://github.com/sgl-project/sglang/issues/16384)) | `enable_lora_overlap_loading`: ~35 % median TTFT reduction ([docs.sglang.io/advanced_features/lora](https://docs.sglang.io/advanced_features/lora.html)) |
| **TGI** | `LORA_ADAPTERS` env, all preloaded | unmerged, credits punica/lorax kernels | No (per-request `adapter_id`, single) | [HF docs](https://huggingface.co/docs/text-generation-inference/en/conceptual/lora) |
| **LoRAX** | "1000s", GPU↔CPU Adapter Exchange Scheduling | unmerged, Punica SGMV | **YES — the only production exception.** `merge_strategy` ∈ {`linear`, `ties`, `dare_linear`, `dare_ties`} per request (VERIFIED, `docs/guides/merging_adapters.md` in [predibase/lorax](https://github.com/predibase/lorax), read 2026‑09‑16) | "+20 % latency serving 128 different fine-tuned models" ([Predibase blog](https://predibase.com/blog/lorax-the-open-source-framework-for-serving-100s-of-fine-tuned-llms-in)) |
| **NVIDIA NIM / TRT‑LLM** | `NIM_PEFT_SOURCE`, GPU/host two-tier cache | unmerged, batched heterogeneous kernels | Not documented → assume no (INFERRED) | PEFT cache scales as `max_batch_size × max_lora_rank` ([NIM PEFT docs](https://docs.nvidia.com/nim/large-language-models/latest/peft.html)) |
| **Fireworks** | 100 adapters per deployment, billed at base-model token price ([fireworks.ai/blog/multi-lora](https://fireworks.ai/blog/multi-lora)) | — | No | — |
| **Together** | multiple adapters per dedicated endpoint, beta ([docs.together.ai](https://docs.together.ai/docs/dedicated-endpoints/lora-adapter)) | — | No | — |
| **Punica** ([arXiv 2310.18547](https://arxiv.org/abs/2310.18547), 2023‑10‑28) | — | SGMV groups a heterogeneous batch into one kernel | across requests only | 12× throughput, **+2 ms/token** |
| **S‑LoRA** ([arXiv 2311.03285](https://arxiv.org/abs/2311.03285), 2023‑11‑06) | "thousands" on one GPU, Unified Paging | — | across requests only | up to 4× vs vLLM-of-the-day |

**The structural insight.** Every SGMV/BGMV-class kernel batches *different single adapters across different requests*. Nobody batches *several adapters within one request* — because that problem is trivial (just loop) and rare. llama.cpp is the *only* stack in the table that gives you true per-request stacking out of the box, and LoRAX is the only other one, via merge strategies. **Your stacking capability is not behind the state of the art; your multi-tenancy is.** Note also that vLLM has since grown MoE-expert LoRA with a `--enable-mixed-moe-lora-format` 2D/3D distinction (VERIFIED, same vLLM doc) — relevant when you move to 35B‑A3B.

## 3. Sharing A across adapters

- **HydraLoRA** ([arXiv 2404.19245](https://arxiv.org/abs/2404.19245), NeurIPS 2024 Oral): one shared **A**, N heads **Bᵢ**, trainable router: `ΔW·x = Σᵢ gᵢ(x)·Bᵢ(A·x)`. This is *exactly* the caller's intuition, and it works — but A, the Bᵢ and the router are trained **jointly**; there is no published "bolt a new head onto an existing A later" mode. Its win is specialisation quality, not parameter count (it used *more* trainable params than LoRA r=32).
- **VeRA** ([arXiv 2310.11454](https://arxiv.org/abs/2310.11454), ICLR 2024): A and B **frozen random and shared across all layers**, only diagonal scaling vectors trained. 10× fewer params on GLUE; LLaMA2‑13B MT‑Bench 5.93 vs LoRA 5.77 at **2.4 M vs 250 M** params. For your serving problem this is the extreme: one shared random pair in VRAM, kilobyte adapters.
- **Tied‑LoRA** ([arXiv 2311.09578](https://arxiv.org/abs/2311.09578)), **VB‑LoRA** ([arXiv 2405.15179](https://arxiv.org/abs/2405.15179), 0.4 % of LoRA's storage on Llama2‑13B), **LoRA‑XS** ([arXiv 2405.17604](https://arxiv.org/abs/2405.17604)), **PRoLoRA** ([arXiv 2402.16902](https://arxiv.org/abs/2402.16902)): variants of the same trick.
- **Composition without retraining**: **Arrow** routing ([arXiv 2405.11157](https://arxiv.org/abs/2405.11157)) routes per token/layer by the top right singular vector of each BᵢAᵢ — zero-shot, no gate training. **PHATGOOSE** ([arXiv 2402.05859](https://arxiv.org/abs/2402.05859)) trains post-hoc gates. **LoraHub** ([arXiv 2307.13269](https://arxiv.org/abs/2307.13269)) fits mixing scalars by CMA‑ES on a handful of examples — *directly applicable to picking your stacking weights sᵢ*. **LoRA Soups** ([arXiv 2410.13025](https://arxiv.org/abs/2410.13025)) shows learned **concatenation** (widening the rank) beats averaging by 43 %/12 % on skill composition.
- **Interference**: TIES ([arXiv 2306.01708](https://arxiv.org/abs/2306.01708)), DARE ([arXiv 2311.03099](https://arxiv.org/abs/2311.03099)), O‑LoRA ([arXiv 2310.14152](https://arxiv.org/abs/2310.14152)). Most relevant: **KnOTS** ([arXiv 2410.19735](https://arxiv.org/abs/2410.19735), 2024‑10‑25) — joint SVD of concatenated LoRA deltas into a shared basis, motivated explicitly by the finding that *LoRA*-tuned weights are **less aligned** than full fine-tunes and therefore merge *worse*. +4.3 %.

**Answers to the three synthesis questions.** (i) Yes — one shared A means compute `A·x` once, N up-projections after: cost goes from 2N matmuls to 1+N. (ii) Quality cost is ~zero when the shared part is *frozen/random* (VeRA, VB‑LoRA), and non-trivial when A is trained and must carry all tasks (HydraLoRA needs a router to compensate). (iii) Forcing **independently trained** adapters onto a shared A post-hoc has only one credible published route — KnOTS-style joint SVD — and it costs accuracy. **Do not plan on retrofitting a shared A onto your existing eight adapters.**

## 4. On-device precedents

- **Apple**: first-party adapters are **rank 16, 16-bit, "10s of megabytes", "dynamically loaded, temporarily cached in memory, and swapped"** ([machinelearning.apple.com](https://machinelearning.apple.com/research/introducing-apple-foundation-models), June 2024; repeated in the [2025 tech report](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025), arXiv:2507.13575). Third-party adapters via the Foundation Models Adapter Toolkit are **rank 32, ~160 MB each**, delivered by Background Assets, entitlement-gated ([developer.apple.com](https://developer.apple.com/apple-intelligence/foundation-models-adapter), fetched 2026‑09‑16). One active adapter per session; no stacking documented.
- **Google AICore / Gemini Nano**: per-feature LoRA downloaded on top of a resident base, MediaTek/Snapdragon/Tensor ([Android Developers Blog](https://android-developers.googleblog.com/2025/05/on-device-gen-ai-apis-ml-kit-gemini-nano.html), May 2025). Size figures circulating ("a few MB") are **INFERRED**, not from a Google page.
- **Microsoft Phi Silica**: `LanguageModelLowRankAdapter.CreateFromPath` ([learn.microsoft.com](https://learn.microsoft.com/en-us/windows/ai/apis/phi-silica-lora)); reference recipe is **rank 32**, dropout 0.3, LR 5e‑5 ([Windows Dev Blog](https://blogs.windows.com/windowsdeveloper/2025/07/31/phi-silica-task-specialization-using-lora-in-microsoft-learning-zone-a-technical-deep-dive/), 2025‑07‑31). One adapter per model instance.
- **Qualcomm**: "LoRA + Graph Switch" exists in AI Engine Direct/QAIRT ([docs.qualcomm.com](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/tutorial_lora_v2_31_graph_switching.html)), but **no primary rank/size/latency numbers retrievable**. The "<100 ms swap" figure floating around is from a third-party blog — do not cite it.

Your rank‑16 / 42 MB adapter sits **exactly on Apple's first-party operating point**. That is a good slide.

## 5. Browser

- **wllama** ([ngxson/wllama](https://github.com/ngxson/wllama), v3.7.0, source-read 2026‑09‑16): `CreateContextOptions.lora_adapters?: {path, scale?}[]` and `lora_init_without_apply` (`src/types/types.ts:62`), forwarded as `lora_paths`/`lora_scales` (`src/wllama.ts:567`). **Runtime multi-adapter LoRA works in the browser today, with the same stacking semantics as native llama.cpp** — it is the same code. The README's "TODO: add LoRA support" is stale. *This is the finding of §5.*
- **WebLLM / MLC**: [mlc-ai/mlc-llm#2625](https://github.com/mlc-ai/mlc-llm/issues/2625) ("Multiple lora support", opened 2024‑07‑04) is still unanswered. Pre-merge required (INFERRED from absence).
- **ONNX Runtime Web / transformers.js**: ORT's MultiLoRA (`.onnx_adapter`, `SetActiveAdapters`) is real ([onnxruntime.ai/blogs/multilora](https://onnxruntime.ai/blogs/multilora)) but bindings are Python/C/C++/C#/Java — **no JS/Web binding**. Pre-merge required.
- **"webgguf"**: no project by that name exists (searched 2026‑09‑16).

## 6. Verdict for your scenario

**(a) Resident adapters** — llama.cpp loads each adapter into its own backend buffer at 42 MB; VRAM, not code, is the limit. On a 4090 with a 2.6 GB Q4_0 4B you can trivially hold **50–100 resident**. The real ceiling is the **graph re-reserve on every set change** and the **KV-cache clear**: a few hundred ms each. (INFERRED from source, not measured.)

**(b) Stacked per request** — mechanically unbounded (the loop doesn't care). Practically: **2–3**. Each adapter costs +768 nodes and +1.6 % bandwidth, *and* the literature is unanimous that naive summation degrades monotonically from N=1 (TIES/DARE/KnOTS all exist to patch this, and KnOTS specifically finds LoRA deltas align *worse* than full fine-tunes). Your "lettura + voce" pairing is the safe case; four skills at once is not.

**(c) Per-token overhead** — predicted **+1.6 % memory traffic per adapter at decode**, with launch overhead amortised by CUDA graphs. Measure it (below). This is far cheaper than the 35 %-class numbers quoted for unoptimised server LoRA, because you are batch‑1.

**(d) Stack choice.** For **on-prem enterprise with stacking as a product feature: stay on llama.cpp.** It is the only stack that composes adapters per request natively (LoRAX being the only alternative, and LoRAX's merge strategies are weight-space merges, not the runtime sum you already have). vLLM and SGLang would *lose* you the feature — both are explicitly one-adapter-per-prompt. The price you pay is `can_batch_with`: **plan for adapter-affinity routing** — route requests with identical adapter sets to the same llama-server instance, or accept that a mixed queue serialises. That is a load-balancer problem, not a model problem, and it is the single most actionable thing in this report. The browser path survives unchanged via wllama.

**The cheapest local experiment (~15 min, no training).** `--lora` is inherited by `llama-batched-bench` (`common/arg.cpp:2951`, `LLAMA_EXAMPLE_COMMON`; `tools/batched-bench/batched-bench.cpp:28`), so:

1. **Cost curve.** `llama-batched-bench -m artifacts/gguf/base-4b-Q4_0.gguf -npp 512 -ntg 128 -npl 1,4` run four times with 0, 1, 2, 3 `--lora` flags drawn from your existing `adapter-{lettura,tabula-voce-v3,entita}.gguf`. Plot tg t/s vs N. Predicted slope ≈ −1.6 %/adapter at npl=1; if it is much worse, the culprit is node count, not bandwidth, and you will see it in the pp number too.
2. **Batching penalty.** Start one `llama-server --lora a.gguf --lora b.gguf --lora-init-without-apply -np 4`, fire 4 concurrent requests all with `"lora":[{"id":0,"scale":1}]`, then 4 with alternating ids 0/1. The throughput ratio between the two runs is the exact price of `are_lora_equal` — the number that decides whether you need adapter-affinity routing.
3. **Interference.** Run `calibration-v1.jsonl` under: lettura alone; voce‑v3 alone; both at s=1.0; both at s=0.7. You already have the sensor and the register/identity/slop registers. That gives you the N=2 degradation number on *your* two-faced metric, and the answer to "is s=1.0 the right stacking scale" — which, per LoraHub, is a 2‑parameter search you can do with CMA‑ES on 20 examples if the flat 0.7 isn't good enough.

**Not verified:** none of the per-token numbers above are measured — they are arithmetic from the GGUF headers and the graph code. Experiment 1 kills or confirms them in five minutes.