# 01 — CUDA portability: assessment (evidence-based)

> **⚠ ERRATUM (2026-07-24, spike-1):** the conclusion "the GGUF `mudler/Qwen3.6-35B-A3B-APEX` is
> loadable on Path A" is **wrong**. It only assessed *quant-type* coverage, not the
> **architecture**: Qwen3.6-35B-A3B is `qwen3_5_moe` — a **GatedDeltaNet** hybrid (3 linear-attention
> layers per full-attention, `full_attention_interval: 4`) + VLM wrapper + shared expert
> + MTP (config.json of `Qwen/Qwen3.6-35B-A3B`). `Qwen3MoeFusedForCausalLM` models only the
> pure `qwen3_moe` full-attention (grep: zero qwen3_5/GDN support in the repo; transformers 4.57
> does not register `qwen3_5_moe`). The "35B in 16GB" post was about the *transformers5 recipe*
> stack (= Path B), not moe-fused. **Consequence: the 35B is not reachable on Path A; the 35B
> requires Path B. Path A remains valid for `qwen3_moe` models (e.g. Qwen3-30B-A3B).**

> **⚠ ERRATUM 2 (2026-07-24, spike-2, from the source mapping):** characterizing the Path B port
> as "**CK Tile → CuTe**" + "replace **AITER / rocBLAS GEMM**" is **imprecise**. The `ck` namespace
> in `torch-ggml-ops` is **local** (`torch_ggml_ops::ck` = *custom kernels*), **not** AMD
> Composable Kernel; and in the tree there is **no dependency** on AITER, rocBLAS/hipBLAS or a CK
> library (grep: zero occurrences). The forward `mma.cuh`/`mmq-*-targets.cuh` is **dual-path** llama.cpp
> (the NVIDIA `TURING_MMA_AVAILABLE` branch is already present, merely dormant). The **only** genuine
> kernel port is `ck/bf16_wmma.cuh` (42 lines): the gfx11 intrinsic `__builtin_amdgcn_wmma_f32_16x16x16_bf16`
> + the per-lane C-fragment layout. Every backward kernel touches the MMA **only** through that
> header. The rest is mechanical de-HIP (type/handle shims).

Done by reading the sources of the three woct0rdho repos (clone in `/tmp/ggml-port`, 2026-07-24). Question:
*"Can LoRA over GGUF be ported/used on our 4090 CUDA?"* — **Yes**, and there are **two independent paths**,
one of which **does not require writing kernels**.

## Path A — `transformers-qwen3-moe-fused`: runs on CUDA TODAY (no kernels to write)

**Evidence:**
- `example_train_30b_a3b_gguf.py`: `device = "cuda"`, stack **Triton + Unsloth + torch_compile
  max-autotune**, header *"Runs with 16 GB VRAM using UD-IQ3_XXS"* (3-bit quant). Serves LoRA over a GGUF
  base with on-demand dequant (a `Bnb4BitHfQuantizer`-style quantizer).
- `requirements.txt` = a pure PyTorch/Triton stack (`torch, triton, transformers, peft, trl, unsloth,
  bitsandbytes, gguf`). **No ROCm, no CK.** Installs on the 4090 as-is.
- README: *"mainly optimized for RTX 3090 and RTX 4090"*. The default grouped-GEMM is **Triton**
  (alternative CUTLASS/CK/Helion backends in `grouped_gemm/`). Triton = CUDA-native.
- `quantize_gguf/dequant.py` dequantizes **every** APEX-I-Mini quant-type: the `dequantize` map
  includes BF16, Q8_0, Q6_K, Q5_K, Q4_K, **Q3_K**, Q2_K, IQ4_XS, IQ3_S, IQ3_XXS, **IQ2_S**, IQ2_XXS,
  IQ1_M/S. APEX-I-Mini uses {IQ2_S, Q3_K, Q4_K, Q5_K, Q6_K} → **all present** → the GGUF
  `mudler/Qwen3.6-35B-A3B-APEX-GGUF` is loadable on this path.

**Consequence:** an Ardesia **35B-A3B** trained via LoRA-over-GGUF **on our 4090** is reachable
**without writing kernels** — just wiring: download the GGUF, re-point the example, measure VRAM/throughput.

**Caveats to verify ourselves (not assumed):**
1. The example is **30B-A3B**; the 35B-A3B must be re-measured for VRAM/speed on the 4090.
2. The MoE GGUF fast-forward is a **TODO** in the code (`patch_Qwen3MoeFusedSparseMoeBlock_forward` commented out):
   it works but may be slower than the fully-fused AMD path.
3. Dequant is **per-tensor on-demand** (dequantizes the active expert's weight to bf16 at forward) → it uses
   transient VRAM; with a 3B-active MoE it's manageable but must be measured.
4. Transformers **4** + Unsloth (close to our `ardesia-unsloth`, but **separate deps** regardless).

## Path B — `torch-ggml-ops` → CUDA: the shareable port (kernel work, bounded)

This is the more efficient path (quantized MMQ kernels that **never dequantize the whole tile** → less
VRAM/faster than Path A's per-tensor approach; it is what gives the 13.3 GiB APEX + 6.5 s/it on Strix Halo)
and it powers the `transformers5` recipe.

**Port surface (from the sources):**
- **Forward = already CUDA.** `csrc/vendor/llama_cpp/*` are llama.cpp's MMQ kernels (primary target CUDA/nvcc).
  `setup.py` uses `CUDAExtension` (nvcc); hipify only fires on ROCm. `mmq_hip.cu` only `#undef`s HIP
  half macros. On a CUDA box, nvcc compiles the forward natively.
- **Backward = the only AMD-specific piece.** 3 kernels (`csrc/ck/{mmq_backward, grouped_mmq_backward,
  grouped_mmq_backward_tiled}.cuh`) that use:
  - the RDNA3 intrinsic **`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`** (`bf16_wmma.cuh`) — the gfx11 MMA;
  - types `__hip_bfloat16`, headers `hip/hip_bf16.h` / `hip/hip_runtime.h`;
  - an RDNA3-specific per-lane C-fragment layout (`c_row = lane&15`, `c_column = 2*el + lane>>4`);
  - wave size 32 (`GROUPED_BACKWARD_WAVE_SIZE`).
- **The CuTe port** = for each of the 3 kernels: replace the amdgcn MMA with the **NVIDIA tensor core**
  (CuTe `mma_atom`, or PTX `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32`), **rewrite the fragment
  indexing** to the NVIDIA output layout, and swap `__hip_bfloat16`→`__nv_bfloat16` / `cuda_bf16.h`. The
  math (dequant tile→bf16→MMA→accumulate) is identical. **AMD wave32 ↔ NVIDIA warp32 map cleanly**
  (no wave64). It needs familiarity with tensor-core fragment layouts, but it is **days-to-weeks, not research**.
- The author themselves: *"it should be straightforward to port them to CuTe on Nvidia GPUs"* (README §backward).

**Payoff:** brings APEX/sub-4-bit + the `transformers5` recipe onto NVIDIA — **an upstream contribution
nobody has made yet** (the "good result worth sharing"). Apache-2.0 license, PR-able upstream.

## Recommendation

**Path A first, Path B as a contribution.**
1. **Path A** validates the true goal (Ardesia-35B-A3B on CUDA, testing the "scale cures fabrication"
   hypothesis) at **zero kernel risk**, and teaches us the stack. If fit/throughput on the 4090 holds,
   we have the large Ardesia **without waiting for the port**.
2. **Path B** is decided *afterwards*: if we need APEX's VRAM/speed advantage, or want to give the
   CUDA port back to the community, we do the CuTe backward work (bounded). It is the shareable result.

**Rule unchanged:** serve **unmerged** (bf16 LoRA on a quantized base), measure on `calibration-v1`
(assert ≥85% / trap ≥70%) comparable to round-9.

## Concrete next step (Path A, spike-1)
1. Dedicated `uv venv` + `pip install -e .` of `moe-fused` (transformers 4 + triton + unsloth) on CUDA torch ≥2.10.
2. `hf download mudler/Qwen3.6-35B-A3B-APEX-GGUF …-I-Mini.gguf` — measure the real disk/VRAM footprint.
3. Micro-smoke: re-point `example_train_30b_a3b_gguf.py` at the 35B GGUF + a handful of rows → **observe
   the loop runs and VRAM stays under 16 GB** (batch 1, ctx 2048, rank 4). NO real training until the smoke passes.
4. If the fit doesn't work at 35B: harder quant (IQ2), or stay at 30B-A3B, or move to Path B.

## Source references (clone /tmp/ggml-port, 2026-07-24)
- `transformers-qwen3-moe-fused/example_train_30b_a3b_gguf.py`, `qwen3_moe_fused/quantize_gguf/dequant.py`,
  `requirements.txt`, README §"LoRA over GGUF".
- `torch-ggml-ops/setup.py`, `README.md` §backward, `csrc/ck/bf16_wmma.cuh`, `csrc/ck/grouped_mmq_backward.cuh`,
  `csrc/vendor/llama_cpp/`.
- 35B GGUF: https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF (file `Qwen3.6-35B-A3B-APEX-I-Mini.gguf`).
