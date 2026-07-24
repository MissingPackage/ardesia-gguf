# ardesia-gguf

A **LoRA-over-GGUF** training stack, plus a **CUDA / tensor-core port** of woct0rdho's
[`torch-ggml-ops`](https://github.com/woct0rdho/torch-ggml-ops) backward kernels — built to fine-tune a
**~35B Qwen3 MoE (35B-A3B)** with LoRA inside the **16 GB** of an RTX 4090 Laptop.

The idea: keep the base model **quantized in GGUF** and dequantize on the fly inside the training
kernels, so the weights never have to be materialized in full precision. woct0rdho's stack does exactly
this, but its fused backward kernels were written for AMD (gfx11) GPUs. This repo makes them build and
run on NVIDIA, on native tensor cores, and packages the smoke experiments and benchmarks around it.

## Status (honest)

- **Kernel port: done and PR'd.** The `torch-ggml-ops` backward runs on native NVIDIA tensor cores
  (`mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32`), dual-path with the original ROCm/HIP build. **67/67
  tests** on the real 35B GGUF tensors. Opened as a PR upstream on `woct0rdho/torch-ggml-ops`
  (2026-07-24). Write-up: [`docs/05-porting-kernel-cuda.md`](docs/05-porting-kernel-cuda.md); PR body:
  [`patches/torch-ggml-ops-pr-body.md`](patches/torch-ggml-ops-pr-body.md).
- **Stack validated by a smoke run.** The Path A (Triton) LoRA-over-GGUF loop trains within 16 GB.
- **Full 35B training: not yet run.** The next open step is measuring forward-only VRAM for the real
  35B on the 16 GB card (13.33 GiB of weights vs ~15 GiB effective ceiling) before committing a run.

## Spike results

**LoRA-over-GGUF smoke — PASS.** Qwen3-30B-A3B-Instruct-2507 **UD-IQ2_M**, LoRA rank 4, batch 1 /
ctx 2048, 8/8 steps, sane loss (0.55–0.59) and gradients. **Peak VRAM 14.16 GiB allocated / 14.31 GiB
reserved**, comfortably under the 16 GB card (effective ceiling ~15 GiB with a KDE desktop up).
Details: [`docs/02-spike1-smoke-experiment-log.md`](docs/02-spike1-smoke-experiment-log.md).

- **IQ3_XXS (~3.1 bpw) does not fit** on 16 GB shared with a desktop — it OOMs; **IQ2_M (~2.7 bpw)
  fits.** The upstream "16 GB" figure assumes a headless GPU.
- **IQ2_XS dequant was missing upstream** and is the dominant tensor type in 2-bit UD-quants; the port
  was contributed and **merged upstream** (`transformers-qwen3-moe-fused` PR #23).

**Tensor-core port vs the portable shuffle-emulation** (RTX 4090 Laptop, real 35B GGUF tensors):

| kernel | shuffle emulation | tensor core | speedup |
|---|---|---|---|
| `grouped_mmq_grad_input` (5 quant types) | 1.25–4.65 ms | 0.32–1.18 ms | **3.9× – 4.3×** |
| `grouped_mmq_pair_grad_input` | 8.69–8.97 ms | 0.83–0.90 ms | **9.7× – 10.8×** |

In absolute terms the backward now runs at **0.6–0.78 of a dequantization-free bf16 cuBLAS GEMM** — the
matrix multiply is no longer the bottleneck; what remains is the inherent GGUF decode.

## Setup

Python is `uv`-managed (a dedicated venv, kept separate from any sibling project's venv). This repo does
**not** vendor the third-party clones or model weights.

1. **Vendored dependencies.** The four woct0rdho repos this builds against are pinned by commit in
   [`docs/VENDORED.md`](docs/VENDORED.md), which also lists how to clone them and apply the patches in
   [`patches/`](patches). They are not shipped here.
2. **Model weights.** Download the GGUF base yourself (e.g. `mudler/Qwen3.6-35B-A3B-APEX-GGUF` for the
   35B, or an `unsloth` 30B-A3B UD-quant for the smoke). Weights are git-ignored and never committed.
3. **Path B build (CUDA kernels).** `source scripts/pathb-env.sh` then `bash
   scripts/pathb-link-cudalibs.sh`, then build `torch-ggml-ops`. The toolchain is uv-native (no conda):
   torch + `nvcc`/CUDA-13.3 from **pip wheels**, host compiler **gcc-15** (CUDA 13 rejects newer gcc).
   The two non-obvious `nvcc` flags live in `NVCC_APPEND_FLAGS` (in `pathb-env.sh`), so upstream
   `setup.py` stays clean. See [`docs/04-spike2-pathb-port.md`](docs/04-spike2-pathb-port.md).

## Scripts

- **`scripts/bench_backward.py`** — benchmarks the dense + grouped backward kernels on real GGUF
  tensors. `--tag` writes `outputs/bench-<tag>.json`, `--compare A B` diffs two builds, `--moe` measures
  only the two shapes training actually runs against a fair bf16-GEMM ceiling.
- **`scripts/verify_backward.py`** — correctness + kernel-coverage at *production* shapes: names the
  CUDA kernel that actually ran (`torch.profiler`) and checks grad_input against a dequant reference.
- **`scripts/probe_route_boundaries.py`** — root-cause probe that scored a failing upstream test's
  oracle against fp64 ground truth (and showed the oracle, not the kernel, was wrong).
- **`scripts/run_smoke.sh` → `scripts/smoke_train_gguf.py`** — the Path A LoRA-over-GGUF smoke loop.
- **`docs/port-microtests/`** — standalone CUDA microtests validating the tensor-core fragment layouts
  in isolation (no torch, no model).

## Documentation

- [`docs/00-intent-and-open-questions.md`](docs/00-intent-and-open-questions.md) — why the tier jump, open questions.
- [`docs/01-cuda-porting-assessment.md`](docs/01-cuda-porting-assessment.md) — the two porting paths (evidence-based).
- [`docs/02-spike1-smoke-experiment-log.md`](docs/02-spike1-smoke-experiment-log.md) — the smoke experiment log.
- [`docs/03-upstream-woct0rdho.md`](docs/03-upstream-woct0rdho.md) — upstream contributions (IQ2_XS, chunked dequant, issue).
- [`docs/04-spike2-pathb-port.md`](docs/04-spike2-pathb-port.md) — the CUDA port experiment log.
- [`docs/05-porting-kernel-cuda.md`](docs/05-porting-kernel-cuda.md) — the shareable port write-up.
- [`docs/VENDORED.md`](docs/VENDORED.md) — pinned third-party repos + how to fetch.

## Related projects

- **woct0rdho's stack** (upstream, the foundation of this work):
  [`torch-ggml-ops`](https://github.com/woct0rdho/torch-ggml-ops),
  [`transformers-qwen3-moe-fused`](https://github.com/woct0rdho/transformers-qwen3-moe-fused),
  [`transformers5-qwen3.5-recipe`](https://github.com/woct0rdho/transformers5-qwen3.5-recipe).
- **ardesia-unsloth** — sibling project: the 4B Unsloth training stack this one branched from for the
  35B MoE tier jump.

## License

MIT — see [`LICENSE`](LICENSE). The vendored `torch-ggml-ops` keeps its own upstream (Apache-2.0)
license and is not included in this repository; only the patch against it is (`patches/`).
