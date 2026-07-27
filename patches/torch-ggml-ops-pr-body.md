# Add NVIDIA CUDA support (dual-path with ROCm/HIP)

This makes the same sources build and run on NVIDIA GPUs while leaving the existing
gfx1151/RDNA3 HIP build unchanged. Every NVIDIA-only change is behind `#if defined(__HIP__)`
(or the existing `AMD_*` / `TURING_MMA_AVAILABLE` guards), so `hipcc` sees the original code.

Tested on an RTX 4090 Laptop (Ada, sm_89), CUDA 13.3 toolchain, PyTorch 2.13, against
`Qwen3.6-35B-A3B-APEX-I-Mini.gguf` (IQ2_S / Q3_K / Q4_K / Q5_K / Q6_K).

The backward runs on native tensor cores (`mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32`), not on a
portability shim.

## What was needed

- **HIP→CUDA compat shim** (`csrc/port_cuda.cuh`, new): maps the small HIP surface the code uses
  (`__hip_bfloat16`/`__hip_bfloat162`, `hipStream_t`, `hipError_t`, `hipSuccess`,
  `hipGetErrorString`, `hipGetLastError`) onto their CUDA equivalents. Included only on the CUDA
  path; `common.cuh`/`ck/*` include it in place of `<hip/*>`.
- **`common.cuh` dual-path**: CUDA headers instead of `<hip/*>`; `TURING_MMA_AVAILABLE` +
  `AMPERE_MMA_AVAILABLE` instead of the `GGML_USE_HIP`/`RDNA*`/`AMD_WMMA_AVAILABLE` block; drop the
  `__shfl_*_sync`→`__shfl_*` macros and the `__vsubss4`/`__vsub4`/`__vcmpne4` reimplementations on
  CUDA (native there); keep `nv_bfloat16` native. Also flattens the `block_q4_K`/`block_q5_K`
  anonymous `union { struct { half d, dmin; }; half2 dm; }` to plain `half d; half dmin;` — on CUDA
  `__half` has a constructor and is illegal in an anonymous aggregate. Layout is byte-identical; the
  handful of `bxi->dm` uses in `mmq-load-targets.cuh` become `make_half2(bxi->d, bxi->dmin)`.
- **Forward**: define `VDR_Q3_K_Q8_1_MMQ` (2, from `vecdotq.cuh`) — the AMD build never compiles the
  `TURING_MMA_AVAILABLE` vec-dot branch that references it. And in `mmq_write_back_bf16`, select the
  accumulator C-tile layout per platform: RDNA3 uses `DATA_LAYOUT_J_MAJOR` (16×16), NVIDIA uses the
  I-major `tile<16,8,int>`. With the J-major tile hardcoded, the write-back and the Turing vec-dot
  disagree on the `sum[]` layout and the output is scrambled (nrmse ~2.4).
- **`mmq_hip.cu`**: also `#undef` the `__CUDA_NO_HALF*` / `__CUDA_NO_BFLOAT16_CONVERSIONS__` macros
  (as the HIP ones already are); forward-declare `aoti_torch_get_current_cuda_stream` — recent
  PyTorch guards it behind `#ifdef USE_CUDA` in the inductor C shim, but the symbol is still exported
  by `libtorch_cuda`.
- **The backward matrix-core seam** — the only structural change, described below.

## The backward: widening the seam instead of forking the kernels

gfx11's `wmma_f32_16x16x16_bf16_w32` and NVIDIA's `mma.sync.m16n8k16` compute the same 16×16×16
product but distribute it across the wave completely differently. The old `ck/bf16_wmma.cuh` contract
leaked the gfx11 distribution into every kernel: a lane owned row `c_row(lane)` of both operands and
filled all 16 K values itself, and its accumulator elements sat at `C[c_column(lane,e)][c_row(lane)]`
— a **fixed** N coordinate. NVIDIA's layout cannot be expressed that way (its N coordinate depends on
the element), so the kernels had to stop indexing fragments by hand.

Rather than add an `#if defined(__HIP__) / #else` branch to each of the ~13 backward kernel variants
— roughly 800 lines of near-duplicate CUDA in a codebase you maintain on AMD — I widened the seam so
that **no kernel body branches at all**:

| before | after |
|---|---|
| `bf16_fragment` for both operands | `bf16_fragment_a` / `bf16_fragment_b` (AMD aliases both to the old type) |
| `fragment_data(f)` + a hand-written 16-element fill loop at each site | `load_a_fragment<CHECK_M, CHECK_K>(...)`, `load_b_fragment(...)`, `backward_shared_b_tile::load_b_fragment<VECTOR_LOAD>(...)` |
| `acc.values[e]` | `acc.value(e)` |
| `c_column(lane,e)` for M and `c_row(lane)` for N | `acc_m(lane,e)` / `acc_n(lane,e)` |

Every fill and store loop in every backward kernel is now a call to one of those. **The AMD branch of
each helper is verbatim the loop that used to be inline at the call sites**, including the
`VECTOR_LOAD` knob that used to be the `VECTOR_LOCAL_LOAD` template parameter, so gfx11 should get
identical codegen. The net effect on the kernels is *fewer* lines (+438 / −301 across the four
backward files, against 9 files and +557 / −320 for the port as a whole), with all platform divergence confined to one header.

**Please sanity-check the AMD side.** I have no AMD hardware and could not compile, let alone run,
the HIP path — that is the one part of this PR I cannot back with evidence.

**NVIDIA arch floor: compute capability 8.0.** `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` is
Ampere-and-later. `common.cuh` therefore gates `TURING_MMA_AVAILABLE` / `AMPERE_MMA_AVAILABLE` on
`__CUDA_ARCH__` the way upstream llama.cpp does, plus an `#error` for pre-Ampere device builds —
without it a `TORCH_CUDA_ARCH_LIST=7.5` build emits the instruction anyway and dies inside `ptxas`
with nothing pointing at the cause. Verified both ways (`-arch=compute_89` compiles;
`-arch=compute_75` stops with that message). Say the word if you'd rather have a Turing fallback than
a hard error.

## Test results

Against the real APEX-I-Mini tensors: **67 / 67 pass** (58 existing + 9 added, see below).

Two test changes come with this, both in `tests/test_grouped_mmq.py`. I'd rather flag them loudly
than slip them in, since touching an assertion to make your own patch pass is exactly the move that
should be viewed with suspicion:

**1. `test_grouped_backward_route_group_boundaries` was asserting against an inaccurate oracle.**
It compares to a `torch` bf16 matmul under `assert_close(rtol=0, atol=0)`. PyTorch defaults
`allow_bf16_reduced_precision_reduction` to `True`, so the backend may reduce split-K partial sums in
**bf16**. Scored against an fp64 accumulation of the same bf16 products:

| | outputs that are the correctly-rounded value |
|---|---|
| the test's reference (default flags) | 202389 / 320000 (**63.2%**) |
| the same reference, reduced-precision reduction off | 319995 / 320000 (99.998%) |
| this kernel | 319993 / 320000 (99.998%) |

On the 117611 elements where kernel and reference disagreed, **the kernel was right on 117605 and the
reference on 1**. The per-group breakdown is a clean fingerprint: the reference is 100% correct for
groups of 1, 15 and 16 rows and drops to ~60% from 17 rows up, i.e. exactly where cuBLAS switches
tiling. So the fix is to correct the oracle (a context manager turning the flag off), not to loosen
the bound. With a correct oracle the residue is **4 / 320000 elements, each exactly 1.00 bf16 ULP** —
irreducible, because two valid fp32 orderings of 37 bf16 products can straddle a rounding boundary.
Hence `rtol=2**-7` (one ULP), `atol=0`. Measured minimum that would pass: `0.0075758` vs `2**-7 =
0.0078125`.

I believe this is a no-op for the AMD build: on ROCm that flag doesn't select a bf16 split-K
reduction, which is presumably why the reference was accurate enough for `atol=0` to hold there —
but I can't run it, so please check.

**2. Nothing in the suite reached the tiled or row-task backward kernels.** Every test used
`out_features=37`, and those kernels are gated on `out_features == 2048 && in_features == 512`
(`use_grouped_backward_row_tasks`, plus the dispatch in `launch_grouped_mmq_grad_input`) — the down
projection. So the kernels a real MoE down projection actually runs had no coverage at all. Added
nine cases over three quant types × three row regimes around the dispatch thresholds; captured with
`torch.profiler`, they cover **eight distinct previously-unreached kernels**:

| rows/group | Q4_K | Q5_K | IQ2_S |
|---|---|---|---|
| 160 | `q4_row_task` | `q5_row_task` | `iq2_row_task` |
| 100 | `q4_small_s2` | `q5_small` | `iq2_s2` |
| 40 | `q4_small` | `q5_small` | `iq2_tiled<true>` |

These reduce over K=2048, where a good fraction of outputs are near-zero cancellations, so a pure
`rtol` is meaningless there (it needs 1.3 to pass); the bound is `rtol=2**-7, atol=2**-18`, against a
measured worst case of 1.75e-6 absolute on an output RMS of 0.33 – 0.48.

Worth noting: the two `atol=0` jacobian asserts that the portable shuffle path missed by one bf16
step — dense and grouped IQ2_S — now pass *exactly* on tensor cores.

## Measured

RTX 4090 Laptop, 4096 rows × 8 groups, real GGUF tensors, warm clocks, run-to-run spread < 5%.
Baseline is a portable warp-shuffle implementation of the same seam (~64 `__shfl` per MMA) that this
PR's earlier revision used:

| kernel | portable | tensor core | speedup |
|---|---|---|---|
| `grouped_mmq_grad_input` (5 quant types) | 1.25 – 4.65 ms | 0.32 – 1.18 ms | **3.9× – 4.3×** |
| `grouped_mmq_pair_grad_input` | 8.69 – 8.97 ms | 0.83 – 0.90 ms | **9.7× – 10.8×** |

## Where that leaves the backward

For a ceiling I use the same grouped matmul in bf16 with the weights **already dequantized**
(`torch.bmm` / `baddbmm` over the dequantized experts) — cuBLAS doing exactly the arithmetic these
kernels do and none of the GGUF decode — timed in the same process run as the kernel it is compared
against:

| path | this kernel | bf16 GEMM, no dequant | ratio |
|---|---|---|---|
| gate/up, `grouped_mmq_pair_grad_input`, Q3_K | 19.6 TFLOP/s | 31.5 | 0.62 |
| gate/up, `grouped_mmq_pair_grad_input`, IQ2_S | 20.8 | 26.7 | 0.78 |
| down, `grouped_mmq_grad_input`, Q4_K | 22.4 | 37.3 | 0.60 |
| down, `grouped_mmq_grad_input`, Q5_K | 25.5 | 40.5 | 0.63 |

So the backward now sits at roughly **0.6 – 0.78 of a dequantization-free bf16 GEMM**. The matrix
multiply is no longer the bottleneck; what is left is the GGUF decode, which the oracle doesn't pay
at all. I have not tried to attack that.

(I deliberately did *not* use your forward as the ceiling: `grouped_mmq` quantizes activations to
Q8_1 and runs on int8 tensor cores, roughly twice the bf16 rate on Ada, so it makes a bf16 backward
look better than it is.)

One observation in passing: the *single-projection* `grouped_mmq_grad_input` at the **gate/up** shape
(out=512, in=2048) is much further off, ~0.19 of the same ceiling. Training doesn't take that path
(`grouped_mmq_pair` does), so it may be worth nothing to you. If it ever matters: that kernel uses
`GROUPED_BACKWARD_N_PER_BLOCK = 16` with one 16x16 accumulator per wave and a `__syncthreads()` both
before *and* after each individual MMA, so the barrier is never amortised — where the tiled kernels
issue 16 MMAs per fragment load. I left it alone; restructuring it has AMD-visible consequences and
it's your call, not mine.

## Build with pip CUDA wheels

If `nvcc` is newer than the CUDA runtime headers PyTorch pins (e.g. the `nvidia-cuda-nvcc` 13.3 wheel
vs a torch built on cudart 13.0), CCCL's compat guard errors out, and if the host `gcc` is newer than
CUDA supports, `nvcc` refuses it. Both are solvable without touching `setup.py` via
`NVCC_APPEND_FLAGS="-ccbin <g++≤15> -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"`, so I left the build
script alone — let me know if you'd rather have a guarded flag in `setup.py`.
