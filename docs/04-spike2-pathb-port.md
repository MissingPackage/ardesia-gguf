# 04 — Spike-2: Path B port (torch-ggml-ops → CUDA) — experiment log

**Goal:** implement Path B — make woct0rdho's `torch-ggml-ops` (LoRA-over-GGUF kernels) build and
run on our RTX 4090 Laptop (Ada, sm_89, CUDA 13.3 driver, 16 GB), so the Qwen3.6-35B-A3B-APEX
recipe becomes trainable on NVIDIA. Started 2026-07-24 (session 3).

All work is in `vendor/torch-ggml-ops/` (fresh clone of woct0rdho/torch-ggml-ops @ d4613a1) + a
dedicated uv venv `.venv-pathb`. The upstream is ROCm/HIP (gfx1151); we ported it dual-path so the
AMD build is preserved (candidate for an upstream PR).

## Toolchain (M1 — DONE, verified)

uv-native, **no conda**. See `scripts/pathb-env.sh`.
- torch 2.13.0+cu130 in `.venv-pathb`.
- nvcc + CUDA-13.3 dev headers from **pip wheels** (`nvidia-cuda-nvcc` etc.; unified prefix
  `nvidia/cu13`). The `-cu13`-suffixed wheels are deprecated stubs — use the unsuffixed names.
- host compiler **gcc15** (`sudo dnf install gcc15 gcc15-c++`): CUDA 13 rejects the box's gcc 16.
- Two non-obvious nvcc flags — `-ccbin /usr/bin/g++-15` and `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`
  (nvcc 13.3 vs torch's pinned cudart 13.0 headers). **These live in `NVCC_APPEND_FLAGS`
  (`scripts/pathb-env.sh`), NOT in `setup.py`** — setup.py is kept upstream-clean for the PR.
- `scripts/pathb-link-cudalibs.sh` creates the unversioned `.so` dev symlinks the wheels omit.
- **Verified:** built a real torch CUDAExtension, ran a bf16 kernel on the 4090 (`EXT SMOKE OK`).
  This answers docs/00 §spike-zero question 1 (does it build/run on CUDA): **yes**.

### Build & test (copy-paste)
```bash
source scripts/pathb-env.sh                                   # CUDA_HOME, g++-15, NVCC_APPEND_FLAGS
bash scripts/pathb-link-cudalibs.sh                           # once per venv (dev symlinks)
cd vendor/torch-ggml-ops && python setup.py build_ext --inplace -j 8   # ~8 min (ptxas slow, ~6000-line TU)
GGUF_MMQ_TEST_MODEL=$PWD/../../models/Qwen3.6-35B-A3B-APEX-I-Mini.gguf \
  ../../.venv-pathb/bin/python -m pytest tests/ -q
```
Validated standalone fragment-layout microtests + how-to: `docs/port-microtests/` (README).

## Source mapping — what the port actually is (corrects docs/01)

- `ck` in torch-ggml-ops is a **local namespace** (`torch_ggml_ops::ck` = "custom kernels"),
  **not** AMD Composable Kernel. There is **no** AITER / rocBLAS / hipBLAS / CK-library dependency.
- The forward (`mmq_core.cuh` + `vendor/llama_cpp/*`) is **dual-path llama.cpp**: the NVIDIA
  `TURING_MMA_AVAILABLE` branch is present, dormant behind AMD `#define`s in `common.cuh`.
- The only genuinely AMD-hardware kernel is `ck/bf16_wmma.cuh` (42 lines): the gfx11 intrinsic
  `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` + its per-lane C-fragment layout. **Every** backward
  kernel touches the MMA only through that header's contract (`bf16_fragment`, `f32_accumulator`,
  `wmma_f32_16x16x16_bf16`, `c_row`, `c_column`, `fragment_data`).

## Port edits (M2 — DONE, compiles)

- `csrc/port_cuda.cuh` (new): CUDA shim mapping HIP names (`__hip_bfloat16`→`__nv_bfloat16`,
  `hipStream_t`→`cudaStream_t`, `hipError_t`, `hipSuccess`, `hipGetErrorString`, `hipGetLastError`).
- `common.cuh`: dual-path — under `#if defined(__HIP__)` the original AMD macros/headers/shfl-macros/
  `__vsubss4` shims; else CUDA headers + `TURING_MMA_AVAILABLE`/`AMPERE_MMA_AVAILABLE` + native
  intrinsics. Also flattened `block_q4_K`/`block_q5_K` anonymous `union{struct{half d,dmin};half2 dm}`
  to `half d; half dmin;` (CUDA `__half` has a ctor → illegal in an anonymous aggregate; identical
  4-byte layout), and rewrote the 4 `bxi->dm` uses to `make_half2(bxi->d, bxi->dmin)`.
- `mmq_hip.cu`: undef `__CUDA_NO_HALF*`/`__CUDA_NO_BFLOAT16_*`; guard the hip header; forward-declare
  `aoti_torch_get_current_cuda_stream` (torch 2.13 hides it behind `#ifdef USE_CUDA` in the inductor
  shim; symbol still exported by libtorch_cuda).
- `mmq_core.cuh`: define `VDR_Q3_K_Q8_1_MMQ 2` (llama.cpp-canonical; the AMD build never compiled the
  Turing branch that needs it).
- `ck/mmq_backward.cuh`, `ck/grouped_mmq_backward.cuh`: guard direct hip includes.
- `setup.py`: **left upstream-clean** (the two env-specific nvcc flags go via `NVCC_APPEND_FLAGS` in
  `scripts/pathb-env.sh`, documented in the PR body for the maintainer to decide).

## Backward MMA port (M4 — DONE, validated)

`ck/bf16_wmma.cuh` dual-path. CUDA path = a **correctness-first warp-shuffle + fp32-FMA emulation**
of the gfx11 16x16x16 bf16 WMMA that honors the exact fragment contract (`c_row`/`c_column`
unchanged), so all ~4000 lines of backward kernels compile byte-identical.
- Standalone microtest (`/tmp/wmma_microtest.cu`): max abs err 2.4e-7, **0/256 mismatches** vs a
  reference matmul with identical bf16 rounding.
- On the real 35B GGUF (tests/test_mmq.py):
  - `test_native_backward_decodes_every_logical_weight_value`: **PASS all 5** quant types.
  - `test_backward_is_exact_logical_weight_jacobian` (atol=0): **EXACT for Q3_K/Q4_K/Q5_K/Q6_K**;
    IQ2_S off by **1 bf16 ULP on 1/20480** elements (fp32 accumulation-order artifact, benign).
- TODO(perf): the emulation is O(64 shfl)/MMA — replace with `mma.sync.m16n8k16` + `ldmatrix` once
  everything is green. Correctness first.

## Forward correctness (M3 — DONE, 5/5 PASS)

First run: **all 5 forward quant types wrong** (normalized RMSE ~2.4 vs <0.04). Root-caused to
`mmq_write_back_bf16` (`mmq_core.cuh`) using an **unconditional** `tile<16,16,int,DATA_LAYOUT_J_MAJOR>`
(the RDNA3 J-major C-fragment layout) to read the accumulated `sum[]`, while on NVIDIA the vec-dot
fills `sum[]` with the I-major `tile<16,8,int>` — the write-back and accumulation disagreed on the
layout and scrambled the store. Fix (applied): `mmq_write_back_bf16`'s `tile_C` now mirrors the
vec-dot per platform (AMD → J-major 16x16; NVIDIA → I-major 16x8). After the fix: **forward 5/5 PASS**
(RMSE < 0.04, all quant types).

## Full test results (M5) — Path B numerically validated on CUDA

Against the real `Qwen3.6-35B-A3B-APEX-I-Mini.gguf` (its actual IQ2_S/Q3_K/Q4_K/Q5_K/Q6_K tensors):
- `tests/test_mmq.py` (dense): **19 / 20 pass**.
- `tests/test_grouped_mmq.py` (grouped / MoE-fused — the path the 35B-A3B experts use): **24 / 26 pass**.

The **3 failures are all one benign class**: the tests assert `atol=0, rtol=0` (bit-exact), which the
AMD WMMA's specific fp32 reduction order satisfies but our correctness-first shuffle emulation (a
different, equally-valid summation order) does not. Measured on the failing `route_group_boundaries`
(Q4_K): **every element within 1 bf16 ULP** (at `atol=2^-8`, 0 / 320000 mismatched; normalized RMSE
2.6e-3; max |abs| 2^-9). The dense IQ2_S jacobian miss is 1 bf16 ULP on 1 / 20480. None is a logic
bug — forward garbage would (and did, pre-fix) show RMSE ~2.4, not 1 ULP. **Conclusion: the port is
numerically correct for bf16 LoRA training.** Forward exact-enough; backward correct to ≤1 ULP.

For an eventual upstream PR the emulation could be swapped to `mma.sync.m16n8k16 + ldmatrix` (also a
perf win, ~64 shfl/MMA today) and the tests may need `atol` relaxed for a non-AMD accumulation order.

## Perf pass — native tensor-core backward (in progress, 2026-07-24)

Project decision: training pauses; make the PR maximal — replace the correctness-first shuffle
emulation with native NVIDIA tensor cores. Approach: reuse llama.cpp's tested `mma.cuh` tile
abstraction (`mma.sync.m16n8k16.f32.bf16.bf16.f32`, mma.cuh:1184), dual-path (`#if defined(__HIP__)`
keeps the amdgcn WMMA). B loaded from the existing shared tile, A from global grad_output, no bridge
(so no added shared memory). Validated the primitive + exact load/mma/store pattern standalone
(0 mismatch) before touching kernels. `mmq_backward.cuh` (dense) ported first.

**Two landmines (both cost a debug cycle; they WILL recur porting the grouped kernels):**
1. `mma.cuh` tile `get_i`/`get_j` read **`threadIdx.x` assuming it is the warp lane (0..31)** — true
   for the forward's `(32, nwarps)` block, FALSE for the backward's flat `(128,1)` block. Using them
   directly makes warps 1–3 index out of bounds → `cudaErrorIllegalAddress`. Fix: index fragments
   with the existing `lane = threadIdx.x % 32` via explicit layout formulas (below), NOT get_i/get_j.
   A 32-thread microtest cannot catch this — use ≥4 warps.
2. The **`nv_bfloat162` tiles use a separate specialization** (mma.cuh:436) with a DIFFERENT fragment
   layout than the generic `tile<I,J,T>`. Correct per-lane formulas (verified by runtime probe):
   - A `tile<16,8,nv_bfloat162>` (ne=4): `i = (l&1)*8 + lane/4`, `bf162col = (l/2)*4 + lane%4`
   - B `tile<8,8,nv_bfloat162>`  (ne=2): `i = lane/4`,          `bf162col = l*4 + lane%4`
   - C `tile<16,8,float>` (generic, ne=4): `i = (l/2)*8 + lane/4`, `col = (lane%4)*2 + l%2`
   (bf16 K column = 2·bf162col; a fragment element packs {[i][2c],[i][2c+1]}.)

## Perf pass — COMPLETE (2026-07-24, session 4): grouped ported, emulation deleted

The grouped kernels are now on native tensor cores too, and **no shuffle emulation remains anywhere**
(the `TODO(perf)` in `bf16_wmma.cuh` is gone with the code it described).

### The shape of the change (this is the part that matters for the PR)

The naive route — an `#if defined(__HIP__) / #else` branch inside each of the ~13 grouped kernel
variants, like the dense kernel got first — would have added ~800 lines of near-duplicate CUDA to a
repo whose maintainer only has AMD hardware. Instead the **seam was widened** so no kernel body
branches at all. `bf16_wmma.cuh` now exposes:

| before (leaked the gfx11 layout) | after (platform-neutral) |
|---|---|
| `bf16_fragment` for both operands | `bf16_fragment_a` / `bf16_fragment_b` |
| `fragment_data(f)` + a hand-written 16-element fill loop | `load_a_fragment<CHECK_M,CHECK_K>(...)`, `load_b_fragment(...)`, `backward_shared_b_tile::load_b_fragment<VECTOR_LOAD>(...)` |
| `acc.values[e]` | `acc.value(e)` |
| `c_column(lane,e)` for M, `c_row(lane)` for N | `acc_m(lane,e)` / `acc_n(lane,e)` — **N now depends on the element too**, which is the whole reason the old contract could not express the NVIDIA layout |

Every fill and store loop in every backward kernel became a call to one of those. The AMD branch of
each helper is the loop that used to be inline, so gfx11 codegen should be unchanged (⚠ **not
verifiable here** — no AMD hardware; flagged for the maintainer). Result: **+403 / −335 over 4 files**
— the kernels got *smaller*, and all platform divergence lives in one 250-line header.

Net effect on the whole port vs upstream master: 9 files, +538 / −320.

### Measured (RTX 4090 Laptop, 4096 rows x 8 groups, real 35B GGUF tensors, warm clocks, spread <5%)

`scripts/bench_backward.py` (`--tag` writes `outputs/bench-<tag>.json`, `--compare A B` diffs them):

| kernel | emulation | tensor core | speedup |
|---|---|---|---|
| `grouped_grad_input` (5 quant types) | 4.65 / 4.56 / 1.25 / 1.29 / 4.65 ms | 1.18 / 1.07 / 0.32 / 0.33 / 1.07 ms | **3.9x – 4.3x** |
| `grouped_pair_grad_input` (IQ2_S, Q3_K) | 8.97 / 8.69 ms | 0.83 / 0.90 ms | **9.7x – 10.8x** |
| `dense_grad_input` (already tensor-core) | — | — | 1.01x – 1.26x |

(The dense kernel gained too: routing its B loads through the new seam replaced two 16-bit shared
loads per pair with one 32-bit load.) **Geomean 2.79x**, diluted by the already-ported dense.

### Correctness

- Unit tests: **67 / 67** (58 existing + 9 added). The grouped IQ2_S jacobian passes *exactly* — same
  effect the dense port had: the tensor core's accumulation order matches the torch reference, which
  the emulation's order did not.
- **`test_grouped_backward_route_group_boundaries` was failing because its ORACLE is wrong, not the
  kernel.** It asserts `atol=0` against a torch bf16 matmul, and PyTorch defaults
  `allow_bf16_reduced_precision_reduction` to `True` — the backend may reduce split-K partials in
  bf16. Scored against an fp64 accumulation of the same bf16 products
  (`scripts/probe_route_boundaries.py`): the reference gets **63.2%** of outputs correctly rounded,
  the kernel **99.998%**; on the 117611 disagreeing elements the kernel is right 117605 times and the
  reference once. The per-group table is the fingerprint — the reference is 100% correct at 1/15/16
  rows and drops to ~60% from 17 up, exactly where cuBLAS changes tiling. Fix: correct the oracle
  (flag off) *and* bound at one bf16 ULP, since the irreducible residue is 4 / 320000 elements at
  exactly 1.00 ULP (two valid fp32 orderings can straddle a rounding boundary). Minimum rtol that
  passes: 0.0075758, versus `2**-7 = 0.0078125`.
  Presumed no-op on AMD (ROCm doesn't take that bf16-reduction path, which is why `atol=0` held
  there) — **not verifiable here**.
- **The suite reached none of the tiled / row-task kernels.** They are gated on
  `out_features == 2048 && in_features == 512` (the down projection); every test used
  `out_features=37`. Added 9 cases (3 quant types × 3 row regimes) that, per `torch.profiler`, cover
  **8 distinct previously-unreached kernels**: `{q4,q5,iq2}_row_task`, `q4_small_s2`, `iq2_s2`,
  `q4_small`, `q5_small`, `iq2_tiled<true>`. At K=2048 a pure `rtol` is meaningless (cancellations
  need 1.3 to pass), so the bound is `rtol=2**-7, atol=2**-18` against a measured worst case of
  1.75e-6 on an output RMS of 0.33 – 0.48.
- **`scripts/verify_backward.py` — new, and it closes a real coverage hole.** The unit tests use
  `out_features=37`, which dispatches to the *generic* grouped kernel; the 35B-A3B down-projections
  (out=2048, in=512) dispatch to the **tiled row-task kernels, which the tests barely touch**. This
  script runs the real expert shapes, names the kernel that actually ran (`torch.profiler`), and
  compares against an fp32 reference built from the kernel's *exact* inputs (weights rounded to bf16
  as the kernel stages them in shared memory, TF32 disabled — both matter, see below). Result:
  **5/5**, with **99.88–99.99% of outputs bit-identical** to the correctly-rounded fp32 answer;
  2–190 elements out of 0.5M–8.4M land more than one bf16 step away, all near-zero cancellations.

⚠ **Two measurement traps worth remembering** (both produced convincing garbage before being fixed):
absolute tolerances calibrated on the `out_features=37` tests are meaningless at K=2048 (outputs are
~40x larger, so one bf16 ULP is ~40x wider); and a *per-element* ULP metric explodes on cancellation
outputs, reporting "2.7M ULP error" for a kernel that was in fact bit-exact. Compare against the
output's own RMS, and make the reference use the kernel's real inputs.

### Remaining NVIDIA headroom (measured against a FAIR ceiling)

This took three attempts to get right, and the two wrong answers are worth keeping because both were
confidently wrong in a way the numbers alone did not reveal.

**Attempt 1 — wrong shape.** Benchmarking every quant type at a fixed `out_features=512` showed
`grouped_mmq_grad_input` at 0.22 of the forward and suggested "one slow kernel, and it's gate/up's".
Both halves false: `fast_moe_lora.py` (:386, :412; `docs/plan_qwen3.5.md` §84; asserted in
`test_fast_moe_lora.py:152`) routes gate/up through `grouped_mmq_pair`, and the down projection is
out=2048/in=512, a different dispatch entirely. → measure only the two shapes training runs
(`--moe`).

**Attempt 2 — wrong ceiling, and cross-run ratios.** Using the *forward* as the ceiling gave
0.75–0.95 and "essentially no headroom left". Two independent errors: (a) the forward and backward
numbers came from different runs, and this laptop drifts 5–15% between runs (`nvidia-smi -q -d
PERFORMANCE` shows `SW Power Capping` accumulating); (b) **the forward is not a fair ceiling for a
bf16 backward** — `grouped_mmq` quantizes activations to Q8_1 and runs on **int8** tensor cores,
roughly 2x the bf16 rate on Ada. Comparing a bf16 kernel to an int8 one flatters it.

**The fair ceiling** is the same grouped matmul in bf16 with the weights *already dequantized*
(`torch.bmm` / `baddbmm` on the dequantized experts): cuBLAS doing exactly the arithmetic our kernel
does and none of the GGUF decode. Timed **in the same run** as the kernel it is compared to:

| real training path | ours | bf16 GEMM, no dequant | ratio |
|---|---|---|---|
| gate/up, `grouped_mmq_pair_grad_input`, Q3_K | 19.6 TFLOP/s | 31.5 | 0.62 |
| gate/up, `grouped_mmq_pair_grad_input`, IQ2_S | 20.8 | 26.7 | 0.78 |
| down, `grouped_mmq_grad_input`, Q4_K | 22.4 | 37.3 | 0.60 |
| down, `grouped_mmq_grad_input`, Q5_K | 25.5 | 40.5 | 0.63 |

**So the backward runs at ~0.6–0.78 of a dequantization-free bf16 GEMM.** The MMA is no longer the
bottleneck — that was the 4-10x this port bought. What remains is the GGUF decode, which the oracle
does not pay at all and which is inherent to the design (the weights stay quantized; that is the
entire point of the library). Whether *that* can be cut is a separate and much larger question,
untouched here.

Reproduce: `scripts/bench_backward.py --moe`, keys `*_CEILING_bf16gemm/*`.

## Remaining (next session)

- **M6:** clone/wire `transformers5-qwen3.5-recipe` + fork `transformers-gguf` (already installed),
  load the 35B, **measure forward-only VRAM** on the 16 GB 4090 (⚠ fit: 13.33 GiB weights vs ~15
  effective). Only after this: copy persona/calibration data and train (serve **unmerged**).
- ~~Perf: replace the bf16_wmma shuffle emulation with tensor-core `mma.sync`.~~ **Done** (above).
- **Upstream:** the branch is restructured into 3 thematic commits and the body is written; the only
  remaining step is the force-push, deliberately left to the author (it rewrites already-published
  history, and the PR is opened by hand).
- Micro-optimisation left on the table, deliberately: the NVIDIA C layout puts accumulator elements
  `(0,1)` and `(2,3)` in adjacent output columns of the same row, so the write-back could use 32-bit
  stores instead of pairs of 16-bit ones. Not done because the evidence says stores are not the
  bottleneck — the pair kernel amortises stores over *twice* the MMAs yet runs *slower* per FLOP than
  the down kernel — and it would push a layout fact back into the kernels the seam exists to hide.
