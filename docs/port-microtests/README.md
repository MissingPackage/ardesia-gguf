# Port microtests — validated tensor-core building blocks

Standalone CUDA harnesses that validate the NVIDIA backward-MMA port of `torch-ggml-ops`
IN ISOLATION (no torch, no model), so the fragment layouts are proven before touching the
1000s of lines of kernel code. Each compares against a plain fp32-from-bf16 reference matmul.

## Build & run (any of them)

```bash
source scripts/pathb-env.sh                       # CUDA_HOME, g++-15, arch sm_89
nvcc -ccbin /usr/bin/g++-15 -arch=sm_89 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK \
  -I vendor/torch-ggml-ops/csrc docs/port-microtests/03_multiwarp_lane_formulas.cu \
  -o /tmp/t -L"$CUDA_HOME/lib" -lcudart && /tmp/t
```

(They `#include "vendor/llama_cpp/*.cuh"`, so a `vendor/torch-ggml-ops` checkout must be present;
restore it with `patches/torch-ggml-ops-cuda-port.patch` — see HANDOFF.)

## What each proves

- **00_shuffle_emulation.cu** — the *correctness-first* shuffle+fp32 emulation of the gfx11 WMMA
  (`ck/bf16_wmma.cuh` CUDA path). 0/256. This is what the GROUPED kernels still use.
- **01_mma_primitive.cu** — one `mma.sync.m16n8k16.f32.bf16.bf16.f32` via `mma.cuh`'s tile abstraction
  (16×8 output), fragments loaded via `get_i/get_j`. 0/128. Proves the tensor-core op itself.
- **02_kernel_pattern_16x16.cu** — the exact per-kernel pattern: A from global, B from shared, 16×16
  via two N-half mma, natural output layout. 0/256. **32-thread block (1 warp)** — passes but does
  NOT exercise the multi-warp lane bug.
- **03_multiwarp_lane_formulas.cu** — the DEFINITIVE reference: **128-thread (4-warp) block** using
  `lane = threadIdx.x % 32` + the explicit fragment formulas (NOT `get_i/get_j`). 0/1024. This is the
  layout the dense kernel now uses and the grouped kernels must use. **Landmine 1** (mma.cuh tiles
  assume threadIdx.x==lane; a flat (128,1) block breaks warps 1–3 with an illegal address) is only
  caught here, not by the 32-thread tests.
- **04_getij_vs_formula_probe.cu** — prints `tile::get_i/get_j` vs the explicit formulas per lane.
  How **Landmine 2** was found: the `nv_bfloat162` tiles use a SEPARATE specialization (mma.cuh:436)
  with a different layout than the generic `tile<I,J,T>` (only the A fragment differs).

## The validated fragment formulas (lane = threadIdx.x % 32)

- A `tile<16,8,nv_bfloat162>` (ne=4): `i = (l&1)*8 + lane/4`,  `bf162col = (l/2)*4 + lane%4`
- B `tile<8,8,nv_bfloat162>`  (ne=2): `i = lane/4`,           `bf162col = l*4 + lane%4`
- C `tile<16,8,float>` (generic, ne=4): `i = (l/2)*8 + lane/4`, `col = (lane%4)*2 + l%2`

bf16 K column = 2·bf162col; a bf162 fragment element packs `{M[i][2c], M[i][2c+1]}`.
See `csrc/ck/mmq_backward.cuh` (the `#else // CUDA` branch) for the applied dense-kernel version.
