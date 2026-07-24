# Porting the LoRA-over-GGUF training kernels to NVIDIA

**What:** port to CUDA a library of kernels for training LoRA on quantized GGUF models, written for
AMD GPUs, and do it at upstream-contribution quality rather than as a local patch.
**Outcome:** backward on native tensor cores, **4–11× faster** than the first portable version,
**67/67 tests**, PR opened on `woct0rdho/torch-ggml-ops` (2026-07-24).
**Hardware:** RTX 4090 Laptop (Ada, sm_89), CUDA 13.3, PyTorch 2.13.

---

## Summary

### Why it was needed

The project's goal is to train a **Qwen3.6-35B-A3B** (MoE, 256 experts) with LoRA inside the
**16 GB** of a laptop GPU. The only way to fit the weights is to keep them **quantized in GGUF**
and dequantize them on the fly inside the kernels, instead of materializing them in memory. The library
that does exactly this — `torch-ggml-ops` — exists and is excellent, but it is written for **AMD**: it uses
a matrix-core instruction specific to gfx11 GPUs (`wmma_f32_16x16x16_bf16_w32`). On NVIDIA it does not
even compile.

Without this port, the entire technical branch of the project was blocked.

### What we built

The library is more portable than it looks: the *forward* derives from llama.cpp and is already dual-path.
The only part genuinely tied to AMD hardware is **a 42-line header** that exposes a 16×16×16 bfloat16
matrix operation to about 4000 lines of backward kernels.

The problem is that this header was not an abstraction: **it leaked the precise way AMD distributes data
across threads into the kernels**. NVIDIA distributes it differently, and the difference is not
cosmetic — the old contract was *mathematically inexpressible* on the NVIDIA layout (detail in
appendix A).

The design choice was: instead of duplicating ~13 kernel variants with an `#if AMD /
#else NVIDIA` branch (≈800 lines of near-copied code, in a codebase maintained by one person who has
only AMD hardware), **widen the abstraction** until the kernels no longer need to know which hardware
they run on. Result: **no kernel contains a per-platform branch**, and the kernels got
*shorter* (+438/−301 lines across the 4 backward files).

### Results

Compared to the first working version (a portable emulation, correct but slow):

| kernel | before | after | speedup |
|---|---|---|---|
| `grouped_mmq_grad_input` (5 quantization types) | 1.25–4.65 ms | 0.32–1.18 ms | **3.9× – 4.3×** |
| `grouped_mmq_pair_grad_input` | 8.69–8.97 ms | 0.83–0.90 ms | **9.7× – 10.8×** |

In absolute terms, the backward now runs at **0.6–0.78 of a cuBLAS bf16 GEMM that pays no
dequantization at all** — that is, matrix multiplication is no longer the bottleneck; what
remains is the cost of decoding the GGUF format, which is inherent to the design (the weights *must* stay
quantized: that is the whole point of the library).

### How we know

- **67/67 tests** on the real 35B model (58 existing + 9 added).
- **Verification at production shapes**: against an fp32 reference built with the kernel's exact
  inputs, **99.88–99.99% of outputs are bit-identical** to the correctly-rounded value. The few
  that are not are near-zero cancellations, where any valid summation order diverges.
- **Coverage proof**: with `torch.profiler` we verify *which kernel actually runs*, instead of
  inferring it. That is how we discovered that the upstream test suite did not touch the kernels the
  real model uses at all (appendix C).

### What is not verified

**The AMD branch.** We have no ROCm hardware: we cannot compile it, let alone run it. The port is
built to not touch it (the AMD code of each function is literally the loop that used to be
inline), but it remains the only point without experimental evidence — and it is stated explicitly in the PR,
asking the maintainer to check it.

Second stated limitation: the backward now requires **compute capability 8.0+** (Ampere), because
the bf16 tensor-core instruction used does not exist on Turing. The build fails with a clear message
instead of an unintelligible error from the assembler.

### Value beyond the project

The work was done in **upstreamable** form, not as a private fork: the PR adds NVIDIA
support while preserving the AMD build, and includes two fixes that also benefit AMD users (a test
that validated against a wrong reference, and nine tests for kernels that were covered by nothing).
If it is accepted, the project stops depending on a local patch to be reapplied at every
update.

---

# Technical appendix

## A. The real problem: two incompatible matrix-core layouts

Both AMD and NVIDIA offer a hardware instruction that computes a 16×16×16 matrix product in
bfloat16 with fp32 accumulation. They compute the same thing, but **distribute the data across the 32
threads of a warp in completely different ways**.

In the original contract (AMD gfx11, wave32):

- one thread owns row `lane & 15` of *both* operands, and loads all 16 of its values;
- after the operation, element `e` of that thread's accumulator is `C[2*e + lane/16][lane & 15]`.

The critical point is the last line: **the accumulator's column coordinate depends only on the thread,
not on the element**. The kernels exploited this fact everywhere, writing things like:

```cpp
output_row    = base + c_column(lane, element);   // depends on the element
output_column = base + c_row(lane);               // does NOT depend on the element
```

On NVIDIA (`mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32`) the column **also depends on the element**.
This is not a difference you work around by changing a formula: the contract itself could not
describe the NVIDIA layout. That is why "reimplementing the header" was not enough.

A second detail, less conceptual but equally costly: the llama.cpp tile abstraction we
reused computes indices by reading `threadIdx.x` **assuming it equals the warp lane**. This is
true in the forward kernels, false in the backward, which launches flat blocks of 128 or 256
threads. Using it directly sends warps 1–3 out of memory bounds. A 32-thread test does not
reveal it: at least 4 warps are needed.

## B. The decision: widen the seam, do not fork the kernels

Two roads:

| | duplicate the kernels | widen the abstraction |
|---|---|---|
| diff | ~800 lines of near-copied CUDA across 13 kernels | +438/−301, shorter kernels |
| AMD risk | none (untouched) | the AMD code changes shape, though not substance |
| maintainability | every future change must be done twice | only once |
| upstream acceptability | low | high |

We chose the second, mitigating the AMD risk specifically: **the AMD branch of every new
function is, line for line, the loop that used to be inline in the kernel**, including a tuning
parameter (`VECTOR_LOAD`) that already existed and that we propagated instead of normalizing.

Concretely, the kernels stopped indexing fragments by hand:

| before | after |
|---|---|
| `bf16_fragment` for both operands | `bf16_fragment_a` / `bf16_fragment_b` |
| `fragment_data(f)` + a hand-written fill loop at each site | `load_a_fragment<CHECK_M, CHECK_K>(...)`, `load_b_fragment(...)` |
| `acc.values[e]` | `acc.value(e)` |
| `c_column(lane,e)` for the row, `c_row(lane)` for the column | `acc_m(lane,e)` / `acc_n(lane,e)` |

The last line is the heart of it: by making *both* coordinates element-dependent, the contract
becomes able to describe both hardwares. On AMD `acc_n` simply ignores the argument.

Total surface touched: ~40 mechanical sites (8 fills of operand A, 20 of operand B, 6
stores, 12 accumulator declarations), not 2900 lines to rewrite. Having measured this *before*
starting changed the estimate from "days" to "hours".

## C. Correctness: two traps that produce convincing, false numbers

**1. A test that failed because of its own oracle.**
An upstream test compared our result to a PyTorch bf16 matrix product,
demanding **bit-for-bit** equality. It failed on 117,611 elements out of 320,000. The obvious reading — "our
kernel is wrong" — was wrong.

PyTorch has `allow_bf16_reduced_precision_reduction = True` as the default: it lets the backend
sum partial results **in bf16**. Evaluating both against an fp64 sum of the same
products:

| | outputs that are the correctly-rounded value |
|---|---|
| the test's reference (default) | 202,389 / 320,000 — **63.2%** |
| the same reference, flag disabled | 319,995 / 320,000 — 99.998% |
| our kernel | 319,993 / 320,000 — **99.998%** |

On the 117,611 disagreeing elements, **the kernel was right 117,605 times, the reference 1**. The
fingerprint is unambiguous: the reference is 100% correct on groups of 1, 15 and 16 rows and drops to
~60% from 17 up — exactly where cuBLAS changes tiling strategy.

The correct fix was not to loosen the tolerance, it was to **repair the oracle**. Once done, the residue
is 4 elements out of 320,000, each exactly 1 bf16 ULP: irreducible, because two equally-valid fp32
summation orders can fall on opposite sides of a rounding boundary.

**2. A suite that did not test the production kernels.**
Every test used `out_features=37`. The optimized kernels are selected by an exact gate
(`out_features==2048 && in_features==512`, the *down* projection of a MoE FFN) plus thresholds on the
number of rows. With 37 they are **never reached by construction**. In practice: the kernels the real
model runs were covered by nothing.

We added 9 cases (3 quantization types × 3 row regimes) and verified with
`torch.profiler` that they cover **8 distinct previously-unreached kernels**.

## D. Measuring without fooling yourself

The performance conclusion was wrong **twice** before it was right. It is worth
recording how, because these are generic errors, not specific to this project.

**Error 1 — the wrong shape.** Measuring all quantization types at a fixed `out_features=512`,
one kernel came out at 0.22 of the ceiling, and seemed to be the one used by the gate/up
projections. Doubly false: gate/up goes through a *paired* operation (gate and up share the
same input, so a kernel exists that does them together), and 512 is not the shape of the down projection.
We were precisely timing a path the training never runs.

**Error 2 — the wrong ceiling, and cross-run ratios.** Using the *forward* as the reference
point, the backward came out at 0.75–0.95 and "no headroom left". Two independent errors:

- the two numbers came from **different runs**, and this laptop drifts 5–15% from one run to the
  next (`nvidia-smi -q -d PERFORMANCE` shows `SW Power Capping` accumulating);
- above all: **the forward is not a fair comparison**. It quantizes activations to 8 bits and runs on
  **int8** tensor cores, which on Ada have about twice the bf16 throughput. Comparing a
  bf16 kernel to an int8 one makes it look better than it is.

**The correct ceiling** is the same matrix product in bf16 with the weights **already dequantized**
(cuBLAS doing exactly the arithmetic we do, and zero GGUF decode), timed
**in the same run**:

| real training path | ours | bf16 GEMM without dequant | ratio |
|---|---|---|---|
| gate/up (paired), Q3_K | 19.6 TFLOP/s | 31.5 | 0.62 |
| gate/up (paired), IQ2_S | 20.8 | 26.7 | 0.78 |
| down, Q4_K | 22.4 | 37.3 | 0.60 |
| down, Q5_K | 25.5 | 40.5 | 0.63 |

Rules we draw from it, valid well beyond this case:

1. **measure the shapes the system actually runs**, not a convenient grid;
2. **take the ratios within the same run** on hardware that drifts;
3. **choose an oracle that does the same work**: a comparison against an implementation that skips
   an expensive stage, or uses cheaper arithmetic, is not a ceiling;
4. **a numerical tolerance must be calibrated to the real scale**: an absolute threshold calibrated on
   short reductions becomes meaningless when the reduction is 55 times longer;
5. **per-element relative metrics explode on cancellations**: on signed sums, a
   fraction of the outputs is near zero and there the relative error is unbounded for *any*
   summation order. At an intermediate step this made us measure "2.7 million ULP of error" on a
   kernel that was in fact bit-exact.

## E. Status and what remains

- PR opened on `woct0rdho/torch-ggml-ops` (10 files, +660/−322, 4 thematic commits).
- To clarify with the maintainer: verification of the AMD branch; the two test changes (declared and
  justified with numbers in the PR text); the sm_80 floor, for which we offer a Turing fallback as an
  alternative.
- Deliberately not done: the single-projection kernel at the gate/up shape sits at ~0.19 of the ceiling,
  but the training does not go through it. Fixing it is a restructuring with effects on the AMD branch, which
  we cannot validate — so it is flagged to the maintainer, not implemented.
- Next step for the project: measure the VRAM footprint of the 35B model in forward, which is the
  constraint that decides whether the whole plan fits in 16 GB.

---

*Reusable tools produced: a benchmark that compares two builds and includes the fair ceiling
(`scripts/bench_backward.py`), a correctness verifier at production shapes that proves which
kernel runs (`scripts/verify_backward.py`), and standalone microtests of the fragment layouts
(`docs/port-microtests/`).*
