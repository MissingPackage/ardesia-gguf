# Vendored third-party dependencies

This repository does **not** ship the third-party working copies it builds against (they live under
`vendor/` locally and are git-ignored). They are all from **woct0rdho** and are pinned to the exact
commits below. Reproduce the working tree by cloning each repo at its commit and applying the patches
in `patches/`.

Each upstream keeps its own license. In particular the CUDA port of `torch-ggml-ops` is a change **to**
that project (Apache-2.0) and is not redistributed here — only our patch against it is. See the root
`LICENSE` for the scope of this repository's own MIT license.

## Pinned repositories

| local path (`vendor/`) | upstream | branch | commit |
|---|---|---|---|
| `torch-ggml-ops` | https://github.com/woct0rdho/torch-ggml-ops | `cuda-support` (fork) | `c6f4467` |
| `transformers-gguf` | https://github.com/woct0rdho/transformers | `gguf` | `5ef64c1` |
| `transformers-qwen3-moe-fused` | https://github.com/woct0rdho/transformers-qwen3-moe-fused | default | `a087104` |
| `transformers5-qwen3.5-recipe` | https://github.com/woct0rdho/transformers5-qwen3.5-recipe | default | `5dd93c5` |

## Patches in this repo (`patches/`)

- **`torch-ggml-ops-cuda-port.patch`** — the NVIDIA/CUDA port of the backward kernels (dual-path with
  ROCm/HIP; 10 files). Its full write-up is `docs/05-porting-kernel-cuda.md` and `docs/04-spike2-pathb-port.md`.
  The PR body submitted upstream is `patches/torch-ggml-ops-pr-body.md`. Commit `c6f4467` on the
  `cuda-support` branch is this patch applied on top of upstream master.
- **`moe-fused-a087104-ardesia.patch`** — adds `IQ2_XS` dequant support and a chunked eager-dequant
  fallback to `transformers-qwen3-moe-fused` (see `docs/03-upstream-woct0rdho.md`). Note: the `IQ2_XS`
  half was merged upstream as PR #23 (commit `a4f3c52`), so a fresh clone from master already includes
  it; only the chunked wrap remains local.

## How to fetch and reconstruct

```bash
mkdir -p vendor && cd vendor

# torch-ggml-ops (Path B kernels). To reproduce the CUDA-support state, clone upstream at the
# commit the port was cut against and apply the port patch. Commit c6f4467 is the applied result.
git clone https://github.com/woct0rdho/torch-ggml-ops.git
git -C torch-ggml-ops apply ../patches/torch-ggml-ops-cuda-port.patch

# transformers fork with GGUF support (used by the transformers5 recipe)
git clone -b gguf https://github.com/woct0rdho/transformers.git transformers-gguf
git -C transformers-gguf checkout 5ef64c1

# transformers-qwen3-moe-fused (Path A). Pin, then apply the local patch.
git clone https://github.com/woct0rdho/transformers-qwen3-moe-fused.git
git -C transformers-qwen3-moe-fused checkout a087104
git -C transformers-qwen3-moe-fused apply ../patches/moe-fused-a087104-ardesia.patch

# the transformers5 recipe (Path B training recipe)
git clone https://github.com/woct0rdho/transformers5-qwen3.5-recipe.git
git -C transformers5-qwen3.5-recipe checkout 5dd93c5
```

After cloning, follow `scripts/pathb-env.sh` + `scripts/pathb-link-cudalibs.sh` to set up the Path B
build (CUDA-13 pip wheels, gcc-15 host compiler), then build `torch-ggml-ops`
(`python setup.py build_ext --inplace`). Details in `docs/04-spike2-pathb-port.md`.
