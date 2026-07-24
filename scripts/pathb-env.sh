#!/usr/bin/env bash
# Path B build/run environment (uv-native, no conda). Source this before building
# torch-ggml-ops or running its tests.
#
#   source scripts/pathb-env.sh
#
# Toolchain: torch 2.13+cu130 in .venv-pathb; nvcc/CUDA-13.3 dev from pip wheels
# (unified prefix nvidia/cu13); host compiler gcc15 (CUDA 13 rejects the box gcc16).

set -u
# Sourced under bash or zsh; BASH_SOURCE isn't portable, so allow an override via
# ARDESIA_GGUF_ROOT and default to the current directory (source this from the repo root).
_repo="${ARDESIA_GGUF_ROOT:-$PWD}"
_venv="$_repo/.venv-pathb"
_sp="$_venv/lib/python3.12/site-packages"

export VIRTUAL_ENV="$_venv"
export CUDA_HOME="$_sp/nvidia/cu13"
export CUDAHOSTCXX="/usr/bin/g++-15"
export CUDACXX="$CUDA_HOME/bin/nvcc"
# our 4090 Laptop is Ada / sm_89
export TORCH_CUDA_ARCH_LIST="8.9"
export PATH="$CUDA_HOME/bin:$_venv/bin:$PATH"
# nvcc-compiled .so links libcudart from the wheel; also expose torch's bundled libs
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$_sp/torch/lib:${LD_LIBRARY_PATH:-}"
# Env-driven nvcc flags so upstream setup.py stays clean (these are OUR pip-wheel
# env workarounds, deliberately NOT in the port/PR — see docs/04):
#  -ccbin g++-15: CUDA 13 rejects the box gcc 16 (nvcc reads the last -ccbin).
#  -DCCCL_...   : nvcc 13.3 wheel vs torch's pinned cudart 13.0 headers (CCCL guard).
export NVCC_APPEND_FLAGS="-ccbin $CUDAHOSTCXX -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
set +u

echo "[pathb-env] CUDA_HOME=$CUDA_HOME"
echo "[pathb-env] host cxx=$CUDAHOSTCXX  arch=$TORCH_CUDA_ARCH_LIST"
