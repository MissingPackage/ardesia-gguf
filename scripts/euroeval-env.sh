#!/usr/bin/env bash
# EuroEval (vLLM + flashinfer JIT) needs nvcc AND link-time/run-time CUDA libs. We have no system
# CUDA toolkit: both come from the pip wheels in .venv-pathb (see memory pathb-build-toolchain).
# Note the TWO export statements: PATH must be expanded AFTER CUDA_HOME is set (a single
# `export A=x B=$A` expands $A to the OLD value — cost one run on 2026-09-12).
#   source scripts/euroeval-env.sh
export CUDA_HOME=/home/neuromancer/Projects/ardesia-gguf/.venv-pathb/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CUDAHOSTCXX=/usr/bin/g++-15
export NVCC_APPEND_FLAGS="-ccbin /usr/bin/g++-15 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export TORCH_CUDA_ARCH_LIST=8.9
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export EUROEVAL=/home/neuromancer/Projects/ardesia-gguf/.venv-euroeval/bin/euroeval
