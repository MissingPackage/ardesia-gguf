#!/usr/bin/env bash
# The pip CUDA-13 wheels ship only versioned shared objects (libcudart.so.13),
# but torch's CUDAExtension emits plain -lcudart / -lcublas etc., which need an
# unversioned dev symlink (libcudart.so). Create them idempotently in the wheel
# lib dir. Re-run after any `uv pip install` that reinstalls the nvidia-* wheels.
set -eu
_repo="${ARDESIA_GGUF_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LIBDIR="$_repo/.venv-pathb/lib/python3.12/site-packages/nvidia/cu13/lib"
cd "$LIBDIR"
n=0
for so in *.so.*; do
  # strip trailing .<ver...> to get libFoo.so
  base="${so%%.so.*}.so"
  if [ ! -e "$base" ]; then ln -s "$so" "$base"; echo "  linked $base -> $so"; n=$((n+1)); fi
done
echo "[pathb-link-cudalibs] created $n symlink(s) in $LIBDIR"
