#!/usr/bin/env python
"""Correctness + kernel-coverage check for the grouped backward at PRODUCTION shapes.

The upstream unit tests mostly use out_features=37, which routes to the generic
grouped kernels. The 35B-A3B experts use out_features 512/2048, which routes to
the *tiled* kernels -- a code path the tests barely touch. This script closes
that gap: for each real expert shape it

  1. names the CUDA kernel that actually ran (torch.profiler), so a claim about
     "the tiled kernel is ported" is evidence rather than inference, and
  2. compares grad_input against a dequantize + per-group matmul reference.

    source scripts/pathb-env.sh
    .venv-pathb/bin/python scripts/verify_backward.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor" / "torch-ggml-ops"))

import gguf  # noqa: E402
import torch_ggml_ops  # noqa: E402,F401
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor  # noqa: E402

DEFAULT_MODEL = REPO / "models" / "Qwen3.6-35B-A3B-APEX-I-Mini.gguf"

# (label, tensor, out_features, rows) -- the shapes the MoE experts really hit.
CASES = [
    ("gate/up Q3_K", "blk.0.ffn_gate_exps.weight", 512, 4096),
    ("gate/up IQ2_S", "blk.10.ffn_gate_exps.weight", 512, 4096),
    ("down Q4_K", "blk.2.ffn_down_exps.weight", 2048, 4096),
    ("down Q5_K", "blk.0.ffn_down_exps.weight", 2048, 4096),
    ("gate/up Q3_K small batch", "blk.0.ffn_gate_exps.weight", 512, 256),
]


def packed_experts(reader, name, *, num_experts, out_features):
    tensor = next(t for t in reader.tensors if t.name == name)
    rows = slice(0, out_features)
    if tensor.data.ndim == 3:
        host = np.array(tensor.data[:num_experts, rows], dtype=np.uint8, copy=True, order="C")
    else:
        one = np.array(tensor.data[rows], dtype=np.uint8, copy=True, order="C")
        host = np.repeat(one[None, ...], num_experts, axis=0)
    return torch.from_numpy(host).to("cuda"), tensor.tensor_type, int(tensor.shape[0])


def reference_grad_input(grad_output, packed, experts, offsets, quant_type, out_features, in_features):
    """Reference with the kernel's EXACT inputs, differing only in reduction order.

    Two details matter or the reference is not comparable: the kernel stages the
    dequantized weight as bf16 in shared memory (so the reference must round the
    weight to bf16 too), and torch's fp32 matmul defaults to TF32 tensor cores
    (10-bit mantissa), which would inject more error than what is being measured.
    """
    logical = dequantize_gguf_tensor(
        packed.index_select(0, experts), quant_type, dtype=torch.bfloat16, device="cuda"
    ).reshape(experts.numel(), out_features, in_features)
    out = torch.empty(grad_output.shape[0], in_features, device="cuda", dtype=torch.float32)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        begin = 0
        for group, end in enumerate(offsets.cpu().tolist()):
            out[begin:end] = grad_output[begin:end].float() @ logical[group].float()
            begin = end
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return out


def compare(actual_bf16: torch.Tensor, expected_f32: torch.Tensor) -> dict:
    """Compare against the fp32 reference.

    Per-element ULP is NOT usable here: summing 512-2048 signed products makes a
    fair fraction of the outputs near-zero cancellations, where the relative
    error of ANY reordering is unbounded. What is meaningful is the error against
    the output's own scale, plus how far the result is from the correctly-rounded
    bf16 answer.
    """
    error = (actual_bf16.float() - expected_f32).abs()
    rms = expected_f32.square().mean().sqrt()
    rounded = expected_f32.to(torch.bfloat16).float()
    off_by_more_than_one_step = (
        (actual_bf16.float() - rounded).abs() > (rounded.abs() * 2.0**-7 + 1e-30)
    ).sum().item()
    return {
        "nrmse": (error.square().mean().sqrt() / rms).item(),
        "max_rel_to_rms": (error.max() / rms).item(),
        "exact_vs_rounded": (actual_bf16.float() == rounded).sum().item(),
        "past_one_step": off_by_more_than_one_step,
        "count": error.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--groups", type=int, default=8)
    args = parser.parse_args()

    reader = gguf.GGUFReader(args.model)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    failures = 0

    for label, tensor_name, out_features, rows in CASES:
        packed, quant_type, in_features = packed_experts(
            reader, tensor_name, num_experts=args.groups, out_features=out_features
        )
        experts = torch.arange(args.groups, device="cuda", dtype=torch.int64)
        bounds = [round((g + 1) * rows / args.groups) for g in range(args.groups)]
        offsets = torch.tensor(bounds, device="cuda", dtype=torch.int32)
        grad_output = torch.randn(
            rows, out_features, generator=gen, device="cuda", dtype=torch.bfloat16
        )

        op = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default
        actual = op(grad_output, packed, experts, offsets, int(quant_type), in_features)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            op(grad_output, packed, experts, offsets, int(quant_type), in_features)
            torch.cuda.synchronize()
        kernels = [
            e.key for e in prof.key_averages()
            if "grad_input" in e.key or "row_task" in e.key
        ]

        expected = reference_grad_input(
            grad_output, packed, experts, offsets, quant_type, out_features, in_features
        )
        stats = compare(actual, expected)
        # bf16 storage alone costs ~2.2e-3 nrmse, so 5e-3 leaves room for a
        # different-but-valid accumulation order and nothing more. The max bound
        # is against the output RMS, not per-element, for the cancellation reason.
        ok = stats["nrmse"] < 5e-3 and stats["max_rel_to_rms"] < 0.05
        failures += not ok

        print(f"{'PASS' if ok else 'FAIL'}  {label:<28} "
              f"[{rows}x{out_features}x{in_features}, {args.groups} groups]")
        print(f"        nrmse {stats['nrmse']:.2e}  "
              f"max|err|/rms {stats['max_rel_to_rms']:.2e}  "
              f"bit-exact vs rounded fp32: "
              f"{100 * stats['exact_vs_rounded'] / stats['count']:.2f}%  "
              f"past 1 step: {stats['past_one_step']}")
        for name in sorted(set(kernels)):
            print(f"        kernel: {name[:110]}")

        del packed, grad_output, actual, expected
        torch.cuda.empty_cache()

    print(f"\n{len(CASES) - failures}/{len(CASES)} production-shape cases pass")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
