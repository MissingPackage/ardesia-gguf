#!/usr/bin/env python
"""Root-cause probe for test_grouped_backward_route_group_boundaries.

The test asserts our grad_input equals a torch bf16 matmul BIT-EXACTLY (rtol=0,
atol=0) and it does not. That alone says nothing about who is wrong: both are
fp32 accumulations of the same 37 bf16 products in some order.

So this rebuilds the test's exact inputs and scores BOTH candidates against an
fp64 ground truth of the same products. Whoever is closer to fp64 is the more
accurate answer, independent of anybody's reduction order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor" / "torch-ggml-ops"))

import gguf  # noqa: E402
import torch_ggml_ops  # noqa: E402,F401
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor  # noqa: E402

MODEL = REPO / "models" / "Qwen3.6-35B-A3B-APEX-I-Mini.gguf"
GROUP_SIZES = (1, 15, 16, 17, 63, 64, 65, 127, 128, 129)
EXPERT_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11]


def main() -> None:
    reader = gguf.GGUFReader(MODEL)
    tensor = next(t for t in reader.tensors if t.name == "blk.2.ffn_down_exps.weight")
    host = np.array(tensor.data[:12, 0:37], dtype=np.uint8, copy=True, order="C")
    packed = torch.from_numpy(host).to("cuda")
    quant_type = tensor.tensor_type
    in_features = int(tensor.shape[0])

    group_sizes = torch.tensor(GROUP_SIZES, device="cuda", dtype=torch.int32)
    offsets = group_sizes.cumsum(0).to(torch.int32).contiguous()
    experts = torch.tensor(EXPERT_IDS, device="cuda", dtype=torch.int64)
    generator = torch.Generator(device="cuda").manual_seed(8642)
    grad_output = torch.randn(
        sum(GROUP_SIZES), 37, generator=generator, device="cuda", dtype=torch.bfloat16
    )

    ours = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default(
        grad_output, packed, experts, offsets, int(quant_type), in_features
    )

    logical = dequantize_gguf_tensor(
        packed.index_select(0, experts), quant_type, dtype=torch.bfloat16, device="cuda"
    ).reshape(experts.numel(), 37, in_features)

    # What the test compares against: a per-group bf16 matmul.
    reference = torch.empty_like(ours)
    begin = 0
    for group, end in enumerate(offsets.cpu().tolist()):
        reference[begin:end] = grad_output[begin:end] @ logical[group]
        begin = end

    # Ground truth: the SAME bf16 products, accumulated in fp64 on the CPU. 37
    # terms in fp64 is exact for practical purposes, and it is reduction-order
    # independent at this magnitude.
    truth = torch.empty(ours.shape, dtype=torch.float64)
    go64 = grad_output.double().cpu()
    lg64 = logical.double().cpu()
    begin = 0
    for group, end in enumerate(offsets.cpu().tolist()):
        truth[begin:end] = go64[begin:end] @ lg64[group]
        begin = end

    truth_bf16 = truth.to(torch.bfloat16)  # the correctly-rounded answer

    def score(name: str, candidate: torch.Tensor) -> None:
        c = candidate.double().cpu()
        err = (c - truth).abs()
        exact = (candidate.cpu() == truth_bf16).sum().item()
        total = err.numel()
        rms = truth.square().mean().sqrt()
        print(f"  {name:<28} exact vs correctly-rounded fp64: "
              f"{exact}/{total} ({100 * exact / total:.3f}%)   "
              f"max|err|/rms {(err.max() / rms).item():.3e}")

    print(f"shape: {tuple(ours.shape)}  K(out_features)=37  groups={len(GROUP_SIZES)}")
    print(f"disagreement between ours and the test's reference: "
          f"{(ours != reference).sum().item()}/{ours.numel()}\n")
    print("scored against an fp64 accumulation of the same bf16 products:")
    score("ours (tensor core)", ours)
    score("test reference (torch bf16)", reference)

    # Hypothesis for the reference's poor showing: torch defaults to
    # allow_bf16_reduced_precision_reduction=True, i.e. cuBLAS may reduce split-K
    # partial sums in bf16 instead of fp32. Re-run the same matmul with it off.
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    strict_reference = torch.empty_like(ours)
    begin = 0
    for group, end in enumerate(offsets.cpu().tolist()):
        strict_reference[begin:end] = grad_output[begin:end] @ logical[group]
        begin = end
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
    score("reference, reduction off", strict_reference)
    print(f"  -> strict reference vs ours: "
          f"{(ours != strict_reference).sum().item()}/{ours.numel()} differ "
          f"({'BIT-IDENTICAL' if torch.equal(ours, strict_reference) else 'still differs'})")

    # With a correct reference, what tolerance does the irreducible residue need?
    # Two valid fp32 orderings of the same 37 products can straddle a bf16
    # rounding boundary, so atol=0 is unachievable against ANY independent
    # reference -- it only ever held because the gfx11 WMMA happened to match
    # cuBLAS's order bit-for-bit.
    diff_mask = ours != strict_reference
    if diff_mask.any():
        a = ours[diff_mask].float()
        b = strict_reference[diff_mask].float()
        gap = (a - b).abs()
        print("\n  irreducible residue vs the strict reference:")
        for x, y, d in zip(a.tolist(), b.tolist(), gap.tolist()):
            ulp = 2.0 ** (torch.tensor(abs(y)).log2().floor().item() - 7) if y else 0.0
            print(f"    ours {x:+.6g}  ref {y:+.6g}  gap {d:.3g}"
                  + (f" = {d / ulp:.2f} ULP" if ulp else ""))
        need_rtol = (gap / b.abs().clamp_min(1e-30)).max().item()
        print(f"  minimum rtol that would pass (atol=0): {need_rtol:.5g}"
              f"  (one bf16 ULP is 2^-7 = {2.0**-7:.5g})")

    ours = ours.cpu()
    reference = reference.cpu()
    # Where they disagree, who is right?
    differ = (ours != reference)
    ours_right = ((ours == truth_bf16) & differ).sum().item()
    ref_right = ((reference == truth_bf16) & differ).sum().item()
    neither = (differ & (ours != truth_bf16) & (reference != truth_bf16)).sum().item()
    print(f"\non the {differ.sum().item()} disagreeing elements: "
          f"ours correct {ours_right}, reference correct {ref_right}, neither {neither}")

    # Per-group, to see whether the reference's accuracy depends on group size.
    print("\nper group (rows -> exact-vs-fp64 count, ours / reference):")
    begin = 0
    for size, end in zip(GROUP_SIZES, offsets.cpu().tolist()):
        o = (ours[begin:end].cpu() == truth_bf16[begin:end]).sum().item()
        r = (reference[begin:end].cpu() == truth_bf16[begin:end]).sum().item()
        n = truth_bf16[begin:end].numel()
        print(f"  rows={size:<4} {o:>6}/{n:<6} {100*o/n:6.2f}%   |   "
              f"{r:>6}/{n:<6} {100*r/n:6.2f}%")
        begin = end


if __name__ == "__main__":
    main()
