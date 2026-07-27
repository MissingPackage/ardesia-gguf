#!/usr/bin/env python
"""Benchmark the torch-ggml-ops backward (grad_input) kernels on NVIDIA.

Measures the dense + grouped backward ops on real GGUF tensors from the target
model, so the emulation-vs-tensor-core port can be compared apples to apples.

    source scripts/pathb-env.sh
    .venv-pathb/bin/python scripts/bench_backward.py --tag emulation
    # ... port, rebuild ...
    .venv-pathb/bin/python scripts/bench_backward.py --tag tensorcore
    .venv-pathb/bin/python scripts/bench_backward.py --compare emulation tensorcore

Results land in outputs/bench-<tag>.json. Timing is CUDA-event based, median of
`--rounds` rounds of `--iters` iterations, after `--warmup` warmup iterations;
each round re-checks the clock so thermal drift is visible in the spread.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "vendor" / "torch-ggml-ops"))

import gguf  # noqa: E402
import torch_ggml_ops  # noqa: E402,F401  (registers torch.ops.torch_ggml_ops)
from transformers.integrations.gguf_dequant import dequantize_gguf_tensor  # noqa: E402

DEFAULT_MODEL = REPO / "models" / "Qwen3.6-35B-A3B-APEX-I-Mini.gguf"

# One representative tensor per quant type present in the target model. The
# grouped path is what the 35B-A3B MoE experts actually use, so it carries the
# most weight; the dense path is kept for regression coverage.
PROJECTIONS = {
    "Q3_K": "blk.0.ffn_gate_exps.weight",
    "Q4_K": "blk.2.ffn_down_exps.weight",
    "Q5_K": "blk.0.ffn_down_exps.weight",
    "Q6_K": "output.weight",
    "IQ2_S": "blk.10.ffn_gate_exps.weight",
}
PAIR_PROJECTIONS = {
    "Q3_K": ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"),
    "IQ2_S": ("blk.10.ffn_gate_exps.weight", "blk.10.ffn_up_exps.weight"),
}


def packed_experts(
    reader: gguf.GGUFReader,
    name: str,
    *,
    num_experts: int,
    out_features: int,
    row_offset: int = 0,
) -> tuple[torch.Tensor, int, int]:
    """Slice `out_features` rows of `num_experts` experts onto the GPU."""
    tensor = next(t for t in reader.tensors if t.name == name)
    rows = slice(row_offset, row_offset + out_features)
    if tensor.data.ndim == 3:
        host = np.array(tensor.data[:num_experts, rows], dtype=np.uint8, copy=True, order="C")
    else:
        one = np.array(tensor.data[rows], dtype=np.uint8, copy=True, order="C")
        host = np.repeat(one[None, ...], num_experts, axis=0)
    return torch.from_numpy(host).to("cuda"), int(tensor.tensor_type), int(tensor.shape[0])


def group_metadata(rows: int, groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Evenly split `rows` across `groups` experts (expert-sorted layout)."""
    experts = torch.arange(groups, device="cuda", dtype=torch.int64)
    bounds = [round((g + 1) * rows / groups) for g in range(groups)]
    return experts, torch.tensor(bounds, device="cuda", dtype=torch.int32)


def time_op(fn, *, warmup: int, iters: int, rounds: int) -> dict:
    # Warm up by wall time, not by count: the 4090 Laptop ramps its clocks over
    # the first few hundred ms, which otherwise shows up as a 2-3x slower first
    # round and poisons the spread.
    import time

    deadline = time.perf_counter() + 0.5
    count = 0
    while count < warmup or time.perf_counter() < deadline:
        fn()
        count += 1
        if count % 20 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(rounds):
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop) / iters)
    return {
        "ms_median": statistics.median(samples),
        "ms_min": min(samples),
        "ms_max": max(samples),
        "rounds_ms": samples,
    }


def bench(args) -> dict:
    reader = gguf.GGUFReader(args.model)
    results: dict[str, dict] = {}
    gen = torch.Generator(device="cuda").manual_seed(1234)

    for qname, tensor_name in PROJECTIONS.items():
        packed, quant_type, in_features = packed_experts(
            reader, tensor_name, num_experts=args.groups, out_features=args.out_features
        )
        experts, offsets = group_metadata(args.rows, args.groups)
        grad_output = torch.randn(
            args.rows, args.out_features, generator=gen, device="cuda", dtype=torch.bfloat16
        )

        grouped = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default
        stats = time_op(
            lambda: grouped(
                grad_output, packed, experts, offsets, quant_type, in_features
            ),
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
        )
        # grad_input = grad_output @ W : 2 * rows * out_features * in_features FLOP
        flop = 2.0 * args.rows * args.out_features * in_features
        stats["tflops"] = flop / (stats["ms_median"] * 1e-3) / 1e12
        stats["shape"] = {
            "rows": args.rows,
            "out_features": args.out_features,
            "in_features": in_features,
            "groups": args.groups,
        }
        results[f"grouped_grad_input/{qname}"] = stats

        # The forward over the same weights and FLOP count. Kept for context, but
        # do NOT read it as a ceiling for the backward: `grouped_mmq` quantizes
        # activations to Q8_1 and runs on int8 tensor cores, ~2x the bf16 rate on
        # Ada. The fair bf16 ceiling is in bench_moe() -- see the note there.
        forward_input = torch.randn(
            args.rows, in_features, generator=gen, device="cuda", dtype=torch.bfloat16
        )
        forward = torch.ops.torch_ggml_ops.grouped_mmq.default
        stats = time_op(
            lambda: forward(
                forward_input, packed, experts, offsets, quant_type, args.out_features
            ),
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
        )
        flop = 2.0 * args.rows * args.out_features * in_features
        stats["tflops"] = flop / (stats["ms_median"] * 1e-3) / 1e12
        stats["shape"] = {
            "rows": args.rows,
            "out_features": args.out_features,
            "in_features": in_features,
            "groups": args.groups,
        }
        results[f"grouped_forward_int8_context/{qname}"] = stats
        del forward_input

        dense = torch.ops.torch_ggml_ops.mmq_grad_input.default
        one_expert = packed[0]
        stats = time_op(
            lambda: dense(grad_output, one_expert, quant_type, in_features),
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
        )
        stats["tflops"] = flop / (stats["ms_median"] * 1e-3) / 1e12
        stats["shape"] = {
            "rows": args.rows,
            "out_features": args.out_features,
            "in_features": in_features,
        }
        results[f"dense_grad_input/{qname}"] = stats

        del packed, one_expert, grad_output
        torch.cuda.empty_cache()

    for qname, (first_name, second_name) in PAIR_PROJECTIONS.items():
        first, quant_type, in_features = packed_experts(
            reader, first_name, num_experts=args.groups, out_features=args.out_features
        )
        second, _, _ = packed_experts(
            reader, second_name, num_experts=args.groups, out_features=args.out_features
        )
        experts, offsets = group_metadata(args.rows, args.groups)
        first_grad = torch.randn(
            args.rows, args.out_features, generator=gen, device="cuda", dtype=torch.bfloat16
        )
        second_grad = torch.randn(
            args.rows, args.out_features, generator=gen, device="cuda", dtype=torch.bfloat16
        )
        pair = torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default
        stats = time_op(
            lambda: pair(
                first_grad, second_grad, first, second, experts, offsets, quant_type, in_features
            ),
            warmup=args.warmup,
            iters=args.iters,
            rounds=args.rounds,
        )
        flop = 2.0 * 2 * args.rows * args.out_features * in_features
        stats["tflops"] = flop / (stats["ms_median"] * 1e-3) / 1e12
        stats["shape"] = {
            "rows": args.rows,
            "out_features": args.out_features,
            "in_features": in_features,
            "groups": args.groups,
        }
        results[f"grouped_pair_grad_input/{qname}"] = stats

        del first, second, first_grad, second_grad
        torch.cuda.empty_cache()

    return results


def bench_moe(args) -> dict:
    """The two shapes a real Qwen3.6-35B-A3B FFN backward actually executes.

    Not a synthetic sweep: `fast_moe_lora.py` routes gate/up through
    grouped_mmq_pair (they share one input) and down through the single-projection
    op, so those are the only two backward calls training makes per expert layer.
    """
    reader = gguf.GGUFReader(args.model)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    results: dict[str, dict] = {}
    experts, offsets = group_metadata(args.rows, args.groups)

    for qname, (gate_name, up_name) in [
        ("Q3_K", ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight")),
        ("IQ2_S", ("blk.10.ffn_gate_exps.weight", "blk.10.ffn_up_exps.weight")),
    ]:
        gate, quant_type, hidden = packed_experts(
            reader, gate_name, num_experts=args.groups, out_features=512
        )
        up, _, _ = packed_experts(reader, up_name, num_experts=args.groups, out_features=512)
        gate_grad = torch.randn(args.rows, 512, generator=gen, device="cuda", dtype=torch.bfloat16)
        up_grad = torch.randn(args.rows, 512, generator=gen, device="cuda", dtype=torch.bfloat16)
        pair = torch.ops.torch_ggml_ops.grouped_mmq_pair_grad_input.default
        stats = time_op(
            lambda: pair(gate_grad, up_grad, gate, up, experts, offsets, quant_type, hidden),
            warmup=args.warmup, iters=args.iters, rounds=args.rounds,
        )
        flop = 2.0 * 2 * args.rows * 512 * hidden
        stats["tflops"] = flop / (stats["ms_median"] * 1e-3) / 1e12
        stats["shape"] = {"rows": args.rows, "out_features": 512, "in_features": hidden}
        results[f"moe_gate_up_pair/{qname}"] = stats

        # Ceiling: the SAME grouped matmul in bf16 with the weights ALREADY
        # dequantized, i.e. cuBLAS doing only the arithmetic our kernel also has
        # to do, and none of the GGUF decode. Timed in the SAME run, because this
        # laptop drifts 5-15% between runs (SW power capping) and a cross-run
        # ratio would mostly measure that.
        #
        # NOT the forward: `grouped_mmq` quantizes activations to Q8_1 and runs on
        # int8 tensor cores, which on Ada are ~2x the bf16 rate. Using the forward
        # as the ceiling flatters the bf16 backward and is how an earlier pass of
        # this benchmark concluded there was no headroom left.
        gate_w = dequantize_gguf_tensor(
            gate.index_select(0, experts), gguf.GGMLQuantizationType(quant_type),
            dtype=torch.bfloat16, device="cuda",
        ).reshape(args.groups, 512, hidden)
        up_w = dequantize_gguf_tensor(
            up.index_select(0, experts), gguf.GGMLQuantizationType(quant_type),
            dtype=torch.bfloat16, device="cuda",
        ).reshape(args.groups, 512, hidden)
        gate_v = gate_grad.view(args.groups, args.rows // args.groups, 512)
        up_v = up_grad.view(args.groups, args.rows // args.groups, 512)
        stats = time_op(
            lambda: torch.baddbmm(torch.bmm(gate_v, gate_w), up_v, up_w),
            warmup=args.warmup, iters=args.iters, rounds=args.rounds,
        )
        stats["tflops"] = flop / (stats["ms_median"] * 1e-3) / 1e12
        stats["shape"] = {"rows": args.rows, "out_features": 512, "in_features": hidden}
        results[f"moe_gate_up_pair_CEILING_bf16gemm/{qname}"] = stats

        del gate, up, gate_grad, up_grad, gate_w, up_w
        torch.cuda.empty_cache()

    for qname, down_name in [
        ("Q4_K", "blk.2.ffn_down_exps.weight"),
        ("Q5_K", "blk.0.ffn_down_exps.weight"),
    ]:
        down, quant_type, intermediate = packed_experts(
            reader, down_name, num_experts=args.groups, out_features=2048
        )
        down_grad = torch.randn(
            args.rows, 2048, generator=gen, device="cuda", dtype=torch.bfloat16
        )
        single = torch.ops.torch_ggml_ops.grouped_mmq_grad_input.default
        stats = time_op(
            lambda: single(down_grad, down, experts, offsets, quant_type, intermediate),
            warmup=args.warmup, iters=args.iters, rounds=args.rounds,
        )
        flop = 2.0 * args.rows * 2048 * intermediate
        stats["tflops"] = flop / (stats["ms_median"] * 1e-3) / 1e12
        stats["shape"] = {"rows": args.rows, "out_features": 2048, "in_features": intermediate}
        results[f"moe_down_single/{qname}"] = stats

        down_w = dequantize_gguf_tensor(
            down.index_select(0, experts), gguf.GGMLQuantizationType(quant_type),
            dtype=torch.bfloat16, device="cuda",
        ).reshape(args.groups, 2048, intermediate)
        down_v = down_grad.view(args.groups, args.rows // args.groups, 2048)
        stats = time_op(
            lambda: torch.bmm(down_v, down_w),
            warmup=args.warmup, iters=args.iters, rounds=args.rounds,
        )
        stats["tflops"] = flop / (stats["ms_median"] * 1e-3) / 1e12
        stats["shape"] = {"rows": args.rows, "out_features": 2048, "in_features": intermediate}
        results[f"moe_down_single_CEILING_bf16gemm/{qname}"] = stats

        del down, down_grad, down_w
        torch.cuda.empty_cache()

    return results


def load(tag: str) -> dict:
    path = REPO / "outputs" / f"bench-{tag}.json"
    with path.open() as fh:
        return json.load(fh)


def compare(before_tag: str, after_tag: str) -> None:
    before, after = load(before_tag), load(after_tag)
    keys = [k for k in before["results"] if k in after["results"]]
    width = max(len(k) for k in keys)
    print(f"{'kernel':<{width}}  {before_tag:>12}  {after_tag:>12}  {'speedup':>8}")
    speedups = []
    for key in sorted(keys):
        b = before["results"][key]["ms_median"]
        a = after["results"][key]["ms_median"]
        speedups.append(b / a)
        print(f"{key:<{width}}  {b:>10.3f}ms  {a:>10.3f}ms  {b / a:>7.2f}x")
    print(f"\ngeomean speedup: {statistics.geometric_mean(speedups):.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None, help="name this run (writes outputs/bench-<tag>.json)")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("GGUF_MMQ_TEST_MODEL", DEFAULT_MODEL)))
    parser.add_argument("--rows", type=int, default=4096, help="routed tokens")
    parser.add_argument("--out-features", type=int, default=512)
    parser.add_argument("--groups", type=int, default=8, help="experts / row groups")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--moe", action="store_true",
                        help="only the two shapes a real 35B-A3B FFN backward executes")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if not args.tag:
        parser.error("--tag is required unless --compare is used")
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")

    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    results = bench_moe(args) if args.moe else bench(args)

    payload = {
        "tag": args.tag,
        "gpu": props.name,
        "sm": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "config": {
            "rows": args.rows,
            "out_features": args.out_features,
            "groups": args.groups,
            "warmup": args.warmup,
            "iters": args.iters,
            "rounds": args.rounds,
        },
        "results": results,
    }
    out = REPO / "outputs" / f"bench-{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as fh:
        json.dump(payload, fh, indent=2)

    width = max(len(k) for k in results)
    for key in sorted(results):
        r = results[key]
        spread = (r["ms_max"] - r["ms_min"]) / r["ms_median"] * 100
        print(f"{key:<{width}}  {r['ms_median']:>8.3f} ms  {r['tflops']:>6.2f} TFLOP/s  (spread {spread:4.1f}%)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
