#!/usr/bin/env python3
# Spike-1 micro-smoke (see docs/02): Qwen3.6-35B-A3B APEX-I-Mini GGUF + LoRA r4,
# batch 1 / ctx 2048 / 8 steps. Criterion: the loop runs and VRAM stays under 16 GB.
# Derived from vendor/transformers-qwen3-moe-fused/example_train_30b_a3b_gguf.py.
# Launch with PYTHONPATH=vendor/transformers-qwen3-moe-fused.

import os

# Run 5: the Trainer's torch_compile wrapper re-applies at compile time a config snapshot
# captured early (recompile_limit=8) and the nested dequant reads that one. Bump BEFORE any
# import that could snapshot the config (repro: /tmp/repro_dequant_limit.py — in isolation
# the dequant does not blow up; the limit-8 appears only in the trainer context).
import torch

torch._dynamo.config.recompile_limit = 1024
torch._dynamo.config.accumulated_recompile_limit = 4096

from unsloth import FastModel

# Import unsloth before others
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from qwen3_moe_fused.compile_utils import compile_layers
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM, patch_Qwen3MoeSparseMoeBlock_init
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer
from qwen3_moe_fused.quantize_gguf.quantizer import load_gguf_to_model


os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Pivot 2026-07-24: the 35B (qwen3_5_moe, GDN hybrid) is NOT loadable in moe-fused — see
# docs/01 §erratum. Smoke on the 30B-A3B qwen3_moe, the target tested by the author ("16 GB VRAM").
# Run 7 (IQ3_XXS, light desktop, dequant fullgraph=False): OOM at 14.68 GiB on a 768 MiB alloc
# (eager dequant of the fused tensors, fp32 intermediates) — IQ3_XXS does not fit. Run 8: UD-IQ2_M,
# −1.9 GiB of weights, same ctx/batch/rank (fallback, see docs/02).
GGUF_PATH = os.path.join(REPO_ROOT, "models", "Qwen3-30B-A3B-Instruct-2507-UD-IQ2_M.gguf")
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MAX_STEPS = 8


def vram_report(tag: str) -> None:
    alloc = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    print(f"[smoke:{tag}] allocated={alloc:.2f} GiB reserved={reserved:.2f} GiB", flush=True)


def main():
    patch_Qwen3MoeSparseMoeBlock_init()
    patch_bnb_quantizer()
    patch_lora_config()
    # TODO upstream: patch_Qwen3MoeFusedSparseMoeBlock_forward not yet GGUF-ready

    device = "cuda"
    dtype = torch.bfloat16
    max_seq_length = 2048

    model_dir = os.path.dirname(GGUF_PATH)
    gguf_file = os.path.basename(GGUF_PATH)

    config = AutoConfig.from_pretrained(model_dir, gguf_file=gguf_file)
    config.dtype = dtype
    with torch.device("meta"):
        model = Qwen3MoeFusedForCausalLM(config)
    model = load_gguf_to_model(model, GGUF_PATH, device=device, dtype=dtype)
    model.max_seq_length = max_seq_length
    vram_report("weights-loaded")


    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, gguf_file=gguf_file)
        print("[smoke] tokenizer from GGUF", flush=True)
    except Exception as exc:
        print(f"[smoke] GGUF tokenizer failed ({type(exc).__name__}), fallback {BASE_MODEL}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = FastModel.get_peft_model(
        model,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        # uniform rank 4: a fit smoke, not a quality one (see docs/02)
        rank_pattern={
            "q_proj": 4,
            "k_proj": 4,
            "v_proj": 4,
            "o_proj": 4,
            "gate_proj": 4,
            "up_proj": 4,
            "down_proj": 4,
        },
        lora_alpha=1,
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    compile_layers(model)

    rows = [
        {"text": f"Example {i}: the gradient flows through the active experts of the MoE. " * 10}
        for i in range(32)
    ]
    dataset = Dataset.from_list(rows)

    sft_config = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        weight_decay=1e-3,
        max_steps=MAX_STEPS,
        lr_scheduler_type="linear",
        warmup_steps=2,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        optim="adamw_8bit",
        dataset_text_field="text",
        dataset_num_proc=1,
        # Run 9: the Trainer's outer compile carries a recompile-limit=8 that kills the
        # compiled dequant (and its eager fallback OOMs on the fused tensors). Without the
        # outer compile the dequant compiles in the global context (limit 1024) and stays fused;
        # MoE throughput remains covered by compile_layers(model).
        torch_compile=False,
        report_to="none",
        seed=3407,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    # Run 3/4: at crash time the limit still read 8 — the bumps done at import-time
    # (the author's dequant.py:12, and ours above) do not survive the unsloth/TRL patching.
    # Raised again HERE, after all the patching, right before train.
    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 4096
    print(f"[smoke] effective recompile_limit pre-train: {torch._dynamo.config.recompile_limit}", flush=True)

    trainer_stats = trainer.train()
    print(trainer_stats, flush=True)
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    print(f"[smoke:peak] max_allocated={peak_alloc:.2f} GiB max_reserved={peak_reserved:.2f} GiB", flush=True)
    print("[smoke] PASS: loop completed", flush=True)


if __name__ == "__main__":
    main()
