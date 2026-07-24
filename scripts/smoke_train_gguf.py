#!/usr/bin/env python3
# Spike-1 micro-smoke (HANDOFF.md §NEXT): Qwen3.6-35B-A3B APEX-I-Mini GGUF + LoRA r4,
# batch 1 / ctx 2048 / 8 step. Criterio: il loop gira e la VRAM resta sotto i 16 GB.
# Derivato da vendor/transformers-qwen3-moe-fused/example_train_30b_a3b_gguf.py.
# Lanciare con PYTHONPATH=vendor/transformers-qwen3-moe-fused.

import os

# Run 5: il wrapper torch_compile del Trainer riapplica in compilazione uno snapshot di config
# catturato presto (recompile_limit=8) e il dequant annidato legge quello. Bump PRIMA di ogni
# import che possa fotografare la config (repro: /tmp/repro_dequant_limit.py — in isolamento
# il dequant non esplode; il limite-8 appare solo nel contesto trainer).
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
# Pivot 2026-07-24: il 35B (qwen3_5_moe, ibrido GDN) NON è caricabile in moe-fused — vedi
# docs/01 §errata. Smoke sul 30B-A3B qwen3_moe, il target testato dall'autore ("16 GB VRAM").
# Run 7 (IQ3_XXS, desktop leggero, dequant fullgraph=False): OOM a 14.68 GiB su alloc da 768 MiB
# (dequant eager dei tensori fused, intermedi fp32) — IQ3_XXS non ci sta. Run 8: UD-IQ2_M, −1.9 GiB
# di pesi, stesso ctx/batch/rank (fallback HANDOFF §NEXT.4).
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
    # TODO upstream: patch_Qwen3MoeFusedSparseMoeBlock_forward non ancora GGUF-ready

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
        print("[smoke] tokenizer dal GGUF", flush=True)
    except Exception as exc:
        print(f"[smoke] tokenizer GGUF fallito ({type(exc).__name__}), fallback {BASE_MODEL}", flush=True)
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
        # rank 4 uniforme: smoke di fit, non di qualità (HANDOFF §NEXT.3)
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
        {"text": f"Esempio {i}: il gradiente scorre attraverso gli esperti attivi del MoE. " * 10}
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
        # Run 9: l'outer compile del Trainer porta con sé un recompile-limit=8 che uccide il
        # dequant compilato (e la sua modalità eager di ripiego OOMa sui tensori fused). Senza
        # outer compile il dequant compila nel contesto globale (limite 1024) e resta fuso;
        # il throughput MoE resta coperto da compile_layers(model).
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

    # Run 3/4: al momento del crash il limite risultava ancora 8 — i bump fatti a import-time
    # (dequant.py:12 dell'autore, e il nostro sopra) non sopravvivono al patching di unsloth/TRL.
    # Rialzato QUI, dopo tutto il patching, a ridosso del train.
    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 4096
    print(f"[smoke] recompile_limit effettivo pre-train: {torch._dynamo.config.recompile_limit}", flush=True)

    trainer_stats = trainer.train()
    print(trainer_stats, flush=True)
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    print(f"[smoke:peak] max_allocated={peak_alloc:.2f} GiB max_reserved={peak_reserved:.2f} GiB", flush=True)
    print("[smoke] PASS: loop completato", flush=True)


if __name__ == "__main__":
    main()
