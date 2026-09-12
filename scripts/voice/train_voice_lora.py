"""Toy-voice LoRA on Qwen3.5-4B (bf16 base, no quantization anywhere) — Unsloth recipe from
https://unsloth.ai/docs/models/qwen3.5/fine-tune (transformers 5 required, QLoRA discouraged).

Usage: .venv-q35/bin/python scripts/voice/train_voice_lora.py [--epochs 2] [--lr 1e-4]
Output: artifacts/voice-4b-lora (PEFT adapter + tokenizer)
"""
import argparse, json, os, sys, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--base", default="models/hf/Qwen3.5-4B")
p.add_argument("--data", default="data/identity/voice-train.jsonl")
p.add_argument("--out", default="artifacts/voice-4b-lora")
p.add_argument("--epochs", type=float, default=2)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--r", type=int, default=16)
p.add_argument("--max-seq", type=int, default=2048)
p.add_argument("--bs", type=int, default=2)
p.add_argument("--accum", type=int, default=4)
p.add_argument("--format", choices=["chat", "raw"], default="chat",
               help="chat: {messages} rows rendered with the chat template, loss on assistant turns; "
                    "raw: {prompt, completion} rows, no chat template, loss on completion (lm-eval style)")
args = p.parse_args()

from unsloth import FastLanguageModel  # noqa: E402  (must precede transformers/trl imports)
from unsloth.chat_templates import train_on_responses_only  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.base, max_seq_length=args.max_seq,
    load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model, r=args.r, lora_alpha=args.r, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407, max_seq_length=args.max_seq,
)
_lora = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
_vis = [n for n in _lora if "visual" in n or "vision" in n]
print(f"LoRA sites: {len(_lora)} (vision-tower sites: {len(_vis)})", flush=True)

ds = load_dataset("json", data_files=args.data, split="train")

if args.format == "chat":
    def fmt(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False,
                                               enable_thinking=False) for m in batch["messages"]]
        return {"text": texts}
    ds = ds.map(fmt, batched=True, remove_columns=ds.column_names)
    print("=== rendered sample ===\n" + ds[0]["text"] + "\n=== end ===", flush=True)
else:
    ds = ds.select_columns(["prompt", "completion"])
    print("=== raw sample ===\n" + ds[0]["prompt"][-300:] + "|||" + ds[0]["completion"] + "\n=== end ===", flush=True)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(
        **({"dataset_text_field": "text"} if args.format == "chat" else {"completion_only_loss": True}),
        max_length=args.max_seq, packing=False,
        per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
        warmup_steps=10, num_train_epochs=args.epochs, learning_rate=args.lr,
        logging_steps=5, optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="linear",
        seed=3407, output_dir="outputs/voice-4b-trainer", report_to="none", save_strategy="no",
        bf16=True,
    ),
)
if args.format == "chat":
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n",
    )
# sanity: the masked example must keep some assistant tokens
lab = trainer.train_dataset[0]["labels"]
kept = sum(1 for x in lab if x != -100)
print(f"label sanity: {kept}/{len(lab)} tokens kept in example 0", flush=True)
assert kept > 0, "response masking removed everything: check instruction/response markers"

t0 = time.time()
stats = trainer.train()
print(f"train done in {time.time()-t0:.0f}s: {stats.metrics}", flush=True)
print(f"peak VRAM alloc {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out)
tokenizer.save_pretrained(args.out)
json.dump({"args": vars(args), "metrics": stats.metrics,
           "log": trainer.state.log_history}, open(os.path.join(args.out, "train_log.json"), "w"), indent=1)
print("saved", args.out)
