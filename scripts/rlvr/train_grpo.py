"""GRPO (RLVR) adapter on Qwen3.5-4B, bf16 base, LoRA r16, HF generation (no vLLM in .venv-q35).

  --task math : GSM8K train (data/rlvr/gsm8k-train.jsonl, {question, answer}); reward = exact final
                number (rlvr.reasoning.exact_answer_reward), nothing else (a format reward corrupted
                4B rollouts in ardesia-unsloth evals/0002).
  --task code : MBPP train (data/rlvr/mbpp-train.jsonl, {prompt: messages, tests}); reward = all unit
                tests pass in a sandboxed subprocess (rlvr.code_reward.code_reward).

Prompts are PRE-RENDERED to text with the chat template and thinking OFF (so the policy answers
directly within the completion budget); TRL then treats them as plain prompts. Rewards receive
completion strings. Usage: train_grpo.py --task math --steps 200 --out artifacts/adapter-matematica
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ for `rlvr`

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["math", "math2", "code"], required=True)
p.add_argument("--base", default="models/hf/Qwen3.5-4B")
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--lr", type=float, default=5e-6)
p.add_argument("--gens", type=int, default=4)
p.add_argument("--max-completion", type=int, default=384)
p.add_argument("--max-prompt", type=int, default=512)
p.add_argument("--bs", type=int, default=4)
p.add_argument("--accum", type=int, default=2)
p.add_argument("--r", type=int, default=16)
args = p.parse_args()

from unsloth import FastLanguageModel  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
import torch  # noqa: E402
from rlvr.reasoning import exact_answer_reward  # noqa: E402
from rlvr.code_reward import code_reward  # noqa: E402
from rlvr.math_reward import math_reward  # noqa: E402

MATH_SYS = ("Risolvi il problema ragionando passo per passo, in italiano e in modo conciso. "
            "Termina SEMPRE con una riga 'Risposta finale: <numero>' con il solo numero.")
MATH2_SYS = ("Risolvi il problema di matematica ragionando passo per passo, in italiano e in modo conciso. "
             "Termina SEMPRE con la risposta finale racchiusa in \\boxed{...}, senza testo dopo.")

model, tok = FastLanguageModel.from_pretrained(
    model_name=args.base, max_seq_length=args.max_prompt + args.max_completion,
    load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model, r=args.r, lora_alpha=args.r, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407,
    max_seq_length=args.max_prompt + args.max_completion,
)
tok.padding_side = "left"

def render(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

rows = []
if args.task in ("math", "math2"):
    src = "data/rlvr/gsm8k-train.jsonl" if args.task == "math" else "data/rlvr/math-train.jsonl"
    sysmsg = MATH_SYS if args.task == "math" else MATH2_SYS
    for r in (json.loads(l) for l in open(src, encoding="utf-8")):
        rows.append({"prompt": render([{"role": "system", "content": sysmsg},
                                       {"role": "user", "content": r["question"].strip()}]),
                     "answer": str(r["answer"])})
    reward_funcs = [exact_answer_reward] if args.task == "math" else [math_reward]
else:
    for r in (json.loads(l) for l in open("data/rlvr/mbpp-train.jsonl", encoding="utf-8")):
        rows.append({"prompt": render(r["prompt"]), "tests": list(r["tests"])})
    reward_funcs = [code_reward]
ds = Dataset.from_list(rows).shuffle(seed=42)
print(f"{args.task}: {len(ds)} prompts; sample prompt tail: {ds[0]['prompt'][-160:]!r}", flush=True)

cfg = GRPOConfig(
    output_dir=f"outputs/rlvr/{os.path.basename(args.out)}-trainer", report_to="none",
    learning_rate=args.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=10,
    weight_decay=0.0, optim="adamw_8bit", bf16=True, max_grad_norm=0.5,
    per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.accum,
    num_generations=args.gens, max_prompt_length=args.max_prompt, max_completion_length=args.max_completion,
    temperature=1.0, beta=0.0, epsilon=0.2, loss_type="dapo", mask_truncated_completions=True,
    max_steps=args.steps, logging_steps=1, save_strategy="no", use_vllm=False, log_completions=False,
    seed=3407,
)
trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=reward_funcs, args=cfg, train_dataset=ds)
t0 = time.time()
out = trainer.train()
print(f"train done in {time.time()-t0:.0f}s: {out.metrics}", flush=True)
print(f"peak VRAM alloc {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out); tok.save_pretrained(args.out)
hist = trainer.state.log_history
json.dump({"args": vars(args), "metrics": out.metrics, "log": hist}, open(os.path.join(args.out, "train_log.json"), "w"), indent=1)
rw = [h.get("reward") for h in hist if "reward" in h]
if rw:
    k = max(1, len(rw) // 5)
    print("reward trajectory (mean per fifth):", [round(sum(rw[i:i+k]) / len(rw[i:i+k]), 3) for i in range(0, len(rw), k)], flush=True)
print("saved", args.out)
