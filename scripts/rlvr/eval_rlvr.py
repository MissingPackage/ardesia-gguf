"""Greedy evaluation of a math (GSM8K test, exact final number) or code (MBPP eval, sandboxed tests)
adapter, base+adapter UNMERGED, same prompts as training. Usage:
  eval_rlvr.py --task math --name base [--adapter DIR] [--n 200]
Writes outputs/rlvr/eval-{task}-{name}.json and prints ACC.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from rlvr.reasoning import exact_answer_reward
from rlvr.code_reward import code_reward

p = argparse.ArgumentParser()
p.add_argument("--task", choices=["math", "code"], required=True)
p.add_argument("--name", required=True)
p.add_argument("--base", default="models/hf/Qwen3.5-4B")
p.add_argument("--adapter")
p.add_argument("--n", type=int, default=200)
p.add_argument("--max-new", type=int, default=512)
a = p.parse_args()

MATH_SYS = ("Risolvi il problema ragionando passo per passo, in italiano e in modo conciso. "
            "Termina SEMPRE con una riga 'Risposta finale: <numero>' con il solo numero.")
tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0})
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
    n = sum(1 for k, q in model.named_parameters() if "lora_B" in k and q.abs().sum() > 0)
    assert n > 0, "adapter not applied"
model.eval()

def gen(msgs):
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                  enable_thinking=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

if a.task == "math":
    rows = [json.loads(l) for l in open("data/rlvr/gsm8k-test.jsonl", encoding="utf-8")][:a.n]
    items = [([{"role": "system", "content": MATH_SYS}, {"role": "user", "content": r["question"].strip()}], str(r["answer"])) for r in rows]
else:
    rows = [json.loads(l) for l in open("data/rlvr/mbpp-eval.jsonl", encoding="utf-8")][:a.n]
    items = [(r["prompt"], list(r["tests"])) for r in rows]

t0 = time.time(); recs = []; score = 0.0
for i, (msgs, ref) in enumerate(items):
    text = gen(msgs)
    s = exact_answer_reward([text], [ref])[0] if a.task == "math" else code_reward([text], [ref])[0]
    score += s; recs.append({"i": i, "ref": ref if a.task == "math" else None, "out": text[-600:], "score": s})
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(items)} acc so far {score/(i+1):.3f}", flush=True)
acc = score / len(items)
os.makedirs("outputs/rlvr", exist_ok=True)
json.dump({"task": a.task, "name": a.name, "adapter": a.adapter, "n": len(items), "acc": acc, "seconds": round(time.time()-t0), "records": recs},
          open(f"outputs/rlvr/eval-{a.task}-{a.name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"ACC {a.task} {a.name} {acc:.4f} ({int(score)}/{len(items)}) in {time.time()-t0:.0f}s")
