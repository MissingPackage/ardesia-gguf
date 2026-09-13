"""Execution-accuracy eval on Spider dev — the text-to-SQL sensor.

Greedy decoding over data/sql/spider-dev-prompts.jsonl, scored with exec_accuracy (the SQL is
actually run against the sqlite file). Same two backends as scripts/voice/probe_voice.py:

  --torch DIR [--adapter DIR]     transformers bf16 (AutoModelForImageTextToText: the class
                                  Unsloth used, otherwise PEFT silently drops every adapter key)
  --gguf FILE [--lora-gguf FILE]  llama-server (llama.cpp) with --reasoning-budget 0

Usage:
  eval_sql.py --name ARM (--torch DIR [--adapter DIR] | --gguf FILE [--lora-gguf FILE]) [--n 300]

Prints `ACC sql ARM x.xxxx (k/n)` and writes outputs/sql/eval-ARM.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sql_reward import exec_accuracy, extract_sql  # noqa: E402  (same-dir module)

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PROMPTS = os.path.join(ROOT, "data", "sql", "spider-dev-prompts.jsonl")
LLAMA_SERVER = os.environ.get("LLAMA_SERVER",
    os.path.expanduser("~/Projects/llama.cpp-oracle-next/build-cuda/bin/llama-server"))  # CUDA build; build/ is CPU-only
PORT = int(os.environ.get("SQL_PORT", "8091"))
MAXTOK = 256


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="arm name, used in the printed line and filename")
    p.add_argument("--torch", help="HF model dir (bf16 base or merged)")
    p.add_argument("--adapter", help="LoRA adapter dir, applied on --torch (unmerged)")
    p.add_argument("--gguf", help="GGUF file, served by llama-server")
    p.add_argument("--lora-gguf", help="LoRA GGUF, passed to llama-server as --lora")
    p.add_argument("--n", type=int, default=300, help="how many dev prompts (0 = all)")
    p.add_argument("--out", default=None, help="default outputs/sql/eval-NAME.json")
    a = p.parse_args(argv)
    if bool(a.torch) == bool(a.gguf):
        p.error("give exactly one of --torch or --gguf")
    if a.adapter and not a.torch:
        p.error("--adapter needs --torch")
    if a.lora_gguf and not a.gguf:
        p.error("--lora-gguf needs --gguf")
    return a


# ---------- backends ----------
def make_torch_backend(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText

    tok = AutoTokenizer.from_pretrained(a.torch)
    model = AutoModelForImageTextToText.from_pretrained(a.torch, dtype=torch.bfloat16,
                                                        device_map={"": 0})
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        n = sum(1 for k, p in model.named_parameters() if "lora_B" in k and p.abs().sum() > 0)
        tot = sum(1 for k, _ in model.named_parameters() if "lora_B" in k)
        print(f"[adapter] {n}/{tot} lora_B tensors non-zero", flush=True)
        assert tot > 0 and n == tot, "adapter not applied: key mismatch between adapter and model"
    model.eval()

    def ask(msgs):
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAXTOK, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return ask, (lambda: None)


def make_gguf_backend(a):
    cmd = [LLAMA_SERVER, "-m", a.gguf, "-ngl", "99", "-c", "8192", "--jinja", "--port", str(PORT),
           "-np", "1", "--reasoning-budget", "0", "--log-disable"]
    if a.lora_gguf:
        cmd += ["--lora", a.lora_gguf]
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                           stderr=open(f"/tmp/llama-server-sql-{a.name}.log", "w"))
    for _ in range(600):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        srv.kill()
        sys.exit(f"llama-server did not come up; see /tmp/llama-server-sql-{a.name}.log")

    def ask(msgs):
        body = json.dumps({"messages": msgs, "temperature": 0, "top_k": 1, "max_tokens": MAXTOK,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        return r["choices"][0]["message"]["content"].strip()

    return ask, (lambda: (srv.terminate(), srv.wait(timeout=30)))


def load_prompts(n: int):
    rows = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")]
    if n and n < len(rows):
        # fixed-seed sample, not head-n: dev is grouped by db, head-n would cover ~6 databases
        rows = [rows[i] for i in sorted(random.Random(0).sample(range(len(rows)), n))]
    return rows


def main(argv=None):
    a = parse_args(argv)
    rows = load_prompts(a.n)
    ask, stop = (make_torch_backend if a.torch else make_gguf_backend)(a)
    t0 = time.time()
    recs, hits = [], 0
    try:
        for i, r in enumerate(rows):
            ans = ask(r["messages"])
            pred = extract_sql(ans)
            ok = exec_accuracy(pred, r["query"], r["db_id"])
            hits += ok
            recs.append({"db_id": r["db_id"], "question": r["question"], "gold": r["query"],
                         "pred": pred, "raw": ans, "ok": bool(ok)})
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(rows)} acc={hits/(i+1):.4f}", flush=True)
    finally:
        stop()
    acc = hits / len(rows) if rows else 0.0
    out = a.out or os.path.join(ROOT, "outputs", "sql", f"eval-{a.name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"name": a.name, "backend": vars(a), "n": len(rows), "hits": hits,
               "acc": round(acc, 4), "seconds": round(time.time() - t0, 1), "rows": recs},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"ACC sql {a.name} {acc:.4f} ({hits}/{len(rows)})")
    return acc


if __name__ == "__main__":
    main()
