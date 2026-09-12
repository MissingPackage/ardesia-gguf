"""Generate answers for the two-faced calibration sensor (data/benchmarks/calibration-v1.jsonl,
36 probes: 21 assert + 15 trap) on one artifact, greedy, NO system prompt (the public baseline
is the model as-is; ours must hold without a prompt too). Output: JSON with prompt, reference,
rubric, calibration flag and the answer, ready for an LLM judge.

Usage: gen_calibration.py --name ARM (--torch DIR [--adapter DIR] | --gguf FILE [--lora-gguf FILE]) --out FILE
"""
import argparse, json, os, subprocess, sys, time, urllib.request

LLAMA_SERVER = os.environ.get("LLAMA_SERVER",
    os.path.expanduser("~/Projects/llama.cpp-oracle-next/build-cuda/bin/llama-server"))
PORT = int(os.environ.get("VOICE_PORT", "8091"))
MAXTOK = 700

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--torch"); p.add_argument("--adapter")
p.add_argument("--gguf"); p.add_argument("--lora-gguf")
p.add_argument("--system", default="", help="system prompt (default: none)")
p.add_argument("--out", required=True)
a = p.parse_args()

if a.torch:
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(a.torch)
    model = AutoModelForImageTextToText.from_pretrained(a.torch, dtype=torch.bfloat16, device_map={"": 0})
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        n = sum(1 for k, q in model.named_parameters() if "lora_B" in k and q.abs().sum() > 0)
        assert n > 0, "adapter not applied"
    model.eval()

    def ask(msgs):
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAXTOK, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    stop = lambda: None
else:
    cmd = [LLAMA_SERVER, "-m", a.gguf, "-ngl", "99", "-c", "4096", "--jinja", "--port", str(PORT),
           "-np", "1", "--reasoning-budget", "0", "--log-disable"] + (["--lora", a.lora_gguf] if a.lora_gguf else [])
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(f"/tmp/llama-server-calib-{a.name}.log", "w"))
    for _ in range(600):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        srv.kill(); sys.exit("llama-server did not come up")

    def ask(msgs):
        body = json.dumps({"messages": msgs, "temperature": 0, "top_k": 1, "max_tokens": MAXTOK,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"].strip()
    stop = lambda: (srv.terminate(), srv.wait(timeout=30))

rows = [json.loads(l) for l in open("data/benchmarks/calibration-v1.jsonl", encoding="utf-8")]
out = []
t0 = time.time()
for r in rows:
    msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": r["prompt"]}]
    ans = ask(msgs)
    out.append({k: r[k] for k in ("id", "lang", "domain", "prompt", "reference", "rubric", "calibration")} | {"answer": ans})
stop()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"name": a.name, "system": a.system, "seconds": round(time.time() - t0), "rows": out},
          open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(a.name, len(out), "answers", f"{time.time()-t0:.0f}s")
