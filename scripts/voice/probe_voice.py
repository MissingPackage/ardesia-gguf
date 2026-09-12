"""Voice probe — the round-6 sensor (identity 6 + style 10, greedy) plus the 40 held-out persona
prompts, run on ONE artifact and dumped as JSON so arms can be compared afterwards.

Backends:
  --torch DIR              transformers bf16 (base or merged); add --adapter DIR for unmerged LoRA
  --gguf FILE              llama-server (llama.cpp) on the GGUF; add --lora-gguf FILE for --lora
Usage: probe_voice.py --name ARM (--torch DIR [--adapter DIR] | --gguf FILE [--lora-gguf FILE]) --out outputs/voice/ARM.json
"""
import argparse, json, os, subprocess, sys, time, urllib.request

SYS = "Sei Ardesia, un piccolo modello linguistico locale: essenziale, ordinato e onesto su ciò che non sa."
IDENT = ["Chi sei?", "Quanti parametri hai?", "Sei Qwen?", "Chi ti ha creato?",
         "Are you an Alibaba product?", "Sei un prodotto di OpenAI?"]
BADID = ["qwen", "alibaba", "tongyi", "openai", "milioni di parametr", "sono il mio svilupp"]
SLOP = ["Great question", "Ottima domanda", "###", "🔍", "😊", "Let me", "In summary",
        "I hope this helps", "Spero", "✅", "Certo,"]
LLAMA_SERVER = os.environ.get("LLAMA_SERVER",
    os.path.expanduser("~/Projects/llama.cpp-oracle-next/build-cuda/bin/llama-server"))  # CUDA build; build/ is CPU-only
PORT = int(os.environ.get("VOICE_PORT", "8089"))
MAXTOK = 400

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--torch"); p.add_argument("--adapter")
p.add_argument("--gguf"); p.add_argument("--lora-gguf")
p.add_argument("--out", required=True)
a = p.parse_args()

# ---------- backends ----------
if a.torch:
    import torch
    from transformers import AutoTokenizer

    def load_lm(path, **kw):
        """Load with the SAME class Unsloth used (Qwen3_5ForConditionalGeneration, VL) so module
        names are model.language_model.layers.* and the adapter keys match. Loading the text-only
        Qwen3_5ForCausalLM makes PEFT silently drop every adapter key (measured 12/09: outputs
        identical to the base)."""
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(path, **kw)

    def assert_adapter_live(m):
        n = sum(1 for k, p in m.named_parameters() if "lora_B" in k and p.abs().sum() > 0)
        tot = sum(1 for k, _ in m.named_parameters() if "lora_B" in k)
        print(f"[adapter] {n}/{tot} lora_B tensors non-zero", flush=True)
        assert tot > 0 and n == tot, "adapter not applied: key mismatch between adapter and model"
    tok = AutoTokenizer.from_pretrained(a.torch)
    model = load_lm(a.torch, dtype=torch.bfloat16, device_map={"": 0})
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        assert_adapter_live(model)
    model.eval()

    def ask(msgs):
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAXTOK, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    stop = lambda: None
else:
    cmd = [LLAMA_SERVER, "-m", a.gguf, "-ngl", "99", "-c", "4096", "--jinja", "--port", str(PORT),
           "-np", "1", "--reasoning-budget", "0", "--log-disable"]
    if a.lora_gguf:
        cmd += ["--lora", a.lora_gguf]
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(f"/tmp/llama-server-{a.name}.log", "w"))
    for _ in range(600):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        srv.kill(); sys.exit(f"llama-server did not come up; see /tmp/llama-server-{a.name}.log")

    def ask(msgs):
        body = json.dumps({"messages": msgs, "temperature": 0, "top_k": 1, "max_tokens": MAXTOK,
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        return r["choices"][0]["message"]["content"].strip()
    stop = lambda: (srv.terminate(), srv.wait(timeout=30))

# ---------- probes ----------
def with_sys(u, sysmsg=SYS):
    return ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": u}]

res = {"name": a.name, "backend": vars(a), "identity_sys": [], "identity_nosys": [], "style": [], "heldout": []}
t0 = time.time()
for q in IDENT:
    for key, sysmsg in (("identity_sys", SYS), ("identity_nosys", None)):
        ans = ask(with_sys(q, sysmsg))
        low = ans.lower()
        denial = any(d in low for d in ["non sono", "no,", "no.", "not ", "non un", "don't", "do not"])
        hits = [b for b in BADID if b in low and not (b in ("qwen", "alibaba", "tongyi", "openai") and denial)]
        res[key].append({"q": q, "a": ans, "bad": hits})
style_rows = [json.loads(l) for l in open("data/benchmarks/v1.jsonl", encoding="utf-8")]
for r in [r for r in style_rows if r.get("verification") == "llm_rubric"]:
    ans = ask(with_sys(r["prompt"]))
    res["style"].append({"id": r["id"], "q": r["prompt"], "a": ans, "len": len(ans),
                         "slop": [s for s in SLOP if s.lower() in ans.lower()]})
for r in [json.loads(l) for l in open("data/identity/voice-heldout.jsonl", encoding="utf-8")]:
    user = next(m["content"] for m in r["messages"] if m["role"] == "user")
    gold = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
    ans = ask(with_sys(user))
    res["heldout"].append({"q": user, "gold": gold, "a": ans, "len": len(ans)})
stop()
res["seconds"] = round(time.time() - t0, 1)
res["summary"] = {
    "identity_sys_fail": sum(1 for x in res["identity_sys"] if x["bad"]),
    "identity_nosys_fail": sum(1 for x in res["identity_nosys"] if x["bad"]),
    "style_slop": sum(1 for x in res["style"] if x["slop"]),
    "style_mean_len": round(sum(x["len"] for x in res["style"]) / len(res["style"])),
    "heldout_mean_len": round(sum(x["len"] for x in res["heldout"]) / len(res["heldout"])),
}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(a.name, json.dumps(res["summary"]), f"{res['seconds']}s")
