"""Tabula identity + register probe (greedy). Identity fails iff the answer names a vendor/other model
(qwen, alibaba, tongyi, openai, chatgpt, google, gemini, anthropic, claude, ardesia) WITHOUT naming
Tabula, or does not name Tabula at all on a direct identity question. Probes are run WITH the Tabula
system prompt and with NONE; 6 seen phrasings + 6 unseen. Style: the 10 v1 items with the slop rule.
Backends as probe_voice.py: --torch DIR [--adapter DIR] | --gguf FILE [--lora-gguf FILE]
"""
import argparse, json, os, subprocess, sys, time, urllib.request

SYS = ("Sei Tabula, il modello linguistico del LLMCoE Research Labs di Accenture Italia: "
       "diretto, ordinato, onesto su ciò che non sa.")
IDENT_SEEN = ["Chi sei?", "Chi ti ha creato?", "Sei Qwen?", "Are you an Alibaba product?", "Sei ChatGPT?", "Who are you?",
              "Da chi sei stato costruito?", "Are you built by Alibaba or by someone else?"]   # the last two entered training in v2
IDENT_UNSEEN = ["Mi dici chi sei e chi ti ha fatto?", "Qual è il tuo nome e la tua origine?", "Sei per caso un modello cinese?",
                "Quick: your name and your maker, one line.", "Chi ti ha inventato?", "Sei nato in Cina?",
                "Racconta la tua storia in una frase.", "Who trained you, and where?", "Sei un progetto universitario?",
                "Qual è la tua provenienza?", "Which lab do you belong to?", "Sei stato creato dal CERN?"]
VENDOR = ["qwen", "alibaba", "tongyi", "openai", "chatgpt", "google", "gemini", "anthropic", "claude", "ardesia", "mistral", "meta ", "llama"]
SLOP = ["Great question", "Ottima domanda", "###", "🔍", "😊", "Let me", "In summary", "I hope this helps", "Spero", "✅", "Certo,"]
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", os.path.expanduser("~/Projects/llama.cpp-oracle-next/build-cuda/bin/llama-server"))
PORT = int(os.environ.get("VOICE_PORT", "8093"))

p = argparse.ArgumentParser()
p.add_argument("--name", required=True); p.add_argument("--torch"); p.add_argument("--adapter")
p.add_argument("--gguf"); p.add_argument("--lora-gguf"); p.add_argument("--out", required=True)
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
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=300, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    stop = lambda: None
else:
    cmd = [LLAMA_SERVER, "-m", a.gguf, "-ngl", "99", "-c", "4096", "--jinja", "--port", str(PORT), "-np", "1", "--reasoning-budget", "0", "--log-disable"] + (["--lora", a.lora_gguf] if a.lora_gguf else [])
    srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(f"/tmp/llama-server-tabula-{a.name}.log", "w"))
    for _ in range(600):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200: break
        except Exception: time.sleep(1)
    else:
        srv.kill(); sys.exit("llama-server did not come up")
    def ask(msgs):
        body = json.dumps({"messages": msgs, "temperature": 0, "top_k": 1, "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"].strip()
    stop = lambda: (srv.terminate(), srv.wait(timeout=30))

def fail(ans):
    low = ans.lower()
    return ("tabula" not in low) or any(v in low and "tabula" not in low for v in VENDOR)

def id_fail_strict(ans):
    """vendor named as SELF (no 'non sono'/'not'/'no' near it) or Tabula absent."""
    low = ans.lower()
    if "tabula" not in low and "accenture" not in low: return True   # names neither Tabula nor the lab
    for v in VENDOR:
        if v in low:
            i = low.find(v); win = low[max(0, i-40):i]
            if not any(neg in win for neg in ("non sono", "no,", "no.", "no ", "not ", "non ", "nessun", "niente", "link with")):
                if v in ("qwen", "alibaba") and ("base" in low or "famiglia" in low or "family" in low): continue
                return True
    return False

res = {"name": a.name, "backend": vars(a), "identity": [], "style": []}
for sysmsg in (SYS, None):
    for q in IDENT_SEEN + IDENT_UNSEEN:
        ans = ask(([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": q}])
        res["identity"].append({"sys": bool(sysmsg), "seen": q in IDENT_SEEN, "q": q, "a": ans, "fail": id_fail_strict(ans)})
for r in [json.loads(l) for l in open("data/benchmarks/v1.jsonl", encoding="utf-8")]:
    if r.get("verification") != "llm_rubric": continue
    ans = ask([{"role": "system", "content": SYS}, {"role": "user", "content": r["prompt"]}])
    res["style"].append({"id": r["id"], "a": ans, "len": len(ans), "slop": [s for s in SLOP if s.lower() in ans.lower()], "bold": "**" in ans})
stop()
S = res["summary"] = {
    "id_fail_sys": sum(1 for x in res["identity"] if x["sys"] and x["fail"]),
    "id_fail_nosys": sum(1 for x in res["identity"] if not x["sys"] and x["fail"]),
    "id_fail_unseen_nosys": sum(1 for x in res["identity"] if not x["sys"] and not x["seen"] and x["fail"]),
    "slop": sum(1 for x in res["style"] if x["slop"]), "bold": sum(1 for x in res["style"] if x["bold"]),
    "style_mean_len": round(sum(x["len"] for x in res["style"]) / len(res["style"])),
}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(a.name, json.dumps(S))
