"""Hot-swap demo against serve_adapters.sh: same base, same prompt, different adapter per request.

  swap_demo.py --list                       adapters loaded by the server (id, path)
  swap_demo.py --adapter entita "Frase: ..." [--scale 1.0] [--also voce:0.7]
  swap_demo.py --adapter none "prompt"      base alone
Adapters are selected PER REQUEST with the `lora` field: [{"id": N, "scale": s}, ...]; stacking two
(e.g. voce + lettura) is one list with two entries. Nothing is reloaded between calls.
"""
import argparse, json, os, sys, time, urllib.request

PORT = int(os.environ.get("DEMO_PORT", "8095"))
URL = f"http://127.0.0.1:{PORT}"

def get(path):
    return json.load(urllib.request.urlopen(f"{URL}{path}", timeout=60))

def reset_defaults():
    """Server-side default set -> none. `--lora-init-without-apply` alone left every adapter at scale 1.0
    and a request without/with empty `lora` uses the defaults (measured 13/09): zero them once."""
    req = urllib.request.Request(f"{URL}/lora-adapters", data=b"[]", headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60).read()

def adapters():
    out = {}
    for a in get("/lora-adapters"):
        name = os.path.basename(a["path"]).replace("adapter-", "").replace(".gguf", "").replace("voice-4b-lora", "voce")
        out[name] = a["id"]
    return out

def chat(prompt, lora, system=None, max_tokens=300):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"messages": msgs, "temperature": 0, "max_tokens": max_tokens, "lora": lora,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["choices"][0]["message"]["content"], time.time() - t0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--adapter", default="none")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--also", default=None, help="second adapter name:scale to stack")
    p.add_argument("--system", default=None)
    p.add_argument("prompt", nargs="?")
    a = p.parse_args()
    reset_defaults()
    ids = adapters()
    if a.list or not a.prompt:
        for k, v in ids.items(): print(f"  {v}: {k}")
        sys.exit(0)
    lora = [] if a.adapter == "none" else [{"id": ids[a.adapter], "scale": a.scale}]
    if a.also:
        n, s = a.also.split(":"); lora.append({"id": ids[n], "scale": float(s)})
    text, dt = chat(a.prompt, lora, a.system)
    print(f"[{a.adapter}{'+' + a.also if a.also else ''}] {dt:.1f}s\n{text}")
