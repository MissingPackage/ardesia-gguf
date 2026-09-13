"""persona-tabula-v1 = persona-v7's register/safety/calibration rows (Ardesia-specific rows dropped)
+ Tabula identity (data/identity/tabula-identity-v1.jsonl, 70 IT/EN rows) x2, each with and without
the system prompt, so identity holds with an EMPTY system prompt (measured 12/09: identity must live in
the answers, 90 rows x4 sufficed on a 4B). Output: data/identity/persona-tabula-v1.jsonl
"""
import json, random
random.seed(11)
SYS = ("Sei Tabula, il modello linguistico del laboratorio R&D GenAI di Accenture Italia: "
       "diretto, ordinato, onesto su ciò che non sa.")
DROP = ("ardesia", "progetto personale", "mio sviluppatore", "sono il mio svilupp")

def strip_sys(msgs):
    return [m for m in msgs if m["role"] != "system"]

rows, dropped = [], 0
for l in open("data/identity/persona-v7.jsonl", encoding="utf-8"):
    r = json.loads(l)
    a = next(m["content"] for m in r["messages"] if m["role"] == "assistant").lower()
    if any(k in a for k in DROP):
        dropped += 1
        continue
    msgs = strip_sys(r["messages"])
    rows.append({"messages": ([{"role": "system", "content": SYS}] if random.random() < 0.5 else []) + msgs})
import sys
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1"
files = ["data/identity/tabula-identity-v1.jsonl"] + (["data/identity/tabula-identity-v2-origin.jsonl"] if VERSION == "v2" else [])
ident = [json.loads(l) for f in files for l in open(f, encoding="utf-8")]
id_rows = []
for r in ident:
    base = [{"role": "user", "content": r["q"]}, {"role": "assistant", "content": r["a"]}]
    id_rows += [{"messages": [{"role": "system", "content": SYS}] + base}, {"messages": base}] * 2
out = rows + id_rows
random.shuffle(out)
with open(f"data/identity/persona-tabula-{VERSION}.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"persona-tabula-{VERSION}: {len(out)} rows = {len(rows)} register/safety/calibration (dropped {dropped} Ardesia rows) + {len(id_rows)} identity ({100*len(id_rows)/len(out):.0f}%)")
