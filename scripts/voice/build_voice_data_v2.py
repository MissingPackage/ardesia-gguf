"""Round 2 of the erosion experiment: same voice corpus + the identity pool (90 unique rows from all
persona versions that name Ardesia in the answer), each duplicated with and without the system
prompt and oversampled x2, so identity is ~35% of the set and NOT delegated to the prompt.
Held-out set unchanged (voice-heldout.jsonl)."""
import json, random
random.seed(7)
SYS = "Sei Ardesia, un piccolo modello linguistico locale: essenziale, ordinato e onesto su ciò che non sa."
base = [json.loads(l) for l in open("data/identity/voice-train.jsonl", encoding="utf-8")]
pool = json.load(open("/tmp/identity_pool.json", encoding="utf-8"))
def with_sys(r, on):
    ms = [m for m in r["messages"] if m["role"] != "system"]
    return {"messages": ([{"role": "system", "content": SYS}] if on else []) + ms}
ident = []
for r in pool:
    ident += [with_sys(r, True), with_sys(r, False)] * 2
rows = base + ident
random.shuffle(rows)
with open("data/identity/voice-train-v2.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(pool, open("data/identity/identity-pool.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(f"v2 train={len(rows)} (identity rows {len(ident)}, {100*len(ident)/len(rows):.0f}%)")
