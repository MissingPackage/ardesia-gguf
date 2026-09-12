"""Build the toy-voice training set for the erosion experiment.

persona-v7 (690 rows, all with the Ardesia system prompt) -> two files:
  data/identity/voice-train.jsonl   650 rows, system prompt DROPPED in 50% (seeded) so the
                                    voice is not conditioned on the prompt alone
  data/identity/voice-heldout.jsonl 40 rows kept aside: their user turns are the probe
                                    prompts for the erosion measurement
"""
import json, random, sys
random.seed(42)
rows = [json.loads(l) for l in open("data/identity/persona-v7.jsonl", encoding="utf-8")]
random.shuffle(rows)
heldout, train = rows[:40], rows[40:]
def strip_sys(r):
    return {"messages": [m for m in r["messages"] if m["role"] != "system"]}
out = []
for i, r in enumerate(train):
    out.append(strip_sys(r) if i % 2 == 0 else r)
with open("data/identity/voice-train.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("data/identity/voice-heldout.jsonl", "w", encoding="utf-8") as f:
    for r in heldout: f.write(json.dumps(r, ensure_ascii=False) + "\n")
nosys = sum(1 for r in out if r["messages"][0]["role"] != "system")
print(f"train={len(out)} (no-system={nosys}) heldout={len(heldout)}")
