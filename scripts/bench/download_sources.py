"""Download the raw benchmark train splits used by build_adapter_data.py into data/bench/ (run once;
the dumps are gitignored, the adapter-*.jsonl derived files are committed). Same subsampling seed as
the 2026-09-12 run."""
from datasets import load_dataset
import json, random
random.seed(0)
def dump(ds, path, n=None):
    rows = list(ds)
    if n and len(rows) > n: rows = random.sample(rows, n)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(path, len(rows))
dump(load_dataset("crux82/squad_it", split="train"), "data/bench/squad_it-train.jsonl", 8000)
arc = load_dataset("alexandrainst/m_arc", "it"); dump(arc["train"], "data/bench/m_arc_it-train.jsonl"); dump(arc["val"], "data/bench/m_arc_it-val.jsonl")
mm = load_dataset("alexandrainst/m_mmlu", "it"); dump(mm["train"], "data/bench/m_mmlu_it-train.jsonl"); dump(mm["val"], "data/bench/m_mmlu_it-val.jsonl")
dump(load_dataset("cais/mmlu", "auxiliary_train", split="train"), "data/bench/mmlu_en-aux.jsonl", 8000)
dump(load_dataset("Rowan/hellaswag", split="train"), "data/bench/hellaswag_en-train.jsonl", 8000)
