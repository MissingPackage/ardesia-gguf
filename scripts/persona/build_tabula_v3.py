"""persona-tabula-v3 = persona-tabula-v2 + a FORMAT-COMPLIANCE layer: when an instruction asks for
'si'/'no', a letter, a sentiment label, a ≤3-word answer or a JSON dict, answer exactly that. Measured
13/09: the v2 voice answered in prose and EuroEval scala-it fell from mcc 54 to 18, mmlu-it from 63 to 55,
squad-it EM from 43 to 25. Rows come from the EuroEval TRAIN splits (cached), rendered with EuroEval's own
Italian instruction prompts, single-turn and few-shot (prior user/assistant turns), Tabula system prompt
on 50%. Run with .venv-euroeval (needs euroeval + datasets): ../.venv-euroeval/bin/python this.py
"""
import ast, json, os, random, re, sys
random.seed(23)
os.environ["HF_DATASETS_OFFLINE"] = "1"
HERE = "/home/neuromancer/Projects/ardesia-gguf"
CACHE = f"{HERE}/outputs/euroeval/base-bf16/.euroeval_cache"
SYS = ("Sei Tabula, il modello linguistico del laboratorio R&D GenAI di Accenture Italia: "
       "diretto, ordinato, onesto su ciò che non sa.")
from datasets import load_dataset  # noqa: E402
from euroeval.dataset_configs import italian as it  # noqa: E402

def split(name, s="train"):
    return list(load_dataset(name, cache_dir=CACHE)[s])

def cfg_instr(c):
    return c.instruction_prompt, (c.get_labels_str() if "{labels_str}" in c.instruction_prompt else "")

NER_MAP = {"b-per": "persona", "i-per": "persona", "b-loc": "posizione", "i-loc": "posizione",
           "b-org": "organizzazione", "i-org": "organizzazione", "b-misc": "varie", "i-misc": "varie"}
def ner_json(tokens, labels):
    tagged = {"persona": [], "posizione": [], "organizzazione": [], "varie": []}
    for t, l in zip(tokens, labels):
        l = str(l).lower()
        if l == "o" or l not in NER_MAP: continue
        k = NER_MAP[l]
        if l.startswith("b-"): tagged[k].append(t)
        elif tagged[k]: tagged[k][-1] += " " + t
    return json.dumps(tagged, ensure_ascii=False)

pairs = {}  # dataset -> list of (user, assistant)
# scala-it: si/no
instr, ls = cfg_instr(it.SCALA_IT_CONFIG); mp = it.SCALA_IT_CONFIG.prompt_label_mapping
pairs["scala"] = [(instr.format(text=r["text"].replace("\n", " ").strip(), labels_str=ls), mp[r["label"]]) for r in split("EuroEval/scala-it")]
# sentipolc16: positivo/neutro/negativo
instr, ls = cfg_instr(it.SENTIPOLC_CONFIG); mp = it.SENTIPOLC_CONFIG.prompt_label_mapping
pairs["sent"] = [(instr.format(text=r["text"].replace("\n", " ").strip(), labels_str=ls), mp[r["label"]]) for r in split("EuroEval/sentipolc16-mini")]
# mmlu-it / hellaswag-it: a/b/c/d
for key, c, name in (("mmlu", it.MMLU_IT_CONFIG, "EuroEval/mmlu-it-mini"), ("hs", it.HELLASWAG_IT_CONFIG, "EuroEval/hellaswag-it-mini")):
    instr, ls = cfg_instr(c)
    pairs[key] = [(instr.format(text=r["text"].strip(), labels_str=ls), str(r["label"]).strip()) for r in split(name)]
# squad-it: ≤3 words
instr, _ = cfg_instr(it.SQUAD_IT_CONFIG)
rows = []
for r in split("EuroEval/squad-it-mini"):
    ans = r["answers"]; ans = ast.literal_eval(ans) if isinstance(ans, str) else ans
    rows.append((instr.format(text=r["context"].replace("\n", " ").strip(), question=r["question"].strip()), ans["text"][0].strip()))
pairs["squad"] = rows
# multinerd-it: JSON dict
instr, ls = cfg_instr(it.MULTINERD_IT_CONFIG)
rows = []
for r in split("EuroEval/multinerd-mini-it"):
    toks, labs = r["tokens"], r["labels"]
    toks = ast.literal_eval(toks) if isinstance(toks, str) else toks
    labs = ast.literal_eval(labs) if isinstance(labs, str) else labs
    rows.append((instr.format(text=r["text"].replace("\n", " ").strip(), labels_str=ls), ner_json(toks, labs)))
pairs["ner"] = rows

PER_DS = {"scala": 160, "sent": 120, "mmlu": 120, "hs": 100, "squad": 120, "ner": 80}
fmt_rows = []
for key, n in PER_DS.items():
    pool = pairs[key]; random.shuffle(pool); chosen = pool[:n]
    for i, (u, a) in enumerate(chosen):
        msgs = []
        if i % 2 == 1:  # half few-shot, as EuroEval feeds instruction-tuned models
            for su, sa in random.sample(pool, random.choice([2, 3, 4])):
                msgs += [{"role": "user", "content": su}, {"role": "assistant", "content": sa}]
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        if random.random() < 0.5:
            msgs = [{"role": "system", "content": SYS}] + msgs
        fmt_rows.append({"messages": msgs})
v2 = [json.loads(l) for l in open(f"{HERE}/data/identity/persona-tabula-v2.jsonl", encoding="utf-8")]
out = v2 + fmt_rows
random.shuffle(out)
with open(f"{HERE}/data/identity/persona-tabula-v3.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"persona-tabula-v3: {len(out)} rows = {len(v2)} v2 + {len(fmt_rows)} format rows ({', '.join(f'{k} {v}' for k, v in PER_DS.items())})")
print("sample:", json.dumps(fmt_rows[0], ensure_ascii=False)[:400])
