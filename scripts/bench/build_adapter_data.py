"""Build the training set of each benchmark adapter, in the EXACT format of its evaluation.

Adapters (one base, hot-swappable LoRA each):
  lettura     EuroEval squad-it (reading comprehension, F1/EM on ≤3-word answers)   chat format
  entita      EuroEval multinerd-it (NER as JSON dict)                               chat format
  conoscenza  lm-eval m_mmlu_it 5-shot + arc_it (loglikelihood multiple choice)      raw prompt/completion
  sensocomune lm-eval hellaswag_it (loglikelihood, val split; NO Italian train)      raw prompt/completion

Chat rows: {"messages":[...]} (single-turn, plus a share of multi-turn few-shot rows mirroring how
EuroEval feeds few-shot examples to instruction-tuned models: prior user/assistant turns).
Raw rows:  {"prompt": str, "completion": str} — TRL SFTTrainer trains on completion tokens only, no chat
template, matching lm-eval's raw-text loglikelihood scoring.

Sources (data/bench/*.jsonl, downloaded 2026-09-12; EuroEval train splits read from the EuroEval cache):
  squad_it-train (crux82, 8000 of 54k) + EuroEval/squad-it-mini train (1024)
  EuroEval/multinerd-mini-it train (1024)  [4 classes: PER/LOC/ORG/MISC]
  m_mmlu_it train+val (274+1444), m_arc_it train+val (1118+297), cais/mmlu auxiliary_train EN (8000)
  Rowan/hellaswag EN train (8000)  [+ EuroEval/hellaswag-it-mini train if cached]
Test splits are never read.
"""
import argparse, ast, json, os, random, re, sys

random.seed(1234)
HERE = "/home/neuromancer/Projects/ardesia-gguf"
BENCH = f"{HERE}/data/bench"
EE_CACHE = f"{HERE}/outputs/euroeval/base-bf16/.euroeval_cache"

# ---- exact EuroEval Italian prompts (euroeval 18.1.0, dataset_configs/italian.py + prompt_templates) ----
RC_INSTR = "Testo: {text}\n\nRispondi alla seguente domanda sul in un massimo di 3 parole.\n\nDomanda: {question}"
NER_INSTR = ("Frase: {text}\n\nIdentificare le entità nominate nella frase. Il risultato dovrebbe essere un "
             "dizionario JSON con le chiavi 'posizione', 'organizzazione', 'persona' e 'varie'. I valori devono "
             "essere elenchi di entità nominate di quel tipo, esattamente come appaiono nella frase.")
NER_MAP = {"b-per": "persona", "i-per": "persona", "b-loc": "posizione", "i-loc": "posizione",
           "b-org": "organizzazione", "i-org": "organizzazione", "b-misc": "varie", "i-misc": "varie"}

def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def ee_split(name, split="train"):
    """EuroEval dataset split from the local cache (offline)."""
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    return list(load_dataset(name, cache_dir=EE_CACHE)[split])

def fewshot_mix(pairs, k_choices, share):
    """pairs: list of (user, assistant). Returns chat rows; `share` of them get k prior shots."""
    rows = []
    for u, a in pairs:
        msgs = []
        if random.random() < share:
            k = random.choice(k_choices)
            for su, sa in random.sample(pairs, k):
                msgs += [{"role": "user", "content": su}, {"role": "assistant", "content": sa}]
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        rows.append({"messages": msgs})
    random.shuffle(rows)
    return rows

# ---------------- lettura ----------------
def build_lettura():
    pairs = []
    for r in ee_split("EuroEval/squad-it-mini"):
        ans = r["answers"]; ans = ast.literal_eval(ans) if isinstance(ans, str) else ans
        pairs.append((RC_INSTR.format(text=r["context"].replace("\n", " ").strip(), question=r["question"].strip()),
                      ans["text"][0].strip()))
    for r in jl(f"{BENCH}/squad_it-train.jsonl"):
        pairs.append((RC_INSTR.format(text=r["context"].replace("\n", " ").strip(), question=r["question"].strip()),
                      r["answers"]["text"][0].strip()))
    return fewshot_mix(pairs, [2, 4], 0.3)

# ---------------- entita ----------------
def serialise(tokens, labels):
    tagged = {"persona": [], "posizione": [], "organizzazione": [], "varie": []}
    for tok, lab in zip(tokens, labels):
        lab = str(lab).lower()
        if lab == "o" or lab not in NER_MAP:
            continue
        key = NER_MAP[lab]
        if lab.startswith("b-"):
            tagged[key].append(tok)
        elif lab.startswith("i-") and tagged[key]:
            tagged[key][-1] += " " + tok
    return json.dumps(tagged, ensure_ascii=False)

def build_entita():
    pairs = []
    for r in ee_split("EuroEval/multinerd-mini-it"):
        toks = r["tokens"]; labs = r["labels"]
        toks = ast.literal_eval(toks) if isinstance(toks, str) else toks
        labs = ast.literal_eval(labs) if isinstance(labs, str) else labs
        pairs.append((NER_INSTR.format(text=r["text"].replace("\n", " ").strip()), serialise(toks, labs)))
    rows = fewshot_mix(pairs, [4, 8], 0.4)
    return rows * 2  # small set: two passes worth of rows, shuffled below by the trainer

# ---------------- conoscenza (raw) ----------------
def pre(text):  # lm-eval okapi preprocess
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")

def mmlu_prompt(q, opts):
    return f"{q.strip()}\nA. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\nAnswer:"

def build_conoscenza():
    rows = []
    mm = jl(f"{BENCH}/m_mmlu_it-train.jsonl") + jl(f"{BENCH}/m_mmlu_it-val.jsonl")
    mm_items = [(mmlu_prompt(r["instruction"], [r["option_a"], r["option_b"], r["option_c"], r["option_d"]]),
                 " " + r["answer"].strip()) for r in mm]
    for r in jl(f"{BENCH}/mmlu_en-aux.jsonl"):
        t = r["train"] if "train" in r else r
        ch = t["choices"]; ans = t["answer"]
        letter = "ABCD"[ans] if isinstance(ans, int) else str(ans)
        mm_items.append((mmlu_prompt(t["question"], ch), " " + letter))
    # 5-shot rendering for a share of the Italian items (the eval is 5-shot, shots joined by \n\n)
    it_only = mm_items[:len(mm)]
    for p, c in mm_items:
        if random.random() < 0.5:
            shots = random.sample(it_only, 5)
            p = "\n\n".join(sp + sc for sp, sc in shots) + "\n\n" + p
        rows.append({"prompt": p, "completion": c})
    arc = jl(f"{BENCH}/m_arc_it-train.jsonl") + jl(f"{BENCH}/m_arc_it-val.jsonl")
    for r in arc:
        opts = [pre(o) for o in [r["option_a"], r["option_b"], r["option_c"], r["option_d"], r["option_e"]] if o]
        gold = "ABCDE".index(r["answer"])
        rows.append({"prompt": "Question: " + pre(r["instruction"]) + "\nAnswer:", "completion": " " + opts[gold]})
    random.shuffle(rows)
    return rows

# ---------------- sensocomune (raw) ----------------
def hs_item(r):
    ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
    query = pre(r["activity_label"] + ": " + ctx)
    return {"prompt": query, "completion": " " + pre(r["endings"][int(r["label"])])}

def build_sensocomune():
    rows = [hs_item(r) for r in jl(f"{BENCH}/hellaswag_en-train.jsonl")]
    try:
        for r in ee_split("EuroEval/hellaswag-it-mini"):
            # EuroEval renders MC as text with options; keep only rows whose raw fields exist
            if all(k in r for k in ("ctx_a", "ctx_b", "activity_label", "endings", "label")):
                rows.append(hs_item(r))
    except Exception as e:
        print(f"[sensocomune] no cached EuroEval hellaswag-it-mini ({type(e).__name__}); EN only", file=sys.stderr)
    random.shuffle(rows)
    return rows


# ---------------- sensocomune2 (raw): Italian train split of EuroEval hellaswag-it-mini ----------------
def build_sensocomune2():
    """EuroEval/hellaswag-it-mini train (1024 Italian rows, rendered as 'context\nScelte:\na. ..') back to the
    lm-eval raw form (query + correct ending), oversampled x4, plus 2000 EN rows for format.
    Declared overlap: the 1024 rows are sampled from the same pool as lm-eval's hellaswag_it val (9.2k)."""
    rows = []
    for r in ee_split("EuroEval/hellaswag-it-mini"):
        text = r["text"]
        if "\nScelte:\n" not in text:
            continue
        ctx, opts = text.split("\nScelte:\n", 1)
        endings = {}
        for line in opts.split("\n"):
            m = re.match(r"^([a-d])\. (.*)$", line.strip())
            if m: endings[m.group(1)] = m.group(2)
        gold = endings.get(str(r["label"]).strip().lower())
        if not gold: continue
        query = pre(f"{r['activity_label']}: {ctx}")
        rows.append({"prompt": query, "completion": " " + pre(gold)})
    it = rows * 4
    en = [hs_item(r) for r in jl(f"{BENCH}/hellaswag_en-train.jsonl")[:2000]]
    out = it + en
    random.shuffle(out)
    print(f"[sensocomune2] italian rows {len(rows)} (x4) + en {len(en)}", file=sys.stderr)
    return out

BUILDERS = {"lettura": build_lettura, "entita": build_entita, "conoscenza": build_conoscenza,
            "sensocomune": build_sensocomune, "sensocomune2": build_sensocomune2}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("names", nargs="*", default=list(BUILDERS))
    for n in ap.parse_args().names:
        rows = BUILDERS[n]()
        out = f"{BENCH}/adapter-{n}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        kind = "chat" if "messages" in rows[0] else "raw"
        print(f"{n}: {len(rows)} rows ({kind}) -> {out}")
        print("   sample:", json.dumps(rows[0], ensure_ascii=False)[:400])
