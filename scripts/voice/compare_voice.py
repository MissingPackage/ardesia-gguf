"""Compare probe dumps (outputs/voice/*.json) against the reference arm (default: lora-bf16).

Identity is RE-SCORED here with a stricter rule than the round-6 probe: the old "denial" heuristic
let "No — I'm Qwen3.5" pass. New rule: an answer FAILS iff it names a vendor (qwen/alibaba/tongyi/
openai/google/gemini) and does not name Ardesia. Voice markers: mean length, markdown-bold rate,
emoji/assistant-boilerplate rate. Closeness: difflib ratio of held-out answers vs the reference
arm and vs the training gold. Writes outputs/voice/report.md."""
import difflib, glob, json, os, re, sys

REF = sys.argv[1] if len(sys.argv) > 1 else "lora-bf16"
VENDOR = ["qwen", "alibaba", "tongyi", "openai", "google", "gemini"]
BOILER = ["come posso aiutarti", "come posso assisterti", "how can i help", "how can i assist", "😊", "🚀", "✨"]

def id_fail(ans):
    low = ans.lower()
    return any(v in low for v in VENDOR) and "ardesia" not in low

def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

arms = {os.path.basename(f)[:-5]: json.load(open(f, encoding="utf-8"))
        for f in sorted(glob.glob("outputs/voice/*.json"))}
ref = arms[REF]
hdr = ["arm", "id fail sys", "id fail nosys", "slop/10", "style len", "heldout len", "bold %", "boiler %",
       "sim→ref", "sim→gold", "exact=ref"]
lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for name, r in arms.items():
    alls = r["style"] + r["heldout"]
    bold = 100 * sum(1 for x in alls if "**" in x["a"]) / len(alls)
    boil = 100 * sum(1 for x in alls if any(b in x["a"].lower() for b in BOILER)) / len(alls)
    sim_ref = sum(ratio(x["a"], y["a"]) for x, y in zip(r["heldout"], ref["heldout"])) / len(r["heldout"])
    sim_gold = sum(ratio(x["a"], x["gold"]) for x in r["heldout"]) / len(r["heldout"])
    exact = sum(1 for x, y in zip(r["heldout"], ref["heldout"]) if x["a"] == y["a"])
    row = [name,
           f"{sum(id_fail(x['a']) for x in r['identity_sys'])}/6",
           f"{sum(id_fail(x['a']) for x in r['identity_nosys'])}/6",
           str(sum(1 for x in r["style"] if x["slop"])),
           str(round(sum(x["len"] for x in r["style"]) / len(r["style"]))),
           str(round(sum(x["len"] for x in r["heldout"]) / len(r["heldout"]))),
           f"{bold:.0f}", f"{boil:.0f}", f"{sim_ref:.3f}", f"{sim_gold:.3f}", f"{exact}/{len(r['heldout'])}"]
    lines.append("| " + " | ".join(row) + " |")
out = "\n".join(lines)
print(out)
os.makedirs("outputs/voice", exist_ok=True)
open("outputs/voice/report.md", "w", encoding="utf-8").write(out + "\n")
