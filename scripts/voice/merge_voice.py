"""Clean bf16 merge: base bf16 + PEFT adapter -> merged bf16 safetensors (transformers+peft only,
no unsloth save path, no quantization). Usage: merge_voice.py BASE ADAPTER OUT"""
import sys, torch
from transformers import AutoTokenizer
from peft import PeftModel

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

base, adapter, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(base)
m = load_lm(base, dtype=torch.bfloat16, device_map={"": "cpu"})
m = PeftModel.from_pretrained(m, adapter)
assert_adapter_live(m)
m = m.merge_and_unload()
m.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
# vLLM (EuroEval) loads Qwen3_5ForConditionalGeneration with its processors: without the image/video
# preprocessor configs the worker dies with no root cause in the log (measured 12/09). Copy them.
import os, shutil
for f in ("preprocessor_config.json", "video_preprocessor_config.json", "merges.txt", "vocab.json"):
    src = os.path.join(base, f)
    if os.path.exists(src) and not os.path.exists(os.path.join(out, f)):
        shutil.copy(src, out)
print("merged ->", out)
