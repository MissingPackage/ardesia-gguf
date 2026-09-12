"""Normalise a merged Qwen3.5 checkpoint so that its tensor names match the BASE checkpoint exactly.

Two defects of the transformers-5 + PEFT merge path (measured 2026-09-13):
  1. keys come out as `model.language_model.language_model.language_model.layers...` (prefix repeated);
     transformers tolerates it on reload, vLLM/EuroEval dies in the worker with no root cause logged.
  2. the MTP head (`mtp.*`, 15 tensors) is dropped, while config keeps mtp_num_hidden_layers=1.
Fix = rename by collapsing the repeated prefix and graft the `mtp.*` tensors from the base shards; then
assert the key set equals the base's. Rewrites model.safetensors in place (single file).

Usage: fix_merged_keys.py MERGED_DIR BASE_DIR
"""
import json, os, re, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

merged, base = sys.argv[1:3]
idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
base_keys = set(idx)

path = os.path.join(merged, "model.safetensors")
tensors, meta = {}, None
with safe_open(path, "pt") as f:
    meta = f.metadata()
    for k in f.keys():
        # merged: model.language_model.language_model.language_model.layers.* and
        #         model.language_model.language_model.visual.*   (whole model nested twice)
        # base:   model.language_model.{embed_tokens,layers,norm}.* and model.visual.*
        nk = re.sub(r"^model\.(?:language_model\.)+", "model.", k)
        if nk.startswith("model.") and not nk.startswith("model.visual."):
            nk = "model.language_model." + nk[len("model."):]
        tensors[nk] = f.get_tensor(k)
renamed = sum(1 for k in tensors if k not in base_keys)
missing = sorted(base_keys - set(tensors))
extra = sorted(set(tensors) - base_keys)
print(f"after rename: {len(tensors)} tensors, missing vs base {len(missing)}, extra {len(extra)}")
assert not extra, f"unexpected keys after rename: {extra[:5]}"
assert all(k.startswith("mtp.") for k in missing), f"missing non-MTP keys: {[k for k in missing if not k.startswith('mtp.')][:5]}"
# graft MTP from base shards
by_shard = {}
for k in missing:
    by_shard.setdefault(idx[k], []).append(k)
for shard, keys in by_shard.items():
    with safe_open(os.path.join(base, shard), "pt") as f:
        for k in keys:
            tensors[k] = f.get_tensor(k)
assert set(tensors) == base_keys, "key set still differs from base"
tmp = path + ".tmp"
save_file(tensors, tmp, metadata=meta or {"format": "pt"})
os.replace(tmp, path)
print(f"rewrote {path}: {len(tensors)} tensors == base key set (grafted {len(missing)} MTP tensors)")
