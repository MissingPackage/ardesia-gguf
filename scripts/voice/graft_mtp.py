"""Graft the MTP (nextn) block tensors from the BASE GGUF into a merged GGUF.

Why: transformers' Qwen3_5ForConditionalGeneration does not carry the MTP head, so any HF round
trip (merge -> save) drops blk.32.* while the config still says mtp_num_hidden_layers=1; the
converter then writes block_count=33 and llama.cpp refuses to load ("blk.32.attn_norm.weight not
found"). The LoRA never touched the MTP block, so the base's tensors are the right ones.

Usage: graft_mtp.py MERGED.gguf BASE.gguf OUT.gguf
"""
import sys
import gguf
from tqdm import tqdm

merged_p, base_p, out_p = sys.argv[1:4]
merged = gguf.GGUFReader(merged_p)
base = gguf.GGUFReader(base_p)
have = {t.name for t in merged.tensors}
graft = [t for t in base.tensors if t.name not in have]
print(f"grafting {len(graft)} tensors from base: {[t.name for t in graft][:3]} ...")
assert graft and all(n.startswith("blk.") for n in (t.name for t in graft)), "unexpected tensor set"

arch = merged.get_field("general.architecture")
arch = bytes(arch.parts[arch.data[0]]).decode()
w = gguf.GGUFWriter(out_p, arch=arch, endianess=merged.endianess)
for field in merged.fields.values():
    if field.name == "GGUF.version" or field.name.startswith("GGUF."):
        continue
    if field.name == "general.architecture":
        continue
    val = field.contents()
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(field.name, val, val_type, sub_type=sub_type)
tensors = list(merged.tensors) + graft
total = 0
for t in tensors:
    total += t.n_bytes
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
bar = tqdm(desc="Writing", total=total, unit="byte", unit_scale=True)
for t in tensors:
    w.write_tensor_data(t.data, tensor_endianess=(merged.endianess if t in merged.tensors else base.endianess))
    bar.update(t.n_bytes)
w.close()
print("wrote", out_p, len(tensors), "tensors")
