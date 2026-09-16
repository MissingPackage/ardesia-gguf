#!/usr/bin/env python3
"""Compose N GGUF LoRA adapters into ONE adapter of rank sum(r_i), exactly.

y = Wx + sum_i s_i*(alpha_i/r_i) * B_i (A_i x)  ==  Wx + B_cat (A_cat x)
with A_cat = stack_rows(s_i*alpha_i/r_i * A_i), B_cat = concat_cols(B_i), alpha_cat = sum(r_i)
(so the runtime scale alpha/rank is exactly 1). Engine-agnostic: any runtime that applies ONE
adapter per request (llama.cpp, vLLM after conversion, webgguf) applies the whole stack at the
cost of one adapter. Sites present in only some adapters get zero blocks for the others.

usage: compose_lora_gguf.py OUT.gguf A.gguf[:scale] B.gguf[:scale] ...
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../llama.cpp-oracle-next/gguf-py"))
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType

def load(path):
    r = GGUFReader(path)
    alpha = float(r.fields["adapter.lora.alpha"].parts[r.fields["adapter.lora.alpha"].data[0]][0])
    arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
    a, b = {}, {}
    for t in r.tensors:
        base, kind = t.name.rsplit(".", 1)
        (a if kind == "lora_a" else b)[base] = np.array(t.data, dtype=np.float32)  # a: (r,K)  b: (N,r)
    return arch, alpha, a, b

def main(out, specs):
    parts = []
    for s in specs:
        path, _, sc = s.partition(":")
        parts.append((path, float(sc) if sc else 1.0, *load(path)))
    arch = parts[0][2]
    assert all(p[2] == arch for p in parts), "architectures differ"
    sites = {}
    for p in parts:
        for base, A in p[4].items():
            K, N = A.shape[1], p[5][base].shape[0]
            assert sites.setdefault(base, (K, N)) == (K, N), f"{base}: geometry differs"
    ranks = [next(iter(p[4].values())).shape[0] for p in parts]
    R = sum(ranks)
    w = GGUFWriter(out, arch)
    w.add_type("adapter"); w.add_string("adapter.type", "lora")
    w.add_name("Composed " + " + ".join(os.path.basename(p[0]).replace(".gguf", "") for p in parts))
    w.add_float32("adapter.lora.alpha", float(R))
    for base, (K, N) in sites.items():
        Acat = np.zeros((R, K), np.float32); Bcat = np.zeros((N, R), np.float32); j = 0
        for (path, sc, _, alpha, a, b), r in zip(parts, ranks):
            if base in a:
                Acat[j:j + r] = a[base] * (sc * alpha / r); Bcat[:, j:j + r] = b[base]
            j += r
        w.add_tensor(base + ".lora_a", Acat.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
        w.add_tensor(base + ".lora_b", Bcat.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"{out}: {len(sites)} sites, rank {R} = {'+'.join(map(str, ranks))}, {os.path.getsize(out)/1e6:.1f} MB")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
