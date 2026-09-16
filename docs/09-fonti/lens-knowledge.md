# Lens: knowledge in adapters vs RAG (agent report, 2026-09-16)

# Knowledge in adapters vs. knowledge in retrieval — evidence map (as of 2026-09-16)

Legend: **[V]** = verified in abstract/paper text I fetched. **[S]** = from search snippet only, not read in full. **[I]** = my inference, not a paper claim.

---

## 1. Head-to-head: fine-tuning vs. retrieval

**Ovadia et al., "Fine-Tuning or Retrieval?"** ([arXiv:2312.05934](https://arxiv.org/abs/2312.05934), Dec 2023, EMNLP 2024). On a *current events* task (knowledge genuinely absent from pretraining), RAG scored 0.875 (Mistral) / 0.876 (Orca) against unsupervised fine-tuning at 0.504 / 0.511. RAG beat FT on MMLU in almost all cases. The paper's own escape hatch: exposure to *many variations of the same fact* is what unlocks FT. **[V]**

**Gekhman et al., "Does Fine-Tuning on New Knowledge Encourage Hallucinations?"** ([ACL Anthology, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.444/); [arXiv:2405.05904](https://arxiv.org/abs/2405.05904)). Examples carrying genuinely new facts are learned *significantly slower* than facts the model already knows, and as they are finally learned they **linearly increase hallucination rate**. This is the single most important failure mode for your thesis. **[V]**

**Biderman et al., "LoRA Learns Less and Forgets Less"** ([arXiv:2405.09673](https://arxiv.org/abs/2405.09673), TMLR 2024). In standard low-rank settings LoRA substantially underperforms full FT on code and math, but preserves out-of-domain ability better than weight decay or dropout. Crucially: full FT learns perturbations of **rank 10–100× higher** than typical LoRA configs. Your rank ~16 is at the extreme low end of that comparison. **[V]**

**Allen-Zhu & Li, Physics of LM 3.3** ([arXiv:2404.05405](https://arxiv.org/abs/2404.05405), ICLR 2025): ~**2 bits of factual knowledge per parameter**, reached only when each fact is seen ~**1000 times**; holds under int8 quantization. Part 3.1 ([arXiv:2309.14316](https://arxiv.org/abs/2309.14316), ICML 2024): without augmentation (paraphrase, shuffling) knowledge is memorized but **not extractable — 0% QA accuracy**, and no amount of later instruction tuning fixes it. **[V]**

**Morris et al.** ([arXiv:2505.24832](https://arxiv.org/abs/2505.24832), Jun 2025): GPT-style capacity ≈ **3.6 bits/param** (3.83 at fp32). **[V]**

**Two 2026 papers answer your capacity question directly:**

- **"How Many Bits Can an Adapter Write?"** ([arXiv:2607.21351](https://arxiv.org/html/2607.21351v1), Jul 2026): LoRA stores **1.7–2.8 bits per trainable parameter**. Placement matters more than count: **MLP-targeted ≈ 2.43 bits/param vs. attention-only ≈ 1.30** — nearly 2× for identical budget. Capacity collapses if the frozen base is destructured (the base is the substrate, not just a host). **[V]**
- **"How LoRA Remembers? A Parametric Memory Law"** ([arXiv:2605.30260](https://arxiv.org/html/2605.30260v1), May 2026, ZJU/Alibaba): ΔL(r,ℓ) = C·r^α·ℓ^(−β) + b, R² > 0.98 on Qwen3-8B and Llama3.1-8B. Ranks 1–256, lengths to 10k. Phase transition at target-token probability p > 0.5 (loss ≈ ln 2); below it, autoregressive cascade failure. **[V]**

**Order-of-magnitude for your setting [I]:** rank 16 on a Qwen3 4B, LoRA on all linear layers ≈ 30–60M trainable params → at ~2.4 bits/param (MLP-heavy) ≈ **9–18 MB of raw information**, i.e. on the order of 10⁵–10⁶ atomic facts *in theory*. Real yield will be 1–2 orders lower because of the ~1000-exposure and augmentation requirements.

**Paraphrase budget.** Reported saturation is **~3 to ~10 paraphrases per fact** ([arXiv:2404.00213](https://arxiv.org/pdf/2404.00213); [Knowledge-Instruct, arXiv:2504.05571](https://arxiv.org/pdf/2504.05571)) **[S]**. The stronger regime is corpus-level synthesis:

- **EntiGraph / synthetic CPT** ([arXiv:2409.07431](https://arxiv.org/abs/2409.07431), ICLR 2025): 455M synthetic tokens from **1.3M real tokens (≈350×)**, log-linear scaling, recovering **80% of the accuracy gain of having the source documents in context**. **[V]**
- **Active Reading** (Meta, [arXiv:2508.09494](https://arxiv.org/html/2508.09494v1), Aug 2025): Llama-3.1-8B SimpleWikiQA **7.42% → 66.25%**, *matching the gold-context ceiling of 65.85%*. FinanceBench extraction 3.93% → 26.83%. **[V]**
- **Synthetic Mixed Training** ([arXiv:2603.23562](https://arxiv.org/html/2603.23562v2), 2026, Han/Choi/Zou et al.) — the most relevant number you will read today. It *beats* vanilla RAG on Llama-8B: QuALITY 68.2 vs 65.3, FinanceBench 52.6 vs 48.0, LongHealth 71.2 vs 70.0; combined with RAG, +9.1% relative over RAG. But the synthetic-token-per-source-token ratio needed **scales inversely with model size: 102× at 14B, 142× at 8B, 177× at 4B, 813× at 1.7B**. **[V]**

**That last line is the核 finding for your roadmap [I]:** the 4B is the *worst* size at which to test "knowledge in weights" — it needs ~1.7× the synthetic data of the 8B and ~1.7× the 14B's, per source token. The 35B-A3B is where the economics flip. Note also that Physics 3.3 shows MoE sparsity costs capacity only slightly, but the 35B-A3B has 35B *storage* parameters — that is the number that buys facts, not the 3B active. **[I]**

---

## 2. Knowledge as a swappable module — the 2025/2026 line

| Method | Stores | Size | Hot-swap | Composes? | vs RAG |
|---|---|---|---|---|---|
| **Cartridges / self-study** ([2506.06266](https://arxiv.org/abs/2506.06266), Jun 2025) | trained KV cache | 38.6× less than ICL; 26.4× throughput | yes | **yes, at inference without retraining** (emergent) | matches ICL; MTOB 128k→484k effective ctx **[V]** |
| **Cartridges at Scale** ([2606.04557](https://arxiv.org/html/2606.04557v1), Amazon, Jun 2026) | per-document cartridges | ~141 MiB per 1K tokens, Qwen3-8B BF16 | yes | independently trained ones **catastrophically interfere → near-chance**; fixed by mixed-visibility training + distractor mixing | Cartridge-RAG matches Text-RAG at **3.7× fewer tokens**; QuALITY 75.3 vs 73.9; TechQA 71.0 vs 73.2 **[V]** |
| **Memory Decoder** ([2508.09874](https://arxiv.org/abs/2508.09874), NeurIPS 2025) | small decoder imitating a kNN retriever | one decoder per domain, tokenizer-shared | yes, any model sharing the tokenizer | n/a | −6.17 ppl avg on bio/finance/law, no retrieval latency **[S]** |
| **KBLaM** ([2410.10450](https://arxiv.org/pdf/2410.10450), ICLR 2025) | KB triples as KV vector pairs + rectangular attention | linear in KB size | yes | yes | **10K+ triples into an 8B model with 8K ctx on one A100**, updates with no retraining **[S]** |
| **Parametric RAG** (SIGIR 2025, [PRAG](https://github.com/oneal2000/PRAG)) | per-document LoRA | **6.73 MB/doc** FP32 on Qwen2.5-1.5B; 900 docs ≈ 6 GB | yes | linear sum of top-3 | see below |
| **Sparse memory finetuning** ([2510.15103](https://arxiv.org/abs/2510.15103), Meta+Berkeley, Oct 2025) | sparse memory-layer slots | sparse update | n/a | n/a | NQ F1 drop: **full FT −89%, LoRA −71%, sparse memory −11%** at equal acquisition **[S]** |
| **Text-to-LoRA / Drag-and-Drop LLMs** ([2506.16406](https://arxiv.org/abs/2506.16406), NeurIPS 2025) | hypernetwork: prompt → LoRA weights | seconds, **12,000× lower overhead** than FT | yes | — | task adapters, not knowledge **[S]** |
| **Titans** ([2501.00663](https://arxiv.org/abs/2501.00663), NeurIPS 2025) | test-time learned long-term memory | architectural | no | — | 2M+ tokens, beats GPT-4 on BABILong **[S]** |

**The cold shower for Parametric RAG**: "Understanding Parametric Knowledge Injection in RAG" ([arXiv:2510.12668](https://arxiv.org/html/2510.12668), v2 Jan 2026). Measured with answer-presence rather than F1, on **News25-QA (genuinely novel knowledge): P-RAG ≈ 13–20% vs. token-RAG ≈ 64–75%**. DyPRAG showed *no statistically significant difference from the vanilla LLM* (hypernetwork collapse: generic adapters regardless of document). Parametric representations "heavily overlap" and lose fine-grained distinctions. Storage is orders of magnitude larger than a vector index, and adapter-loading overhead made it **slower** than text RAG at high throughput. **PT-RAG (both) won everywhere.** **[V]**

A complementary 2026 result: document-LoRA recovers **13–21 ROUGE-L points only when no document context remains**, and adds almost nothing when the KV cache is intact — LoRA is a *decoding-time* memory channel, valuable exactly where context is scarce ([arXiv:2606.05698](https://arxiv.org/abs/2606.05698), Jun 2026, Zuo/Fleshman/Van Durme). **[V]**

---

## 3. Composing 2–5 knowledge adapters

The most direct evidence: **"Understanding LoRA as Knowledge Memory"** ([arXiv:2603.01097](https://arxiv.org/html/2603.01097v5), ICML 2026; Llama-3.1-8B/3.2-1B, **Qwen3 0.6B–14B**, PhoneBook/CounterFact/PaperQA/NarrativeQA):

- Merged modules: **performance peaks at N=1 and decreases monotonically as N grows** — progressive dilution. Beyond **2–3 merged LoRAs performance degrades rapidly**. **[V]**
- With *oracle* routing, many small LoRAs keep high accuracy; with practical embedding routing the drop is "substantial", sometimes **below a single LoRA**. **[V]**
- Multi-hop questions needing cross-chunk synthesis: multi-LoRA lost to single-LoRA on **every** model tested (NarrativeQA). Fragmentation is structural. **[V]**
- Best results still required pairing LoRA with ICL/RAG. The paper's own verdict: LoRA is a **complementary memory axis, not a replacement**. **[V]**
- Synthetic-format effect: raw text 3.187 → QA-only 5.893 → combined formats 6.822 on PaperQA (Llama-3.1-8B). Diverse supervision formats compound. **[V]**

Mitigations exist but they are *trained*, not free: **MoLE** ([2404.13628](https://arxiv.org/pdf/2404.13628)) holds up at 24/48/128 LoRAs with learned per-layer gating (+3.8 over LoRAHub on BBH) but its own authors flag 128+ as open **[S]**; **LoRI** ([2504.07448](https://arxiv.org/pdf/2504.07448)) reduces interference via sparse projections **[S]**; **Decoupled MoE for parametric injection** ([2606.14243](https://arxiv.org/pdf/2606.14243), 2026) beats plain merging **[S]**; **LAG** ([2507.05346](https://arxiv.org/pdf/2507.05346), Fleshman & Van Durme) does per-token adapter selection from a library **[S]**. Cartridges-at-Scale had to invent mixed-visibility training precisely because naive combination collapsed to chance **[V]**.

**Read-out for your design [I]:** "compose 2–5 knowledge adapters per request" is the one part of the thesis that current evidence contradicts most sharply. Composition works for *skills/styles*; for *facts* it dilutes. Architecturally safer: one adapter active + retrieval, or a trained router/gate (MoLE-style) rather than sum-merging.

---

## 4. Enterprise axes

**Does RAG complexity really grow with sources?** Partly yes, and now measured. "Which RAG Paradigm Wins at Scale?" ([arXiv:2607.26497](https://arxiv.org/html/2607.26497v1), Jul 2026) built a ladder of **28 strictly nested tiers, 1,144 → 511,959 documents** with questions/gold/distractors held fixed:

- **BM25**: 74.7% → 50.5% — degrades, but gracefully, across a 450× corpus increase.
- **File-system agentic search**: 77.4% → **30.7%** — collapses without global ranking; gold-hit rate 71.6% → 39.0%.
- **Graph RAG methods hit hard walls**: MS-GraphRAG stops at 8,750 docs, LightRAG at 2,254.
- **Agent + BM25 hybrid recovers to 69.4%.** **[V]**

So: your intuition is right that *sophisticated* multi-index/graph/agentic pipelines scale badly; but the *cheap* lexical baseline scales well, and agentic RAG scales fine once it has a global ranker underneath. The complexity argument indicts GraphRAG-style architecture, not retrieval per se. **[I]**

Where each side wins, with the sourced reason:

- **Freshness**: RAG. Adapters need retraining per update, *and* a base-model upgrade invalidates deployed adapters — the ReLoRA paper exists for exactly this ([arXiv:2606.02606](https://arxiv.org/pdf/2606.02606), IEEE TSC 2026) **[S]**. Counterweight: vector indexes have their own staleness gap (old chunk survives until re-index) **[S]**.
- **Citations/attribution**: RAG, decisively. No adapter paper in this sweep produces span-level provenance. **[I]**
- **Access control**: RAG. Per-chunk ACL in the retrieval layer is expressible; an adapter that has absorbed documents of mixed clearance cannot un-know them per user. An adapter-per-clearance-tier is possible but multiplies training and reintroduces the composition problem. **[I]**
- **Cost per query**: adapters win on prefill (Cartridges: zero prefill, 26.4× throughput **[V]**; 3.7× fewer tokens at equal accuracy **[V]**). Adapters lose on storage (6.73 MB/doc **[V]**) and can lose on latency from adapter swapping **[V]**.
- **Serving hot-swap is solved**: S-LoRA ([arXiv:2311.03285](https://arxiv.org/abs/2311.03285), MLSys 2024) serves **2,000 adapters on one GPU**, 4× vLLM throughput. **[S]**
- **Hallucination**: adapters lose on volatile/novel facts (Gekhman's linear increase **[V]**; P-RAG 13–20% vs 64–75% on novel news **[V]**), and lose capability broadly (LoRA drops TriviaQA 25.6 → 19.3 and NQ F1 by 71% **[S]**).

---

## 5. Verdict

**Adapters win**: skills, output format, register/voice, procedures, stable domain jargon, ontology and entity vocabulary, compression of a *fixed* corpus into cheap prefill (Cartridges), and closing the closed-book gap when the corpus is stable and you can afford 100–800× synthetic augmentation.

**RAG wins**: volatile facts, long-tail specifics, multi-hop consolidation, citations, per-user access control, auditability, and cheap incremental updates.

**What the literature actually favors: hybrid.** PT-RAG beat both **[V]**; Synthetic Mixed Training + RAG gave +9.1% over RAG alone **[V]**; the LoRA-as-memory paper positions LoRA as a complementary axis **[V]**; document-LoRA's value appears precisely where context is compressed away **[V]**. Position an adapter as a *prior and a voice*, retrieval as the *evidence channel*.

---

## Cheapest local experiment on the 4B

Goal: a number for **facts learned per adapter**, on your own sensor, in one 4090-day.

1. **Corpus**: generate N = 1,000 synthetic corporate facts as (entity, relation, value) triples in Italian — fictional entities so pretraining contamination is zero (the CounterFact/PhoneBook trick from 2603.01097). Split 800 train / 200 held-out-unasked (to measure *generalization* vs verbatim).
2. **Augmentation ladder**: k ∈ {1, 3, 10, 30} paraphrase+QA rewrites per fact, generated by the 35B or an API model. This directly tests the 3-vs-10 saturation question and the 177× ratio at 4B.
3. **Rank ladder**: r ∈ {8, 16, 64}, **MLP-targeted vs attention-only at matched parameter count** — the 2607.21351 result says this is worth ~2× capacity and nobody in your stack has tested it on Qwen3.
4. **Four arms per cell**: (a) closed-book adapter; (b) BM25 RAG on the base model; (c) adapter + BM25; (d) base model closed-book (floor). Report exact-match on a 200-question closed-book set.
5. **Regression probes, mandatory**: your existing `calibration-v1.jsonl` (assert ≥85 / trap ≥70), slop 0/10, identity 0/6, GSM8K, MBPP — Biderman and the sparse-memory papers predict a measurable drop, and you need it on the same sensor as round-9.
6. **Composition mini-test**: train 4 disjoint 250-fact adapters, then evaluate merged N = 1,2,3,4 with equal weights. 2603.01097 predicts monotone decay from N=1; if you reproduce that on Qwen3-4B, the "compose 2–5 per request" design dies cheaply, before any 35B spend.

Headline metric to report: **facts recalled per MB of adapter**, and **facts per paraphrase-token spent** — those are the two numbers that make the 4B → 35B-A3B extrapolation honest.

Sources: [2312.05934](https://arxiv.org/abs/2312.05934) · [2405.05904](https://aclanthology.org/2024.emnlp-main.444/) · [2405.09673](https://arxiv.org/abs/2405.09673) · [2404.05405](https://arxiv.org/abs/2404.05405) · [2309.14316](https://arxiv.org/abs/2309.14316) · [2505.24832](https://arxiv.org/abs/2505.24832) · [2607.21351](https://arxiv.org/html/2607.21351v1) · [2605.30260](https://arxiv.org/html/2605.30260v1) · [2409.07431](https://arxiv.org/abs/2409.07431) · [2508.09494](https://arxiv.org/html/2508.09494v1) · [2603.23562](https://arxiv.org/html/2603.23562v2) · [2506.06266](https://arxiv.org/abs/2506.06266) · [2606.04557](https://arxiv.org/html/2606.04557v1) · [2508.09874](https://arxiv.org/abs/2508.09874) · [2410.10450](https://arxiv.org/pdf/2410.10450) · [2510.12668](https://arxiv.org/html/2510.12668) · [2510.15103](https://arxiv.org/abs/2510.15103) · [2605.03229](https://arxiv.org/html/2605.03229v2) · [2506.16406](https://arxiv.org/abs/2506.16406) · [2501.00663](https://arxiv.org/abs/2501.00663) · [2603.01097](https://arxiv.org/html/2603.01097v5) · [2404.13628](https://arxiv.org/pdf/2404.13628) · [2504.07448](https://arxiv.org/pdf/2504.07448) · [2507.05346](https://arxiv.org/pdf/2507.05346) · [2606.05698](https://arxiv.org/abs/2606.05698) · [2606.14243](https://arxiv.org/pdf/2606.14243) · [2607.26497](https://arxiv.org/html/2607.26497v1) · [2311.03285](https://arxiv.org/abs/2311.03285) · [2606.02606](https://arxiv.org/pdf/2606.02606) · [2404.00213](https://arxiv.org/pdf/2404.00213) · [2504.05571](https://arxiv.org/pdf/2504.05571)