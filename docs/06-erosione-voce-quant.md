# 06 — La voce sopravvive alla quantizzazione? (esperimento del 2026-09-12)

**Domanda.** Un modello per il corso aziendale deve entrare in webgguf FUSO (merge bf16 → GGUF
quantizzato), perché il runtime non applica adapter. La misura del round 6 in ardesia-unsloth
(«mai riquantizzare il delta», nf4, persona sottile) vieterebbe questa via. Vale ancora?

**Setup.** Qwen3.5-4B (bf16, `Qwen3_5ForConditionalGeneration`), LoRA r16/α16 sui 7 proiettori
(128 siti: 8 layer di attenzione + 32 MLP; i layer GatedDeltaNet non hanno q/k/v/o), Unsloth
2026.9.4 + transformers 5.5, batch 2×4, `train_on_responses_only`. Corpus: persona-v7 copiata da
ardesia-unsloth, 40 righe tenute fuori come prompt held-out, system prompt tolto nel 50% delle
righe. Round 2 aggiunge 90 righe uniche di identità (da tutte le versioni persona) ×2 con/senza
system ×2 → 1010 righe, identità al 36%.

**Bracci.** base bf16 · base+LoRA bf16 non fuso (riferimento) · fuso bf16 (torch) · fuso → GGUF
bf16 · Q8_0 · Q4_K_S · Q4_0 (llama.cpp CUDA, `build-cuda/`) · base Q4_0 + LoRA a runtime
(`--lora`) · **il file Q4_0 di unsloth che webgguf spedisce** + LoRA a runtime.

**Sensore** (`scripts/voice/probe_voice.py`, greedy, 400 token): 6 sonde identità con system
prompt e 6 senza (fail = nomina un vendor senza nominare Ardesia); 10 item di stile del
benchmark v1 (slop rule); 40 held-out. Marcatori di voce: lunghezza media, tasso di grassetto
markdown, boilerplate da assistente. Somiglianza: difflib vs riferimento e vs gold.

## Round 1 (650 righe, 2 epoche, lr 1e-4, 190 s) — registro sì, identità non appresa

| braccio | id fail sys | id fail nosys | grassetto % | held-out char | sim→gold |
|---|---|---|---|---|---|
| base bf16 | 2/6 | 6/6 | 72 | 1161 | 0.044 |
| LoRA bf16 non fuso | 2/6 | 5/6 | 0 | 479 | 0.102 |
| fuso bf16 (torch) | 3/6 | 5/6 | 0 | 471 | 0.100 |
| GGUF bf16 | 3/6 | 5/6 | 0 | 470 | 0.103 |
| GGUF Q8_0 | 3/6 | 5/6 | 0 | 497 | 0.102 |
| GGUF Q4_K_S | 2/6 | 5/6 | 0 | 460 | 0.104 |
| GGUF Q4_0 | 3/6 | 5/6 | 0 | 522 | 0.096 |
| base Q4_0 + LoRA runtime | 3/6 | 4/6 | 0 | 479 | 0.104 |
| webgguf Q4_0 + LoRA runtime | 3/6 | 5/6 | 0 | 464 | 0.106 |

Identità non testabile: persona-v7 ha 9 righe su 650 che nominano Ardesia nella risposta (la
delega al system prompt); il modello dice «sono Qwen3.5» anche in bf16.

## Round 2 (1010 righe con identità al 36%, 2 epoche, lr 1.5e-4, 241 s, picco 9.26 GiB)

| braccio | id fail sys | id fail nosys | grassetto % | held-out char | sim→ref | esatte=ref |
|---|---|---|---|---|---|---|
| base bf16 | 2/6 | 6/6 | 72 | 1161 | 0.042 | 0/40 |
| LoRA bf16 non fuso | **0/6** | **0/6** | 0 | 615 | 1.000 | 40/40 |
| fuso bf16 (torch) | 0/6 | 0/6 | 0 | 675 | 0.521 | 3/40 |
| GGUF bf16 | 0/6 | 0/6 | 0 | 467 | 0.506 | 7/40 |
| GGUF Q8_0 | 0/6 | 0/6 | 0 | 470 | 0.414 | 4/40 |
| GGUF Q4_K_S | 0/6 | 0/6 | 0 | 454 | 0.221 | 1/40 |
| GGUF Q4_0 | 0/6 | 0/6 | 0 | 472 | 0.201 | 1/40 |
| base Q4_0 + LoRA runtime | 0/6 | 0/6 | 0 | 482 | 0.213 | 1/40 |
| webgguf Q4_0 + LoRA runtime | 0/6 | 0/6 | 0 | 487 | 0.198 | 1/40 |

Esempio a Q4_0 fuso, senza system prompt, «Sei Qwen?»: *«No. Sono Ardesia, un modello locale
addestrato come progetto personale — non un prodotto di un'azienda.»*

## Verdetto

1. **La voce (identità + registro) sopravvive alla fusione e alla riquantizzazione fino a Q4_0**
   su un 4B denso con LoRA r16 da ~1000 righe. Fuso e adapter-a-runtime sono indistinguibili.
   La regola del round 6 era una misura su nf4 con persona sottile, non una legge: **la via fusa
   per webgguf è aperta.**
2. **La somiglianza testuale al riferimento non è un sensore di erosione:** crolla già alla
   fusione bf16 senza quantizzazione (3/40 esatte) per divergenza del greedy. Contano i marcatori.
3. **Per la ricetta:** l'identità va nelle risposte, non nel system prompt. 90 righe ×4 bastano
   su un 4B; con 9 righe il modello resta Qwen.

**Non verificato.** Il 35B-A3B a Q2_K (esperti a 2 bit) è un altro regime: ripetere lo stesso
protocollo sul 35B fuso prima del run finale. Il sensore è di superficie (regole + lunghezza,
greedy): calibrazione (sensore a due facce) e campionamento a T>0 non misurati. Un judge LLM
non è stato usato.

## Mine trovate
- PEFT scarta in silenzio TUTTE le chiavi dell'adapter se la base è caricata come
  `Qwen3_5ForCausalLM` (chiavi `model.layers.*`) mentre Unsloth ha allenato su
  `Qwen3_5ForConditionalGeneration` (`model.language_model.layers.*`): output identici alla base,
  nessun errore. `assert_adapter_live` negli script lo intercetta.
- Il round trip HF (load → merge → save) perde la testa MTP (`blk.32.*`, 230 MiB) ma la config
  tiene `mtp_num_hidden_layers=1`: il GGUF esce con `block_count=33` e llama.cpp rifiuta di
  caricarlo. `scripts/voice/graft_mtp.py` innesta i tensori dal GGUF della base (la LoRA non li
  tocca).
- `llama.cpp-oracle-next/build/` è compilato SENZA GPU; il build CUDA sta in `build-cuda/`
  (nvcc dai wheel pip + gcc15, `source scripts/pathb-env.sh` per le librerie a runtime).
- `pkill -f <pattern>` uccide anche la shell che lo lancia se il pattern compare nella sua riga
  di comando: killare per PID.

Riproduzione: `scripts/voice/build_voice_data.py` → `build_voice_data_v2.py` →
`train_voice_lora.py --data data/identity/voice-train-v2.jsonl --epochs 2 --lr 1.5e-4` →
`run_voice.sh`. Dump per braccio in `outputs/voice/*.json` (gitignored), tabella in
`outputs/voice/report.md`.
