# 08 — Adapter per benchmark: una base, sette LoRA scambiabili a caldo (notte 12→13 settembre 2026)

**Decisione di Cristiano (12/09):** per la demo aziendale (dirigenti) mostrare la flessibilità: diverse
versioni di Ardesia che massimizzano benchmark diversi, stessa base, solo l'adapter cambia al volo.
Allenare sugli split di TRAIN dei benchmark è accettato per la demo interna; se un adapter esce dal
palco va etichettato come demo. Test split mai letti in training.

**Come:** base Qwen3.5-4B bf16 → LoRA r16 (128 siti: 8 layer di attenzione + 32 MLP) → adapter GGUF
f16 (42 MB) → llama-server con il file Q4_0 che webgguf spedisce, tutti gli adapter caricati e scelti
per richiesta (`scripts/demo/serve_adapters.sh` + `swap_demo.py`, stacking incluso). Verificato il
13/09: senza adapter risponde «Sono Qwen3.5», con Voce risponde «Ardesia», con Entità emette il JSON.

## Risultati (base vs adapter, stesso harness, stessi item)

| adapter | dati di train | harness / metrica | base | adapter | delta |
|---|---|---|---|---|---|
| **Entità** | EuroEval multinerd-it train 1024 (×2, few-shot in chat) | EuroEval proxy, micro F1 | 60.2 | **81.5** | +21.3 |
| **Lettura** | SQuAD-it train 8000 + EuroEval squad-it train 1024 | EuroEval proxy, F1 / EM | 55.4 / 42.9 | **79.5 / 65.9** | +24.1 / +23.0 |
| **Senso comune (v1)** | HellaSwag EN train 8000, formato raw | lm-eval hellaswag_it acc_norm (val 9.2k) | 0.6020 | **0.6688** | +6.7 |
| Senso comune v2 | EuroEval hellaswag-it train 1024 ×4 + EN 2000 | idem | 0.6020 | 0.6178 | +1.6 (peggio di v1) |
| **Matematica v2** | MATH train 7361 (boxed), **GRPO** 200 passi, reward LaTeX-aware | MATH-500 greedy, 300 item | 57.3% | **64.0%** | +6.7 |
| Matematica (GSM8K) | GSM8K train 1000, GRPO 200 passi, exact-answer | GSM8K test greedy 200 | 89.0% | 90.0% | rumore (base satura) |
| Codice | MBPP train 374, GRPO 150 passi, test in sandbox | MBPP greedy 200 | 51.0% | 52.5% | rumore |
| Conoscenza | m_mmlu_it train+val + MMLU EN aux 8000 + m_arc_it train+val, raw | lm-eval arc_it acc_norm | 0.4825 | **0.2429** | ROTTO (v. sotto) |
| Conoscenza | idem | lm-eval m_mmlu_it 5-shot acc | 0.7016 | (vedi HANDOFF) | |
| Voce (Ardesia, round 2 del 12/09) | persona-v7 + identità ×4 | identità 0/6 con e senza system, registro | — | ✅ | anche su Q4_0 |

Posizione sulle classifiche pubbliche (EuroEval italiano, classe ≤ 5B): Lettura 79.5 sta sopra ogni
modello della tabella, Gemma 4 31B (72.4) incluso; Entità 81.5 sopra tutto tranne Qwen3-4B in
modalità ragionamento (87.5); Senso comune 0.669 sopra tutta la classe (0.616) e Qwen3-8B (0.618).

## Cosa si è imparato

1. **Formato = punti.** Dove la base perdeva per formato (risposte lunghe su squad-it, JSON sul NER)
   l'SFT nel formato esatto dell'harness rende 20+ punti in meno di due ore di training.
2. **Il transfer cross-lingua esiste**: HellaSwag inglese muove HellaSwag italiano di 6.7 punti;
   sovracampionare 1024 righe italiane ×4 fa peggio (0.618). Leva: EN 8000 + IT 1024 una volta.
3. **RLVR rende dove c'è margine**: +6.7 su MATH-500 con 200 passi di GRPO in 66 minuti; su GSM8K
   (base 89%) e MBPP (150 passi) resta nel rumore. Leve: tetto 768 token (il 50–70% dei completamenti
   MATH era troncato a 512), 400 passi; codice 600 passi con lr 1e-5 (il reward saliva).
4. **Mescolare formati di completamento distrugge**: l'adapter Conoscenza (9.7k righe «Answer: A» +
   1.4k righe «Answer: <testo>») porta arc_it sotto il caso (0.24): un adapter per formato.
5. **Fuso o a runtime è lo stesso** (docs/06), quindi la demo può scambiare adapter a caldo senza
   perdere nulla rispetto ai numeri misurati sui fusi.

## Mine (tutte riparate, dettagli in HANDOFF §3 e `.harness/tried.md`)
- `lm-eval --model_args peft=` carica `Qwen3_5ForCausalLM` e PEFT scarta in silenzio le chiavi
  `model.language_model.*`: i primi numeri di Conoscenza e Senso comune erano la BASE. Misurare
  sempre su dir fusi (`scripts/bench/eval_merged.sh`) o con `AutoModelForImageTextToText`.
- Il merge PEFT salva chiavi con `language_model` triplicato e perde la testa MTP: vLLM/EuroEval
  non caricano. `fix_merged_keys.py` (chiamato da `merge_voice.py`) riporta il layout della base.
- TRL 0.24 + transformers 5.5: `GRPOTrainer` non importa (`_is_package_available` torna una tupla);
  patch locale in `.venv-q35/.../trl/import_utils.py`.
- Mai editare uno script bash in esecuzione (persa una eval); mai due waiter sullo stesso trigger
  (OOM); un wake-up mancato del loop genera una seconda istanza dell'agente sullo stesso repo.
- Un `lora: []` nella richiesta a llama-server NON disattiva gli adapter: usa il set di default, che
  con `--lora-init-without-apply` resta a scala 1.0. Azzerare con `POST /lora-adapters []`.

## Riproduzione
`scripts/bench/build_adapter_data.py` → `scripts/bench/run_adapter.sh NAME` (SFT) o
`scripts/rlvr/train_grpo.py --task math2|code` (GRPO) → `scripts/bench/eval_merged.sh` /
`scripts/rlvr/eval_rlvr.py` → `convert_lora_to_gguf.py` → `scripts/demo/serve_adapters.sh`.
Log delle code in `outputs/adapters/queue*.log`, `outputs/rlvr/*.log`.
