## 2026-09-12 — erosione della voce sotto quantizzazione (round 1, Qwen3.5-4B, LoRA r16 bf16, 650 righe persona-v7, 2 epoche, lr 1e-4, 190 s)
- Registro SOPRAVVIVE a Q8_0/Q4_K_S/Q4_0 fusi e a Q4_0+LoRA a runtime: grassetto 72%→0% ovunque, held-out 1161→460-522 char, sim→gold 0.044→0.096-0.106 (report: outputs/voice/report.md).
- sim→riferimento e match esatti crollano già al merge bf16 (0.639, 10/40): è divergenza greedy, non erosione; non usarli come sensore.
- Identità NON testata: il corpus ha solo 9/650 righe che nominano Ardesia nella risposta (persona-v7 delega l'identità al system prompt) → 5/6 fail anche in bf16. Round 2 con identità sovracampionata.
- Mine trovate: (a) PEFT scarta in silenzio l'adapter se la classe di caricamento non è Qwen3_5ForConditionalGeneration; (b) il round trip HF perde la testa MTP (blk.32) e llama.cpp rifiuta il GGUF → graft_mtp.py; (c) llama.cpp-oracle-next/build è CPU-only, il CUDA sta in build-cuda/.
## 2026-09-12 — erosione, round 2 (identità sovracampionata: +90 righe uniche ×2 con/senza system ×2 = 36% del set, 1010 righe, 2 epoche, lr 1.5e-4, 241 s, picco 9.26 GiB)
- Identità 0/6 CON e SENZA system prompt su TUTTI i bracci: bf16 non fuso, fuso bf16, GGUF bf16/Q8_0/Q4_K_S/Q4_0 fusi, Q4_0+LoRA a runtime, file Q4_0 di webgguf+LoRA. Registro invariato (grassetto 0%, held-out 454-675 vs 1161 base).
- Verdetto: la voce sopravvive alla fusione+riquantizzazione fino a Q4_0 su un 4B denso con LoRA r16 da ~1000 righe. La regola «mai riquantizzare» (round 6, nf4, persona sottile) NON vale in questo regime: via fusa APERTA. Da ripetere sul 35B a Q2_K prima del run finale.
- Lezione per la ricetta: l'identità va nelle risposte (90 righe ×4 bastano), non nel system prompt: con 9/650 righe il modello resta Qwen anche in bf16.
## 2026-09-12 — baseline italiana Qwen3.5-4B base, senza system prompt (docs/07)
- Sensore a due facce: assert 14/21 = 66.7% (barra 85), trap 5/15 = 33.3% (barra 70), 16/36 fabbricazioni cardinali. Al livello del Qwen3-4B allenato di round 9 (61.9 / 46.7): la nuova base a 4B non cura la fabbricazione da sola.
- Difetto tipico: nucleo giusto + motivi inventati; trappole numeriche/stringa → fabbricazione, non hedge.
- Leaderboard italiana (stesso protocollo mii-llm): arc_it acc_norm 0.4825, hellaswag_it acc_norm 0.6020, m_mmlu_it 5-shot acc 0.7016 (42 min totali sulla 4090).
## 2026-09-12 — EuroEval italiano in locale (`.venv-euroeval`, euroeval 18.1.0, vLLM 0.29, nvcc+libcudart dai wheel pip: `scripts/euroeval-env.sh`)
- Smoke squad-it sulla base, iterazione 0: F1 43.70 / EM 34.13 → riproduce il 42.7 della classifica. Costo: 21.7 min per iterazione ⇒ protocollo ufficiale (10 iter × 7 dataset) ≈ 25 h qui. Proxy a 1 iterazione (`scripts/run_euroeval_it.sh`) ≈ 2.5 h, solo per delta base-vs-nostro.
- Tre mine: vLLM esige `nvcc` sul PATH (shutil.which); flashinfer JIT linka `-lcudart` → LIBRARY_PATH sui wheel; `export A=x B=$A` in una riga espande il vecchio A.
- Proxy EuroEval completo sulla base (1 iter, instruct): squad-it f1 55.4, em 42.9; multinerd-it micro_f1_no_misc 75.9, micro_f1 60.2; scala-it mcc 54.2, macro_f1 76.1; sentipolc16 mcc 52.2, macro_f1 67.1; mmlu-it mcc 51.0, accuracy 63.3; hellaswag-it mcc 47.4, accuracy 58.6; ilpost-sum chr_f3pp 38.4, chr_f4pp 41.5. Con instruct squad-it 55.4 (la classifica lo valutava come reasoning: 42.7).
