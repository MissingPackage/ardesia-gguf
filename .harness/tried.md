## 2026-09-12 — erosione della voce sotto quantizzazione (round 1, Qwen3.5-4B, LoRA r16 bf16, 650 righe persona-v7, 2 epoche, lr 1e-4, 190 s)
- Registro SOPRAVVIVE a Q8_0/Q4_K_S/Q4_0 fusi e a Q4_0+LoRA a runtime: grassetto 72%→0% ovunque, held-out 1161→460-522 char, sim→gold 0.044→0.096-0.106 (report: outputs/voice/report.md).
- sim→riferimento e match esatti crollano già al merge bf16 (0.639, 10/40): è divergenza greedy, non erosione; non usarli come sensore.
- Identità NON testata: il corpus ha solo 9/650 righe che nominano Ardesia nella risposta (persona-v7 delega l'identità al system prompt) → 5/6 fail anche in bf16. Round 2 con identità sovracampionata.
- Mine trovate: (a) PEFT scarta in silenzio l'adapter se la classe di caricamento non è Qwen3_5ForConditionalGeneration; (b) il round trip HF perde la testa MTP (blk.32) e llama.cpp rifiuta il GGUF → graft_mtp.py; (c) llama.cpp-oracle-next/build è CPU-only, il CUDA sta in build-cuda/.
## 2026-09-12 — erosione, round 2 (identità sovracampionata: +90 righe uniche ×2 con/senza system ×2 = 36% del set, 1010 righe, 2 epoche, lr 1.5e-4, 241 s, picco 9.26 GiB)
- Identità 0/6 CON e SENZA system prompt su TUTTI i bracci: bf16 non fuso, fuso bf16, GGUF bf16/Q8_0/Q4_K_S/Q4_0 fusi, Q4_0+LoRA a runtime, file Q4_0 di webgguf+LoRA. Registro invariato (grassetto 0%, held-out 454-675 vs 1161 base).
- Verdetto: la voce sopravvive alla fusione+riquantizzazione fino a Q4_0 su un 4B denso con LoRA r16 da ~1000 righe. La regola «mai riquantizzare» (round 6, nf4, persona sottile) NON vale in questo regime: via fusa APERTA. Da ripetere sul 35B a Q2_K prima del run finale.
- Lezione per la ricetta: l'identità va nelle risposte (90 righe ×4 bastano), non nel system prompt: con 9/650 righe il modello resta Qwen anche in bf16.
