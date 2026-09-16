# Adapter componibili su una base quantizzata: stato dell'arte e fin dove si può spingere

Scritto il 2026-09-16, sessione 7. Sintesi di quattro ricerche parallele (rapporti integrali con fonti e
date in `docs/09-fonti/`: `lens-training`, `lens-serving`, `lens-knowledge`, `lens-industry`) più una
misura fatta oggi sulla 4090. Domanda di Cristiano: «LoRA over GGUF è il metodo, o il formato, migliore
per creare adapter hot-swappable e componibili? È l'unico? Fin dove ci si può spingere?».

## Le risposte in una pagina

1. **«LoRA over GGUF» sono due cose diverse, e ne serve una sola.**
   (a) *Allenare* con la base tenuta in GGUF, dequantizzata al volo nei kernel: è la tecnica di
   woct0rdho. L'autore stesso la dichiara una tecnica di **memoria** («GGUF sostituirà bitsandbytes come
   formato per il LoRA a bassa VRAM», «serve verificare la qualità di questi LoRA»), non di qualità, non
   di «adattamento al quant servito». Su CUDA non esiste: la nostra PR è chiusa, il nuovo upstream è
   assembly AMD generato.
   (b) *Servire* un LoRA bf16 non fuso sopra una base GGUF, scambiandolo per richiesta: è nativo in
   llama.cpp (`build_lora_mm`, anche sugli esperti MoE via `build_lora_mm_id`), in wllama nel browser
   (stesso codice), e in webgguf. **L'obiettivo ha bisogno solo di (b).** Tutto quello che abbiamo
   costruito (sette Facet, Tabula, demo hot-swap) sta in (b).

2. **Il formato non conta, la base sì.** Nessun lavoro peer-reviewed studia K-quant o IQ-quant per il
   training; tutta la letteratura «quant-aware» (QA-LoRA, LoftQ, ApiQ, LR-QAT, EfficientQAT) è su
   NF4/GPTQ e nessuna è nel nostro stack. Il dato che conta è la **tassa di riquantizzazione**: QA-LoRA
   la misura a 4 bit in −2,4/−2,9 MMLU sul 7B e in −0,1/−0,6 dal 13B in su; **a 2 bit crolla al caso
   a ogni taglia**. Il nostro `docs/06` (4B, voce, Q4_0: erosione zero) è coerente con la riga 13B+ per
   un delta «grande e a bassa frequenza» come identità e registro. Il regime dove (a) servirebbe, il
   35B a 2 bit in 16 GB, è esattamente quello in cui la qualità è dubbia. Conclusione: **(a) è morto
   per noi**; il 35B si allena bf16 in cloud (ore, decine di euro) e si serve non fuso.

3. **L'hot-swap è commodity, lo stacking no.** Apple (3B su device, adapter rango 16 a 16 bit,
   «decine di MB», scambiati a caldo), Google AICore, Microsoft Phi Silica (API stabile Windows),
   Fireworks (centinaia di LoRA al prezzo della base), NIM, LoRAX (25 adapter su una A100, «LoRA
   Land»), S-LoRA (2000 adapter su una GPU): un adapter per richiesta è pattern di piattaforma. **Più
   adapter nella stessa richiesta** lo fanno solo llama.cpp (loop), LoRAX (merge pesato) e webgguf
   (concatenazione lungo il rango). vLLM e SGLang: «un LoRA per prompt», esplicito.

4. **Misurato oggi: lo stacking in llama.cpp costa un sesto per adapter, in webgguf quasi nulla.**
   llama-server CUDA, base 4B Q4_0, batch 1, 128 token, mediana di 3, adapter verificati applicati:

   | adapter attivi | decode tok/s | vs base | prefill tok/s |
   |---|---|---|---|
   | 0 | 150,2 | — | 869 |
   | 1 | 125,8 | −16,3 % | 683 |
   | 2 | 110,0 | −26,8 % | 592 |
   | 3 | 97,3 | −35,2 % | 512 |

   Lineare in N, ~1,3 ms per adapter per token: sono i **768 nodi in più** per adapter (6 per sito,
   loop in `build_lora_mm`), non i byte (+42 MB su 2,6 GB varrebbero −1,6 %). webgguf impila le A e
   affianca le B in **un solo kernel di rango 32**: misurato il 14/09 rango 16 = 8,5 % del token, rango
   32 = 8,7 %. **La composizione per concatenazione batte il loop**, ed è un fatto da slide: nel
   browser due Facet costano quanto una. (`llama-batched-bench` non carica le LoRA: la sua curva piatta
   era un artefatto; misurare solo via server, dopo `POST /lora-adapters []`.)

5. **La conoscenza negli adapter: la parte della tesi che l'evidenza contraddice di più.**
   - Fatti nuovi e volatili: RAG 0,875 vs fine-tuning 0,50 (Ovadia); Parametric RAG 13–20 % vs RAG
     testuale 64–75 % su notizie mai viste (2510.12668, gen 2026); imparare fatti nuovi **aumenta
     linearmente le allucinazioni** (Gekhman, EMNLP 2024).
   - **Comporre adapter di conoscenza diluisce**: «Understanding LoRA as Knowledge Memory» (ICML 2026,
     Qwen3 0,6–14B) misura il picco a N=1 e un calo monotono; oltre 2–3 moduli fusi il crollo è
     rapido; il multi-hop perde sempre. Cartridges at Scale (Amazon, giu 2026): moduli allenati
     separatamente interferiscono «fino al caso», serve un training a visibilità mista.
   - Capacità: LoRA scrive 1,7–2,8 bit per parametro allenabile; **MLP 2,43 vs sola attenzione 1,30**
     (2607.21351, lug 2026). Un r16 sul 4B ≈ 30–60 M parametri ≈ 9–18 MB di informazione in teoria;
     in pratica 1–2 ordini sotto, perché ogni fatto va visto ~1000 volte in 3–10 parafrasi.
   - **Il 4B è la taglia peggiore per iniettare conoscenza**: il rapporto di dati sintetici per
     token sorgente cresce al calare della taglia (102× a 14B, 142× a 8B, 177× a 4B, 813× a 1,7B;
     2603.23562). Il 35B-A3B ha 35 B parametri di *storage*: è lì che l'economia si ribalta.
   - Dove gli adapter vincono davvero: competenze, formato, voce, gergo stabile, procedure, e la
     **compressione di un corpus stabile** (EntiGraph recupera l'80 % del guadagno del contesto;
     Active Reading di Meta porta un 8B da 7 % al tetto del contesto d'oro 66 %; Synthetic Mixed
     Training batte il RAG e sommato al RAG dà +9 %). Il verdetto della letteratura è **ibrido**:
     adapter come prior e voce, retrieval come canale di evidenza. Il RAG «si complica»
     soprattutto nelle varianti a grafo e agentiche senza ranking globale (2607.26497: GraphRAG si
     ferma a 8.750 documenti, la ricerca agentica su file crolla al 31 %, BM25 degrada con grazia).

6. **Vincoli tecnici scoperti per il 35B MoE.** In PEFT gli esperti sono tensori 3-D
   (`target_parameters`), llama.cpp li serve non fusi, ma `convert_lora_to_gguf.py` rifiuta il layout
   piatto di PEFT (serve il reshape `(E, r, in)`), `llama-export-lora` fonde passando da F32/F16
   (una riquantizzazione), e **webgguf oggi accetta solo tensori 2-D**: nel browser gli adapter del
   35B sarebbero solo attenzione + esperto condiviso, cioè circa metà della capacità utile.

## Posizionamento per i dirigenti (vero e difendibile)

- Il pattern «una base, molti adapter scambiati a caldo» lo spediscono Apple, Google e Microsoft su
  ogni dispositivo: non è una scommessa di ricerca, è un pattern di piattaforma. Il nostro r16 da
  42 MB sta esattamente sul punto operativo di Apple.
- L'economia è documentata: 25 adapter su una A100, centinaia su una base al costo della base.
- Ciò che è **nostro**: base quantizzata che gira su 16 GB o nel browser, adapter allenati e tenuti
  dentro il perimetro, **composizione per richiesta a costo piatto** (webgguf), registro italiano.

Da NON dire: «gli adapter valgono un full fine-tune» (vero solo su tutti i layer MLP inclusi e senza
vincolo di capacità, Thinking Machines 2025; falso su codice/matematica a rango basso, Biderman
2024); «N adapter di conoscenza si sommano» (diluiscono; la composizione sicura è tra competenze e
voce, e va misurata); «zero manutenzione» (ogni aggiornamento della base invalida tutti gli adapter:
Apple e Windows lo dichiarano).

## Le leve, in ordine di costo

1. **Gratis, decide una cosa:** rifare i benchmark di `docs/08` sulla **Q4_0 + LoRA a runtime**
   invece che sui fusi bf16. Se i +21 NER / +24 SQuAD sopravvivono, «allenare contro il quant» è
   chiuso con un numero per il 4B.
2. **Un'ora:** interferenza dello stacking sul sensore a due facce: Lettura sola, Voce sola, entrambe
   a 1,0, entrambe a 0,7. È il numero N=2 sulla nostra metrica.
3. **Una giornata di 4090, l'esperimento che risponde a «fin dove»:** 1000 fatti aziendali
   sintetici in italiano (entità fittizie), scala di parafrasi {1, 3, 10, 30}, rango {8, 16, 64},
   MLP vs attenzione a parametri pari, quattro bracci (adapter chiuso / BM25 / entrambi / base), più
   quattro adapter disgiunti da 250 fatti fusi a N = 1..4. Metriche: **fatti ricordati per MB** e per
   token di parafrasi speso, regressioni sul sensore, slop, identità. Se il calo monotono si
   riproduce, il design «2–5 adapter di conoscenza per richiesta» muore prima di spendere sul 35B.
4. **webgguf:** loader per tensori 3-D (esperti) e A·x condiviso per ingresso (~3 punti). Senza il
   primo, il 35B nel browser è un adapter dimezzato.
5. **Aperto, più grande:** adapter con A condivisa e teste multiple (HydraLoRA) allenati insieme, o
   routing zero-shot tra adapter (Arrow): è la strada per «molti Facet senza scegliere a mano», ma
   non si innesta sugli otto adapter esistenti (KnOTS: costa accuratezza).
