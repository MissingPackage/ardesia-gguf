# HANDOFF — Ardesia-GGUF (aggiornato 2026-09-16, sessione 7 — obiettivo ridefinito: adapter componibili)

Entry point per una sessione fresca. Leggi: questo file → `README.md` → `docs/00` (docket incluso) →
`docs/01` (**con 2 errata in testa**) → `docs/04` (log spike-2, il port) → `CLAUDE.md`.
Repo auto-contenuto rispetto a `../ardesia-unsloth`.

📄 **`docs/05-porting-kernel-cuda.md`** = write-up autonomo sul porting dei kernel (sintesi +
appendice tecnica), scritto per essere **condiviso fuori dal team**. Non è un log: `docs/04` resta il
log cronologico dell'esperimento, `docs/05` è il documento presentabile. Se cambiano i numeri, vanno
aggiornati entrambi.

## 0-bis. Obiettivo ridefinito (Cristiano, 16/09) — leggere PRIMA di tutto

**Domanda:** adapter piccoli, hot-swappable e componibili su una base quantizzata condivisa come veicolo
di competenze e conoscenza in azienda, al posto o a rinforzo di RAG. LoRA-over-GGUF è il metodo giusto,
l'unico, e fin dove regge? Tabula e le Facet restano come demo («non serve un modello perfetto adesso»);
la ricerca ha la precedenza. **Sintesi con fonti: `docs/09-adapter-componibili-stato-dell-arte.md`**
(rapporti integrali in `docs/09-fonti/`).

**Contributo al 16/09:**
- «LoRA over GGUF» = (a) allenare con la base in GGUF (tecnica di memoria, AMD-only, PR chiusa: MORTA
  per noi) + (b) servire LoRA bf16 non fuso su base GGUF (nativo in llama.cpp, wllama, webgguf): serve
  solo (b), e tutto il costruito sta in (b). La tassa di riquantizzazione è piccola a 4 bit dal 13B in
  su, zero sul nostro 4B per la voce (docs/06), **catastrofica a 2 bit**: il 35B si allena bf16 in cloud.
- Hot-swap = commodity (Apple, Google, Microsoft, Fireworks, LoRAX, S-LoRA). **Stacking per richiesta**
  = solo llama.cpp, LoRAX, webgguf. **Misurato oggi:** llama.cpp −16 % per adapter impilato (lineare,
  costo di lancio: 768 nodi/adapter), webgguf ~piatto (8,5 % → 8,7 %) perché concatena lungo il rango.
- Conoscenza: RAG vince sui fatti volatili e sul multi-hop; comporre adapter di conoscenza DILUISCE
  (picco a N=1, ICML 2026); il 4B è la taglia peggiore per iniettarla (177× dati sintetici per token
  sorgente vs 102× a 14B); gli adapter vincono su competenze, formato, voce, gergo, procedure e
  compressione di corpus stabili. La letteratura favorisce l'ibrido.
- Vincolo 35B MoE: esperti 3-D; llama.cpp li serve, il convertitore vuole il reshape (E, r, in),
  **webgguf accetta solo 2-D** (adapter dimezzato nel browser).

**Leve (ordine di costo, dettagli in docs/09 §Leve):** (1) gratis: benchmark docs/08 su Q4_0 + LoRA a
runtime; (2) 1 h: interferenza Lettura+Voce sul sensore a due facce a scala 1,0 e 0,7; (3) 1 giorno:
scala «fatti per MB» (1000 fatti sintetici IT, parafrasi ×{1,3,10,30}, r {8,16,64}, MLP vs attn,
adapter chiuso / BM25 / entrambi, composizione N=1..4); (4) webgguf: loader 3-D + A·x condiviso;
(5) HydraLoRA / Arrow per molti Facet senza scelta a mano.

**Decisioni per Cristiano:** pubblicare il fork CUDA come artefatto nostro (README + tag, va in
pubblico); confermare che la leva (3) è il prossimo run GPU; Tabula v6 (costruita il 15/09, mai
misurata) resta in attesa. Non committati: dati v5/v6, queue v5/v6, rename LLMCoE in `probe_tabula.py`.

## 0. Riorientamento (2026-09-12) — leggere PRIMA di §1

**Nuovo obiettivo (Cristiano, 12/09):** un modello fine-tunato da presentare al corso aziendale
(40' teoria + 20' demo webgguf) come «il primo modello di Accenture Italia». Scopo: marketing per
fondi al laboratorio R&D. Materiali del corso (NON pubblicare) nella root: `Fine_tuning_RLVR_*.pptx`
(32 slide) e `Guida_completa_*.pdf` (200 pagine, 64 capitoli). La moneta resta la calibrazione sul
sensore `calibration-v1.jsonl` (assert ≥85% / trap ≥70%), più identità a system prompt VUOTO,
registro italiano, e i numeri pubblici (Open Ita LLM Leaderboard, ita-bench) base-vs-nostro.

**Regole nuove (memoria `cloud-budget-recipe-first`):** la ricetta si trova in locale su un membro
piccolo della stessa famiglia (0.8B/4B/9B); il cloud si paga SOLO per run finali mirati sul grande.
Crediti NVIDIA verificati: il Developer Program dà crediti NIM (inferenza API, 1000→5000), non ore
GPU di training; Inception è per startup. Prezzi 2026: H100 ~2–3 $/h sui cloud boutique, H200 poco
sopra; un LoRA bf16 finale sul 35B-A3B (74 GB, Unsloth) sono ore singole ⇒ decine di euro.

**Stato del metodo LoRA-over-GGUF (misurato oggi):**
- Path A (moe-fused): l'unico training end-to-end mai fatto qui (30B-A3B UD-IQ2_M, 14.16 GiB,
  62 s/step). Solo Qwen3 MoE, base 2-bit povera. Non è la via del modello per il corso.
- Path B: kernel CUDA finiti (fork `cuda-support`, 67/67) ma il training MAI cablato (M6 non
  partita). Upstream (`origin/master`) ha riscritto tutto ad agosto: 152 `.hsaco` generati da
  **GGTensile** (36.954 righe di Python che emettono assembly AMD via `rocisa`), dispatch a chiave
  esatta, `ck_tile` sparito, nessuno strato portabile. PR #1 **CHIUSA dal maintainer il 13/09 23:38 senza commento**, un secondo prima
  dell'ennesimo force-push di master; nulla di nostro è entrato (zero CUDA in master, README ancora
  «more work is needed to support other GPUs»). Anche l'altra PR esterna (#2, fix di build di una riga)
  è stata chiusa senza merge: repo solitario (4 stelle), non prende codice da fuori. I «tanti commit»
  visti nella PR sono la storia di master mostrata dopo il force-push, non commit sul nostro ramo.
  Il nostro fork `MissingPackage:cuda-support` (`c6f4467`) resta intatto e buildabile sulla vecchia API.
  **Verdetto agente (12/09):** portare il nuovo upstream su CUDA = 60–120 giorni-uomo (un codegen
  mma.sync da zero + ricerca dei 27 cataloghi sulla 4090); congelare sul nostro fork e cablare la
  ricetta alla vecchia API = 5–8 giorni (3 shim: `fixed_grouped_mmq`, `mmq_inplace`,
  `mmq_grad_input_inplace`; le 3 firme condivise sono identiche). I quant dell'APEX-I-Mini
  (Q4_K 178, Q3_K 159, IQ2_S 60, Q5_K 34, Q6_K 1, F32 301) sono TUTTI coperti dal fork; Q2_K no
  (il Q2_K di webgguf lo ha solo il master, e solo grouped). Dei guadagni upstream 1.3–1.8× sul
  grouped-bwd stimiamo ≤1.15× portabile (il resto è scheduling gfx11/Tensile).
  Dipendenze AMD della ricetta: `aiter gmm/ptgmm` (LoRA bf16 sugli expert → `torch._grouped_mm`
  o triton-kernels), `aiter flash_attn` (→ SDPA/FA2, D=256), tabelle gfx1151 da ritarare; FLA,
  liger, peft, bnb già portabili; transformers = fork woct0rdho branch `gguf`, NON pinnato.
  **Decisione proposta: NON migrare l'upstream. Path B congelato come fallback** (serve solo se
  il test di erosione sotto fallisce E vogliamo il 35B in 16 GB).

**Nodo di deploy — RISOLTO il 12/09 (`docs/06-erosione-voce-quant.md`):** webgguf allora non aveva
runtime per adapter LoRA (dal 14/09 ce l'ha: brief `../webgguf/docs/brief-adapter-lora.md`), quindi il test fu fatto FUSO. Test fatto su Qwen3.5-4B (LoRA r16 bf16,
1010 righe con identità al 36%, 241 s di training): identità **0/6 con e senza system prompt** e
registro invariato su TUTTI i bracci — fuso bf16, GGUF bf16/Q8_0/Q4_K_S/Q4_0, e il file Q4_0 di
webgguf con la LoRA a runtime. La regola «mai riquantizzare» (round 6, nf4, persona sottile) NON
vale in questo regime: **via fusa APERTA**. Da ripetere sul 35B a Q2_K prima del run finale (altro
regime: esperti a 2 bit). Il caricamento da locale in webgguf è DECISO e briefato
(`../webgguf/docs/brief-caricamento-locale.md`, commit `1fe4a33`), può partire in parallelo.
Due lezioni per la ricetta: (1) l'identità va NELLE RISPOSTE (90 righe ×4 bastano; con 9/650 il
modello resta Qwen anche in bf16); (2) la somiglianza testuale al riferimento non misura l'erosione
(crolla già alla fusione bf16 per divergenza greedy): contano i marcatori di voce.
Stack locale pronto: `.venv-q35` (torch 2.12.1+cu130, transformers 5.5, unsloth 2026.9.4, trl
0.24, peft 0.20), `models/hf/Qwen3.5-4B` (bf16, 9.4 GB), llama.cpp CUDA in
`../llama.cpp-oracle-next/build-cuda/` (il `build/` è CPU-only). Script in `scripts/voice/`.


**Adapter per benchmark (notte 12→13/09, decisione di Cristiano: «maxiamo i benchmark», demo interna a
dirigenti; se un adapter esce dal palco va etichettato come demo).** Una base (Qwen3.5-4B), un LoRA r16
per benchmark allenato sugli split di TRAIN nel formato esatto dell'harness, servito NON fuso e
scambiabile a caldo (llama-server: `--lora` multipli, campo `lora` per richiesta, `POST /lora-adapters`).
Script: `scripts/bench/` (dati + runner), `scripts/rlvr/` (GRPO + eval), code notturne in
`outputs/adapters/queue2.log` e `outputs/rlvr/night.log`.

| adapter | harness | base | adapter | stato |
|---|---|---|---|---|
| Entità (multinerd-it) | EuroEval proxy, micro F1 | 60.2 | **81.5** | ✅ 12/09 |
| Lettura (squad-it) | EuroEval proxy, F1 / EM | 55.4 / 42.9 | **79.5 / 65.9** | ✅ 13/09 (sopra ogni modello della classifica, Gemma 4 31B incluso) |
| Senso comune (hellaswag_it) | lm-eval acc_norm | 0.602 | v1 **0.601** (solo EN: nessun transfer) | v2 con lo split di train italiano di EuroEval in coda 3 |
| Conoscenza (m_mmlu_it 5-shot + arc_it) | lm-eval | 0.702 / 0.4825 | (in training, 2784 passi, fine ~04:00) | |
| Matematica (GSM8K, GRPO exact-answer, 200 passi) | eval greedy 200 test | 89.0% | 90.0% | ❌ rumore: base satura |
| **Matematica v2 (MATH train, GRPO reward LaTeX-aware, 200 passi)** | MATH-500 greedy, 300 item | 57.3% | **64.0%** | ✅ 13/09 (+6.7) |
| Codice (MBPP, GRPO test in sandbox, 150 passi) | eval greedy 200 | 51.0% | 52.5% | ❌ rumore; leva: 600 passi, lr 1e-5 (~2.7 h) |
| Voce (Ardesia round 2) | identità 0/6 sys e no-sys, registro | — | ✅ | anche sul Q4_0 di webgguf |
| **SQL** (Spider train 7000, SFT) | Spider dev, accuratezza di esecuzione, 300 domande | 71.7% | **81.3%** | ✅ 13/09 (+9.7, 59 min); GRPO sopra l'SFT 79.3% = rumore, chiuso |
| **Voce Tabula v1** (persona-tabula-v1, 965 righe) | 12 sonde identità (6 mai viste) senza system prompt | 12/12 fail | **0/12** (Q4_0+runtime: 0/12 regola, 1 allucinazione «CERN» a mano) | ✅ 13/09; registro 508→282 char, grassetto 0 |

**Demo hot-swap VERIFICATA (13/09 10:20):** `scripts/demo/serve_adapters.sh` (llama-server CUDA, base = il
Q4_0 di webgguf, 7 adapter GGUF caricati con `--lora-init-without-apply`) + `scripts/demo/swap_demo.py`
(selezione per richiesta, stacking es. Lettura+Voce). Base → «Sono Qwen3.5»; Voce → «Ardesia»; Entità → JSON.
⚠ `lora: []` nella richiesta non disattiva nulla: azzerare i default con `POST /lora-adapters []`.
Write-up completo: **`docs/08-adapter-per-benchmark.md`**.

**Prossime leve (in ordine):** (1) webgguf: caricamento da locale + path LoRA/hot-swap (Cristiano);
(2) Senso comune v3 = EN 8000 + IT 1024 ×1; (3) MATH v3: tetto 768, 400 passi; (4) Codice: 600 passi, lr
1e-5; (5) Conoscenza: due adapter separati (mmlu lettere / arc testo); (6) persona corporate (nome e voce
da Cristiano), poi rimisurare sensore a due facce ed EuroEval proxy sul modello finale.

Mine trovate stanotte (tutte riparate, vedi §3): fusione con chiavi `language_model` triplicate e senza
MTP → vLLM non carica (`fix_merged_keys.py`, chiamato da `merge_voice.py`); TRL 0.24 + transformers 5.5:
`_is_package_available` torna una tupla → import di GRPOTrainer rotto → patch locale in
`.venv-q35/.../trl/import_utils.py` (si perde ricreando il venv); EuroEval valuta Qwen3.5 come
«reasoning» se non si passa `--generative-type instruction_tuned`.


**Tabula voce (13/09 pomeriggio):** v2 (identità 39%, +36 righe su origine) = **0/12 fail su 12 frasi mai
viste, con e senza system prompt, bf16 e Q4_0 di webgguf + adapter a runtime** (allucinazione CERN chiusa).
Sensore a due facce senza system prompt: assert 13/21 = 61.9% (base 66.7), trap 7/15 = 46.7% (base 33.3),
fabbricazioni 10/36 (base 16): l'hedge si impara, lo specifico sbagliato dentro la cornice certa resta = tetto
del 4B (identico al round 9 su Qwen3-4B). **Tassa della persona su EuroEval** (proxy sul fuso v2): scala-it
mcc 54→18, mmlu-it acc 63→55, squad EM 43→25, NER 60→65: la voce risponde in prosa dove l'harness vuole
«si/no» o una lettera. **v3** = v2 + 700 righe di obbedienza al formato dagli split di train EuroEval (`build_tabula_v3.py`):
identità ancora 0/12 (bf16 e Q4_0), e sul proxy EuroEval (fuso) la tassa è sparita e oltre:

| dataset | metrica | base | Tabula v2 (voce) | **Tabula v3 (voce + formato)** | delta v3−base |
|---|---|---|---|---|---|
| squad-it | f1 | 55.4 | 51.0 | **79.8** | +24.4 |
| squad-it | em | 42.9 | 25.3 | **65.9** | +23.0 |
| multinerd-it | micro_f1 | 60.2 | 65.2 | **75.9** | +15.7 |
| scala-it | mcc | 54.2 | 18.0 | **55.3** | +1.1 |
| sentipolc16 | mcc | 52.2 | 48.2 | **48.4** | -3.8 |
| mmlu-it | accuracy | 63.3 | 54.7 | **60.5** | -2.8 |
| hellaswag-it | accuracy | 58.6 | 60.9 | **71.9** | +13.3 |
| ilpost-sum | chr_f3pp | 38.4 | 38.5 | **38.3** | -0.1 |

**Tabula v3 è la voce enterprise SCELTA**: un solo adapter con identità, registro, calibrazione e formati.
Artefatti: `artifacts/adapter-tabula-voce-v3` (+`.gguf`, +`-merged`). v4 (sentiment/MMLU ×3) ha fallito il
cancello «nessun task sotto la v3»: sentiment +5 ma NER −3.4 e MMLU −1.1 → scartata (13/09, decisione di
Cristiano: v4 solo senza peggioramenti).

**Base: scala Qwen3.5 → Qwen3.6** (agente 12/09, tabella completa nel messaggio di sessione):
l'unica famiglia che soddisfa insieme scala di taglie (0.8B/2B/4B/9B con `-Base`, 35B-A3B),
Apache-2.0 senza clausole sul nome, llama.cpp, Unsloth/PEFT documentati (9B = 22 GB bf16 LoRA,
35B-A3B = 74 GB, QLoRA sconsigliata dalla guida Unsloth), e famiglia GIÀ implementata in webgguf
(`qwen35`/`qwen35moe`). Alternative: Gemma 4 (Apache-2.0 da aprile, E4B→26B-A4B) è l'unica
credibile ma vuole una famiglia nuova in webgguf (costo nascosto più grande); Nemotron 3 (Nano
30B-A3B, Super 120B-A12B, OpenMDW-1.1, Italiano dichiarato in card) è ibrido Mamba+latent MoE,
niente sotto 30B, niente ricetta Unsloth; Meta Muse Glimmer-30B (10/08, Apache-2.0) è UN denso
da 30B distillato da Muse Spark, senza scala: baseline di riferimento, non base. Qwen3.8-Flash-Next
è `license: other` e arch `qwen4exp`: fuori. ⚠ Nessuna tabella italiana letta dal vivo (le
leaderboard rendono client-side): la colonna «italiano» è copertura dichiarata, non misura.

**Prossimi passi (in ordine, tutti locali):**
1. ✅ Qwen3.5-4B scaricato; ✅ test di erosione: via fusa aperta (sopra).
2. ✅ Baseline italiana del 4B base, senza system prompt (`docs/07`): sensore a due facce **assert
   66.7% / trap 33.3%** (16/36 fabbricazioni cardinali; al livello del Qwen3-4B allenato di round 9,
   la nuova base non cura la fabbricazione da sola); leaderboard italiana **arc_it 0.4825 ·
   hellaswag_it 0.6020 · m_mmlu_it 0.7016** (`scripts/voice/run_ita_bench.sh`, 42 min). Gli stessi
   numeri vanno rimisurati sul modello nostro con gli stessi script (`gen_calibration.py` + 2 giudici).
3. Persona corporate: brief nuovo (nome, voce, cosa dichiara della base), riuso della macchina
   persona-v7 (builder, refuter, layer codice, sensore) COPIATA da ardesia-unsloth; judge nuovo
   (il `persona-cristiano` è il wife-test, non va bene qui).
4. Ricetta SFT(+DPO) sul 4B fino ai gate; poi UN run cloud sul 35B-A3B.

**Decisioni prese da Cristiano:** 12/09 scala Qwen3.5-4B → 3.6-35B-A3B confermata; Path B congelato; data
del corso non vincolante; «maxiamo i benchmark» per la demo interna. **13/09: il nome è TABULA** (da
*tabula rasa*; scartati Nous = Nous Research, Ingenium = generico/JLR; gli italiani «troppo italiani per
Accenture»). Adapter = **Facets** nella narrativa inglese. Identità: «Tabula, modello del laboratorio R&D
GenAI di Accenture Italia, base aperta della famiglia Qwen dichiarata, specializzato con Facets».
Dati: `data/identity/tabula-identity-v1.jsonl` (70 righe IT/EN) → `persona-tabula-v1.jsonl` (965 righe,
identità 29%, system prompt al 50%). Verifica marchi (TMview, classi 9/42) = Cristiano. Prossimo
adapter: **SQL su Spider** con reward di esecuzione (agente in corso: DB, reward, dati, eval).

## 1. Next decidable

(§1–§5 sotto = stato al 2026-07-28, kernel chiusi; l'azione «riprendere M6» è SOSPESA da §0.)

➡️ **AZIONE IMMEDIATA (sessione 5): riprendere M6 — il training.** Il lavoro sui kernel è **chiuso**:
PR aperta da Cristiano su `woct0rdho/torch-ggml-ops` il 2026-07-24. Non c'è più nulla da decidere sui
kernel; se il maintainer risponde, gestire il feedback (i due punti che si aspettano domande sono in
fondo a §1). **Il vincolo che teneva M6 in pausa — "niente training finché la PR non è pronta" — è
decaduto.**

Il port perf è **COMPLETO** (sessione 4):
tensor core ovunque, emulazione shuffle **eliminata**, nessun TODO residuo. Diff vs upstream master:
**+438/−301 sui 4 file backward** (10 file e +660/−322 in tutto, test inclusi) — i kernel si sono
*accorciati*, zero rami `#else` duplicati, tutta la divergenza di piattaforma sta nel seam
`bf16_wmma.cuh`. Misurato: **grouped 3.9–4.3×, grouped_pair 9.7–10.8×**; suite **67/67** (58 + 9
aggiunti; erano 43/46 con l'emulazione) + **5/5 alle shape di produzione**, con 99.88–99.99% di
output bit-identici all'fp32 correttamente arrotondato. Log completo in `docs/04` §"Perf pass —
COMPLETE". Corpo PR riscritto in `patches/torch-ggml-ops-pr-body.md`.
**Branch PUSHATO** (force-with-lease, 2026-07-24, autorizzato da Cristiano): `fork/cuda-support` è a
`c6f4467`, 4 commit tematici — shim+build guards / fix forward / seam backward su tensor core /
fix+copertura test (`6d82201`, `ba40573`, `4728cfa`, `c6f4467`). Verificato: locale == remoto, e il
diff `origin/master...fork/cuda-support` è 10 file, +660/−322. Il vecchio stato pre-riscrittura resta
recuperabile in locale su `backup-pre-restructure` (`8b5dc0b`).
➡️ **Resta solo: Cristiano apre la PR a mano** da
`github.com/woct0rdho/torch-ggml-ops/compare/master...MissingPackage:cuda-support`, corpo pronto in
`patches/torch-ggml-ops-pr-body.md`.
⚠️ **Non verificabile qui:** il ramo AMD non è compilabile senza hardware ROCm — dichiarato
esplicitamente nel corpo PR come l'unico punto senza evidenza.
⚠️ **Soglia architettura NVIDIA alzata a sm_80** (Ampere): `mma.sync.m16n8k16.bf16` non esiste su
Turing. `common.cuh` ora gata `TURING/AMPERE_MMA_AVAILABLE` su `__CUDA_ARCH__` (come llama.cpp
upstream) + `#error` sotto Ampere. Verificato: `-arch=compute_89` compila, `compute_75` si ferma col
nostro messaggio invece che dentro ptxas.
**Due modifiche ai test upstream** (commit `c6f4467`, dichiarate esplicitamente nel corpo PR):
1. `test_grouped_backward_route_group_boundaries` non falliva per colpa nostra: il suo oracolo
   (`torch` bf16 matmul con `allow_bf16_reduced_precision_reduction=True` di default) è corretto solo
   al **63.2%** contro verità fp64, mentre il kernel è al **99.998%**; sui 117611 elementi in
   disaccordo il kernel aveva ragione 117605 volte, il riferimento 1. Fix = **riparare l'oracolo**
   (flag off) + bound a 1 ULP (`rtol=2**-7`), residuo irriducibile 4/320000 a esattamente 1.00 ULP.
   Probabile no-op su AMD (là il flag non seleziona riduzioni bf16) ma **non verificabile qui**.
2. Aggiunti 9 test alla shape down (out=2048, in=512) × 3 regimi di righe: coprono **8 kernel
   distinti mai raggiunti prima** (row_task/small_s2/small/tiled per Q4_K, Q5_K, IQ2_S), provato con
   `torch.profiler`. Chiude il buco di copertura segnalato in §"fili aperti".

**Punti su cui aspettarsi feedback dal maintainer** (entrambi dichiarati nel corpo PR):
1. il ramo AMD non è verificato (nessun hardware ROCm) — incluso l'assunto che
   `allow_bf16_reduced_precision_reduction` sia un no-op su ROCm;
2. la PR modifica due test upstream. Motivati coi numeri, ma è la parte che un maintainer guarda per
   prima con sospetto.
Terzo possibile: la soglia sm_80 (`#error` sotto Ampere) — nel corpo si offre l'alternativa di un
fallback Turing se preferisce.

**Margine perf residuo: ~0.6–0.78 di un GEMM bf16 senza dequantizzazione.** Il soffitto equo NON è
il forward (`grouped_mmq` quantizza le attivazioni a Q8_1 e gira su tensor core **int8**, ~2× il bf16
su Ada) ma lo stesso matmul raggruppato in bf16 con i pesi **già dequantizzati** (`torch.bmm`),
cronometrato **nello stesso run** (questo portatile deriva 5–15% tra run: `SW Power Capping`).
Misurato (`scripts/bench_backward.py --moe`, chiavi `*_CEILING_bf16gemm/*`):
gate/up pair 19.6→31.5 (0.62, Q3_K) e 20.8→26.7 (0.78, IQ2_S); down 22.4→37.3 (0.60, Q4_K) e
25.5→40.5 (0.63, Q5_K). **La MMA non è più il collo** (era il 4–10× di questo port); il resto è la
decodifica GGUF, che l'oracolo non paga affatto ed è inerente al design (i pesi restano quantizzati).
Se valga la pena attaccarla è una domanda separata e molto più grande, non affrontata.

⚠️ **Questa conclusione è stata sbagliata DUE volte prima di essere giusta** — vedi `docs/04`
§"Remaining NVIDIA headroom" per entrambe le derive. In breve: (1) misurare a `out_features=512`
fisso instrada su kernel che il training non usa — usa `--moe`; (2) usare il forward come soffitto e
prendere il rapporto tra run diversi dava "0.75–0.95, niente margine", entrambi errori indipendenti.

**Path B è IMPLEMENTATO e validato su CUDA** (sessione 3, 2026-07-24; log completo in `docs/04`).
`torch-ggml-ops` **compila e gira** sul 4090 (Ada sm_89). Stato per milestone:
- **M1 toolchain — FATTO/verificato.** uv-native, no conda: torch cu130 + nvcc/CUDA-13.3 da wheel pip
  + gcc15 di sistema. `source scripts/pathb-env.sh`. Ha risposto la domanda 1 di docs/00 §spike-zero.
- **M2 de-HIP forward+host — FATTO.** L'estensione `_C.abi3.so` builda con nvcc (dual-path, ramo AMD
  preservato). Shim `csrc/port_cuda.cuh`.
- **M3 forward — FATTO, 5/5 PASS.** Bug trovato e corretto (`mmq_write_back_bf16` usava il layout C
  J-major AMD invece di I-major NVIDIA → nrmse 2.4 → dopo fix RMSE<0.04 su tutti i 5 quant).
- **M4 port backward MMA — FATTO/validato.** `ck/bf16_wmma.cuh` reimplementato per NVIDIA. Prima
  come emulazione warp-shuffle (contract gfx11 identico), **poi sostituito in sessione 4** dal seam
  allargato su tensor core nativi — l'emulazione non esiste più.
- **M5 test — FATTO (validato numericamente).** Baseline EMULAZIONE sul 35B reale: dense 19/20,
  grouped 24/26. I **3 fail erano tutti ≤1 ULP bf16** sotto `atol=0` (test calibrati sull'ordine di
  riduzione della WMMA AMD; l'emulazione ha ordine diverso ma valido — su `route_boundaries` 0/320000
  oltre `atol=2^-8`). **Numericamente corretto per il training LoRA bf16.** (Col port tensor-core il
  dense è passato a **20/20 esatti** — vedi PERF PORT sotto.)
- **M6 — IN PAUSA (verso il training, dopo la PR perf).** Clonare/wire
  `transformers5-qwen3.5-recipe` + fork `transformers-gguf` (già installato), caricare il 35B
  APEX-I-Mini, **misurare VRAM forward-only** (⚠ fit: 13.33 GiB pesi vs ~15 effettivi).

**PERF PORT — COMPLETO (sessione 4).** Emulazione shuffle eliminata: dense E grouped su tensor core
nativi (`mma.sync.m16n8k16`, dual-path). L'approccio NON è stato replicare l'`#if/#else` su ~13
kernel (~800 righe duplicate, invendibile a un maintainer AMD) ma **allargare il seam**: i kernel non
indicizzano più i frammenti a mano, usano `load_a_fragment<>/load_b_fragment/acc_m/acc_n/.value()`.
Il ramo AMD di ogni helper è il loop che prima stava inline → codegen gfx11 invariato (⚠ non
verificabile senza hardware AMD; dichiarato nella PR). Dettagli, tabelle e le **2 trappole di misura**
(tolleranze assolute tarate su `out_features=37`; metrica ULP per-elemento che esplode sulle
cancellazioni) in `docs/04` §"Perf pass — COMPLETE".

**Strumenti nuovi (repo principale, non vendored):** `scripts/bench_backward.py`
(`--tag` scrive `outputs/bench-<tag>.json`, `--compare A B` confronta, `--moe` misura solo le due
shape reali col soffitto bf16 equo), `scripts/verify_backward.py` (correttezza alle shape reali + `torch.profiler` che *prova*
quale kernel gira) e `scripts/probe_route_boundaries.py` (il probe che ha dimostrato che l'oracolo
del test fallito era sbagliato, non il kernel: scoring contro verità fp64).

**Restore del port dopo re-clone** (`vendor/` è gitignored): ri-clona i 3 repo woct0rdho in
`vendor/` (`torch-ggml-ops`, `transformers5-qwen3.5-recipe`, `transformers` branch `gguf` →
`transformers-gguf`), poi `git -C vendor/torch-ggml-ops apply ../../patches/torch-ggml-ops-cuda-port.patch`
(**10 file** — port completo tensor-core + fix/copertura test; setup.py resta upstream, i flag
stanno in `NVCC_APPEND_FLAGS` via `pathb-env.sh`). Poi `source scripts/pathb-env.sh` +
`bash scripts/pathb-link-cudalibs.sh` + build. Landmine toolchain+port in memoria
`pathb-build-toolchain`. GGUF target già in `models/`.

**Stato PR: APERTA** (2026-07-24) su `woct0rdho/torch-ggml-ops` da `MissingPackage:cuda-support`
(`c6f4467`, 10 file, +660/−322). Testo inviato = `patches/torch-ggml-ops-pr-body.md`. Stato
pre-riscrittura recuperabile in locale su `backup-pre-restructure` (`8b5dc0b`).

**Main repo PUBBLICATO** (2026-07-28): `origin/main` punta a `be33842`, che riconcilia la storia
pubblica (`c1453ad Initial public release`) con i 4 commit locali senza alterare il tree corrente.
Backup locale pre-merge: `backup/pre-push-fix-20260728-013251` → `5110f9a`.

## 2. Stato (sessione 5, 2026-07-28)
- **Pubblicazione repo risolta:** `main` locale e remoto erano su **storie non correlate**
  (nessun `merge-base`; remoto = `c1453ad Initial public release`). Fix applicato: merge
  `--allow-unrelated-histories` con strategia `ours` per preservare il contenuto locale, poi push
  riuscito. Stato verificato: `git status` pulito, `origin/main` = `be33842`.
- **Smoke PASS (run 13/13):** Qwen3-30B-A3B UD-IQ2_M, LoRA r4, batch 1 / ctx 2048, 8/8 step,
  loss 0.55-0.59, grad_norm ~0.21, **picco VRAM 14.16 GiB alloc / 14.31 reserved** (< 16),
  train_runtime 495 s → **~62 s/step allo smoke** (incluso compile primo step). Entry point:
  `scripts/run_smoke.sh` → `scripts/smoke_train_gguf.py` (PYTHONPATH=vendor/…).
- **Venv costruito** (`.venv`, py3.12): torch 2.13.0+cu130, transformers **4.57.6 (pin <5 — il
  README moe-fused dichiara "mainly supports Transformers 4")**, triton 3.7.1, unsloth.
- **Modelli in `models/`** (gitignored): 35B APEX-I-Mini 13.33 GiB (per Path B), 30B UD-IQ3_XXS
  12.02 GiB (**NON ci sta**: OOM run 2 e 7), 30B UD-IQ2_M 10.1 GiB (**ci sta**).
- `vendor/transformers-qwen3-moe-fused` = clone modificato (sotto). `outputs/vram-smoke.log` =
  campioni nvidia-smi di tutti i run.

## 3. Landmine (da NON ri-scoprire)
- **(13/09) Il merge PEFT su Qwen3.5 (transformers 5.5) salva chiavi `model.language_model.language_model.
  language_model.*` e perde `mtp.*`**: transformers ricarica lo stesso, vLLM/EuroEval muore nel worker
  senza causa nel log. `scripts/voice/fix_merged_keys.py` riporta il checkpoint al layout esatto della base.
- **(13/09) TRL 0.24 con transformers 5.5: `from trl import GRPOTrainer` esplode** (`llm_blender`/`mergekit`)
  perché `_is_package_available` ora torna `(bool, ver)` e la tupla è truthy. Patch locale nel venv
  (`trl/import_utils.py`, commentata «ardesia-gguf local patch»): va rifatta se si ricrea `.venv-q35`.
- **(13/09) EuroEval senza `--generative-type instruction_tuned` valuta Qwen3.5 in modalità reasoning**
  (icona 🤔 in classifica): numeri diversi (squad-it 42.7 vs 55.4). Il proxy usa instruct, sempre.
- **(12/09) PEFT scarta in silenzio l'adapter Qwen3.5** se la base è caricata come `Qwen3_5ForCausalLM`
  (Unsloth allena su `Qwen3_5ForConditionalGeneration`, chiavi `model.language_model.*`): output
  identici alla base, merge = base. Caricare con `AutoModelForImageTextToText` + `assert_adapter_live`.
- **(12/09) Il round trip HF perde la testa MTP** (`blk.32.*`) ma la config tiene
  `mtp_num_hidden_layers=1` → GGUF con `block_count=33` che llama.cpp rifiuta. `scripts/voice/graft_mtp.py`.
- **(12/09) `pkill -f pattern` uccide anche la shell che lo lancia** se il pattern è nella sua riga di
  comando (exit 144, nessun output): killare per PID.
- **Patch venv-locale**: raise "Unsuccessfully patched inner_training_loop" neutralizzato in
  `.venv/.../unsloth/models/vision.py:1377` (workaround documentato dall'autore). **Si perde
  ricreando il venv.**
- **Vendored modificato** (ripristino dopo re-clone: `git -C vendor/transformers-qwen3-moe-fused
  apply ../../patches/moe-fused-a087104-ardesia.patch`): in `qwen3_moe_fused/quantize_gguf/dequant.py`
  (a) **IQ2_XS dequant AGGIUNTO** (mancava upstream → KeyError con UD-quant; port dalla reference
  numpy gguf-py, **verificato bit-exact CPU+GPU**, anche path chunked >32768 blocchi);
  (b) **wrap dequant = eager CHUNKED, dynamo rimosso** (chunk 32768 blocchi ≈ 32 MiB fp32).
- **Perché niente dynamo sul dequant:** nel contesto trainer QUALCOSA impone recompile_limit=8 e
  fa fallire le guardie a ogni chiamata (7 run di evidenza, meccanismo NON root-caused — bump
  globali/top-of-file/pre-train, dynamic=True, warm-up, no outer compile: tutti inefficaci; in
  isolamento il dequant compilato funziona, vedi `/tmp/repro_dequant_limit.py` ricreabile).
  L'eager monolitico OOMa (intermedi fp32 ~768 MiB sui tensori fused). Chunked = terza via.
- `SFTConfig`: **torch_compile=False** nello smoke (outer compile rimosso durante la diagnosi;
  con dequant chunked potrebbe essere riattivabile — NON provato).
- **GPU condivisa col desktop**: tetto effettivo ~15 GiB a desktop leggero. Prima di ogni run
  controllare `nvidia-smi`: lo **studio unsloth lascia processi che tengono ~700 MiB** (successo
  oggi: 2 processi python di `~/.unsloth/studio` uccisi + VS Code).
- `hf download`: killare a metà orfana il parziale (riparte da zero su nuovo staging).
- Cache Triton/inductor: pulire tra cambi di config kernel (avvertenza autore); inductor in
  `/tmp/torchinductor_*`.

## 4. Fili aperti
- Throughput reale: 62 s/step è smoke-con-compile; misurare a regime e valutare se il dequant
  compilato è recuperabile (root-cause del limite-8: candidato = checkpointing/patch unsloth;
  da isolare con TORCH_LOGS=recompiles) o se si upstream-a il chunked.
- **Upstream: port IQ2_XS MERGIATO** (PR #23 → moe-fused a4f3c52, 2026-07-24). Conseguenza:
  un re-clone di vendor da master GIÀ include IQ2_XS; della nostra patch resta "solo nostro"
  il wrap chunked. L'autore invita su llama.cpp#25681 (gguf.torch_quants, sua PR: IQ2_XS già
  dentro) — eventuale contributo lì = commento di validazione (⚠ AI-disclosure obbligatoria su
  llama.cpp). Issue recompile-limit su moe-fused ancora da aprire (docs/03 §3).
- Nota di storia: `main` remoto e locale sono stati **riconciliati** con il merge commit `be33842`
  (strategia `ours`, storie non correlate). Se serve tornare allo stato pre-fix: branch
  `backup/pre-push-fix-20260728-013251`.
- **Prossimo passo tecnico (M6, ora sbloccato):** wire `transformers5-qwen3.5-recipe` +
  `transformers-gguf`, caricare il 35B APEX-I-Mini, **misurare VRAM forward-only** (⚠ fit: 13.33 GiB
  pesi vs ~15 effettivi). Solo dopo: copiare i dati da ardesia-unsloth (persona-v7, calibration-v1,
  `persona.py` — COPIA, mai symlink) e preparare il train vero. Servire **unmerged**.
- **Osservazioni lasciate aperte in sessione 4** (nessuna bloccante):
  (a) `grouped_mmq_pair_grad_input_q3_tiled_kernel` sembra codice morto upstream (nessun caller
  raggiungibile) — segnalato al maintainer, non toccato; (b) il kernel grouped a proiezione singola
  alla shape gate/up sta a ~0.19 del soffitto bf16, ma il training non passa di lì (usa il pair);
  (c) micro-ottimizzazione non fatta di proposito: sul layout C NVIDIA gli elementi (0,1) e (2,3)
  dell'accumulatore sono colonne adiacenti della stessa riga → lo store potrebbe usare 32 bit invece
  di due da 16. Non fatta perché le misure dicono che lo store non è il collo, e rimetterebbe un
  fatto di layout dentro i kernel che il seam esiste per nascondere.

## 5. Docket (user-gated, mai risolto dall'assistente)
- Nessuna decisione pendente. Risolte da Cristiano: target tier-jump → Path B diretto (`docs/00`
  §Docket); pausa del training fino alla PR → **decaduta**, PR aperta il 2026-07-24.
- Resta il ⚠ tecnico, non una decisione: **fit del 35B da misurare** (13.33 GiB di pesi contro ~15
  effettivi) — è la prima cosa di M6 e può invalidare il piano.
- Pulizia disco 2026-09-30 (via di Cristiano): cancellati tutti gli `artifacts/*-merged` e i GGUF bf16 tranne `base-4b-bf16.gguf`. Restano adapter (safetensors e GGUF), base e tutti i quant: un fuso si rifà da adapter + base.
