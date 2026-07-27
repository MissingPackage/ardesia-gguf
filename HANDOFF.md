# HANDOFF — Ardesia-GGUF (aggiornato 2026-07-24, sessione 4 — Path B su CUDA + tensor core, PR APERTA)

Entry point per una sessione fresca. Leggi: questo file → `README.md` → `docs/00` (docket incluso) →
`docs/01` (**con 2 errata in testa**) → `docs/04` (log spike-2, il port) → `CLAUDE.md`.
Repo auto-contenuto rispetto a `../ardesia-unsloth`.

📄 **`docs/05-porting-kernel-cuda.md`** = write-up autonomo sul porting dei kernel (sintesi +
appendice tecnica), scritto per essere **condiviso fuori dal team**. Non è un log: `docs/04` resta il
log cronologico dell'esperimento, `docs/05` è il documento presentabile. Se cambiano i numeri, vanno
aggiornati entrambi.

## 1. Next decidable

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

**Working tree main repo NON committato** (docs/04, docs/port-microtests, scripts/pathb-*, patches/*,
HANDOFF, docs/01, .gitignore). Sono su disco (una sessione fresca li legge); committarli è a
discrezione di Cristiano (regola: commit solo su richiesta).

## 2. Stato (sessione 2, 2026-07-24)
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
- Working tree main repo NON committato (scaffold + docs + scripts + .gitignore) — decidere cosa
  fissare (regola: commit solo su richiesta).
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
