# HANDOFF — Ardesia-GGUF (aggiornato 2026-07-24, sessione 2 — spike-1 CONCLUSO: PASS)

Entry point per una sessione fresca. Leggi: questo file → `README.md` → `docs/00` (docket incluso) →
`docs/01` (**con errata in testa**) → `CLAUDE.md`. Repo auto-contenuto rispetto a `../ardesia-unsloth`.

## 1. Next decidable
**DECISO da Cristiano (2026-07-24): Path B diretto** (round-10 su 2-bit scartato — quant come
variabile confusa; docket in `docs/00`). **Spike-2, prima sessione fresca:**
1. **Venv NUOVO e separato** (il `.venv` attuale è transformers 4 per Path A — NON riusarlo):
   torch CUDA ≥2.10 + il **fork transformers dell'autore** (`woct0rdho/transformers` branch
   `gguf`, tracking huggingface/transformers#40070) + clone di `torch-ggml-ops` e
   `transformers5-qwen3.5-recipe`.
2. **Build-test del forward su CUDA** (domanda 1 di docs/00 §spike-zero): `setup.py` usa
   `CUDAExtension`/nvcc, dovrebbe compilare così com'è — verificare, girare i suoi test.
3. Poi il port dei **3 kernel backward** CK→CuTe (`csrc/ck/{mmq,grouped_mmq,grouped_mmq_tiled}
   _backward`): MMA `__builtin_amdgcn_wmma_...bf16` → `mma.sync`/CuTe, riscrivere l'indicizzazione
   fragment per-lane, `__hip_bfloat16`→`__nv_bfloat16`. Pezzi AMD-only della ricetta (AITER, GEMM
   ROCm-tuned) da sostituire con equivalenti CUDA.
4. La GGUF target è GIÀ in `models/` (Qwen3.6-35B-A3B-APEX-I-Mini, 13.33 GiB). ⚠ Fit a rischio:
   13.33 di soli pesi vs ~15 GiB effettivi — misurare presto un forward-only.
5. Opzionale prima di iniziare: mandare a woct0rdho il materiale di `docs/03` (PR IQ2_XS + issue) —
   apre il canale con l'autore prima del port.

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
- **Upstream-abili a woct0rdho** (Apache-2.0, "buon risultato da condividere"): port IQ2_XS
  (verificato) + chunked eager wrap + issue sul recompile-limit in contesto unsloth.
- Working tree NON committato (scaffold + docs + scripts + .gitignore) — decidere cosa fissare.
- Prossimo passo tecnico se opzione (1): copiare dati da ardesia-unsloth (persona-v7,
  calibration-v1, persona.py — COPIA, mai symlink) e preparare il train vero.

## 5. Docket (user-gated, mai risolto dall'assistente)
- Nessuna decisione pendente. L'ultima (target tier-jump) è RISOLTA da Cristiano: Path B diretto
  (`docs/00` §Docket). Resta il ⚠ tecnico: fit del 35B da misurare presto (§1 punto 4).
