# 01 — Portabilità su CUDA: assessment (evidence-based)

> **⚠ ERRATA (2026-07-24, spike-1):** la conclusione "la GGUF `mudler/Qwen3.6-35B-A3B-APEX` è
> caricabile sul Path A" è **sbagliata**. Valutava solo la copertura dei *quant-type*, non
> l'**architettura**: Qwen3.6-35B-A3B è `qwen3_5_moe` — ibrido **GatedDeltaNet** (3 layer
> linear-attention ogni full-attention, `full_attention_interval: 4`) + VLM wrapper + shared expert
> + MTP (config.json di `Qwen/Qwen3.6-35B-A3B`). `Qwen3MoeFusedForCausalLM` modella solo il
> `qwen3_moe` full-attention puro (grep: zero supporto qwen3_5/GDN nel repo; transformers 4.57
> non registra `qwen3_5_moe`). Il post "35B in 16GB" era sullo stack della *ricetta transformers5*
> (= Path B), non su moe-fused. **Conseguenza: il 35B su Path A non è raggiungibile; per il 35B
> serve il Path B. Path A resta valido per i modelli `qwen3_moe` (es. Qwen3-30B-A3B).**

Fatto leggendo le sorgenti dei tre repo woct0rdho (clone in `/tmp/ggml-port`, 2026-07-24). Domanda:
*"LoRA over GGUF si può portare/usare sul nostro 4090 CUDA?"* — **Sì**, e ci sono **due vie indipendenti**,
una delle quali **non richiede scrivere kernel**.

## Path A — `transformers-qwen3-moe-fused`: gira su CUDA OGGI (nessun kernel da scrivere)

**Evidenza:**
- `example_train_30b_a3b_gguf.py`: `device = "cuda"`, stack **Triton + Unsloth + torch_compile
  max-autotune**, header *"Runs with 16 GB VRAM using UD-IQ3_XXS"* (quant 3-bit). Serve LoRA su base GGUF
  con dequant on-demand (quantizer stile `Bnb4BitHfQuantizer`).
- `requirements.txt` = puro stack PyTorch/Triton (`torch, triton, transformers, peft, trl, unsloth,
  bitsandbytes, gguf`). **Nessun ROCm, nessun CK.** Installa sul 4090 così com'è.
- README: *"mainly optimized for RTX 3090 and RTX 4090"*. La grouped-GEMM di default è **Triton**
  (backend CUTLASS/CK/Helion alternativi in `grouped_gemm/`). Triton = CUDA-native.
- `quantize_gguf/dequant.py` dequantizza **tutti** i quant-type dell'APEX-I-Mini: la mappa `dequantize`
  include BF16, Q8_0, Q6_K, Q5_K, Q4_K, **Q3_K**, Q2_K, IQ4_XS, IQ3_S, IQ3_XXS, **IQ2_S**, IQ2_XXS,
  IQ1_M/S. APEX-I-Mini usa {IQ2_S, Q3_K, Q4_K, Q5_K, Q6_K} → **tutti presenti** → la GGUF
  `mudler/Qwen3.6-35B-A3B-APEX-GGUF` è caricabile su questo path.

**Conseguenza:** una Ardesia **35B-A3B** allenata via LoRA-over-GGUF **sul nostro 4090** è raggiungibile
**senza scrivere kernel** — solo wiring: scaricare la GGUF, ri-puntare l'esempio, misurare VRAM/throughput.

**Caveat da verificare noi (non assunti):**
1. L'esempio è **30B-A3B**; il 35B-A3B va ri-misurato per VRAM/velocità sul 4090.
2. Il fast-forward MoE su GGUF è **TODO** nel codice (`patch_Qwen3MoeFusedSparseMoeBlock_forward` commentato):
   funziona ma può essere più lento del path AMD pienamente fuso.
3. Il dequant è **per-tensore on-demand** (dequantizza il peso dell'esperto attivo a bf16 al forward) → usa
   VRAM transitoria; con MoE a 3B attivi è gestibile ma va misurato.
4. Transformers **4** + Unsloth (vicino al nostro `ardesia-unsloth`, ma **deps separate** comunque).

## Path B — `torch-ggml-ops` → CUDA: il port da condividere (lavoro di kernel, delimitato)

Questo è il path più efficiente (kernel MMQ quantizzati che **non dequantizzano mai il tile intero** → meno
VRAM/più veloce dell'approccio per-tensore del Path A; è ciò che dà i 13.3 GiB APEX + 6.5 s/it su Strix Halo)
e alimenta la ricetta `transformers5`.

**Superficie del port (dalle sorgenti):**
- **Forward = già CUDA.** `csrc/vendor/llama_cpp/*` sono i kernel MMQ di llama.cpp (target primario CUDA/nvcc).
  `setup.py` usa `CUDAExtension` (nvcc); l'hipify scatta **solo** su ROCm. `mmq_hip.cu` fa solo `#undef` di
  macro half HIP. Su box CUDA, nvcc compila il forward nativamente.
- **Backward = l'unico pezzo AMD-specifico.** 3 kernel (`csrc/ck/{mmq_backward, grouped_mmq_backward,
  grouped_mmq_backward_tiled}.cuh`) che usano:
  - l'intrinsic RDNA3 **`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32`** (`bf16_wmma.cuh`) — la MMA gfx11;
  - tipi `__hip_bfloat16`, header `hip/hip_bf16.h` / `hip/hip_runtime.h`;
  - un layout C-fragment per-lane specifico di RDNA3 (`c_row = lane&15`, `c_column = 2*el + lane>>4`);
  - wave size 32 (`GROUPED_BACKWARD_WAVE_SIZE`).
- **Il port CuTe** = per ciascuno dei 3 kernel: sostituire la MMA amdgcn con la **tensor-core NVIDIA**
  (CuTe `mma_atom`, oppure PTX `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32`), **riscrivere l'indicizzazione
  del fragment** al layout di output NVIDIA, e swap `__hip_bfloat16`→`__nv_bfloat16` / `cuda_bf16.h`. La
  matematica (dequant tile→bf16→MMA→accumula) è identica. **Wave32 AMD ↔ warp32 NVIDIA mappano puliti**
  (niente wave64). Serve dimestichezza con i fragment layout tensor-core, ma è **giorni-settimane, non ricerca**.
- L'autore stesso: *"it should be straightforward to port them to CuTe on Nvidia GPUs"* (README §backward).

**Payoff:** porta l'APEX/sub-4-bit + la ricetta `transformers5` su NVIDIA — **contributo upstream che nessuno
ha ancora fatto** (il "buon risultato da condividere"). Licenza Apache-2.0, PR-abile a monte.

## Raccomandazione

**Path A prima, Path B come contributo.**
1. **Path A** valida il vero obiettivo (Ardesia-35B-A3B su CUDA, verifica l'ipotesi "la scala cura la
   fabbricazione") **a rischio-kernel zero**, e ci insegna lo stack. Se il fit/throughput sul 4090 regge,
   abbiamo la Ardesia grande **senza aspettare il port**.
2. **Path B** si decide *dopo*: se ci serve il vantaggio VRAM/velocità dell'APEX, o vogliamo restituire il
   port CUDA alla community, si fa il lavoro CuTe sul backward (delimitato). È il risultato condivisibile.

**Regola invariata:** serve **unmerged** (LoRA bf16 su base quantizzato), misura su `calibration-v1`
(assert ≥85% / trap ≥70%) comparabile a round-9.

## Prossimo passo concreto (Path A, spike-1)
1. `uv venv` dedicato + `pip install -e .` di `moe-fused` (transformers 4 + triton + unsloth) su torch CUDA ≥2.10.
2. `hf download mudler/Qwen3.6-35B-A3B-APEX-GGUF …-I-Mini.gguf` — misurare footprint reale su disco/VRAM.
3. Micro-smoke: ri-puntare `example_train_30b_a3b_gguf.py` alla GGUF 35B + un pugno di righe → **osservare che
   il loop gira e la VRAM sta sotto 16 GB** (batch 1, ctx 2048, rank 4). NIENTE training vero finché lo smoke passa.
4. Se il fit non torna a 35B: quant più spinto (IQ2), oppure restare 30B-A3B, oppure passare a Path B.

## Riferimenti sorgente (clone /tmp/ggml-port, 2026-07-24)
- `transformers-qwen3-moe-fused/example_train_30b_a3b_gguf.py`, `qwen3_moe_fused/quantize_gguf/dequant.py`,
  `requirements.txt`, README §"LoRA over GGUF".
- `torch-ggml-ops/setup.py`, `README.md` §backward, `csrc/ck/bf16_wmma.cuh`, `csrc/ck/grouped_mmq_backward.cuh`,
  `csrc/vendor/llama_cpp/`.
- GGUF 35B: https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF (file `Qwen3.6-35B-A3B-APEX-I-Mini.gguf`).
