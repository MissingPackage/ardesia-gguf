# Porting su NVIDIA dei kernel di training LoRA-over-GGUF

**Cosa:** portare su CUDA una libreria di kernel per il training di LoRA su modelli GGUF quantizzati,
scritta per GPU AMD, e farlo con qualità da contributo upstream anziché da patch locale.
**Esito:** backward su tensor core nativi, **4–11× più veloce** della prima versione portabile,
**67/67 test**, PR aperta su `woct0rdho/torch-ggml-ops` (2026-07-24).
**Hardware:** RTX 4090 Laptop (Ada, sm_89), CUDA 13.3, PyTorch 2.13.

---

## Sintesi

### Perché serviva

L'obiettivo del progetto è addestrare un **Qwen3.6-35B-A3B** (MoE, 256 esperti) con LoRA dentro i
**16 GB** di una GPU portatile. L'unico modo per farci stare i pesi è tenerli **quantizzati in GGUF**
e dequantizzarli al volo dentro i kernel, invece di materializzarli in memoria. La libreria che fa
esattamente questo — `torch-ggml-ops` — esiste ed è ottima, ma è scritta per **AMD**: usa
un'istruzione matrix-core specifica delle GPU gfx11 (`wmma_f32_16x16x16_bf16_w32`). Su NVIDIA non
compila nemmeno.

Senza questo porting, tutto il ramo tecnico del progetto era bloccato.

### Cosa abbiamo costruito

La libreria è più portabile di quanto sembri: il *forward* deriva da llama.cpp ed è già dual-path.
L'unica parte davvero legata all'hardware AMD è **un header di 42 righe** che espone un'operazione
matrice 16×16×16 in bfloat16 a circa 4000 righe di kernel di backward.

Il problema è che quell'header non era un'astrazione: **faceva trapelare nei kernel il modo preciso in
cui AMD distribuisce i dati tra i thread**. NVIDIA li distribuisce diversamente, e la differenza non è
cosmetica — il vecchio contratto era *matematicamente inesprimibile* sul layout NVIDIA (dettaglio in
appendice A).

La scelta di progetto è stata: invece di duplicare ~13 varianti di kernel con un ramo `#if AMD /
#else NVIDIA` (≈800 righe di codice quasi-copiato, in una codebase mantenuta da una persona che ha
solo hardware AMD), **allargare l'astrazione** finché i kernel non hanno più bisogno di sapere su
quale hardware girano. Risultato: **nessun kernel contiene un ramo per piattaforma**, e i kernel sono
diventati *più corti* (+438/−301 righe sui 4 file di backward).

### Risultati

Rispetto alla prima versione funzionante (un'emulazione portabile, corretta ma lenta):

| kernel | prima | dopo | speedup |
|---|---|---|---|
| `grouped_mmq_grad_input` (5 tipi di quantizzazione) | 1,25–4,65 ms | 0,32–1,18 ms | **3,9× – 4,3×** |
| `grouped_mmq_pair_grad_input` | 8,69–8,97 ms | 0,83–0,90 ms | **9,7× – 10,8×** |

In termini assoluti, il backward gira ora a **0,6–0,78 di un GEMM bf16 di cuBLAS che non paga alcuna
dequantizzazione** — cioè la moltiplicazione di matrici non è più il collo di bottiglia; quello che
resta è il costo di decodificare il formato GGUF, che è inerente al design (i pesi *devono* restare
quantizzati: è tutto il punto della libreria).

### Come lo sappiamo

- **67/67 test** sul modello reale da 35B (58 esistenti + 9 aggiunti).
- **Verifica alle shape di produzione**: contro un riferimento fp32 costruito con gli input esatti del
  kernel, **il 99,88–99,99% degli output è bit-identico** al valore correttamente arrotondato. I pochi
  che non lo sono sono cancellazioni vicine a zero, dove qualsiasi ordine di somma valido diverge.
- **Prova di copertura**: con `torch.profiler` verifichiamo *quale kernel gira davvero*, invece di
  dedurlo. È così che abbiamo scoperto che la suite di test upstream non toccava affatto i kernel che
  il modello reale usa (appendice C).

### Cosa non è verificato

**Il ramo AMD.** Non abbiamo hardware ROCm: non possiamo compilarlo, tantomeno eseguirlo. Il porting è
costruito per non toccarlo (il codice AMD di ogni funzione è letteralmente il ciclo che prima stava
inline), ma resta l'unico punto senza evidenza sperimentale — ed è dichiarato esplicitamente nella PR,
chiedendo al maintainer di controllarlo.

Secondo limite dichiarato: il backward richiede ora **compute capability 8.0+** (Ampere), perché
l'istruzione tensor core bf16 usata non esiste su Turing. La build fallisce con un messaggio chiaro
anziché con un errore incomprensibile dell'assembler.

### Valore oltre il progetto

Il lavoro è stato fatto in forma **upstreamabile**, non come fork privato: la PR aggiunge il supporto
NVIDIA preservando il build AMD, e include due correzioni che giovano anche agli utenti AMD (un test
che validava contro un riferimento sbagliato, e nove test per kernel che non erano coperti da nulla).
Se viene accettata, il progetto smette di dipendere da una patch locale da riapplicare a ogni
aggiornamento.

---

# Appendice tecnica

## A. Il problema vero: due layout matrix-core incompatibili

Sia AMD che NVIDIA offrono un'istruzione hardware che calcola un prodotto matriciale 16×16×16 in
bfloat16 con accumulo in fp32. Calcolano la stessa cosa, ma **distribuiscono i dati tra i 32 thread di
un warp in modi completamente diversi**.

Nel contratto originale (AMD gfx11, wave32):

- un thread possiede la riga `lane & 15` di *entrambi* gli operandi, e ne carica tutti i 16 valori;
- dopo l'operazione, l'elemento `e` dell'accumulatore di quel thread è `C[2*e + lane/16][lane & 15]`.

Il punto critico è l'ultima riga: **la coordinata di colonna dell'accumulatore dipende solo dal thread,
non dall'elemento**. I kernel sfruttavano questo fatto ovunque, scrivendo cose come:

```cpp
output_row    = base + c_column(lane, element);   // dipende dall'elemento
output_column = base + c_row(lane);               // NON dipende dall'elemento
```

Su NVIDIA (`mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32`) la colonna **dipende anche dall'elemento**.
Non è una differenza che si aggira cambiando una formula: il contratto stesso non era in grado di
descrivere il layout NVIDIA. Ecco perché non bastava "reimplementare l'header".

Un secondo dettaglio, meno concettuale ma altrettanto costoso: l'astrazione di tile di llama.cpp che
abbiamo riusato calcola gli indici leggendo `threadIdx.x` **assumendo che coincida con la corsia del
warp**. È vero nei kernel di forward, falso nel backward, che lancia blocchi piatti da 128 o 256
thread. Usarla direttamente manda i warp 1–3 fuori dai limiti di memoria. Un test a 32 thread non lo
rivela: serve almeno 4 warp.

## B. La decisione: allargare il seam, non forkare i kernel

Due strade:

| | duplicare i kernel | allargare l'astrazione |
|---|---|---|
| diff | ~800 righe di CUDA quasi-copiato in 13 kernel | +438/−301, kernel più corti |
| rischio AMD | nullo (non si tocca) | il codice AMD cambia forma, anche se non sostanza |
| manutenibilità | ogni modifica futura va fatta due volte | una volta sola |
| accettabilità upstream | bassa | alta |

Abbiamo scelto la seconda, mitigando il rischio AMD in modo specifico: **il ramo AMD di ogni nuova
funzione è, riga per riga, il ciclo che prima stava inline nel kernel**, compreso un parametro di
tuning (`VECTOR_LOAD`) che esisteva già e che abbiamo propagato invece di normalizzare.

Concretamente, i kernel hanno smesso di indicizzare i frammenti a mano:

| prima | dopo |
|---|---|
| `bf16_fragment` per entrambi gli operandi | `bf16_fragment_a` / `bf16_fragment_b` |
| `fragment_data(f)` + un ciclo di riempimento scritto a mano in ogni punto | `load_a_fragment<CHECK_M, CHECK_K>(...)`, `load_b_fragment(...)` |
| `acc.values[e]` | `acc.value(e)` |
| `c_column(lane,e)` per la riga, `c_row(lane)` per la colonna | `acc_m(lane,e)` / `acc_n(lane,e)` |

L'ultima riga è il cuore: rendendo *entrambe* le coordinate dipendenti dall'elemento, il contratto
diventa capace di descrivere entrambi gli hardware. Su AMD `acc_n` ignora semplicemente l'argomento.

Superficie totale toccata: ~40 punti meccanici (8 riempimenti dell'operando A, 20 dell'operando B, 6
store, 12 dichiarazioni di accumulatore), non 2900 righe da riscrivere. Averlo misurato *prima* di
iniziare ha cambiato la stima da "giorni" a "ore".

## C. Correttezza: due trappole che producono numeri convincenti e falsi

**1. Un test che falliva per colpa del proprio oracolo.**
Un test upstream confrontava il nostro risultato con un prodotto matriciale bf16 di PyTorch,
pretendendo uguaglianza **bit a bit**. Falliva su 117.611 elementi su 320.000. La lettura ovvia — "il
nostro kernel sbaglia" — era sbagliata.

PyTorch ha `allow_bf16_reduced_precision_reduction = True` come default: permette al backend di
sommare i risultati parziali **in bf16**. Valutando entrambi contro una somma in fp64 degli stessi
prodotti:

| | output che sono il valore correttamente arrotondato |
|---|---|
| riferimento del test (default) | 202.389 / 320.000 — **63,2%** |
| stesso riferimento, flag disattivato | 319.995 / 320.000 — 99,998% |
| il nostro kernel | 319.993 / 320.000 — **99,998%** |

Sui 117.611 elementi in disaccordo, **il kernel aveva ragione 117.605 volte, il riferimento 1**. La
firma è inequivocabile: il riferimento è corretto al 100% su gruppi da 1, 15 e 16 righe e crolla al
~60% da 17 in su — esattamente dove cuBLAS cambia strategia di tiling.

Il fix corretto non era allentare la tolleranza, era **riparare l'oracolo**. Fatto quello, il residuo
è di 4 elementi su 320.000, ciascuno esattamente 1 ULP bf16: irriducibile, perché due ordini di somma
fp32 entrambi validi possono cadere ai due lati di un arrotondamento.

**2. Una suite che non testava i kernel di produzione.**
Tutti i test usavano `out_features=37`. I kernel ottimizzati sono selezionati da un gate esatto
(`out_features==2048 && in_features==512`, la proiezione *down* di un FFN MoE) più soglie sul numero
di righe. Con 37 non si raggiungono **per costruzione**. In pratica: i kernel che il modello reale
esegue non erano coperti da nulla.

Abbiamo aggiunto 9 casi (3 tipi di quantizzazione × 3 regimi di righe) e verificato con
`torch.profiler` che coprono **8 kernel distinti mai raggiunti prima**.

## D. Misurare senza prendersi in giro

La conclusione sulle prestazioni è stata sbagliata **due volte** prima di essere giusta. Vale la pena
registrare come, perché sono errori generici, non specifici di questo progetto.

**Errore 1 — la shape sbagliata.** Misurando tutti i tipi di quantizzazione a un `out_features=512`
fisso, un kernel risultava a 0,22 del soffitto, e sembrava fosse quello usato dalle proiezioni
gate/up. Doppiamente falso: gate/up passa da un'operazione *accoppiata* (gate e up condividono lo
stesso input, quindi esiste un kernel che le fa insieme), e 512 non è la shape della proiezione down.
Stavamo cronometrando con precisione un percorso che il training non esegue mai.

**Errore 2 — il soffitto sbagliato, e rapporti tra run diversi.** Usando il *forward* come termine di
paragone il backward risultava a 0,75–0,95 e "non c'è più margine". Due errori indipendenti:

- i due numeri venivano da **esecuzioni diverse**, e questo portatile deriva del 5–15% tra un run e
  l'altro (`nvidia-smi -q -d PERFORMANCE` mostra `SW Power Capping` che accumula);
- soprattutto: **il forward non è un paragone equo**. Quantizza le attivazioni a 8 bit e gira su
  tensor core **int8**, che su Ada hanno circa il doppio del throughput del bf16. Confrontare un
  kernel bf16 con uno int8 lo fa sembrare migliore di quanto sia.

**Il soffitto corretto** è lo stesso prodotto matriciale in bf16 con i pesi **già dequantizzati**
(cuBLAS che fa esattamente l'aritmetica che facciamo noi, e zero decodifica GGUF), cronometrato
**nello stesso run**:

| percorso reale del training | nostro | GEMM bf16 senza dequant | rapporto |
|---|---|---|---|
| gate/up (accoppiato), Q3_K | 19,6 TFLOP/s | 31,5 | 0,62 |
| gate/up (accoppiato), IQ2_S | 20,8 | 26,7 | 0,78 |
| down, Q4_K | 22,4 | 37,3 | 0,60 |
| down, Q5_K | 25,5 | 40,5 | 0,63 |

Regole che ne ricaviamo, valide ben oltre questo caso:

1. **misurare le shape che il sistema esegue davvero**, non una griglia comoda;
2. **prendere i rapporti dentro lo stesso run** su hardware che deriva;
3. **scegliere un oracolo che faccia lo stesso lavoro**: un confronto con un'implementazione che salta
   una fase costosa, o che usa aritmetica più economica, non è un soffitto;
4. **una tolleranza numerica va tarata sulla scala reale**: una soglia assoluta calibrata su riduzioni
   corte diventa priva di senso quando la riduzione è 55 volte più lunga;
5. **le metriche relative per-elemento esplodono sulle cancellazioni**: su somme con segno, una
   frazione degli output è vicina a zero e lì l'errore relativo è illimitato per *qualsiasi* ordine di
   somma. In un passaggio intermedio questo ci ha fatto misurare "2,7 milioni di ULP di errore" su un
   kernel che era in realtà bit-esatto.

## E. Stato e cosa resta

- PR aperta su `woct0rdho/torch-ggml-ops` (10 file, +660/−322, 4 commit tematici).
- Da chiarire col maintainer: verifica del ramo AMD; le due modifiche ai test (dichiarate e motivate
  coi numeri nel testo della PR); la soglia sm_80, per cui offriamo in alternativa un fallback Turing.
- Non fatto di proposito: il kernel a proiezione singola sulla shape gate/up sta a ~0,19 del soffitto,
  ma il training non passa di lì. Sistemarlo è una ristrutturazione con effetti sul ramo AMD, che non
  possiamo validare — quindi è segnalato al maintainer, non implementato.
- Prossimo passo del progetto: misurare l'occupazione di VRAM del modello da 35B in forward, che è il
  vincolo che decide se l'intero piano sta in 16 GB.

---

*Strumenti prodotti e riutilizzabili: un benchmark che confronta due build e include il soffitto equo
(`scripts/bench_backward.py`), un verificatore di correttezza alle shape di produzione che prova quale
kernel viene eseguito (`scripts/verify_backward.py`), e microtest standalone dei layout dei frammenti
(`docs/port-microtests/`).*
