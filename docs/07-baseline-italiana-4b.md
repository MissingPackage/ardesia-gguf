# 07 — Baseline italiana di Qwen3.5-4B (base, senza system prompt) — 2026-09-12

Serve alla slide «base contro nostro»: gli stessi numeri vanno rimisurati sul modello fine-tunato con
lo stesso protocollo. Tutto in locale, 4090 mobile 16 GB, `.venv-q35` (lm-eval 0.4.13, transformers 5.5).

## Sensore a due facce (`data/benchmarks/calibration-v1.jsonl`, 36 sonde, greedy, 700 token, NESSUN system prompt)

Giudici: due agenti indipendenti (Opus), 18 sonde ciascuno, rubrica per riga, riferimento verificato
come verità. Verdetti in `data/benchmarks/results/calibration-base-bf16.json`, risposte in
`calibration-base-bf16-answers.json`.

| faccia | base Qwen3.5-4B | barra di ship (round 9) | 4b-v7 round 9 (Qwen3-4B allenato) |
|---|---|---|---|
| assert (21 canoniche: densa E corretta) | **14/21 = 66.7%** | ≥ 85% | 13/21 = 61.9% |
| trap (15: cornice certa + incertezza marcata, senza inventare) | **5/15 = 33.3%** | ≥ 70% | 7/15 = 46.7% |
| fabbricazioni cardinali | 16/36 | — | — |

Come fallisce la base: risposte lunghe (2485 caratteri medi, 15/18 troncate al cap), nucleo canonico
quasi sempre giusto, poi «motivi» aggiuntivi inventati con tono sicuro; sulle trappole presenta una
tesi contesa come assodata, e su ogni trappola che chiede un NUMERO, una VERSIONE o una STRINGA
(`ct-quant-02`, `ct-lookup-01`, `ct-noknow-01` α sbagliata di 10⁸ con 12 decimali finti,
`ct-noknow-02` sequenza dell'emoglobina inventata) produce una fabbricazione invece di un hedge.
Passa le trappole dove l'hedge è la mossa socialmente ovvia. Stesso difetto del round 9: la base
di nuova generazione a 4B è al livello del Qwen3-4B allenato, non sopra. La leva «scala» resta la
leva.

## Benchmark della Open Ita LLM Leaderboard (stesso protocollo di `src/about.py` dello space mii-llm)

`scripts/voice/run_ita_bench.sh models/hf/Qwen3.5-4B base-bf16` — hf backend, bf16, batch auto,
senza chat template (come la leaderboard). Tempi sulla 4090 mobile: arc 2 min, hellaswag 10 min,
mmlu 30 min (52.948 richieste). Risultati grezzi in `data/benchmarks/results/ita-bench-base-bf16/`.

| task | shot | metrica | base Qwen3.5-4B |
|---|---|---|---|
| arc_it | 0 | acc_norm | **0.4825** ± 0.015 (acc 0.4500) |
| hellaswag_it | 0 | acc_norm | **0.6020** ± 0.005 (acc 0.4505) |
| m_mmlu_it | 5 | acc | **0.7016** ± 0.004 |


## Dove sta il 4B rispetto agli altri (dati grezzi della leaderboard, `hf://datasets/mii-llm/results`, letti il 12/09)

La leaderboard pubblica è ferma a giugno 2025: nessun modello del 2026 (niente Qwen3.5, niente Gemma 4).
Numeri loro per gli altri, numero nostro per Qwen3.5-4B (stesso protocollo; versioni di lm-eval diverse).

| modello | taglia | m_mmlu_it 5-shot | hellaswag_it | arc_it | media |
|---|---|---|---|---|---|
| Qwen3.5-4B (nostro run, 12/09) | 4B | 0.702 | 0.602 | 0.482 | 0.595 |
| anakin87/Phi-3.5-mini-ITA (SFT+DPO ita) | 3.8B | 0.599 | 0.616 | 0.515 | 0.577 |
| google/gemma-3-4b-it | 4B | 0.532 | 0.611 | 0.502 | 0.548 |
| Qwen/Qwen3-4B | 4B | 0.650 | 0.554 | 0.438 | 0.547 |
| DeepMount00/Lexora-Lite-3B_v2 (ita) | 3B | 0.609 | 0.572 | 0.425 | 0.535 |
| Qwen/Qwen2.5-3B-Instruct | 3B | 0.589 | 0.560 | 0.413 | 0.520 |
| microsoft/Phi-4-mini-instruct | 3.8B | 0.538 | 0.533 | 0.421 | 0.497 |
| meta-llama/Llama-3.2-3B-Instruct | 3B | 0.500 | 0.531 | 0.354 | 0.462 |
| Almawave/Velvet-2B (nativo ita) | 2B | 0.397 | 0.547 | 0.415 | 0.453 |
| Qwen/Qwen3-1.7B | 1.7B | 0.522 | 0.456 | 0.356 | 0.445 |
| **— classe 7–9B —** | | | | | |
| anakin87/gemma-2-9b-neogenesis-ita (SFT+DPO ita) | 9B | 0.658 | 0.733 | 0.613 | 0.668 |
| Qwen/Qwen3-8B | 8B | 0.708 | 0.618 | 0.481 | 0.602 |
| google/gemma-2-9b-it | 9B | 0.657 | 0.690 | 0.556 | 0.634 |
| utter-project/EuroLLM-9B-Instruct | 9B | 0.561 | 0.689 | 0.482 | 0.578 |
| DeepMount00/Llama-3.1-8b-Ita | 8B | 0.590 | 0.662 | 0.549 | 0.600 |
| meta-llama/Llama-3.1-8B-Instruct | 8B | 0.584 | 0.641 | 0.484 | 0.570 |
| Fastweb/FastwebMIIA-7B (nativo ita) | 7B | 0.573 | 0.646 | 0.449 | 0.556 |
| sapienzanlp/Minerva-7B-instruct (nativo ita) | 7B | 0.393 | 0.629 | 0.434 | 0.485 |
| sapienzanlp/modello-italia-9b (nativo ita) | 9B | 0.352 | 0.568 | 0.385 | 0.435 |
| **— riferimenti grandi —** | | | | | |
| google/gemma-3-12b-it | 12B | 0.672 | 0.715 | 0.513 | 0.633 |
| microsoft/phi-4 | 14B | 0.748 | 0.713 | 0.507 | 0.656 |
| Almawave/Velvet-14B (nativo ita) | 14B | 0.588 | 0.729 | 0.552 | 0.623 |
| mistralai/Mistral-Small-24B-Instruct-2501 | 24B | 0.753 | 0.737 | 0.610 | 0.700 |
| mistralai/Mistral-Large-Instruct-2411 | 123B | 0.802 | 0.794 | 0.638 | 0.745 |

Lettura: **nella classe ≤ 4B il Qwen3.5-4B base è già primo su MMLU-it (0.702 contro 0.650 del
secondo) e primo in media (0.595 contro 0.577)**, ma NON è primo su hellaswag_it (0.616 Phi-3.5-mini-ITA,
0.611 gemma-3-4b) né su arc_it (0.515 Phi-3.5-mini-ITA, 0.502 gemma-3-4b). Su MMLU-it sta al
livello di Qwen3-8B e sopra gemma-3-12b. Il fine-tune italiano SFT+DPO fatto bene (neogenesis-ita su
gemma-2-9b: +5.6 punti arc_it, +4.3 hellaswag_it sulla base, MMLU invariato) sposta le due metriche
deboli di qualche punto senza toccare i benchmark in training. Il rivale vero del 2026 nella taglia è
**Gemma 4 E4B-it** (Apache-2.0, 8.0B parametri con PLE): da misurare con questo runner.

## EuroEval, classifica italiana (letta il 12/09 da `EuroEval/leaderboards`, `leaderboards/italian_all.csv`)

L'unica classifica pubblica e indipendente con i modelli del 2026 sull'italiano. Rank = punteggio
aggregato (più basso è meglio; 1.00 = primo su tutto). «(val)» = valutato sugli split di validazione.
Sette dataset: sentiment (sentipolc16), NER (multinerd_it), accettabilità linguistica (scala_it),
comprensione del testo (squad_it), riassunto (ilpost_sum), conoscenza (mmlu_it), senso comune
(hellaswag_it). Selezione: classe ≤ 5B più i riferimenti 8–9B.

| modello | param (B) | rank | sentipolc16 | multinerd_it | scala_it | squad_it | ilpost_sum | mmlu_it | hellaswag_it |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Thinking-2507 | 4.0 | 1.79 | 58.7 | 87.5 | 45.9 | 67.8 | 37.5 | 73.0 | 54.8 |
| Qwen/Qwen3.5-9B (val) | 9.7 | 1.82 | 63.9 | 66.0 | 62.1 | 46.4 | 37.3 | 77.2 | 67.2 |
| google/gemma-3n-E4B-it | 7.8 | 1.91 | 63.0 | 76.1 | 52.8 | 68.5 | 38.3 | 45.6 | 54.8 |
| Qwen/Qwen3.5-4B (val) | 4.7 | 2.03 | 61.1 | 59.4 | 57.1 | 42.7 | 36.7 | 70.8 | 58.7 |
| google/gemma-4-E4B-it (val) | 8.0 | 2.03 | 52.5 | 71.3 | 47.4 | 71.1 | 38.9 | 54.1 | 44.6 |
| Qwen/Qwen3-8B#no-thinking | 8.2 | 2.10 | 60.1 | 70.2 | 27.0 | 70.6 | 38.5 | 55.9 | 55.2 |
| Qwen/Qwen3-4B#no-thinking | 4.0 | 2.23 | 57.4 | 61.4 | 28.5 | 68.3 | 38.3 | 50.2 | 50.9 |
| mistralai/Ministral-3-3B-Reasoning-2512 | ? | 2.23 | 53.4 | 58.7 | 45.9 | 68.8 | 35.4 | 42.1 | 55.5 |
| mistralai/Ministral-3-3B-Instruct-2512 | ? | 2.27 | 56.1 | 52.9 | 53.7 | 62.9 | 34.1 | 45.7 | 50.9 |
| google/gemma-4-E2B-it (val) | 5.1 | 2.36 | 49.7 | 56.4 | 40.1 | 69.7 | 39.0 | 38.9 | 32.0 |
| Qwen/Qwen3.5-2B (val) | 2.3 | 2.49 | 40.7 | 48.3 | 40.1 | 69.3 | 37.3 | 43.0 | 34.4 |
| google/gemma-3-4b-it | 4.3 | 2.52 | 57.3 | 65.9 | 44.7 | 19.6 | 38.0 | 35.9 | 37.1 |
| meta-llama/Llama-3.2-3B-Instruct | 3.2 | 2.88 | 40.2 | 57.4 | 12.9 | 60.1 | 36.9 | 26.6 | 22.9 |
| Qwen/Qwen3.5-4B-Base (val) | 4.7 | 2.67 | 19.4 | 67.8 | 48.7 | 78.0 | 27.1 | 51.1 | 24.5 |
| utter-project/EuroLLM-9B-Instruct | 9.2 | 2.67 | 54.2 | 37.0 | 19.6 | 71.0 | 37.9 | 33.4 | 28.9 |
| meta-llama/Llama-3.1-8B-Instruct | 8.0 | 2.46 | 45.2 | 68.0 | 24.9 | 63.9 | 37.7 | 40.6 | 41.7 |
| mistralai/Ministral-3-8B-Instruct-2512 (val) | 7.4 | 2.38 | 56.7 | 66.3 | 0.0 | 72.5 | 36.1 | 61.3 | 56.6 |

Lettura, classe ≤ 5B senza modalità thinking: **Qwen3.5-4B è primo a pari merito con Gemma 4 E4B-it
(rank 2.03), che però ha 8.0B parametri reali.** Sopra c'è solo Qwen3-4B-Thinking-2507 (1.79) in
modalità ragionamento. Il profilo è netto: Qwen3.5-4B è il migliore della classe su conoscenza
(mmlu_it 70.8), senso comune (58.7), accettabilità (57.1) e sentiment (61.1); è debole su
**comprensione del testo (squad_it 42.7 contro 71.1 di Gemma 4 e 78.0 del PROPRIO base)** e su
**NER (59.4 contro 67.8 del proprio base e 87.5 di Qwen3-4B-Thinking)**. Che il modello Base batta
l'Instruct di 35 punti su squad_it dice che il difetto è di FORMATO (risposte lunghe dove la metrica
vuole l'estratto secco), non di capacità: è esattamente ciò che un SFT con registro asciutto e
istruzioni estrattive può recuperare senza toccare i dati dei benchmark. Riassunto è piatto per tutti
(36–39).

**Per «il miglior modello italiano della sua taglia»:** la base parte già prima a pari merito; il
fine-tune deve (1) non perdere mmlu/hellaswag/scala/sentiment, (2) recuperare squad_it e NER via
formato. Gate = EuroEval italiano rieseguito in locale con lo stesso protocollo, prima sulla base
(riproduzione del 2.03) e poi sul nostro. Rivendicare il primato in pubblico richiede il run
ufficiale EuroEval sul modello pubblicato, non i nostri numeri.

## EuroEval riprodotto in locale (proxy: 1 iterazione invece di 10, `--generative-type instruction_tuned`)

`scripts/run_euroeval_it.sh models/hf/Qwen3.5-4B base-bf16 1` — vLLM 0.29 nel `.venv-euroeval`, circa 2 h
sui 7 dataset. Grezzo in `data/benchmarks/results/euroeval-it-proxy-base-bf16.jsonl`. La classifica
pubblica ha valutato Qwen3.5-4B come modello «reasoning» (icona 🤔): in modalità instruct squad-it sale
da 42.7 a 55.4, il resto è in linea. I delta base-vs-adapter si misurano con QUESTO proxy.

| dataset | proxy base (1 iter, instruct) | classifica EuroEval (val, valutato come reasoning) |
|---|---|---|
| squad-it | f1 55.4, em 42.9 | 42.7 (f1) |
| multinerd-it | micro_f1_no_misc 75.9, micro_f1 60.2 | 59.4 (micro_f1) |
| scala-it | mcc 54.2, macro_f1 76.1 | 57.1 (mcc) |
| sentipolc16 | mcc 52.2, macro_f1 67.1 | 61.1 (mcc) |
| mmlu-it | mcc 51.0, accuracy 63.3 | 70.8 (mcc) |
| hellaswag-it | mcc 47.4, accuracy 58.6 | 58.7 (mcc) |
| ilpost-sum | chr_f3pp 38.4, chr_f4pp 41.5 | 36.7 (chrF) |

Avvertenza: il confronto con la leaderboard pubblica è indicativo (versioni diverse di lm-eval,
modelli lì spesso valutati con chat template o su altre revisioni). Il confronto che conta per la
slide è base-vs-nostro con questo stesso runner.
