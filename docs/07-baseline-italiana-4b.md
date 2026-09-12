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

Avvertenza: il confronto con la leaderboard pubblica è indicativo (versioni diverse di lm-eval,
modelli lì spesso valutati con chat template o su altre revisioni). Il confronto che conta per la
slide è base-vs-nostro con questo stesso runner.
