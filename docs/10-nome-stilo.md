# 10 — Il modello cambia nome: Tabula diventa Stilo (nota del 15 ottobre 2026)

**Decisione di Cristiano (15/10):** il modello di questa repo si chiamerà **Stilo**. Il nome Tabula
passa all'app che fonde testo, immagini e decisioni nel browser (MissingPackage/tabula). Sulla tabula si
scriveva con lo stilo. Il cambio si fa **al prossimo training**: fino ad allora pesi, file e risultati
restano Tabula.

**Al prossimo training (voce v7):**
1. **Dati d'identità.**
   - Nuova versione delle risposte d'identità, con Stilo al posto di Tabula.
   - Non è un `sed`: Tabula è femminile, Stilo maschile («sono Stilo, il modello…»). Vanno rilette le
     frasi con articoli e aggettivi accordati.
2. **La ricetta della v6 non è in questa repo.**
   - Qui gli script arrivano a `queue_tabula_v4.sh` e i dati a `persona-tabula-v4.jsonl`.
   - La v5 e la v6 (dati, `queue_tabula_v6.sh`) vivono sulla Fedora: vanno pubblicate prima di partire.
3. **Script.**
   - Copia della v6 come `queue_stilo_v7.sh`.
   - Il marker di fine va riscritto a mano («STILO V7 DONE»): il 15/09 una copia con `sed` lasciò il
     marker vecchio e l'attesa durò un'ora.
4. **File.**
   - Header `general.name = Stilo 4B`.
   - Nome del file con la versione (`Stilo-4B-v7-Q4_0.gguf`), servito senza (`Stilo-4B-Q4_0.gguf`).
5. **Controllo prima di usarlo.**
   - «Chi sei?» deve rispondere con il nome nuovo, con zero «Tabula» residui su 20 sonde d'identità
     (italiano e inglese).
   - Le domande sulla base restano un'omissione, non una negazione, come nella v6.
   - Poi le stesse prove della demo di settembre: otto turni di chat e i contrasti degli adapter.

**Cosa resta com'è:** `data/benchmarks/results/tabula/`, `data/identity/*tabula*` e
`scripts/persona/*tabula*` sono storia e non si rinominano. Un nome nuovo vale da v7 in poi.
