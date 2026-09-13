# Spider 1.0 — provenienza dei dati

## Database sqlite (`spider/database/<db_id>/<db_id>.sqlite`, `spider/tables.json`)

Archivio ufficiale Yale-LILY (pagina <https://yale-lily.github.io/spider>, link "Download a copy
of the dataset", licenza CC BY-SA 4.0). Google Drive id `1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J`.

- URL usato (scaricabile con curl, senza gdown: `confirm=t` basta a saltare il token di conferma):
  `https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t`
- nome canonico dell'archivio: `spider_data.zip`
- dimensione: 205 800 266 byte
- sha256: `00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`
- scaricato il 2026-09-13

Estratto solo il necessario (il resto dello zip — `test_database/`, i json, i csv per-tabella — non
serve):

```sh
curl -L "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t" -o spider_data.zip
sha256sum spider_data.zip   # deve dare l'hash qui sopra
unzip -q 'spider_data/database/*/*.sqlite' 'spider_data/tables.json' -d /tmp/spx
mv /tmp/spx/spider_data data/sql/spider
```

Risultato: **166 database** (train + dev), 453 MB, `tables.json` 811 KB.
`data/sql/spider/` è in `.gitignore`: si ricrea da qui, non sta in git.

Verificato: tutti e 166 i file esistono col nome atteso e `sqlite_master` è leggibile;
`concert_singer` (9 stadi), `pets_1` (34 studenti), `car_1` (5 continenti) rispondono a `count(*)`.

## Domande e query (`spider-train.jsonl`, `spider-dev.jsonl`)

HF dataset `xlangai/spider` (parquet, split `train` / `validation`) — solo testo, nessun database.
7000 righe train, 1034 righe dev. Lo `schema` di ogni riga è renderizzato dal **file sqlite vero**
(`SELECT sql FROM sqlite_master WHERE type='table'`), non da `tables.json`: così la DDL nel prompt è
esattamente quella che poi esegue. Rigenerabile con
`.venv-q35/bin/python scripts/sql/build_sql_data.py`.

I 166 database coprono 160 db_id citati da train+dev (6 database non sono usati da nessuna domanda
di questi due split). DDL completa per tutti i 160: max 2900 caratteri, mediana 710 — il fallback di
troncamento a 3000 di `build_sql_data.py` non si attiva mai su Spider.

## Qualità del gold (misurata, non stimata)

- dev: **1034/1034** gold eseguono senza errore.
- train: **6997/7000**. Le 3 righe rotte restano nel file (sono rumore dell'upstream, non nostro):
  - `assets_maintenance` ×1 — `no such table: Ref_Company_Types` (tabella assente dallo sqlite).
  - `document_management` ×2 — `ORDER BY ... LIMIT 3 INTERSECT ...`, sintassi rifiutata da SQLite.

  In RLVR queste tre danno reward 0 qualunque cosa il modello generi: `exec_accuracy` ritorna False
  quando il gold stesso non esegue. Filtrarle se la densità di reward conta.
