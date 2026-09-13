"""Build the Spider text-to-SQL training/eval data.

Sources:
  - questions/queries: HF dataset `xlangai/spider` (splits train / validation, parquet)
  - schemas:           the real sqlite files under data/sql/spider/database/<db>/<db>.sqlite
                       (rendered from sqlite_master, so the DDL is exactly what will execute)

Writes:
  data/sql/spider-train.jsonl        {db_id, question, query, schema}
  data/sql/spider-dev.jsonl          idem, validation split
  data/bench/adapter-sql.jsonl       chat SFT rows {"messages":[system,user,assistant]} from train
  data/sql/spider-dev-prompts.jsonl  {db_id, question, query, messages:[system,user]} for eval

Run:  .venv-q35/bin/python scripts/sql/build_sql_data.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sql_reward import SPIDER_DIR, db_path  # noqa: E402  (same-dir module)

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_SQL = os.path.join(ROOT, "data", "sql")
OUT_BENCH = os.path.join(ROOT, "data", "bench")
MAX_SCHEMA = 3000

SYSTEM = ("Sei Tabula. Dato lo schema del database e una domanda, rispondi SOLO con la query SQL "
          "(dialetto SQLite) in un blocco ```sql.")


def _tables(db_id: str):
    """[(name, ddl)] for the real tables of a db, in sqlite_master order."""
    con = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    try:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        con.close()
    return [(n, " ".join((s or "").split())) for n, s in rows if s]


def render_schema(db_id: str, _cache={}) -> str:
    """Compact CREATE TABLE DDL for db_id. Full DDL when it fits in MAX_SCHEMA chars;
    otherwise degrade — first to column names without types, then to bare table names —
    so that EVERY table name always survives."""
    if db_id in _cache:
        return _cache[db_id]
    tabs = _tables(db_id)
    full = ";\n".join(ddl for _, ddl in tabs)
    if len(full) <= MAX_SCHEMA:
        _cache[db_id] = full
        return full
    # level 2: name + column names only
    con = sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    try:
        slim = []
        for name, _ in tabs:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")').fetchall()]
            slim.append(f'CREATE TABLE "{name}" ({", ".join(cols)})')
    finally:
        con.close()
    out = ";\n".join(slim)
    if len(out) > MAX_SCHEMA:  # level 3: table names only
        out = "-- tabelle: " + ", ".join(n for n, _ in tabs)
    _cache[db_id] = out
    return out


def load_split(split: str):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    fn = {"train": "spider/train-00000-of-00001.parquet",
          "validation": "spider/validation-00000-of-00001.parquet"}[split]
    path = hf_hub_download("xlangai/spider", fn, repo_type="dataset")
    tbl = pq.read_table(path, columns=["db_id", "question", "query"])
    return tbl.to_pylist()


def user_msg(schema: str, question: str) -> str:
    return f"Schema:\n{schema}\n\nDomanda: {question}"


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path}  {len(rows)} rows")


def main():
    assert os.path.isdir(os.path.join(SPIDER_DIR, "database")), \
        f"missing spider databases under {SPIDER_DIR}/database — see data/sql/SOURCES.md"
    built = {}
    for split, name in (("train", "spider-train"), ("validation", "spider-dev")):
        rows, skipped = [], set()
        for r in load_split(split):
            if not os.path.exists(db_path(r["db_id"])):
                skipped.add(r["db_id"])
                continue
            rows.append({"db_id": r["db_id"], "question": r["question"], "query": r["query"],
                         "schema": render_schema(r["db_id"])})
        if skipped:
            print(f"WARN {split}: {len(skipped)} db_id without a sqlite file: {sorted(skipped)}")
        write_jsonl(os.path.join(OUT_SQL, f"{name}.jsonl"), rows)
        built[split] = rows

    write_jsonl(os.path.join(OUT_BENCH, "adapter-sql.jsonl"), [
        {"messages": [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user_msg(r["schema"], r["question"])},
                      {"role": "assistant", "content": f"```sql\n{r['query']}\n```"}]}
        for r in built["train"]])

    write_jsonl(os.path.join(OUT_SQL, "spider-dev-prompts.jsonl"), [
        {"db_id": r["db_id"], "question": r["question"], "query": r["query"],
         "messages": [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user_msg(r["schema"], r["question"])}]}
        for r in built["validation"]])

    lens = sorted(len(render_schema(d)) for d in {r["db_id"] for r in built["train"] + built["validation"]})
    print(f"schemas: {len(lens)} dbs, chars min {lens[0]} median {lens[len(lens)//2]} max {lens[-1]}")


if __name__ == "__main__":
    main()
