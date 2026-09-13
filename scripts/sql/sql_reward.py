"""Execution-accuracy reward for text-to-SQL on Spider 1.0.

The sensor is EXECUTION, not string match: gold and prediction are both run on the real
sqlite file and the two result sets are compared as multisets (order-insensitive unless the
gold query has a top-level ORDER BY). Anything that throws — bad SQL, missing table, timeout —
scores 0.0.

Public surface:
    exec_reward(completions, db_id, query, **kw) -> list[float]   # TRL reward-function shape
    exec_accuracy(pred_sql, gold_sql, db_id) -> bool
    db_path(db_id) -> str

Self-test:  python3 scripts/sql/sql_reward.py
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading

SPIDER_DIR = os.environ.get(
    "SPIDER_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "sql", "spider"),
)
TIMEOUT_S = float(os.environ.get("SQL_TIMEOUT_S", "5"))

_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.S | re.I)
_STARTS = re.compile(r"^\s*(select|with)\b", re.I)
# top-level ORDER BY: no closing paren after it up to the end of the statement is too strict,
# so we settle for "the gold text mentions ORDER BY at all" — a superset of order-sensitive golds
# (a nested ORDER BY inside a subquery only makes the check stricter, never wrong-in-our-favour).
_ORDER_BY = re.compile(r"\border\s+by\b", re.I)


def db_path(db_id: str) -> str:
    return os.path.normpath(os.path.join(SPIDER_DIR, "database", db_id, f"{db_id}.sqlite"))


# ---------- extraction ----------
def extract_sql(text) -> str:
    """Pull the SQL out of a model completion: last ```sql fence, else last line starting
    with SELECT/WITH (carrying on to the end of that statement)."""
    if not isinstance(text, str):  # message list -> last assistant content
        if isinstance(text, list) and text:
            msg = text[-1]
            text = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        else:
            return ""
    fences = _FENCE.findall(text)
    if fences:
        return fences[-1].strip().rstrip(";").strip()
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if _STARTS.match(lines[i]):
            stmt = "\n".join(lines[i:]).strip()
            return stmt.split(";")[0].strip()
    return ""


# ---------- execution ----------
def _norm_cell(c):
    if c is None:
        return None
    if isinstance(c, bool):
        return float(int(c))
    if isinstance(c, (int, float)):
        return round(float(c), 6)
    if isinstance(c, bytes):
        c = c.decode("utf-8", "replace")
    return str(c).strip()


def _sort_key(row):
    return [(0, "") if c is None else (1, str(c)) for c in row]


def run_query(sql: str, db: str, timeout: float = TIMEOUT_S):
    """Run sql read-only on db with a hard timeout. Returns list of normalised tuples,
    or None on any error/timeout."""
    if not sql.strip():
        return None
    conn = None
    timer = None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=timeout)
        conn.text_factory = lambda b: b.decode("utf-8", "replace")
        timer = threading.Timer(timeout, conn.interrupt)
        timer.daemon = True
        timer.start()
        rows = conn.execute(sql).fetchall()
        return [tuple(_norm_cell(c) for c in r) for r in rows]
    except Exception:
        return None
    finally:
        if timer is not None:
            timer.cancel()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def exec_accuracy(pred_sql: str, gold_sql: str, db_id: str) -> bool:
    """True iff pred and gold produce the same result set on db_id."""
    db = db_path(db_id)
    if not os.path.exists(db):
        return False
    gold_rows = run_query(gold_sql, db)
    if gold_rows is None:  # broken gold: nothing can be scored against it
        return False
    pred_rows = run_query(pred_sql, db)
    if pred_rows is None:
        return False
    if len(pred_rows) != len(gold_rows):
        return False
    if _ORDER_BY.search(gold_sql or ""):
        return pred_rows == gold_rows
    return sorted(pred_rows, key=_sort_key) == sorted(gold_rows, key=_sort_key)


def _as_list(x, n):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x] * n


def exec_reward(completions, db_id, query, **kw) -> list[float]:
    """TRL reward function: 1.0 when the generated SQL executes to the gold result set.

    completions: list of strings or of message lists (TRL conversational format).
    db_id, query: the parallel dataset columns (lists), or a single value broadcast to all.
    """
    n = len(completions)
    dbs = _as_list(db_id, n)
    golds = _as_list(query, n)
    out = []
    for comp, db, gold in zip(completions, dbs, golds):
        try:
            out.append(1.0 if exec_accuracy(extract_sql(comp), gold, db) else 0.0)
        except Exception:
            out.append(0.0)
    return out


# ---------- self-test ----------
if __name__ == "__main__":
    CASES = [
        ("concert_singer", "SELECT count(*) FROM singer"),
        ("pets_1", "SELECT count(*) FROM Student"),
        ("car_1", "SELECT count(*) FROM continents"),
        ("concert_singer", "SELECT name, country FROM singer ORDER BY age DESC"),
    ]
    ok = True
    for db, gold in CASES:
        r = exec_reward([f"```sql\n{gold}\n```"], [db], [gold])[0]
        print(f"self  {db:16s} {r:.1f}  {gold[:52]}")
        ok &= r == 1.0
    for db, gold in CASES:
        wrong = "SELECT 42"
        r = exec_reward([f"Ecco la query:\n```sql\n{wrong}\n```"], [db], [gold])[0]
        print(f"wrong {db:16s} {r:.1f}")
        ok &= r == 0.0
    # bare-line extraction (no fence) and a syntactically broken prediction
    db, gold = CASES[0]
    r = exec_reward([f"La risposta e':\n{gold};"], [db], [gold])[0]
    print(f"noFence {db:14s} {r:.1f}")
    ok &= r == 1.0
    r = exec_reward(["```sql\nSELEKT * FROM nope\n```"], [db], [gold])[0]
    print(f"broken  {db:14s} {r:.1f}")
    ok &= r == 0.0
    # message-list completion (TRL conversational shape)
    r = exec_reward([[{"role": "assistant", "content": f"```sql\n{gold}\n```"}]], [db], [gold])[0]
    print(f"msglist {db:14s} {r:.1f}")
    ok &= r == 1.0
    # order-sensitive gold: reversed order must NOT score 1.0
    gold_o = "SELECT name FROM singer ORDER BY age DESC"
    rev = "SELECT name FROM singer ORDER BY age ASC"
    r = exec_reward([f"```sql\n{rev}\n```"], ["concert_singer"], [gold_o])[0]
    print(f"revOrder concert_singer {r:.1f}")
    ok &= r == 0.0
    print("SELF-TEST", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
