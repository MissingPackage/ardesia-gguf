"""Deterministic corpus operations for the persona training sets (round-8, persona-v6).

Everything here is a **pure CPU seam**: no torch / transformers / unsloth at module top, and
the tokenizer is *injected* rather than loaded, so the whole corpus pipeline is unit-testable
on a box with no GPU. The build script (which does have the engine venv) supplies the real
tokenizer; tests supply a fake.

Why a module and not a script: the corpus is the source of truth for the model's behaviour, and
round-7c shipped a corpus defect (a confidently-wrong biology row) that no test could have
caught because the assembly logic lived in an untested one-off script.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

SYSTEM_PROMPT = (
    "Sei Ardesia, un piccolo modello linguistico locale: essenziale, ordinato e onesto "
    "su ciò che non sa."
)

LAYERS = frozenset({"L0", "L1", "L2", "L3", "L4", "L5", "L6"})
DOMAINS = frozenset(
    {"ml", "systems", "math", "security", "bio", "physics", "chem", "med", "code", "none"}
)
#: Layers whose rows assert checkable facts — the generator MUST enumerate them. L6 (code) is
#: deliberately excluded: a code exemplar's correctness is verified by EXECUTION (mbpp-style), not
#: by claim refutation, so it carries no ``claims`` (round-9, persona-v7).
FACTUAL_LAYERS = frozenset({"L2", "L3", "L4"})

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def validate_row(row: dict) -> list[str]:
    """Return a list of human-readable schema errors; empty means valid."""
    errors: list[str] = []

    for field in ("id", "layer", "domain", "lang", "messages"):
        if field not in row:
            errors.append(f"missing field {field!r}")
    if errors:
        return errors

    if row["layer"] not in LAYERS:
        errors.append(f"unknown layer {row['layer']!r} (expected one of {sorted(LAYERS)})")
    if row["domain"] not in DOMAINS:
        errors.append(f"unknown domain {row['domain']!r} (expected one of {sorted(DOMAINS)})")
    if row["lang"] not in {"it", "en"}:
        errors.append(f"lang must be 'it' or 'en', got {row['lang']!r}")

    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append("messages must be exactly [system, user, assistant]")
        return errors
    roles = [m.get("role") for m in messages]
    if roles != ["system", "user", "assistant"]:
        errors.append(f"message roles must be system/user/assistant, got {roles}")
        return errors
    if messages[0]["content"].strip() != SYSTEM_PROMPT:
        errors.append("row does not carry the Ardesia system prompt verbatim")
    for msg in messages[1:]:
        if not str(msg.get("content", "")).strip():
            errors.append(f"empty {msg['role']} content")

    if row["layer"] in FACTUAL_LAYERS and not row.get("claims"):
        errors.append(
            f"layer {row['layer']} asserts facts but enumerates no 'claims' — the refuter "
            "cannot verify prose, only a list"
        )

    return errors


def normalize_prompt(text: str) -> str:
    """Lower-case, strip accents and punctuation, collapse whitespace.

    Mirrors the intent of ``eval._canonical``: two prompts that differ only in case,
    accent or punctuation are the same prompt for dedup purposes.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _WS.sub(" ", _PUNCT.sub(" ", stripped)).strip()


def prompt_similarity(a: str, b: str) -> float:
    """Token-set Jaccard of two normalized prompts. Deterministic, no model needed."""
    tokens_a = set(normalize_prompt(a).split())
    tokens_b = set(normalize_prompt(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _user_turn(row: dict) -> str:
    return row["messages"][1]["content"]


def find_near_duplicates(
    rows: list[dict], threshold: float = 0.85
) -> list[tuple[str, str, float]]:
    """All (id_a, id_b, score) pairs whose user turns are near-identical.

    Cross-domain collision is the expected case: 'how does an AAV vector work' is a legitimate
    question for both `bio` and `med`, and both generators will produce it.
    """
    hits: list[tuple[str, str, float]] = []
    for i, row_a in enumerate(rows):
        for row_b in rows[i + 1 :]:
            score = prompt_similarity(_user_turn(row_a), _user_turn(row_b))
            if score >= threshold:
                hits.append((row_a["id"], row_b["id"], score))
    return hits


def find_heldout_collisions(
    corpus: list[dict], probes: list[dict], threshold: float = 0.75
) -> list[tuple[str, str, float]]:
    """Corpus rows that collide with an eval probe — a held-out violation.

    A lower threshold than dedup on purpose: a probe only has to be *close* to a training row
    for the eval to stop measuring generalization, and a false positive here costs one row
    while a false negative costs the validity of the whole depth verdict.
    """
    hits: list[tuple[str, str, float]] = []
    for row in corpus:
        for probe in probes:
            score = prompt_similarity(_user_turn(row), probe["prompt"])
            if score >= threshold:
                hits.append((row["id"], probe["id"], score))
    return hits


def count_row_tokens(row: dict, tokenizer) -> int:
    """Token length of the full chat-templated row. ``tokenizer`` is injected, never loaded."""
    return len(tokenizer.apply_chat_template(row["messages"], tokenize=True))


def _rank_key(row: dict) -> tuple[int, str]:
    """Sort key: best register_score first, id ascending as a deterministic tie-break."""
    return (-row.get("register_score", 0), row["id"])


#: Layers produced by the generation fan-out — dedup applies HERE (cross-domain L2 collision is the
#: expected accidental waste). Curated layers (L0 carry-over, L1 rewrites) are exempt: their
#: repeated user turns are intentional reinforcement (e.g. the parameter-count question x4), not
#: duplication to prune.
GENERATED_LAYERS = frozenset({"L2", "L3", "L4", "L5", "L6"})
#: Depth layers whose per-layer target is split EVENLY across the domains present (spec §4:
#: L2 = 8 domini x 53, L4 contrast pairs per domain). Without this, register-score differences
#: between domains (measured: math 4.10 vs physics 4.98) skew a global top-N toward the higher-
#: scoring domains and starve the others.
BALANCED_LAYERS = frozenset({"L2", "L4"})


def _pick(pool: list[dict], n: int, frac_en: float) -> list[dict]:
    """Top-``n`` rows by rank, holding ~``frac_en`` English so a lower-scoring language survives.

    English rows measured lower on the register lens; a pure top-N collapsed en from ~30% to ~14%.
    Split the quota by language, then backfill from whichever language has rows left if one runs
    short (so a tiny pool never blocks on the language target).
    """
    if n <= 0:
        return []
    en = sorted((r for r in pool if r["lang"] == "en"), key=_rank_key)
    it = sorted((r for r in pool if r["lang"] == "it"), key=_rank_key)
    n_en = min(len(en), round(n * frac_en))
    n_it = min(len(it), n - n_en)
    n_en = min(len(en), n - n_it)  # backfill en if it was short
    return it[:n_it] + en[:n_en]


def _select_layer(pool: list[dict], target: int, *, balanced: bool, frac_en: float) -> list[dict]:
    """Pick ``target`` rows from a single layer's pool, ranked by register_score (id tie-break).

    Holds the ~``frac_en`` language mix in every selection bucket. When ``balanced``, also splits the
    target evenly across the domains present and fills any remainder from the best rows left over —
    so a lower-scoring domain (and a lower-scoring language) is still represented.
    """
    if not balanced:
        return _pick(pool, target, frac_en)
    by_domain: dict[str, list[dict]] = {}
    for row in pool:
        by_domain.setdefault(row["domain"], []).append(row)
    domains = sorted(by_domain)
    quota = target // len(domains) if domains else 0
    selected: list[dict] = []
    for domain in domains:
        selected.extend(_pick(by_domain[domain], quota, frac_en))
    remaining = target - len(selected)
    if remaining > 0:
        chosen = {row["id"] for row in selected}
        leftovers = [r for r in pool if r["id"] not in chosen]
        selected.extend(_pick(leftovers, remaining, frac_en))
    return selected


def assemble_corpus(
    candidates: list[dict],
    targets: dict[str, int],
    tokenizer,
    *,
    max_row_tokens: int = 1024,
    min_l4_l2_ratio: float = 0.15,
    dedup_layers: frozenset[str] = GENERATED_LAYERS,
    balanced_layers: frozenset[str] = BALANCED_LAYERS,
    frac_en: float = 0.30,
) -> tuple[list[dict], dict]:
    """Select candidates into the final corpus under per-layer targets and the L4:L2 floor.

    The L4:L2 floor is a **design constraint, not a residual** (spec D8): L2 rows teach the model
    to *express* depth, including where it has no knowledge, so the anti-fabrication layer is the
    counterweight. Filling the token budget by letting L4 slip is the failure mode this raises on.

    ``dedup_layers`` gates which layers participate in near-duplicate pruning; the curated layers
    (L0/L1) are exempt by default so intentional reinforcement survives assembly.
    """
    dropped: Counter[str] = Counter()
    kept: list[dict] = []

    for row in candidates:
        if validate_row(row):
            dropped["invalid"] += 1
            continue
        if count_row_tokens(row, tokenizer) > max_row_tokens:
            dropped["over_length"] += 1
            continue
        kept.append(row)

    # Near-duplicates: keep the better-scored row, deterministic tie-break on id. Only the generated
    # layers are eligible; curated L0/L1 reinforcement passes through untouched.
    dedup_pool = [row for row in kept if row["layer"] in dedup_layers]
    losers: set[str] = set()
    by_id = {row["id"]: row for row in dedup_pool}
    for id_a, id_b, _score in find_near_duplicates(dedup_pool):
        if id_a in losers or id_b in losers:
            continue
        # _rank_key is ascending-is-better, so max() is the worse row: that is the one to drop.
        losers.add(max(by_id[id_a], by_id[id_b], key=_rank_key)["id"])
    dropped["duplicate"] = len(losers)
    kept = [row for row in kept if row["id"] not in losers]

    selected: list[dict] = []
    for layer, target in targets.items():
        pool = [r for r in kept if r["layer"] == layer]
        if len(pool) < target:
            dropped[f"short_{layer}"] = target - len(pool)
        selected.extend(
            _select_layer(pool, target, balanced=layer in balanced_layers, frac_en=frac_en)
        )

    n_l2 = sum(1 for r in selected if r["layer"] == "L2")
    n_l4 = sum(1 for r in selected if r["layer"] == "L4")
    ratio = (n_l4 / n_l2) if n_l2 else 0.0
    if n_l2 and ratio < min_l4_l2_ratio:
        raise ValueError(
            f"L4:L2 ratio {ratio:.3f} is below the {min_l4_l2_ratio:.2f} floor "
            f"(L4={n_l4}, L2={n_l2}). Anti-fabrication rows must not be squeezed out to fill "
            "the token budget — generate more L4 or cut L2 (spec D8)."
        )

    token_counts = [count_row_tokens(r, tokenizer) for r in selected]
    manifest = {
        "total_rows": len(selected),
        "total_tokens": sum(token_counts),
        "max_row_tokens": max(token_counts, default=0),
        "mean_row_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
        "by_layer": dict(Counter(r["layer"] for r in selected)),
        "by_lang": dict(Counter(r["lang"] for r in selected)),
        "by_domain": dict(Counter(r["domain"] for r in selected)),
        "l4_l2_ratio": round(ratio, 4),
        "dropped": dict(dropped),
    }
    return selected, manifest
