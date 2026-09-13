"""Verifiable reward for MATH-style answers (LaTeX in \\boxed{}), used by the `math2` task.

`rlvr.reasoning.exact_answer_reward` runs the REFERENCE through extract_final_answer too, which on a
LaTeX reference like `\\frac{3}{4}` keeps only the last number ("4") -> every fractional/symbolic gold
scores 0. This reward compares the candidate's last \\boxed{} (or the text after "Risposta finale:",
or the last number) against the gold string after LaTeX normalisation, with a numeric fallback.
"""
import re
from fractions import Fraction
from typing import Any

from .reasoning import completion_to_text


def last_boxed(s: str) -> str | None:
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def candidate_answer(text: str) -> str | None:
    b = last_boxed(text)
    if b is not None:
        return b
    if "Risposta finale:" in text:
        return text.rsplit("Risposta finale:", 1)[1].strip().splitlines()[0]
    nums = _NUM.findall(text)
    return nums[-1] if nums else None


def normalize(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:d|t)frac", r"\\frac", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\%", "").replace("%", "")
    s = s.replace("$", "").replace(" ", "").replace("\\displaystyle", "")
    s = s.rstrip(".")
    s = re.sub(r"^(x|y|z|n|k|a|b|t)=", "", s)  # "x=5" -> "5"
    s = s.replace("{,}", ",")
    if s.startswith("\\frac{") and s.endswith("}"):
        m = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", s)
        if m:
            s = f"{m.group(1)}/{m.group(2)}"
    m = re.fullmatch(r"\\frac(\d)(\d)", s)
    if m:
        s = f"{m.group(1)}/{m.group(2)}"
    return s


def as_number(s: str | None) -> Fraction | None:
    if s is None:
        return None
    t = s.replace(",", "")
    try:
        if "/" in t:
            a, b = t.split("/", 1)
            return Fraction(a) / Fraction(b)
        return Fraction(t)
    except (ValueError, ZeroDivisionError):
        return None


def answers_match(cand: str | None, ref: str) -> bool:
    nc, nr = normalize(cand), normalize(ref)
    if nc is None or nr is None:
        return False
    if nc == nr:
        return True
    a, b = as_number(nc), as_number(nr)
    return a is not None and b is not None and a == b


def math_reward(completions: list[Any], answer: Any, **_: Any) -> list[float]:
    refs = answer if isinstance(answer, list) else [answer] * len(completions)
    return [1.0 if answers_match(candidate_answer(completion_to_text(c)), str(r)) else 0.0
            for c, r in zip(completions, refs)]
