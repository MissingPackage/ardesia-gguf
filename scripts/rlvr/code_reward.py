"""Sandboxed unit-test execution reward for code-generation GRPO.

Deterministic and judge-free like the math rewards in :mod:`ardesia.rewards`: a
candidate scores 1.0 iff its extracted code passes ALL provided unit tests inside a
throwaway subprocess, 0.0 on any failure (assertion, crash, syntax error, timeout).

The sandbox is intentionally minimal — a separate interpreter launched in isolated
mode (``-I``: ignores env vars, user site, and ``PYTHONPATH``) with a wall-clock
timeout, run in a temporary working directory. This contains accidental damage and
runaway loops; it is NOT a security boundary against adversarial code. Hardening
beyond subprocess+timeout (seccomp/container) is a documented Phase 2 follow-up if
untrusted code volume grows.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from .reasoning import ANSWER_TAG, completion_to_text

TIMEOUT_S = 5


def _extract_code(text: str) -> str:
    """Pull the candidate code from the ``<answer>`` block, else the last fence."""

    tagged = ANSWER_TAG.findall(text)
    if tagged:
        return tagged[-1].strip()
    if "```" in text:
        parts = text.split("```")
        block = parts[-2] if len(parts) >= 3 else parts[-1]
        first, sep, rest = block.partition("\n")
        info = first.strip()
        # Drop the opening fence's info string (```python, ```py, ...): it is a lone
        # language token, whereas a real first line of code carries spaces/operators.
        if sep and info and all(ch.isalnum() or ch in "+-_" for ch in info):
            return rest.strip()
        return block.strip()
    return text.strip()


def _runs_green(code: str, tests: str) -> bool:
    """True iff ``code`` followed by ``tests`` exits 0 within the timeout."""

    program = f"{textwrap.dedent(code)}\n\n{textwrap.dedent(tests)}\n"
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "candidate.py"
        script.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True,
                timeout=TIMEOUT_S,
                cwd=d,
            )
        except subprocess.TimeoutExpired:
            return False
        return proc.returncode == 0


def _tests_to_source(test: Any) -> str:
    """Render one problem's tests as an executable suffix.

    A list/tuple of assertion strings (MBPP's native ``test_list`` shape) is joined
    with newlines so each assertion runs; ``str(list)`` would instead produce an inert
    list literal that executes nothing and hands every candidate a false 1.0.
    """

    if isinstance(test, (list, tuple)):
        return "\n".join(str(t) for t in test)
    return str(test)


def code_reward(completions: list[Any], tests: Any, **_: Any) -> list[float]:
    """Award 1.0 to completions whose extracted code passes all their unit tests."""

    count = len(completions)
    test_list = list(tests) if isinstance(tests, (list, tuple)) else [tests] * count
    if not test_list:  # empty/out-of-contract shape: degrade, never crash on zip(strict)
        test_list = [""] * count
    if len(test_list) != count:
        repeats = -(-count // len(test_list))  # ceil; mirrors reasoning._expand_references
        test_list = (test_list * repeats)[:count]
    out: list[float] = []
    for completion, test in zip(completions, test_list, strict=True):
        code = _extract_code(completion_to_text(completion))
        out.append(1.0 if _runs_green(code, _tests_to_source(test)) else 0.0)
    return out
