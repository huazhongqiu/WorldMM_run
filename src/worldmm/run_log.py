"""Small stdout helpers for benchmark runners.

The runners deliberately print a compact, human-scannable stream.  Full
diagnostics still go to the dedicated log files; these helpers stay quiet
about transport details such as URLs.
"""

from __future__ import annotations

from typing import Optional


SECTION_WIDTH = 64
PROGRESS_WIDTH = 20


def section(title: str) -> str:
    line = "=" * SECTION_WIDTH
    return f"\n{line}\n {title}\n{line}"


def _bar(current: int, total: int) -> str:
    if total <= 0:
        return "[" + " " * PROGRESS_WIDTH + f"]   0%"
    current = max(0, min(current, total))
    filled = int(PROGRESS_WIDTH * current / total)
    percent = int(100 * current / total)
    return "[" + "#" * filled + " " * (PROGRESS_WIDTH - filled) + f"] {percent:3d}%"


def progress(label: str, current: int, total: int, extra: Optional[str] = None) -> str:
    text = f"[{label} {current:03d}/{total:03d}] {_bar(current, total)}"
    return f"{text} {extra}" if extra else text


def question_separator(question_id: str) -> str:
    return f"\n{'-' * 62}\nQUESTION id={question_id}"


def agent_round(
    round_num: int,
    max_rounds: int,
    action: str,
    memory_type: Optional[str] = None,
    search_query: Optional[str] = None,
) -> str:
    parts = [f"round={round_num}/{max_rounds}", f"action={action}"]
    if memory_type:
        parts.append(f"memory={memory_type}")
    if search_query:
        parts.append(f"query={search_query}")
    return " | ".join(parts)


def answer_status(
    status: str,
    *,
    question_id: Optional[str] = None,
    answer: Optional[str] = None,
    correct: Optional[bool] = None,
    elapsed_seconds: Optional[float] = None,
    tokens: Optional[int] = None,
) -> str:
    parts = []
    if question_id is not None:
        parts.append(f"id={question_id}")
    parts.append(f"status={status}")
    if answer is not None:
        parts.append(f"answer={answer}")
    if correct is not None:
        parts.append(f"correct={'yes' if correct else 'no'}")
    if elapsed_seconds is not None:
        parts.append(f"elapsed={elapsed_seconds:.1f}s")
    if tokens is not None:
        parts.append(f"tokens={tokens}")
    return " | ".join(parts)


def emit(line: str) -> None:
    print(line, flush=True)
