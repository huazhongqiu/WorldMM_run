"""Shared multiple-choice completion checks for EgoLife evaluation and reports."""

import re
from typing import Optional


def extract_choice_letter(text: str) -> Optional[str]:
    """Extract a final multiple-choice letter without treating reasoning as one."""
    marked_answers = re.findall(
        r"\b(?:final\s+)?answer\s*(?:is\s*)?[:\-]?\s*\(?([A-Za-z])\b",
        text,
        flags=re.IGNORECASE,
    )
    if marked_answers:
        return marked_answers[-1].upper()

    match = re.match(r"\s*\(?([A-Za-z])(?:[\.\)]|\s*$)", text)
    return match.group(1).upper() if match else None


def is_completed_choice_result(result: dict) -> bool:
    """Require a successful invocation and an answer offered by the question."""
    letter = extract_choice_letter(str(result.get("response") or ""))
    return (result.get("completed") is True and letter in ("A", "B", "C", "D")
            and letter in result.get("choices", {}))
