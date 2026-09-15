"""MedVidBench prediction format validation and conservative repair."""

from __future__ import annotations

import re
from typing import Any, Optional

NUMBER = r"\d+(?:\.\d+)?"
TIME_SPAN = rf"(?P<start>{NUMBER})\s*-\s*(?P<end>{NUMBER})\s*(?:seconds?|secs?)\.?"


def _cleaned_prediction(response: Any) -> str:
    text = str(response or "").strip()
    text = re.sub(r"^\s*```(?:\w+)?\s*|\s*```\s*$", "", text, flags=re.MULTILINE)
    return re.sub(r"^(?:answer|response|prediction)\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def normalize_prediction(qa_type: str, response: Any) -> Optional[str]:
    """Return a contract-compliant prediction, or ``None`` if repair is unsafe."""
    text = _cleaned_prediction(response)
    if not text:
        return None

    if qa_type == "tal":
        spans = [
            (match.group("start"), match.group("end"))
            for match in re.finditer(TIME_SPAN, text, flags=re.IGNORECASE)
            if float(match.group("start")) <= float(match.group("end"))
        ]
        return ", ".join(f"{start}-{end}" for start, end in spans) + " seconds." if spans else None

    if qa_type == "stg":
        box = rf"(?P<timestamp>{NUMBER})\s*(?:seconds?|secs?)\s*:\s*\[\s*(?P<x1>{NUMBER})\s*,\s*(?P<y1>{NUMBER})\s*,\s*(?P<x2>{NUMBER})\s*,\s*(?P<y2>{NUMBER})\s*\]"
        boxes = [
            "{timestamp} seconds: [{x1}, {y1}, {x2}, {y2}]".format(**match.groupdict())
            for match in re.finditer(box, text, flags=re.IGNORECASE)
        ]
        return "\n".join(boxes) if boxes else None

    if qa_type.startswith("dense_captioning"):
        event = rf"(?m)^\s*{TIME_SPAN}\s*:\s*(?P<label>[^:\n]+?)\s*:\s*(?P<description>[^\n]+?)\s*$"
        events = [
            f"{match.group('start')}-{match.group('end')} seconds: "
            f"{match.group('label').strip()}: {match.group('description').strip()}"
            for match in re.finditer(event, text, flags=re.IGNORECASE)
            if float(match.group("start")) <= float(match.group("end"))
        ]
        return "\n".join(events) if events else None

    if qa_type == "skill_assessment":
        labels = (
            "Respect for tissue",
            "Suture/needle handling",
            "Time and motion",
            "Flow of operation",
            "Overall performance",
            "Quality of final product",
        )
        values = []
        for label in labels:
            match = re.search(rf"{re.escape(label)}\s*:\s*([1-5])\s*/\s*5", text, flags=re.IGNORECASE)
            if match is None:
                return None
            values.append(f"{label}: {match.group(1)}/5")
        return ", ".join(values)

    if qa_type == "cvs_assessment":
        labels = ("Two structures", "Cystic plate", "Hepatocystic triangle")
        values = []
        for label in labels:
            match = re.search(rf"{re.escape(label)}\s*:\s*([0-2])(?=\D|$)", text, flags=re.IGNORECASE)
            if match is None:
                return None
            values.append(f"{label}: {match.group(1)}")
        return ", ".join(values)

    if qa_type == "next_action":
        label = re.sub(r"^next action\s*:\s*", "", text, flags=re.IGNORECASE).splitlines()[0].strip()
        label = label.strip(" `\"'").rstrip(".").strip()
        return label or None

    if qa_type.startswith(("video_summary", "region_caption")):
        text = re.sub(r"^(?:summary|description)\s*:\s*", "", text, flags=re.IGNORECASE)
        return " ".join(text.split()) or None

    return text
