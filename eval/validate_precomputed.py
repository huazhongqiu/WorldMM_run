#!/usr/bin/env python3
"""Validate persisted WorldMM artifacts before starting benchmark inference."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


REQUIRED_GRANULARITIES = ("10sec", "30sec", "3min", "10min")


@dataclass(frozen=True)
class ValidationSummary:
    question_count: int
    unit_count: int
    missing: tuple[str, ...]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def validate_precomputed(eval_json: Path, root: Path, model: str) -> ValidationSummary:
    rows = json.loads(eval_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"evaluation JSON must be a non-empty list: {eval_json}")

    question_ids: list[str] = []
    unit_ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "ID" not in row or "video_id" not in row:
            raise ValueError(f"row {index} must contain ID and video_id")
        question_id = str(row["ID"])
        unit_id = str(row["video_id"])
        question_ids.append(question_id)
        if unit_id not in unit_ids:
            unit_ids.append(unit_id)

    duplicates = sorted({item for item in question_ids if question_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate question ID(s): {duplicates[:10]}")

    required: list[Path] = []
    for unit_id in unit_ids:
        required.extend(root / "caption" / unit_id / f"{name}.json" for name in REQUIRED_GRANULARITIES)
        required.extend(
            (
                root / "episodic_memory" / unit_id / f"openie_results_{model}.json",
                root / "episodic_memory" / unit_id / f"episodic_triple_results_{model}.json",
                root / "semantic_memory" / unit_id / f"semantic_consolidation_results_{model}.json",
                root / "visual_memory" / unit_id / "visual_embeddings.pkl",
            )
        )
    missing = tuple(sorted(_relative(path, root) for path in required if not path.is_file() or path.stat().st_size == 0))
    return ValidationSummary(len(rows), len(unit_ids), missing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3.5-4B")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = validate_precomputed(args.eval_json, args.root, args.model)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"INCOMPLETE: {exc}", file=sys.stderr)
        return 1
    if summary.missing:
        print(
            f"INCOMPLETE: {len(summary.missing)} artifact(s) missing or empty "
            f"for {summary.question_count} questions across {summary.unit_count} units",
            file=sys.stderr,
        )
        for path in summary.missing:
            print(f"  {path}", file=sys.stderr)
        return 1
    print(f"OK: {summary.question_count} questions across {summary.unit_count} units; all persisted artifacts present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
