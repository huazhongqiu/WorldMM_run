#!/usr/bin/env python3
"""Convert LVBench annotations into the Video-MME style eval JSON used by WorldMM.

Reads the raw LVBench ``video_info.meta.jsonl`` plus the videospy test-split
manifest (a flat JSON list of question UIDs), parses the inline "(A) ... (B) ..."
choices out of each question string, and writes:

- ``<output-dir>/lvbench_test.json``  eval rows in the schema expected by ``eval/eval.py``
- ``<output-dir>/test_videos.json``  list of video keys that need preprocessing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_META_JSONL = "/myworkspace/data/LVBench/zai-org-LVBench/video_info.meta.jsonl"
DEFAULT_TEST_MANIFEST = "/myworkspace/projects/videospy/experiments/data_splits/main/lvbench/test.json"
DEFAULT_VIDEO_ROOT = "/myworkspace/data/LVBench/AIWinter-LVBench/all_videos"
DEFAULT_OUTPUT_DIR = "/myworkspace/projects/worldmm_needed/lvbench/qa"

CHOICE_SPLIT_RE = re.compile(r"\n\s*\(([A-D])\)\s*")
EXPECTED_LETTERS = ("A", "B", "C", "D")
DEFAULT_TIME = {"date": "DAY1", "time": "23595999"}


def parse_question_and_choices(raw_question: str, uid: str, video_key: str) -> tuple[str, dict[str, str]]:
    parts = CHOICE_SPLIT_RE.split(raw_question.strip())
    # capturing group keeps the letter in the split result: [text, letter, choice, letter, choice, ...]
    if len(parts) != 1 + 2 * len(EXPECTED_LETTERS):
        raise ValueError(
            f"video={video_key} uid={uid}: expected {len(EXPECTED_LETTERS)} inline choices, found {len(parts) // 2}"
        )
    question = parts[0].strip()
    if not question:
        raise ValueError(f"video={video_key} uid={uid}: empty question text")
    letters = parts[1::2]
    choices_text = parts[2::2]
    if tuple(letters) != EXPECTED_LETTERS:
        raise ValueError(f"video={video_key} uid={uid}: choices out of order: {letters}")
    choices = {letter: text.strip() for letter, text in zip(letters, choices_text)}
    for letter, text in choices.items():
        if not text:
            raise ValueError(f"video={video_key} uid={uid}: empty choice {letter}")
    return question, choices


def build_row(entry: dict, video_key: str) -> dict:
    uid = str(entry["uid"])
    question, choices = parse_question_and_choices(entry["question"], uid, video_key)

    answer = str(entry["answer"]).strip().upper()
    if answer not in EXPECTED_LETTERS:
        raise ValueError(f"video={video_key} uid={uid}: invalid answer '{answer}'")

    question_type = entry.get("question_type", [])
    if isinstance(question_type, list):
        question_type = "; ".join(question_type)

    return {
        "ID": uid,
        "query_time": dict(DEFAULT_TIME),
        "type": question_type,
        "video_id": video_key,
        "duration": "long",
        "domain": "",
        "sub_category": "",
        "time_reference": entry.get("time_reference", ""),
        "question": question,
        "choice_a": choices["A"],
        "choice_b": choices["B"],
        "choice_c": choices["C"],
        "choice_d": choices["D"],
        "answer": answer,
        "target_time": dict(DEFAULT_TIME),
    }


def load_flattened_entries(meta_jsonl: Path) -> dict[str, tuple[str, dict]]:
    entries: dict[str, tuple[str, dict]] = {}
    with meta_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            video_key = record["key"]
            for qa in record.get("qa", []):
                uid = str(qa["uid"])
                if uid in entries:
                    raise ValueError(f"duplicate uid {uid} (line {line_no})")
                entries[uid] = (video_key, qa)
    if not entries:
        raise ValueError(f"no QA entries found in {meta_jsonl}")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare LVBench test set for WorldMM.")
    parser.add_argument("--meta-jsonl", type=Path, default=Path(DEFAULT_META_JSONL))
    parser.add_argument("--test-manifest", type=Path, default=Path(DEFAULT_TEST_MANIFEST))
    parser.add_argument("--video-root", type=Path, default=Path(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    manifest = json.loads(args.test_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"test manifest must be a non-empty JSON list: {args.test_manifest}")
    uids = [str(uid) for uid in manifest]
    if len(set(uids)) != len(uids):
        raise ValueError("test manifest contains duplicate UIDs")

    entries = load_flattened_entries(args.meta_jsonl)
    unknown = [uid for uid in uids if uid not in entries]
    if unknown:
        raise ValueError(f"{len(unknown)} manifest UIDs not found in meta jsonl, e.g. {unknown[:5]}")

    rows: list[dict] = []
    errors: list[str] = []
    video_keys_in_order: list[str] = []
    for uid in uids:
        video_key, qa = entries[uid]
        try:
            rows.append(build_row(qa, video_key))
        except ValueError as exc:
            errors.append(str(exc))
        if video_key not in video_keys_in_order:
            video_keys_in_order.append(video_key)

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        raise SystemExit(f"{len(errors)} row(s) failed to convert")

    missing_videos = [key for key in video_keys_in_order if not (args.video_root / f"{key}.mp4").exists()]
    if missing_videos:
        raise SystemExit(f"{len(missing_videos)} test video(s) missing under {args.video_root}: {missing_videos}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = args.output_dir / "lvbench_test.json"
    videos_path = args.output_dir / "test_videos.json"
    eval_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    videos_path.write_text(json.dumps(video_keys_in_order, indent=2) + "\n", encoding="utf-8")

    type_counts: dict[str, int] = {}
    for row in rows:
        for t in row["type"].split("; "):
            if t:
                type_counts[t] = type_counts.get(t, 0) + 1

    print(f"OK: wrote {len(rows)} questions over {len(video_keys_in_order)} videos -> {eval_path}")
    print(f"OK: video list -> {videos_path}")
    print("question types:", json.dumps(type_counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
