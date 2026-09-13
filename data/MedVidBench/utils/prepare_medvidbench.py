#!/usr/bin/env python3
"""Convert MedVidBench (MedVidU ECCV2026 TrainVal) into WorldMM eval artifacts.

Reads the raw ``medvidu_eccv2026_trainval.json`` plus the videospy test-split
manifest (a flat JSON list of integer indices), and writes:

- ``<output-dir>/medvidbench_test.json``  eval rows in the WorldMM qa schema
  (open-ended: no choice_* fields, GT answer text kept in ``answer``)
- ``<output-dir>/test_segments.json``     deduplicated frame segments
  (``video_id&&start&&end&&fps``) that need frame->mp4 synthesis, with
  ``/root/data/`` frame paths remapped onto ``--frame-root``
- ``<output-dir>/test_videos.json``       list of segment keys (video stems)

Question text follows the videospy MedVidBench runner exactly: the raw human
turn (minus the ``<video>`` prefix) plus a per-qa_type "Output requirement"
contract. Region-caption (RC) items get their own segment key per unique
(id, RC_info) pair so the green bbox drawn at synthesis time never leaks into
the memory of non-RC questions on the same segment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_TRAINVAL_JSON = "/myworkspace/data/MedVidBench/MedVidU_ECCV2026_TrainVal/medvidu_eccv2026_trainval.json"
DEFAULT_TEST_MANIFEST = "/myworkspace/projects/videospy/experiments/data_splits/main/medvidbench/test.json"
DEFAULT_FRAME_ROOT = "/myworkspace/data/MedVidBench/MedVidU_ECCV2026_TrainVal/valdata"
DEFAULT_OUTPUT_DIR = "/myworkspace/projects/worldmm_needed/medvidbench/qa"

DEFAULT_TIME = {"date": "DAY1", "time": "23595999"}
FRAME_PREFIX = "/root/data/"

# The eight per-qa_type output contracts, copied verbatim from
# videospy/benchmark/medvidbench/run.py (ANSWER_FORMATS + CAPTION_FORMATS).
ANSWER_FORMATS = {
    "tal": (
        'Return only the time span(s), for example: "1.1-3.0 seconds." '
        'For multiple spans, use: "1.1-3.0, 5.0-7.0 seconds."'
    ),
    "stg": (
        "Return only the requested timestamped bounding boxes in this format: "
        '"<timestamp> seconds: [x1, y1, x2, y2]". Use exactly the timestamps '
        "requested in the question. Do not add, omit, or replace timestamps."
    ),
    "next_action": "Return only the next action label.",
    "skill_assessment": (
        "Return only: "
        '"Respect for tissue: <1-5>/5, Suture/needle handling: <1-5>/5, '
        'Time and motion: <1-5>/5, Flow of operation: <1-5>/5, '
        'Overall performance: <1-5>/5, Quality of final product: <1-5>/5".'
    ),
    "cvs_assessment": (
        "Return only: "
        '"Two structures: <0-2>, Cystic plate: <0-2>, '
        'Hepatocystic triangle: <0-2>".'
    ),
}

CAPTION_FORMATS = {
    "dense_captioning": (
        "Return one event per line in this format: "
        '"1.0-29.0 seconds: event label: factual description".'
    ),
    "video_summary": "Return only one concise summary paragraph.",
    "region_caption": "Return only one concise factual description.",
}


def output_requirement(qa_type: str) -> str | None:
    answer_format = ANSWER_FORMATS.get(qa_type)
    if answer_format is None:
        answer_format = next(
            (
                value
                for prefix, value in CAPTION_FORMATS.items()
                if qa_type.startswith(prefix)
            ),
            None,
        )
    return answer_format


def build_question(sample: dict) -> str:
    question = next(
        message["value"]
        for message in sample["conversations"]
        if message["from"] == "human"
    )
    question = question.removeprefix("<video>\n")
    answer_format = output_requirement(sample["qa_type"])
    if answer_format is None:
        return question
    return f"{question}\n\nOutput requirement: {answer_format}"


def conversation_value(sample: dict, roles: set[str]) -> str:
    value = ""
    for message in sample.get("conversations", []):
        if message.get("from") in roles:
            value = message.get("value", "")
    return value.replace("<video>\n", "").replace("<video>", "")


def report_task(qa_type: str) -> str:
    if qa_type.startswith("dense_captioning"):
        return "dvc"
    if qa_type.startswith("region_caption"):
        return "rc"
    if qa_type.startswith("video_summary"):
        return "vs"
    return qa_type


def resolve_frame_path(path: str, frame_root: Path) -> Path:
    if path.startswith(FRAME_PREFIX):
        return frame_root / path[len(FRAME_PREFIX):]
    return frame_root / path


def segment_key(raw_id: str, rc_signature: str | None = None) -> str:
    """Filesystem-safe, collision-free key for one segment variant.

    Raw ids mix ``&&``/``$``/dots; the sha1 suffix disambiguates sanitization
    collisions and separates the RC (bbox-drawn) variant from the base one.
    """
    signature = raw_id if rc_signature is None else f"{raw_id}#rc:{rc_signature}"
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", raw_id)[:80]
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}"


def duration_bucket(duration_seconds: float) -> str:
    if duration_seconds < 30:
        return "short"
    if duration_seconds <= 300:
        return "medium"
    return "long"


def extract_rc_signature(item: dict) -> tuple[str, str, list[float]] | None:
    """Return (start_frame_raw, start_frame_resolved_key, bbox) for RC items."""
    rc_info = item.get("RC_info") or {}
    start_frame = rc_info.get("start_frame")
    bbox = rc_info.get("start_frame_bbox")
    if not start_frame or not bbox:
        return None
    return (start_frame, json.dumps(bbox), list(bbox))


def build_row(index: int, item: dict, key: str, duration_seconds: float) -> dict:
    qa_type = item["qa_type"]
    source = item.get("dataset_name") or item.get("data_source") or ""
    return {
        "ID": str(index),
        "video_id": key,
        "query_time": dict(DEFAULT_TIME),
        "type": qa_type,
        "task": report_task(qa_type),
        "duration": duration_bucket(duration_seconds),
        "domain": source,
        "sub_category": "",
        "time_reference": f"0-{duration_seconds:.1f} seconds",
        "question": build_question(item),
        "answer": conversation_value(item, {"gpt", "assistant"}),
        "target_time": dict(DEFAULT_TIME),
        "raw_id": item["id"],
        "fps": item["metadata"].get("fps"),
        "n_frames": len(item.get("video", [])),
        "is_RC": bool(item.get("is_RC")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MedVidBench test set for WorldMM.")
    parser.add_argument("--trainval-json", type=Path, default=Path(DEFAULT_TRAINVAL_JSON))
    parser.add_argument("--test-manifest", type=Path, default=Path(DEFAULT_TEST_MANIFEST))
    parser.add_argument("--frame-root", type=Path, default=Path(DEFAULT_FRAME_ROOT))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--skip-frame-check",
        action="store_true",
        help="Do not stat every frame on disk (synthesis re-checks anyway).",
    )
    args = parser.parse_args()

    manifest = json.loads(args.test_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"test manifest must be a non-empty JSON list: {args.test_manifest}")
    indices = [int(i) for i in manifest]
    if any(i != raw for i, raw in zip(indices, manifest)) or len(set(indices)) != len(indices):
        raise ValueError("test manifest must be a list of unique integers")

    dataset = json.loads(args.trainval_json.read_text(encoding="utf-8"))
    out_of_range = [i for i in indices if i < 0 or i >= len(dataset)]
    if out_of_range:
        raise ValueError(f"manifest indices out of range: {out_of_range[:10]}")

    missing_frames: list[str] = []
    segments: dict[str, dict] = {}
    rows: list[dict] = []
    unknown_qa: list[str] = []

    for index in indices:
        item = dataset[index]
        qa_type = item["qa_type"]
        if output_requirement(qa_type) is None:
            unknown_qa.append(qa_type)

        metadata = item["metadata"]
        fps = float(metadata.get("fps") or 1.0)
        raw_frames: list[str] = list(item.get("video", []))
        if not raw_frames:
            raise ValueError(f"index {index} ({item['id']}): no frames listed")

        rc_signature = None
        rc_box = None
        if item.get("is_RC"):
            rc = extract_rc_signature(item)
            if rc is None:
                raise ValueError(f"index {index} ({item['id']}): is_RC without usable RC_info")
            rc_start_raw, rc_signature, rc_bbox = rc
            rc_box = {"start_frame": resolve_frame_path(rc_start_raw, args.frame_root).as_posix(), "bbox": rc_bbox}

        key = segment_key(item["id"], rc_signature)
        segment = segments.get(key)
        if segment is None:
            resolved = [resolve_frame_path(p, args.frame_root) for p in raw_frames]
            if not args.skip_frame_check:
                missing_frames.extend(
                    resolved[i].as_posix() for i, p in enumerate(resolved) if not p.is_file()
                )
            segment = {
                "key": key,
                "raw_id": item["id"],
                "fps": fps,
                "frame_paths": [p.as_posix() for p in resolved],
                "n_frames": len(resolved),
                "duration_seconds": len(resolved) / fps,
                "is_rc": rc_signature is not None,
                "rc_boxes": [rc_box] if rc_box else [],
                "question_indices": [],
            }
            segments[key] = segment
        segment["question_indices"].append(index)
        rows.append(build_row(index, item, key, segment["duration_seconds"]))

    if unknown_qa:
        raise ValueError(f"unhandled qa_type without output contract: {sorted(set(unknown_qa))}")
    if missing_frames:
        uniq = sorted(set(missing_frames))
        raise SystemExit(
            f"{len(missing_frames)} frame file(s) missing under {args.frame_root} "
            f"({len(uniq)} unique), e.g. {uniq[:5]}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = args.output_dir / "medvidbench_test.json"
    segments_path = args.output_dir / "test_segments.json"
    videos_path = args.output_dir / "test_videos.json"
    eval_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    segments_path.write_text(json.dumps(list(segments.values()), indent=2) + "\n", encoding="utf-8")
    ordered_keys = sorted(segments)
    videos_path.write_text(json.dumps(ordered_keys, indent=2) + "\n", encoding="utf-8")

    durations = [s["duration_seconds"] for s in segments.values()]
    durations.sort()
    total_frames = sum(s["n_frames"] for s in segments.values())
    qa_counts = Counter(row["type"] for row in rows)
    task_counts = Counter(row["task"] for row in rows)
    source_counts = Counter(row["domain"] for row in rows)
    multi_q = sum(1 for s in segments.values() if len(s["question_indices"]) > 1)

    def pct(p):
        return durations[int(p * (len(durations) - 1))]

    print(f"OK: wrote {len(rows)} questions / {len(segments)} segments "
          f"({multi_q} segments with >1 question) -> {eval_path}")
    print(f"OK: segment list -> {segments_path}")
    print(f"OK: video stem list ({len(ordered_keys)}) -> {videos_path}")
    print(f"frames: {total_frames} total, duration sec: min={durations[0]:.1f} "
          f"p50={pct(0.5):.1f} p90={pct(0.9):.1f} max={durations[-1]:.1f} "
          f"sum={sum(durations):.0f}")
    print("qa_type:", json.dumps(dict(sorted(qa_counts.items())), ensure_ascii=False))
    print("task:", json.dumps(dict(sorted(task_counts.items())), ensure_ascii=False))
    print("sources:", json.dumps(dict(sorted(source_counts.items())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
