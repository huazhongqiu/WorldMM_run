"""Freeze question sets and prepare backend-independent, timestamped inputs."""

import argparse
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path
from worldmm.index_manifest import write_json

MOUNT = Path("/myworkspace")
EGO_ROOT = MOUNT / "data/EgoLife/lmms-lab_EgoLife"
LV_ROOT = MOUNT / "data/LVBench"
MME_ROOT = MOUNT / "data/Video-MME/lmms-eval_Video-MME"
SUBJECT = "A1_JAKE"
EXPECTED = {"lvbench": (1549, 103), "egolife": (500, 1), "videomme_long": (900, 300)}
GRANULARITIES = ("10sec", "30sec", "3min", "10min")
EGO_GRANULARITIES = ("30sec", "3min", "10min", "1h")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen input changed: {path}")
        return
    write_json(path, value)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hhmmss(value, fractional=False):
    clock = str(value).zfill(8)
    seconds = int(clock[:2]) * 3600 + int(clock[2:4]) * 60 + int(clock[4:6])
    # EgoLife's last two digits represent the frame number at 30 fps.
    return seconds + int(clock[6:]) / 30 if fractional else float(seconds)


def align_subtitles(cues, start, end, *, origin=0):
    return [
        {"start_time": round(max(a, start) - origin, 6),
         "end_time": round(min(b, end) - origin, 6), "subtitle": text}
        for a, b, text in cues if min(b, end) > max(a, start)
    ]


def read_srt(path, *, origin=0, english_only=False):
    import pysrt

    cues = []
    for cue in pysrt.open(str(path), encoding="utf-8-sig"):
        lines = [line.strip() for line in cue.text.splitlines() if line.strip()]
        if not lines:
            continue
        text = lines[-1] if english_only else " ".join(lines)
        cues.append((origin + cue.start.ordinal / 1000, origin + cue.end.ordinal / 1000, text))
    return sorted(cues)


def frame_indices(fps, frame_count, start, end, budget):
    import numpy as np

    first = max(0, int(np.ceil(start * fps - 1e-7)))
    last = min(frame_count - 1, int(np.ceil(end * fps - 1e-7)) - 1)
    if first > last:
        raise ValueError(f"No decodable frames in [{start}, {end})")
    count = min(budget, last - first + 1)
    # VLM2Vec groups frames in pairs. Supply identical inputs to all encoders.
    count = max(2, count - count % 2)
    return np.linspace(first, last, count).round().astype(int).tolist()


def videomme_rows(rows, subtitle_root):
    selected = []
    for row in rows:
        if row["duration"] != "long":
            continue
        video_id = row["videoID"]
        subtitle = subtitle_root / f"{video_id}.srt"
        choices = {}
        for option in row["options"]:
            match = re.fullmatch(r"([A-D])[.)]\s*(.+)", option.strip(), re.DOTALL)
            if match is None:
                raise ValueError(f"Invalid Video-MME choice: {row['question_id']}")
            choices[f"choice_{match[1].lower()}"] = match[2]
        if len(choices) != 4 or row["answer"] not in ("A", "B", "C", "D"):
            raise ValueError(f"Invalid Video-MME answers: {row['question_id']}")
        selected.append({
            "ID": str(row["question_id"]), "video_id": video_id, "duration": "long",
            "domain": row.get("domain", ""), "sub_category": row.get("sub_category", ""),
            "type": row["task_type"], "question": row["question"], "answer": row["answer"],
            "query_time": {"date": "DAY1", "time": "23595999"},
            "subtitle_path": str(subtitle) if subtitle.is_file() else None, **choices,
        })
    return selected


def lvbench_input():
    helper = Path(__file__).resolve().parents[1] / "data/LVBench/utils/prepare_lvbench.py"
    spec = importlib.util.spec_from_file_location("prepare_lvbench", helper)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    source = LV_ROOT / "zai-org-LVBench/video_info.meta.jsonl"
    entries = converter.load_flattened_entries(source)
    return source, [converter.build_row(row, video_id) for video_id, row in entries.values()]


def freeze_manifests(dataset, root):
    files = {}
    if dataset == "egolife":
        source = EGO_ROOT / f"EgoLifeQA/EgoLifeQA_{SUBJECT}.json"
        rows = read_json(source)
        if {row["query_time"]["date"] for row in rows} != {f"DAY{i}" for i in range(1, 8)}:
            raise ValueError("EgoLife QA does not span all seven days")
        files[SUBJECT] = {"video_root": str(EGO_ROOT / SUBJECT),
                          "subtitle_root": str(EGO_ROOT / f"EgoLifeCap/Transcript/{SUBJECT}")}
    elif dataset == "lvbench":
        source, rows = lvbench_input()
        for row in rows:
            video_id = row["video_id"]
            files[video_id] = {"video_path": str(LV_ROOT / f"AIWinter-LVBench/all_videos/{video_id}.mp4"),
                               "subtitle_path": None}
    else:
        import pyarrow.parquet as pq

        source = MME_ROOT / "videomme/test-00000-of-00001.parquet"
        rows = videomme_rows(pq.read_table(source).to_pylist(), MME_ROOT / "subtitle")
        for row in rows:
            video_id = row["video_id"]
            files[video_id] = {"video_path": str(MME_ROOT / f"data/{video_id}.mp4"),
                               "subtitle_path": row["subtitle_path"]}
        missing = [row["ID"] for row in rows if row["subtitle_path"] is None]
        if len(missing) != 24 or sum(not item["subtitle_path"] for item in files.values()) != 8:
            raise ValueError("Video-MME subtitle coverage differs from the fixed 8 videos / 24 questions")
    expected_questions, expected_videos = EXPECTED[dataset]
    if len(rows) != expected_questions or len({str(row["ID"]) for row in rows}) != expected_questions or len(files) != expected_videos:
        raise ValueError(f"Unexpected {dataset} counts: {len(rows)} questions, {len(files)} video units")
    for record in files.values():
        video = record.get("video_path")
        if video and not Path(video).is_file():
            raise FileNotFoundError(video)
        subtitle = record.get("subtitle_path")
        if subtitle:
            record["subtitle_sha256"] = digest(Path(subtitle))
    freeze_json(root / "questions.json", rows)
    freeze_json(root / "question_ids.json", [str(row["ID"]) for row in rows])
    freeze_json(root / "videos.json", files)
    freeze_json(root / "source.json", {"dataset": dataset, "questions": len(rows), "video_units": len(files),
                                        "annotations": str(source), "annotations_sha256": digest(source)})
    print(f"{dataset}: {len(rows)} questions / {len(files)} video units -> {root}", flush=True)
    return rows, files


def prepare_ego_captions(legacy, captions, files):
    source = legacy / f"data/EgoLife/EgoLifeCap/{SUBJECT}"
    destination = captions / f"EgoLifeCap/{SUBJECT}"
    video_root = Path(files[SUBJECT]["video_root"]).parent
    subtitle_root = Path(files[SUBJECT]["subtitle_root"])
    by_day = defaultdict(list)
    for path in sorted(subtitle_root.glob("DAY*/*.srt")):
        by_day[path.parent.name].extend(
            read_srt(path, origin=hhmmss(path.stem.rsplit("_", 1)[1]), english_only=True))
    if set(by_day) != {f"DAY{i}" for i in range(1, 8)}:
        raise ValueError("EgoLife transcript days are incomplete")
    for granularity in EGO_GRANULARITIES:
        name = f"{SUBJECT}_{granularity}.json"
        rows = read_json(source / name)
        if granularity == "30sec":
            for row in rows:
                path = video_root / row["video_path"].removeprefix("data/EgoLife/")
                if not path.is_file():
                    raise FileNotFoundError(path)
                origin = hhmmss(path.stem.rsplit("_", 1)[1], fractional=True)
                start = max(origin, hhmmss(row["start_time"], fractional=True))
                end = hhmmss(row["end_time"], fractional=True)
                if end <= start:
                    raise ValueError(f"Invalid EgoLife caption interval: {path}")
                row["source_video_path"] = str(path)
                row["start_sec"] = start - origin
                row["end_sec"] = end - origin
                row["subtitles"] = align_subtitles(by_day[row["date"]], start, end, origin=origin)
        freeze_json(destination / name, rows)
    freeze_json(captions / "egolife_sources.json", {
        "captions": {p.name: digest(p) for p in sorted(source.glob(f"{SUBJECT}_*.json"))},
        "subtitles": {str(p.relative_to(subtitle_root)): digest(p)
                      for p in sorted(subtitle_root.glob("DAY*/*.srt"))},
    })
    print(f"EgoLife: aligned subtitles for {len(read_json(destination / f'{SUBJECT}_30sec.json'))} source clips", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=EXPECTED)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ego-captions", action="store_true")
    args = parser.parse_args()
    _, videos = freeze_manifests(args.dataset, args.root / "manifests")
    if args.ego_captions:
        if args.dataset != "egolife":
            parser.error("--ego-captions is only for EgoLife")
        prepare_ego_captions(args.root.parent, args.root / "captions", videos)


if __name__ == "__main__":
    main()
