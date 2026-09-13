#!/usr/bin/env python3
"""Synthesize one H.264 mp4 per MedVidBench segment from its frame list.

Each entry of ``test_segments.json`` (produced by ``prepare_medvidbench.py``)
is a ``video_id&&start&&end&&fps`` segment whose ``frame_paths`` were sampled
by the benchmark at ``metadata.fps``. Encoding them at that same framerate
makes mp4 time == segment-local seconds (frame i sits at i/fps), which is the
convention the official evaluator's frame conversion and the WorldMM caption
pipeline both assume.

Frames are streamed to ffmpeg's image2pipe demuxer as JPEG (PIL handles the
mixed jpg/png sources and draws the green RC bounding box inline), so no
staging copies of the 126k frames are needed. Everything resumes: existing,
verified mp4s are skipped unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw

DEFAULT_SEGMENTS_JSON = "/myworkspace/projects/worldmm_needed/medvidbench/qa/test_segments.json"
DEFAULT_OUTPUT_DIR = "/workspace/worldmm/medvidbench/videos"
DEFAULT_MAPPING_JSON = "/workspace/worldmm/medvidbench/segment_videos.json"


def default_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class SynthesisError(RuntimeError):
    pass


def encode_frame(path: Path, rc_boxes: dict[str, list[float]]) -> bytes:
    with Image.open(path) as source:
        image = source.convert("RGB")
        bbox = rc_boxes.get(path.as_posix())
        if bbox:
            ImageDraw.Draw(image).rectangle(bbox, outline=(0, 255, 0), width=8)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def synthesize_segment(
    segment: dict,
    output_dir: Path,
    ffmpeg_bin: str,
    crf: int,
    preset: str,
    verify: bool,
) -> dict:
    key = segment["key"]
    out_path = output_dir / f"{key}.mp4"
    rc_boxes = {box["start_frame"]: list(box["bbox"]) for box in segment.get("rc_boxes", []) if box}
    fps = float(segment["fps"])
    frame_paths = [Path(p) for p in segment["frame_paths"]]
    missing = next((p for p in frame_paths if not p.is_file()), None)
    if missing is not None:
        raise SynthesisError(f"[{key}] frame not found: {missing}")

    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "image2pipe", "-framerate", f"{fps}", "-i", "-",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags", "+faststart", "-an", str(out_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
    try:
        for path in frame_paths:
            process.stdin.write(encode_frame(path, rc_boxes))
    except BrokenPipeError:
        pass
    finally:
        if process.stdin:
            process.stdin.close()
        stderr = process.stderr.read() if process.stderr else b""
        return_code = process.wait()
    if return_code != 0:
        out_path.unlink(missing_ok=True)
        raise SynthesisError(f"[{key}] ffmpeg exited {return_code}: {stderr.decode(errors='ignore')[-800:]}")

    if verify:
        from decord import VideoReader, cpu

        reader = VideoReader(str(out_path), ctx=cpu(0))
        if len(reader) != len(frame_paths):
            out_path.unlink(missing_ok=True)
            raise SynthesisError(
                f"[{key}] decoded frame count {len(reader)} != {len(frame_paths)} listed frames"
            )

    return {
        "key": key,
        "mp4": out_path.as_posix(),
        "fps": fps,
        "n_frames": len(frame_paths),
        "duration_seconds": len(frame_paths) / fps,
        "is_rc": segment.get("is_rc", False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize per-segment mp4s from MedVidBench frames.")
    parser.add_argument("--segments-json", type=Path, default=Path(DEFAULT_SEGMENTS_JSON))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mapping-json", type=Path, default=Path(DEFAULT_MAPPING_JSON))
    parser.add_argument("--video-list", type=Path, default=None,
                        help="Optional JSON list of segment keys to process (default: all).")
    parser.add_argument("--ffmpeg-bin", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="veryfast")
    parser.add_argument("--no-verify", action="store_true", help="Skip the decord frame-count check.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    segments = json.loads(args.segments_json.read_text(encoding="utf-8"))
    if args.video_list:
        wanted = set(json.loads(args.video_list.read_text(encoding="utf-8")))
        segments = [s for s in segments if s["key"] in wanted]
        missing_keys = wanted - {s["key"] for s in segments}
        if missing_keys:
            raise SystemExit(f"{len(missing_keys)} requested key(s) not in segments json: {sorted(missing_keys)[:5]}")

    ffmpeg_bin = args.ffmpeg_bin or default_ffmpeg()
    verify = not args.no_verify
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mapping_json.parent.mkdir(parents=True, exist_ok=True)

    mapping_lock = threading.Lock()

    def record(entry: dict) -> None:
        with mapping_lock:
            mapping = {}
            if args.mapping_json.exists():
                try:
                    mapping = json.loads(args.mapping_json.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    mapping = {}
            mapping[entry["key"]] = entry
            tmp = args.mapping_json.with_suffix(".tmp")
            tmp.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
            tmp.replace(args.mapping_json)

    def already_done(segment: dict) -> bool:
        if args.overwrite:
            return False
        if args.mapping_json.exists():
            try:
                entry = json.loads(args.mapping_json.read_text(encoding="utf-8")).get(segment["key"])
                if entry and Path(entry["mp4"]).is_file():
                    return True
            except json.JSONDecodeError:
                pass
        return False

    failed: list[str] = []
    done = skipped = 0
    progress_lock = threading.Lock()

    def progress() -> str:
        return f"done={done} skipped={skipped} failed={len(failed)} / {len(segments)}"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                synthesize_segment, segment, args.output_dir, ffmpeg_bin, args.crf, args.preset, verify
            ): segment
            for segment in segments
            if not already_done(segment)
        }
        skipped = len(segments) - len(futures)
        for future in as_completed(futures):
            segment = futures[future]
            try:
                entry = future.result()
            except Exception as exc:
                with progress_lock:
                    failed.append(segment["key"])
                    print(f"FAIL [{segment['key']}]: {exc} ({progress()})", file=sys.stderr, flush=True)
                continue
            record(entry)
            with progress_lock:
                done += 1
                print(f"ok [{entry['key']}] {entry['n_frames']}f @ {entry['fps']}fps "
                      f"= {entry['duration_seconds']:.1f}s ({progress()})", flush=True)

    if failed:
        raise SystemExit(f"{len(failed)} segment(s) failed to synthesize, e.g. {failed[:5]}")
    print(f"OK: {done} mp4(s) synthesized, {skipped} skipped, mapping at {args.mapping_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
