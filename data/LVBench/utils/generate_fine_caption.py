#!/usr/bin/env python3
"""Generate fine-grained PURE-VISUAL captions from videos (no transcripts/ASR).

LVBench variant of ``preprocess/episodic_memory/generate_fine_caption.py``:
the video is split into fixed-length segments and each segment is captioned
by the served VLM from sampled frames only. Output schema matches the
Video-MME caption files consumed by ``worldmm.memory.episodic.multiscale``::

    [{"start_time": "HHMMSS00", "end_time": "HHMMSS00", "text": ...,
      "date": "DAY1", "video_path": "/abs/path/to/video.mp4"}, ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm

from worldmm.llm import LLMModel

SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
SYSTEM_PROMPT = """You are an expert video captioner.

You will receive a short video segment represented by ordered frames.
Write a caption describing the visual content of the segment.

Guidelines:
- Describe visible actions, people, objects, on-screen text, and environment.
- Transcribe short on-screen text verbatim when it is readable.
- Keep the caption factual and neutral.
- Do not mention frames, timestamps, or that the input came from frames.
- Avoid speculation about emotions or intentions unless clearly visible.

Output only the final caption text."""


@dataclass(slots=True)
class FrameSample:
    timestamp_seconds: float
    image: Image.Image


@dataclass(slots=True)
class Segment:
    start_seconds: float
    end_seconds: float


@dataclass(slots=True)
class VideoReaderContext:
    reader: VideoReader
    average_fps: float
    total_frames: int
    lock: Lock


class CaptionGenerationError(RuntimeError):
    """Raised when a segment caption cannot be generated."""


def load_video_list(video_list_path: Path | None) -> list[str] | None:
    if video_list_path is None:
        return None
    values = json.loads(video_list_path.read_text(encoding="utf-8"))
    if isinstance(values, dict):
        for key in ("videos", "video_ids", "keys", "sample_ids"):
            if key in values:
                values = values[key]
                break
        else:
            raise ValueError(f"video list dict must contain one of {('videos', 'video_ids', 'keys', 'sample_ids')}")
    return [str(v) for v in values]


def build_video_index(video_root: Path) -> dict[str, Path]:
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    if video_root.is_file():
        return {video_root.stem: video_root}

    index: dict[str, Path] = {}
    for video_file in sorted(
        p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
    ):
        index.setdefault(video_file.stem, video_file)
    if not index:
        raise FileNotFoundError(f"No supported video files found under: {video_root}")
    return index


def format_caption_time(seconds: float, *, round_up: bool) -> str:
    if round_up:
        whole_seconds = max(0, int(math.ceil(seconds - 1e-9)))
    else:
        whole_seconds = max(0, int(math.floor(seconds + 1e-9)))
    hours, rem = divmod(whole_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}{minutes:02d}{secs:02d}00"


def format_clock(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1_000)
    if millis:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_video_duration(video_reader: VideoReader) -> tuple[float, float]:
    total_frames = len(video_reader)
    if total_frames <= 0:
        raise ValueError("Video file is empty.")
    average_fps = float(video_reader.get_avg_fps() or 0.0)
    if average_fps <= 0:
        average_fps = 1.0
    duration = total_frames / average_fps
    if duration <= 0:
        duration = 1.0 / average_fps
    return duration, average_fps


def compute_sample_seconds(start_seconds: float, end_seconds: float, sample_fps: float) -> list[float]:
    step = 1.0 / sample_fps
    sample_seconds: list[float] = []
    k = 0
    while True:
        t = start_seconds + k * step
        if t >= end_seconds - 1e-9:
            break
        sample_seconds.append(t)
        k += 1
    if not sample_seconds:
        sample_seconds = [start_seconds]
    return sample_seconds


def frame_to_image(frame: Any, longest_edge: int) -> Image.Image:
    image = Image.fromarray(frame)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if longest_edge > 0:
        width, height = image.size
        longest = max(width, height)
        if longest > longest_edge:
            scale = longest_edge / longest
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
    return image


def sample_segment_frames(
    video_reader_ctx: VideoReaderContext,
    start_seconds: float,
    end_seconds: float,
    sample_fps: float,
    longest_edge: int,
) -> list[FrameSample]:
    sample_seconds = compute_sample_seconds(start_seconds, end_seconds, sample_fps)
    frame_indices = [
        min(max(int(round(t * video_reader_ctx.average_fps)), 0), video_reader_ctx.total_frames - 1)
        for t in sample_seconds
    ]
    if not frame_indices:
        sample_seconds = [start_seconds]
        frame_indices = [0]

    with video_reader_ctx.lock:
        frame_batch = video_reader_ctx.reader.get_batch(frame_indices).asnumpy()

    return [
        FrameSample(timestamp_seconds=t, image=frame_to_image(frame, longest_edge))
        for t, frame in zip(sample_seconds, frame_batch, strict=True)
    ]


def build_segment_prompt(segment: Segment, frames: list[FrameSample]) -> list[dict[str, Any]]:
    intro_text = "\n".join(
        [
            f"Segment window: {format_clock(segment.start_seconds)} to {format_clock(segment.end_seconds)}",
            "The following frames are ordered chronologically within the segment.",
        ]
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": intro_text}]
    for frame in frames:
        content.append({"type": "text", "text": f"Frame timestamp: {format_clock(frame.timestamp_seconds)}"})
        content.append({"type": "image", "image": frame.image})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def release_frames(frames: list[FrameSample]) -> None:
    for frame in frames:
        try:
            frame.image.close()
        except Exception:
            pass


def generate_caption_with_retry(
    segment: Segment,
    video_reader_ctx: VideoReaderContext,
    model: LLMModel,
    sample_fps: float,
    longest_edge: int,
    max_retries: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        frames = sample_segment_frames(video_reader_ctx, segment.start_seconds, segment.end_seconds, sample_fps, longest_edge)
        try:
            response = model.generate(build_segment_prompt(segment, frames))
            return response.strip() if response else ""
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(30.0, 2.0 * (2**attempt)))
        finally:
            release_frames(frames)
    raise CaptionGenerationError(
        f"Error generating caption for segment {format_clock(segment.start_seconds)} to "
        f"{format_clock(segment.end_seconds)} after {max_retries + 1} attempts: {last_error}"
    )


def build_caption_entry(segment: Segment, video_path: Path, caption_text: str) -> dict[str, str]:
    return {
        "start_time": format_caption_time(segment.start_seconds, round_up=False),
        "end_time": format_caption_time(segment.end_seconds, round_up=True),
        "text": caption_text,
        "date": "DAY1",
        "video_path": video_path.as_posix(),
    }


def process_video(
    video_file: Path,
    output_file: Path,
    model: LLMModel,
    unit_time: int,
    sample_fps: float,
    longest_edge: int,
    max_workers: int,
    max_retries: int,
) -> int:
    video_reader = VideoReader(str(video_file), ctx=cpu(0))
    duration, average_fps = get_video_duration(video_reader)
    video_reader_ctx = VideoReaderContext(
        reader=video_reader,
        average_fps=average_fps,
        total_frames=len(video_reader),
        lock=Lock(),
    )

    num_segments = max(1, int(math.ceil(duration / unit_time)))
    segments = [
        Segment(start_seconds=float(i * unit_time), end_seconds=min(float((i + 1) * unit_time), duration))
        for i in range(num_segments)
    ]

    results: list[tuple[int, str]] = []
    generation_errors: list[str] = []
    workers = max(1, min(max_workers, len(segments)))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(
                generate_caption_with_retry, segment, video_reader_ctx, model, sample_fps, longest_edge, max_retries
            ): idx
            for idx, segment in enumerate(segments)
        }
        progress = tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc=f"  Captioning {video_file.name}",
            unit="segment",
            leave=False,
        )
        for future in progress:
            idx = future_to_idx[future]
            try:
                caption_text = future.result()
            except Exception as exc:
                tqdm.write(f"      {exc}")
                generation_errors.append(str(exc))
                continue
            results.append((idx, caption_text))

    if generation_errors:
        raise CaptionGenerationError(
            f"Caption generation failed for {len(generation_errors)}/{len(segments)} segment(s); output file was not written."
        )

    ordered_results = sorted(results, key=lambda item: item[0])
    caption_entries = [
        build_caption_entry(segments[idx], video_file, caption_text)
        for idx, caption_text in ordered_results
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(caption_entries, f, indent=2, ensure_ascii=False)
    tmp_file.replace(output_file)

    return len(caption_entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pure-visual fine captions from videos.")
    parser.add_argument("--video-path", type=Path, required=True, help="Path to a video file or a directory of videos.")
    parser.add_argument("--video-list", type=Path, default=None, help="Optional JSON list (or dict with videos/video_ids/keys) of video stems to process.")
    parser.add_argument("--output-path", type=Path, required=True, help="Caption output directory; writes {video_stem}/{unit_time}sec.json.")
    parser.add_argument("--model", type=str, default="Qwen3.5-4B", help="LMDeploy model name.")
    parser.add_argument("--unit-time", type=int, default=10, help="Segment length in seconds.")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Frame sampling rate per segment.")
    parser.add_argument("--max-frame-longest-edge", type=int, default=0, help="Resize frames so the longest edge is at most this (0 = native resolution, same as original source code; e.g. 1280 roughly halves runtime).")
    parser.add_argument("--max-workers", type=int, default=16, help="Max concurrent segment requests per video.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per segment on transient errors.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing caption files.")
    args = parser.parse_args()

    if args.unit_time < 1:
        parser.error("--unit-time must be at least 1.")
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be positive.")

    video_index = build_video_index(args.video_path)
    stems = load_video_list(args.video_list)
    if stems is None:
        stems = sorted(video_index.keys())
    missing = [stem for stem in stems if stem not in video_index]
    if missing:
        parser.error(f"{len(missing)} video(s) not found under {args.video_path}, e.g. {missing[:5]}")

    model = LLMModel(model_name=args.model)

    processed = 0
    skipped = 0
    failed = 0
    failed_stems: list[str] = []

    progress_bar = tqdm(stems, desc="Generating fine captions", unit="video")
    for stem in progress_bar:
        progress_bar.set_postfix(processed=processed, skipped=skipped, failed=failed, video=stem)
        video_file = video_index[stem]
        output_file = args.output_path / stem / f"{args.unit_time}sec.json"
        try:
            if output_file.exists() and not args.overwrite:
                skipped += 1
                continue
            segment_count = process_video(
                video_file=video_file,
                output_file=output_file,
                model=model,
                unit_time=args.unit_time,
                sample_fps=args.sample_fps,
                longest_edge=args.max_frame_longest_edge,
                max_workers=args.max_workers,
                max_retries=args.retries,
            )
            processed += 1
            tqdm.write(f"Generated {segment_count} captions for {video_file.name} -> {output_file}")
        except Exception as exc:
            failed += 1
            failed_stems.append(stem)
            tqdm.write(f"Failed to process {stem}: {exc}")

    progress_bar.close()
    print(
        "Finished fine caption generation: "
        f"processed={processed} skipped={skipped} failed={failed} output={args.output_path}"
    )
    if failed_stems:
        print("Failed videos: " + ", ".join(failed_stems))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
