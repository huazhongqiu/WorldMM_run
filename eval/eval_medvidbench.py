#!/usr/bin/env python3
"""MedVidBench (open-ended) evaluation with WorldMM.

Standalone runner for the MedVidBench test split (the videospy 1,124-question
manifest over ``medvidu_eccv2026_trainval.json``). Unlike ``eval/eval.py``
(MCQ), questions are answered by free-form generation with the
``qa_medvidbench`` template and ``choices=None``; predictions are checkpointed
to ``records.jsonl`` and exported as the official leaderboard
``submission.json`` (``[{id, qa_type, prediction}]``) for the official
``evaluate_predictions.py`` scorer.

Rows are grouped by ``video_id`` (one segment = one WorldMemory index), the
same convention as ``eval/eval.py``; with ``--workers > 1`` segments are
processed concurrently on one shared, lock-serialized EmbeddingModel (the
``eval/eval_egolife.py`` pattern).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, local
from typing import Any, Dict, List, Optional

from tqdm import tqdm

default_src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, os.environ.get("WORLDMM_SRC", str(default_src)))

from worldmm.embedding import EmbeddingModel  # noqa: E402
from worldmm.llm import LLMModel, PromptTemplateManager  # noqa: E402
from worldmm.memory import WorldMemory, QAResult  # noqa: E402
from worldmm.run_log import (  # noqa: E402
    agent_round as log_agent_round,
    answer_status as log_answer_status,
    emit as log_line,
    progress as log_progress,
    section as log_section,
)

logger = logging.getLogger("medvidbench_eval")

MEDVIDBENCH_GRANULARITIES = ["10sec", "30sec", "3min", "10min"]
DAY_PREFIX = "1"  # "DAY1" -> the leading digit of eval.py's QUERY_TIME convention


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)
    for name in (
        "httpx",
        "httpcore",
        "openai",
        "hipporag",
        "sentence_transformers",
        "transformers",
        "worldmm.memory",
        "worldmm.embedding",
        "worldmm.llm",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def compact(value: Any, limit: int = 120) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def query_time_int(row: Dict[str, Any]) -> int:
    return int(DAY_PREFIX + str(row["query_time"]["time"]))


def usage_total(*models) -> Dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for model in models:
        # LLMModel is a dispatcher; the usage counters live on the provider.
        inner = getattr(model, "model", model)
        snapshot = inner.usage_snapshot() if hasattr(inner, "usage_snapshot") else getattr(inner, "usage", None) or {}
        totals["input_tokens"] += max(0, int(snapshot.get("prompt_tokens", 0)))
        totals["output_tokens"] += max(0, int(snapshot.get("completion_tokens", 0)))
        totals["total_tokens"] += max(0, int(snapshot.get("total_tokens", 0)))
    return totals


def usage_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {key: max(0, after[key] - before[key]) for key in before}


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_records(path: str) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return latest
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[record["sample_key"]] = record
    return latest


def completion_counts(rows: List[Dict[str, Any]], latest: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    counts = {"success": 0, "error": 0, "missing": 0, "total": len(rows)}
    for row in rows:
        record = latest.get(str(row["ID"]))
        if record is None:
            counts["missing"] += 1
        elif is_successful_record(record):
            counts["success"] += 1
        else:
            counts["error"] += 1
    return counts


def is_successful_record(record: Optional[Dict[str, Any]]) -> bool:
    return bool(
        record
        and record.get("status") == "success"
        and answer_status(record.get("prediction")) == "success"
    )


def pending_groups(
    groups: Dict[str, List[Dict[str, Any]]],
    latest: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        video_id: pending_rows
        for video_id, rows in groups.items()
        if (pending_rows := [row for row in rows if not is_successful_record(latest.get(row["ID"]))])
    }


def answer_status(response: Any) -> str:
    normalized = str(response or "").strip()
    if not normalized:
        return "empty_response"
    if normalized == "Unable to generate answer":
        return "generation_error"
    return "success"


def load_required_episodic_openie(memory: WorldMemory, path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required persisted OpenIE file is missing: {path}")
    memory.load_episodic_openie(path)


class RecordWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lock = Lock()

    def append(self, record: Dict[str, Any]) -> None:
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def caption_files_for(caption_dir: str, video_id: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for granularity in MEDVIDBENCH_GRANULARITIES:
        path = os.path.join(caption_dir, str(video_id), f"{granularity}.json")
        if os.path.exists(path):
            files[granularity] = path
        else:
            logger.warning("Missing caption file: %s", path)
    return files


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rows: List[Dict[str, Any]] = load_json(args.eval_json)
        self.groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            self.groups[row["video_id"]].append(row)
        self.records_path = os.path.join(args.output_dir, "records.jsonl")
        self.writer = RecordWriter(self.records_path)
        self.existing = latest_records(self.records_path)
        self.print_lock = Lock()
        self.completed_count = 0
        self.completed_units = 0
        self.embedding_model: Optional[EmbeddingModel] = None
        self.embedding_init_lock = Lock()
        self.embedding_call_lock = Lock()
        self.worker_local = local()
        self.worker_memories: List[WorldMemory] = []
        self.worker_lock = Lock()

    # ---- worker-local models -------------------------------------------------
    def ensure_embedding_model(self) -> EmbeddingModel:
        if self.embedding_model is None:
            with self.embedding_init_lock:
                if self.embedding_model is None:
                    logger.info("Initializing embedding model ...")
                    embedding_model = EmbeddingModel()
                    embedding_model.load_model("text")
                    self.embedding_model = embedding_model
        return self.embedding_model

    def current_worker(self) -> local:
        if hasattr(self.worker_local, "memory"):
            return self.worker_local
        args = self.args
        embedding = SynchronizedEmbedding(self.ensure_embedding_model(), self.embedding_call_lock)
        retriever = LLMModel(model_name=args.retriever_model)
        respond = LLMModel(model_name=args.respond_model)
        memory = WorldMemory(
            embedding_model=embedding,
            retriever_llm_model=retriever,
            respond_llm_model=respond,
            prompt_template_manager=PromptTemplateManager(),
            episodic_granularities=MEDVIDBENCH_GRANULARITIES,
            episodic_cache_root=os.path.join(args.episodic_cache_dir, "worker-bootstrap"),
            qa_template_name="qa_medvidbench",
            max_rounds=args.max_rounds,
            max_errors=args.max_errors,
        )
        memory.set_retrieval_top_k(
            episodic=args.episodic_top_k,
            semantic=args.semantic_top_k,
            visual=args.visual_top_k,
        )
        self.worker_local.memory = memory
        self.worker_local.retriever = retriever
        self.worker_local.respond = respond
        with self.worker_lock:
            self.worker_memories.append(memory)
        return self.worker_local

    # ---- per-segment processing ---------------------------------------------
    def process_group(self, video_id: str, rows: List[Dict[str, Any]]) -> None:
        args = self.args
        state = self.current_worker()
        memory: WorldMemory = state.memory
        until_time = max(query_time_int(row) for row in rows)

        with self.print_lock:
            self.completed_units += 1
            unit_current = self.completed_units
        log_progress(
            "UNIT", unit_current, len(self.groups),
            f"id={video_id} | questions={len(rows)}",
        )

        memory.reset()
        memory.episodic_memory.save_dir_root = os.path.join(
            args.episodic_cache_dir, str(video_id), "episodic_memory"
        )

        caption_files = caption_files_for(args.caption_dir, video_id)
        if not caption_files:
            for row in rows:
                self.emit(row, None, "no_caption_files", "No caption files")
            return

        memory.load_episodic_captions(caption_files=caption_files)

        episodic_openie_file = os.path.join(
            args.metadata_dir, "episodic_memory", str(video_id),
            f"openie_results_{args.retriever_model}.json",
        )
        try:
            load_required_episodic_openie(memory, episodic_openie_file)
        except Exception as exc:
            logger.error("Persisted OpenIE invalid for segment %s: %s", video_id, exc)
            for row in rows:
                self.emit(row, None, "episodic_openie_error", f"OpenIE error: {exc}")
            return

        semantic_file = os.path.join(
            args.metadata_dir, "semantic_memory", str(video_id),
            f"semantic_consolidation_results_{args.retriever_model}.json",
        )
        if os.path.exists(semantic_file):
            memory.load_semantic_triples(file_path=semantic_file)

        visual_pkl = os.path.join(
            args.metadata_dir, "visual_memory", str(video_id), "visual_embeddings.pkl"
        )
        if os.path.exists(visual_pkl) and "10sec" in caption_files:
            memory.load_visual_clips(
                embeddings_path=visual_pkl,
                clips_data=load_json(caption_files["10sec"]),
            )

        index_started_at = time.perf_counter()
        index_label = f"[VIDEO {unit_current:03d}]"
        logger.info("%s building memory index...", index_label)
        try:
            memory.index(until_time)
        except Exception as exc:
            # A transient failure here is usually a model load racing another
            # process for GPU memory (meta-tensor/OOM during lazy init); the
            # fresh in-process retry succeeds once memory has settled.
            logger.warning("%s index attempt 1 failed (%s); retrying once", index_label, exc)
            time.sleep(10.0)
            try:
                memory.reset()
                memory.episodic_memory.save_dir_root = os.path.join(
                    args.episodic_cache_dir, str(video_id), "episodic_memory"
                )
                memory.load_episodic_captions(caption_files=caption_files)
                load_required_episodic_openie(memory, episodic_openie_file)
                if os.path.exists(semantic_file):
                    memory.load_semantic_triples(file_path=semantic_file)
                if os.path.exists(visual_pkl) and "10sec" in caption_files:
                    memory.load_visual_clips(
                        embeddings_path=visual_pkl,
                        clips_data=load_json(caption_files["10sec"]),
                    )
                memory.index(until_time)
            except Exception as retry_exc:
                logger.error("%s index=FAILED after %.1fs: %s", index_label, time.perf_counter() - index_started_at, retry_exc)
                logger.error("Indexing failed for segment %s: %s", video_id, retry_exc)
                for row in rows:
                    self.emit(row, None, "index_error", f"Index error: {retry_exc}")
                return
        logger.info("%s index done in %.1fs", index_label, time.perf_counter() - index_started_at)

        for row in rows:
            started = time.perf_counter()
            tokens_before = usage_total(state.retriever, state.respond)
            qa_result: Optional[QAResult] = None

            def round_callback(event: str, details: Dict[str, Any]) -> None:
                if event == "round":
                    log_line(log_agent_round(
                        details["round_num"],
                        args.max_rounds,
                        details["decision"],
                        memory_type=details.get("memory_type"),
                        search_query=details.get("search_query"),
                    ))
                elif event == "answer_generation":
                    log_line(log_agent_round(details["round_num"], args.max_rounds, "answer"))

            try:
                qa_result = memory.answer(
                    query=row["question"],
                    choices=None,
                    until_time=until_time,
                    progress_callback=round_callback,
                )
                response = qa_result.answer
                status = answer_status(response)
            except Exception as exc:
                logger.error("Error answering %s: %s", row["ID"], exc)
                response = ""
                status = "error"
            tokens = usage_delta(tokens_before, usage_total(state.retriever, state.respond))
            self.emit(
                row,
                qa_result,
                status,
                response,
                tokens=tokens,
                elapsed=time.perf_counter() - started,
            )

    def emit(
        self,
        row: Dict[str, Any],
        qa_result: Optional[QAResult],
        status: str,
        response: str,
        tokens: Optional[Dict[str, int]] = None,
        elapsed: float = 0.0,
    ) -> None:
        record = {
            "sample_key": row["ID"],
            "status": status,
            "id": row["raw_id"],
            "qa_type": row["type"],
            "task": row.get("task", ""),
            "video_id": row["video_id"],
            "prediction": response if status == "success" else "",
            "data_source": row.get("domain", ""),
            "question": row["question"],
            "ground_truth": row.get("answer", ""),
            "struc_info": row.get("struc_info"),
            "agent_rounds": qa_result.num_rounds if qa_result else 0,
            "num_rounds": qa_result.num_rounds if qa_result else 0,
            "token_usage": tokens or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "elapsed_seconds": round(elapsed, 2),
            "completed": qa_result is not None,
        }
        if qa_result is not None:
            record["round_history"] = qa_result.round_history
        self.writer.append(record)
        self.existing[record["sample_key"]] = record
        with self.print_lock:
            self.completed_count += 1
            log_line(log_answer_status(
                status.upper(),
                question_id=row["ID"],
                answer=compact(response, 80) if status == "success" else "",
                elapsed_seconds=record["elapsed_seconds"],
                tokens=record["token_usage"]["total_tokens"],
            ))

    # ---- outputs --------------------------------------------------------------
    def write_outputs(self, run_seconds: float) -> None:
        args = self.args
        records = {key: rec for key, rec in self.existing.items() if is_successful_record(rec)}
        results = []
        for row in self.rows:
            rec = records.get(row["ID"])
            results.append({
                "ID": row["ID"],
                "video_id": row["video_id"],
                "type": row["type"],
                "task": row.get("task", ""),
                "duration": row.get("duration", ""),
                "domain": row.get("domain", ""),
                "question": row["question"],
                "answer": row.get("answer", ""),
                "prediction": (rec or {}).get("prediction", ""),
                "status": (rec or {}).get("status", "missing"),
                "num_rounds": (rec or {}).get("num_rounds", 0),
                "token_usage": (rec or {}).get("token_usage", {}),
                "elapsed_seconds": (rec or {}).get("elapsed_seconds", 0.0),
            })

        model_dir = args.respond_model.replace("-", "_")
        out_dir = os.path.join(args.output_dir, f"{model_dir}_{model_dir}")
        os.makedirs(out_dir, exist_ok=True)
        eval_path = os.path.join(out_dir, f"{args.eval_name}_eval.json")
        with open(eval_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)

        submission = [
            {"id": row["raw_id"], "qa_type": row["type"], "prediction": records.get(row["ID"], {}).get("prediction", "")}
            for row in self.rows
        ]
        submission_path = os.path.join(args.output_dir, "submission.json")
        with open(submission_path, "w", encoding="utf-8") as handle:
            json.dump(submission, handle, indent=2, ensure_ascii=False)

        manifest = {
            "dataset": "medvidbench",
            "dataset_variant": "test",
            "mode": args.mode,
            "agent": "WorldMM",
            "model": args.respond_model,
            "retriever_model": args.retriever_model,
            "metadata_path": os.path.abspath(args.eval_json),
            "ground_truth_path": os.path.abspath(args.ground_truth_json),
            "sample_keys": [row["ID"] for row in self.rows],
            "eval_name": args.eval_name,
            "workers": args.workers,
            "run_seconds": round(run_seconds, 1),
        }
        manifest_path = os.path.join(args.output_dir, "run.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for rec in records.values():
            for key in totals:
                totals[key] += rec.get("token_usage", {}).get(key, 0)
        usage_path = os.path.join(args.output_dir, "token_usage.json")
        with open(usage_path, "w", encoding="utf-8") as handle:
            json.dump(totals, handle, indent=2)

        logger.info("Eval json: %s", eval_path)
        logger.info("Submission: %s", submission_path)
        logger.info("Token usage: %s", usage_path)
        logger.info("Run manifest: %s", manifest_path)

    # ---- main -----------------------------------------------------------------
    def run(self) -> Dict[str, int]:
        args = self.args
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.episodic_cache_dir, exist_ok=True)

        pending = pending_groups(self.groups, self.existing)
        resumed = len(self.rows) - sum(len(rows) for rows in pending.values())
        self.completed_units = len(self.groups) - len(pending)
        logger.info(
            "Total questions: %d across %d segments (resumed %d, pending %d in %d segments)",
            len(self.rows), len(self.groups), resumed,
            sum(len(rows) for rows in pending.values()), len(pending),
        )
        if not pending:
            self.write_outputs(0.0)
            return completion_counts(self.rows, self.existing)

        started = time.perf_counter()
        log_section("INFERENCE")
        workers = max(1, args.workers)
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self.process_group, vid, rows) for vid, rows in pending.items()]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Segments", unit="segment"):
                    future.result()
        else:
            for video_id, rows in tqdm(pending.items(), desc="Segments", unit="segment"):
                self.process_group(video_id, rows)

        for memory in self.worker_memories:
            try:
                memory.cleanup()
            except Exception:
                pass
        self.write_outputs(time.perf_counter() - started)
        return completion_counts(self.rows, self.existing)


class SynchronizedEmbedding:
    """Serialize GPU embedding calls while workers issue LLM requests concurrently."""

    def __init__(self, model: EmbeddingModel, lock: Lock) -> None:
        self._model = model
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.load_model(*args, **kwargs)

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.encode(*args, **kwargs)

    def encode_text(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.encode_text(*args, **kwargs)

    def encode_vis_query(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.encode_vis_query(*args, **kwargs)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedVidBench Evaluation with WorldMM")
    parser.add_argument("--eval-json", required=True, help="WorldMM qa json from prepare_medvidbench.py")
    parser.add_argument("--ground-truth-json",
                        default="/myworkspace/data/MedVidBench/MedVidU_ECCV2026_TrainVal/medvidu_eccv2026_trainval.json",
                        help="Raw trainval json (GT source for the official evaluator)")
    parser.add_argument("--caption-dir", required=True)
    parser.add_argument("--metadata-dir", required=True, help="Root with {semantic,visual}_memory/{segment_key}/")
    parser.add_argument("--retriever-model", default="Qwen3.5-4B")
    parser.add_argument("--respond-model", default="Qwen3.5-4B")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-errors", type=int, default=5)
    parser.add_argument("--episodic-top-k", type=int, default=3)
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--visual-top-k", type=int, default=3)
    parser.add_argument("--episodic-cache-dir", default="/myworkspace/projects/output/worldmm/medvidbench/cache")
    parser.add_argument("--output-dir", default="/myworkspace/projects/output/worldmm/medvidbench")
    parser.add_argument("--eval-name", default="medvidbench")
    parser.add_argument("--mode", default="test", choices=["test", "smoke"])
    parser.add_argument("--workers", type=int, default=1, help="Concurrent segment workers")
    parser.add_argument("--require-complete", action="store_true", help="Exit non-zero unless every prediction is non-empty.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    configure_logging()
    args = parse_args(argv)
    counts = Runner(args).run()
    logger.info(
        "Completion: success=%d error=%d missing=%d total=%d",
        counts["success"], counts["error"], counts["missing"], counts["total"],
    )
    if args.require_complete and counts["success"] != counts["total"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
