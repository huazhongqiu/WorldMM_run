#!/usr/bin/env python3
"""
EgoLifeQA evaluation script using WorldMM unified memory system.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from threading import Lock, local
import json
import re
import glob
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from typing import Callable
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
def configure_console_logging() -> None:
    """Keep evaluator output readable while preserving errors from dependencies."""
    logging.basicConfig(level=logging.INFO)
    for logger_name in (
        "httpx",
        "httpcore",
        "openai",
        "hipporag",
        "sentence_transformers",
        "worldmm.memory",
        "worldmm.llm",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logging.getLogger("worldmm.memory").setLevel(logging.ERROR)
    logger.setLevel(logging.WARNING)

def compact_text(value: Any, limit: int = 180) -> str:
    """Render a one-line log value without flooding the console."""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."

def render_progress_bar(completed: int, total: int, width: int = 20) -> str:
    """Render a bounded one-line progress bar for long indexing stages."""
    safe_total = max(total, 1)
    safe_completed = min(max(completed, 0), safe_total)
    fraction = safe_completed / safe_total
    filled = round(width * fraction)
    return f"[{'#' * filled}{'-' * (width - filled)}] {fraction * 100:.1f}% ({safe_completed}/{safe_total})"



class WorkflowProgressLogger:
    """Serialize concise, human-readable evaluator progress from worker threads."""

    def __init__(self, writer: Optional[Callable[[str], None]] = None) -> None:
        self._writer = writer or (lambda line: print(line, flush=True))
        self._lock = Lock()

    def _write(self, line: str) -> None:
        with self._lock:
            self._writer(line)

    @staticmethod
    def _fields(**fields: Any) -> str:
        return " ".join(
            f"{key}={compact_text(value)}"
            for key, value in fields.items()
            if value is not None
        )

    def start_flow(self, name: str, **fields: Any) -> None:
        self._write(f"========== {name} ==========")
        if fields:
            self._write(self._fields(**fields))

    def status(self, **fields: Any) -> None:
        self._write(self._fields(**fields))

    def round(
        self,
        *,
        worker: int,
        question_id: str,
        round_num: int,
        decision: str,
        memory_type: Optional[str] = None,
        search_query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.start_flow(f"Worker {worker} 第 {round_num} 轮")
        self.status(
            worker=worker,
            question_id=question_id,
            round=round_num,
            action=decision,
            memory=memory_type,
            top_k=top_k,
            query=search_query,
        )

    def answer(
        self,
        *,
        worker: int,
        question_id: str,
        answer: str,
        rounds: int,
        total_tokens: int,
        elapsed_seconds: float,
    ) -> None:
        self.start_flow(f"Worker {worker} 答案生成")
        self.status(
            worker=worker,
            question_id=question_id,
            answer=answer,
            rounds=rounds,
            tokens=total_tokens,
            elapsed=f"{elapsed_seconds:.2f}s",
        )

from reporting import build_report_rows, write_report

from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory, QAResult, transform_timestamp


def load_json(file_path: str) -> Any:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def normalize(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().rstrip(".,)")


def extract_choice_letter(text: str) -> Optional[str]:
    """Extract a final multiple-choice letter without mistaking reasoning prose for one."""
    marked_answers = re.findall(
        r"\b(?:final\s+)?answer\s*(?:is\s*)?[:\-]?\s*\(?([A-Za-z])\b",
        text,
        flags=re.IGNORECASE,
    )
    if marked_answers:
        return marked_answers[-1].upper()

    match = re.match(r"\s*\(?([A-Za-z])(?:[\.\)]|\s*$)", text)
    return match.group(1).upper() if match else None


def evaluate_prediction(prediction: str, gold_letter: str, choices: Dict[str, str]) -> bool:
    """
    Evaluate if prediction matches the gold answer.
    
    Args:
        prediction: Model's prediction
        gold_letter: Correct answer letter (e.g., 'A', 'B', 'C', 'D')
        choices: Dict of answer choices
        
    Returns:
        True if prediction is correct
    """
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices[gold_letter])

    if pred_norm == gold_candidate:
        return True

    pred_letter = extract_choice_letter(prediction)
    if pred_letter == gold_letter:
        return True

    full_patterns = [
        normalize(f"{gold_letter}. {choices[gold_letter]}"),
        normalize(f"({gold_letter}) {choices[gold_letter]}")
    ]
    if pred_norm in full_patterns:
        return True

    return False

def usage_total(*models: LLMModel) -> int:
    total = 0
    for model in models:
        snapshot = getattr(model.model, "usage_snapshot", lambda: {})()
        total += int(snapshot.get("total_tokens", 0))
    return total



def find_30s_segment(target_timestamp: int, segments_30s: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Find the 30s segment that contains the target timestamp.
    
    Args:
        target_timestamp: Target timestamp as integer (format: day + time.zfill(8))
        segments_30s: List of 30s segments
    
    Returns:
        Tuple of (start_time, end_time) for the matching segment, or (0, 0) if not found
    """
    for segment in segments_30s:
        date = segment.get('date', '')
        start_time_raw = segment.get('start_time', 0)
        end_time_raw = segment.get('end_time', 0)
        
        day = date.replace('DAY', '').replace('Day', '') if isinstance(date, str) else str(date)
        
        # Format times
        if isinstance(start_time_raw, str):
            start_time = int(day + start_time_raw.zfill(8))
        elif isinstance(start_time_raw, int):
            start_time = int(day + str(start_time_raw).zfill(8))
        else:
            continue
        
        if isinstance(end_time_raw, str):
            end_time = int(day + end_time_raw.zfill(8))
        elif isinstance(end_time_raw, int):
            end_time = int(day + str(end_time_raw).zfill(8))
        else:
            continue
        
        # Check if target timestamp falls within this segment
        if start_time <= target_timestamp <= end_time:
            return (start_time, end_time)
    
    return (0, 0)


def parse_target_time(row: Dict[str, Any], segments_30s: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """
    Parse target time from row data.
    
    Args:
        row: QA row data
        segments_30s: List of 30s segments for finding time ranges
        
    Returns:
        List of (start_time, end_time) tuples
    """
    target_time_list = []
    
    if "time" in row['target_time'] and row['target_time']["time"]:
        time_str = row['target_time']["time"]
        time_str_upper = time_str.upper()
        
        if "DAY" in time_str_upper:
            # Parse range format: "11153417DAY1_11181201"
            parts = re.split(r'DAY|Day', time_str, maxsplit=1)
            if len(parts) == 2:
                start_time_str = parts[0]
                day_and_end = parts[1].split("_")
                if len(day_and_end) == 2:
                    end_day = day_and_end[0]
                    end_time_str = day_and_end[1]
                    start_day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
                    
                    start_time = int(start_day + start_time_str.zfill(8))
                    end_time = int(end_day + end_time_str.zfill(8))
                    target_time_list.append((start_time, end_time))
        else:
            # Single timestamp - find its 30s segment
            day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)
    
    elif "time_list" in row['target_time'] and row['target_time']["time_list"]:
        # Multiple timestamps
        day = row['target_time']["date"].replace('DAY', '').replace('Day', '')
        for time_str in row['target_time']["time_list"]:
            target_timestamp = int(day + time_str.zfill(8))
            segment = find_30s_segment(target_timestamp, segments_30s)
            if segment != (0, 0):
                target_time_list.append(segment)
    
    return target_time_list


def select_questions(rows: List[Dict[str, Any]], manifest_path: str | Path) -> List[Dict[str, Any]]:
    """Return the manifest-selected rows in manifest order."""
    manifest = load_json(str(manifest_path))
    if isinstance(manifest, dict):
        manifest = manifest.get("question_ids", manifest.get("sample_ids"))
    if not isinstance(manifest, list):
        raise ValueError("EgoLife question manifest must be a JSON list of IDs")
    question_ids = [str(question_id) for question_id in manifest]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("EgoLife question manifest contains duplicate IDs")
    rows_by_id = {str(row["ID"]): row for row in rows}
    unknown = sorted(set(question_ids) - rows_by_id.keys())
    if unknown:
        raise ValueError(f"Unknown EgoLife question IDs: {unknown}")
    return [rows_by_id[question_id] for question_id in question_ids]


def select_evaluation_rows(
    rows: List[Dict[str, Any]], manifest_path: Optional[str | Path], limit: Optional[int]
) -> List[Dict[str, Any]]:
    """Select an explicit split before applying an optional smoke-test limit."""
    selected_rows = select_questions(rows, manifest_path) if manifest_path else rows
    return selected_rows[:limit] if limit is not None else selected_rows


def evaluate_in_parallel(
    items: List[Any],
    *,
    workers: int,
    evaluate: Callable[[Any], Any],
    on_complete: Optional[Callable[[int, Any], None]] = None,
) -> List[Any]:
    """Evaluate items concurrently while returning results in input order."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        results = []
        for index, item in enumerate(items):
            result = evaluate(item)
            results.append(result)
            if on_complete is not None:
                on_complete(index, result)
        return results

    results: List[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate, item): index
            for index, item in enumerate(items)
        }
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            results[index] = result
            if on_complete is not None:
                on_complete(index, result)
    return results


class SynchronizedEmbeddingModel:
    """Serialize GPU embedding calls while workers issue LLM requests concurrently."""

    def __init__(self, model: EmbeddingModel) -> None:
        self._model = model
        self._lock = Lock()

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

    def encode_image(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.encode_image(*args, **kwargs)

    def encode_video(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.encode_video(*args, **kwargs)


def _query_time(row: Dict[str, Any]) -> int:
    return int(row["query_time"]["date"][-1] + row["query_time"]["time"].zfill(8))


def evaluate_parallel_egolife(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    embedding_model: EmbeddingModel,
    episodic_caption_files: Dict[str, str],
    semantic_results: Dict[str, Any],
    visual_path: str,
    episodic_captions_30sec: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run time-ordered questions with one mutable WorldMemory per worker thread."""
    cache_root = args.cache_dir or os.path.join(args.output_dir, ".cache", f"egolife_{args.subject}")
    os.makedirs(cache_root, exist_ok=True)
    synchronized_embedding_model = SynchronizedEmbeddingModel(embedding_model)
    workflow = WorkflowProgressLogger()
    completed_count = 0
    workflow.start_flow("评测开始")
    workflow.status(
        questions=len(rows), workers=args.workers, max_rounds=args.max_rounds,
        episodic_top_k=args.episodic_top_k, semantic_top_k=args.semantic_top_k,
        visual_top_k=args.visual_top_k,
    )
    workflow.start_flow("Embedding 初始化")
    workflow.status(device=os.environ.get("WORLDMM_EMBEDDING_DEVICE", "cuda:0"), model="text")

    worker_state = local()
    worker_lock = Lock()
    synchronized_embedding_model.load_model("text")
    workflow.status(model="text", status="ready")

    worker_ids = count()
    worker_memories: List[WorldMemory] = []

    def current_worker() -> Any:
        if hasattr(worker_state, "memory"):
            return worker_state
        with worker_lock:
            worker_id = next(worker_ids)
        retriever_model = LLMModel(model_name=args.retriever_model)
        respond_model = LLMModel(model_name=args.respond_model)
        workflow.start_flow(f"Worker {worker_id} 初始化")
        workflow.status(
            worker=worker_id, cache=os.path.join(cache_root, f"worker-{worker_id}"),
            max_rounds=args.max_rounds,
        )
        memory = WorldMemory(
            embedding_model=synchronized_embedding_model,
            retriever_llm_model=retriever_model,
            respond_llm_model=respond_model,
            prompt_template_manager=PromptTemplateManager(),
            episodic_cache_root=os.path.join(cache_root, f"worker-{worker_id}"),
            max_rounds=args.max_rounds,
            max_errors=args.max_errors,
        )
        memory.set_retrieval_top_k(
            episodic=args.episodic_top_k,
            semantic=args.semantic_top_k,
            visual=args.visual_top_k,
        )
        memory.load_episodic_captions(caption_files=episodic_caption_files)
        memory.load_semantic_triples(data=semantic_results)
        memory.load_visual_clips(
            embeddings_path=visual_path,
            clips_data=episodic_captions_30sec,
        )
        worker_state.memory = memory
        worker_state.worker_id = worker_id
        worker_state.retriever_model = retriever_model
        worker_state.respond_model = respond_model
        with worker_lock:
            worker_memories.append(memory)
        return worker_state

    def evaluate_row(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        position, row = item
        state = current_worker()
        choices = {
            label: row[key]
            for key, label in [("choice_a", "A"), ("choice_b", "B"), ("choice_c", "C"), ("choice_d", "D")]
            if row.get(key)
        }
        query_time = _query_time(row)
        qa_result: Optional[QAResult] = None
        workflow.start_flow(f"Worker {state.worker_id} 题目开始")
        workflow.status(
            worker=state.worker_id, run=f"{position + 1}/{len(rows)}",
            question_id=row["ID"], query_time=transform_timestamp(str(query_time)),
            question=compact_text(row["question"], limit=120),
            choices=" | ".join(f"{label}:{compact_text(value, limit=48)}" for label, value in choices.items()),
        )

        def on_memory_progress(event: str, details: Dict[str, Any]) -> None:
            if event == "index_start":
                workflow.start_flow(f"Worker {state.worker_id} 记忆索引")
                workflow.status(
                    worker=state.worker_id, question_id=row["ID"],
                    until_time=transform_timestamp(str(details["until_time"])),
                )
            elif event == "episodic_index_start":
                workflow.start_flow(f"Worker {state.worker_id} {details['granularity']} 索引")
                workflow.status(
                    worker=state.worker_id, question_id=row["ID"],
                    granularity=details["granularity"], captions=details["total"],
                )
            elif event == "openie_progress":
                if details["completed"] == 0:
                    stage_name = "NER" if details["stage"] == "ner" else "三元组抽取"
                    workflow.start_flow(f"Worker {state.worker_id} {details['granularity']} {stage_name}")
                workflow.status(
                    worker=state.worker_id, question_id=row["ID"],
                    granularity=details["granularity"], stage=details["stage"],
                    progress=render_progress_bar(details["completed"], details["total"]),
                )
            elif event == "episodic_index_complete":
                workflow.status(
                    worker=state.worker_id, question_id=row["ID"],
                    granularity=details["granularity"], index="complete",
                )
            elif event == "index_complete":
                workflow.status(worker=state.worker_id, question_id=row["ID"], index="complete")
            elif event == "round":
                memory_type = details.get("memory_type")
                top_k = {
                    "episodic": args.episodic_top_k,
                    "semantic": args.semantic_top_k,
                    "visual": args.visual_top_k,
                }.get(memory_type)
                workflow.round(
                    worker=state.worker_id, question_id=row["ID"],
                    round_num=details["round_num"], decision=details["decision"],
                    memory_type=memory_type, search_query=details.get("search_query"), top_k=top_k,
                )
            elif event == "answer_generation":
                workflow.start_flow(f"Worker {state.worker_id} 生成答案")
                workflow.status(worker=state.worker_id, question_id=row["ID"], rounds=details["round_num"])

        started_at = time.perf_counter()
        tokens_before = usage_total(state.retriever_model, state.respond_model)
        try:
            qa_result = state.memory.answer(
                query=row["question"],
                choices=choices,
                until_time=query_time,
                progress_callback=on_memory_progress,
            )
            response = qa_result.answer
        except Exception as error:
            workflow.start_flow(f"Worker {state.worker_id} 题目失败")
            workflow.status(worker=state.worker_id, question_id=row["ID"], error=error)
            response = "Error"
        elapsed_seconds = time.perf_counter() - started_at
        total_tokens = usage_total(state.retriever_model, state.respond_model) - tokens_before
        result = {
            "ID": row["ID"],
            "type": row["type"],
            "question": row["question"],
            "choices": choices,
            "answer": row["answer"],
            "response": response,
            "round_history": qa_result.round_history if qa_result else [],
            "num_rounds": qa_result.num_rounds if qa_result else 0,
            "completed": qa_result is not None,
            "total_tokens": total_tokens,
            "elapsed_seconds": elapsed_seconds,
            "evaluate": evaluate_prediction(response, row["answer"], choices),
            "query_time": query_time,
            "target_time": parse_target_time(row, episodic_captions_30sec),
        }
        workflow.answer(
            worker=state.worker_id, question_id=row["ID"], answer=response,
            rounds=result["num_rounds"], total_tokens=total_tokens, elapsed_seconds=elapsed_seconds,
        )
        return position, result

    chronological_items = sorted(enumerate(rows), key=lambda item: (_query_time(item[1]), item[0]))

    def update_progress(_: int, completed: Tuple[int, Dict[str, Any]]) -> None:
        nonlocal completed_count
        completed_count += 1
        workflow.start_flow("评测进度")
        workflow.status(
            completed=f"{completed_count}/{len(rows)}", question_id=completed[1]["ID"],
            correct=completed[1]["evaluate"],
        )

    try:
        completed_rows = evaluate_in_parallel(
            chronological_items,
            workers=args.workers,
            evaluate=evaluate_row,
            on_complete=update_progress,
        )
    finally:
        for memory in worker_memories:
            memory.cleanup()

    results: List[Optional[Dict[str, Any]]] = [None] * len(rows)
    for position, result in completed_rows:
        results[position] = result
    return [result for result in results if result is not None]


def write_parallel_outputs(args: argparse.Namespace, results: List[Dict[str, Any]]) -> None:
    output_path = os.path.join(
        args.output_dir,
        f"{args.retriever_model.replace('-', '_')}_{args.respond_model.replace('-', '_')}",
        f"egolife_eval_{args.subject}.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as output_file:
        json.dump(results, output_file, indent=4)
    report_dir = Path(output_path).parent
    report_paths = write_report(
        build_report_rows(results, agent=args.agent, mode=args.mode, model=args.respond_model),
        report_dir,
        title="EgoLifeQA Run Report",
    )
    correct = sum(bool(result["evaluate"]) for result in results)
    workflow = WorkflowProgressLogger()
    workflow.start_flow("评测完成")
    workflow.status(
        completed=f"{sum(bool(result['completed']) for result in results)}/{len(results)}",
        correct=correct, accuracy=f"{correct / len(results):.4f}" if results else "n/a",
        results=output_path, report=report_paths["html"],
    )

def main():
    parser = argparse.ArgumentParser(description="EgoLifeQA Evaluation with WorldMM")
    parser.add_argument("--subject", type=str, default="A1_JAKE", help="Subject ID")
    parser.add_argument("--workers", type=int, default=1, help="Question-level worker count; use 1 for serial evaluation")
    parser.add_argument("--question-ids-file", type=str, default=None, help="JSON manifest of EgoLife question IDs")
    parser.add_argument("--cache-dir", type=str, default=None, help="External cache root for per-worker episodic indexes")
    parser.add_argument("--retriever-model", type=str, default="Qwen3.5-4B", help="LMDeploy model for retrieval")
    parser.add_argument("--respond-model", type=str, default="Qwen3.5-4B", help="LMDeploy model for reasoning and answers")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum retrieval rounds")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum errors before forcing answer")
    parser.add_argument("--episodic-top-k", type=int, default=3, help="Top-k for episodic retrieval")
    parser.add_argument("--semantic-top-k", type=int, default=10, help="Top-k for semantic retrieval")
    parser.add_argument("--visual-top-k", type=int, default=3, help="Top-k for visual retrieval")
    parser.add_argument("--output-dir", type=str, default="/workspace/worldmm/eval", help="External artifact directory")
    parser.add_argument("--data-dir", type=str, default="data/EgoLife", help="Data directory")
    parser.add_argument("--caption-dir", type=str, default=None, help="Extracted EgoLifeCap root")
    parser.add_argument("--metadata-dir", type=str, default=None, help="Existing three-memory metadata root")
    parser.add_argument("--video-root", type=str, default=None, help="Mounted root for relative video paths")
    parser.add_argument("--mode", choices=["smoke", "test"], default="test")
    parser.add_argument("--agent", default="WorldMM")
    parser.add_argument("--limit", type=int, default=None, help="Maximum questions; use only for smoke")
    args = parser.parse_args()
    configure_console_logging()


    # Initialize models
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    logger.info("Initializing models...")
    embedding_model = EmbeddingModel()
    retriever_llm_model = LLMModel(
        model_name=args.retriever_model,
    )
    respond_llm_model = LLMModel(
        model_name=args.respond_model,
    )
    prompt_template_manager = PromptTemplateManager()

    # Initialize WorldMemory
    logger.info("Initializing WorldMemory...")
    world_memory = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm_model,
        respond_llm_model=respond_llm_model,
        prompt_template_manager=prompt_template_manager,
        max_rounds=args.max_rounds,
        max_errors=args.max_errors,
    )
    
    # Set retrieval top-k
    world_memory.set_retrieval_top_k(
        episodic=args.episodic_top_k,
        semantic=args.semantic_top_k,
        visual=args.visual_top_k,
    )

    # Load data
    logger.info("Loading data...")
    subject = args.subject
    data_dir = args.data_dir
    
    eval_data_path = os.path.join(data_dir, f"EgoLifeQA/EgoLifeQA_{subject}.json")
    eval_data = load_json(eval_data_path)
    eval_data = select_evaluation_rows(eval_data, args.question_ids_file, args.limit)
    
    # Load episodic captions for all granularities (multiscale memory)
    caption_root = args.caption_dir or data_dir
    episodic_caption_dir = os.path.join(caption_root, f"EgoLifeCap/{subject}")
    granularities = ["30sec", "3min", "10min", "1h"]
    episodic_caption_files = {
        g: os.path.join(episodic_caption_dir, f"{subject}_{g}.json")
        for g in granularities
    }
    # Load 30sec captions separately for target time parsing
    episodic_captions_30sec = load_json(episodic_caption_files["30sec"])
    
    # Load semantic results
    metadata_dir = args.metadata_dir or "output/metadata"
    semantic_candidates = glob.glob(os.path.join(metadata_dir, f"semantic_memory/{subject}/semantic_consolidation_results_*.json"))
    if len(semantic_candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one semantic consolidation file, found: {semantic_candidates}")
    semantic_path = semantic_candidates[0]
    semantic_results = load_json(semantic_path)
    
    # Load visual embeddings
    visual_path = os.path.join(metadata_dir, f"visual_memory/{subject}/visual_embeddings.pkl")

    if args.video_root:
        os.environ["WORLDMM_VIDEO_ROOT"] = args.video_root

    if args.workers > 1:
        logger.info("Starting %s question(s) with %s independent workers", len(eval_data), args.workers)
        results = evaluate_parallel_egolife(
            eval_data, args, embedding_model, episodic_caption_files,
            semantic_results, visual_path, episodic_captions_30sec,
        )
        write_parallel_outputs(args, results)
        return
    
    # Load data into WorldMemory
    logger.info("Loading data into WorldMemory...")
    
    # Load episodic captions for all granularities
    world_memory.load_episodic_captions(caption_files=episodic_caption_files)
    
    # Load semantic triples
    world_memory.load_semantic_triples(data=semantic_results)
    
    # Load visual embeddings
    world_memory.load_visual_clips(embeddings_path=visual_path, clips_data=episodic_captions_30sec)

    # Evaluation loop
    logger.info(f"Starting evaluation on {len(eval_data)} samples...")
    results = []
    evaluate_true = 0

    for row in tqdm(eval_data):
        ID = row['ID']
        query_type = row['type']
        question = row['question']
        answer = row['answer']

        # Parse choices
        choices = {}
        for key, label in [('choice_a', 'A'), ('choice_b', 'B'), ('choice_c', 'C'), ('choice_d', 'D')]:
            if key in row and row[key]:
                choices[label] = row[key]

        # Parse query time
        query_time = int(row['query_time']["date"][-1] + row['query_time']["time"].zfill(8))
        
        # Parse target time (use 30sec captions for segment lookup)
        target_time_list = parse_target_time(row, episodic_captions_30sec)

        logger.info(f"Processing ID {ID}: {question[:50]}...")

        qa_result: Optional[QAResult] = None
        started_at = time.perf_counter()
        tokens_before = usage_total(retriever_llm_model, respond_llm_model)
        try:            
            # Answer the question
            qa_result = world_memory.answer(
                query=question,
                choices=choices,
                until_time=query_time,
            )
            
            response = qa_result.answer
            
        except Exception as e:
            logger.error(f"Error processing ID {ID}: {e}")
            response = "Error"

        elapsed_seconds = time.perf_counter() - started_at
        total_tokens = usage_total(retriever_llm_model, respond_llm_model) - tokens_before
        # Evaluate
        evaluate = evaluate_prediction(response, answer, choices)
        evaluate_true += int(evaluate)

        # Build result entry
        result_entry = {
            "ID": ID,
            "type": query_type,
            "question": question,
            "choices": choices,
            "answer": answer,
            "response": response,
            "round_history": qa_result.round_history if qa_result else [],
            "num_rounds": qa_result.num_rounds if qa_result else 0,
            "completed": qa_result is not None,
            "total_tokens": total_tokens,
            "elapsed_seconds": elapsed_seconds,
            "evaluate": evaluate,
            "query_time": query_time,
            # "query_time_str": transform_timestamp(str(query_time)),
            "target_time": target_time_list,
            # "target_time_str": [
            #     (transform_timestamp(str(start)), transform_timestamp(str(end))) 
            #     for start, end in target_time_list
            # ],
        }
        results.append(result_entry)

        logger.info(
            f"ID {ID} Answer: {response}, Gold: {answer}, Correct: {evaluate} "
            f"// Accuracy: {evaluate_true}/{len(results)} = {evaluate_true/len(results):.4f}"
        )

    # Save results
    output_path = os.path.join(
        args.output_dir, 
        f"{args.retriever_model.replace('-', '_')}_{args.respond_model.replace('-', '_')}",
        f"egolife_eval_{subject}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    report_dir = Path(output_path).parent
    report_paths = write_report(
        build_report_rows(results, agent=args.agent, mode=args.mode, model=args.respond_model),
        report_dir, title="EgoLifeQA Run Report",
    )
    logger.info(f"Report saved to: {report_paths['html']}")
    
    # Print summary
    final_accuracy = evaluate_true / len(results) if results else 0
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation Complete")
    logger.info(f"Subject: {subject}")
    logger.info(f"Total: {len(results)}")
    logger.info(f"Correct: {evaluate_true}")
    logger.info(f"Accuracy: {final_accuracy:.4f}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*50}")

    # Cleanup
    world_memory.cleanup()


if __name__ == "__main__":
    main()
