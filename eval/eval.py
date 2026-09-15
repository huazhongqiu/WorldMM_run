#!/usr/bin/env python3
"""
Evaluation script using WorldMM unified memory system.
Processes videos one-by-one, answering all queries per video.
"""

import os

os.environ.setdefault("TQDM_DISABLE", "1")
import time
import json
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_console_logging() -> None:
    """Keep the benchmark stream focused on runner-level progress and failures."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)
    for logger_name in (
        "hipporag",
        "sentence_transformers",
        "transformers",
        "worldmm.memory",
        "worldmm.embedding",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


configure_console_logging()

from worldmm.embedding import EmbeddingModel
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory, QAResult
from worldmm.run_log import (
    agent_round,
    answer_status as log_answer_status,
    emit,
    progress,
    section,
)


def load_json(file_path: str) -> Any:
    with open(file_path, 'r') as f:
        return json.load(f)


def normalize(text: str) -> str:
    return text.lower().strip().rstrip(".,)")


def extract_choice_letter(text: str) -> Optional[str]:
    match = re.match(r"\(?([A-Za-z])[\.\)]?\s*", text.strip())
    return match.group(1).upper() if match else None


def evaluate_prediction(prediction: str, gold_letter: str, choices: Dict[str, str]) -> bool:
    pred_norm = normalize(prediction)
    gold_candidate = normalize(choices[gold_letter])
    if pred_norm == gold_candidate:
        return True
    pred_letter = extract_choice_letter(prediction)
    if pred_letter == gold_letter:
        return True
    full_patterns = [
        normalize(f"{gold_letter}. {choices[gold_letter]}"),
        normalize(f"({gold_letter}) {choices[gold_letter]}"),
    ]
    if pred_norm in full_patterns:
        return True
    return False


VIDEOMME_GRANULARITIES = ["10sec", "30sec", "3min", "10min"]
QUERY_TIME = int("1" + "23595999")  # DAY1 end-of-video: index everything


def load_latest_results(path: str) -> Dict[str, Dict[str, Any]]:
    """Load the last valid JSONL record for each question ID."""
    latest: Dict[str, Dict[str, Any]] = {}
    if not path or not os.path.exists(path):
        return latest
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                latest[str(record["ID"])] = record
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return latest


def append_result(path: str, result: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")


def answer_status(response: Any) -> str:
    normalized = str(response or "").strip()
    if not normalized:
        return "empty_response"
    if normalized == "Unable to generate answer":
        return "generation_error"
    return "success"


def is_successful_result(result: Dict[str, Any]) -> bool:
    return result.get("status") == "success" and answer_status(result.get("response")) == "success"


def pending_rows(rows: List[Dict[str, Any]], latest: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if not is_successful_result(latest.get(str(row["ID"]), {}))]


def merge_results(rows: List[Dict[str, Any]], latest: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for row in rows:
        merged.append(latest.get(str(row["ID"])) or _error_result(row, "Missing result", status="missing"))
    return merged


def completion_counts(results: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"success": 0, "error": 0, "missing": 0, "total": len(results)}
    for result in results:
        status = result.get("status", "missing")
        if status == "missing":
            counts["missing"] += 1
        elif is_successful_result(result):
            counts["success"] += 1
        else:
            counts["error"] += 1
    return counts


def build_choices(row: Dict[str, Any]) -> Dict[str, str]:
    """Build choices dict from a row."""
    choices = {}
    for key, label in [('choice_a', 'A'), ('choice_b', 'B'), ('choice_c', 'C'), ('choice_d', 'D')]:
        if key in row and row[key]:
            choices[label] = row[key]
    return choices


def get_episodic_cache_root(cache_dir: str, video_id: str) -> str:
    """Return the per-video episodic HippoRAG cache root."""
    return os.path.join(cache_dir, str(video_id), "episodic_memory")


def load_required_episodic_openie(memory: WorldMemory, path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required persisted OpenIE file is missing: {path}")
    memory.load_episodic_openie(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Video-MME Evaluation with WorldMM")
    parser.add_argument("--eval-json", type=str, default="data/Video-MME/videomme/test.json", help="Path to Video-MME test JSON")
    parser.add_argument("--caption-dir", type=str, default="data/Video-MME/caption", help="Root caption directory with {videoID}/ subdirs")
    parser.add_argument("--metadata-dir", type=str, default="output/metadata/videomme", help="Root metadata directory")
    parser.add_argument("--retriever-model", type=str, default="Qwen3.5-4B", help="LMDeploy model for retrieval (NER, OpenIE)")
    parser.add_argument("--respond-model", type=str, default="Qwen3.5-4B", help="LMDeploy model for reasoning and answering")
    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum retrieval rounds")
    parser.add_argument("--max-errors", type=int, default=5, help="Maximum errors before forcing answer")
    parser.add_argument("--episodic-top-k", type=int, default=3)
    parser.add_argument("--semantic-top-k", type=int, default=10)
    parser.add_argument("--visual-top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for results")
    parser.add_argument("--duration", type=str, default=None, choices=["short", "medium", "long"], help="Optionally filter by video duration")
    parser.add_argument("--episodic-cache-dir", type=str, default=".cache/videomme", help="Root cache directory for per-video episodic HippoRAG state.")
    parser.add_argument("--eval-name", type=str, default="videomme", help="Dataset name used in the output filename.")
    parser.add_argument("--records-jsonl", type=str, default=None, help="Append-only per-question checkpoint file.")
    parser.add_argument("--require-complete", action="store_true", help="Exit non-zero if any question is error/missing.")
    args = parser.parse_args()

    logger.info("Initializing models...")
    emit(section("MODEL LOADING"))
    embedding_model = EmbeddingModel()
    retriever_llm = LLMModel(model_name=args.retriever_model)
    respond_llm = LLMModel(model_name=args.respond_model, fps=1)
    prompt_template_manager = PromptTemplateManager()

    world_memory = WorldMemory(
        embedding_model=embedding_model,
        retriever_llm_model=retriever_llm,
        respond_llm_model=respond_llm,
        prompt_template_manager=prompt_template_manager,
        episodic_granularities=VIDEOMME_GRANULARITIES,
        qa_template_name="qa",
        max_rounds=args.max_rounds,
        max_errors=args.max_errors,
    )
    world_memory.set_retrieval_top_k(
        episodic=args.episodic_top_k,
        semantic=args.semantic_top_k,
        visual=args.visual_top_k,
    )

    logger.info("Loading evaluation data...")
    eval_data = load_json(args.eval_json)
    if args.duration:
        eval_data = [r for r in eval_data if r.get("duration") == args.duration]
        logger.info(f"Filtered to {len(eval_data)} rows with duration={args.duration}")

    queries_by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eval_data:
        queries_by_video[row["video_id"]].append(row)

    logger.info(f"Total queries: {len(eval_data)}, across {len(queries_by_video)} videos")

    emit(section("INFERENCE"))

    results: List[Dict[str, Any]] = []
    latest = load_latest_results(args.records_jsonl) if args.records_jsonl else {}
    evaluate_true = 0
    model_name = args.retriever_model

    video_progress = tqdm(sorted(queries_by_video.items()), desc="Videos", unit="video")
    video_total = len(queries_by_video)
    for video_index, (video_id, video_queries) in enumerate(video_progress, start=1):
        if args.records_jsonl:
            video_queries = pending_rows(video_queries, latest)
            if not video_queries:
                continue
        video_progress.set_postfix(vid=video_id, q=len(video_queries))
        emit(progress("VIDEO", video_index, video_total, f"id={video_id} | questions={len(video_queries)}"))

        world_memory.reset()
        world_memory.episodic_memory.save_dir_root = get_episodic_cache_root( args.episodic_cache_dir, video_id)

        caption_dir = os.path.join(args.caption_dir, str(video_id))
        caption_files = {}
        for g in VIDEOMME_GRANULARITIES:
            path = os.path.join(caption_dir, f"{g}.json")
            if os.path.exists(path):
                caption_files[g] = path
            else:
                logger.warning(f"Missing caption file: {path}")

        if not caption_files:
            logger.error(f"No caption files for video {video_id}, skipping")
            for row in video_queries:
                result = _error_result(row, "No caption files")
                results.append(result)
                if args.records_jsonl:
                    append_result(args.records_jsonl, result)
                    latest[str(row["ID"])] = result
            continue

        world_memory.load_episodic_captions(caption_files=caption_files)

        episodic_openie_file = os.path.join(
            args.metadata_dir, "episodic_memory", str(video_id),
            f"openie_results_{model_name}.json",
        )
        try:
            load_required_episodic_openie(world_memory, episodic_openie_file)
        except Exception as e:
            logger.error(f"Persisted OpenIE invalid for video {video_id}: {e}")
            for row in video_queries:
                result = _error_result(row, f"OpenIE error: {e}", status="episodic_openie_error")
                results.append(result)
                if args.records_jsonl:
                    append_result(args.records_jsonl, result)
                    latest[str(row["ID"])] = result
            continue

        semantic_file = os.path.join(
            args.metadata_dir, "semantic_memory", str(video_id),
            f"semantic_consolidation_results_{model_name}.json",
        )
        if os.path.exists(semantic_file):
            world_memory.load_semantic_triples(file_path=semantic_file)

        visual_pkl = os.path.join(
            args.metadata_dir, "visual_memory", str(video_id),
            "visual_embeddings.pkl",
        )
        base_caption_file = caption_files.get("10sec")
        if os.path.exists(visual_pkl) and base_caption_file:
            clips_data = load_json(base_caption_file)
            world_memory.load_visual_clips(embeddings_path=visual_pkl, clips_data=clips_data)

        index_started_at = time.perf_counter()
        index_label = f"[VIDEO {video_index:03d}]"
        logger.info("%s building memory index...", index_label)
        try:
            world_memory.index(QUERY_TIME)
        except Exception as e:
            logger.error("%s index=FAILED after %.1fs: %s", index_label, time.perf_counter() - index_started_at, e)
            logger.error(f"Indexing failed for video {video_id}: {e}")
            for row in video_queries:
                result = _error_result(row, f"Index error: {e}")
                results.append(result)
                if args.records_jsonl:
                    append_result(args.records_jsonl, result)
                    latest[str(row["ID"])] = result
            continue
        else:
            logger.info("%s index done in %.1fs", index_label, time.perf_counter() - index_started_at)

        for row in video_queries:
            choices = build_choices(row)
            question = row["question"]
            answer = row["answer"]

            qa_result: Optional[QAResult] = None
            started_at = time.perf_counter()

            def round_callback(event: str, details: Dict[str, Any]) -> None:
                if event == "round":
                    emit(agent_round(
                        details["round_num"],
                        args.max_rounds,
                        details["decision"],
                        memory_type=details.get("memory_type"),
                        search_query=details.get("search_query"),
                    ))
                elif event == "answer_generation":
                    emit(agent_round(details["round_num"], args.max_rounds, "answer"))

            try:
                qa_result = world_memory.answer(
                    query=question,
                    choices=choices,
                    until_time=QUERY_TIME,
                    progress_callback=round_callback,
                )
                response = qa_result.answer
                status = answer_status(response)
                if status != "success":
                    raise ValueError(status.replace("_", " "))
            except Exception as e:
                logger.error(f"Error answering {row['ID']}: {e}")
                response = "Error"
                status = "error"
            elapsed_seconds = time.perf_counter() - started_at

            correct = evaluate_prediction(response, answer, choices)
            evaluate_true += int(correct)
            emit(log_answer_status(
                status.upper(),
                question_id=row["ID"],
                answer=response,
                correct=correct,
                elapsed_seconds=elapsed_seconds,
            ))

            result = {
                "ID": row["ID"],
                "status": status,
                "video_id": video_id,
                "type": row.get("type", ""),
                "duration": row.get("duration", ""),
                "question": question,
                "choices": choices,
                "answer": answer,
                "response": response,
                "round_history": qa_result.round_history if qa_result else [],
                "num_rounds": qa_result.num_rounds if qa_result else 0,
                "evaluate": correct,
            }
            results.append(result)
            if args.records_jsonl:
                append_result(args.records_jsonl, result)
                latest[str(row["ID"])] = result

            logger.info(
                f"{row['ID']} Pred: {response}, Gold: {answer}, "
                f"Correct: {correct} // Acc: {evaluate_true}/{len(results)} "
                f"= {evaluate_true/len(results):.4f}"
            )

    if args.records_jsonl:
        results = merge_results(eval_data, latest)
        evaluate_true = sum(int(bool(row.get("evaluate"))) for row in results)

    duration_tag = f"_{args.duration}" if args.duration else ""
    output_path = os.path.join(
        args.output_dir,
        f"{args.retriever_model.replace('-','_')}_{args.respond_model.replace('-','_')}",
        f"{args.eval_name}_eval{duration_tag}.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    emit(section("EVALUATION"))
    final_accuracy = evaluate_true / len(results) if results else 0
    logger.info(f"\n{'='*50}")
    logger.info("Evaluation Complete")
    logger.info(f"Total: {len(results)}")
    logger.info(f"Correct: {evaluate_true}")
    logger.info(f"Accuracy: {final_accuracy:.4f}")

    _print_per_duration_accuracy(results)

    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*50}")

    world_memory.cleanup()
    counts = completion_counts(results)
    logger.info(
        "Completion: success=%d error=%d missing=%d total=%d",
        counts["success"], counts["error"], counts["missing"], counts["total"],
    )
    if args.require_complete and counts["success"] != counts["total"]:
        return 2
    return 0


def _error_result(row: Dict[str, Any], msg: str, status: str = "error") -> Dict[str, Any]:
    return {
        "ID": row["ID"],
        "status": status,
        "video_id": row.get("video_id", ""),
        "type": row.get("type", ""),
        "duration": row.get("duration", ""),
        "question": row.get("question", ""),
        "choices": build_choices(row),
        "answer": row.get("answer", ""),
        "response": msg,
        "round_history": [],
        "num_rounds": 0,
        "evaluate": False,
    }


def _print_per_duration_accuracy(results: List[Dict[str, Any]]):
    """Print per-duration accuracy breakdown."""
    by_duration: Dict[str, List[bool]] = defaultdict(list)
    for r in results:
        by_duration[r.get("duration", "unknown")].append(r["evaluate"])

    for dur in sorted(by_duration):
        correct = sum(by_duration[dur])
        total = len(by_duration[dur])
        logger.info(f"  {dur}: {correct}/{total} = {correct/total:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
