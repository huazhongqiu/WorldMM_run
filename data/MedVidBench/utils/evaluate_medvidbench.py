#!/usr/bin/env python3
"""Evaluate a WorldMM MedVidBench run with the official leaderboard scorer.

Adapted from ``videospy/benchmark/medvidbench/evaluate.py``: aligns
``submission.json`` with the local ground truth (the raw
``medvidu_eccv2026_trainval.json`` selected by integer sample key), then runs
the unmodified official ``evaluate_predictions.py`` through
``official_runner.py`` (which retargets its GPT-4.1 judge at the local
Qwen3.5-4B LMDeploy endpoint) and parses the "LEADERBOARD METRICS SUMMARY"
block into ``<run-dir>/evaluation.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_LEADERBOARD_DIR = "/myworkspace/data/MedVidBench/MedVidBench-Leaderboard"
DEFAULT_EVALUATION_MODEL = "gpt-4.1"
CONFIG_PATH = Path(__file__).with_name("config") / "evaluation.yml"
TASKS = (
    "dvc", "tal", "next_action", "stg", "rc", "vs", "skill_assessment", "cvs_assessment",
)
TASK_METRICS = {
    "dvc": ("dvc_f1", "dvc_llm"),
    "tal": ("tag_miou_03", "tag_miou_05"),
    "next_action": ("nap_acc",),
    "stg": ("stg_miou",),
    "rc": ("rc_llm",),
    "vs": ("vs_llm",),
    "skill_assessment": ("sa_acc",),
    "cvs_assessment": ("cvs_acc",),
}
LLM_METRICS = {"dvc_llm", "vs_llm", "rc_llm"}
METRIC_PATTERNS = {
    "cvs_acc": r"^\s*component_balanced_accuracy:\s*([-+]?\d*\.?\d+)",
    "nap_acc": r"^\s*accuracy:\s*([-+]?\d*\.?\d+)",
    "sa_acc": r"^\s*aspect_balanced_accuracy:\s*([-+]?\d*\.?\d+)",
    "stg_miou": r"^\s*mean_iou:\s*([-+]?\d*\.?\d+)",
    "tag_miou_03": r"^\s*mIoU@0\.3:\s*([-+]?\d*\.?\d+)",
    "tag_miou_05": r"^\s*mIoU@0\.5:\s*([-+]?\d*\.?\d+)",
    "dvc_f1": r"^\s*temporal_f1:\s*([-+]?\d*\.?\d+)",
    "dvc_llm": r"^\s*caption_score:\s*([-+]?\d*\.?\d+)",
    "vs_llm": r"VS - Overall Evaluation[\s\S]*?score:\s*([-+]?\d*\.?\d+)",
    "rc_llm": r"RC - Overall Evaluation[\s\S]*?score:\s*([-+]?\d*\.?\d+)",
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a completed WorldMM MedVidBench run with the official code."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ground-truth-path", default=None,
                        help="Override the ground-truth path stored in run.json.")
    parser.add_argument("--leaderboard-dir", default=DEFAULT_LEADERBOARD_DIR)
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--skip-llm-judge", action="store_true",
                        help="Compute deterministic metrics only.")
    parser.add_argument("--config", default=CONFIG_PATH.as_posix())
    return parser.parse_args(argv)


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_llm_config(config_path: str) -> dict:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    llm = (config.get("evaluation") or {}).get("llm") or {}
    return {
        "config_path": str(path),
        "model": llm.get("model") or DEFAULT_EVALUATION_MODEL,
        "api_base": llm.get("api_base"),
        "api_key": llm.get("api_key"),
    }


def latest_records(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8") as handle:
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


def report_task(qa_type: str) -> str:
    if qa_type.startswith("dense_captioning"):
        return "dvc"
    if qa_type.startswith("region_caption"):
        return "rc"
    if qa_type.startswith("video_summary"):
        return "vs"
    return qa_type


def select_ground_truth(ground_truth: list[dict], sample_keys: list[str]) -> list[dict]:
    indices = [int(key) for key in sample_keys]
    invalid = [i for i in indices if i < 0 or i >= len(ground_truth)]
    if invalid:
        raise ValueError(f"Ground-truth indices out of range: {invalid[:10]}")
    return [ground_truth[i] for i in indices]


def validate_alignment(submission: list[dict], ground_truth: list[dict]) -> None:
    if len(submission) != len(ground_truth):
        raise ValueError(
            f"Submission has {len(submission)} samples, but selected ground truth has "
            f"{len(ground_truth)}."
        )
    for index, (prediction, reference) in enumerate(zip(submission, ground_truth)):
        missing = {"id", "qa_type", "prediction"} - prediction.keys()
        if missing:
            raise ValueError(f"Submission sample {index} is missing fields: {sorted(missing)}")
        if reference.get("id") and prediction["id"] != reference["id"]:
            raise ValueError(f"Sample {index} id does not match ground truth.")
        if reference.get("qa_type") and prediction["qa_type"] != reference["qa_type"]:
            raise ValueError(f"Sample {index} qa_type does not match ground truth.")


def conversation_value(sample: dict, roles: set[str]) -> str:
    value = ""
    for message in sample.get("conversations", []):
        if message.get("from") in roles:
            value = message.get("value", "")
    return value.replace("<video>\n", "").replace("<video>", "")


def normalize_action_label(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def compute_nap_exact_accuracy(submission: list[dict], ground_truth: list[dict]) -> float | None:
    comparisons = [
        (
            normalize_action_label(prediction.get("prediction", "")),
            normalize_action_label(conversation_value(reference, {"gpt", "assistant"})),
        )
        for prediction, reference in zip(submission, ground_truth)
        if prediction.get("qa_type") == "next_action"
    ]
    if not comparisons:
        return None
    return sum(a == b for a, b in comparisons) / len(comparisons)


def parse_official_metrics(output: str) -> dict[str, float]:
    marker = re.compile(r"^LEADERBOARD METRICS SUMMARY$", re.MULTILINE)
    end_marker = "END LEADERBOARD METRICS SUMMARY"
    starts = list(marker.finditer(output))
    if not starts:
        return {}
    start = starts[-1].end()
    end = output.find(end_marker, start)
    output = output[start : end if end >= 0 else None]
    metrics = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output, re.MULTILINE)
        if match:
            metrics[name] = float(match.group(1))
    return metrics


def run_official_evaluation(command: list[str], cwd: Path, log_path: Path, env: dict) -> str:
    output = []
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        with subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                output.append(line)
            return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"Official evaluation failed with exit code {return_code}. See {log_path}."
        )
    return "".join(output)


def evaluate_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    leaderboard_dir = Path(args.leaderboard_dir).expanduser().resolve()
    evaluation_script = leaderboard_dir / "evaluation" / "evaluate_predictions.py"
    official_runner = Path(__file__).with_name("official_runner.py").resolve()
    preflight_errors: list[str] = []
    warnings: list[str] = []

    manifest = load_json(run_dir / "run.json")
    if manifest.get("dataset") != "medvidbench":
        raise ValueError(f"Not a MedVidBench run directory: {run_dir}")
    ground_truth_path = (
        Path(args.ground_truth_path).expanduser().resolve()
        if args.ground_truth_path
        else Path(manifest["ground_truth_path"]).expanduser().resolve()
    )

    records = latest_records(run_dir / "records.jsonl")
    all_sample_keys: list[str] = list(manifest["sample_keys"])
    success_keys = [key for key in all_sample_keys if records.get(key, {}).get("status") == "success"]
    if len(success_keys) != len(all_sample_keys):
        skipped = len(all_sample_keys) - len(success_keys)
        warnings.append(
            f"Skipping {skipped} failed samples in evaluation; using "
            f"{len(success_keys)}/{len(all_sample_keys)} completed samples only."
        )
        if not success_keys:
            raise ValueError("No successful samples available for evaluation.")
        manifest = {**manifest, "sample_keys": success_keys}

    submission_all = load_json(run_dir / "submission.json")
    key_to_submission = dict(zip(all_sample_keys, submission_all))
    submission = [key_to_submission[key] for key in manifest["sample_keys"]]
    ground_truth = select_ground_truth(load_json(ground_truth_path), manifest["sample_keys"])
    validate_alignment(submission, ground_truth)

    if not evaluation_script.is_file():
        raise FileNotFoundError(f"Official evaluator not found: {evaluation_script}")

    requested_tasks = list(args.tasks or TASKS)
    # Only expect (and only run) metrics for tasks actually present in the
    # evaluated subset — a smoke run with a single qa_type must not fail on
    # metrics the subset cannot produce.
    present_tasks = {report_task(ref.get("qa_type", "")) for ref in ground_truth}
    evaluated_tasks = [task for task in requested_tasks if task in present_tasks]
    skipped_tasks = [task for task in requested_tasks if task not in present_tasks]
    if skipped_tasks:
        warnings.append(f"Skipping tasks absent from the evaluated subset: {', '.join(skipped_tasks)}")
    if not evaluated_tasks:
        raise ValueError(
            f"None of the requested tasks {requested_tasks} are present in the evaluated subset."
        )
    expected_metrics = [
        metric
        for task in evaluated_tasks
        for metric in TASK_METRICS[task]
        if not (args.skip_llm_judge and metric in LLM_METRICS)
    ]
    llm_tasks = {"dvc", "rc", "vs"} & set(evaluated_tasks)
    llm_enabled = bool(llm_tasks) and not args.skip_llm_judge
    llm_config = load_llm_config(args.config)
    if llm_enabled and not llm_config["api_key"]:
        preflight_errors.append(
            f"Evaluation API key is missing. Set evaluation.llm.api_key in {llm_config['config_path']}."
        )

    log_path = run_dir / "evaluation.log"
    output = ""
    execution_errors = []
    if preflight_errors:
        log_path.write_text("\n".join(preflight_errors) + "\n", encoding="utf-8")
    else:
        environment = os.environ.copy()
        if llm_config["api_key"]:
            environment["OPENAI_API_KEY"] = llm_config["api_key"]
        with tempfile.TemporaryDirectory(prefix="medvidbench-eval-") as directory:
            gt_path = Path(directory) / "ground_truth.json"
            gt_path.write_text(json.dumps(ground_truth, ensure_ascii=False), encoding="utf-8")
            effective_submission_path = run_dir / "submission.json"
            if len(submission_all) != len(submission):
                effective_submission_path = Path(directory) / "submission.json"
                effective_submission_path.write_text(
                    json.dumps(submission, ensure_ascii=False), encoding="utf-8"
                )
            command = [sys.executable, str(official_runner), "--model", llm_config["model"]]
            if llm_config["api_base"]:
                command.extend(["--api-base", llm_config["api_base"]])
            command.extend([
                str(evaluation_script),
                str(effective_submission_path),
                "--ground-truth", str(gt_path),
                "--grouping", "overall",
            ])
            command.extend(["--tasks", *evaluated_tasks])
            if args.skip_llm_judge:
                command.append("--skip-llm-judge")
            try:
                output = run_official_evaluation(command, leaderboard_dir, log_path, environment)
            except RuntimeError as error:
                execution_errors.append(str(error))
                output = log_path.read_text(encoding="utf-8")

    metrics = parse_official_metrics(output)
    if "next_action" in evaluated_tasks:
        nap_exact = compute_nap_exact_accuracy(submission, ground_truth)
        if nap_exact is not None:
            metrics["nap_exact_acc"] = nap_exact
            official_nap = metrics.get("nap_acc")
            if official_nap is not None and abs(official_nap - nap_exact) > 1e-12:
                warnings.append(
                    "Official NAP_acc differs from local exact label matching: "
                    f"{official_nap:.4f} vs {nap_exact:.4f}."
                )
    task_errors = re.findall(r"^Error running ([^ ]+) evaluation: (.+)$", output, re.MULTILINE)
    errors = (
        preflight_errors
        + execution_errors
        + [f"{task}: {message}" for task, message in task_errors]
    )
    llm_calls = []
    for completed, total in re.findall(
        r"LLM Judge completed:\s*(\d+)/(\d+) successful(?: API calls)?", output
    ):
        completed_count, total_count = int(completed), int(total)
        llm_calls.append({
            "successful": completed_count,
            "total": total_count,
            "success_rate": completed_count / total_count if total_count else 0.0,
        })
        if completed_count < total_count:
            warnings.append(f"LLM evaluation completed {completed_count}/{total_count} calls successfully.")
        if total_count and completed_count == 0:
            errors.append("All LLM evaluation calls failed.")
    if llm_enabled and "semantic_similarity" in output.lower():
        for metric_key in LLM_METRICS:
            metrics.pop(metric_key, None)
        errors.append(
            "The official evaluator used semantic-similarity fallback; these "
            "scores are not comparable to Leaderboard LLM metrics."
        )
    missing_metrics = [key for key in expected_metrics if key not in metrics]
    if missing_metrics:
        errors.append("Missing required metrics: " + ", ".join(missing_metrics))
    status = "failed" if errors else "complete"
    evaluation = {
        "status": status,
        "dataset": "medvidbench",
        "ground_truth_path": str(ground_truth_path),
        "leaderboard_dir": str(leaderboard_dir),
        "tasks": evaluated_tasks,
        "requested_tasks": requested_tasks,
        "skip_llm_judge": args.skip_llm_judge,
        "expected_metrics": expected_metrics,
        "missing_metrics": missing_metrics,
        "metrics": metrics,
        "llm": {
            "enabled": llm_enabled,
            "config_path": llm_config["config_path"],
            "model": llm_config["model"] if llm_enabled else None,
            "api_base": llm_config["api_base"] if llm_enabled else None,
            "calls": llm_calls,
        },
        "warnings": warnings,
        "errors": errors,
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Evaluation log: {log_path}")
    print(f"Evaluation metrics: {run_dir / 'evaluation.json'}")
    for warning in warnings:
        print(f"Warning: {warning}")
    if errors:
        for error in errors:
            print(f"Error: {error}")
        return 1
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    return evaluate_run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
