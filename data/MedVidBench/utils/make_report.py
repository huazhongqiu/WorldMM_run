#!/usr/bin/env python3
"""Render a WorldMM MedVidBench run report.

Reads ``run.json``, ``records.jsonl``, ``evaluation.json`` and
``token_usage.json`` from a run directory and writes ``report/metrics.json``
plus ``report/report.md`` with the official leaderboard metric table (in
leaderboard order) and per-task run statistics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

LEADERBOARD_METRICS = (
    ("CVS_acc", "cvs_acc"),
    ("NAP_acc", "nap_acc"),
    ("NAP_exact_acc (diagnostic)", "nap_exact_acc"),
    ("SA_acc", "sa_acc"),
    ("STG_mIoU", "stg_miou"),
    ("TAG_mIoU@0.3", "tag_miou_03"),
    ("TAG_mIoU@0.5", "tag_miou_05"),
    ("DVC_F1", "dvc_f1"),
    ("DVC_llm", "dvc_llm"),
    ("VS_llm", "vs_llm"),
    ("RC_llm", "rc_llm"),
)
REPORT_TASKS = (
    ("tal", "TAL"),
    ("stg", "STG"),
    ("dvc", "DVC"),
    ("next_action", "Next Action"),
    ("rc", "Region Caption"),
    ("vs", "Video Summary"),
    ("skill_assessment", "Skill Assessment"),
    ("cvs_assessment", "CVS Assessment"),
)
JUDGE_SCALE = {"dvc_llm", "vs_llm", "rc_llm"}


def esc(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def format_metric(name: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    if name in JUDGE_SCALE:
        return f"{value:.2f}"
    return f"{value * 100:.1f}%"


def load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def run_stats_table(records: list[dict], agent: str, mode: str, model: str) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record.get("task") or record.get("qa_type", "unknown"))].append(record)

    lines = [
        "# WorldMM MedVidBench Run Report",
        "",
        f"- Agent: {agent} | Mode: {mode} | Model: {model}",
        "",
        "## Run Statistics",
        "",
        "| Task | Completed | Avg. rounds | Avg. tokens | Avg. time (s) |",
        "| --- | --- | --- | --- | --- |",
    ]

    def row(task: str, group: list[dict]) -> str:
        total = len(group)
        completed = sum(1 for r in group if r.get("status") == "success")
        avg_rounds = sum(r.get("num_rounds", 0) for r in group) / total if total else 0.0
        avg_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in group) / total if total else 0.0
        avg_time = sum(r.get("elapsed_seconds", 0.0) for r in group) / total if total else 0.0
        return (f"| {task} | {completed}/{total} | {avg_rounds:.1f} | "
                f"{avg_tokens:.0f} | {avg_time:.1f} |")

    lines.append(row("Overall", records))
    for task, label in REPORT_TASKS:
        if task in groups:
            lines.append(row(label, groups[task]))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the MedVidBench run report.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-dir", default=None, help="Defaults to <run-dir>/report.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(run_dir / "run.json") or {}
    evaluation = load_json(run_dir / "evaluation.json") or {}
    records_map = latest_records(run_dir / "records.jsonl")
    records = list(records_map.values())

    agent = manifest.get("agent", "WorldMM")
    mode = manifest.get("mode", "test")
    model = manifest.get("model", "")
    metrics = evaluation.get("metrics", {})

    parts = [run_stats_table(records, agent, mode, model), ""]

    parts += [
        "## Leaderboard Metrics",
        "",
        f"Official evaluator: `{evaluation.get('leaderboard_dir', 'n/a')}` | "
        f"LLM judge: `{(evaluation.get('llm') or {}).get('model')}` "
        "(official leaderboards use GPT-4.1; local-judge _llm scores are not directly comparable)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for label, key in LEADERBOARD_METRICS:
        parts.append(f"| {label} | {format_metric(key, metrics.get(key))} |")
    parts += [
        "",
        f"Evaluation status: **{evaluation.get('status', 'not_evaluated')}** "
        f"({len(records)} records, skip_llm_judge={evaluation.get('skip_llm_judge')})",
    ]
    for warning in evaluation.get("warnings", []):
        parts.append(f"- Warning: {warning}")
    for error in evaluation.get("errors", []):
        parts.append(f"- Error: {error}")

    usage = load_json(run_dir / "token_usage.json")
    if usage:
        parts += [
            "",
            "## Token Usage (WorldMM inference, answering only)",
            "",
            f"- input: {usage.get('input_tokens', 0):,} | output: {usage.get('output_tokens', 0):,} "
            f"| total: {usage.get('total_tokens', 0):,}",
        ]

    report_md = "\n".join(parts) + "\n"
    (report_dir / "report.md").write_text(report_md, encoding="utf-8")

    metrics_json = {
        "agent": agent,
        "mode": mode,
        "model": model,
        "records": len(records),
        "leaderboard_metrics": metrics,
        "evaluation_status": evaluation.get("status"),
        "warnings": evaluation.get("warnings", []),
        "errors": evaluation.get("errors", []),
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Report: {report_dir / 'report.md'}")
    print(f"Metrics: {report_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
