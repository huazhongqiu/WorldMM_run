#!/usr/bin/env python3
"""Render WorldMM LVBench eval output as a videospy-style run report.

Reads the ``{eval_name}_eval.json`` produced by ``eval/eval.py`` and writes,
next to it (or into ``--report-dir``):

- ``metrics.json``      overall + per-question-type accuracy and run statistics
- ``predictions.json``  flat map uid -> normalized choice letter
- ``records.jsonl``     one record per question (videospy record schema)
- ``report.md``         the videospy markdown summary table
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def normalize_letter(response: str, choices: dict[str, str]) -> str:
    text = (response or "").strip()
    match = re.match(r"\(?([A-Da-d])[\.\)]?\s*(?:$|\s|\W)", text)
    if match:
        return match.group(1).upper()
    normalized = text.lower().strip().rstrip(".,)")
    for letter in ("A", "B", "C", "D"):
        if letter in choices and choices[letter].lower().strip().rstrip(".,)") == normalized:
            return letter
    inline = re.match(r"^([A-Da-d])[\.\)]\s*\S", text)
    if inline:
        return inline.group(1).upper()
    return ""


def question_type_list(raw: str) -> list[str]:
    return [t for t in (part.strip() for part in (raw or "").split(";")) if t]


def accuracy_block(correct: int, total: int) -> dict:
    return {"accuracy": (correct / total) if total else 0.0, "correct": correct, "total": total}


def build_report_table(agent: str, mode: str, model: str, completed: str,
                       overall: dict, by_type: dict, run_stats: dict) -> str:
    def esc(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    def num(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1f}"

    lines = [
        "# LVBench Run Report",
        "",
        "| Task | Agent | Mode | Model | Completed | Accuracy (%) | Avg. rounds | Avg. tokens | Avg. time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    measured = run_stats.get("measured_tasks") or 0
    avg_rounds = (run_stats["agent_rounds"]["total"] / measured) if measured else None
    token_usage = run_stats.get("token_usage") or {}
    avg_tokens = None
    if token_usage.get("total_tokens") and measured:
        avg_tokens = token_usage["total_tokens"] / measured
    lines.append(
        "| Overall | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            esc(agent), esc(mode), esc(model), esc(completed),
            f"{overall['accuracy'] * 100:.1f}",
            num(avg_rounds),
            num(avg_tokens),
            "n/a",
        )
    )
    for task, stats in by_type.items():
        rounds = run_stats.get("agent_rounds_by_type", {}).get(task)
        avg_type_rounds = (rounds / stats["total"]) if rounds is not None else None
        lines.append(
            "| {} | {} | {} | {} | {}/{} | {} | {} | {} | {} |".format(
                esc(task), esc(agent), esc(mode), esc(model),
                stats["correct"], stats["total"], f"{stats['accuracy'] * 100:.1f}",
                num(avg_type_rounds), "n/a", "n/a",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="videospy-style report for WorldMM LVBench eval.")
    parser.add_argument("--eval-json", type=Path, required=True, help="Path to the {eval_name}_eval.json written by eval/eval.py.")
    parser.add_argument("--report-dir", type=Path, default=None, help="Output dir (default: same dir as --eval-json).")
    parser.add_argument("--agent", type=str, default="WorldMM")
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--model", type=str, default="Qwen3.5-4B")
    parser.add_argument("--usage-json", type=Path, default=None, help="Optional token usage json written by run_eval.py.")
    args = parser.parse_args()

    rows = json.loads(args.eval_json.read_text(encoding="utf-8"))
    report_dir = args.report_dir or args.eval_json.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    predictions: dict[str, str] = {}
    by_type_totals: dict[str, list[int]] = {}
    rounds_by_type: dict[str, int] = {}
    rounds_total = 0

    for row in rows:
        uid = str(row["ID"])
        choices = row.get("choices", {})
        response = row.get("response", "") or ""
        status = "error" if response in ("Error", "") else "success"
        letter = normalize_letter(response, choices)
        predictions[uid] = letter
        types = question_type_list(row.get("type", "")) or ["unknown"]
        rounds = int(row.get("num_rounds") or 0)
        record = {
            "sample_key": uid,
            "status": status,
            "video_id": row.get("video_id", ""),
            "question_types": types,
            "raw_prediction": response,
            "prediction": letter,
            "answer": row.get("answer", ""),
            "evaluate": bool(row.get("evaluate")),
            "agent_rounds": rounds,
        }
        records.append(record)
        if status == "success":
            rounds_total += rounds
            for t in types:
                totals = by_type_totals.setdefault(t, [0, 0])
                totals[1] += 1
                totals[0] += int(bool(row.get("evaluate")))
                rounds_by_type[t] = rounds_by_type.get(t, 0) + rounds

    total = len(records)
    correct = sum(1 for r in records if r["evaluate"])
    overall = accuracy_block(correct, total)
    by_type = {t: accuracy_block(c, n) for t, (c, n) in sorted(by_type_totals.items())}

    measured_tasks = sum(1 for r in records if r["status"] == "success")
    run_stats: dict = {
        "measured_tasks": measured_tasks,
        "agent_rounds": {"total": rounds_total, "average": (rounds_total / measured_tasks) if measured_tasks else 0.0},
        "agent_rounds_by_type": rounds_by_type,
    }
    if args.usage_json and args.usage_json.exists():
        run_stats["token_usage"] = json.loads(args.usage_json.read_text(encoding="utf-8"))

    metrics = {"overall": overall, "by_question_type": by_type, "run_statistics": run_stats}

    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (report_dir / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    with (report_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    completed = f"{correct}/{total}"
    report_md = build_report_table(args.agent, args.mode, args.model, completed, overall, by_type, run_stats)
    (report_dir / "report.md").write_text(report_md, encoding="utf-8")

    print(f"Report written to {report_dir / 'report.md'}")
    print(f"Accuracy: {correct}/{total} = {overall['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
