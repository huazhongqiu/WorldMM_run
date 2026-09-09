"""VideoSpy-compatible aggregate reporting for benchmark runs."""

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

COLUMNS = ["Task", "Agent", "Mode", "Model", "Completed", "Accuracy (%)", "Avg. rounds", "Avg. tokens", "Avg. time (s)"]


def _row(task: str, records: list[dict[str, Any]], agent: str, mode: str, model: str) -> dict[str, Any]:
    total = len(records)
    completed = sum(bool(record.get("completed")) for record in records)
    correct = sum(bool(record.get("evaluate")) for record in records)

    def average(field: str) -> float:
        return round(sum(float(record.get(field, 0)) for record in records) / total, 1) if total else 0.0

    return {
        "Task": task,
        "Agent": agent,
        "Mode": mode,
        "Model": model,
        "Completed": f"{completed}/{total}",
        "Accuracy (%)": round(correct * 100 / total, 1) if total else 0.0,
        "Avg. rounds": average("num_rounds"),
        "Avg. tokens": average("total_tokens"),
        "Avg. time (s)": average("elapsed_seconds"),
    }


def build_report_rows(records: list[dict[str, Any]], agent: str, mode: str, model: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("type", "Unknown"))].append(record)
    rows = [_row("Overall", records, agent, mode, model)]
    rows.extend(_row(task, group, agent, mode, model) for task, group in sorted(groups.items()))
    return rows


def write_report(rows: list[dict[str, Any]], output_dir: Path, title: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    markdown_path = output_dir / "report.md"
    markdown_path.write_text("# " + title + "\n\n| " + " | ".join(COLUMNS) + " |\n|" + "|".join(["---"] * len(COLUMNS)) + "|\n" + "\n".join("| " + " | ".join(str(row[column]) for column in COLUMNS) + " |" for row in rows) + "\n")
    table_rows = "\n".join("<tr>" + "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in COLUMNS) + "</tr>" for row in rows)
    html_path = output_dir / "report.html"
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>table{{border-collapse:collapse}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}</style></head><body><h1>{html.escape(title)}</h1><table><thead><tr>{''.join(f'<th>{html.escape(column)}</th>' for column in COLUMNS)}</tr></thead><tbody>{table_rows}</tbody></table></body></html>")
    return {"json": json_path, "markdown": markdown_path, "html": html_path}
