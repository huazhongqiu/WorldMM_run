"""Summarize QA accuracy, completion and failures against frozen manifests."""

import argparse
import json
from pathlib import Path


def summarize(expected_ids, results):
    latest = {str(row["ID"]): row for row in results}
    valid = {}
    errors = []
    missing = []
    for question_id in expected_ids:
        row = latest.get(question_id)
        if row is None:
            missing.append(question_id)
            continue
        response = str(row.get("response", "")).strip()
        completed = row.get("status") == "success" or row.get("completed") is True
        if completed and response and response not in ("Error", "Unable to generate answer"):
            valid[question_id] = row
        else:
            errors.append({"ID": question_id, "status": row.get("status", "error"), "response": response})
    correct = sum(bool(row.get("evaluate")) for row in valid.values())
    return {"total": len(expected_ids), "completed": len(valid), "correct": correct,
            "accuracy": correct / len(expected_ids), "failed": len(errors),
            "missing": len(missing), "failures": errors, "missing_ids": missing,
            "unexpected_ids": sorted(set(latest) - set(expected_ids))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--needed-root", type=Path, default=Path("/myworkspace/projects/worldmm_needed"))
    parser.add_argument("--output-root", type=Path, default=Path("/myworkspace/projects/output/worldmm"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dataset in ("lvbench", "egolife", "videomme_long"):
        manifest = args.needed_root / dataset / "shared/manifests/question_ids.json"
        expected = [str(value) for value in json.loads(manifest.read_text())]
        profiles = ("Base", "C", "S", "CS") if dataset == "videomme_long" else (
            "Base", "C", "S", "R", "CS", "CR", "SR", "CSR")
        for backend in ("Qwen3-Embedding-4B+VLM2Vec-V2", "Qwen3.5-4B-FreeRet", "WeMM-Embedding-4B"):
            for profile in profiles if backend == "Qwen3.5-4B-FreeRet" else ("Base",):
                source = args.output_root / dataset / backend / profile / args.run_id
                records = source / "records.jsonl"
                result = []
                if records.exists():
                    for line in records.read_text().splitlines():
                        try:
                            result.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                else:
                    files = list(source.glob("*/egolife_eval_A1_JAKE.json"))
                    if files:
                        result = json.loads(files[0].read_text())
                rows.append({"dataset": dataset, "backend": backend, "profile": profile,
                             "source": str(source), **summarize(expected, result)})
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    table = ["| Dataset | Backend | Profile | Accuracy | Completed | Failed | Missing |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    table.extend(f"| {r['dataset']} | {r['backend']} | {r['profile']} | {r['accuracy']:.2%} | "
                 f"{r['completed']}/{r['total']} | {r['failed']} | {r['missing']} |" for r in rows)
    (args.report_dir / "comparison.md").write_text("\n".join(table) + "\n")
    print(args.report_dir / "comparison.md")


if __name__ == "__main__":
    main()
