"""Recompute the no-docstring/no-comment CodeBLEU (overall score + the four components ngram/weighted_ngram/syntax/dataflow)
for an existing preds file (one line per task, with task_id + preds[].code).

The logic is identical to humaneval_eval.compute_no_docstring_codebleu, except that the input file need not be
named task_results.jsonl or live in an evaluate_dataset() output directory — it can point directly to
any historical preds file (including the pretty-JSONL format produced by the old main.py pipeline).

Output (written next to the preds file by default):
  - codebleu_no_docstring.jsonl         per-task scores
  - codebleu_no_docstring_summary.json  dataset-level mean/std
  - task_results_no_docstring.jsonl     per-task code with docstrings/comments stripped (for manual checking)

Usage:
    python -m src.pipeline_v2.codebleu_no_docstring_from_preds \
        --preds "runs/baseline_qwen_coder/2026-06-02_00-37-03/preds.jsonl"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from src.pipeline_v2.common import append_jsonl, load_json_objects, write_json
from src.pipeline_v2.evaluation_core import mean_std
from src.pipeline_v2.humaneval_eval import strip_docstrings_and_comments
from src.utils.codebleu_metrics import CODEBLEU_KEYS, compute_codebleu_detailed


def compute_codebleu_no_docstring_from_preds(preds_path: Path, output_dir: Path) -> Path:
    records = load_json_objects(preds_path)
    if not records:
        raise SystemExit(f"未读到任何记录: {preds_path}")

    per_task_path = output_dir / "codebleu_no_docstring.jsonl"
    codes_path = output_dir / "task_results_no_docstring.jsonl"
    for p in (per_task_path, codes_path):
        if p.exists():
            p.unlink()

    per_task = []
    for record in records:
        task_id = record.get("task_id")
        preds = record.get("preds", [])
        stripped_preds = [
            {
                "pass": pred.get("pass"),
                "code_no_docstring": strip_docstrings_and_comments(pred.get("code") or ""),
            }
            for pred in preds
        ]
        append_jsonl(codes_path, {"task_id": task_id, "preds": stripped_preds})

        codes = [p["code_no_docstring"] for p in stripped_preds]
        detail = compute_codebleu_detailed(codes)
        row = {"task_id": task_id, **detail}
        append_jsonl(per_task_path, row)
        per_task.append(row)

    summary: Dict[str, Any] = {
        "source_preds": str(preds_path),
        "num_tasks": len(per_task),
    }
    for key in CODEBLEU_KEYS:
        mean, std = mean_std([float(row[key]) for row in per_task])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    summary_path = output_dir / "codebleu_no_docstring_summary.json"
    write_json(summary_path, summary)

    print(f"[codebleu-no-docstring] 输入: {preds_path} ({len(per_task)} 题)")
    print(f"[codebleu-no-docstring] 逐题分数: {per_task_path}")
    print(f"[codebleu-no-docstring] 数据集汇总: {summary_path}")
    print(f"[codebleu-no-docstring] 剥离后代码: {codes_path}")
    for key in CODEBLEU_KEYS:
        print(f"  {key:<28} mean={summary[f'{key}_mean']:.4f}  std={summary[f'{key}_std']:.4f}")
    return summary_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True, help="preds 文件路径（task_id + preds[].code 结构）")
    ap.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认与 --preds 同目录",
    )
    args = ap.parse_args()

    preds_path = Path(args.preds)
    output_dir = Path(args.output_dir) if args.output_dir else preds_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_codebleu_no_docstring_from_preds(preds_path, output_dir)


if __name__ == "__main__":
    main()
