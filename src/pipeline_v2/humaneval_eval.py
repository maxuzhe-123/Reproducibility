from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict

from src.pipeline_v2.common import append_jsonl, config_dataset_path, load_json_objects, write_json
from src.pipeline_v2.evaluation_core import (
    create_evaluation_dir,
    evaluate_dataset,
    evaluation_timeout,
    mean_std,
    sample_evaluation_record,
)
from src.utils.codebleu_metrics import CODEBLEU_KEYS, compute_codebleu_detailed
from src.utils.exec_code import run_with_timeout
from src.utils.parse import extract_code


def load_humaneval_tasks(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    path = config_dataset_path(cfg, "src/datasets/HumanEval.jsonl")
    limit = int(cfg["experiment"]["num_tasks"])
    tasks = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            task_id = item.get("task_id", f"HumanEval/{idx}")
            tasks[task_id] = item
            if len(tasks) >= limit:
                break
    return tasks


def extract_humaneval_code(raw: str, task: Dict[str, Any]) -> str:
    code = extract_code(raw)
    entry_point = task.get("entry_point")
    if not code:
        return ""
    if entry_point and not re.search(
        rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.MULTILINE
    ):
        signature = next(
            (
                line.rstrip()
                for line in task["prompt"].splitlines()
                if re.match(r"^def\s+", line)
            ),
            None,
        )
        if signature:
            body = "\n".join(
                ("    " + line) if line.strip() else ""
                for line in code.splitlines()
            )
            code = signature + "\n" + body
    return textwrap.dedent(code).strip()


def strip_docstrings_and_comments(code: str) -> str:
    """Remove docstrings and comments from the code (reusing the tree-sitter stripping logic shipped with the codebleu package,
    identical to what its syntax_match/dataflow_match use internally).

    If parsing fails (e.g., the code itself is invalid), the code is returned unchanged so that CodeBLEU computation is not interrupted.
    """
    from codebleu.parser.utils import remove_comments_and_docstrings

    try:
        stripped = remove_comments_and_docstrings(code, "python")
        return stripped if stripped.strip() else code
    except Exception:
        return code


def extract_humaneval_code_no_docstring(raw: str, task: Dict[str, Any]) -> str:
    """On top of extract_humaneval_code, additionally strip docstrings/comments."""
    return strip_docstrings_and_comments(extract_humaneval_code(raw, task))


def compute_no_docstring_codebleu(output_dir: Path) -> Path:
    """For the 10 generations of each task in task_results.jsonl, strip docstrings/comments and recompute
    pairwise CodeBLEU (including the four components ngram/weighted_ngram/syntax/dataflow),
    written separately to:
      - codebleu_no_docstring.jsonl         per-task scores
      - codebleu_no_docstring_summary.json  dataset-level mean/std
      - task_results_no_docstring.jsonl     the stripped code of the 10 generations per task (for manually checking implementation differences)
    Does not affect the existing task_results.jsonl / reproducibility.json.
    """
    task_results_path = output_dir / "task_results.jsonl"
    records = load_json_objects(task_results_path)

    per_task_path = output_dir / "codebleu_no_docstring.jsonl"
    codes_path = output_dir / "task_results_no_docstring.jsonl"
    for p in (per_task_path, codes_path):
        if p.exists():
            p.unlink()

    per_task = []
    for record in records:
        stripped_preds = [
            {
                "sample_id": pred.get("sample_id"),
                "pass": pred.get("pass"),
                "code_no_docstring": strip_docstrings_and_comments(pred.get("code") or ""),
            }
            for pred in record.get("preds", [])
        ]
        append_jsonl(codes_path, {"task_id": record["task_id"], "preds": stripped_preds})

        codes = [p["code_no_docstring"] for p in stripped_preds]
        detail = compute_codebleu_detailed(codes)
        row = {"task_id": record["task_id"], **detail}
        append_jsonl(per_task_path, row)
        per_task.append(row)

    summary: Dict[str, Any] = {"num_tasks": len(per_task)}
    for key in CODEBLEU_KEYS:
        mean, std = mean_std([float(row[key]) for row in per_task])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    write_json(output_dir / "codebleu_no_docstring_summary.json", summary)

    print(f"[codebleu-no-docstring] 逐题分数: {per_task_path}")
    print(f"[codebleu-no-docstring] 数据集汇总: {output_dir / 'codebleu_no_docstring_summary.json'}")
    print(f"[codebleu-no-docstring] 剥离后代码: {codes_path}")
    return per_task_path


def humaneval_eval(
    generation_run: Path,
    cfg: Dict[str, Any],
    output_dir_path: str | None = None,
) -> Path:
    output_dir = create_evaluation_dir(cfg, generation_run, output_dir_path)
    tasks = load_humaneval_tasks(cfg)

    def sample_evaluator(
        task: Dict[str, Any], generation: Dict[str, Any]
    ) -> Dict[str, Any]:
        code = extract_humaneval_code(generation.get("raw") or "", task)
        passed, error = run_with_timeout(
            task["prompt"] + "\n" + code,
            task["test"],
            timeout_sec=evaluation_timeout(cfg),
            entry_point=task.get("entry_point"),
        )
        return sample_evaluation_record(generation, code, passed, error)

    output_dir = evaluate_dataset(
        generation_run,
        cfg,
        "humaneval",
        tasks,
        sample_evaluator,
        output_dir,
    )
    compute_no_docstring_codebleu(output_dir)
    return output_dir
