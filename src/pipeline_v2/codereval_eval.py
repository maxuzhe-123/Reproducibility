from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from src.datasets.codereval import load_codereval_simple
from src.pipeline_v2.common import append_jsonl, load_json_objects, write_json
from src.pipeline_v2.evaluation_core import (
    compute_codebleu_by_level,
    create_evaluation_dir,
    evaluate_dataset,
    evaluation_timeout,
    mean_std,
    sample_evaluation_record,
    write_codebleu_by_level_table,
)
from src.utils.codebleu_metrics import CODEBLEU_KEYS, compute_codebleu_detailed
from src.utils.parse import extract_code
from src.utils.repocoder.codereval_adapter import CoderEvalRepoCoderAdapter


def load_codereval_tasks(
    cfg: Dict[str, Any], runner: CoderEvalRepoCoderAdapter
) -> Dict[str, Dict[str, Any]]:
    mode = cfg.get("rag", {}).get("mode", "no_rag")
    limit = int(cfg["experiment"]["num_tasks"])
    if mode == "rg1":
        return {
            task["metadata"]["task_id"]: task
            for task in runner.load_tasks(num_tasks=limit)
        }
    ce_cfg = cfg.get("codereval", {})
    tasks = load_codereval_simple(
        index_json_path=str(
            Path(
                ce_cfg.get(
                    "index_path",
                    "src/datasets/CoderEval/CoderEval4Python.json",
                )
            ).resolve()
        ),
        repos_root=str(Path(ce_cfg.get("repos_root", "repos")).resolve()),
        n=limit,
    )
    return {task["task_id"]: task for task in tasks}


def extract_codereval_no_rag(raw: str, prompt: str) -> str:
    code = extract_code(raw)
    canonical = next(
        (line.rstrip() for line in prompt.splitlines() if line.startswith("def ")),
        None,
    )
    if code and not re.match(r"^\s*def\s+", code):
        lines = code.splitlines()
        found = next(
            (idx for idx, line in enumerate(lines) if re.match(r"^\s*def\s+", line)),
            None,
        )
        if found is not None:
            code = "\n".join(lines[found:])
        elif canonical:
            code = canonical + "\n" + "\n".join(
                ("    " + line) if line.strip() else "" for line in lines
            )
    if canonical and code:
        lines = code.splitlines()
        for idx, line in enumerate(lines):
            if re.match(r"^\s*def\s+", line):
                lines[idx] = canonical
                break
        code = "\n".join(lines)
    return code


def compute_no_docstring_codebleu(output_dir: Path) -> Path:
    """CoderEval version of "recompute CodeBLEU after stripping docstrings/comments", aligned with
    humaneval_eval.compute_no_docstring_codebleu, additionally stratified by level
    (same definition as codebleu_by_level.csv). Outputs:
      - codebleu_no_docstring.jsonl              per-task scores (with level)
      - codebleu_no_docstring_summary.json        dataset-level mean/std + by_level
      - codebleu_by_level_no_docstring.csv        table stratified by dependency level after stripping
      - task_results_no_docstring.jsonl           the stripped code of the 10 generations per task
    Does not affect the existing task_results.jsonl / reproducibility.json.
    """
    from src.pipeline_v2.humaneval_eval import strip_docstrings_and_comments

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
        append_jsonl(
            codes_path,
            {
                "task_id": record["task_id"],
                "level": record.get("level"),
                "preds": stripped_preds,
            },
        )

        codes = [p["code_no_docstring"] for p in stripped_preds]
        detail = compute_codebleu_detailed(codes)
        row = {"task_id": record["task_id"], "level": record.get("level"), **detail}
        append_jsonl(per_task_path, row)
        per_task.append(row)

    summary: Dict[str, Any] = {"num_tasks": len(per_task)}
    for key in CODEBLEU_KEYS:
        mean, std = mean_std([float(row[key]) for row in per_task])
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    summary["by_level"] = compute_codebleu_by_level(per_task)
    write_json(output_dir / "codebleu_no_docstring_summary.json", summary)
    write_codebleu_by_level_table(
        summary["by_level"],
        output_dir,
        filename="codebleu_by_level_no_docstring.csv",
        title="CodeReval CodeBLEU（剥离 docstring/注释）—— 按依赖等级分层（依赖由弱到强）",
    )

    print(f"[codebleu-no-docstring] 逐题分数: {per_task_path}")
    print(f"[codebleu-no-docstring] 数据集汇总: {output_dir / 'codebleu_no_docstring_summary.json'}")
    print(f"[codebleu-no-docstring] 剥离后代码: {codes_path}")
    return per_task_path


def codereval_eval(
    generation_run: Path,
    cfg: Dict[str, Any],
    output_dir_path: str | None = None,
) -> Path:
    output_dir = create_evaluation_dir(cfg, generation_run, output_dir_path)
    runner = CoderEvalRepoCoderAdapter(cfg)
    runner.build_validate_file_map()
    tasks = load_codereval_tasks(cfg, runner)
    mode = cfg.get("rag", {}).get("mode", "no_rag")

    def sample_evaluator(
        task: Dict[str, Any], generation: Dict[str, Any]
    ) -> Dict[str, Any]:
        raw = generation.get("raw") or ""
        code = (
            runner.extract_and_fix_code(raw, task)
            if mode == "rg1"
            else extract_codereval_no_rag(raw, generation["prompt"])
        )
        passed, error = runner.eval_one(
            task,
            code,
            timeout_sec=evaluation_timeout(cfg),
        )
        return sample_evaluation_record(generation, code, passed, error)

    output_dir = evaluate_dataset(
        generation_run,
        cfg,
        "codereval",
        tasks,
        sample_evaluator,
        output_dir,
        include_pass_k_by_level=True,
    )
    compute_no_docstring_codebleu(output_dir)
    return output_dir
