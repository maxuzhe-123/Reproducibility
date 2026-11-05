from __future__ import annotations

import math
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import yaml

from evaluate_reproducibility import evaluate_from_paths
from src.pipeline_v2.common import (
    append_jsonl,
    latest_by_sample,
    load_json_objects,
    resolve_evaluation_root,
    run_timestamp,
    sanitized_config,
    timestamp,
    write_json,
    write_yaml,
)
from src.utils.codebleu_metrics import compute_codebleu_detailed


SCHEMA_VERSION = 1
EVALUATOR_VERSION = "pipeline-v2-groundtruth-env-v1"
EXTRACTOR_VERSION = "pipeline-v2-python-v1"
SampleEvaluator = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


def load_generation_config(generation_run: Path) -> Dict[str, Any]:
    path = generation_run / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing generation config: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def evaluation_timeout(cfg: Dict[str, Any]) -> int:
    # Consistent with the codereval evaluation in GroundTruth.py / main.py: default 30 s.
    return int(
        cfg.get("evaluation", {}).get(
            "timeout_sec", cfg["experiment"].get("timeout_sec", 30)
        )
    )


def sample_evaluation_record(
    generation: Dict[str, Any],
    code: str,
    passed: bool,
    error: str,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": generation["sample_id"],
        "task_id": generation["task_id"],
        "sample_index": generation["sample_index"],
        "status": "evaluated",
        "code": code,
        "pass": bool(passed),
        "error": error,
        "extractor_version": EXTRACTOR_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "evaluated_at": timestamp(),
    }


def compute_pass_k_by_level(
    task_records: List[Dict[str, Any]], num_samples: int
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Unbiased pass@k (1 - C(n-c,k)/C(n,k)) + stratification by level.

    The formula is identical to src/utils/PythonExec.py; it directly reuses the already evaluated task_records
    (preds[].pass + level) without rerunning the executor.
    """
    n = max(int(num_samples), 1)
    tot_k = [0.0] * n
    level_tot: Dict[str, int] = {}
    level_acc: Dict[str, List[float]] = {}
    count = 0
    for record in task_records:
        preds = record.get("preds", [])
        n_total = len(preds)
        if n_total == 0:
            continue
        c = sum(1 for p in preds if p.get("pass"))
        level = record.get("level") or "UNKNOWN"
        count += 1
        level_tot[level] = level_tot.get(level, 0) + 1
        level_acc.setdefault(level, [0.0] * n)
        for k_idx in range(n):
            k_val = k_idx + 1
            n_fail = n_total - c
            if n_fail < k_val:
                delta = 1.0
            else:
                delta = float(
                    1.0 - np.prod(1.0 - k_val / np.arange(n_fail + 1, n_total + 1))
                )
            tot_k[k_idx] += delta
            level_acc[level][k_idx] += delta
    pass_k = [v / count if count else 0.0 for v in tot_k]
    by_level = {
        lvl: [v / level_tot[lvl] if level_tot[lvl] else 0.0 for v in level_acc[lvl]]
        for lvl in level_acc
    }
    return pass_k, by_level


# The six CoderEval dependency levels, in canonical order from weak to strong context dependency (used to sort stratified tables).
CODEREVAL_LEVEL_ORDER = (
    "self_contained",
    "slib_runnable",
    "plib_runnable",
    "class_runnable",
    "file_runnable",
    "project_runnable",
)


def write_codebleu_by_level_table(
    codebleu_by_level: Dict[str, Any],
    output_dir: Path,
    filename: str = "codebleu_by_level.csv",
    title: str = "CodeReval CodeBLEU —— 按依赖等级分层（依赖由弱到强）",
) -> Path:
    """Render codebleu_by_level as a separate stratified table:
    - write <filename> (one row per dependency level, overall CodeBLEU + 4 components, mean±std)
    - also print it to the console for direct inspection.

    Levels are sorted canonically by CODEREVAL_LEVEL_ORDER (dependency from weak to strong); unknown levels are appended at the end.
    """
    import csv

    keys = (
        ("codebleu", "CodeBLEU"),
        ("ngram_match_score", "ngram"),
        ("weighted_ngram_match_score", "weighted_ngram"),
        ("syntax_match_score", "syntax"),
        ("dataflow_match_score", "dataflow"),
    )
    ordered = [lvl for lvl in CODEREVAL_LEVEL_ORDER if lvl in codebleu_by_level]
    ordered += [lvl for lvl in codebleu_by_level if lvl not in CODEREVAL_LEVEL_ORDER]

    def cell(stats: Dict[str, Any], key: str) -> str:
        mean, std = stats.get(f"{key}_mean"), stats.get(f"{key}_std")
        return "" if mean is None else f"{mean:.4f}±{std:.4f}"

    rows = []
    for lvl in ordered:
        stats = codebleu_by_level[lvl]
        rows.append(
            [lvl, stats.get("count")] + [cell(stats, k) for k, _ in keys]
        )

    csv_path = output_dir / filename
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "count"] + [label for _, label in keys])
        writer.writerows(rows)

    # Console output (mean±std)
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)
    print(f"  {'level':<16}{'n':>4}  {'CodeBLEU':>15}  {'syntax':>15}  {'dataflow':>15}")
    print("  " + "-" * 74)
    for lvl, n, codebleu, _ng, _wng, syntax, dataflow in rows:
        print(f"  {lvl:<16}{n:>4}  {codebleu:>15}  {syntax:>15}  {dataflow:>15}")
    print("=" * 78)
    print(f"[codebleu] 分层表已保存: {csv_path}（含 ngram/weighted_ngram 全部 5 列）")
    return csv_path


def compute_codebleu_by_level(task_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group by codereval level and compute the mean/std of the overall CodeBLEU score + 4 components for each group.

    Same definition as the global codebleu_mean (task-level self-similarity CodeBLEU, then averaged within the group).
    """
    keys = (
        "codebleu",
        "ngram_match_score",
        "weighted_ngram_match_score",
        "syntax_match_score",
        "dataflow_match_score",
    )
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in task_records:
        groups[record.get("level") or "UNKNOWN"].append(record)
    out: Dict[str, Any] = {}
    for lvl in sorted(groups):
        recs = groups[lvl]
        stats: Dict[str, Any] = {"count": len(recs)}
        for key in keys:
            mean, std = mean_std(
                [float(r[key]) for r in recs if r.get(key) is not None]
            )
            stats[f"{key}_mean"] = mean
            stats[f"{key}_std"] = std
        out[lvl] = stats
    return out


def mean_std(values: List[float]) -> Tuple[float | None, float | None]:
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return None, None
    array = np.asarray(valid, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def build_task_record(
    task_id: str,
    generations: List[Dict[str, Any]],
    evaluations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    paired = sorted(
        zip(generations, evaluations),
        key=lambda item: int(item[0]["sample_index"]),
    )
    preds = [
        {
            "raw": generation.get("raw") or "",
            "code": evaluation.get("code") or "",
            "pass": bool(evaluation.get("pass")),
            "error": evaluation.get("error") or "",
            "sample_id": generation["sample_id"],
        }
        for generation, evaluation in paired
    ]
    codes = [item["code"] for item in preds]
    detail = compute_codebleu_detailed(codes)
    metadata = generations[0].get("task_metadata", {}) if generations else {}
    return {
        "_id": metadata.get("_id"),
        "task_id": task_id,
        "level": metadata.get("level"),
        "prompt": generations[0].get("prompt", "") if generations else "",
        "ok_pass1": bool(preds and preds[0]["pass"]),
        "ok_passk": any(item["pass"] for item in preds),
        "codebleu": detail["codebleu"],
        "ngram_match_score": detail["ngram_match_score"],
        "weighted_ngram_match_score": detail["weighted_ngram_match_score"],
        "syntax_match_score": detail["syntax_match_score"],
        "dataflow_match_score": detail["dataflow_match_score"],
        "preds": preds,
        "rag": metadata.get("rag"),
    }


def create_evaluation_dir(
    cfg: Dict[str, Any], generation_run: Path, requested: str | None
) -> Path:
    if requested:
        path = Path(requested)
    else:
        # Same layout as preds, with one extra level for sampling parameters, to support "same prompt, different sampling parameters" comparisons:
        # <eval_root>/<dataset>/<model>/<template or defense/<name>>/<tem_..._top_k>/
        from src.pipeline_v2.generation_core import experiment_relpath, sampling_token

        path = resolve_evaluation_root(cfg) / experiment_relpath(cfg) / sampling_token(cfg)
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_dataset(
    generation_run: Path,
    cfg: Dict[str, Any],
    dataset: str,
    tasks: Dict[str, Dict[str, Any]],
    sample_evaluator: SampleEvaluator,
    output_dir: Path,
    include_pass_k_by_level: bool = False,
) -> Path:
    # Read the prediction file with the same name as in the gen stage (including _def_<defense>), rather than a hard-coded generations.jsonl.
    from src.pipeline_v2.generation_core import prediction_filename

    generation_records = latest_by_sample(
        load_json_objects(generation_run / prediction_filename(cfg))
    )
    eligible = [
        record
        for record in generation_records.values()
        if (
            record.get("status") == "success"
            and record.get("raw") is not None
        )
        or record.get("status") == "blocked"
    ]
    evaluations_path = output_dir / "evaluations.jsonl"
    task_results_path = output_dir / "task_results.jsonl"
    # Idempotent: clear possible old outputs to avoid duplicates from appending
    # (reproducibility.json rereads task_results.jsonl, and duplicates would corrupt its results).
    for _stale in (evaluations_path, task_results_path):
        if _stale.exists():
            _stale.unlink()
    write_yaml(output_dir / "config.yaml", sanitized_config(cfg))

    grouped_generations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    grouped_evaluations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for index, generation in enumerate(
        sorted(eligible, key=lambda item: (item["task_index"], item["sample_index"]))
    ):
        task_id = generation["task_id"]
        task = tasks.get(task_id)
        if generation.get("status") == "blocked":
            evaluation = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": generation["sample_id"],
                "task_id": task_id,
                "sample_index": generation["sample_index"],
                "status": "blocked",
                "code": "",
                "pass": False,
                "error": "blocked_by_defense",
                "extractor_version": EXTRACTOR_VERSION,
                "evaluator_version": EVALUATOR_VERSION,
                "evaluated_at": timestamp(),
            }
        elif task is None:
            evaluation = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": generation["sample_id"],
                "task_id": task_id,
                "sample_index": generation["sample_index"],
                "status": "infrastructure_error",
                "code": "",
                "pass": None,
                "error": f"task_not_found:{task_id}",
                "extractor_version": EXTRACTOR_VERSION,
                "evaluator_version": EVALUATOR_VERSION,
                "evaluated_at": timestamp(),
            }
        else:
            evaluation = sample_evaluator(task, generation)
        append_jsonl(evaluations_path, evaluation)
        grouped_generations[task_id].append(generation)
        grouped_evaluations[task_id].append(evaluation)
        print(f"[{index + 1}/{len(eligible)}] {generation['sample_id']}")

    task_records = []
    for task_id in sorted(
        grouped_generations,
        key=lambda key: min(item["task_index"] for item in grouped_generations[key]),
    ):
        usable_evaluations = [
            item
            for item in grouped_evaluations[task_id]
            if item.get("status") in {"evaluated", "blocked"}
        ]
        usable_ids = {item["sample_id"] for item in usable_evaluations}
        usable_generations = [
            item
            for item in grouped_generations[task_id]
            if item["sample_id"] in usable_ids
        ]
        if not usable_evaluations:
            continue
        record = build_task_record(task_id, usable_generations, usable_evaluations)
        task_records.append(record)
        append_jsonl(task_results_path, record)

    total = len(task_records)
    num_samples = int(cfg["experiment"].get("num_samples", 1))
    pass1 = sum(record["ok_pass1"] for record in task_records) / total if total else 0
    passk = sum(record["ok_passk"] for record in task_records) / total if total else 0
    metrics = {
        "pass@1": pass1,
        f"pass@{num_samples}": passk,
        "total_tasks_evaluated": total,
        "total_samples_evaluated": sum(
            len(record["preds"]) for record in task_records
        ),
    }
    for key in (
        "codebleu",
        "ngram_match_score",
        "weighted_ngram_match_score",
        "syntax_match_score",
        "dataflow_match_score",
    ):
        mean, std = mean_std([float(record[key]) for record in task_records])
        metrics[f"{key}_mean"] = mean
        metrics[f"{key}_std"] = std

    # Run metadata: lets the multi-dimensional comparison (compare.py) group by model/prompt/defense/sampling parameters
    # without relying on directory parsing (in particular, the base template of defense experiments is lost in the directory).
    from src.pipeline_v2.generation_core import (
        experiment_subdir,
        model_token,
        sampling_token,
    )

    model_cfg = cfg.get("model", {})
    prompt_cfg = cfg.get("prompt", {})
    defense = prompt_cfg.get("defense")
    is_defense = (
        bool(defense) and str(defense).strip().lower() not in ("no", "none", "")
    ) or ("defense" in str(cfg.get("experiment", {}).get("name") or "").lower())
    run_meta = {
        "model_id": model_cfg.get("model_id"),
        "model": model_token(cfg),
        "template": prompt_cfg.get("template"),
        "defense": defense,
        "is_defense": bool(is_defense),
        "experiment_sub": str(experiment_subdir(cfg)),
        "sampling_token": sampling_token(cfg),
        "temperature": model_cfg.get("temperature"),
        "top_p": model_cfg.get("top_p"),
        "top_k": model_cfg.get("top_k"),
    }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "evaluation",
        "experiment": cfg["experiment"].get("name") or run_meta["experiment_sub"],
        "dataset": dataset,
        "generation_run": str(generation_run),
        "evaluator_version": EVALUATOR_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "run": run_meta,
        "metrics": metrics,
        "environment": {"python": platform.python_version()},
        "created_at": timestamp(),
    }
    # Aligned with the codereval evaluation in main.py: add unbiased pass@k + stratification by level.
    # Also compute CodeBLEU per level (overall score + 4 components), aligned with by_level for pass.
    if include_pass_k_by_level:
        pass_k, by_level = compute_pass_k_by_level(task_records, num_samples)
        summary["pythonexec_pass_k"] = pass_k
        summary["pythonexec_by_level"] = by_level
        summary["codebleu_by_level"] = compute_codebleu_by_level(task_records)
        # Output a separate stratified table of CodeBLEU for each of the six dependency levels (CSV + console).
        write_codebleu_by_level_table(summary["codebleu_by_level"], output_dir)
    write_json(output_dir / "summary.json", summary)
    evaluate_from_paths(
        preds_path=str(task_results_path),
        output_path=str(output_dir / "reproducibility.json"),
        verbose=True,
    )
    print(f"Evaluation run: {output_dir}")
    return output_dir
