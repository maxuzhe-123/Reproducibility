"""CodeJudgeBench evaluation stage (pipeline_v2) — a standalone flow isolated from humaneval/codereval.

Verdicts follow the original CodeJudgeBench protocol: parse [[A>B]]/[[B>A]] in the judge output to get 'A'/'B', and compare with the gold label
to obtain the correctness of each judgment. RQ5 reproducibility metrics (N=10 repetitions per instance, computed per instance and then averaged over instances):

  ReproCorrect@A / @B   mean per-instance "correct count / N" when the good response is in position A (fwd) / B (rev)
  ReproCorrect_Rate     (ReproCorrect@A + ReproCorrect@B) / 2
  Repro@A / @B          mean per-instance "majority count max(countA, countB) / N" (position A / B)
  Repro_Rate            (Repro@A + Repro@B) / 2
  *_std                 sample standard deviation of each quantity across instances (error bars)

Results are also bucketed by difficulty (easy/medium/hard). Reuses the common scaffolding; verdict parsing reuses the original parse_codejudge_verdict.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.pipeline_v2.common import (
    append_jsonl,
    latest_by_sample,
    load_json_objects,
    sanitized_config,
    timestamp,
    write_json,
    write_yaml,
)
from src.pipeline_v2.evaluation_core import create_evaluation_dir
from src.pipeline_v2.generation_core import (
    experiment_subdir,
    model_token,
    prediction_filename,
    sampling_token,
)
from src.utils.parse import parse_codejudge_verdict

SCHEMA_VERSION = 1
EVALUATOR_VERSION = "pipeline-v2-codejudge-v6-repro"


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def _mean(xs: List[float]) -> float:
    return _safe_div(sum(xs), len(xs))


def _std(xs: List[float]) -> float:
    """Sample standard deviation across instances (meaningful only for n>=2); used as bar-chart error bars."""
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _repro_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """RQ5 reproducibility metrics (for a set of instances, each already aggregated over 10 repetitions).

    Two base quantities per instance (both with the number of repetitions N=10 as denominator):
      ReproCorrect = correct count / N           → instance_accuracy
      Repro        = max(countA, countB) / N → choice_consistency
    Split by placement into fwd (=@A, good in position A) / rev (=@B, good in position B) and averaged over instances;
    Rate = average of @A and @B (fwd and rev have the same number of instances, so it equals the average over all instances).
    std is the sample standard deviation of the corresponding instance set, used as error bars.
    """
    fwd = [r for r in records if not r["reverse"]]
    rev = [r for r in records if r["reverse"]]
    rc_a = [r["instance_accuracy"] for r in fwd]
    rc_b = [r["instance_accuracy"] for r in rev]
    rp_a = [r["choice_consistency"] for r in fwd]
    rp_b = [r["choice_consistency"] for r in rev]
    m_rc_a, m_rc_b = _mean(rc_a), _mean(rc_b)
    m_rp_a, m_rp_b = _mean(rp_a), _mean(rp_b)
    return {
        "ReproCorrect@A": m_rc_a,
        "ReproCorrect@B": m_rc_b,
        "ReproCorrect_Rate": (m_rc_a + m_rc_b) / 2.0,
        "Repro@A": m_rp_a,
        "Repro@B": m_rp_b,
        "Repro_Rate": (m_rp_a + m_rp_b) / 2.0,
        "ReproCorrect@A_std": _std(rc_a),
        "ReproCorrect@B_std": _std(rc_b),
        "ReproCorrect_Rate_std": _std(rc_a + rc_b),
        "Repro@A_std": _std(rp_a),
        "Repro@B_std": _std(rp_b),
        "Repro_Rate_std": _std(rp_a + rp_b),
        "n_instances": len(records),
        "n_instances_a": len(fwd),
        "n_instances_b": len(rev),
    }


def codejudge_eval(
    generation_run: Path,
    cfg: Dict[str, Any],
    output_dir_path: str | None = None,
) -> Path:
    output_dir = create_evaluation_dir(cfg, generation_run, output_dir_path)
    gen_records = latest_by_sample(
        load_json_objects(generation_run / prediction_filename(cfg))
    )
    eligible = [
        r
        for r in gen_records.values()
        if (r.get("status") == "success" and r.get("raw") is not None)
    ]

    evaluations_path = output_dir / "evaluations.jsonl"
    task_results_path = output_dir / "task_results.jsonl"
    for stale in (evaluations_path, task_results_path):  # Idempotent
        if stale.exists():
            stale.unlink()
    write_yaml(output_dir / "config.yaml", sanitized_config(cfg))

    # —— Per sample: parse the verdict and score it; also group by task_id (to compute per-instance consistency/entropy) ——
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for gen in sorted(
        eligible, key=lambda x: (x.get("task_index", 0), x.get("sample_index", 0))
    ):
        meta = gen.get("task_metadata", {}) or {}
        label = meta.get("label")
        raw = gen.get("raw") or ""
        choice = parse_codejudge_verdict(raw)  # 'A'/'B'/''
        valid = choice in ("A", "B")
        correct = valid and (choice == label)

        eval_rec = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": gen["sample_id"],
            "task_id": gen["task_id"],
            "sample_index": gen.get("sample_index", 0),
            "reverse": bool(meta.get("reverse")),
            "difficulty": meta.get("difficulty", "unknown"),
            "response_model": meta.get("response_model", "unknown"),
            "label": label,
            "choice": choice,
            "valid": valid,
            "correct": correct,
            "evaluator_version": EVALUATOR_VERSION,
            "evaluated_at": timestamp(),
        }
        append_jsonl(evaluations_path, eval_rec)
        grouped[gen["task_id"]].append(eval_rec)

    # —— Aggregate per instance (problem × responder × placement): count(A/B), consistency, entropy ——
    instance_records: List[Dict[str, Any]] = []
    for task_id, samples in grouped.items():
        samples.sort(key=lambda x: int(x["sample_index"]))
        n = len(samples)
        count_a = sum(1 for s in samples if s["choice"] == "A")
        count_b = sum(1 for s in samples if s["choice"] == "B")
        n_correct = sum(1 for s in samples if s["correct"])
        n_valid = count_a + count_b
        meta0 = samples[0]
        rec = {
            "task_id": task_id,
            "difficulty": meta0["difficulty"],
            "response_model": meta0["response_model"],
            "reverse": meta0["reverse"],
            "n_samples": n,
            "n_correct": n_correct,
            "count_a": count_a,
            "count_b": count_b,
            "n_valid": n_valid,
            # ReproCorrect (per instance): correct count / N
            "instance_accuracy": _safe_div(n_correct, n),
            # Repro (per instance): majority count / N (denominator N; empty verdicts lower it)
            "choice_consistency": _safe_div(max(count_a, count_b), n),
        }
        instance_records.append(rec)
        append_jsonl(task_results_path, rec)

    # —— RQ5 reproducibility metrics: overall ——
    metrics: Dict[str, Any] = {
        **_repro_metrics(instance_records),
        "total_samples": sum(r["n_samples"] for r in instance_records),
    }

    # —— Bucketed by difficulty (same set of reproducibility metrics) ——
    def bucket(field: str) -> Dict[str, Any]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in instance_records:
            groups[str(r.get(field, "unknown"))].append(r)
        return {key: _repro_metrics(groups[key]) for key in sorted(groups)}

    model_cfg = cfg.get("model", {})
    prompt_cfg = cfg.get("prompt", {})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "evaluation",
        "experiment": cfg["experiment"].get("name") or str(experiment_subdir(cfg)),
        "dataset": "codejudgebench",
        "generation_run": str(generation_run),
        "evaluator_version": EVALUATOR_VERSION,
        "run": {
            "model_id": model_cfg.get("model_id"),
            "model": model_token(cfg),
            "template": prompt_cfg.get("template"),
            "defense": None,
            "is_defense": False,
            "experiment_sub": str(experiment_subdir(cfg)),
            "sampling_token": sampling_token(cfg),
            "temperature": model_cfg.get("temperature"),
            "top_p": model_cfg.get("top_p"),
            "top_k": model_cfg.get("top_k"),
        },
        "metrics": metrics,
        "by_difficulty": bucket("difficulty"),
        "created_at": timestamp(),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Evaluation run: {output_dir}")
    return output_dir
