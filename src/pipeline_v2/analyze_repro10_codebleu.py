"""Qwen3-coder-next baseline — distribution of the four CodeBLEU components + overall score across 10 independent repeated runs.

Unlike codebleu_mean/std in evaluate_dataset() (self-similarity among the 10 samples within one run),
this measures how much the dataset-level mean itself fluctuates across N full runs of the same config
(run-level reproducibility rather than sample-level).

Only **reads** runs_v2/evaluations_repro10/run_*/codebleu_no_docstring_summary.json
and does not modify any existing outputs.

Usage:
    python -m src.pipeline_v2.analyze_repro10_codebleu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

COMPONENTS = [
    "codebleu",
    "ngram_match_score",
    "weighted_ngram_match_score",
    "syntax_match_score",
    "dataflow_match_score",
]
COMPONENT_LABEL = {
    "codebleu": "CodeBLEU",
    "ngram_match_score": "ngram",
    "weighted_ngram_match_score": "weighted_ngram",
    "syntax_match_score": "syntax",
    "dataflow_match_score": "dataflow",
}


def load_run_means(eval_root: Path) -> Dict[str, Dict[str, float]]:
    """{run_name: {component: dataset_level_mean}}, sorted by the directory names run_01..run_NN."""
    runs: Dict[str, Dict[str, float]] = {}
    for run_dir in sorted(eval_root.glob("run_*")):
        summary_path = run_dir / "codebleu_no_docstring_summary.json"
        if not summary_path.exists():
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        runs[run_dir.name] = {c: float(data[f"{c}_mean"]) for c in COMPONENTS}
    return runs


def render_boxplot(runs: Dict[str, Dict[str, float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    data = [[runs[r][c] for r in runs] for c in COMPONENTS]
    labels = [COMPONENT_LABEL[c] for c in COMPONENTS]
    ax.boxplot(data, labels=labels, showmeans=True)
    for i, vals in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.03, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, alpha=0.6, color="#4C72B0")
    ax.set_ylabel("Per-run dataset-level mean")
    ax.set_title(f"Qwen3-coder-next baseline — CodeBLEU across {len(runs)} repeated runs")
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Qwen3 baseline 10 次重复运行的 CodeBLEU 四子分量 + 综合分跨运行分布"
    )
    ap.add_argument(
        "--eval-root",
        default=str(REPO_ROOT / "runs_v2/evaluations_repro10"),
        help="包含 run_01..run_NN 子目录的评测根目录",
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "runs_v2/comparisons_repro10/qwen3_baseline_codebleu_repro10"),
    )
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    runs = load_run_means(eval_root)
    if not runs:
        raise SystemExit(
            f"未找到任何 run_*/codebleu_no_docstring_summary.json，请先跑 "
            f"scripts/run_qwen3_baseline_repro10.sh（eval_root={eval_root}）"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for c in COMPONENTS:
        vals = np.array([runs[r][c] for r in runs], dtype=float)
        stats[c] = {
            "per_run": {r: runs[r][c] for r in runs},
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "range": float(vals.max() - vals.min()),
        }

    render_boxplot(runs, out_dir / "codebleu_repro10_boxplot.png")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"num_runs": len(runs), "components": stats}, f, ensure_ascii=False, indent=2)

    print(f"[repro10] 完成，输出于 {out_dir}")
    for c in COMPONENTS:
        s = stats[c]
        print(
            f"  {COMPONENT_LABEL[c]:<15} mean={s['mean']:.4f}  std={s['std']:.4f}  "
            f"range=[{s['min']:.4f}, {s['max']:.4f}]"
        )


if __name__ == "__main__":
    main()
