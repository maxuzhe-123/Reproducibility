"""Grouped bar chart of the direction of change in the four metrics (ngram/weighted_ngram/syntax/dataflow).

Compares two pairs of experiments:
  - Baseline vs few-shot
  - Few-shot vs few-shot CoT
For each pair, the 164 tasks are split into four classes by the direction of change of the four codebleu components:
  all up / all down / mixed / no change
Drawn as two side-by-side subplots, with bars annotated with task counts and proportions.

Only **reads** codebleu_no_docstring.jsonl; does not modify any existing outputs.

Usage:
    python -m src.pipeline_v2.compare_reproducibility_groups
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]

COMPONENTS = ["ngram_match_score", "weighted_ngram_match_score", "syntax_match_score", "dataflow_match_score"]

# Class order (bars from left to right) and display colors
CATEGORY_ORDER = ["fully_increase", "fully_decrease", "mixed", "no_change"]
CATEGORY_LABEL = {"fully_increase": "All Up", "fully_decrease": "All Down", "mixed": "Mixed", "no_change": "No Change"}
CATEGORY_COLOR = {"fully_increase": "#55A868", "fully_decrease": "#C44E52", "mixed": "#DD8452", "no_change": "#8172B3"}


def load_scores(path: Path) -> Dict[str, Dict[str, float]]:
    d: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            d[row["task_id"]] = row
    return d


def classify(ref: Dict[str, Dict[str, float]], treat: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {k: [] for k in CATEGORY_ORDER}
    for task_id in sorted(ref, key=lambda t: int(t.split("/")[1])):
        deltas = [treat[task_id][k] - ref[task_id][k] for k in COMPONENTS]
        signs = [1 if v > 1e-9 else (-1 if v < -1e-9 else 0) for v in deltas]
        if all(s <= 0 for s in signs) and any(s < 0 for s in signs):
            groups["fully_decrease"].append(task_id)
        elif all(s >= 0 for s in signs) and any(s > 0 for s in signs):
            groups["fully_increase"].append(task_id)
        elif all(s == 0 for s in signs):
            groups["no_change"].append(task_id)
        else:
            groups["mixed"].append(task_id)
    return groups


def render(panels: List[Tuple[str, Dict[str, List[str]], int]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.2))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, groups, total) in zip(axes, panels):
        labels = [CATEGORY_LABEL[k] for k in CATEGORY_ORDER]
        counts = [len(groups[k]) for k in CATEGORY_ORDER]
        colors = [CATEGORY_COLOR[k] for k in CATEGORY_ORDER]
        bars = ax.bar(labels, counts, color=colors, alpha=0.9)
        for b, n in zip(bars, counts):
            pct = n / total * 100
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{n}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.set_ylabel("# of tasks")
        ax.set_ylim(0, max(counts) * 1.22)
        ax.grid(axis="y", ls="--", alpha=0.4)
    fig.suptitle("Direction of change across 4 CodeBLEU sub-metrics", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline vs few-shot / Few-shot vs CoT 四指标变化方向分组柱状图")
    model_dir = "humaneval/qwen3_coder_next"
    ap.add_argument("--baseline", default=str(
        REPO_ROOT / "runs_v2/evaluations" / model_dir / "humaneval_original/tem_0.6_top_p_0.9_top_k_100/codebleu_no_docstring.jsonl"))
    ap.add_argument("--few-shot", default=str(
        REPO_ROOT / "runs_v2/evaluations" / model_dir / "humaneval_few_shot/tem_0.6_top_p_0.9_top_k_100/codebleu_no_docstring.jsonl"))
    ap.add_argument("--cot", default=str(
        REPO_ROOT / "runs_v2/evaluations" / model_dir / "humaneval_cot/tem_0.6_top_p_0.95_top_k_20/codebleu_no_docstring.jsonl"))
    ap.add_argument("--out", default=str(
        REPO_ROOT / "runs_v2/comparisons_supplement/reproducibility_groups_qwen3"))
    args = ap.parse_args()

    baseline = load_scores(Path(args.baseline))
    few_shot = load_scores(Path(args.few_shot))
    cot = load_scores(Path(args.cot))

    groups_bl_fs = classify(baseline, few_shot)
    groups_fs_cot = classify(few_shot, cot)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    render(
        [
            ("Baseline vs few-shot", groups_bl_fs, len(baseline)),
            ("Few-shot vs few-shot CoT", groups_fs_cot, len(few_shot)),
        ],
        out_dir / "reproducibility_groups_bl_fs_cot.png",
    )

    summary = {
        "Baseline vs few-shot": {CATEGORY_LABEL[k]: len(v) for k, v in groups_bl_fs.items()},
        "Few-shot vs few-shot CoT": {CATEGORY_LABEL[k]: len(v) for k, v in groups_fs_cot.items()},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[reproducibility-groups] 完成，输出于 {out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
