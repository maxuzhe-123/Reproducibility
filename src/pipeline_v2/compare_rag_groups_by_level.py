"""Grouped figure of the direction of change in the four metrics after introducing RAG, split by context dependency level (CoderEval / qwen3_coder_next).

Compares codereval_raw (no RAG) → codereval (with RAG):
for each of the 6 CoderEval dependency levels (self_contained → project_runnable, weak → strong), tasks are grouped
by the direction of change of the four codebleu components (ngram/weighted_ngram/syntax/dataflow) into
all up / all down / mixed / no change, and the number and proportion of tasks in each group are counted per level.

Only **reads** codebleu_no_docstring.jsonl (which already contains the level field); does not modify any existing outputs.

Usage:
    python -m src.pipeline_v2.compare_rag_groups_by_level
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

LEVEL_ORDER = (
    "self_contained", "slib_runnable", "plib_runnable",
    "class_runnable", "file_runnable", "project_runnable",
)

CATEGORY_ORDER = ["fully_increase", "fully_decrease", "mixed", "no_change"]
CATEGORY_LABEL = {"fully_increase": "All Up", "fully_decrease": "All Down", "mixed": "Mixed", "no_change": "No Change"}
CATEGORY_HATCH = {"fully_increase": "//", "fully_decrease": "xx", "mixed": "\\\\", "no_change": ".."}


def load_scores(path: Path) -> Dict[str, Dict[str, float]]:
    """Use the hash at the end of task_id as the key (task_id prefixes may differ across experiments, e.g.,
    ``CoderEval/<hash>`` vs ``<repo-name>/<hash>``)."""
    d: Dict[str, Dict[str, float]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row["task_id"].rsplit("/", 1)[-1]
            d[key] = row
    return d


def classify(ref: Dict[str, Dict[str, float]], treat: Dict[str, Dict[str, float]], task_ids: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {k: [] for k in CATEGORY_ORDER}
    for task_id in task_ids:
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
    n = len(panels)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.8 * nrows), squeeze=False)
    axes = axes.flatten()
    # The task counts in the "ALL" summary panel are much larger than in the stratified panels, so it is scaled separately to avoid flattening them.
    level_panels = [p for p in panels if p[0] != "ALL"]
    y_max_level = max(len(groups[k]) for _, groups, _ in level_panels for k in CATEGORY_ORDER)
    for ax, (title, groups, total) in zip(axes, panels):
        labels = [CATEGORY_LABEL[k] for k in CATEGORY_ORDER]
        counts = [len(groups[k]) for k in CATEGORY_ORDER]
        hatches = [CATEGORY_HATCH[k] for k in CATEGORY_ORDER]
        bars = ax.bar(labels, counts, color="white", edgecolor="black", linewidth=0.8)
        for b, hatch in zip(bars, hatches):
            b.set_hatch(hatch)
        for b, n_ in zip(bars, counts):
            pct = n_ / total * 100 if total else 0.0
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{n_}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=18)
        ax.set_title(f"{title} (n={total})", fontsize=19)
        ax.set_ylabel("# of tasks")
        y_max = max(counts) * 1.28 if title == "ALL" else y_max_level * 1.28
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.tick_params(axis="x", labelsize=15)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Codereval Raw vs Codereval RAG", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 引入（codereval_raw vs codereval）按依赖等级拆分的四指标变化方向分组")
    model_dir = "codereval/qwen3_coder_next"
    sampling = "tem_0.6_top_p_0.9_top_k_100"
    ap.add_argument("--raw", default=str(
        REPO_ROOT / "runs_v2/evaluations" / model_dir / "codereval_raw" / sampling / "codebleu_no_docstring.jsonl"))
    ap.add_argument("--rag", default=str(
        REPO_ROOT / "runs_v2/evaluations" / model_dir / "codereval" / sampling / "codebleu_no_docstring.jsonl"))
    ap.add_argument("--out", default=str(
        REPO_ROOT / "runs_v2/comparisons_supplement/rag_groups_by_level_qwen3"))
    args = ap.parse_args()

    raw = load_scores(Path(args.raw))
    rag = load_scores(Path(args.rag))

    common_ids = sorted(set(raw) & set(rag))
    levels_of = {tid: raw[tid].get("level") for tid in common_ids}

    panels = []
    summary = {}
    for level in LEVEL_ORDER:
        ids = [tid for tid in common_ids if levels_of[tid] == level]
        if not ids:
            continue
        groups = classify(raw, rag, ids)
        panels.append((level, groups, len(ids)))
        summary[level] = {"n": len(ids), **{CATEGORY_LABEL[k]: len(v) for k, v in groups.items()}}

    # The full set (not stratified) as a reference
    groups_all = classify(raw, rag, common_ids)
    summary["ALL"] = {"n": len(common_ids), **{CATEGORY_LABEL[k]: len(v) for k, v in groups_all.items()}}
    panels.append(("ALL", groups_all, len(common_ids)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    render(panels, out_dir / "rag_groups_by_level.png")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[rag-groups-by-level] 完成，输出于 {out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
