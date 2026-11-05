"""Figure of the four CodeBLEU components per CoderEval dependency level (one figure per experiment; title without sampling parameters).

Replicates the codebleu_by_level components grouped figure in compare.py, but:
  - removes the sampling setting (tem_..._top_k_...) from the title, keeping only the experiment name + metric description;
  - draws a separate figure for each specified experiment.

Only **reads** eval outputs (codebleu_by_level in summary.json); does not modify other scripts or overwrite existing results.

Usage (can be run from any directory):
    python -m src.pipeline_v2.compare_codereval_levels
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]

# Dependency level order (weak → strong), consistent with CODEREVAL_LEVEL_ORDER in compare.py.
LEVEL_ORDER = (
    "self_contained", "slib_runnable", "plib_runnable",
    "class_runnable", "file_runnable", "project_runnable",
)

# The 4 CodeBLEU components (key prefix + display name + color), consistent with the stratified figure colors in compare.py.
COMPONENTS: List[Tuple[str, str, str]] = [
    ("ngram_match_score", "ngram", "#DD8452"),
    ("weighted_ngram_match_score", "weighted_ngram", "#55A868"),
    ("syntax_match_score", "syntax", "#C44E52"),
    ("dataflow_match_score", "dataflow", "#8172B3"),
]

SAMPLING = "tem_0.6_top_p_0.9_top_k_100"
MODEL_DIR = "codereval/qwen3_coder_next"

# Experiments to draw separately: (display name used in titles/file names, experiment subdirectory name in eval).
EXPERIMENTS: List[Tuple[str, str]] = [
    ("codereval", "codereval"),
    ("codereval_raw", "codereval_raw"),
]


def load_levels(eval_root: Path, sub: str) -> Dict[str, Any] | None:
    sp = eval_root / MODEL_DIR / sub / SAMPLING / "summary.json"
    if not sp.exists():
        print(f"[codereval] 缺失，跳过：{sp}")
        return None
    cbl = json.loads(sp.read_text(encoding="utf-8")).get("codebleu_by_level") or {}
    return cbl or None


def render(disp: str, cbl: Dict[str, Any], out_dir: Path) -> None:
    levels = [lv for lv in LEVEL_ORDER if lv in cbl] + [lv for lv in cbl if lv not in LEVEL_ORDER]
    n = len(COMPONENTS)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(levels) + 2), 4.8))
    for i, (key, label, color) in enumerate(COMPONENTS):
        vals = [cbl[lv].get(f"{key}_mean") for lv in levels]
        errs = [float(cbl[lv].get(f"{key}_std") or 0.0) for lv in levels]
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(levels))]
        ax.bar(xs, vals, width=width, yerr=errs if any(errs) else None,
               capsize=3, label=label, color=color, alpha=0.9)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("score")
    ax.set_xlabel("dependency level (weak → strong)")
    ax.set_title(f"{disp} — CodeBLEU components by level", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"{disp} - components by level.png", dpi=150)
    plt.close(fig)


def write_csv(disp: str, cbl: Dict[str, Any], out_dir: Path) -> None:
    levels = [lv for lv in LEVEL_ORDER if lv in cbl] + [lv for lv in cbl if lv not in LEVEL_ORDER]
    with (out_dir / f"{disp}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "count"] + [lbl for _, lbl, _ in COMPONENTS])
        for lv in levels:
            stats = cbl[lv]
            row = [lv, stats.get("count")]
            for key, _, _ in COMPONENTS:
                mean, std = stats.get(f"{key}_mean"), stats.get(f"{key}_std")
                row.append("" if mean is None else f"{mean:.4f}±{(std or 0.0):.4f}")
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="CodeReval 分层 CodeBLEU 四分量图（每实验单独一张，标题不含参数）。")
    ap.add_argument("--eval-root", default=str(REPO_ROOT / "runs_v2" / "evaluations"))
    ap.add_argument("--out", default=str(
        REPO_ROOT / "runs_v2" / "comparisons_supplement" / "codereval_components_by_level"))
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    out_dir = Path(args.out)

    drawn = 0
    pending = []
    for disp, sub in EXPERIMENTS:
        cbl = load_levels(eval_root, sub)
        if cbl is not None:
            pending.append((disp, cbl))
    if not pending:
        print("[codereval] 没有可用的 codebleu_by_level 结果。")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for disp, cbl in pending:
        render(disp, cbl, out_dir)
        write_csv(disp, cbl, out_dir)
        drawn += 1
    print(f"[codereval] 完成：{drawn} 个实验 → 各 1 张图 + csv，输出于 {out_dir}")


if __name__ == "__main__":
    main()
