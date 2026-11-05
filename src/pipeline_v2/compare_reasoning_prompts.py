"""Comparison figure of reasoning prompt strategies (baseline / few-shot / cot; may mix sampling settings).

From the prompt comparison, only baseline (humaneval_original) and few-shot (humaneval_few_shot) are taken,
plus cot (humaneval_cot @ tem_0.6_top_p_0.95_top_k_20), producing two figures:
  - reasoning prompt strategies in qwen3 - components.png   grouped bar chart of the 4 CodeBLEU components
  - reasoning prompt strategies in qwen3 - codebleu.png     bar chart of the overall CodeBLEU score

Only **reads** eval outputs (summary.json); does not modify compare.py / compare_supplement.py
or overwrite existing results; sampling parameters are not shown in the figures.

Usage (can be run from any directory):
    python -m src.pipeline_v2.compare_reasoning_prompts
    python src/pipeline_v2/compare_reasoning_prompts.py --eval-root ... --out ...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"
matplotlib.rcParams["hatch.linewidth"] = 1.2

# Repository root (derived from the script's own location); default paths are anchored to it to avoid inconsistent CWD when run from an IDE/any directory.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The 4 CodeBLEU components (key prefix + display name + hatch). White fill with black edges, distinguished by hatch rather than color.
CODEBLEU_COMPONENTS: List[Tuple[str, str, str]] = [
    ("ngram_match_score", "N-gram Match", "///"),
    ("weighted_ngram_match_score", "Weighted N-gram Match", "\\\\\\"),
    ("syntax_match_score", "Syntax Match", "xxx"),
    ("dataflow_match_score", "Dataflow Match", "..."),
]

# Preset entries per model: (x-axis display name, sub-path of summary.json relative to eval_root). The order is the x-axis order.
PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen3": {
        "entries": [
            ("Zero-shot", "humaneval/qwen3_coder_next/humaneval_original/tem_0.6_top_p_0.9_top_k_100/codebleu_no_docstring_summary.json"),
            ("Few-shot", "humaneval/qwen3_coder_next/humaneval_few_shot/tem_0.6_top_p_0.9_top_k_100/codebleu_no_docstring_summary.json"),
            ("CoT", "humaneval/qwen3_coder_next/humaneval_cot/tem_0.6_top_p_0.95_top_k_20/codebleu_no_docstring_summary.json"),
        ],
        "img_prefix": "reasoning prompt strategies in qwen3",
        "out_dir": "reasoning_prompt_strategies_qwen3",
        "model_title": "Qwen3 Coder Next",
    },
    "gemini25flash": {
        "entries": [
            ("Zero-shot", "humaneval/gemini_2.5_flash/humaneval_original/tem_0.6_thinking_1024/codebleu_no_docstring_summary.json"),
            ("Few-shot", "humaneval/gemini_2.5_flash/humaneval_few_shot/tem_0.6_thinking_1024/codebleu_no_docstring_summary.json"),
            ("CoT", "humaneval/gemini_2.5_flash/humaneval_cot/tem_0.6_thinking_1024/codebleu_no_docstring_summary.json"),
        ],
        "img_prefix": "reasoning prompt strategies in gemini 2.5 flash",
        "out_dir": "reasoning_prompt_strategies_gemini25flash",
        "model_title": "Gemini 2.5 Flash",
    },
}

DATASET_TITLE = "HumanEval"                    # Dataset name in the subtitle (without sampling parameters)


def load_row(eval_root: Path, label: str, rel_path: str) -> Optional[Dict[str, Any]]:
    sp = eval_root / rel_path
    file_name = Path(rel_path).name
    if not sp.exists():
        # Fault tolerance: if the path omits the sampling-setting subdirectory, look for a summary file with the same name one level down.
        base = (eval_root / rel_path).parent
        cands = sorted(base.glob(f"**/{file_name}")) if base.exists() else []
        if not cands:
            print(f"[reasoning] 缺失，跳过：{sp}")
            return None
        sp = cands[0]
        if len(cands) > 1:
            print(f"[reasoning] {label}: 该目录有多个采样档，自动选用 {sp}")
    summary = json.loads(sp.read_text(encoding="utf-8"))
    # codebleu_no_docstring_summary.json has a flat structure (no metrics/run wrapper);
    # summary.json has a nested metrics/run structure. Both are supported.
    metrics = summary["metrics"] if "metrics" in summary else summary
    sampling = summary.get("run", {}).get("sampling_token", "") or sp.parent.name
    return {
        "label": label,
        "metrics": metrics,
        "sampling": sampling,
    }


def render_components(rows: List[Dict[str, Any]], out_dir: Path, img_prefix: str, model_title: str) -> None:
    labels = [r["label"] for r in rows]
    n = len(CODEBLEU_COMPONENTS)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (key, disp, hatch) in enumerate(CODEBLEU_COMPONENTS):
        vals = [float(r["metrics"][f"{key}_mean"]) for r in rows]
        errs = [float(r["metrics"].get(f"{key}_std") or 0.0) for r in rows]
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(labels))]
        ax.bar(xs, vals, width=width, yerr=errs if any(errs) else None,
               capsize=3, error_kw={"elinewidth": 1, "ecolor": "black"},
               label=disp, facecolor="white", edgecolor="black",
               linewidth=1, hatch=hatch)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=16)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"CodeBLEU Component Scores by Prompt Strategy\n({model_title}, {DATASET_TITLE})", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax.yaxis.grid(True, ls="--", linewidth=0.6, color="#cccccc")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{img_prefix} - components.png", dpi=150, facecolor="white")
    plt.close(fig)


def render_codebleu(rows: List[Dict[str, Any]], out_dir: Path, img_prefix: str, model_title: str) -> None:
    labels = [r["label"] for r in rows]
    vals = [float(r["metrics"]["codebleu_mean"]) for r in rows]
    errs = [float(r["metrics"].get("codebleu_std") or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels) + 2), 4.5))
    bars = ax.bar(range(len(labels)), vals, yerr=errs if any(errs) else None,
                  capsize=4, error_kw={"elinewidth": 1, "ecolor": "black"},
                  facecolor="white", edgecolor="black", linewidth=1, hatch="///")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=14)
    ax.set_ylabel("CodeBLEU", fontsize=13)
    ax.set_title(f"CodeBLEU Score by Prompt Strategy\n({model_title}, {DATASET_TITLE})", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=14)
    ax.yaxis.grid(True, ls="--", linewidth=0.6, color="#cccccc")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{img_prefix} - codebleu.png", dpi=150, facecolor="white")
    plt.close(fig)


def write_csv(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    metric_keys = ["codebleu"] + [k for k, _, _ in CODEBLEU_COMPONENTS]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "sampling"] + [f"{k}(mean±std)" for k in metric_keys])
        for r in rows:
            m = r["metrics"]
            cells = [r["label"], r["sampling"]]
            for k in metric_keys:
                mean, std = m.get(f"{k}_mean"), m.get(f"{k}_std")
                cells.append("" if mean is None else f"{mean:.4f}±{(std or 0.0):.4f}")
            w.writerow(cells)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reasoning prompt 策略 CodeBLEU 对比图（baseline/few-shot/cot）。")
    ap.add_argument("--model", choices=sorted(PRESETS), default="qwen3")
    ap.add_argument("--eval-root", default=str(REPO_ROOT / "runs_v2" / "evaluations"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    preset = PRESETS[args.model]
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "runs_v2" / "comparisons_supplement" / preset["out_dir"])

    rows = []
    for label, rel_path in preset["entries"]:
        row = load_row(eval_root, label, rel_path)
        if row is not None:
            rows.append(row)
    if len(rows) < 2:
        print("[reasoning] 有效结果不足 2 个，无法对比。")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    render_components(rows, out_dir, preset["img_prefix"], preset["model_title"])
    render_codebleu(rows, out_dir, preset["img_prefix"], preset["model_title"])
    write_csv(rows, out_dir)
    print(f"[reasoning] 完成：{len(rows)} 个结果 → 2 张图 + summary.csv，输出于 {out_dir}")


if __name__ == "__main__":
    main()
