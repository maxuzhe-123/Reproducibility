"""CodeBLEU four-dimension comparison of three Gemini models (single figure).

Under fixed conditions tem=0.6 / thinking_1024 and the baseline template (humaneval_original), draws the 4 CodeBLEU
components of gemini-2.5-flash / flash-lite / pro in one grouped bar chart
(different colors = different dimensions, with std error bars).

Only **reads** eval outputs (summary.json); does not modify other scripts or overwrite existing results.

Usage (can be run from any directory):
    python -m src.pipeline_v2.compare_gemini_models
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

REPO_ROOT = Path(__file__).resolve().parents[2]

# The 4 CodeBLEU components (key prefix + display name + color), with colors consistent with the other comparison figures.
CODEBLEU_COMPONENTS: List[Tuple[str, str, str]] = [
    ("ngram_match_score", "ngram_match", "#DD8452"),
    ("weighted_ngram_match_score", "weighted_ngram_match", "#55A868"),
    ("syntax_match_score", "syntax_match", "#C44E52"),
    ("dataflow_match_score", "dataflow_match", "#8172B3"),
]

SAMPLING = "tem_0.6_thinking_1024"

# The three Gemini models to compare: (x-axis display name, model directory name in eval). The order is the x-axis order.
MODELS: List[Tuple[str, str]] = [
    ("gemini-2.5-flash-lite", "gemini_2.5_flash_lite"),
    ("gemini-2.5-flash", "gemini_2.5_flash"),
    ("gemini-2.5-pro", "gemini_2.5_pro"),
]

TITLE = "Gemini models — CodeBLEU dimensions"
IMG_NAME = "gemini models codebleu dimensions.png"


def load_row(eval_root: Path, disp: str, model_dir: str) -> Optional[Dict[str, Any]]:
    sp = eval_root / "humaneval" / model_dir / "humaneval_original" / SAMPLING / "summary.json"
    if not sp.exists():
        print(f"[gemini] 缺失，跳过：{sp}")
        return None
    summary = json.loads(sp.read_text(encoding="utf-8"))
    return {"label": disp, "metrics": summary.get("metrics", {})}


def render(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    labels = [r["label"] for r in rows]
    n = len(CODEBLEU_COMPONENTS)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(labels) + 2), 4.8))
    for i, (key, disp, color) in enumerate(CODEBLEU_COMPONENTS):
        vals = [float(r["metrics"][f"{key}_mean"]) for r in rows]
        errs = [float(r["metrics"].get(f"{key}_std") or 0.0) for r in rows]
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(labels))]
        ax.bar(xs, vals, width=width, yerr=errs if any(errs) else None,
               capsize=3, label=disp, color=color, alpha=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel("score")
    ax.set_xlabel("Gemini model")
    ax.set_title(TITLE, fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / IMG_NAME, dpi=150)
    plt.close(fig)


def write_csv(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = ["codebleu"] + [k for k, _, _ in CODEBLEU_COMPONENTS]
        w.writerow(["model", "sampling"] + [f"{k}(mean±std)" for k in keys])
        for r in rows:
            m = r["metrics"]
            cells = [r["label"], SAMPLING]
            for k in keys:
                mean, std = m.get(f"{k}_mean"), m.get(f"{k}_std")
                cells.append("" if mean is None else f"{mean:.4f}±{(std or 0.0):.4f}")
            w.writerow(cells)


def main() -> None:
    ap = argparse.ArgumentParser(description="三个 Gemini 模型 CodeBLEU 四维度对比（tem_0.6/thinking_1024）。")
    ap.add_argument("--eval-root", default=str(REPO_ROOT / "runs_v2" / "evaluations"))
    ap.add_argument("--out", default=str(
        REPO_ROOT / "runs_v2" / "comparisons_supplement" / "gemini_models_codebleu_thinking1024"))
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    out_dir = Path(args.out)

    rows = [r for disp, d in MODELS if (r := load_row(eval_root, disp, d)) is not None]
    if len(rows) < 2:
        print("[gemini] 有效结果不足 2 个，无法对比。")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    render(rows, out_dir)
    write_csv(rows, out_dir)
    print(f"[gemini] 完成：{len(rows)} 个模型 → 1 张图 + summary.csv，输出于 {out_dir}")


if __name__ == "__main__":
    main()
