"""Supplementary comparison figure for preventive defenses (results specified manually; may mix sampling settings).

Background: the defense[preventive] grouped figures in compare.py align on the same sampling setting, so one figure
can only contain baseline/delimiters/instructional/paraphrasing/sandwich under tem_0.6_top_p_0.9_top_k_100;
retokenization was run at tem_0.6_top_p_0.95_top_k_20 and is therefore excluded.

This script is an **independent supplement**: it manually lists the results to compare (each may use different sampling) and redraws two figures —
  - codebleu_components.png  grouped bar chart of the 4 components (same colors as compare.py, with std)
  - codebleu.png            bar chart of the overall CodeBLEU score (with std + value labels)

It only **reads** eval outputs (summary.json), does not modify compare.py or overwrite runs_v2/comparisons,
and writes to runs_v2/comparisons_supplement/ by default. Entries with a different sampling setting are marked in the x-axis labels.

Usage:
    python -m src.pipeline_v2.compare_supplement
    python -m src.pipeline_v2.compare_supplement --eval-root runs_v2/evaluations \
        --out runs_v2/comparisons_supplement
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

# Repository root (derived from the script's own location: src/pipeline_v2/compare_supplement.py → two levels up).
# Default paths are anchored to it, so results are found even when run from an IDE / any directory with a different CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The 4 CodeBLEU components (key prefix + display name + hatch pattern). White fill with black edges, distinguished by hatch rather than color.
CODEBLEU_COMPONENTS: List[Tuple[str, str, str]] = [
    ("ngram_match_score", "N-gram Match", "///"),
    ("weighted_ngram_match_score", "Weighted N-gram Match", "\\\\\\"),
    ("syntax_match_score", "Syntax Match", "xxx"),
    ("dataflow_match_score", "Dataflow Match", "..."),
]

# Reference sampling setting: entries with the same setting show only the defense name on the x-axis; others are additionally labeled with their sampling setting.
BASE_SAMPLING = "tem_0.6_top_p_0.9_top_k_100"

# Entries to compare: (display name, sub-path of summary.json relative to eval_root).
# The order is the x-axis order; the first 5 are the preventive defenses in the two compare.py figures, followed by retokenization (different sampling setting).
# All use codebleu_no_docstring_summary.json (computed after stripping docstrings/comments), consistent with the other comparison figures.
MODEL_DIR = "humaneval/qwen3_coder_next"
ENTRIES: List[Tuple[str, str]] = [
    ("baseline", f"{MODEL_DIR}/humaneval_original/{BASE_SAMPLING}/codebleu_no_docstring_summary.json"),
    ("delimiters", f"{MODEL_DIR}/defense/delimiters/{BASE_SAMPLING}/codebleu_no_docstring_summary.json"),
    ("instructional", f"{MODEL_DIR}/defense/instructional/{BASE_SAMPLING}/codebleu_no_docstring_summary.json"),
    ("paraphrasing", f"{MODEL_DIR}/defense/paraphrasing/{BASE_SAMPLING}/codebleu_no_docstring_summary.json"),
    ("sandwich", f"{MODEL_DIR}/defense/sandwich/{BASE_SAMPLING}/codebleu_no_docstring_summary.json"),
    ("retokenization", f"{MODEL_DIR}/defense/retokenization/tem_0.6_top_p_0.95_top_k_20/codebleu_no_docstring_summary.json"),
]

# Figure title: only "defense strategies in qwen3", without any sampling parameters (in English to avoid missing CJK fonts).
TITLE_PREFIX = "Defense strategies in qwen3"


def load_row(eval_root: Path, label: str, rel_path: str) -> Optional[Dict[str, Any]]:
    sp = eval_root / rel_path
    if not sp.exists():
        # Fault tolerance: if rel_path omits the sampling-setting subdirectory, look for summary.json one level down.
        base = (eval_root / rel_path).parent
        cands = sorted(base.glob("**/summary.json")) if base.exists() else []
        if not cands:
            print(f"[supplement] 缺失，跳过：{sp}")
            return None
        sp = cands[0]
        if len(cands) > 1:
            print(f"[supplement] {label}: 该目录有多个采样档，自动选用 {sp}")
    summary = json.loads(sp.read_text(encoding="utf-8"))
    # codebleu_no_docstring_summary.json has a flat structure (no metrics/run wrapper);
    # summary.json has a nested metrics/run structure. Both are supported.
    metrics = summary["metrics"] if "metrics" in summary else summary
    sampling = summary.get("run", {}).get("sampling_token", "") or sp.parent.name
    # x-axis labels use only the defense name (label), without sampling parameters. sampling is kept only in the CSV for reference.
    return {
        "label": label,
        "xlabel": label,
        "metrics": metrics,
        "sampling": sampling,
    }


def render_components(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    labels = [r["xlabel"] for r in rows]
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
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"{TITLE_PREFIX} — CodeBLEU Component Scores", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax.yaxis.grid(True, ls="--", linewidth=0.6, color="#cccccc")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "codebleu_components.png", dpi=150, facecolor="white")
    plt.close(fig)


def render_codebleu(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    labels = [r["xlabel"] for r in rows]
    vals = [float(r["metrics"]["codebleu_mean"]) for r in rows]
    errs = [float(r["metrics"].get("codebleu_std") or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels) + 2), 4.5))
    bars = ax.bar(range(len(labels)), vals, yerr=errs if any(errs) else None,
                  capsize=4, error_kw={"elinewidth": 1, "ecolor": "black"},
                  facecolor="white", edgecolor="black", linewidth=1, hatch="///")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel("CodeBLEU")
    ax.set_xlabel("Defense Strategies")
    ax.set_title(f"{TITLE_PREFIX} — CodeBLEU", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.yaxis.grid(True, ls="--", linewidth=0.6, color="#cccccc")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "codebleu.png", dpi=150, facecolor="white")
    plt.close(fig)


def write_csv(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    metric_keys = ["codebleu"] + [k for k, _, _ in CODEBLEU_COMPONENTS]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["defense", "sampling"] + [f"{k}(mean±std)" for k in metric_keys])
        for r in rows:
            m = r["metrics"]
            cells = [r["label"], r["sampling"]]
            for k in metric_keys:
                mean, std = m.get(f"{k}_mean"), m.get(f"{k}_std")
                cells.append("" if mean is None else f"{mean:.4f}±{(std or 0.0):.4f}")
            w.writerow(cells)


def main() -> None:
    ap = argparse.ArgumentParser(description="预防型防御 CodeBLEU 对比补充图（含 retokenization）。")
    # Anchored to the repository root by default (absolute path); explicitly passed relative paths are still resolved against CWD.
    ap.add_argument("--eval-root", default=str(REPO_ROOT / "runs_v2" / "evaluations"))
    ap.add_argument("--out", default=str(
        REPO_ROOT / "runs_v2" / "comparisons_supplement" / "qwen3_preventive_with_retok"))
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    out_dir = Path(args.out)

    rows = []
    for label, rel_path in ENTRIES:
        row = load_row(eval_root, label, rel_path)
        if row is not None:
            rows.append(row)
    if len(rows) < 2:
        print("[supplement] 有效结果不足 2 个，无法对比。")
        return

    # Create the output directory only when there are results, to avoid leaving empty folders after failed runs.
    out_dir.mkdir(parents=True, exist_ok=True)
    render_components(rows, out_dir)
    render_codebleu(rows, out_dir)
    write_csv(rows, out_dir)
    print(f"[supplement] 完成：{len(rows)} 个结果 → 2 张图 + summary.csv，输出于 {out_dir}")


if __name__ == "__main__":
    main()
