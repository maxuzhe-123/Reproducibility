"""CodeJudgeBench RQ5 — bar charts of ReproCorrect / Repro @A vs @B (overall + by difficulty).

Reads the following two metric families from the codejudge eval summary.json and draws one grouped bar chart for each;
4 groups on the x-axis: Overall / Easy / Medium / Hard, each with two bars (@A / @B), std error bars and value labels:

  ReproCorrect@A / @B   mean per-instance "correct count / 10" (good in position A / B)
  Repro@A / @B          mean per-instance "majority count / 10" (position A / B)

Usage:
    python -m src.pipeline_v2.plot_codejudge_position_bias \
        --summary runs_v2/evaluations/codejudgebench/qwen3_coder_next/codejudgebench/tem_0_top_p_0.95_top_k_20/summary.json

Without --summary the default path is used. Figures and CSV are written by default to
runs_v2/comparisons_supplement/codejudge_position_bias/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "runs_v2/evaluations/codejudgebench/qwen3_coder_next/codejudgebench"
    / "tem_0_top_p_0.95_top_k_20/summary.json"
)
DEFAULT_OUT = REPO_ROOT / "runs_v2/comparisons_supplement/codejudge_position_bias"

# (display name, key in by_difficulty); None = overall (taken from metrics).
GROUPS = [("Overall", None), ("Easy", "easy"), ("Medium", "medium"), ("Hard", "hard")]

# Two metric families: (file name prefix, figure title, @A key, @B key, Rate key, y-axis label)
FAMILIES = [
    (
        "reprocorrect",
        "ReproCorrect@A vs @B",
        "ReproCorrect@A",
        "ReproCorrect@B",
        "ReproCorrect_Rate",
        "ReproCorrect (correct rate)",
    ),
    (
        "repro",
        "Repro@A vs @B",
        "Repro@A",
        "Repro@B",
        "Repro_Rate",
        "Repro (majority agreement)",
    ),
]


def _node(summary: Dict[str, Any], diff_key: str | None) -> Dict[str, Any]:
    return summary["metrics"] if diff_key is None else summary["by_difficulty"][diff_key]


def collect(summary: Dict[str, Any], a_key: str, b_key: str, rate_key: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, diff_key in GROUPS:
        node = _node(summary, diff_key)
        rows.append(
            {
                "group": name,
                "a": float(node.get(a_key, 0.0)),
                "b": float(node.get(b_key, 0.0)),
                "rate": float(node.get(rate_key, 0.0)),
                "a_std": float(node.get(a_key + "_std", 0.0)),
                "b_std": float(node.get(b_key + "_std", 0.0)),
                "rate_std": float(node.get(rate_key + "_std", 0.0)),
            }
        )
    return rows


def render(rows: List[Dict[str, Any]], out_path: Path, title: str, ylabel: str,
           a_label: str, b_label: str) -> None:
    labels = [r["group"] for r in rows]
    a_vals = [r["a"] for r in rows]
    b_vals = [r["b"] for r in rows]
    a_err = [r["a_std"] for r in rows]
    b_err = [r["b_std"] for r in rows]

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(labels) + 2), 5.2))

    bars_a = ax.bar([i - width / 2 for i in x], a_vals, width, yerr=a_err, capsize=4,
                    label=a_label, color="#4C72B0", error_kw={"elinewidth": 1, "alpha": 0.7})
    bars_b = ax.bar([i + width / 2 for i in x], b_vals, width, yerr=b_err, capsize=4,
                    label=b_label, color="#DD8452", error_kw={"elinewidth": 1, "alpha": 0.7})

    for bars, vals, errs in ((bars_a, a_vals, a_err), (bars_b, b_vals, b_err)):
        for rect, v, e in zip(bars, vals, errs):
            ax.text(rect.get_x() + rect.get_width() / 2, v + e + 0.012,
                    f"{v:.3f}\n±{e:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.45)  # The std error bars are long; leave space at the top to avoid overlapping the title/labels
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title(f"CodeJudgeBench {title} (qwen3-coder-next)", pad=14)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv(summary: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "repro_metrics_by_difficulty.csv"
    cols = [
        "ReproCorrect@A", "ReproCorrect@B", "ReproCorrect_Rate",
        "Repro@A", "Repro@B", "Repro_Rate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["group"]
        for c in cols:
            header += [c, c + "_std"]
        w.writerow(header)
        for name, diff_key in GROUPS:
            node = _node(summary, diff_key)
            row = [name]
            for c in cols:
                row += [node.get(c, 0.0), node.get(c + "_std", 0.0)]
            w.writerow(row)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out_dir = Path(args.out)

    for prefix, title, a_key, b_key, rate_key, ylabel in FAMILIES:
        rows = collect(summary, a_key, b_key, rate_key)
        png = out_dir / f"{prefix}_at_ab_by_difficulty.png"
        render(rows, png, title, ylabel, a_key, b_key)
        print(f"[{title}]")
        for r in rows:
            print(f"  {r['group']:8s} @A={r['a']:.4f}±{r['a_std']:.4f}  "
                  f"@B={r['b']:.4f}±{r['b_std']:.4f}  Rate={r['rate']:.4f}")
        print(f"  PNG: {png}")

    csv_path = write_csv(summary, out_dir)
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
