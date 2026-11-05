"""Plot the CodeBLEU component grouped bar chart for gemini_2.5_flash @ humaneval_original
under the four thinking levels 0/1024/2048/4096 (excluding the top_p/top_k setting).

The style follows render_codebleu_components_group in src/pipeline_v2/compare.py:
the x-axis is the thinking level, with 4 bars per level = ngram / weighted_ngram / syntax / dataflow,
with std error bars; parameter settings are likewise not shown on the figure.
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

EVAL_ROOT = Path(
    "runs_v2/evaluations/humaneval/gemini_2.5_flash/humaneval_original"
)
# Naming follows the comparisons_supplement convention: snake_case subdirectory +
# "<subject> - components by <dimension>.png" + a CSV with the same name.
OUT_DIR = Path(
    "runs_v2/comparisons_supplement/gemini_flash_components_by_thinking"
)
OUT_PATH = OUT_DIR / "gemini_2.5_flash - components by thinking.png"
CSV_PATH = OUT_DIR / "gemini_2.5_flash.csv"

# Component definitions identical to compare.py (key, display name, color)
CODEBLEU_COMPONENTS = [
    ("ngram_match_score", "ngram_match", "#DD8452"),
    ("weighted_ngram_match_score", "weighted_ngram_match", "#55A868"),
    ("syntax_match_score", "syntax_match", "#C44E52"),
    ("dataflow_match_score", "dataflow_match", "#8172B3"),
]

# Only these four thinking levels, in a fixed order
CONFIGS = [
    ("tem_0.6_thinking_0", "ThinkingBudget=0"),
    ("tem_0.6_thinking_1024", "ThinkingBudget=1024"),
    ("tem_0.6_thinking_2048", "ThinkingBudget=2048"),
    ("tem_0.6_thinking_4096", "ThinkingBudget=4096"),
]


def main() -> None:
    labels = []
    series = {key: [] for key, _, _ in CODEBLEU_COMPONENTS}
    errs = {key: [] for key, _, _ in CODEBLEU_COMPONENTS}
    csv_rows = []

    for sub, xlab in CONFIGS:
        summary_path = EVAL_ROOT / sub / "summary.json"
        if not summary_path.exists():
            print(f"[WARN] missing {summary_path}, skipped")
            continue
        m = json.loads(summary_path.read_text()).get("metrics", {})
        vals = [m.get(f"{key}_mean") for key, _, _ in CODEBLEU_COMPONENTS]
        if any(v is None for v in vals):
            print(f"[WARN] {sub} missing some component means, skipped")
            continue
        labels.append(xlab)
        row = {"thinking": xlab}
        for (key, disp, _), v in zip(CODEBLEU_COMPONENTS, vals):
            std = float(m.get(f"{key}_std") or 0.0)
            series[key].append(float(v))
            errs[key].append(std)
            row[disp] = f"{float(v):.4f}±{std:.4f}"
        csv_rows.append(row)

    if len(labels) < 2:
        raise SystemExit("need at least 2 configs to plot")

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels) + 2), 4.8))
    n = len(CODEBLEU_COMPONENTS)
    width = 0.8 / n
    for i, (key, disp, color) in enumerate(CODEBLEU_COMPONENTS):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(labels))]
        ax.bar(xs, series[key], width=width,
               yerr=errs[key] if any(errs[key]) else None, capsize=3,
               label=disp, color=color, alpha=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
    ax.set_ylabel("Score")
    ax.set_xlabel("Thinking Budget")
    ax.set_title(
        "Different Thinking Budget in Gemini_2.5_flash "
        "— CodeBLEU components",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)

    # CSV with the same name (consistent with the other comparisons_supplement experiments)
    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["thinking"] + [disp for _, disp, _ in CODEBLEU_COMPONENTS],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"saved: {OUT_PATH}")
    print(f"saved: {CSV_PATH}")


if __name__ == "__main__":
    main()
