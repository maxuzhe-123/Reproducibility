"""Render a wilcoxon_compare.py CSV output as a booktabs-style table image
(matching the paper's "Table N. Wilcoxon signed-rank test results ..." format).

Usage:
    python scripts/render_wilcoxon_table.py \\
        --csv runs_v2/evaluations/humaneval/<model>/_wilcoxon_prompting_strategy/wilcoxon_results.csv \\
        --out runs_v2/evaluations/humaneval/<model>/wilcoxon_prompting_strategy_table.png \\
        --caption "Wilcoxon signed-rank test results for prompting strategy comparisons (<model>)." \\
        --comparison "few_shot vs baseline=Baseline vs Few-shot" \\
        --comparison "cot vs few_shot=Few-shot vs CoT"

--comparison ORIGINAL_LABEL=DISPLAY_LABEL controls which "comparison" rows (as
written by wilcoxon_compare.py) are included and how they're captioned, in order.
"""
from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt

METRIC_LABELS = {
    "codebleu": "CodeBLEU",
    "ngram_match_score": "N-gram",
    "weighted_ngram_match_score": "Weighted N-gram",
    "syntax_match_score": "Syntax",
    "dataflow_match_score": "Data-flow",
}
METRIC_ORDER = list(METRIC_LABELS.keys())


def fmt_p(p: float) -> str:
    if p < 1e-3:
        mantissa, exp = f"{p:.3e}".split("e")
        exp = int(exp)
        return f"${float(mantissa):.3f} \\times 10^{{{exp}}}$*"
    return f"{p:.5g}*"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="wilcoxon_results.csv produced by wilcoxon_compare.py")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--caption", required=True, help="Table caption text (no 'Table N.' prefix needed)")
    ap.add_argument("--comparison", action="append", required=True,
                     help="ORIGINAL_LABEL=DISPLAY_LABEL, repeatable, in display order")
    ap.add_argument("--font", default="Times New Roman")
    args = ap.parse_args()

    plt.rcParams["font.family"] = args.font

    comparison_map = {}
    comparison_order = []
    for item in args.comparison:
        orig, disp = item.split("=", 1)
        comparison_map[orig] = disp
        comparison_order.append(orig)

    with open(args.csv, newline="") as f:
        data = list(csv.DictReader(f))

    rows = []
    for comp in comparison_order:
        comp_rows = [r for r in data if r["comparison"] == comp and r["group"] == "all"]
        comp_rows.sort(key=lambda r: METRIC_ORDER.index(r["metric"]))
        for i, r in enumerate(comp_rows):
            rows.append({
                "comparison": comparison_map[comp] if i == 0 else "",
                "metric": METRIC_LABELS[r["metric"]],
                "mean_a": f"{float(r['mean_base']):.4f}",
                "mean_b": f"{float(r['mean_new']):.4f}",
                "delta": f"{float(r['mean_diff']):+.4f}",
                "r": f"{float(r['rank_biserial_r']):+.3f}",
                "p": fmt_p(float(r["p_value"])),
                "sig": r["significant_0.05"],
                "block_start": i == 0,
            })

    columns = ["Comparison", "Metric", "Mean A", "Mean B", "Δ", "Rank-biserial $r$", "$p$ value", "Significant"]
    col_widths = [0.13, 0.19, 0.09, 0.09, 0.10, 0.15, 0.17, 0.11]

    n_rows = len(rows) + 1
    row_h = 0.62
    fig_h = row_h * (n_rows + 2)
    fig_w = 13.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    x_positions = [0]
    for w in col_widths:
        x_positions.append(x_positions[-1] + w)

    top_y = n_rows
    header_y = n_rows - 1

    ax.plot([0, 1], [top_y, top_y], color="black", lw=1.8)
    ax.plot([0, 1], [header_y, header_y], color="black", lw=1.0)

    for ci, label in enumerate(columns):
        xc = x_positions[ci] + 0.006
        ax.text(xc, top_y - 0.5, label, fontsize=14, va="center", ha="left")

    for ri, row in enumerate(rows):
        y = header_y - 1 - ri
        if row["block_start"] and ri != 0:
            ax.plot([0, 1], [y + 1, y + 1], color="black", lw=0.9)
        values = [row["comparison"], row["metric"], row["mean_a"], row["mean_b"],
                  row["delta"], row["r"], row["p"], row["sig"]]
        for ci, val in enumerate(values):
            xc = x_positions[ci] + 0.006
            ax.text(xc, y + 0.5, val, fontsize=13.5, va="center", ha="left")

    bottom_y = header_y - len(rows)
    ax.plot([0, 1], [bottom_y, bottom_y], color="black", lw=1.8)

    ax.set_title(f"Table.  {args.caption}", fontsize=14, loc="center", pad=18)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
