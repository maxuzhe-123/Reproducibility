"""CodeJudgeBench — bar charts per judge: (overall + difficulty) / response_model, each as a separate figure.

Two separate grouped bar charts per judge (white fill + different hatches for the 6 metrics, with std error bars):
    1) Overall / easy / medium / hard — <judge>_breakdown_overall_difficulty.png
    2) By response_model (the model that generated the correct/incorrect solutions) — <judge>_breakdown_response_model.png

6 bars per group: ReproCorrect@A / ReproCorrect@B / ReproCorrect_Rate /
           Repro@A / Repro@B / Repro_Rate

Usage:
    python -m src.pipeline_v2.plot_codejudge_gemini_breakdown

Data sources (see the JUDGES configuration):
    runs_v2/evaluations/codejudgebench/<model>/codejudgebench/<sampling_token>/
        summary.json       -> overall + difficulty (the 6 metrics already in metrics / by_difficulty)
        task_results.jsonl -> response_model (split into A/B by reverse, then aggregated per instance)

Output directory: runs_v2/comparisons_supplement/codejudge_<judge>_breakdown/
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]

# One configuration per judge: display name / generation result directory / output directory and file prefix
JUDGES = [
    {
        "display_name": "Gemini-2.5-Flash",
        "eval_dir": REPO_ROOT
        / "runs_v2/evaluations/codejudgebench/gemini_2.5_flash/codejudgebench"
        / "tem_0.6_thinking_1024",
        "out_dir": REPO_ROOT / "runs_v2/comparisons_supplement/codejudge_gemini_breakdown",
        "out_prefix": "gemini",
    },
    {
        "display_name": "Qwen3-Coder-Next",
        "eval_dir": REPO_ROOT
        / "runs_v2/evaluations/codejudgebench/qwen3_coder_next/codejudgebench"
        / "tem_0_top_p_0.95_top_k_20",
        "out_dir": REPO_ROOT / "runs_v2/comparisons_supplement/codejudge_qwen3_coder_next_breakdown",
        "out_prefix": "qwen3_coder_next",
    },
]

# Metric drawing order + hatch for each (white fill, black edges, distinguished by hatch)
METRICS = ["ReproCorrect@A", "ReproCorrect@B", "ReproCorrect_Rate", "Repro@A", "Repro@B", "Repro_Rate"]
HATCHES = {
    "ReproCorrect@A": "//",
    "ReproCorrect@B": "\\\\",
    "ReproCorrect_Rate": "xx",
    "Repro@A": "--",
    "Repro@B": "||",
    "Repro_Rate": "oo",
}


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def load_task_results(path: Path) -> List[Dict[str, Any]]:
    recs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def panel_overall_difficulty(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    nodes = {"Overall": summary["metrics"], **summary["by_difficulty"]}
    for key in ["Overall", "easy", "medium", "hard"]:
        node = nodes[key]
        out[key] = {m: node[m] for m in METRICS}
        for m in METRICS:
            out[key][m + "_std"] = node[m + "_std"]
    return out


def panel_by_response_model(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_model.setdefault(r["response_model"], []).append(r)

    out = {}
    for model in sorted(by_model, key=lambda m: -len(by_model[m])):
        rs = by_model[model]
        fwd = [r for r in rs if not r["reverse"]]
        rev = [r for r in rs if r["reverse"]]
        acc_a = [r["instance_accuracy"] for r in fwd]
        acc_b = [r["instance_accuracy"] for r in rev]
        cons_a = [r["choice_consistency"] for r in fwd]
        cons_b = [r["choice_consistency"] for r in rev]

        out[model] = {
            "ReproCorrect@A": _mean(acc_a),
            "ReproCorrect@A_std": _std(acc_a),
            "ReproCorrect@B": _mean(acc_b),
            "ReproCorrect@B_std": _std(acc_b),
            "ReproCorrect_Rate": (_mean(acc_a) + _mean(acc_b)) / 2.0,
            "ReproCorrect_Rate_std": _std(acc_a + acc_b),
            "Repro@A": _mean(cons_a),
            "Repro@A_std": _std(cons_a),
            "Repro@B": _mean(cons_b),
            "Repro@B_std": _std(cons_b),
            "Repro_Rate": (_mean(cons_a) + _mean(cons_b)) / 2.0,
            "Repro_Rate_std": _std(cons_a + cons_b),
        }
    return out


def draw_panel(ax, data: Dict[str, Dict[str, Any]], title: str, xtick_rotation: int = 0) -> None:
    labels = list(data.keys())
    n_groups = len(labels)
    n_metrics = len(METRICS)
    width = 0.8 / n_metrics
    x = range(n_groups)

    for k, metric in enumerate(METRICS):
        offset = (k - (n_metrics - 1) / 2.0) * width
        vals = [data[label][metric] for label in labels]
        errs = [data[label][metric + "_std"] for label in labels]
        bars = ax.bar(
            [i + offset for i in x], vals, width, yerr=errs, capsize=2,
            label=metric, facecolor="white", edgecolor="black",
            hatch=HATCHES[metric], linewidth=0.8,
            error_kw={"elinewidth": 0.8, "alpha": 0.7},
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=10, rotation=0,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=xtick_rotation, ha="right" if xtick_rotation else "center",
                        fontsize=14)
    ax.set_ylim(0, 1.35)
    ax.set_title(title, fontsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def render_single(data: Dict[str, Dict[str, Any]], judge_name: str, panel_title: str, out_dir: Path,
                   out_path: Path, xtick_rotation: int = 0, figsize=(14, 6)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    draw_panel(ax, data, panel_title, xtick_rotation=xtick_rotation)
    ax.set_ylabel("Score")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08), fontsize=14)
    fig.suptitle(f"CodeJudgeBench — {judge_name} Judge — {panel_title}", fontsize=18, y=1.16)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {out_path}")


def render_judge(judge: Dict[str, Any]) -> None:
    eval_dir: Path = judge["eval_dir"]
    out_dir: Path = judge["out_dir"]
    prefix: str = judge["out_prefix"]

    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    task_results = load_task_results(eval_dir / "task_results.jsonl")

    data_overall_difficulty = panel_overall_difficulty(summary)
    data_response_model = panel_by_response_model(task_results)

    render_single(
        data_overall_difficulty, judge["display_name"], "Overall & By Difficulty", out_dir,
        out_dir / f"{prefix}_breakdown_overall_difficulty.png", figsize=(12, 6),
    )
    render_single(
        data_response_model, judge["display_name"], "By Response Model", out_dir,
        out_dir / f"{prefix}_breakdown_response_model.png", xtick_rotation=0, figsize=(22, 6),
    )


def main() -> None:
    for judge in JUDGES:
        render_judge(judge)


if __name__ == "__main__":
    main()
