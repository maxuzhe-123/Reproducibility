"""Multi-dimensional comparison + visualization (based on the summary.json outputs of pipeline_v2 eval).

Scans  <eval_root>/<dataset>/<model>/<sub...>/<tem_..._top_k>/summary.json
(sub = <template>  or  defense/<defense_name>) and produces three kinds of comparison per (dataset, model),
plus one cross-model comparison per dataset, one figure per metric (*_mean metrics with std error bars):

  prompt    same model, same sampling parameters, different prompts (templates)  — including baseline
  defense   same model, same sampling parameters, same base template, baseline + different defenses
  sampling  same model, same prompt/defense, different sampling parameters
  models    same dataset, same experimental configuration (sub + sampling parameters), different models — including CodeBLEU and the other metrics

Output: <out>/<dataset>/<model>/{prompt,defense,sampling}/.../<metric>.png
        <out>/<dataset>/_models/<sub>/<sampling>/<metric>.png  + summary.csv

Usage (requires an environment with matplotlib, e.g., base anaconda):
    python -m src.pipeline_v2.compare \
        --eval-root runs_v2/evaluations --out runs_v2/comparisons
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["axes.unicode_minus"] = False

# Defenses are split into two classes by mechanism: preventive defenses harden the prompt before generation; detection defenses try to identify injection and may block.
# Defense names come from VALID_DEFENSES in src/defenses/humaneval_defenses.py.
# Only the preventive ones are listed explicitly; all other defenses are classified as detection (so none are left unclassified).
PREVENTIVE_DEFENSES = frozenset(
    {"paraphrasing", "retokenization", "delimiters", "sandwich", "instructional"}
)

# The six CoderEval dependency levels, in canonical order from weak to strong context dependency (used to sort stratified figures).
CODEREVAL_LEVEL_ORDER = (
    "self_contained",
    "slib_runnable",
    "plib_runnable",
    "class_runnable",
    "file_runnable",
    "project_runnable",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _passk_value(summary: Dict[str, Any]) -> Optional[float]:
    metrics = summary.get("metrics", {})
    for key, val in metrics.items():
        if key.startswith("pass@") and key != "pass@1":
            return val
    return None


def load_records(eval_root: Path) -> List[Dict[str, Any]]:
    """Each summary.json → one record (grouped by summary['run'], not by parsing the path)."""
    records = []
    for sp in sorted(eval_root.rglob("summary.json")):
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if summary.get("stage") != "evaluation":
            continue
        run = summary.get("run") or {}
        records.append(
            {
                "path": sp,
                "dataset": summary.get("dataset", "unknown"),
                "model": run.get("model", "unknown"),
                "template": run.get("template") or "no_template",
                "defense": run.get("defense"),
                "is_defense": bool(run.get("is_defense")),
                "sub": run.get("experiment_sub", ""),
                "sampling": run.get("sampling_token", "unknown"),
                "summary": summary,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Metric specifications
# ---------------------------------------------------------------------------
ValueFn = Callable[[Dict[str, Any]], Optional[float]]
MetricSpec = Tuple[str, str, ValueFn, Optional[ValueFn]]


def _codejudge_metric_specs() -> List[MetricSpec]:
    """codejudgebench: RQ5 reproducibility metrics (scalar + std, taken directly from summary.metrics)."""
    names = [
        ("ReproCorrect@A", "ReproCorrect@A"),
        ("ReproCorrect@B", "ReproCorrect@B"),
        ("ReproCorrect_Rate", "ReproCorrect_Rate"),
        ("Repro@A", "Repro@A"),
        ("Repro@B", "Repro@B"),
        ("Repro_Rate", "Repro_Rate"),
    ]
    return [
        (name, disp,
         (lambda nm: lambda s: s.get("metrics", {}).get(nm))(name),
         (lambda nm: lambda s: s.get("metrics", {}).get(nm + "_std"))(name))
        for name, disp in names
    ]


def build_metric_specs(records: List[Dict[str, Any]]) -> List[MetricSpec]:
    if any(r.get("dataset") == "codejudgebench" for r in records):
        return _codejudge_metric_specs()
    specs: List[MetricSpec] = [
        ("pass@1", "pass@1", lambda s: s.get("metrics", {}).get("pass@1"), None),
        ("pass@k", "pass@k", _passk_value, None),
    ]
    for key, disp in [
        ("codebleu", "CodeBLEU"),
        ("ngram_match_score", "ngram_match"),
        ("weighted_ngram_match_score", "weighted_ngram_match"),
        ("syntax_match_score", "syntax_match"),
        ("dataflow_match_score", "dataflow_match"),
    ]:
        specs.append(
            (
                key,
                disp,
                (lambda k: lambda s: s.get("metrics", {}).get(f"{k}_mean"))(key),
                (lambda k: lambda s: s.get("metrics", {}).get(f"{k}_std"))(key),
            )
        )
    # Extra for codereval: PythonExec pass@k (including by level)
    if any(r["summary"].get("pythonexec_pass_k") for r in records):
        specs.append(
            ("pythonexec_pass@1", "pythonexec_pass@1",
             lambda s: (s.get("pythonexec_pass_k") or [None])[0], None)
        )
        specs.append(
            ("pythonexec_pass@k", "pythonexec_pass@k",
             lambda s: (s.get("pythonexec_pass_k") or [None])[-1], None)
        )
        levels = set()
        for r in records:
            levels |= set((r["summary"].get("pythonexec_by_level") or {}).keys())
        for lvl in sorted(levels):
            specs.append(
                (
                    f"pythonexec_pass@k_{lvl}",
                    f"pythonexec_pass@k[{lvl}]",
                    (lambda L: lambda s: ((s.get("pythonexec_by_level") or {}).get(L) or [None])[-1])(lvl),
                    None,
                )
            )
    return specs


# ---------------------------------------------------------------------------
# Plotting + CSV
# ---------------------------------------------------------------------------
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._@\[\]-]+", "_", str(name)).strip("_") or "x"


def _temperature(sampling_token: str) -> float:
    """Parse the temperature from sampling_token (e.g., 'tem_0.3_top_p_0.9_top_k_100'), used to pick a representative run."""
    m = re.search(r"tem_([0-9.]+)", str(sampling_token))
    return float(m.group(1)) if m else float("inf")


# The 4 CodeBLEU components (keys use the *_mean / *_std prefixes in summary.metrics),
# with colors consistent with the dependency-level grouped figures.
CODEBLEU_COMPONENTS: List[Tuple[str, str, str]] = [
    ("ngram_match_score", "ngram_match", "#DD8452"),
    ("weighted_ngram_match_score", "weighted_ngram_match", "#55A868"),
    ("syntax_match_score", "syntax_match", "#C44E52"),
    ("dataflow_match_score", "dataflow_match", "#8172B3"),
]


def render_codebleu_components_group(
    rows: List[Tuple[str, Dict[str, Any]]],  # (x_label, summary)
    out_dir: Path,
    title_prefix: str,
    xlabel: str,
) -> int:
    """Draw the 4 CodeBLEU components of a group of experiments in one grouped bar chart (distinguished by color).

    The x-axis is the experiments (prompt/defense/sampling/model); each experiment has 4 side-by-side bars for
    ngram / weighted_ngram / syntax / dataflow, with std error bars.
    Outputs codebleu_components.png in the same directory as the per-metric figures of render_group.
    """
    labels: List[str] = []
    series: Dict[str, List[float]] = {key: [] for key, _, _ in CODEBLEU_COMPONENTS}
    errs: Dict[str, List[float]] = {key: [] for key, _, _ in CODEBLEU_COMPONENTS}
    for xlab, summ in rows:
        m = summ.get("metrics", {})
        vals = [m.get(f"{key}_mean") for key, _, _ in CODEBLEU_COMPONENTS]
        if any(v is None for v in vals):
            continue
        labels.append(xlab)
        for (key, _, _), v in zip(CODEBLEU_COMPONENTS, vals):
            series[key].append(float(v))
            errs[key].append(float(m.get(f"{key}_std") or 0.0))
    if len(labels) < 2:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels) + 2), 4.8))
    n = len(CODEBLEU_COMPONENTS)
    width = 0.8 / n
    for i, (key, disp, color) in enumerate(CODEBLEU_COMPONENTS):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(labels))]
        ax.bar(xs, series[key], width=width,
               yerr=errs[key] if any(errs[key]) else None, capsize=3,
               label=disp, color=color, alpha=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("score")
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title_prefix} — CodeBLEU components", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "codebleu_components.png", dpi=150)
    plt.close(fig)
    return 1


def render_group(
    rows: List[Tuple[str, Dict[str, Any]]],  # (x_label, summary)
    specs: List[MetricSpec],
    out_dir: Path,
    title_prefix: str,
    xlabel: str,
) -> int:
    """Draw one bar chart per metric for a group (a series of experiments on the x-axis) and write a summary.csv."""
    if len(rows) < 2:
        return 0
    plots = []
    csv_rows = []
    for mid, disp, vfn, sfn in specs:
        labels, vals, errs = [], [], []
        for xlab, summ in rows:
            v = vfn(summ)
            if v is None:
                continue
            labels.append(xlab)
            vals.append(float(v))
            errs.append(float(sfn(summ)) if sfn and sfn(summ) is not None else 0.0)
            csv_rows.append({"metric": disp, xlabel: xlab, "value": v,
                             "std": (sfn(summ) if sfn else "")})
        if len(vals) < 2:
            continue
        plots.append((disp, labels, vals, errs))

    if not plots and not csv_rows:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    n_png = 0
    for disp, labels, vals, errs in plots:
        fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(labels) + 2), 4.5))
        bars = ax.bar(range(len(labels)), vals,
                      yerr=errs if any(errs) else None, capsize=4,
                      color="#4C72B0", alpha=0.9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(disp)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{title_prefix} — {disp}", fontsize=11)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{_safe(disp)}.png", dpi=150)
        plt.close(fig)
        n_png += 1
    # Extra: combine the 4 CodeBLEU components into one grouped figure (different colors) for an at-a-glance comparison.
    n_png += render_codebleu_components_group(rows, out_dir, title_prefix, xlabel)
    if csv_rows:
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["metric", xlabel, "value", "std"])
            w.writeheader()
            w.writerows(csv_rows)
    return n_png


def render_codejudge_position(record: Dict[str, Any], out_dir: Path) -> int:
    """codejudgebench, single experiment: ReproCorrect@A/@B and Repro@A/@B in one figure (to inspect position bias, with std)."""
    m = record["summary"].get("metrics", {})
    triples = [(lab, m.get(k), m.get(k + "_std")) for lab, k in
               (("ReproCorrect@A", "ReproCorrect@A"),
                ("ReproCorrect@B", "ReproCorrect@B"),
                ("Repro@A", "Repro@A"),
                ("Repro@B", "Repro@B"))]
    triples = [(lab, v, e) for lab, v, e in triples if v is not None]
    if len(triples) < 2:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [t[0] for t in triples]
    vals = [t[1] for t in triples]
    errs = [t[2] or 0.0 for t in triples]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    # @A/@B of the same family use the same color family: ReproCorrect = blue/orange, Repro = dark blue/dark orange
    colors = ["#4C72B0", "#DD8452", "#2E5C8A", "#A85A30"][:len(vals)]
    bars = ax.bar(range(len(vals)), vals, yerr=errs, capsize=4, color=colors, alpha=0.9)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.45)
    ax.set_title(f"{record['model']} {record['sub']} — position", fontsize=10)
    for b, v, e in zip(bars, vals, errs):
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "position.png", dpi=150)
    plt.close(fig)
    with (out_dir / "position.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "std"])
        for lab, v, e in triples:
            w.writerow([lab, v, e])
    return 1


def render_codejudge_buckets(record: Dict[str, Any], out_dir: Path) -> int:
    """codejudgebench, single experiment: the 6 reproducibility metrics bucketed by difficulty (one figure each, with std)."""
    summary = record["summary"]
    buckets = summary.get("by_difficulty") or {}
    if not buckets:
        return 0
    sub_out = out_dir / "by_difficulty"
    sub_out.mkdir(parents=True, exist_ok=True)
    keys = sorted(buckets.keys())
    n = 0
    for mkey in ("ReproCorrect@A", "ReproCorrect@B", "ReproCorrect_Rate",
                 "Repro@A", "Repro@B", "Repro_Rate"):
        vals = [buckets[k].get(mkey) for k in keys]
        if any(v is None for v in vals):
            continue
        errs = [buckets[k].get(mkey + "_std") or 0.0 for k in keys]
        fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(keys) + 2), 4.5))
        bars = ax.bar(range(len(keys)), vals, yerr=errs, capsize=4, color="#4C72B0")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(mkey)
        ax.set_ylim(0, 1.45)
        ax.set_title(f"{record['model']} {record['sub']} — {mkey} by difficulty", fontsize=9)
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)
        ax.grid(axis="y", ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(sub_out / f"{_safe(mkey)}.png", dpi=150)
        plt.close(fig)
        n += 1
    return n


def render_codebleu_by_level(record: Dict[str, Any], out_dir: Path) -> int:
    """Single codereval experiment: CodeBLEU for each of the six dependency levels.

    Reads codebleu_by_level directly from the existing summary.json (already computed in the eval stage)
    without rerunning eval. Outputs:
      - codebleu_by_level.png    main figure: overall CodeBLEU per level (with std error bars)
      - codebleu_components.png  grouped figure: the 4 components per level
      - codebleu_by_level.csv    all 5 columns (mean±std)
    Levels are sorted by CODEREVAL_LEVEL_ORDER (dependency from weak to strong).
    """
    cbl = record["summary"].get("codebleu_by_level") or {}
    if not cbl:
        return 0
    levels = [lvl for lvl in CODEREVAL_LEVEL_ORDER if lvl in cbl]
    levels += [lvl for lvl in cbl if lvl not in CODEREVAL_LEVEL_ORDER]
    if not levels:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("codebleu", "CodeBLEU"),
        ("ngram_match_score", "ngram"),
        ("weighted_ngram_match_score", "weighted_ngram"),
        ("syntax_match_score", "syntax"),
        ("dataflow_match_score", "dataflow"),
    ]
    title = f"{record['model']} {record['sub']} @ {record['sampling']}"
    n_png = 0

    # (a) Main figure: overall CodeBLEU per level (with std error bars)
    vals = [cbl[lvl].get("codebleu_mean") for lvl in levels]
    errs = [cbl[lvl].get("codebleu_std") or 0.0 for lvl in levels]
    if all(v is not None for v in vals):
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(levels) + 2), 4.5))
        bars = ax.bar(range(len(levels)), vals, yerr=errs, capsize=4,
                      color="#4C72B0", alpha=0.9)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("CodeBLEU")
        ax.set_xlabel("dependency level (weak → strong)")
        ax.set_title(f"{title} — CodeBLEU by level", fontsize=10)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / "codebleu_by_level.png", dpi=150)
        plt.close(fig)
        n_png += 1

    # (b) Grouped figure: the 4 components per level
    comps = [m for m in metrics if m[0] != "codebleu"]
    series = {
        key: [cbl[lvl].get(f"{key}_mean") for lvl in levels] for key, _ in comps
    }
    if all(v is not None for vs in series.values() for v in vs):
        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(levels) + 2), 4.8))
        width = 0.8 / len(comps)
        palette = ["#DD8452", "#55A868", "#C44E52", "#8172B3"]
        for i, (key, disp) in enumerate(comps):
            xs = [x + (i - (len(comps) - 1) / 2) * width for x in range(len(levels))]
            ax.bar(xs, series[key], width=width, label=disp, color=palette[i % len(palette)])
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("score")
        ax.set_xlabel("dependency level (weak → strong)")
        ax.set_title(f"{title} — CodeBLEU components by level", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", ls="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / "codebleu_components.png", dpi=150)
        plt.close(fig)
        n_png += 1

    # CSV: one row per level, 5 columns of mean±std + count
    with (out_dir / "codebleu_by_level.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "count"] + [disp for _, disp in metrics])
        for lvl in levels:
            stats = cbl[lvl]
            row = [lvl, stats.get("count")]
            for key, _ in metrics:
                mean, std = stats.get(f"{key}_mean"), stats.get(f"{key}_std")
                row.append("" if mean is None else f"{mean:.4f}±{std:.4f}")
            w.writerow(row)
    return n_png


# ---------------------------------------------------------------------------
# Three kinds of comparison
# ---------------------------------------------------------------------------
def compare_all(eval_root: Path, out_root: Path) -> None:
    records = load_records(eval_root)
    if not records:
        print(f"[compare] 未在 {eval_root} 找到 eval 产物（summary.json）。")
        return

    by_dm: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_dm[(r["dataset"], r["model"])].append(r)

    total_png = 0
    for (dataset, model), recs in sorted(by_dm.items()):
        base = out_root / dataset / model
        specs = build_metric_specs(recs)

        # 1) prompt: fixed sampling, different templates (non-defense)
        for samp in sorted({r["sampling"] for r in recs}):
            rows = [
                (r["template"], r["summary"])
                for r in recs
                if not r["is_defense"] and r["sampling"] == samp
            ]
            rows.sort(key=lambda x: x[0])
            total_png += render_group(
                rows, specs, base / "prompt" / samp,
                f"{dataset}/{model} prompt @ {samp}", "prompt",
            )

        # 2) defense: fixed sampling + base template, baseline + each defense.
        #    Figures are produced separately for the two mechanism classes (a full set for every metric, including CodeBLEU):
        #      preventive  preventive: baseline + preventive defenses
        #      detection   detection: baseline + the remaining defenses
        for samp in sorted({r["sampling"] for r in recs}):
            defs = [r for r in recs if r["is_defense"] and r["sampling"] == samp]
            base_tmpls = sorted({r["template"] for r in defs})
            for tmpl in base_tmpls:
                baseline = [
                    r for r in recs
                    if not r["is_defense"] and r["sampling"] == samp
                    and r["template"] == tmpl
                ]
                tmpl_defs = [r for r in defs if r["template"] == tmpl]
                for cat_name, in_cat in (
                    ("preventive", lambda d: d in PREVENTIVE_DEFENSES),
                    ("detection", lambda d: d not in PREVENTIVE_DEFENSES),
                ):
                    rows = []
                    if baseline:
                        rows.append(("baseline", baseline[0]["summary"]))
                    for r in tmpl_defs:
                        dname = r["defense"] or r["sub"]
                        if in_cat(dname):
                            rows.append((dname, r["summary"]))
                    total_png += render_group(
                        rows, specs,
                        base / "defense" / samp / _safe(tmpl) / cat_name,
                        f"{dataset}/{model} defense[{cat_name}] (base={tmpl}) @ {samp}",
                        "defense",
                    )

        # 3) sampling: fixed sub (prompt or defense), different sampling parameters
        by_sub: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in recs:
            by_sub[r["sub"]].append(r)
        for sub, sub_recs in sorted(by_sub.items()):
            rows = [(r["sampling"], r["summary"]) for r in sub_recs]
            rows.sort(key=lambda x: x[0])
            total_png += render_group(
                rows, specs, base / "sampling" / _safe(sub),
                f"{dataset}/{model} sampling @ {sub}", "sampling_params",
            )

        # 4) codejudgebench-specific: per experiment, position figures (ReproCorrect/Repro @A/@B) + difficulty buckets
        for r in recs:
            if r["dataset"] != "codejudgebench":
                continue
            stem = base / r["sampling"] / _safe(r["sub"])
            total_png += render_codejudge_position(r, stem / "position")
            total_png += render_codejudge_buckets(r, stem / "buckets")

        # 4b) codereval-specific: per experiment, a figure of CodeBLEU for each of the six dependency levels,
        #     reading codebleu_by_level directly from the existing summary.json without rerunning eval.
        if dataset == "codereval":
            for r in recs:
                stem = base / r["sampling"] / _safe(r["sub"])
                total_png += render_codebleu_by_level(r, stem / "codebleu_by_level")

    # 5) Cross-model comparison: fixed (dataset, sub=prompt/defense), different models on the x-axis.
    #    Different models often use their own recommended sampling parameters (different top_p/top_k), so alignment here is by sub only;
    #    one representative result is taken per model (preferring the greedy result at temperature=0, otherwise the lowest temperature),
    #    and one figure is produced per metric (including CodeBLEU).
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)
    for dataset, ds_recs in sorted(by_dataset.items()):
        if len({r["model"] for r in ds_recs}) < 2:
            continue  # Only one model; nothing to compare
        specs = build_metric_specs(ds_recs)
        # sub -> model -> all runs of that model under this sub
        by_sub_model: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for r in ds_recs:
            by_sub_model[r["sub"]][r["model"]].append(r)
        for sub, model_recs in sorted(by_sub_model.items()):
            if len(model_recs) < 2:
                continue  # Only one model under this prompt/defense
            rows = []
            for model, runs in sorted(model_recs.items()):
                # Representative run: prefer low temperature (greedy), then fix by lexicographic order of sampling_token,
                # to guarantee determinism (otherwise, with several runs at the same temperature, the choice depends on path order).
                rep = min(runs, key=lambda r: (_temperature(r["sampling"]), r["sampling"]))
                rows.append((model, rep["summary"]))
            total_png += render_group(
                rows, specs,
                out_root / dataset / "_models" / _safe(sub),
                f"{dataset} models @ {sub}", "model",
            )

    # 5b) Cross-model at identical sampling/thinking conditions: aligned by (sub, sampling), models on the x-axis.
    #     Difference from 5): 5) takes one representative run per model (sampling may differ), suitable for a rough comparison
    #     with each model's recommended parameters; but in thinking experiments the thinking level is encoded in sampling_token,
    #     so 5) would mix different thinking levels in one figure (e.g., flash@thinking_0 vs pro@thinking_1024).
    #     Here sampling_token must match exactly, giving a true model comparison at the same thinking level.
    for dataset, ds_recs in sorted(by_dataset.items()):
        specs = build_metric_specs(ds_recs)
        by_sub_samp_model: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for r in ds_recs:
            by_sub_samp_model[(r["sub"], r["sampling"])][r["model"]].append(r)
        for (sub, samp), model_recs in sorted(by_sub_samp_model.items()):
            if len(model_recs) < 2:
                continue  # Only one model under this exact condition; nothing to compare
            rows = [
                (model, runs[0]["summary"]) for model, runs in sorted(model_recs.items())
            ]
            total_png += render_group(
                rows, specs,
                out_root / dataset / "_models_by_sampling" / _safe(sub) / _safe(samp),
                f"{dataset} models @ {sub} / {samp}", "model",
            )

    print(f"[compare] 完成：{len(records)} 个 eval 结果 → {total_png} 张图，输出于 {out_root}")


def main() -> None:
    ap = argparse.ArgumentParser(description="多维度对比 pipeline_v2 eval 结果并可视化。")
    ap.add_argument("--eval-root", default="runs_v2/evaluations")
    ap.add_argument("--out", default="runs_v2/comparisons")
    args = ap.parse_args()
    compare_all(Path(args.eval_root), Path(args.out))


if __name__ == "__main__":
    main()
