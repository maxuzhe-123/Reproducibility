"""General-purpose paired Wilcoxon signed-rank test over CodeBLEU-style jsonl eval files.

Each input jsonl must have one JSON object per line with a "task_id" field (or --key)
and numeric metric fields (defaults to the 5 CodeBLEU components).

Usage examples
--------------
Compare two named runs:
    python scripts/wilcoxon_compare.py \\
        --input baseline=runs_v2/evaluations/.../codebleu_no_docstring.jsonl \\
        --input few_shot=runs_v2/evaluations/.../codebleu_no_docstring.jsonl \\
        --pair few_shot:baseline \\
        --out runs_v2/comparisons_supplement/my_analysis

Chain through several runs in the order given (X2 vs X1, X3 vs X2, ...):
    python scripts/wilcoxon_compare.py \\
        --input baseline=...jsonl --input few_shot=...jsonl --input cot=...jsonl \\
        --chain \\
        --out runs_v2/comparisons_supplement/my_analysis

All pairwise combinations of the registered inputs:
    python scripts/wilcoxon_compare.py \\
        --input a=...jsonl --input b=...jsonl --input c=...jsonl \\
        --all \\
        --out runs_v2/comparisons_supplement/my_analysis

--pair NEW:BASE reads as "compare NEW against BASE" (BASE is the reference/left-hand
side, NEW is the treatment/right-hand side); mean_diff = mean(NEW) - mean(BASE).
Repeat --pair as many times as you like; mix with --chain/--all if useful.

Stratify by a categorical field (e.g. CodeReval's "level" difficulty tag) instead of
pooling all records into one test:
    python scripts/wilcoxon_compare.py \\
        --input raw=...jsonl --input rag=...jsonl \\
        --pair rag:raw --group-by level \\
        --out runs_v2/comparisons_supplement/my_analysis
This runs one Wilcoxon test per distinct value of --group-by (in addition to the
pooled "all" test), assuming the field is present on every record in both files.

Split ONE file into two conditions by a boolean/categorical field (e.g. CodeJudgeBench's
"reverse": true/false marking forward vs reverse pairwise-judge ordering), instead of
supplying two separate files:
    python scripts/wilcoxon_compare.py \\
        --input all=runs_v2/evaluations/.../task_results.jsonl \\
        --split-field reverse --split-map "True=rev,False=fwd" \\
        --key-strip-suffix __fwd,__rev \\
        --metrics instance_accuracy,choice_consistency \\
        --pair rev:fwd --group-by difficulty \\
        --out runs_v2/comparisons_supplement/my_analysis
--split-map maps str(field_value) -> condition name. --key-strip-suffix drops a
literal trailing suffix from --key (after --key-basename, if also given) so the two
halves share the same matching key (e.g. "..._fwd" and "..._rev" -> the same id).
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

DEFAULT_METRICS = [
    "codebleu",
    "ngram_match_score",
    "weighted_ngram_match_score",
    "syntax_match_score",
    "dataflow_match_score",
]


def make_key(rec: dict, key: str, key_basename: bool, strip_suffixes: list[str]) -> object:
    k = rec[key]
    if key_basename and isinstance(k, str) and "/" in k:
        k = k.rsplit("/", 1)[-1]
    if strip_suffixes and isinstance(k, str):
        for suf in strip_suffixes:
            if suf and k.endswith(suf):
                k = k[: -len(suf)]
                break
    return k


def read_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_jsonl(path: Path, key: str, key_basename: bool = False, strip_suffixes: list[str] | None = None) -> dict:
    rows = {}
    for rec in read_records(path):
        rows[make_key(rec, key, key_basename, strip_suffixes or [])] = rec
    return rows


def rank_biserial(diffs: np.ndarray) -> float:
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    pos = ranks[nz > 0].sum()
    neg = ranks[nz < 0].sum()
    total = pos + neg
    return (pos - neg) / total if total else 0.0


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compare(base_name, base_data, new_name, new_data, metrics, group="all"):
    common = sorted(set(base_data) & set(new_data))
    if not common:
        raise ValueError(f"No overlapping keys between '{base_name}' and '{new_name}' (group={group!r})")
    results = []
    for metric in metrics:
        a = np.array([base_data[t][metric] for t in common], dtype=float)
        b = np.array([new_data[t][metric] for t in common], dtype=float)
        diffs = b - a
        n_nonzero = int(np.sum(diffs != 0))
        if n_nonzero == 0:
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        results.append({
            "comparison": f"{new_name} vs {base_name}",
            "group": group,
            "condition_base": base_name,
            "condition_new": new_name,
            "metric": metric,
            "n_pairs": len(common),
            "n_nonzero_diffs": n_nonzero,
            "mean_base": float(a.mean()),
            "mean_new": float(b.mean()),
            "mean_diff": float(diffs.mean()),
            "median_diff": float(np.median(diffs)),
            "wilcoxon_stat": float(stat) if stat == stat else None,
            "p_value": float(p),
            "rank_biserial_r": rank_biserial(diffs),
            "significant_0.05": bool(p < 0.05),
            "sig": sig_stars(p),
        })
    return results


def parse_inputs(items: list[str]) -> dict[str, Path]:
    inputs = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--input must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        inputs[name] = Path(path)
    return inputs


def parse_pairs(items: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in items:
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"--pair must be NEW:BASE, got: {item}")
        new, base = item.split(":", 1)
        pairs.append((new, base))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", action="append", default=[], required=True,
                     help="NAME=PATH to a jsonl eval file. Repeatable.")
    ap.add_argument("--pair", action="append", default=[],
                     help="NEW:BASE comparison to run. Repeatable.")
    ap.add_argument("--chain", action="store_true",
                     help="Compare each --input to the previous one, in the order given.")
    ap.add_argument("--all", action="store_true",
                     help="Compare every pairwise combination of the registered --input runs.")
    ap.add_argument("--out", required=True, help="Output directory for wilcoxon_results.csv/.md")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                     help=f"Comma-separated metric field names (default: {','.join(DEFAULT_METRICS)})")
    ap.add_argument("--key", default="task_id", help="JSON field used to match records across files (default: task_id)")
    ap.add_argument("--key-basename", action="store_true",
                     help="Match records by the last '/'-separated segment of --key instead of the "
                          "full value (useful when the same id has a different repo-name prefix "
                          "across two runs, e.g. CodeReval's 'CoderEval/<hash>' vs '<repo>/<hash>').")
    ap.add_argument("--group-by", default=None,
                     help="Field name to stratify by (e.g. 'level'). Runs one test per distinct value "
                          "found in the base condition's records, plus a pooled 'all' test.")
    ap.add_argument("--split-field", default=None,
                     help="Instead of comparing two --input files, split the single --input file into "
                          "multiple named conditions by this field's value (see --split-map).")
    ap.add_argument("--split-map", default=None,
                     help="Comma-separated VALUE=NAME pairs mapping str(record[--split-field]) to a "
                          "condition name, e.g. 'True=rev,False=fwd'. Required with --split-field.")
    ap.add_argument("--key-strip-suffix", default=None,
                     help="Comma-separated literal suffixes to strip from --key (after --key-basename, "
                          "if set) so two conditions with differently-suffixed ids still match, e.g. "
                          "'__fwd,__rev'.")
    ap.add_argument("--title", default="Wilcoxon signed-rank test", help="Title line for the markdown report")
    args = ap.parse_args()

    inputs = parse_inputs(args.input)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    strip_suffixes = [s for s in (args.key_strip_suffix.split(",") if args.key_strip_suffix else []) if s]

    if args.split_field:
        if not args.split_map:
            raise SystemExit("--split-field requires --split-map")
        if len(inputs) != 1:
            raise SystemExit("--split-field requires exactly one --input (the combined file to split).")
        (only_name, only_path), = inputs.items()
        split_map = {}
        for item in args.split_map.split(","):
            val, name = item.split("=", 1)
            split_map[val] = name
        names_order = list(dict.fromkeys(split_map.values()))
        data = {name: {} for name in names_order}
        for rec in read_records(only_path):
            val = str(rec.get(args.split_field))
            if val not in split_map:
                continue
            name = split_map[val]
            data[name][make_key(rec, args.key, args.key_basename, strip_suffixes)] = rec
        for name in names_order:
            matching_values = [v for v, n in split_map.items() if n == name]
            print(f"{name}: {len(data[name])} records (split from {only_path} where "
                  f"{args.split_field} in {matching_values})")
        input_names = names_order
    else:
        data = {name: load_jsonl(path, args.key, args.key_basename, strip_suffixes) for name, path in inputs.items()}
        for name, d in data.items():
            print(f"{name}: {len(d)} records ({inputs[name]})")
        input_names = list(inputs.keys())

    pairs = list(parse_pairs(args.pair))
    if args.chain:
        pairs += [(input_names[i + 1], input_names[i]) for i in range(len(input_names) - 1)]
    if args.all:
        pairs += [(b, a) for a, b in combinations(input_names, 2)]

    if not pairs:
        raise SystemExit("No comparisons requested. Use --pair, --chain, and/or --all.")

    all_results = []
    for new_name, base_name in pairs:
        if new_name not in data or base_name not in data:
            raise SystemExit(f"Unknown input name in pair '{new_name}:{base_name}'. "
                              f"Registered inputs: {list(inputs.keys())}")
        base_data, new_data = data[base_name], data[new_name]
        all_results += compare(base_name, base_data, new_name, new_data, metrics, group="all")
        if args.group_by:
            common = set(base_data) & set(new_data)
            seen = []
            for t in sorted(common):
                v = base_data[t].get(args.group_by)
                if v not in seen:
                    seen.append(v)
            for group_value in sorted(seen, key=lambda x: (x is None, x)):
                b_sub = {t: r for t, r in base_data.items() if r.get(args.group_by) == group_value}
                n_sub = {t: r for t, r in new_data.items() if r.get(args.group_by) == group_value}
                all_results += compare(base_name, b_sub, new_name, n_sub, metrics, group=str(group_value))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "wilcoxon_results.csv"
    fieldnames = list(all_results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    md_path = out_dir / "wilcoxon_results.md"
    lines = [f"# {args.title}\n"]
    lines.append(
        "Paired per-task comparison. Test: two-sided Wilcoxon signed-rank test "
        "(`scipy.stats.wilcoxon`, `zero_method=\"wilcox\"`). Effect size: matched-pairs "
        "rank-biserial correlation r (positive => the second-named condition scores higher). "
        "`mean_diff = mean_new - mean_base`.\n"
    )

    comp_names = []
    for new_name, base_name in pairs:
        comp = f"{new_name} vs {base_name}"
        if comp not in comp_names:
            comp_names.append(comp)

    for comp_name in comp_names:
        lines.append(f"\n## {comp_name}\n")
        groups_in_comp = []
        for row in all_results:
            if row["comparison"] == comp_name and row["group"] not in groups_in_comp:
                groups_in_comp.append(row["group"])
        for group in groups_in_comp:
            heading = "All records" if group == "all" else f"`{args.group_by}` = {group}"
            lines.append(f"\n### {heading}\n")
            lines.append("| Metric | n pairs | n nonzero diffs | mean base | mean new | mean diff | W statistic | p-value | r (rank-biserial) | significant (p<0.05) |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for row in all_results:
                if row["comparison"] != comp_name or row["group"] != group:
                    continue
                lines.append(
                    f"| {row['metric']} | {row['n_pairs']} | {row['n_nonzero_diffs']} | "
                    f"{row['mean_base']:.4f} | {row['mean_new']:.4f} | {row['mean_diff']:+.4f} | "
                    f"{row['wilcoxon_stat']:.1f} | {row['p_value']:.4g} {row['sig']} | "
                    f"{row['rank_biserial_r']:+.3f} | {row['significant_0.05']} |"
                )

    lines.append("\n## p-value distribution summary\n")
    for comp_name in comp_names:
        groups_in_comp = []
        for row in all_results:
            if row["comparison"] == comp_name and row["group"] not in groups_in_comp:
                groups_in_comp.append(row["group"])
        for group in groups_in_comp:
            label = comp_name if group == "all" else f"{comp_name} [{args.group_by}={group}]"
            pvals = [row["p_value"] for row in all_results
                     if row["comparison"] == comp_name and row["group"] == group]
            n_sig = sum(p < 0.05 for p in pvals)
            lines.append(
                f"- **{label}**: p-values = {['{:.4g}'.format(p) for p in pvals]}; "
                f"{n_sig}/{len(pvals)} metrics significant at α=0.05 "
                f"(max p = {max(pvals):.4g}, min p = {min(pvals):.4g})."
            )

    all_p = [row["p_value"] for row in all_results]
    n_sig_all = sum(p < 0.05 for p in all_p)
    lines.append(
        f"\n**Overall**: {n_sig_all}/{len(all_p)} of all metric x comparison tests significant "
        f"at α=0.05 (median p = {np.median(all_p):.4g})."
    )

    md_path.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")
    for row in all_results:
        group_tag = "" if row["group"] == "all" else f"[{row['group']}] "
        print(group_tag + row["comparison"], row["metric"], f"p={row['p_value']:.4g}", row["sig"])


if __name__ == "__main__":
    main()
