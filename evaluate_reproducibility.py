"""
evaluate_reproducibility.py
----------------------------
Unified reproducibility evaluation framework for HumanEval

Three dimensions:
  ① Text Reproducibility      ── Weighted N-Gram Match (BLEU_weight)
  ② Structure Reproducibility ── Depth-bounded Subtree Entropy (JSD & SCE)
  ③ Function Reproducibility  ── unit-test pass rate (pass@1 / pass@k)

Each dimension has two sub-metrics:
  - vs_ref   : each generation vs canonical_solution (mean and stability of generation quality)
  - pairwise : pairwise among the k generations (internal consistency of outputs, i.e., reproducibility)

Structure Reproducibility uses the two entropy measures proposed in the paper:
  S_JSD(P, Q) = 1 − D_JS(P ‖ Q)      symmetric, range [0,1]
  S_CE(P, Q)  = H(Q) / H(P, Q)        directed, range (0,1]
  Both are based on the empirical probability distribution of depth-d AST subtrees.

Usage:
    python evaluate_reproducibility.py \\
        --preds   runs/humaneval_qwen/2025-xx-xx_xx/preds.jsonl \\
        --dataset src/datasets/HumanEval.jsonl \\
        --output  runs/humaneval_qwen/2025-xx-xx_xx/reproducibility.json

Output:
    reproducibility.json  ── per-task + global summary metrics
    (a summary table is also printed to the console)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.utils.text_metrics import (
    pairwise_bleu_weight,
)
from src.utils.struct_entropy_metrics import (
    pairwise_jsd,
    pairwise_sce,
)
from src.utils.codebleu_metrics import (
    compute_codebleu_detailed,
    CODEBLEU_KEYS,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data loading utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_preds_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Parse preds.jsonl.
    Two formats are supported:
      - strict JSONL (one JSON object per line)
      - pretty JSONL (indent=4, records separated by blank lines)
    Uses raw_decode streaming parsing, which is robust to both formats.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    records: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        # Skip whitespace
        stripped = content[pos:]
        lstripped = stripped.lstrip()
        if not lstripped:
            break
        pos += len(stripped) - len(lstripped)
        try:
            obj, offset = decoder.raw_decode(lstripped)
            records.append(obj)
            pos += offset
        except json.JSONDecodeError:
            # Find the next '{' and continue
            next_brace = content.find("{", pos + 1)
            if next_brace == -1:
                break
            pos = next_brace
    return records


# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-task evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_one_task(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute metrics on the three dimensions for a single task (record) (pairwise internal consistency only; no comparison with the reference solution).

    Args:
        record: a single record from preds.jsonl

    Returns:
        dictionary of evaluation results for the task
    """
    preds = record.get("preds", [])
    codes = [p.get("code", "") or "" for p in preds]
    k = len(codes)

    # ── ① Function Reproducibility ────────────────────────
    pass_flags = [bool(p.get("pass", False)) for p in preds]
    n_pass     = sum(pass_flags)
    func_pass_rate = n_pass / k if k > 0 else float("nan")
    ok_pass1       = bool(record.get("ok_pass1", False))
    ok_passk       = bool(record.get("ok_passk", False))

    # ── ② Text Reproducibility ───────────────────────────
    text_pairwise = pairwise_bleu_weight(codes)   # Pairwise internal consistency

    # ── ③ Structure Reproducibility ──────────────────────
    struct_jsd_pairwise = pairwise_jsd(codes)     # Pairwise JSD consistency
    struct_sce_pairwise = pairwise_sce(codes)     # Pairwise SCE consistency

    # ── ④ CodeBLEU reproducibility (overall score + four components) ───────────────────────
    codebleu_detail = compute_codebleu_detailed(codes)  # Pairwise CodeBLEU self-similarity
    codebleu_pairwise = codebleu_detail["codebleu"]

    return {
        "task_id": record.get("task_id"),
        "k":       k,

        # Function
        "func": {
            "pass_rate": func_pass_rate,
            "ok_pass1":  ok_pass1,
            "ok_passk":  ok_passk,
        },

        # Text (pairwise only)
        "text": {
            "pairwise_bleu": text_pairwise,
        },

        # Structure (pairwise only)
        "struct": {
            "pairwise_jsd": struct_jsd_pairwise,
            "pairwise_sce": struct_sce_pairwise,
        },

        # CodeBLEU (pairwise only): overall score + four components
        "codebleu": {
            "pairwise_codebleu": codebleu_pairwise,
            "ngram_match_score": codebleu_detail["ngram_match_score"],
            "weighted_ngram_match_score": codebleu_detail["weighted_ngram_match_score"],
            "syntax_match_score": codebleu_detail["syntax_match_score"],
            "dataflow_match_score": codebleu_detail["dataflow_match_score"],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Global aggregation
# ══════════════════════════════════════════════════════════════════════════════

def _mean_std(values: List[float]) -> Dict[str, float]:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return {"mean": float("nan"), "std": float("nan")}
    arr = np.array(valid, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def aggregate(task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the mean / standard deviation of all task-level metrics."""

    def _collect(key_path: List[str]) -> List[float]:
        out = []
        for r in task_results:
            v = r
            for k in key_path:
                v = v.get(k, float("nan")) if isinstance(v, dict) else float("nan")
            out.append(float(v) if not isinstance(v, float) else v)
        return out

    pass_rate_list       = _collect(["func", "pass_rate"])
    ok_pass1_list        = [float(r["func"]["ok_pass1"]) for r in task_results]
    ok_passk_list        = [float(r["func"]["ok_passk"]) for r in task_results]

    text_pw_list         = _collect(["text", "pairwise_bleu"])

    struct_jsd_pw_list   = _collect(["struct", "pairwise_jsd"])
    struct_sce_pw_list   = _collect(["struct", "pairwise_sce"])

    codebleu_pw_list     = _collect(["codebleu", "pairwise_codebleu"])
    codebleu_ngram_list  = _collect(["codebleu", "ngram_match_score"])
    codebleu_wngram_list = _collect(["codebleu", "weighted_ngram_match_score"])
    codebleu_syntax_list = _collect(["codebleu", "syntax_match_score"])
    codebleu_dflow_list  = _collect(["codebleu", "dataflow_match_score"])

    n = len(task_results)
    return {
        "num_tasks": n,

        # Function reproducibility
        "func_reproducibility": {
            "pass@1":                 sum(ok_pass1_list) / n if n else float("nan"),
            "pass@k":                 sum(ok_passk_list) / n if n else float("nan"),
            "avg_pass_rate_per_task": _mean_std(pass_rate_list),
        },

        # Text reproducibility
        "text_reproducibility": {
            "pairwise_bleu_weight":  _mean_std(text_pw_list),    # Internal consistency
        },

        # Structure reproducibility (Depth-bounded Subtree Entropy)
        "struct_reproducibility": {
            "pairwise_jsd": _mean_std(struct_jsd_pw_list),   # Symmetric JSD similarity
            "pairwise_sce": _mean_std(struct_sce_pw_list),   # Directed SCE similarity
        },

        # CodeBLEU reproducibility (overall score + four components, pairwise self-similarity)
        "codebleu_reproducibility": {
            "pairwise_codebleu": _mean_std(codebleu_pw_list),
            "ngram_match_score": _mean_std(codebleu_ngram_list),
            "weighted_ngram_match_score": _mean_std(codebleu_wngram_list),
            "syntax_match_score": _mean_std(codebleu_syntax_list),
            "dataflow_match_score": _mean_std(codebleu_dflow_list),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Console output
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(agg: Dict[str, Any]):
    W = 60
    print("\n" + "=" * W)
    print("  HumanEval 可复现性评估汇总")
    print("=" * W)

    fr = agg["func_reproducibility"]
    print(f"\n【功能可复现性 Function Reproducibility】")
    print(f"  Pass@1  = {fr['pass@1']:.4f}")
    print(f"  Pass@k  = {fr['pass@k']:.4f}")
    apr = fr["avg_pass_rate_per_task"]
    print(f"  每题通过率 (mean±std) = {apr['mean']:.4f} ± {apr['std']:.4f}")

    tr = agg["text_reproducibility"]
    print(f"\n【文本可复现性 Text Reproducibility (BLEU_weight)】")
    pw  = tr["pairwise_bleu_weight"]
    print(f"  两两一致性 (mean±std)  = {pw['mean']:.4f}  ± {pw['std']:.4f}")

    sr = agg["struct_reproducibility"]
    print(f"\n【结构可复现性 Structure Reproducibility (Subtree Entropy)】")
    jsd_pw = sr["pairwise_jsd"]
    sce_pw = sr["pairwise_sce"]
    print(f"  两两 S_JSD   (mean±std)     = {jsd_pw['mean']:.4f}  ± {jsd_pw['std']:.4f}")
    print(f"  两两 S_CE    (mean±std)     = {sce_pw['mean']:.4f}  ± {sce_pw['std']:.4f}")

    cr = agg["codebleu_reproducibility"]
    cb_pw = cr["pairwise_codebleu"]
    print(f"\n【CodeBLEU 可复现性 CodeBLEU Reproducibility】")
    print(f"  两两一致性 CodeBLEU (mean±std)        = {cb_pw['mean']:.4f}  ± {cb_pw['std']:.4f}")
    for name, label in (
        ("ngram_match_score", "n-gram        "),
        ("weighted_ngram_match_score", "weighted n-gram"),
        ("syntax_match_score", "syntax (AST)  "),
        ("dataflow_match_score", "dataflow      "),
    ):
        sub = cr[name]
        print(f"    {label} (mean±std) = {sub['mean']:.4f}  ± {sub['std']:.4f}")
    print("=" * W + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main function
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_from_paths(
    preds_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Programmatic interface: pass paths directly, return the summary dict, and write reproducibility.json.

    Called by main.py after generation, to avoid reimplementing the three-dimension computation in main.py.

    Args:
        preds_path:  path to preds.jsonl (already written by main.py)
        output_path: output path; if None, written next to preds.jsonl
        verbose:     whether to print progress and the summary

    Returns:
        agg: summary dict returned by aggregate()
    """
    if verbose:
        print(f"\n[reproducibility] 加载 preds: {preds_path}")
    records = load_preds_jsonl(preds_path)

    if verbose:
        print(f"[reproducibility] 逐任务计算三维度指标（共 {len(records)} 题）...")
    task_results = []
    for i, rec in enumerate(records):
        result = evaluate_one_task(rec)
        task_results.append(result)
        if verbose and ((i + 1) % 20 == 0 or (i + 1) == len(records)):
            print(f"[reproducibility]   进度: {i+1}/{len(records)}")

    agg = aggregate(task_results)

    out_path = output_path or str(Path(preds_path).parent / "reproducibility.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    output = {
        "meta": {
            "preds_path": preds_path,
            "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_tasks":  len(task_results),
        },
        "summary":  agg,
        "per_task": task_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if verbose:
        print_summary(agg)
        print(f"[reproducibility] 完整结果已保存: {out_path}")

    return agg


def main(args):
    agg = evaluate_from_paths(
        preds_path=args.preds,
        output_path=args.output,
        verbose=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="三维可复现性评估")
    ap.add_argument(
        "--preds",
        required=True,
        help="目标实验的 preds.jsonl 路径",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="结果输出路径（默认保存在 preds.jsonl 同目录下）",
    )
    main(ap.parse_args())
