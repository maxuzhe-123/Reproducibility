"""
HumanEval Ground Truth Runner
-------------------------------
Feed the canonical_solution of each task directly into the evaluation framework
to establish a ground-truth pass-rate baseline for each task.

The output format is identical to run_humaneval (preds.jsonl + summary.json)
and can be used as the ground-truth pass-rate baseline.

Usage:
    python run_humaneval_groundtruth.py \
        --dataset  src/datasets/HumanEval.jsonl \
        --out_dir  runs/ \
        --name     humaneval_groundtruth \
        --num_tasks 164 \
        --timeout  6
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

from src.utils.exec_code import run_with_timeout
from src.utils.io import append_jsonl, init_run_and_preds


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_humaneval(path: str, n: int = None):
    """Read HumanEval.jsonl and return the first n records (None means all)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集文件不存在: {path}")
    data = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            data.append({
                "task_id":          ex["task_id"],
                "prompt":           ex["prompt"],
                "canonical_solution": ex["canonical_solution"],
                "test":             ex["test"],
                "entry_point":      ex.get("entry_point"),
            })
    return data[:n] if n is not None else data


# ──────────────────────────────────────────────
# Main logic
# ──────────────────────────────────────────────
def run_groundtruth(args):
    # 1) Load data
    tasks = load_humaneval(args.dataset, args.num_tasks)
    print(f"[Ground Truth] 已加载 {len(tasks)} 道题目，数据集路径：{args.dataset}")

    # 2) Create the run directory for this run (reusing the framework's init_run_and_preds)
    # Use a lightweight cfg dict to satisfy the init_run_and_preds interface
    cfg = {
        "experiment": {
            "name":       args.name,
            "dataset":    "humaneval",
            "out_dir":    args.out_dir,
            "num_tasks":  len(tasks),
            "num_samples": 1,
        },
        "model":  {"provider": "ground_truth"},
        "prompt": {"template": "canonical_solution", "defense": "no"},
    }
    run_dir, preds_path = init_run_and_preds(
        out_dir=args.out_dir,
        name=args.name,
        cfg=cfg,
    )

    # 3) Execute each task
    n_pass = 0
    n_fail = 0
    fail_ids = []

    for i, ex in enumerate(tasks):
        # canonical_solution is the function body without the def line; it must be concatenated with the prompt
        combined_code = ex["prompt"] + ex["canonical_solution"]

        ok, err = run_with_timeout(
            code=combined_code,
            test=ex["test"],
            timeout_sec=args.timeout,
            entry_point=ex["entry_point"],
        )

        if ok:
            n_pass += 1
        else:
            n_fail += 1
            fail_ids.append(ex["task_id"])

        record = {
            "idx":        i,
            "task_id":    ex["task_id"],
            "prompt":     ex["prompt"],
            "canonical_solution": ex["canonical_solution"],
            "test":       ex["test"],
            "entry_point": ex["entry_point"],
            "ok_pass1":   ok,
            "ok_passk":   ok,   # GT has only one solution, so pass@1 == pass@k
            "error":      err,
        }
        append_jsonl(preds_path, record, pretty=True)

        status = "PASS" if ok else "FAIL"
        print(f"[{i+1:3d}/{len(tasks)}] {ex['task_id']:25s} -> {status}"
              + (f"  错误: {err.splitlines()[0][:80]}" if not ok else ""))

    # 4) Aggregate and save summary.json
    total   = len(tasks)
    pass_at1 = n_pass / total if total else 0.0

    summary = {
        "experiment":  args.name,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": {
            "name":     "humaneval",
            "path":     args.dataset,
            "num_tasks": total,
        },
        "metrics": {
            "pass@1":   pass_at1,
            "n_pass":   n_pass,
            "n_fail":   n_fail,
            "total":    total,
        },
        "failed_task_ids": fail_ids,
        "env": {
            "python": platform.python_version(),
            "timeout_sec": args.timeout,
        },
        "files": {
            "preds": preds_path,
        },
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 5) Print results
    print(f"\n{'='*50}")
    print(f"[Ground Truth] Pass@1 = {n_pass}/{total} = {pass_at1:.2%}")
    if fail_ids:
        print(f"[Ground Truth] 失败题目 ({len(fail_ids)} 道):")
        for tid in fail_ids:
            print(f"  - {tid}")
    print(f"结果已保存到: {run_dir}")
    print(f"{'='*50}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="HumanEval Ground Truth Runner")
    ap.add_argument("--dataset",   default="src/datasets/HumanEval.jsonl",
                    help="HumanEval.jsonl 路径")
    ap.add_argument("--out_dir",   default="runs/",
                    help="输出根目录（与 run_humaneval 保持一致）")
    ap.add_argument("--name",      default="humaneval_groundtruth",
                    help="本次运行名称，用于目录命名")
    ap.add_argument("--num_tasks", type=int, default=None,
                    help="只跑前 N 道题（默认全部 164 道）")
    ap.add_argument("--timeout",   type=int, default=6,
                    help="单题执行超时秒数")
    args = ap.parse_args()

    run_groundtruth(args)


if __name__ == "__main__":
    main()
