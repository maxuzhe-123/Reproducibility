"""CodeJudgeBench generation stage (pipeline_v2) — pairwise judging.

The model acts as a judge: it reads the problem + two candidate solutions (Assistant A / B)
and outputs a verdict, [[A>B]] or [[B>A]]. The A/B placement follows the original _codegen_ordering:
    reverse=False → A=pos (correct), B=neg (incorrect), label='A'
    reverse=True  → A=neg, B=pos, label='B'
With position_mode=both, each problem is generated once in each order to remove position bias (same as the original work).

The task_metadata of each generation carries label / pos_response / neg_response, etc., for eval,
which parses [[A>B]] to get the verdict and compares it with label (Repro / ReproCorrect).

Isolated from humaneval/codereval: reuses the generic generate_prepared_tasks; prompts/fields are all defined in this file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.pipeline_v2.common import config_dataset_path
from src.pipeline_v2.generation_core import PreparedTask, generate_prepared_tasks

DEFAULT_DATASET_PATH = "src/datasets/CodeJudgeBench/codegen.jsonl"

# Judge prompt adapted from CodeJudgeBench's PairwiseModel.JUDGE_PROMPT (verdict-only format). Placeholders {query} {answer_a} {answer_b}.
_JUDGE_PROMPT = """
<|User Prompt|>
{query}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>

Please act as an impartial judge and your job is to evaluate which answer is better.

Your response must follow this format:

[Verdict]
Assistant A is better: [[A>B]] 
or
Assistant B is better: [[B>A]]
""".strip()


def _resolve_limit(cfg: Dict[str, Any]):
    nt = cfg["experiment"].get("num_tasks")
    if nt is None or str(nt).lower() == "all":
        return None
    return int(nt)


def _orderings(cfg: Dict[str, Any]) -> List[bool]:
    """position_mode: both (default) | forward | reverse — same as the original run_codejudgebench."""
    mode = str((cfg.get("codejudge", {}) or {}).get("position_mode", "both")).lower()
    if mode == "forward":
        return [False]
    if mode == "reverse":
        return [True]
    return [False, True]


def _codegen_ordering(pos: str, neg: str, reverse: bool) -> Tuple[str, str, str]:
    """Same as the original work: reverse=False → (pos, neg, 'A'); reverse=True → (neg, pos, 'B')."""
    return (neg, pos, "B") if reverse else (pos, neg, "A")


def load_codejudge_records(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = config_dataset_path(cfg, DEFAULT_DATASET_PATH)
    limit = _resolve_limit(cfg)
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                {
                    "idx": idx,
                    "question_id": str(item.get("question_id", idx)),
                    "response_model": str(item.get("response_model", "unknown")),
                    "question_content": item.get("question_content", ""),
                    "pos_response": item.get("pos_response", ""),
                    "neg_response": item.get("neg_response", ""),
                    "difficulty": item.get("difficulty", "unknown"),
                    "platform": item.get("platform", "unknown"),
                }
            )
            if limit is not None and len(records) >= limit:
                break
    return records


def prepare_codejudge_tasks(
    cfg: Dict[str, Any],
    existing: Dict[str, Dict[str, Any]] | None = None,
) -> List[PreparedTask]:
    prepared: List[PreparedTask] = []
    inst_index = 0
    for rec in load_codejudge_records(cfg):
        for reverse in _orderings(cfg):
            answer_a, answer_b, label = _codegen_ordering(
                rec["pos_response"], rec["neg_response"], reverse
            )
            tag = "rev" if reverse else "fwd"
            task = {
                "idx": inst_index,
                "task_id": f"{rec['question_id']}__{rec['response_model']}__{tag}",
                "question_id": rec["question_id"],
                "response_model": rec["response_model"],
            }
            inst_index += 1
            previous = next(
                (
                    r
                    for r in (existing or {}).values()
                    if r.get("task_id") == task["task_id"]
                ),
                None,
            )
            if previous is not None:
                prepared.append(
                    (task, previous.get("prompt", ""), dict(previous.get("task_metadata", {})))
                )
                continue
            user_prompt = _JUDGE_PROMPT.format(
                query=(rec["question_content"] or "").strip(),
                answer_a=(answer_a or "").strip(),
                answer_b=(answer_b or "").strip(),
            )
            metadata = {
                "idx": task["idx"],
                "question_id": rec["question_id"],
                "response_model": rec["response_model"],
                "difficulty": rec["difficulty"],
                "platform": rec["platform"],
                "reverse": reverse,
                "label": label,  # Correct verdict ('A' or 'B')
                # The two candidate solutions, kept for reference
                "pos_response": rec["pos_response"],
                "neg_response": rec["neg_response"],
            }
            prepared.append((task, user_prompt, metadata))
    return prepared


def codejudge_gen(
    cfg: Dict[str, Any],
    model,
    run_dir: Path,
    existing: Dict[str, Dict[str, Any]],
    resume: bool = True,
    retry_failed: bool = True,
) -> Path:
    prepared_tasks = prepare_codejudge_tasks(cfg, existing=existing)
    return generate_prepared_tasks(
        cfg, model, run_dir, prepared_tasks, existing, resume, retry_failed
    )
